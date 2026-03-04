from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from .models import (
    ArtifactManifest,
    MusicBrief,
    QCReport,
    RunCreateRequest,
    RunCreateResponse,
    RunState,
    RunStatusResponse,
    StemContract,
    TimelineContract,
)
from .settings import Settings
from .store import RunStore
from .workflow import Stage1Workflow, WorkflowState


@dataclass
class RunJob:
    run_id: str
    prompt: str
    references: list[str]
    overrides: dict[str, Any]
    retry_target: Literal["drums", "bass", "pads", "all"] = "all"
    retry_index: int = 0


class RunManager:
    def __init__(self, settings: Settings, store: RunStore):
        self.settings = settings
        self.store = store
        self.workflow = Stage1Workflow(settings, store)
        self.queue: asyncio.Queue[RunJob] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(self._worker_loop(), name="run-worker")

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def create_run(self, request: RunCreateRequest) -> RunCreateResponse:
        run_id = str(uuid4())
        brief = self.workflow.llm.build_initial_brief(request.prompt, request.references, request.overrides)
        self.store.create_run(run_id, request.prompt, brief.model_dump())
        self.store.add_log(run_id, "info", "api", "Run created and queued", {"references": request.references})

        await self.queue.put(
            RunJob(
                run_id=run_id,
                prompt=request.prompt,
                references=request.references,
                overrides=request.overrides,
                retry_target="all",
                retry_index=0,
            )
        )
        return RunCreateResponse(run_id=run_id, status=RunState.queued, music_brief=brief)

    async def retry_run(self, run_id: str, target: Literal["drums", "bass", "pads", "all"], reason: str) -> int:
        row = self.store.get_run(run_id)
        if not row:
            raise KeyError(run_id)

        retry_count = int(row["retry_count"])
        if retry_count >= self.settings.max_run_retries:
            raise RuntimeError(
                f"Retry limit reached for run {run_id}. retry_count={retry_count}, max={self.settings.max_run_retries}"
            )

        retry_count = self.store.increment_retry(run_id)
        self.store.set_status(run_id, RunState.queued)
        self.store.add_log(run_id, "info", "retry", "Manual retry queued", {"target": target, "reason": reason})

        await self.queue.put(
            RunJob(
                run_id=run_id,
                prompt=row["prompt"],
                references=[],
                overrides={},
                retry_target=target,
                retry_index=retry_count,
            )
        )
        return retry_count

    def get_status(self, run_id: str) -> RunStatusResponse:
        row = self.store.get_run(run_id)
        if not row:
            raise KeyError(run_id)

        def parse(model_type, key):
            val = row.get(key)
            if val is None:
                return None
            return model_type.model_validate(val)

        music_brief = parse(MusicBrief, "music_brief_json")
        timeline = parse(TimelineContract, "timeline_json")

        stem_contracts_raw = row.get("stem_contracts_json") or []
        stem_contracts = [StemContract.model_validate(item) for item in stem_contracts_raw]

        qc_report = parse(QCReport, "qc_report_json")
        manifest = parse(ArtifactManifest, "manifest_json")

        logs = self.store.get_logs(run_id)

        return RunStatusResponse(
            run_id=run_id,
            status=RunState(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            prompt=row["prompt"],
            music_brief=music_brief,
            timeline_contract=timeline,
            stem_contracts=stem_contracts,
            qc_report=qc_report,
            artifact_manifest=manifest,
            error=row.get("error"),
            logs=logs,
        )

    def get_artifacts(self, run_id: str) -> ArtifactManifest:
        row = self.store.get_run(run_id)
        if not row:
            raise KeyError(run_id)
        raw = row.get("manifest_json")
        if not raw:
            raise RuntimeError(f"Artifacts not available for run {run_id}")
        return ArtifactManifest.model_validate(raw)

    async def _worker_loop(self) -> None:
        while True:
            job = await self.queue.get()
            await self._execute_job(job)
            self.queue.task_done()

    async def _execute_job(self, job: RunJob) -> None:
        run_id = job.run_id
        row = self.store.get_run(run_id)
        if not row:
            return

        prior_brief = row.get("music_brief_json")
        prior_timeline = row.get("timeline_json")
        prior_contracts = row.get("stem_contracts_json")
        prior_generation = row.get("generation_json") or {}
        prior_manifest = row.get("manifest_json") or {}

        prior_stem_wavs: dict[str, str] = {}
        if isinstance(prior_manifest, dict):
            prior_stem_wavs = prior_manifest.get("stems") or {}

        state: WorkflowState = {
            "run_id": run_id,
            "prompt": job.prompt,
            "references": job.references,
            "overrides": job.overrides,
            "retry_target": job.retry_target,
            "retry_index": job.retry_index,
        }

        if prior_brief:
            state["music_brief"] = MusicBrief.model_validate(prior_brief)
        if prior_timeline:
            state["timeline_contract"] = TimelineContract.model_validate(prior_timeline)
        if prior_contracts:
            state["stem_contracts"] = [StemContract.model_validate(item) for item in prior_contracts]
        if prior_generation:
            state["generation_results"] = prior_generation
        if prior_stem_wavs:
            state["stem_wavs"] = prior_stem_wavs

        try:
            attempts_used = 0
            max_attempts = self.settings.max_run_retries + 1
            current_state = state
            while attempts_used < max_attempts:
                out = await self.workflow.execute(current_state)
                report = out.get("qc_report")
                if report and report.passed and out.get("artifact_manifest"):
                    self.store.set_status(run_id, RunState.done)
                    self.store.update_fields(run_id, error=None)
                    self.store.add_log(run_id, "info", "workflow", "Run completed")
                    return

                attempts_used += 1
                if attempts_used >= max_attempts:
                    break

                if report:
                    failing = [
                        m.stem_name
                        for m in report.stem_metrics
                        if not (m.duration_pass and m.clipping_pass and m.silence_pass and m.tempo_pass and m.stem_sanity_pass)
                    ]
                else:
                    failing = ["all"]

                target = job.retry_target if job.retry_target != "all" else (failing[0] if failing else "all")
                retry_count = self.store.increment_retry(run_id)
                self.store.add_log(run_id, "info", "retry", "Automatic retry", {"target": target, "retry_count": retry_count})

                current_state = {
                    **out,
                    "run_id": run_id,
                    "prompt": job.prompt,
                    "references": job.references,
                    "overrides": job.overrides,
                    "retry_target": target,
                    "retry_index": retry_count,
                }

            self.store.set_status(run_id, RunState.failed)
            self.store.update_fields(run_id, error="QC failed after retry budget")
            self.store.add_log(run_id, "error", "workflow", "Run failed after retry budget")
        except Exception as exc:
            err_msg = str(exc) or repr(exc)
            self.store.set_status(run_id, RunState.failed)
            self.store.update_fields(run_id, error=err_msg)
            self.store.add_log(
                run_id,
                "error",
                "workflow",
                "Run failed with exception",
                {"error": err_msg, "exception_type": exc.__class__.__name__},
            )
