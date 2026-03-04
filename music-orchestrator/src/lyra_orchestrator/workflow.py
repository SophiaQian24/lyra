from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .audio import mix_stems, process_stem
from .llm import LLMClient
from .models import ArtifactManifest, GenerationResult, MusicBrief, QCReport, RunState, StemContract, TimelineContract
from .qc import compute_qc_report, is_borderline
from .settings import Settings
from .store import RunStore
from .suno_client import SunoClient
from .timeline import build_timeline_contract


class WorkflowState(TypedDict, total=False):
    run_id: str
    prompt: str
    references: list[str]
    overrides: dict[str, Any]
    retry_target: Literal["drums", "bass", "pads", "all"]
    retry_index: int

    music_brief: MusicBrief
    timeline_contract: TimelineContract
    stem_contracts: list[StemContract]

    generation_results: dict[str, dict[str, Any]]
    stem_wavs: dict[str, str]
    qc_report: QCReport
    artifact_manifest: ArtifactManifest


class Stage1Workflow:
    def __init__(self, settings: Settings, store: RunStore):
        self.settings = settings
        self.store = store
        self.llm = LLMClient(settings)
        self.suno = SunoClient(settings.suno_api_base_url, timeout_seconds=settings.suno_timeout_seconds)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("intent", self.intent_node)
        graph.add_node("conductor", self.conductor_node)
        graph.add_node("preflight", self.preflight_node)
        graph.add_node("generate", self.generate_node)
        graph.add_node("qc", self.qc_node)
        graph.add_node("mix", self.mix_node)

        graph.add_edge(START, "intent")
        graph.add_edge("intent", "conductor")
        graph.add_edge("conductor", "preflight")
        graph.add_edge("preflight", "generate")
        graph.add_edge("generate", "qc")
        graph.add_conditional_edges("qc", self.qc_route, {"mix": "mix", "fail": END})
        graph.add_edge("mix", END)
        return graph.compile()

    def _log(self, run_id: str, step: str, message: str, payload: dict[str, Any] | None = None) -> None:
        self.store.add_log(run_id, "info", step, message, payload)

    async def execute(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        self.store.set_status(run_id, RunState.planning)
        self._log(run_id, "workflow", "Run started", {"retry_target": state.get("retry_target", "all")})
        out = await self._graph.ainvoke(state)
        return out

    async def intent_node(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        if state.get("music_brief"):
            self._log(run_id, "intent", "Reusing existing MusicBrief")
            return {}

        brief = self.llm.build_initial_brief(
            prompt=state["prompt"],
            references=state.get("references", []),
            overrides=state.get("overrides", {}),
        )
        self.store.update_fields(run_id, music_brief=brief.model_dump())
        self._log(run_id, "intent", "MusicBrief generated")
        return {"music_brief": brief}

    async def conductor_node(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        if state.get("timeline_contract") and state.get("stem_contracts"):
            self._log(run_id, "conductor", "Reusing existing timeline and stem contracts")
            return {}

        brief = state["music_brief"]
        timeline = build_timeline_contract(brief)
        contracts = self.llm.build_stem_contracts(brief, timeline.total_seconds)

        self.store.update_fields(
            run_id,
            timeline_contract=timeline.model_dump(),
            stem_contracts=[c.model_dump() for c in contracts],
        )
        self._log(run_id, "conductor", "Timeline and stem contracts generated")
        return {"timeline_contract": timeline, "stem_contracts": contracts}

    async def preflight_node(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        limit = await self.suno.get_limit()
        credits_left = int(limit.get("credits_left", 0))
        self._log(run_id, "preflight", "Fetched Suno credits", limit)
        if credits_left < self.settings.min_credits_threshold:
            raise RuntimeError(
                f"Insufficient credits: credits_left={credits_left}, required>={self.settings.min_credits_threshold}"
            )
        return {}

    def _run_dir(self, run_id: str) -> Path:
        return self.settings.artifacts_root / run_id

    def _contract_by_name(self, contracts: list[StemContract]) -> dict[str, StemContract]:
        return {item.stem_name: item for item in contracts}

    async def _wait_for_audio_url(self, ids: list[str]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        deadline = time.monotonic() + float(self.settings.suno_poll_timeout_seconds)
        last_polled: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            polled = await self.suno.get(ids)
            last_polled = polled
            for clip in polled:
                if clip.get("audio_url") and clip.get("status") in {"streaming", "complete"}:
                    return clip, polled
            await asyncio.sleep(5)
        return None, last_polled

    async def generate_node(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        self.store.set_status(run_id, RunState.generating)
        self._log(run_id, "generate", "Starting stem generation")

        run_dir = self._run_dir(run_id)
        raw_dir = run_dir / "raw"
        stems_dir = run_dir / "stems"
        reports_dir = run_dir / "reports"
        mix_dir = run_dir / "mix"
        for d in [raw_dir, stems_dir, reports_dir, mix_dir]:
            d.mkdir(parents=True, exist_ok=True)

        contracts = self._contract_by_name(state["stem_contracts"])
        retry_target = state.get("retry_target", "all")
        retry_index = state.get("retry_index", 0)
        timeline = state["timeline_contract"]
        brief = state["music_brief"]

        target_stems = ["drums", "bass", "pads"] if retry_target == "all" else [retry_target]
        existing = dict(state.get("stem_wavs") or {})
        generation_results = dict(state.get("generation_results") or {})
        stem_wavs = dict(existing)

        for stem_name in ["drums", "bass", "pads"]:
            if stem_name not in target_stems:
                if stem_name in existing and Path(existing[stem_name]).exists():
                    self._log(run_id, "generate", f"Reusing existing stem {stem_name}")
                    continue
                target_stems.append(stem_name)

        for stem_name in target_stems:
            contract = contracts[stem_name]
            stem_payload = self.llm.build_stem_payload(brief, contract, retry_index=retry_index)
            payload = {
                "prompt": stem_payload.prompt,
                "tags": stem_payload.tags,
                "title": stem_payload.title,
                "negative_tags": stem_payload.negative_tags,
                "make_instrumental": True,
                "model": self.settings.suno_model,
                "wait_audio": False,
            }
            self._log(run_id, "suno_request", f"Generating {stem_name}", payload)
            response = await self.suno.custom_generate(**payload)
            self._log(run_id, "suno_response", f"Received Suno response for {stem_name}", {"count": len(response)})

            if not response:
                raise RuntimeError(f"Suno returned empty response for stem={stem_name}")

            selected = None
            for clip in response:
                if clip.get("audio_url"):
                    selected = clip
                    break
            if not selected:
                ids = [clip.get("id") for clip in response if clip.get("id")]
                if ids:
                    self._log(
                        run_id,
                        "suno_poll",
                        f"Polling /api/get for {stem_name}",
                        {"ids": ids, "poll_timeout_seconds": self.settings.suno_poll_timeout_seconds},
                    )
                    selected, polled = await self._wait_for_audio_url(ids)
                    response = polled if polled else response
            if not selected:
                raise RuntimeError(
                    f"No audio_url became available for stem={stem_name} "
                    f"within {self.settings.suno_poll_timeout_seconds}s"
                )

            extension = ".mp3"
            raw_path = raw_dir / f"{stem_name}{extension}"
            await self.suno.download_to_file(selected["audio_url"], str(raw_path))

            normalized_path = stems_dir / f"{stem_name}.wav"
            duration_seconds = process_stem(
                str(raw_path),
                str(normalized_path),
                target_duration_seconds=timeline.total_seconds,
                target_rms_db=self.settings.qc_target_lufs,
            )

            result = GenerationResult(
                stem_name=stem_name,  # type: ignore[arg-type]
                suno_ids=[c.get("id") for c in response if c.get("id")],
                audio_url=selected["audio_url"],
                local_raw_path=str(raw_path.resolve()),
                normalized_wav_path=str(normalized_path.resolve()),
                duration_seconds=duration_seconds,
                status=selected.get("status", "complete"),
                provider_payload=selected,
            )

            generation_results[stem_name] = result.model_dump()
            stem_wavs[stem_name] = str(normalized_path.resolve())

        self.store.update_fields(run_id, generation_results=generation_results)
        return {"generation_results": generation_results, "stem_wavs": stem_wavs}

    async def qc_node(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        self.store.set_status(run_id, RunState.qc)
        timeline = state["timeline_contract"]
        report = compute_qc_report(
            run_id=run_id,
            target_bpm=timeline.bpm,
            target_duration_seconds=timeline.total_seconds,
            stems=state["stem_wavs"],
            duration_tolerance_ms=float(self.settings.qc_duration_tolerance_ms),
            min_pass_ratio=float(self.settings.qc_min_pass_ratio),
        )

        if is_borderline(report):
            decision = self.llm.qc_judge(
                {
                    "numeric_passed": report.passed,
                    "target_bpm": report.target_bpm,
                    "target_duration_seconds": report.target_duration_seconds,
                    "stem_metrics": [m.model_dump() for m in report.stem_metrics],
                }
            )
            report.passed = decision.passed
            report.judge_notes = decision.notes

        self.store.update_fields(run_id, qc_report=report.model_dump())
        self._log(run_id, "qc", "QC evaluated", report.model_dump())
        return {"qc_report": report}

    def qc_route(self, state: WorkflowState) -> str:
        report = state.get("qc_report")
        if report and report.passed:
            return "mix"
        return "fail"

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    async def mix_node(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        self.store.set_status(run_id, RunState.mixing)

        run_dir = self._run_dir(run_id)
        stems = state["stem_wavs"]

        mix_path = run_dir / "mix" / "instrumental_mix.wav"
        mix_stems([stems["drums"], stems["bass"], stems["pads"]], str(mix_path), target_rms_db=self.settings.qc_target_lufs)

        music_brief_path = run_dir / "reports" / "music_brief.json"
        timeline_path = run_dir / "reports" / "timeline_contract.json"
        qc_path = run_dir / "reports" / "qc_report.json"
        manifest_path = run_dir / "reports" / "manifest.json"

        self._write_json(music_brief_path, state["music_brief"].model_dump())
        self._write_json(timeline_path, state["timeline_contract"].model_dump())
        self._write_json(qc_path, state["qc_report"].model_dump())

        manifest = ArtifactManifest(
            run_id=run_id,
            stems={
                "drums": str(Path(stems["drums"]).resolve()),
                "bass": str(Path(stems["bass"]).resolve()),
                "pads": str(Path(stems["pads"]).resolve()),
            },
            mix=str(mix_path.resolve()),
            reports={
                "music_brief": str(music_brief_path.resolve()),
                "timeline_contract": str(timeline_path.resolve()),
                "qc_report": str(qc_path.resolve()),
                "manifest": str(manifest_path.resolve()),
            },
        )
        self._write_json(manifest_path, manifest.model_dump(mode="json"))

        self.store.update_fields(run_id, artifact_manifest=manifest.model_dump(mode="json"))
        self._log(run_id, "mix", "Mix and manifest exported", manifest.model_dump(mode="json"))
        return {"artifact_manifest": manifest}
