from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from .manager import RunManager
from .models import RetryRequest, RetryResponse, RunCreateRequest, RunCreateResponse, RunStatusResponse
from .settings import Settings, get_settings
from .store import RunStore


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    store = RunStore(cfg.db_path)
    manager = RunManager(cfg, store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(title="Lyra Music Orchestrator", version="0.1.0", lifespan=lifespan)

    def get_manager() -> RunManager:
        return manager

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/runs", response_model=RunCreateResponse)
    async def create_run(payload: RunCreateRequest, rm: RunManager = Depends(get_manager)) -> RunCreateResponse:
        try:
            return await rm.create_run(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/runs/{run_id}", response_model=RunStatusResponse)
    async def get_run(run_id: str, rm: RunManager = Depends(get_manager)) -> RunStatusResponse:
        try:
            return rm.get_status(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/runs/{run_id}/artifacts")
    async def get_artifacts(run_id: str, rm: RunManager = Depends(get_manager)) -> dict:
        try:
            return rm.get_artifacts(run_id).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/runs/{run_id}/retry", response_model=RetryResponse)
    async def retry_run(run_id: str, payload: RetryRequest, rm: RunManager = Depends(get_manager)) -> RetryResponse:
        try:
            retry_count = await rm.retry_run(run_id, payload.target, payload.reason)
            return RetryResponse(run_id=run_id, queued=True, target=payload.target, retry_count=retry_count)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
