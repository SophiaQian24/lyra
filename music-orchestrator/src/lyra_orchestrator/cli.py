from __future__ import annotations

import json
import os

import httpx
import typer

app = typer.Typer(help="Lyra stage-1 orchestrator CLI")


def _base_url(base_url: str | None) -> str:
    return base_url or os.getenv("LYRA_API_BASE_URL", "http://127.0.0.1:8000")


def _print_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=True))


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("lyra_orchestrator.main:app", host=host, port=port, reload=False)


@app.command("run")
def run_command(
    prompt: str = typer.Option(..., help="User intent prompt"),
    references: list[str] = typer.Option([], help="Optional reference strings"),
    base_url: str | None = typer.Option(None, help="Orchestrator API base URL"),
) -> None:
    payload = {"prompt": prompt, "references": references, "overrides": {}}
    with httpx.Client(base_url=_base_url(base_url), timeout=60) as client:
        resp = client.post("/v1/runs", json=payload)
    if resp.status_code >= 400:
        raise typer.BadParameter(f"Request failed: {resp.status_code} {resp.text}")
    _print_json(resp.json())


@app.command("status")
def status_command(
    run_id: str = typer.Option(..., help="Run id"),
    base_url: str | None = typer.Option(None, help="Orchestrator API base URL"),
) -> None:
    with httpx.Client(base_url=_base_url(base_url), timeout=60) as client:
        resp = client.get(f"/v1/runs/{run_id}")
    if resp.status_code >= 400:
        raise typer.BadParameter(f"Request failed: {resp.status_code} {resp.text}")
    _print_json(resp.json())


@app.command("artifacts")
def artifacts_command(
    run_id: str = typer.Option(..., help="Run id"),
    base_url: str | None = typer.Option(None, help="Orchestrator API base URL"),
) -> None:
    with httpx.Client(base_url=_base_url(base_url), timeout=60) as client:
        resp = client.get(f"/v1/runs/{run_id}/artifacts")
    if resp.status_code >= 400:
        raise typer.BadParameter(f"Request failed: {resp.status_code} {resp.text}")
    _print_json(resp.json())


@app.command("retry")
def retry_command(
    run_id: str = typer.Option(..., help="Run id"),
    target: str = typer.Option(..., help="drums|bass|pads|all"),
    reason: str = typer.Option("manual retry", help="Retry reason"),
    base_url: str | None = typer.Option(None, help="Orchestrator API base URL"),
) -> None:
    payload = {"target": target, "reason": reason}
    with httpx.Client(base_url=_base_url(base_url), timeout=60) as client:
        resp = client.post(f"/v1/runs/{run_id}/retry", json=payload)
    if resp.status_code >= 400:
        raise typer.BadParameter(f"Request failed: {resp.status_code} {resp.text}")
    _print_json(resp.json())


if __name__ == "__main__":
    app()
