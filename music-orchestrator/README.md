# Music Orchestrator (Stage 1)

Local-first multi-agent instrumental generation service (GPT + Suno).

## What It Implements

- FastAPI API:
  - `POST /v1/runs`
  - `GET /v1/runs/{run_id}`
  - `GET /v1/runs/{run_id}/artifacts`
  - `POST /v1/runs/{run_id}/retry`
- CLI:
  - `lyra run --prompt "..."`
  - `lyra status --run-id <id>`
  - `lyra artifacts --run-id <id>`
  - `lyra retry --run-id <id> --target drums`
- Canonical contracts (`MusicBrief`, `TimelineContract`, `StemContract`, `QCReport`, `ArtifactManifest`)
- LangGraph workflow agents:
  - `IntentAgent` (GPT default `gpt-4.1-mini`)
  - `ConductorAgent` (`gpt-4.1`)
  - `StemPromptAgent` (`gpt-4.1-mini`)
  - `QCJudgeAgent` (`gpt-4.1`)
- Suno integration via existing local wrapper (`http://localhost:3000`)
- 3 pseudo-stems (`drums`, `bass`, `pads`) + `instrumental_mix.wav`
- SQLite run store + persisted logs + payload snapshots

## Requirements

- Python 3.11+
- `ffmpeg` in `PATH`
- Running Suno wrapper service at `http://localhost:3000`
- OpenAI API key for GPT agents

## Setup

```powershell
cd d:\Git\lyra\music-orchestrator
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.
If Suno generation is slow, increase `SUNO_TIMEOUT_SECONDS` and `SUNO_POLL_TIMEOUT_SECONDS`.

## Run

```powershell
# API server
lyra serve --host 127.0.0.1 --port 8000

# Create run
lyra run --prompt "dark cinematic synthwave, 128 bpm, no vocals"

# Check run
lyra status --run-id <run_id>

# Get artifact manifest
lyra artifacts --run-id <run_id>

# Retry one stem
lyra retry --run-id <run_id> --target drums --reason "tighten groove"
```

## API Examples

```bash
curl -X POST http://127.0.0.1:8000/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"prompt":"melodic house 124 bpm, warm bass, no vocals","references":[],"overrides":{}}'

curl http://127.0.0.1:8000/v1/runs/<run_id>
curl http://127.0.0.1:8000/v1/runs/<run_id>/artifacts
curl -X POST http://127.0.0.1:8000/v1/runs/<run_id>/retry \
  -H "Content-Type: application/json" \
  -d '{"target":"bass","reason":"better low-end focus"}'
```

## Artifact Layout

Outputs are stored under `ARTIFACTS_ROOT/<run_id>/`:

- `raw/` downloaded provider audio
- `stems/drums.wav`
- `stems/bass.wav`
- `stems/pads.wav`
- `mix/instrumental_mix.wav`
- `reports/music_brief.json`
- `reports/timeline_contract.json`
- `reports/qc_report.json`
- `reports/manifest.json`

## QC Gates

Hard stop unless all three stems pass:

- duration tolerance: `<= 250 ms`
- clipping: `<= -1 dBTP` (sample peak approximation)
- silence ratio gate
- tempo coherence near target BPM
- stem sanity heuristics:
  - drums transient density
  - bass low-end dominance
  - pads sustained profile

## Troubleshooting

- Suno 402 / credit issues:
  - Check `http://localhost:3000/api/get_limit`
  - Raise credits or lower `MIN_CREDITS_THRESHOLD`
- CAPTCHA instability:
  - Keep `suno-api` browser/cookie settings valid
  - Retry run with `lyra retry`
- `ffmpeg` not found:
  - Install ffmpeg and verify `ffmpeg -version`
- OpenAI not configured:
  - Set `OPENAI_API_KEY`
  - Without key, fallback logic is used for deterministic scaffolding only

## Tests

```powershell
pytest
```

Includes unit tests for contracts/timeline/prompt-retry/QC and API contract tests with workflow mocking.
