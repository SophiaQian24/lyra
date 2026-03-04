from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .models import RunState


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunStore:
    db_path: Path

    def __post_init__(self) -> None:
        self._lock = Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    music_brief_json TEXT,
                    timeline_json TEXT,
                    stem_contracts_json TEXT,
                    generation_json TEXT,
                    qc_report_json TEXT,
                    manifest_json TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    step TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                """
            )

    def create_run(self, run_id: str, prompt: str, music_brief: dict[str, Any]) -> None:
        now = utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, prompt, status, music_brief_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, prompt, RunState.queued.value, json.dumps(music_brief), now, now),
            )

    def increment_retry(self, run_id: str) -> int:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE runs SET retry_count = retry_count + 1, updated_at = ? WHERE run_id = ?", (utc_now(), run_id))
            row = conn.execute("SELECT retry_count FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if not row:
                raise KeyError(run_id)
            return int(row["retry_count"])

    def set_status(self, run_id: str, status: RunState) -> None:
        self.update_fields(run_id, status=status.value)

    def update_fields(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        mapped = dict(fields)
        mapped["updated_at"] = utc_now()

        sql_parts = []
        values = []
        for key, value in mapped.items():
            column = {
                "music_brief": "music_brief_json",
                "timeline_contract": "timeline_json",
                "stem_contracts": "stem_contracts_json",
                "generation_results": "generation_json",
                "qc_report": "qc_report_json",
                "artifact_manifest": "manifest_json",
            }.get(key, key)

            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            sql_parts.append(f"{column} = ?")
            values.append(value)
        values.append(run_id)

        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE runs SET {', '.join(sql_parts)} WHERE run_id = ?", values)

    def add_log(self, run_id: str, level: str, step: str, message: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_logs (run_id, ts, level, step, message, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, utc_now(), level, step, message, json.dumps(payload) if payload else None),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None

        raw = dict(row)
        for key in [
            "music_brief_json",
            "timeline_json",
            "stem_contracts_json",
            "generation_json",
            "qc_report_json",
            "manifest_json",
        ]:
            if raw.get(key):
                raw[key] = json.loads(raw[key])
        return raw

    def get_logs(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM run_logs WHERE run_id = ? ORDER BY id ASC", (run_id,)).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if item.get("payload_json"):
                item["payload_json"] = json.loads(item["payload_json"])
            output.append(item)
        return output
