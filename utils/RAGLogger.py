from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR
SHANGHAI_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def new_trace_id() -> str:
    return uuid.uuid4().hex


def shanghai_now() -> datetime:
    return datetime.now(SHANGHAI_TZ)


def shanghai_now_iso() -> str:
    return shanghai_now().isoformat()


def daily_log_file(log_root: str | Path, now: Optional[datetime] = None) -> Path:
    current = now or shanghai_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=SHANGHAI_TZ)
    else:
        current = current.astimezone(SHANGHAI_TZ)
    root = Path(log_root)
    return root / f"{current:%Y}" / f"{current:%m}" / f"{current:%Y-%m-%d}.jsonl"


def safe_json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [safe_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    return str(value)


class RAGLogger:
    def __init__(
        self,
        log_file: str | Path = DEFAULT_LOG_FILE,
        enabled: bool = True,
        pipeline_name: str = "",
    ) -> None:
        self.log_target = Path(log_file)
        self.enabled = enabled
        self.pipeline_name = pipeline_name

    def resolve_log_file(self) -> Path:
        if self.log_target.suffix:
            return self.log_target
        return daily_log_file(self.log_target)

    def log(self, event: str, trace_id: str, **fields: Any) -> None:
        if not self.enabled:
            return

        payload: Dict[str, Any] = {
            "ts": shanghai_now_iso(),
            "timezone": "Asia/Shanghai",
            "pipeline": self.pipeline_name,
            "trace_id": trace_id,
            "event": event,
        }
        payload.update({key: safe_json_value(value) for key, value in fields.items()})

        log_file = self.resolve_log_file()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    def stage_start(self, trace_id: str, stage: str, metrics: Optional[Dict[str, Any]] = None) -> float:
        started_at = time.perf_counter()
        self.log(
            event="stage_start",
            trace_id=trace_id,
            stage=stage,
            metrics=metrics or {},
        )
        return started_at

    def stage_end(
        self,
        trace_id: str,
        stage: str,
        started_at: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log(
            event="stage_end",
            trace_id=trace_id,
            stage=stage,
            status="ok",
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 3),
            metrics=metrics or {},
        )

    def stage_error(self, trace_id: str, stage: str, started_at: float, error: BaseException) -> None:
        self.log(
            event="stage_error",
            trace_id=trace_id,
            stage=stage,
            status="error",
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 3),
            error_type=type(error).__name__,
            error_message=str(error),
        )
