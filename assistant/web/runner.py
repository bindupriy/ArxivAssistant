"""Graph execution and bounded concurrent background jobs for the local web app."""

from __future__ import annotations

import json
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterator

from assistant.config import get_config
from assistant.graph import get_graph


def _now() -> str:
    return datetime.utcnow().isoformat()


def invoke_graph(state: dict[str, Any]) -> dict[str, Any]:
    return get_graph().invoke(
        state,
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )


def stream_graph(state: dict[str, Any]) -> Iterator[str]:
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def run() -> None:
        final: dict[str, Any] = {}
        try:
            stream = get_graph().stream(
                state,
                config={"configurable": {"thread_id": str(uuid.uuid4())}},
                stream_mode=["custom", "values"],
            )
            for item in stream:
                if isinstance(item, tuple) and len(item) == 2:
                    mode, payload = item
                else:
                    mode, payload = "values", item
                if mode == "custom":
                    events.put(dict(payload))
                elif isinstance(payload, dict):
                    final = payload
            events.put({"type": "final", "result": final})
        except Exception as exc:  # noqa: BLE001
            events.put({"type": "error", "message": str(exc)})
        finally:
            events.put(None)

    threading.Thread(target=run, name="assistant-chat", daemon=True).start()
    while True:
        event = events.get()
        if event is None:
            break
        yield json.dumps(event, ensure_ascii=True) + "\n"


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


class JobManager:
    def __init__(self, max_workers: int | None = None) -> None:
        workers = max_workers if max_workers is not None else get_config().concurrency.background_workers
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="assistant-job")
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def submit(self, kind: str, label: str, operation: Callable[[], dict[str, Any]]) -> dict:
        job = Job(id=str(uuid.uuid4()), kind=kind, label=label)
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            self._update(job.id, status="running")
            try:
                self._update(job.id, status="completed", result=operation())
            except Exception as exc:  # noqa: BLE001
                self._update(job.id, status="failed", error=str(exc))

        self._executor.submit(run)
        return asdict(job)

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [asdict(job) for job in jobs]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


jobs = JobManager()