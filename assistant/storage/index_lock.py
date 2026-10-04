"""In-process coordination for embedded search indexes and individual papers.

The server must still use one process: these locks do not coordinate multiple
processes opening the same embedded Qdrant store.
"""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from typing import Iterator


class _IndexLock:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer: int | None = None
        self._write_depth = 0
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        thread = threading.get_ident()
        with self._condition:
            owned = self._writer == thread
            if not owned:
                while self._writer is not None or self._waiting_writers:
                    self._condition.wait()
                self._readers += 1
        try:
            yield
        finally:
            if not owned:
                with self._condition:
                    self._readers -= 1
                    if not self._readers:
                        self._condition.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        thread = threading.get_ident()
        with self._condition:
            if self._writer == thread:
                self._write_depth += 1
            else:
                self._waiting_writers += 1
                try:
                    while self._writer is not None or self._readers:
                        self._condition.wait()
                    self._writer = thread
                    self._write_depth = 1
                finally:
                    self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._condition:
                self._write_depth -= 1
                if not self._write_depth:
                    self._writer = None
                    self._condition.notify_all()


index_lock = _IndexLock()
_resource_guard = threading.Lock()
_resource_locks: dict[str, threading.Lock] = {}
_version = re.compile(r"v\d+$", re.IGNORECASE)


@contextmanager
def resource_lock(key: str) -> Iterator[None]:
    with _resource_guard:
        lock = _resource_locks.setdefault(key, threading.Lock())
    with lock:
        yield


@contextmanager
def paper_lock(paper_id: str) -> Iterator[None]:
    """Serialize repeated ingests/removal of the same paper, not other papers."""
    key = _version.sub("", paper_id.strip()).lower()
    with resource_lock(f"paper:{key}"):
        yield