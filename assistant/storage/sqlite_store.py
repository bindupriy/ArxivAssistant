"""SQLite engine + session factory.

Lazily creates the schema on first use. Single engine per process.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from assistant.config import get_config
from assistant.storage.schema import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    cfg = get_config()
    cfg.storage.ensure_dirs()
    url = f"sqlite:///{cfg.storage.sqlite_path.as_posix()}"
    engine = create_engine(url, future=True, connect_args={"timeout": 30})
    # Multiple chat readers can run while a background ingest commits.
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        columns = {column["name"] for column in inspect(connection).get_columns("papers")}
        if "reading_status" not in columns:
            connection.exec_driver_sql(
                "ALTER TABLE papers ADD COLUMN reading_status VARCHAR(16) NOT NULL DEFAULT 'new'"
            )
        if "is_favorite" not in columns:
            connection.exec_driver_sql(
                "ALTER TABLE papers ADD COLUMN is_favorite BOOLEAN NOT NULL DEFAULT 0"
            )
        chunk_columns = {column["name"] for column in inspect(connection).get_columns("chunks")}
        if "page_number" not in chunk_columns:
            connection.exec_driver_sql("ALTER TABLE chunks ADD COLUMN page_number INTEGER")
    return engine


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Usage:
        with session_scope() as s:
            s.add(...); ...
    """
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
