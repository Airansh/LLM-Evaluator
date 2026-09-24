"""Engine and session management.

Schema initialization is intentionally lightweight: `init_db` calls
`Base.metadata.create_all`, which creates missing tables and never alters existing
ones. There are no migrations in the MVP; changing a column means deleting the local
SQLite file (or adding Alembic, listed as a next step in the README).
"""

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _enable_sqlite_pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def configure_engine(database_url: str) -> Engine:
    """(Re)create the global engine. Called at startup and by tests."""
    global _engine, _SessionLocal

    url = make_url(database_url)
    connect_args: dict[str, object] = {}
    if url.get_backend_name() == "sqlite":
        # The background runner writes from a different thread than the request.
        connect_args["check_same_thread"] = False
        if url.database and url.database != ":memory:":
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)

    if _engine is not None:
        _engine.dispose()
    _engine = create_engine(database_url, connect_args=connect_args)
    if url.get_backend_name() == "sqlite":
        event.listen(_engine, "connect", _enable_sqlite_pragmas)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db() -> None:
    Base.metadata.create_all(get_engine())


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Database engine not configured; call configure_engine() first")
    return _engine


def session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        raise RuntimeError("Database engine not configured; call configure_engine() first")
    return _SessionLocal


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a short-lived session."""
    with session_factory()() as session:
        yield session
