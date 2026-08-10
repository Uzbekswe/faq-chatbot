"""SQLite storage for opaque anonymous browser sessions."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from .domain import Message, Session, Source


def _now() -> datetime:
    return datetime.now(UTC)


class SQLiteConversationStore:
    def __init__(self, path: Path, ttl_hours: int) -> None:
        self.path = path
        self.ttl = timedelta(hours=ttl_hours)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    request_window_start TEXT,
                    request_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL REFERENCES sessions(token_hash) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    retrieval_query TEXT,
                    sources_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS messages_session_time ON messages(token_hash, created_at);
                """
            )

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _purge_expired(self, db: sqlite3.Connection) -> None:
        db.execute("DELETE FROM sessions WHERE expires_at <= ?", (_now().isoformat(),))

    def create_session(self) -> tuple[str, Session]:
        token = secrets.token_urlsafe(32)
        token_hash = self._hash(token)
        now = _now()
        session = Session(id=str(uuid4()), created_at=now, expires_at=now + self.ttl)
        with self._connect() as db:
            self._purge_expired(db)
            db.execute(
                "INSERT INTO sessions(token_hash, session_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (
                    token_hash,
                    session.id,
                    session.created_at.isoformat(),
                    session.expires_at.isoformat(),
                ),
            )
        return token, session

    def _find(self, token: str) -> sqlite3.Row | None:
        with self._connect() as db:
            self._purge_expired(db)
            row = db.execute(
                "SELECT * FROM sessions WHERE token_hash = ?", (self._hash(token),)
            ).fetchone()
            if row:
                expires_at = (_now() + self.ttl).isoformat()
                db.execute(
                    "UPDATE sessions SET expires_at = ? WHERE token_hash = ?",
                    (expires_at, self._hash(token)),
                )
            return row

    def get_session(self, token: str) -> Session | None:
        row = self._find(token)
        if row is None:
            return None
        # Sliding expiry is the current expiry, rather than the pre-update database row.
        return Session(
            id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=_now() + self.ttl,
        )

    def delete_session(self, token: str) -> bool:
        with self._connect() as db:
            result = db.execute("DELETE FROM sessions WHERE token_hash = ?", (self._hash(token),))
            return result.rowcount > 0

    def messages(self, token: str) -> list[Message]:
        if self._find(token) is None:
            return []
        import json

        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM messages WHERE token_hash = ? ORDER BY created_at",
                (self._hash(token),),
            ).fetchall()
        return [
            Message(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                created_at=datetime.fromisoformat(row["created_at"]),
                retrieval_query=row["retrieval_query"],
                sources=[Source.model_validate(item) for item in json.loads(row["sources_json"])],
            )
            for row in rows
        ]

    def save_message(self, token: str, message: Message) -> None:
        import json

        if self._find(token) is None:
            raise KeyError("anonymous session no longer exists")
        with self._connect() as db:
            db.execute(
                """INSERT INTO messages(id, token_hash, role, content, created_at, retrieval_query, sources_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    message.id,
                    self._hash(token),
                    message.role,
                    message.content,
                    message.created_at.isoformat(),
                    message.retrieval_query,
                    json.dumps([item.model_dump() for item in message.sources], ensure_ascii=False),
                ),
            )

    def rate_limit_ok(self, token: str, limit: int) -> bool:
        now = _now()
        token_hash = self._hash(token)
        with self._connect() as db:
            self._purge_expired(db)
            row = db.execute(
                "SELECT request_window_start, request_count FROM sessions WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return False
            start = (
                datetime.fromisoformat(row["request_window_start"])
                if row["request_window_start"]
                else now
            )
            count = row["request_count"] if now - start < timedelta(minutes=1) else 0
            if count >= limit:
                return False
            db.execute(
                "UPDATE sessions SET request_window_start = ?, request_count = ? WHERE token_hash = ?",
                (now.isoformat() if count == 0 else start.isoformat(), count + 1, token_hash),
            )
        return True
