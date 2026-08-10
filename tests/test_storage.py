from __future__ import annotations

from datetime import UTC, datetime, timedelta

from faq_chatbot.domain import Message
from faq_chatbot.storage import SQLiteConversationStore


def test_sessions_persist_and_delete_in_isolation(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chat.sqlite", ttl_hours=24)
    first_token, first = store.create_session()
    second_token, second = store.create_session()
    message = Message(id="one", role="user", content="hello", created_at=datetime.now(UTC))
    store.save_message(first_token, message)
    assert [item.content for item in store.messages(first_token)] == ["hello"]
    assert store.messages(second_token) == []
    assert store.delete_session(first_token) is True
    assert store.get_session(first_token) is None
    assert store.get_session(second_token).id == second.id
    assert first.id != second.id


def test_rate_limit_is_per_session(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chat.sqlite", ttl_hours=24)
    token, _ = store.create_session()
    assert store.rate_limit_ok(token, 1)
    assert not store.rate_limit_ok(token, 1)


def test_sessions_survive_reopen_and_expire_with_messages(monkeypatch, tmp_path):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr("faq_chatbot.storage._now", lambda: now)
    path = tmp_path / "chat.sqlite"
    store = SQLiteConversationStore(path, ttl_hours=24)
    token, _ = store.create_session()
    store.save_message(
        token,
        Message(id="persisted", role="user", content="hello", created_at=now),
    )

    reopened = SQLiteConversationStore(path, ttl_hours=24)
    assert reopened.get_session(token) is not None
    assert [message.id for message in reopened.messages(token)] == ["persisted"]

    now += timedelta(hours=25)
    assert reopened.get_session(token) is None
    assert reopened.messages(token) == []
