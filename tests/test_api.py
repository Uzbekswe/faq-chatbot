from __future__ import annotations

from fastapi.testclient import TestClient

from faq_chatbot.config import Settings
from faq_chatbot.domain import IndexFingerprint, Source, StreamEvent
from faq_chatbot.main import create_app


class StaticRetriever:
    async def search(self, query: str, limit: int) -> list[Source]:
        return [Source(id="faq1", question="배송", answer_excerpt="내일 배송", score=0.9)]


def _settings(tmp_path, **overrides):
    return Settings(database_url=f"sqlite:///{tmp_path / 'chat.sqlite'}", **overrides)


def test_config_without_key_and_session_lifecycle(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/v1/config").json() == {
            "ready": False,
            "missing": ["OPENAI_API_KEY", "FAQ_INDEX"],
        }
        assert client.get("/health/ready").status_code == 503
        assert (
            client.post("/api/v1/chat/stream", json={"message": "배송은 언제?"}).status_code == 401
        )
        created = client.post("/api/v1/sessions")
        assert created.status_code == 201
        assert (
            client.post("/api/v1/chat/stream", json={"message": "배송은 언제?"}).status_code == 503
        )
        assert client.get("/api/v1/sessions/current/messages").json() == []
        assert client.delete("/api/v1/sessions/current").status_code == 204
        assert client.get("/api/v1/sessions/current/messages").status_code == 401


def test_sse_contains_sources_delta_and_completion(tmp_path):
    class FakeService:
        async def stream(self, token, question):
            yield StreamEvent(event="sources", data={"sources": []})
            yield StreamEvent(event="delta", data={"text": "배송 안내"})
            yield StreamEvent(event="completed", data={"message_id": "fake", "out_of_scope": False})

    app = create_app(_settings(tmp_path, openai_api_key="not-a-real-key"), service=FakeService())
    with TestClient(app) as client:
        client.post("/api/v1/sessions")
        response = client.post("/api/v1/chat/stream", json={"message": "배송은 언제?"})
    assert response.status_code == 200
    assert "event: sources" in response.text
    assert "event: delta" in response.text
    assert "event: completed" in response.text


def test_injected_service_enables_a_deterministic_browser_server(tmp_path):
    class FakeService:
        async def stream(self, token, question):
            yield StreamEvent(event="sources", data={"sources": []})
            yield StreamEvent(event="delta", data={"text": "테스트"})
            yield StreamEvent(event="completed", data={"message_id": "fake", "out_of_scope": False})

    app = create_app(_settings(tmp_path), service=FakeService())
    with TestClient(app) as client:
        assert client.get("/api/v1/config").json() == {"ready": True, "missing": []}
        assert client.post("/api/v1/sessions").status_code == 201
        streamed = client.post("/api/v1/chat/stream", json={"message": "배송은 언제?"})
    assert "테스트" in streamed.text


def test_configuration_requires_a_valid_index_when_an_api_key_exists(tmp_path):
    index = tmp_path / "chroma"
    index.mkdir()
    (index / "chroma.sqlite3").touch()
    (index / "fingerprint.json").write_text(
        IndexFingerprint(dataset_sha256="dataset", embedding_model="embed").model_dump_json(),
        encoding="utf-8",
    )
    app = create_app(
        _settings(
            tmp_path,
            openai_api_key="not-a-real-key",
            chroma_path=index,
            openai_embedding_model="embed",
        )
    )
    with TestClient(app) as client:
        assert client.get("/api/v1/config").json() == {"ready": True, "missing": []}
