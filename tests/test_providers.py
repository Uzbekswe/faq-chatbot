from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from faq_chatbot.providers import ChromaRetriever, OpenAIAnswerGenerator


@pytest.mark.asyncio
async def test_openai_generator_uses_response_streaming(monkeypatch):
    calls = []

    class Stream:
        def __init__(self):
            self.events = [
                SimpleNamespace(type="response.created"),
                SimpleNamespace(type="response.output_text.delta", delta="안녕"),
            ]

        def __aiter__(self):
            self.iterator = iter(self.events)
            return self

        async def __anext__(self):
            try:
                return next(self.iterator)
            except StopIteration as error:
                raise StopAsyncIteration from error

    class Responses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return Stream()

    monkeypatch.setattr(
        "faq_chatbot.providers.AsyncOpenAI", lambda **kwargs: SimpleNamespace(responses=Responses())
    )
    generator = OpenAIAnswerGenerator("key", "model", 30)
    assert [
        item
        async for item in generator.stream(
            system="rules", history=[], question="질문", safety_identifier="safe"
        )
    ] == ["안녕"]
    assert calls[0]["stream"] is True
    assert calls[0]["safety_identifier"] == "safe"


@pytest.mark.asyncio
async def test_chroma_retriever_maps_cosine_distance(monkeypatch, tmp_path):
    (tmp_path / "fingerprint.json").write_text(
        '{"dataset_sha256":"digest","embedding_model":"embed"}', encoding="utf-8"
    )
    (tmp_path / "chroma.sqlite3").touch()

    class Embeddings:
        def create(self, **kwargs):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    class Collection:
        def query(self, **kwargs):
            return {
                "ids": [["faq1"]],
                "documents": [["배송은 언제?"]],
                "metadatas": [[{"answer": "내일입니다."}]],
                "distances": [[0.1]],
            }

    fake_chroma = SimpleNamespace(
        PersistentClient=lambda path: SimpleNamespace(get_collection=lambda name: Collection())
    )
    monkeypatch.setitem(sys.modules, "chromadb", fake_chroma)
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace(embeddings=Embeddings()))
    found = await ChromaRetriever(
        path=str(tmp_path), api_key="key", embedding_model="embed"
    ).search("배송", 1)
    assert found[0].question == "배송은 언제?"
    assert found[0].score == pytest.approx(0.9)
