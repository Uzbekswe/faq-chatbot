"""OpenAI and Chroma adapters; imports that need optional services stay at the edge."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI

from .domain import IndexFingerprint, Message, Source


def index_is_ready(path: Path, embedding_model: str) -> bool:
    """Check the persisted fingerprint and Chroma database without exposing implementation details."""
    fingerprint_file = path / "fingerprint.json"
    try:
        fingerprint = IndexFingerprint.model_validate_json(
            fingerprint_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    return (
        fingerprint.schema_version == 1
        and fingerprint.metric == "cosine"
        and fingerprint.embedding_model == embedding_model
        and (path / "chroma.sqlite3").is_file()
    )


class OpenAIAnswerGenerator:
    def __init__(self, api_key: str, model: str, timeout_seconds: float) -> None:
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=2)
        self.model = model

    async def stream(
        self, *, system: str, history: Sequence[Message], question: str, safety_identifier: str
    ) -> AsyncIterator[str]:
        input_items: list[dict[str, str]] = [
            {"role": "developer", "content": system},
            *({"role": item.role, "content": item.content} for item in history),
            {"role": "user", "content": question},
        ]
        stream = await self.client.responses.create(
            model=self.model,
            input=cast(Any, input_items),
            stream=True,
            safety_identifier=safety_identifier,
        )
        async for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta


class HistoryQueryRewriter:
    """Conservative local heuristic: only follow-up cues receive prior turn context."""

    async def rewrite(self, question: str, history: Sequence[Message]) -> str:
        followup_cues = ("그것", "그거", "그 경우", "그러면", "이것", "it", "that")
        if history and any(cue in question.lower() for cue in followup_cues):
            previous_user = next(
                (message.content for message in reversed(history) if message.role == "user"), ""
            )
            if previous_user:
                return f"{previous_user} — 후속 질문: {question}"
        return question


class ChromaRetriever:
    def __init__(self, *, path: str, api_key: str, embedding_model: str) -> None:
        self.path, self.api_key, self.embedding_model = path, api_key, embedding_model

    async def search(self, query: str, limit: int) -> Sequence[Source]:
        return await asyncio.to_thread(self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> list[Source]:
        import chromadb
        from openai import OpenAI

        if not index_is_ready(Path(self.path), self.embedding_model):
            raise RuntimeError("FAQ index was built with a different embedding model; rebuild it")
        vector = (
            OpenAI(api_key=self.api_key)
            .embeddings.create(model=self.embedding_model, input=query)
            .data[0]
            .embedding
        )
        result = (
            chromadb.PersistentClient(path=self.path)
            .get_collection("faqs")
            .query(
                query_embeddings=[vector],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
        )
        ids = result["ids"][0] if result["ids"] else []
        documents = result["documents"][0] if result["documents"] else []
        metadata = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        return [
            Source(
                id=faq_id,
                question=str(question),
                answer=str(item["answer"]),
                answer_excerpt=str(item["answer"])[:320],
                score=max(0, min(1, 1 - distance)),
            )
            for faq_id, question, item, distance in zip(
                ids, documents, metadata, distances, strict=True
            )
        ]


class DeterministicAnswerGenerator:
    """A test-only provider with deterministic token boundaries."""

    def __init__(self, answer: str = "테스트 응답입니다.") -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    async def stream(
        self, *, system: str, history: Sequence[Message], question: str, safety_identifier: str
    ) -> AsyncIterator[str]:
        self.calls.append(
            {
                "system": system,
                "history": list(history),
                "question": question,
                "safety_identifier": safety_identifier,
            }
        )
        for token in self.answer.split(" "):
            yield token + " "


def safety_identifier(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]
