"""Validated domain data and provider boundaries."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class FAQRecord(BaseModel):
    id: str = Field(min_length=8)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    source: str = "smartstore"
    content_hash: str = Field(min_length=8)


class Source(BaseModel):
    id: str
    question: str
    answer_excerpt: str
    answer: str = Field(default="", exclude=True, repr=False)
    score: float = Field(ge=0, le=1)


class Message(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    retrieval_query: str | None = None
    sources: list[Source] = Field(default_factory=list)


class Session(BaseModel):
    id: str
    created_at: datetime
    expires_at: datetime


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=2_000)


class IndexFingerprint(BaseModel):
    schema_version: int = 1
    dataset_sha256: str
    embedding_model: str
    metric: Literal["cosine"] = "cosine"


class StreamEvent(BaseModel):
    event: Literal["sources", "delta", "completed", "error"]
    data: dict[str, object]


class Retriever(Protocol):
    async def search(self, query: str, limit: int) -> Sequence[Source]: ...


class AnswerGenerator(Protocol):
    def stream(
        self, *, system: str, history: Sequence[Message], question: str, safety_identifier: str
    ) -> AsyncIterator[str]: ...


class QueryRewriter(Protocol):
    async def rewrite(self, question: str, history: Sequence[Message]) -> str: ...


class ConversationStore(Protocol):
    def create_session(self) -> tuple[str, Session]: ...
    def get_session(self, token: str) -> Session | None: ...
    def delete_session(self, token: str) -> bool: ...
    def messages(self, token: str) -> list[Message]: ...
    def save_message(self, token: str, message: Message) -> None: ...
    def rate_limit_ok(self, token: str, limit: int) -> bool: ...
