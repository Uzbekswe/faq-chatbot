from __future__ import annotations

import asyncio

import pytest

from faq_chatbot.chat import ChatError, ChatService, OUT_OF_SCOPE
from faq_chatbot.domain import Source
from faq_chatbot.providers import DeterministicAnswerGenerator, HistoryQueryRewriter
from faq_chatbot.storage import SQLiteConversationStore


class StaticRetriever:
    def __init__(self, sources: list[Source]):
        self.sources = sources
        self.queries: list[str] = []

    async def search(self, query: str, limit: int) -> list[Source]:
        self.queries.append(query)
        return self.sources[:limit]


@pytest.mark.asyncio
async def test_chat_keeps_raw_message_and_does_not_duplicate_prompt(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chat.sqlite", ttl_hours=24)
    token, _ = store.create_session()
    generator = DeterministicAnswerGenerator("배송은 내일입니다.")
    retriever = StaticRetriever(
        [Source(id="faq1", question="배송", answer_excerpt="내일 배송", score=0.9)]
    )
    service = ChatService(
        store=store,
        retriever=retriever,
        generator=generator,
        rewriter=HistoryQueryRewriter(),
        top_k=4,
        similarity_threshold=0.35,
        rate_limit=20,
    )
    events = [event async for event in service.stream(token, "배송은 언제예요?")]
    assert events[0].event == "sources"
    assert events[-1].event == "completed"
    assert [event.event for event in events].count("delta") >= 1
    saved = store.messages(token)
    assert saved[0].content == "배송은 언제예요?"
    assert saved[0].retrieval_query == "배송은 언제예요?"
    assert [item.content for item in generator.calls[0]["history"]].count("배송은 언제예요?") == 0
    assert generator.calls[0]["question"] == "배송은 언제예요?"


@pytest.mark.asyncio
async def test_irrelevant_question_refuses_without_generator(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chat.sqlite", ttl_hours=24)
    token, _ = store.create_session()
    generator = DeterministicAnswerGenerator()
    service = ChatService(
        store=store,
        retriever=StaticRetriever([]),
        generator=generator,
        rewriter=HistoryQueryRewriter(),
        top_k=4,
        similarity_threshold=0.35,
        rate_limit=20,
    )
    events = [event async for event in service.stream(token, "날씨는 어때요?")]
    assert events[1].data["text"] == OUT_OF_SCOPE
    assert generator.calls == []


@pytest.mark.asyncio
async def test_rate_limit_and_contextual_rewrite(tmp_path):
    store = SQLiteConversationStore(tmp_path / "chat.sqlite", ttl_hours=24)
    token, _ = store.create_session()
    source = Source(id="faq1", question="배송", answer_excerpt="내일 배송", score=0.9)
    generator = DeterministicAnswerGenerator()
    service = ChatService(
        store=store,
        retriever=StaticRetriever([source]),
        generator=generator,
        rewriter=HistoryQueryRewriter(),
        top_k=4,
        similarity_threshold=0.35,
        rate_limit=1,
    )
    _ = [event async for event in service.stream(token, "배송은 언제예요?")]
    with pytest.raises(Exception, match="Too many requests"):
        _ = [event async for event in service.stream(token, "그러면 주말에도?")]
    rewritten = await HistoryQueryRewriter().rewrite("그것은요?", store.messages(token))
    assert rewritten.startswith("배송은 언제예요?")


@pytest.mark.asyncio
async def test_only_one_stream_can_run_for_a_session(tmp_path):
    class BlockingGenerator:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, **kwargs):
            self.started.set()
            await self.release.wait()
            yield "완료"

    store = SQLiteConversationStore(tmp_path / "chat.sqlite", ttl_hours=24)
    token, _ = store.create_session()
    generator = BlockingGenerator()
    service = ChatService(
        store=store,
        retriever=StaticRetriever(
            [Source(id="faq1", question="배송", answer_excerpt="내일 배송", score=0.9)]
        ),
        generator=generator,
        rewriter=HistoryQueryRewriter(),
        top_k=4,
        similarity_threshold=0.35,
        rate_limit=20,
    )

    first = service.stream(token, "첫 질문")
    assert (await anext(first)).event == "sources"
    pending = asyncio.create_task(anext(first))
    await generator.started.wait()
    with pytest.raises(ChatError, match="Another answer is already streaming") as error:
        _ = [event async for event in service.stream(token, "두 번째 질문")]
    assert error.value.status_code == 429
    generator.release.set()
    assert (await pending).event == "delta"
    await first.aclose()
