"""Real-browser smoke flow used locally and in CI without an OpenAI key."""

from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import uvicorn
from playwright.sync_api import sync_playwright

from faq_chatbot.chat import ChatService
from faq_chatbot.config import Settings
from faq_chatbot.domain import Message, Source
from faq_chatbot.main import create_app
from faq_chatbot.providers import HistoryQueryRewriter
from faq_chatbot.storage import SQLiteConversationStore


class StaticRetriever:
    async def search(self, query: str, limit: int) -> list[Source]:
        return [
            Source(
                id="faq-shipping",
                question="배송 지연은 어떻게 처리하나요?",
                answer="배송 지연 사유와 새로운 발송 예정일을 구매자에게 안내해 주세요.",
                answer_excerpt="배송 지연 사유와 새로운 발송 예정일을 구매자에게 안내해 주세요.",
                score=0.92,
            )
        ]


class ScenarioGenerator:
    async def stream(
        self, *, system: str, history: Sequence[Message], question: str, safety_identifier: str
    ) -> AsyncIterator[str]:
        if "오류" in question:
            raise RuntimeError("simulated provider failure")
        if "취소" in question:
            yield "처리 중"
            await asyncio.sleep(2)
            yield " 완료"
            return
        yield "배송 지연 안내입니다."


def available_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def wait_until_ready(url: str) -> None:
    from urllib.request import urlopen

    deadline = monotonic() + 10
    while monotonic() < deadline:
        try:
            with urlopen(f"{url}/health/live", timeout=1) as response:  # noqa: S310
                if response.status == 200:
                    return
        except OSError:
            sleep(0.05)
    raise RuntimeError("browser smoke server did not start")


def main() -> None:
    with TemporaryDirectory(prefix="faq-browser-") as temporary:
        root = Path(temporary)
        settings = Settings(
            database_url=f"sqlite:///{root / 'chat.sqlite3'}",
            chroma_path=root / "chroma",
        )
        store = SQLiteConversationStore(settings.database_path, settings.session_ttl_hours)
        service = ChatService(
            store=store,
            retriever=StaticRetriever(),
            generator=ScenarioGenerator(),
            rewriter=HistoryQueryRewriter(),
            top_k=4,
            similarity_threshold=0.35,
            rate_limit=20,
        )
        app = create_app(settings, service=service)
        port = available_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{port}"
        wait_until_ready(url)

        output = Path("output/playwright")
        output.mkdir(parents=True, exist_ok=True)
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={"width": 1280, "height": 1000})
                page.goto(url, wait_until="networkidle")
                page.get_by_text("Ready", exact=True).wait_for()
                page.get_by_role("button", name="상품 등록은 어떻게 하나요?").click()
                assistant = page.locator('.message[data-role="assistant"] .message-body')
                assistant.get_by_text("배송 지연 안내입니다.", exact=False).wait_for()
                page.locator('.message[data-role="assistant"] .sources summary').click()
                page.get_by_text("배송 지연은 어떻게 처리하나요?", exact=True).wait_for()
                page.get_by_role("textbox", name="Ask a SmartStore question in Korean").focus()
                page.screenshot(path=output / "chat-stream.png")

                page.reload(wait_until="networkidle")
                page.get_by_text("배송 지연 안내입니다.", exact=True).wait_for()

                page.get_by_role("textbox", name="Ask a SmartStore question in Korean").fill(
                    "오류 테스트"
                )
                page.get_by_role("textbox", name="Ask a SmartStore question in Korean").press(
                    "Enter"
                )
                page.get_by_text(
                    "The answer service is temporarily unavailable.", exact=True
                ).wait_for()
                page.get_by_role("textbox", name="Ask a SmartStore question in Korean").fill(
                    "다시 배송 질문"
                )
                page.get_by_role("textbox", name="Ask a SmartStore question in Korean").press(
                    "Enter"
                )
                page.get_by_text("배송 지연 안내입니다.", exact=True).last.wait_for()

                page.get_by_role("textbox", name="Ask a SmartStore question in Korean").fill(
                    "취소 테스트"
                )
                page.get_by_role("textbox", name="Ask a SmartStore question in Korean").press(
                    "Enter"
                )
                page.get_by_text("처리 중", exact=True).wait_for()
                page.get_by_role("button", name="Stop", exact=True).click()
                page.get_by_role("button", name="Send message").wait_for()

                page.get_by_role("button", name="New chat", exact=True).click()
                page.get_by_role("heading", name="Clear answers, grounded in the FAQ.").wait_for()

                page.set_viewport_size({"width": 375, "height": 812})
                page.reload(wait_until="networkidle")
                overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
                assert overflow is False, "mobile layout has horizontal overflow"
                page.get_by_role("textbox", name="Ask a SmartStore question in Korean").wait_for()
                page.screenshot(path=output / "chat-mobile.png", full_page=True)
                browser.close()
        finally:
            server.should_exit = True
            thread.join(timeout=5)

        unconfigured_port = available_port()
        unconfigured = uvicorn.Server(
            uvicorn.Config(
                create_app(
                    Settings(
                        database_url=f"sqlite:///{root / 'unconfigured.sqlite3'}",
                        chroma_path=root / "missing-index",
                    )
                ),
                host="127.0.0.1",
                port=unconfigured_port,
                log_level="warning",
            )
        )
        unconfigured_thread = threading.Thread(target=unconfigured.run, daemon=True)
        unconfigured_thread.start()
        unconfigured_url = f"http://127.0.0.1:{unconfigured_port}"
        wait_until_ready(unconfigured_url)
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page(viewport={"width": 1280, "height": 800})
                page.goto(unconfigured_url, wait_until="networkidle")
                page.get_by_text("Setup needed", exact=True).wait_for()
                assert page.get_by_role(
                    "textbox", name="Ask a SmartStore question in Korean"
                ).is_disabled()
                assert page.locator('input[type="password"]').count() == 0
                page.screenshot(path=output / "setup-needed.png", full_page=True)
                browser.close()
        finally:
            unconfigured.should_exit = True
            unconfigured_thread.join(timeout=5)


if __name__ == "__main__":
    main()
