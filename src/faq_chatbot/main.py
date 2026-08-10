"""FastAPI entrypoint and HTTP boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .chat import ChatError, ChatService
from .config import Settings, get_settings
from .domain import ChatRequest, StreamEvent
from .providers import ChromaRetriever, HistoryQueryRewriter, OpenAIAnswerGenerator, index_is_ready
from .storage import SQLiteConversationStore

COOKIE_NAME = "faq_session"


class UnconfiguredService:
    async def stream(self, token: str, question: str) -> AsyncIterator[StreamEvent]:
        raise ChatError("not_configured", "The chatbot is not configured yet.", 503)
        yield  # pragma: no cover


def _sse(event: StreamEvent) -> str:
    return f"event: {event.event}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"


async def _event_stream(
    service: ChatService | UnconfiguredService, token: str, question: str
) -> AsyncIterator[str]:
    try:
        async for event in service.stream(token, question):
            yield _sse(event)
    except ChatError as error:
        yield _sse(
            StreamEvent(
                event="error",
                data={"code": error.code, "message": error.message, "retryable": error.retryable},
            )
        )
    except asyncio.CancelledError:
        raise


def _build_service(
    settings: Settings, store: SQLiteConversationStore
) -> ChatService | UnconfiguredService:
    if not settings.is_configured:
        return UnconfiguredService()
    key = settings.openai_api_key.get_secret_value()  # type: ignore[union-attr]
    return ChatService(
        store=store,
        retriever=ChromaRetriever(
            path=str(settings.chroma_path),
            api_key=key,
            embedding_model=settings.openai_embedding_model,
        ),
        generator=OpenAIAnswerGenerator(
            key, settings.openai_chat_model, settings.provider_timeout_seconds
        ),
        rewriter=HistoryQueryRewriter(),
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
        rate_limit=settings.session_rate_limit,
    )


def _missing_configuration(settings: Settings) -> list[str]:
    missing = settings.missing_configuration
    if not index_is_ready(settings.chroma_path, settings.openai_embedding_model):
        missing.append("FAQ_INDEX")
    return missing


def create_app(
    settings: Settings | None = None, service: ChatService | UnconfiguredService | None = None
) -> FastAPI:
    settings = settings or get_settings()
    store = SQLiteConversationStore(settings.database_path, settings.session_ttl_hours)
    configured = (
        settings.is_configured and not _missing_configuration(settings)
    ) or service is not None
    active_service = service or _build_service(settings, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = store
        app.state.service = active_service
        yield

    app = FastAPI(title="FAQ Chatbot", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.service = active_service
    app.state.configured = configured
    web_path = Path(__file__).parent / "web"
    if web_path.exists():
        app.mount("/static", StaticFiles(directory=web_path), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        html = web_path / "index.html"
        if html.exists():
            return FileResponse(html)
        return JSONResponse({"name": "FAQ Chatbot", "ready": configured})

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready(request: Request) -> JSONResponse:
        configured = request.app.state.configured
        # A missing OpenAI key is an intentional local configuration state, not a crash.
        missing = [] if configured else _missing_configuration(request.app.state.settings)
        return JSONResponse(
            status_code=200 if configured else 503,
            content={"ready": configured, "missing": missing},
        )

    @app.get("/api/v1/config")
    async def config(request: Request) -> dict[str, object]:
        current: Settings = request.app.state.settings
        configured = request.app.state.configured
        return {
            "ready": configured,
            "missing": [] if configured else _missing_configuration(current),
        }

    @app.post("/api/v1/sessions", status_code=201)
    async def create_session(request: Request, response: Response) -> dict[str, object]:
        token, session = request.app.state.store.create_session()
        response.set_cookie(
            COOKIE_NAME,
            token,
            httponly=True,
            samesite="lax",
            secure=request.app.state.settings.cookie_secure,
            max_age=request.app.state.settings.session_ttl_hours * 3600,
            path="/",
        )
        return {"id": session.id, "expires_at": session.expires_at.isoformat()}

    def require_token(token: str | None) -> str:
        if not token:
            raise HTTPException(status_code=401, detail="Start a new conversation first.")
        return token

    @app.get("/api/v1/sessions/current/messages")
    async def get_messages(
        request: Request, faq_session: str | None = Cookie(default=None)
    ) -> list[dict[str, object]]:
        token = require_token(faq_session)
        if request.app.state.store.get_session(token) is None:
            raise HTTPException(
                status_code=401, detail="Session expired. Start a new conversation."
            )
        return [
            message.model_dump(mode="json") for message in request.app.state.store.messages(token)
        ]

    @app.delete("/api/v1/sessions/current", status_code=204)
    async def delete_session(
        request: Request, response: Response, faq_session: str | None = Cookie(default=None)
    ) -> Response:
        token = require_token(faq_session)
        request.app.state.store.delete_session(token)
        response.delete_cookie(COOKIE_NAME, path="/")
        return Response(status_code=204)

    @app.post("/api/v1/chat/stream")
    async def chat_stream(
        request: Request, body: ChatRequest, faq_session: str | None = Cookie(default=None)
    ) -> StreamingResponse:
        token = require_token(faq_session)
        if not request.app.state.configured:
            raise HTTPException(status_code=503, detail="The chatbot is not configured yet.")
        if len(body.message) > request.app.state.settings.max_message_chars:
            raise HTTPException(status_code=422, detail="Message is too long.")
        return StreamingResponse(
            _event_stream(request.app.state.service, token, body.message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()


def run() -> None:
    uvicorn.run("faq_chatbot.main:app", host="0.0.0.0", port=8000, reload=True)
