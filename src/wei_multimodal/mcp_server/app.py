"""FastMCP application assembly for the six-tool CRC-LNM core service.

The model, artifact store and case repository are created once in the official
FastMCP lifespan. Tool calls retrieve the immutable runtime from their MCP
``Context`` instead of reconstructing weights for every request.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from wei_multimodal.mcp_server.artifact_store import ArtifactStore
from wei_multimodal.mcp_server.case_repository import CaseRepository
from wei_multimodal.mcp_server.services import RuntimeDependencies
from wei_multimodal.mcp_server.settings import MCPSettings, load_mcp_settings
from wei_multimodal.mcp_server.tools import register_core_tools
from wei_multimodal.service.prediction import PredictionService

DEFAULT_CONFIG_PATH = Path(
    str(
        files("wei_multimodal.resources")
        .joinpath("configs")
        .joinpath("mcp.hosted.yaml")
    )
)


@dataclass(slots=True)
class RuntimeState:
    """Small non-sensitive state used only by live/ready health endpoints."""

    ready: bool = False
    model_version: str | None = None
    runtime: RuntimeDependencies | None = None


def _build_runtime(settings: MCPSettings) -> RuntimeDependencies:
    """Create all long-lived dependencies after validating the deployment bundle."""

    prediction_service = PredictionService(
        settings.bundle_directory,
        device=settings.device,
    )
    info = prediction_service.get_info()
    if not info.integrity_verified or info.member_count != 5:
        raise RuntimeError("The approved deployment bundle is not ready.")
    artifact_store = ArtifactStore(
        settings.artifact_root,
        ttl_seconds=settings.artifact_ttl_seconds,
        max_artifact_bytes=settings.max_artifact_bytes,
        max_artifacts_per_trace=settings.max_artifacts_per_trace,
        max_total_bytes=settings.max_artifact_store_bytes,
    )
    case_repository = CaseRepository(
        settings.case_root,
        max_file_bytes=settings.max_case_file_bytes,
    )

    return RuntimeDependencies.create(
        prediction_service=prediction_service,
        artifact_store=artifact_store,
        case_repository=case_repository,
        tool_timeout_seconds=settings.tool_timeout_seconds,
        max_concurrency=settings.max_concurrency,
    )


def create_mcp(settings: MCPSettings) -> tuple[FastMCP, RuntimeState]:
    """Construct a six-tool FastMCP server and its readiness state."""

    state = RuntimeState()

    @asynccontextmanager
    async def lifespan(_: FastMCP[Any]) -> AsyncIterator[RuntimeDependencies]:
        owns_runtime = state.runtime is None
        runtime = state.runtime
        if runtime is None:
            runtime = await asyncio.to_thread(_build_runtime, settings)
            state.runtime = runtime
            state.model_version = runtime.prediction_service.get_info().model_version
            state.ready = True
        try:
            yield runtime
        finally:
            if owns_runtime:
                state.ready = False
                state.model_version = None
                state.runtime = None
                runtime.artifact_store.cleanup_expired()

    mcp = FastMCP(
        name="crc_lnm_mcp",
        instructions=(
            "Research-assistance only. Accept only allowlisted case_ref values "
            "whose packages contain precomputed 1409-dimensional CT features, "
            "768-dimensional pathology features, and clinical fields. Do not "
            "request or claim support for raw NIfTI, DICOM, WSI, or patch files. "
            "Run case QC before feature preparation, then require both modalities "
            "before prediction."
        ),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        lifespan=lifespan,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(settings.allowed_hosts),
            allowed_origins=list(settings.allowed_origins),
        ),
    )
    register_core_tools(mcp)
    return mcp, state


class StaticBearerAuthMiddleware:
    """Pure ASGI bearer check that does not buffer Streamable HTTP responses."""

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self._app = app
        self._token = token

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if self._token and scope.get("type") == "http" and scope.get("path", "").startswith("/mcp"):
            headers = {
                key.lower(): value
                for key, value in scope.get("headers", [])
            }
            supplied = headers.get(b"authorization", b"").decode("latin-1")
            expected = f"Bearer {self._token}"
            if not hmac.compare_digest(supplied, expected):
                body = json.dumps(
                    {"error": "unauthorized"},
                    separators=(",", ":"),
                ).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                            (b"www-authenticate", b"Bearer"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self._app(scope, receive, send)


class RequestBodyLimitMiddleware:
    """Reject oversized MCP request bodies before JSON parsing or tool execution."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope.get("type") != "http" or not scope.get("path", "").startswith("/mcp"):
            await self._app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self._max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send)
                return

        buffered: list[Message] = []
        total = 0
        more_body = True
        while more_body:
            message = await receive()
            buffered.append(message)
            if message.get("type") != "http.request":
                break
            total += len(message.get("body", b""))
            if total > self._max_bytes:
                await self._reject(send)
                return
            more_body = bool(message.get("more_body", False))

        async def replay() -> Message:
            if buffered:
                return buffered.pop(0)
            return await receive()

        await self._app(scope, replay, send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = b'{"error":"request_too_large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_http_app(settings: MCPSettings) -> Any:
    """Return the Streamable HTTP ASGI app with health routes and optional auth."""

    mcp, state = create_mcp(settings)
    app = mcp.streamable_http_app()
    session_manager_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def http_lifespan(starlette_app: Any) -> AsyncIterator[None]:
        runtime = await asyncio.to_thread(_build_runtime, settings)
        state.runtime = runtime
        state.model_version = runtime.prediction_service.get_info().model_version
        state.ready = True
        try:
            async with session_manager_lifespan(starlette_app):
                yield
        finally:
            state.ready = False
            state.model_version = None
            state.runtime = None
            runtime.artifact_store.cleanup_expired()

    app.router.lifespan_context = http_lifespan

    async def live(_: Request) -> Response:
        return JSONResponse({"status": "live"})

    async def ready(_: Request) -> Response:
        if not state.ready:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse(
            {"status": "ready", "model_version": state.model_version}
        )

    app.routes.insert(0, Route("/health/ready", ready, methods=["GET"]))
    app.routes.insert(0, Route("/health/live", live, methods=["GET"]))

    token = os.getenv("CRC_LNM_MCP_BEARER_TOKEN")
    if settings.require_bearer_token and not token:
        raise RuntimeError("Production bearer token is required.")
    limited_app = RequestBodyLimitMiddleware(app, settings.max_request_body_bytes)
    return StaticBearerAuthMiddleware(limited_app, token)


def load_default_settings(path: Path | None = None) -> MCPSettings:
    """Load the configured service without depending on the process working directory."""

    selected = path or Path(os.getenv("CRC_LNM_MCP_CONFIG", DEFAULT_CONFIG_PATH))
    return load_mcp_settings(selected)


__all__ = [
    "RuntimeState",
    "RequestBodyLimitMiddleware",
    "StaticBearerAuthMiddleware",
    "create_http_app",
    "create_mcp",
    "load_default_settings",
]
