"""Cooperative execution deadlines for synchronous MCP tool operations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from wei_multimodal.mcp_server.errors import ContractError, ErrorCode

_DEADLINE: ContextVar[float | None] = ContextVar("mcp_tool_deadline", default=None)


@contextmanager
def tool_execution_deadline(deadline: float) -> Iterator[None]:
    """Apply one monotonic deadline to blocking work and artifact commits."""

    token = _DEADLINE.set(deadline)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def require_active_execution() -> None:
    """Prevent a timed-out worker from committing an artifact after its deadline."""

    deadline = _DEADLINE.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise ContractError(
            ErrorCode.TOOL_TIMEOUT,
            message="Tool execution exceeded the configured time limit.",
        )


__all__ = ["require_active_execution", "tool_execution_deadline"]
