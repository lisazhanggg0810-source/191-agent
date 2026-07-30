from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from wei_multimodal.mcp_server.artifact_store import ArtifactStore
from wei_multimodal.mcp_server.contracts import GetModelInfoRequest, GetModelInfoResponse
from wei_multimodal.mcp_server.errors import ContractError, ErrorCode
from wei_multimodal.mcp_server.execution import tool_execution_deadline
from wei_multimodal.mcp_server.services import RuntimeDependencies
from wei_multimodal.mcp_server.tools.common import execute_tool


class _PredictionService:
    def get_info(self) -> SimpleNamespace:
        return SimpleNamespace(
            model_version="a" * 64,
            schema_version="1.0.0",
            feature_order_sha256="b" * 64,
        )


def _runtime(timeout: float = 0.05) -> RuntimeDependencies:
    return RuntimeDependencies(
        prediction_service=cast(object, _PredictionService()),
        artifact_store=cast(object, object()),
        case_repository=cast(object, object()),
        tool_timeout_seconds=timeout,
        tool_semaphore=asyncio.Semaphore(1),
    )


def _request() -> GetModelInfoRequest:
    return GetModelInfoRequest(request_id=uuid4(), trace_id=uuid4(), input={})


def test_timeout_retains_capacity_until_blocking_operation_finishes() -> None:
    async def exercise() -> None:
        runtime = _runtime()
        started = threading.Event()
        release = threading.Event()
        second_started = threading.Event()

        def blocked_operation() -> object:
            started.set()
            assert release.wait(timeout=2)
            return object()

        first = await execute_tool(
            request=_request(),
            runtime=runtime,
            tool_name="crc_lnm_get_model_info",
            response_type=GetModelInfoResponse,
            operation=blocked_operation,
            internal_error=ErrorCode.SERVICE_UNAVAILABLE,
        )
        assert first.status.name == "TOOL_TIMEOUT"
        assert started.is_set()

        def second_operation() -> object:
            second_started.set()
            return object()

        second = await execute_tool(
            request=_request(),
            runtime=runtime,
            tool_name="crc_lnm_get_model_info",
            response_type=GetModelInfoResponse,
            operation=second_operation,
            internal_error=ErrorCode.SERVICE_UNAVAILABLE,
        )
        assert second.status.name == "TOOL_TIMEOUT"
        assert second_started.is_set() is False

        release.set()
        await asyncio.wait_for(runtime.tool_semaphore.acquire(), timeout=1)
        runtime.tool_semaphore.release()

    asyncio.run(exercise())


def test_expired_execution_cannot_commit_an_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with tool_execution_deadline(time.monotonic() - 1), pytest.raises(ContractError) as exc_info:
        store.put(
            artifact_type="case_qc",
            trace_id="trace-1",
            case_binding_sha256="a" * 64,
            payload={"passed": True},
        )
    assert exc_info.value.code == ErrorCode.TOOL_TIMEOUT


def test_execution_expiring_during_serialization_leaves_no_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    original_serialize = ArtifactStore._serialize_payload

    def delayed_serialize(payload: object) -> tuple[bytes, str]:
        time.sleep(0.05)
        return original_serialize(payload)

    monkeypatch.setattr(
        ArtifactStore,
        "_serialize_payload",
        staticmethod(delayed_serialize),
    )

    with tool_execution_deadline(time.monotonic() + 0.01), pytest.raises(ContractError) as exc_info:
        store.put(
            artifact_type="case_qc",
            trace_id="trace-1",
            case_binding_sha256="a" * 64,
            payload={"passed": True},
        )

    assert exc_info.value.code == ErrorCode.TOOL_TIMEOUT
    assert list(tmp_path.iterdir()) == []
