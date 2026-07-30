"""Transport helpers shared by all FastMCP tool wrappers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations

from wei_multimodal.mcp_server.contracts import (
    ErrorPayload,
    ProvenancePayload,
    StatusCode,
    StatusName,
    StatusPayload,
    StrictContract,
    ToolResponse,
    WarningPayload,
)
from wei_multimodal.mcp_server.contracts.inputs import BaseRequest
from wei_multimodal.mcp_server.errors import ContractError, ErrorCode
from wei_multimodal.mcp_server.execution import tool_execution_deadline
from wei_multimodal.mcp_server.services import RuntimeDependencies

SERVICE_VERSION = "1.0.0"
LOGGER = logging.getLogger(__name__)
READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
ARTIFACT_WRITING_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)


def runtime_from_context(ctx: Context[Any, RuntimeDependencies, Any]) -> RuntimeDependencies:
    """Retrieve the lifespan-owned runtime; never instantiate models per request."""

    runtime = ctx.request_context.lifespan_context
    if not isinstance(runtime, RuntimeDependencies):
        raise ContractError(
            ErrorCode.SERVICE_UNAVAILABLE,
            message="Core MCP runtime is unavailable.",
        )
    return runtime


def request_sha256(request: StrictContract) -> str:
    """Hash the validated logical request deterministically for audit provenance."""

    raw = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _provenance(
    request: StrictContract,
    runtime: RuntimeDependencies,
    *,
    started: float,
) -> ProvenancePayload:
    """Build a patient-data-free provenance record from the loaded bundle."""

    info = runtime.prediction_service.get_info()
    return ProvenancePayload(
        service_version=SERVICE_VERSION,
        model_version=info.model_version,
        model_schema_version=info.schema_version,
        model_feature_order_sha256=info.feature_order_sha256,
        input_sha256=request_sha256(request),
        created_at=datetime.now(UTC),
        duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
    )


_SUGGESTIONS: dict[ErrorCode, str] = {
    ErrorCode.MISSING_MODALITY: "Provide the complete approved three-modality case package.",
    ErrorCode.RESEARCH_ID_MISMATCH: "Regenerate the case package with one research identifier.",
    ErrorCode.PII_DETECTED: "Remove direct identifiers and submit a deidentified case package.",
    ErrorCode.INVALID_FILE_FORMAT: "Regenerate the package using manifest schema 1.0.0.",
    ErrorCode.FEATURE_SCHEMA_MISMATCH: "Use features exported with model schema 1.0.0.",
    ErrorCode.NON_FINITE_VALUE: "Regenerate features without NaN or Infinity values.",
    ErrorCode.INVALID_CLINICAL_CATEGORY: "Use a category present in the trained vocabulary.",
    ErrorCode.UNSUPPORTED_SOURCE_MODE: "Use the approved precomputed feature workflow.",
    ErrorCode.QC_REQUIRED_OR_FAILED: "Run case quality control successfully before continuing.",
    ErrorCode.ARTIFACT_EXPIRED: "Rerun the preceding workflow stages.",
    ErrorCode.ARTIFACT_UNAVAILABLE: "Rerun the preceding stage in the same trace.",
    ErrorCode.INVALID_STAGE_ORDER: "Follow the documented core MCP workflow order.",
    ErrorCode.ARTIFACT_TYPE_MISMATCH: "Use the artifact produced by the required preceding tool.",
    ErrorCode.CASE_BINDING_MISMATCH: "Use artifacts produced for this case and trace.",
    ErrorCode.MODEL_INCOMPATIBLE: "Regenerate all features with the loaded model schema.",
    ErrorCode.INFERENCE_FAILURE: "Retry only after the service operator reviews server logs.",
    ErrorCode.REPORT_GENERATION_FAILURE: "Retry report generation once with the same artifacts.",
    ErrorCode.SERVICE_UNAVAILABLE: "Retry after the core service reports ready.",
    ErrorCode.CAPACITY_EXCEEDED: "Retry after temporary artifact capacity becomes available.",
}


async def execute_tool[
    RequestT: BaseRequest,
    DataT: StrictContract,
    ResponseT: ToolResponse[Any],
](
    *,
    request: RequestT,
    runtime: RuntimeDependencies,
    tool_name: str,
    response_type: type[ResponseT],
    operation: Callable[[], DataT],
    internal_error: ErrorCode,
    warning_builder: Callable[[DataT], list[WarningPayload]] | None = None,
) -> ResponseT:
    """Execute one service call and map all outcomes to its exact response model.

    Known ``ContractError`` instances retain their stable public code.  Unexpected
    exceptions are deliberately replaced with a tool-specific fixed error and no
    stack, path or exception text is returned to the caller.
    """

    started = time.perf_counter()
    deadline = time.monotonic() + runtime.tool_timeout_seconds
    try:

        async def run_with_capacity_limit() -> DataT:
            """Retain capacity until a timed-out worker has actually finished."""

            await runtime.tool_semaphore.acquire()

            def run_operation() -> DataT:
                with tool_execution_deadline(deadline):
                    return operation()

            task = asyncio.create_task(asyncio.to_thread(run_operation))

            def release_capacity(completed: asyncio.Task[DataT]) -> None:
                if not completed.cancelled():
                    completed.exception()
                runtime.tool_semaphore.release()

            task.add_done_callback(release_capacity)
            return await asyncio.shield(task)

        data = await asyncio.wait_for(
            run_with_capacity_limit(),
            timeout=runtime.tool_timeout_seconds,
        )
        warnings = warning_builder(data) if warning_builder is not None else []
        code = StatusCode.OK_WITH_WARNINGS if warnings else StatusCode.OK
        status = StatusPayload(
            code=code,
            name=StatusName[code.name],
            message=(
                "Tool completed with review warnings."
                if warnings
                else "Tool completed successfully."
            ),
            severity="warning" if warnings else "info",
            retryable=False,
        )
        return response_type.model_validate(
            {
                "request_id": request.request_id,
                "trace_id": request.trace_id,
                "tool_name": tool_name,
                "status": status,
                "data": data,
                "errors": [],
                "warnings": warnings,
                "provenance": _provenance(request, runtime, started=started),
            }
        )
    except TimeoutError:
        public_error = ContractError(
            ErrorCode.TOOL_TIMEOUT,
            message="Tool execution exceeded the configured time limit.",
        )
    except ContractError as exc:
        public_error = exc
    except Exception:
        LOGGER.exception(
            "Unexpected MCP tool failure",
            extra={"tool_name": tool_name, "trace_id": str(request.trace_id)},
        )
        public_error = ContractError(
            internal_error,
            message={
                ErrorCode.INFERENCE_FAILURE: "Model inference could not be completed.",
                ErrorCode.REPORT_GENERATION_FAILURE: "Report generation could not be completed.",
            }.get(internal_error, "Tool execution could not be completed."),
        )
    code = StatusCode(int(public_error.code))
    return response_type.model_validate(
        {
            "request_id": request.request_id,
            "trace_id": request.trace_id,
            "tool_name": tool_name,
            "status": StatusPayload(
                code=code,
                name=StatusName[code.name],
                message=public_error.message,
                severity="error",
                retryable=bool(public_error.retryable),
            ),
            "data": None,
            "errors": [
                ErrorPayload(
                    field=public_error.field,
                    code=StatusName[code.name],
                    message=public_error.message,
                    suggestion=_SUGGESTIONS.get(ErrorCode(int(public_error.code))),
                )
            ],
            "warnings": [],
            "provenance": _provenance(request, runtime, started=started),
        }
    )


__all__ = [
    "ARTIFACT_WRITING_ANNOTATIONS",
    "READ_ONLY_ANNOTATIONS",
    "execute_tool",
    "runtime_from_context",
]
