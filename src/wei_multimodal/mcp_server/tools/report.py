"""FastMCP wrapper for deterministic research report generation."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import Context
from pydantic import UUID4

from wei_multimodal.mcp_server.contracts import GenerateReportRequest, GenerateReportResponse
from wei_multimodal.mcp_server.contracts.inputs import CaseRef, GenerateReportInput
from wei_multimodal.mcp_server.errors import ErrorCode
from wei_multimodal.mcp_server.services import RuntimeDependencies, generate_report

from .common import execute_tool, runtime_from_context


async def report_tool(
    contract_version: Literal["1.1.0"],
    request_id: UUID4,
    trace_id: UUID4,
    case_ref: CaseRef,
    input: GenerateReportInput,
    ctx: Context[Any, RuntimeDependencies, Any],
) -> GenerateReportResponse:
    """Build a fixed-section escaped HTML report from server-side artifacts."""

    runtime = runtime_from_context(ctx)
    request = GenerateReportRequest(
        contract_version=contract_version,
        request_id=request_id,
        trace_id=trace_id,
        case_ref=case_ref,
        input=input,
    )
    return await execute_tool(
        request=request,
        runtime=runtime,
        tool_name="crc_lnm_generate_report",
        response_type=GenerateReportResponse,
        operation=lambda: generate_report(request, runtime),
        internal_error=ErrorCode.REPORT_GENERATION_FAILURE,
    )


__all__ = ["report_tool"]
