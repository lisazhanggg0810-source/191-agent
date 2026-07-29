"""Registration entry point for the exact six-tool core FastMCP surface."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .case_qc import case_qc_tool
from .common import READ_ONLY_ANNOTATIONS
from .ct_features import ct_features_tool
from .model_info import model_info_tool
from .pathology_features import pathology_features_tool
from .prediction import prediction_tool
from .report import report_tool


def register_core_tools(mcp: FastMCP) -> None:
    """Register exactly six lifespan-context-backed, read-only core tools."""

    registrations = (
        (
            model_info_tool,
            "crc_lnm_get_model_info",
            "Return integrity-checked model dimensions, version, hashes and threshold.",
        ),
        (
            case_qc_tool,
            "crc_lnm_case_data_qc",
            "Validate deidentified case integrity, privacy and required modalities.",
        ),
        (
            ct_features_tool,
            "crc_lnm_prepare_ct_features",
            "Validate and retain approved precomputed 1409-dimensional CT features.",
        ),
        (
            pathology_features_tool,
            "crc_lnm_prepare_pathology_features",
            "Validate and retain approved 768-dimensional pathology features.",
        ),
        (
            prediction_tool,
            "crc_lnm_predict_multimodal",
            "Run the locked five-member multimodal ensemble after compatibility gates.",
        ),
        (
            report_tool,
            "crc_lnm_generate_report",
            "Generate a deterministic escaped research-assistance report.",
        ),
    )
    for function, name, description in registrations:
        mcp.add_tool(
            function,
            name=name,
            description=description,
            annotations=READ_ONLY_ANNOTATIONS,
            structured_output=True,
        )


__all__ = ["register_core_tools"]
