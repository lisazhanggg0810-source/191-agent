from __future__ import annotations

import asyncio
import json
from pathlib import Path

from wei_multimodal.mcp_server.app import create_mcp
from wei_multimodal.mcp_server.contracts import PrepareCTRequest
from wei_multimodal.mcp_server.settings import load_mcp_settings

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_release_ct_schema_exposes_only_precomputed() -> None:
    encoded = json.dumps(PrepareCTRequest.model_json_schema(), ensure_ascii=False)
    assert '"const": "precomputed"' in encoded
    assert "ct_package" not in encoded


def test_release_contains_no_sidecar_or_python_cache() -> None:
    assert not (
        RELEASE_ROOT
        / "src/wei_multimodal/mcp_server/services/dicom_sidecar_client.py"
    ).exists()
    assert not list(RELEASE_ROOT.rglob("__pycache__"))
    assert not list(RELEASE_ROOT.rglob("*.pyc"))
    source_text = "\n".join(
        path.read_text("utf-8")
        for path in (RELEASE_ROOT / "src").rglob("*.py")
    )
    assert "crc_lnm_dicom_mcp" not in source_text
    assert "DICOM_MCP_" not in source_text


def test_release_registers_exactly_six_tools() -> None:
    settings = load_mcp_settings(RELEASE_ROOT / "configs/mcp.yaml")
    mcp, _ = create_mcp(settings)
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert names == {
        "crc_lnm_get_model_info",
        "crc_lnm_case_data_qc",
        "crc_lnm_prepare_ct_features",
        "crc_lnm_prepare_pathology_features",
        "crc_lnm_predict_multimodal",
        "crc_lnm_generate_report",
    }


def test_release_cannot_enable_dicom_mode_from_environment() -> None:
    settings = load_mcp_settings(
        RELEASE_ROOT / "configs/mcp.yaml",
        environ={"DICOM_RADIOMICS_MODE": "enabled"},
    )
    assert settings.dicom_radiomics_mode == "off"
