from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from wei_multimodal.mcp_server.app import create_mcp
from wei_multimodal.mcp_server.contracts import PrepareCTRequest, build_contract_schemas
from wei_multimodal.mcp_server.settings import load_mcp_settings

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_release_ct_schema_exposes_only_precomputed() -> None:
    encoded = json.dumps(PrepareCTRequest.model_json_schema(), ensure_ascii=False)
    assert '"const": "precomputed"' in encoded
    assert "ct_package" not in encoded


def test_checked_in_contract_schemas_match_precomputed_only_models() -> None:
    generated = build_contract_schemas()
    for name, schema in generated.items():
        path = RELEASE_ROOT / "schemas/mcp/v1" / f"{name}.schema.json"
        assert json.loads(path.read_text(encoding="utf-8")) == schema
        serialized = json.dumps(schema, ensure_ascii=False).lower()
        assert "dicom" not in serialized
        assert "sidecar" not in serialized
        assert "ct_package" not in serialized


def test_release_contains_no_sidecar_or_python_cache() -> None:
    """Check release source code does not contain forbidden content (isolated subprocess)."""
    # Use subprocess to check without being affected by current process bytecode
    check_source = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            f"root = Path({str(RELEASE_ROOT)!r})",
            "checks = []",
            "sidecar_path = root / 'src' / 'wei_multimodal' / 'mcp_server' / 'services'",
            "sidecar_path = sidecar_path / 'dicom_sidecar_client.py'",
            "if sidecar_path.exists():",
            "    checks.append('dicom_sidecar_exists')",
            "src_cache = list((root / 'src').rglob('__pycache__'))",
            "if src_cache:",
            "    checks.append(f'pycache_in_src:{src_cache}')",
            "source_text = '\\n'.join(p.read_text('utf-8') for p in (root / 'src').rglob('*.py'))",
            "if 'crc_lnm_dicom_mcp' in source_text:",
            "    checks.append('dicom_ref_in_source')",
            "sys.exit(0 if not checks else 1)",
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",  # Don't write bytecode
            "-c",
            check_source,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Release check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    source_paths = (RELEASE_ROOT / "src").rglob("*.py")
    source_text = "\n".join(path.read_text("utf-8") for path in source_paths)
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
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    assert tools["crc_lnm_get_model_info"].annotations.readOnlyHint is True
    assert tools["crc_lnm_get_model_info"].annotations.idempotentHint is True
    for name in names - {"crc_lnm_get_model_info"}:
        assert tools[name].annotations.readOnlyHint is False
        assert tools[name].annotations.idempotentHint is False


def test_release_configuration_rejects_removed_dicom_switch() -> None:
    payload = (RELEASE_ROOT / "configs/mcp.yaml").read_text(encoding="utf-8")
    assert "dicom_radiomics_mode" not in payload
