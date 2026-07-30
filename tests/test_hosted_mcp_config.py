from __future__ import annotations

import json
import tempfile
from pathlib import Path

from wei_multimodal.mcp_server.app import load_default_settings
from wei_multimodal.mcp_server.settings import load_mcp_settings

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_modelscope_import_config_has_standard_mcp_servers_root() -> None:
    payload = json.loads(
        (RELEASE_ROOT / "configs/modelscope-mcp.json").read_text(encoding="utf-8")
    )

    assert set(payload) == {"mcpServers"}
    server = payload["mcpServers"]["crc-lnm-research-assistant"]
    assert server["command"] == "uvx"
    assert server["args"] == [
        "--index",
        "https://download.pytorch.org/whl/cpu",
        "--from",
        "crc-lnm-medical-agent==1.0.1",
        "crc-lnm-mcp",
        "--transport",
        "stdio",
    ]


def test_packaged_hosted_config_is_self_contained_and_writable() -> None:
    settings = load_default_settings()

    assert settings.require_bearer_token is False
    assert settings.bundle_directory.is_dir()
    assert settings.case_root.is_dir()
    assert len(list(settings.bundle_directory.rglob("model_state.pt"))) == 5
    assert settings.artifact_root == Path(tempfile.gettempdir()) / "crc_lnm_artifacts"


def test_hosted_stdio_config_uses_only_repo_relative_paths() -> None:
    settings = load_mcp_settings(RELEASE_ROOT / "configs/mcp.local.yaml")

    assert settings.require_bearer_token is False
    assert settings.bundle_directory == RELEASE_ROOT / "models/deployment_bundle"
    assert settings.case_root == RELEASE_ROOT / "demo/cases"
    assert settings.artifact_root == RELEASE_ROOT / "artifacts_local"
