from __future__ import annotations

import json
import re
import tomllib
from importlib.resources import files
from pathlib import Path

from wei_multimodal.mcp_server.__main__ import _parser
from wei_multimodal.mcp_server.app import load_default_settings
from wei_multimodal.mcp_server.settings import load_mcp_settings

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _modelscope_import_config() -> dict[str, object]:
    return json.loads(
        (RELEASE_ROOT / "configs/modelscope-mcp.json").read_text(encoding="utf-8")
    )


def test_modelscope_import_config_has_standard_mcp_servers_root() -> None:
    payload = _modelscope_import_config()

    assert set(payload) == {"mcpServers"}
    server = payload["mcpServers"]["crc-lnm-research-assistant"]
    assert server["command"] == "uvx"
    assert server["args"] == [
        "--no-progress",
        "--torch-backend",
        "cpu",
        "--from",
        "crc-lnm-medical-agent==1.0.6",
        "crc-lnm-medical-agent",
    ]
    assert "env" not in server


def test_root_readme_exposes_same_parseable_modelscope_configuration() -> None:
    readme = (RELEASE_ROOT / "README.md").read_text(encoding="utf-8")
    json_blocks = re.findall(r"```json\s*\n(.*?)\n```", readme, flags=re.DOTALL)

    assert len(json_blocks) == 1
    assert json.loads(json_blocks[0]) == _modelscope_import_config()
    assert (
        "https://github.com/lisazhanggg0810-source/191-agent/blob/main/README.md"
        in readme
    )


def test_hosted_console_entry_point_defaults_to_stdio() -> None:
    assert _parser().parse_args([]).transport == "stdio"


def test_hosted_uvx_package_name_is_a_packaged_console_script() -> None:
    project = tomllib.loads((RELEASE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == "1.0.6"
    assert project["project"]["scripts"]["crc-lnm-medical-agent"] == (
        "wei_multimodal.mcp_server.__main__:main"
    )


def test_hosted_stdio_config_uses_only_repo_relative_paths() -> None:
    settings = load_mcp_settings(RELEASE_ROOT / "configs/mcp.local.yaml")

    assert settings.require_bearer_token is False
    assert settings.bundle_directory == RELEASE_ROOT / "models/deployment_bundle"
    assert settings.case_root == RELEASE_ROOT / "artifacts_local/cases"
    assert settings.artifact_root == RELEASE_ROOT / "artifacts_local"
    assert settings.case_package_jsonl == RELEASE_ROOT / "data/release_case_package_groups.jsonl"


def test_hosted_stdio_defaults_use_packaged_immutable_assets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CRC_LNM_MCP_CONFIG", raising=False)
    monkeypatch.setenv("CRC_LNM_MCP_RUNTIME_ROOT", str(tmp_path / "runtime"))
    settings = load_default_settings()
    assets = files("wei_multimodal.mcp_server.runtime_assets")

    assert settings.require_bearer_token is False
    assert settings.bundle_directory == Path(str(assets.joinpath("models/deployment_bundle")))
    assert settings.case_package_jsonl == Path(
        str(assets.joinpath("release_case_package_groups.jsonl"))
    )
    assert settings.case_root == tmp_path / "runtime/cases"
    assert settings.artifact_root == tmp_path / "runtime/artifacts"
    assert settings.bundle_directory.is_dir()
    assert settings.case_package_jsonl.is_file()
