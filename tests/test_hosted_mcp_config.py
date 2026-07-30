from __future__ import annotations

import json
import re
import tomllib
from importlib.resources import files
from pathlib import Path

from wei_multimodal.mcp_server.app import load_default_settings

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_modelscope_import_config_has_standard_mcp_servers_root() -> None:
    payload = json.loads(
        (RELEASE_ROOT / "configs/modelscope-mcp.json").read_text(encoding="utf-8")
    )

    assert set(payload) == {"mcpServers"}
    server = payload["mcpServers"]["crc-lnm-research-assistant"]
    assert server["command"] == "uvx"
    assert server["args"] == [
        "crc-lnm-medical-agent@latest",
        "--transport",
        "stdio",
    ]


def test_hosted_uvx_package_name_is_a_packaged_console_script() -> None:
    project = tomllib.loads((RELEASE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["crc-lnm-medical-agent"] == (
        "wei_multimodal.mcp_server.__main__:main"
    )


def test_root_readme_exposes_the_same_parseable_stdio_configuration() -> None:
    readme = (RELEASE_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"```json\n(?P<config>\{.*?\})\n```", readme, flags=re.DOTALL)
    assert match is not None
    documented = json.loads(match.group("config"))
    configured = json.loads(
        (RELEASE_ROOT / "configs/modelscope-mcp.json").read_text(encoding="utf-8")
    )
    assert documented == configured


def test_dockerfile_uses_the_locked_dependency_graph_and_keeps_schemas() -> None:
    dockerfile = (RELEASE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "uv sync --locked --no-dev --extra mcp" in dockerfile
    assert "COPY schemas /app/schemas" in dockerfile
    assert "pip install" not in dockerfile


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
