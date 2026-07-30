"""核心 MCP 的非秘密配置与操作员环境覆盖。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MCPSettings(BaseModel):
    """经过验证且不可变的核心服务配置。

    YAML 只允许非秘密项。Bearer token 等凭据由实际 HTTP 鉴权层直接从环境或 secret
    mount 获取，绝不保存在本模型或日志中。DICOM 模式本阶段固定默认 ``off``。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_directory: Path
    case_root: Path
    artifact_root: Path
    case_package_jsonl: Path | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    device: Literal["cpu"] = "cpu"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    require_bearer_token: bool = False
    artifact_ttl_seconds: int = Field(default=1800, ge=60, le=86_400)
    max_request_body_bytes: int = Field(default=1_048_576, ge=1024, le=67_108_864)
    max_case_file_bytes: int = Field(default=16_777_216, ge=1024, le=67_108_864)
    max_artifact_bytes: int = Field(default=16_777_216, ge=1024, le=67_108_864)
    max_artifacts_per_trace: int = Field(default=10, ge=1, le=100)
    max_artifact_store_bytes: int = Field(default=268_435_456, ge=16_777_216)
    tool_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    max_concurrency: int = Field(default=2, ge=1, le=32)
    allowed_origins: tuple[str, ...] = ("http://127.0.0.1", "http://localhost")
    allowed_hosts: tuple[str, ...] = ("127.0.0.1:*", "localhost:*")

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """拒绝通配 Origin，避免浏览器侧任意站点调用。"""

        if not value or any(origin == "*" for origin in value):
            raise ValueError("allowed_origins must be an explicit non-empty allowlist")
        return value

    @field_validator("allowed_hosts")
    @classmethod
    def validate_hosts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require an explicit Host allowlist for DNS rebinding protection."""

        if not value or any(host == "*" for host in value):
            raise ValueError("allowed_hosts must be an explicit non-empty allowlist")
        return value

def _csv_tuple(value: str) -> tuple[str, ...]:
    """Parse a non-empty comma-separated runtime allowlist."""

    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("allowlist override must not be empty")
    return items


_ENV_OVERRIDES: dict[str, tuple[str, Any]] = {
    "CRC_LNM_MCP_HOST": ("host", str),
    "CRC_LNM_MCP_PORT": ("port", int),
    "CRC_LNM_MCP_LOG_LEVEL": ("log_level", str),
    "CRC_LNM_MCP_ALLOWED_HOSTS": ("allowed_hosts", _csv_tuple),
    "CRC_LNM_MCP_ALLOWED_ORIGINS": ("allowed_origins", _csv_tuple),
}


def _load_yaml_object(path: Path) -> dict[str, Any]:
    """读取 YAML object；错误中只给固定文本，避免对外暴露路径。"""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("MCP configuration is unavailable or invalid.") from exc
    if not isinstance(payload, dict):
        raise ValueError("MCP configuration root must be a mapping.")
    return payload


def load_mcp_settings(
    path: Path,
    *,
    environ: dict[str, str] | None = None,
) -> MCPSettings:
    """加载配置、解析相对路径并应用受控的操作员环境覆盖。

    相对路径始终以配置文件所在目录为基准，绝不受进程当前工作目录影响。只有固定白名单
    中的环境变量可覆盖配置，聊天请求无法改变服务运行模式。
    """

    config_path = path.resolve()
    payload = _load_yaml_object(config_path)
    environment = os.environ if environ is None else environ
    for variable, (field, converter) in _ENV_OVERRIDES.items():
        if variable in environment:
            payload[field] = converter(environment[variable])
    for field in ("bundle_directory", "case_root", "artifact_root", "case_package_jsonl"):
        raw_value = payload.get(field)
        if raw_value is None:
            continue
        value = Path(str(raw_value))
        payload[field] = (
            value.resolve()
            if value.is_absolute()
            else (config_path.parent / value).resolve()
        )
    return MCPSettings.model_validate(payload)
