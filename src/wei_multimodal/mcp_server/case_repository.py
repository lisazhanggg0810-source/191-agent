"""受控的预提取特征病例仓库。

仓库只接受 ``case_ref``，不接受调用者路径，不递归扫描目录。V1 manifest 必须明确列出
CT、病理和临床三个 JSON 文件及其真实 SHA-256；原始 DICOM/WSI 不在本模块范围内。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from wei_multimodal.mcp_server.errors import ContractError, ErrorCode
from wei_multimodal.mcp_server.security import (
    SHA256_PATTERN,
    read_json_object,
    reject_forbidden_fields,
    resolve_case_directory,
    resolve_case_file,
    sha256_file,
)

RESEARCH_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
REQUIRED_MODALITIES: Final[tuple[str, ...]] = (
    "ct_features",
    "pathology_features",
    "clinical",
)
EXPECTED_FILENAMES: Final[dict[str, str]] = {
    "ct_features": "ct_features.json",
    "pathology_features": "pathology_features.json",
    "clinical": "clinical.json",
}
MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_version", "case_ref", "research_id", "input_mode", "files", "sha256"}
)


@dataclass(frozen=True, slots=True)
class CaseManifest:
    """已校验的脱敏病例 manifest。"""

    schema_version: str
    case_ref: str
    research_id: str
    input_mode: str
    files: dict[str, str]
    sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class CasePackage:
    """服务端受控病例对象；``paths`` 绝不直接写入 MCP 响应。"""

    manifest: CaseManifest
    paths: dict[str, Path]
    case_binding_sha256: str

    @property
    def case_ref(self) -> str:
        return self.manifest.case_ref

    @property
    def research_id(self) -> str:
        return self.manifest.research_id

    def safe_summary(self) -> dict[str, Any]:
        """返回不含路径和特征值的质控摘要。"""

        return {
            "case_ref": self.case_ref,
            "input_mode": self.manifest.input_mode,
            "modalities": list(REQUIRED_MODALITIES),
            "file_count": len(self.paths),
            "case_binding_sha256": self.case_binding_sha256,
        }


class CaseRepository:
    """从单一 allowlist 根目录读取预提取 JSON 病例包。"""

    def __init__(self, root: Path, *, max_file_bytes: int = 16 * 1024 * 1024) -> None:
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        self._root = root.resolve()
        self._max_file_bytes = max_file_bytes

    def _parse_manifest(self, payload: dict[str, Any], case_ref: str) -> CaseManifest:
        """严格验证 manifest 字段、模式、逻辑文件名和真实哈希格式。"""

        if set(payload) != MANIFEST_FIELDS:
            raise ContractError(
                ErrorCode.INVALID_FILE_FORMAT,
                message="Case manifest fields do not match schema 1.0.0.",
            )
        if payload.get("schema_version") != "1.0.0":
            raise ContractError(
                ErrorCode.INVALID_FILE_FORMAT,
                message="Unsupported case manifest schema version.",
            )
        if payload.get("case_ref") != case_ref:
            raise ContractError(
                ErrorCode.RESEARCH_ID_MISMATCH,
                message="Manifest case reference does not match the requested case.",
            )
        research_id = payload.get("research_id")
        if not isinstance(research_id, str) or not RESEARCH_ID_PATTERN.fullmatch(research_id):
            raise ContractError(
                ErrorCode.INVALID_FILE_FORMAT,
                message="Manifest research identifier is invalid.",
            )
        if payload.get("input_mode") != "precomputed":
            raise ContractError(
                ErrorCode.UNSUPPORTED_SOURCE_MODE,
                message="Only precomputed feature case packages are enabled.",
            )
        files = payload.get("files")
        digests = payload.get("sha256")
        if not isinstance(files, dict) or set(files) != set(REQUIRED_MODALITIES):
            raise ContractError(
                ErrorCode.MISSING_MODALITY,
                message="CT, pathology and clinical inputs are all required.",
            )
        if any(files.get(key) != EXPECTED_FILENAMES[key] for key in REQUIRED_MODALITIES):
            raise ContractError(
                ErrorCode.INVALID_FILE_FORMAT,
                message="Manifest contains an unsupported case file reference.",
            )
        if not isinstance(digests, dict) or set(digests) != set(files.values()):
            raise ContractError(
                ErrorCode.INVALID_FILE_FORMAT,
                message="Manifest SHA-256 entries do not match case files.",
            )
        for digest in digests.values():
            if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
                raise ContractError(
                    ErrorCode.INVALID_FILE_FORMAT,
                    message="Manifest contains an invalid SHA-256 digest.",
                )
            # 设计文档中的全 0/1/2 哈希仅是占位符，不能进入实际病例包。
            if len(set(digest)) == 1:
                raise ContractError(
                    ErrorCode.INVALID_FILE_FORMAT,
                    message="Manifest contains a placeholder SHA-256 digest.",
                )
        return CaseManifest(
            schema_version="1.0.0",
            case_ref=case_ref,
            research_id=research_id,
            input_mode="precomputed",
            files={str(key): str(value) for key, value in files.items()},
            sha256={str(key): str(value) for key, value in digests.items()},
        )

    def load(self, case_ref: str) -> CasePackage:
        """加载并完整验证病例包，任何失败都不返回半成品对象。"""

        case_directory = resolve_case_directory(self._root, case_ref)
        manifest_path = resolve_case_file(case_directory, "manifest.json")
        manifest_payload = read_json_object(
            manifest_path,
            max_bytes=self._max_file_bytes,
        )
        reject_forbidden_fields(manifest_payload)
        manifest = self._parse_manifest(manifest_payload, case_ref)

        paths: dict[str, Path] = {}
        for modality in REQUIRED_MODALITIES:
            filename = manifest.files[modality]
            path = resolve_case_file(case_directory, filename)
            actual_digest = sha256_file(path, max_bytes=self._max_file_bytes)
            if not hmac.compare_digest(actual_digest, manifest.sha256[filename]):
                raise ContractError(
                    ErrorCode.INVALID_FILE_FORMAT,
                    message="Case file integrity verification failed.",
                )
            # 这里做隐私/标签入口检查；维度和模型 schema 由后续准备工具检查。
            payload = read_json_object(path, max_bytes=self._max_file_bytes)
            reject_forbidden_fields(payload)
            paths[modality] = path

        # 病例绑定只依赖脱敏逻辑内容，不包含服务器绝对路径。
        binding_payload = {
            "schema_version": manifest.schema_version,
            "case_ref": manifest.case_ref,
            "research_id": manifest.research_id,
            "input_mode": manifest.input_mode,
            "sha256": manifest.sha256,
        }
        binding_bytes = json.dumps(
            binding_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return CasePackage(
            manifest=manifest,
            paths=paths,
            case_binding_sha256=hashlib.sha256(binding_bytes).hexdigest(),
        )

    def load_case(self, case_ref: str) -> CasePackage:
        """``load`` 的显式别名，便于服务层阅读。"""

        return self.load(case_ref)

    def _coerce_case(self, case: CasePackage | str) -> CasePackage:
        """允许服务层复用已加载对象，避免同一次请求重复哈希全部文件。"""

        return self.load(case) if isinstance(case, str) else case

    def _read_modality(self, case: CasePackage | str, modality: str) -> dict[str, Any]:
        """读取已受控模态；调用者无法指定文件名或路径。"""

        package = self._coerce_case(case)
        path = package.paths[modality]
        # 再次核对哈希，防止 QC 后、特征准备前文件被替换（TOCTOU）。
        expected = package.manifest.sha256[package.manifest.files[modality]]
        actual = sha256_file(path, max_bytes=self._max_file_bytes)
        if not hmac.compare_digest(actual, expected):
            raise ContractError(
                ErrorCode.INVALID_FILE_FORMAT,
                message="Case file integrity verification failed.",
            )
        payload = read_json_object(path, max_bytes=self._max_file_bytes)
        reject_forbidden_fields(payload)
        return payload

    def read_ct_features(self, case: CasePackage | str) -> dict[str, Any]:
        """返回服务器内部 CT 特征对象，不进入 MCP 响应。"""

        return self._read_modality(case, "ct_features")

    def read_pathology_features(self, case: CasePackage | str) -> dict[str, Any]:
        """返回服务器内部病理特征对象，不进入 MCP 响应。"""

        return self._read_modality(case, "pathology_features")

    def read_clinical(self, case: CasePackage | str) -> dict[str, Any]:
        """返回病例包中的四项临床输入，具体类别由预测合同继续校验。"""

        return self._read_modality(case, "clinical")
