"""病例文件系统边界与隐私检查。

所有外部调用只允许提交短 ``case_ref``。路径解析、文件名解析和 JSON 读取全部在
服务器的 allowlist 根目录内完成，调用者不能提交绝对路径、URL 或目录遍历片段。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Final

from wei_multimodal.mcp_server.errors import ContractError, ErrorCode

CASE_REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SAFE_JSON_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,126}\.json$"
)
SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-f0-9]{64}$")

# 只做“字段名”硬拦截，避免把普通自由文本中的相似词误判为患者身份。
PII_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "name",
        "patient_name",
        "patientname",
        "姓名",
        "hospital_id",
        "hospital_number",
        "inpatient_id",
        "admission_id",
        "住院号",
        "id_card",
        "identity_card",
        "national_id",
        "身份证号",
        "身份证",
        "phone",
        "phone_number",
        "mobile",
        "mobile_number",
        "telephone",
        "手机号",
        "patient_id",
        "medical_record_number",
        "mrn",
    }
)

# 标签/结局不能进入模型输入病例包，否则可能造成评测泄漏。
LABEL_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "label",
        "target",
        "outcome",
        "ground_truth",
        "groundtruth",
        "true_label",
        "lnm_label",
        "lnm",
        "y",
        "lymph_node_metastasis",
        "淋巴结转移标签",
        "淋巴结转移",
        "结局",
    }
)


def _invalid_file(message: str = "The case package format is invalid.") -> ContractError:
    """返回不包含服务器路径的统一文件格式错误。"""

    return ContractError(ErrorCode.INVALID_FILE_FORMAT, message=message)


def resolve_case_directory(root: Path, case_ref: str, *, must_exist: bool = True) -> Path:
    """在病例 allowlist 根目录内解析一个病例目录。

    ``case_ref`` 只允许字母、数字、下划线和短横线。解析后还会再次检查真实路径，
    因而即使目录本身是指向根目录外的符号链接也会被拒绝。
    """

    if not CASE_REF_PATTERN.fullmatch(case_ref):
        raise _invalid_file("Invalid case_ref.")
    resolved_root = root.resolve()
    raw_candidate = resolved_root / case_ref
    # 必须在 ``resolve`` 前检查最后一级符号链接，否则 resolve 后已无法判断原入口类型。
    if raw_candidate.is_symlink():
        raise _invalid_file("Unsafe case_ref.")
    candidate = raw_candidate.resolve()
    if candidate.parent != resolved_root:
        raise _invalid_file("Unsafe case_ref.")
    if must_exist and not candidate.is_dir():
        raise _invalid_file("Case package is unavailable or invalid.")
    return candidate


def resolve_case_file(
    case_directory: Path,
    filename: str,
    *,
    allowed_suffixes: frozenset[str] = frozenset({".json"}),
    must_exist: bool = True,
) -> Path:
    """解析 manifest 中的逻辑文件名，拒绝目录、URL和符号链接逃逸。"""

    if not SAFE_JSON_FILENAME_PATTERN.fullmatch(filename):
        raise _invalid_file("Unsafe case file reference.")
    if Path(filename).suffix.lower() not in allowed_suffixes:
        raise _invalid_file("Case file type is not allowed.")
    resolved_directory = case_directory.resolve()
    raw_candidate = resolved_directory / filename
    # 同样先检查 manifest 指向的文件本身是否为符号链接，再解析真实路径。
    if raw_candidate.is_symlink():
        raise _invalid_file("Unsafe case file reference.")
    candidate = raw_candidate.resolve()
    if candidate.parent != resolved_directory:
        raise _invalid_file("Unsafe case file reference.")
    if must_exist and not candidate.is_file():
        raise _invalid_file("Required case file is unavailable or invalid.")
    return candidate


def validate_file_size(path: Path, *, max_bytes: int) -> int:
    """校验普通文件大小并返回字节数，不把真实路径写入异常。"""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _invalid_file("Case file cannot be inspected.") from exc
    if size > max_bytes:
        raise _invalid_file("Case file exceeds the configured size limit.")
    return size


def sha256_file(path: Path, *, max_bytes: int) -> str:
    """以流式方式计算文件 SHA-256，并在读取前执行容量限制。"""

    validate_file_size(path, max_bytes=max_bytes)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _invalid_file("Case file cannot be read.") from exc
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    """Python JSON 默认接受 NaN/Infinity；病例输入必须遵守 RFC 8259。"""

    raise _invalid_file(f"Non-standard JSON numeric constant is not allowed: {value}.")


def read_json_object(path: Path, *, max_bytes: int) -> dict[str, Any]:
    """读取严格 UTF-8 JSON object，拒绝 BOM、NaN/Inf、重复键和非对象根。"""

    validate_file_size(path, max_bytes=max_bytes)

    def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _invalid_file("Duplicate JSON keys are not allowed.")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        # ``utf-8`` 而非 ``utf-8-sig``：病例 JSON 明确禁止 BOM。
        text = raw.decode("utf-8")
        if text.startswith("\ufeff"):
            raise _invalid_file("UTF-8 BOM is not allowed.")
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=no_duplicate_pairs,
        )
    except ContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid_file("Case file is not valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise _invalid_file("Case JSON root must be an object.")
    return payload


def find_forbidden_fields(payload: Any) -> tuple[set[str], set[str]]:
    """递归返回直接身份字段和标签泄漏字段，不返回对应患者值。"""

    pii: set[str] = set()
    labels: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                normalized = str(raw_key).strip().casefold()
                if normalized in PII_FIELD_NAMES:
                    pii.add(normalized)
                if normalized in LABEL_FIELD_NAMES:
                    labels.add(normalized)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return pii, labels


def reject_forbidden_fields(payload: Any) -> None:
    """发现 PII 或标签泄漏时立即终止，不尝试自动删除或修复。"""

    pii, labels = find_forbidden_fields(payload)
    if pii:
        raise ContractError(
            ErrorCode.PII_DETECTED,
            message="Direct patient identifiers are not allowed in case packages.",
        )
    if labels:
        raise _invalid_file("Outcome or label fields are not allowed in model inputs.")
