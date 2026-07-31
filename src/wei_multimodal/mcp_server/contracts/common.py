"""MCP JSON 合同的公共类型与不变量。

本模块只定义传输层合同，不访问病例文件、模型或网络。将关键约束放在
Pydantic 模型层，可以保证 FastMCP、单元测试和离线 Schema 验证使用同一规则。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import IntEnum, StrEnum
from typing import Annotated, Literal

from pydantic import (
    UUID4,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

CONTRACT_VERSION: Literal["1.1.0"] = "1.1.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
ARTIFACT_ID_PATTERN = r"^(qc|qcpermit|ctf|ctpkg|pathf|pred|rpt)_[a-f0-9]{32}$"


def _require_utc(value: object) -> object:
    """拒绝非 UTC 时间；字符串形式还必须显式使用 ``Z`` 后缀。"""

    if isinstance(value, str) and not value.endswith("Z"):
        raise ValueError("timestamp must be RFC 3339 UTC and end with Z")
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be timezone-aware UTC")
    return value


# Pydantic 会将 UTC datetime 序列化为 Z；BeforeValidator 同时拦截 +00:00 文本，
# 从而严格满足设计规格的“线上 JSON 时间必须以 Z 结尾”。
UtcDateTime = Annotated[datetime, BeforeValidator(_require_utc)]
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
ArtifactId = Annotated[str, Field(pattern=ARTIFACT_ID_PATTERN)]
ShortText = Annotated[str, Field(min_length=1, max_length=256)]


class StrictContract(BaseModel):
    """所有合同模型的严格基类。

    ``extra=forbid`` 防止调用者夹带路径、阈值或兼容性结论；
    ``allow_inf_nan=False`` 让所有浮点字段继承有限数值约束；冻结模型可避免
    已校验的请求/响应在后续流程中被原地修改。
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        str_max_length=2000,
    )


class StatusCode(IntEnum):
    """设计规格冻结的机器状态码。"""

    OK = 2000
    OK_WITH_WARNINGS = 2001
    MISSING_MODALITY = 4001
    RESEARCH_ID_MISMATCH = 4002
    PII_DETECTED = 4003
    INVALID_FILE_FORMAT = 4004
    FEATURE_SCHEMA_MISMATCH = 4005
    NON_FINITE_VALUE = 4006
    INVALID_CLINICAL_CATEGORY = 4007
    UNSUPPORTED_SOURCE_MODE = 4008
    QC_REQUIRED_OR_FAILED = 4010
    TOOL_TIMEOUT = 4032
    ARTIFACT_EXPIRED = 4040
    ARTIFACT_UNAVAILABLE = 4041
    INVALID_STAGE_ORDER = 4090
    ARTIFACT_TYPE_MISMATCH = 4091
    CASE_BINDING_MISMATCH = 4092
    DICOM_SERIES_NOT_FOUND = 4220
    DICOM_SERIES_AMBIGUOUS = 4221
    DICOM_GEOMETRY_INVALID = 4222
    ROI_REQUIRED = 4223
    ROI_EMPTY = 4224
    ROI_GEOMETRY_MISMATCH = 4225
    EXTRACTION_PROFILE_UNKNOWN = 4226
    EXTRACTION_PROFILE_UNVERIFIED = 4227
    MODEL_INCOMPATIBLE = 4228
    BUNDLE_INTEGRITY_FAILURE = 5001
    INFERENCE_FAILURE = 5002
    REPORT_GENERATION_FAILURE = 5003
    RADIOMICS_EXTRACTION_FAILURE = 5004
    SERVICE_UNAVAILABLE = 5030
    CAPACITY_EXCEEDED = 5031


class StatusName(StrEnum):
    """与 :class:`StatusCode` 一一对应的稳定名称。"""

    OK = "OK"
    OK_WITH_WARNINGS = "OK_WITH_WARNINGS"
    MISSING_MODALITY = "MISSING_MODALITY"
    RESEARCH_ID_MISMATCH = "RESEARCH_ID_MISMATCH"
    PII_DETECTED = "PII_DETECTED"
    INVALID_FILE_FORMAT = "INVALID_FILE_FORMAT"
    FEATURE_SCHEMA_MISMATCH = "FEATURE_SCHEMA_MISMATCH"
    NON_FINITE_VALUE = "NON_FINITE_VALUE"
    INVALID_CLINICAL_CATEGORY = "INVALID_CLINICAL_CATEGORY"
    UNSUPPORTED_SOURCE_MODE = "UNSUPPORTED_SOURCE_MODE"
    QC_REQUIRED_OR_FAILED = "QC_REQUIRED_OR_FAILED"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    ARTIFACT_EXPIRED = "ARTIFACT_EXPIRED"
    ARTIFACT_UNAVAILABLE = "ARTIFACT_UNAVAILABLE"
    INVALID_STAGE_ORDER = "INVALID_STAGE_ORDER"
    ARTIFACT_TYPE_MISMATCH = "ARTIFACT_TYPE_MISMATCH"
    CASE_BINDING_MISMATCH = "CASE_BINDING_MISMATCH"
    DICOM_SERIES_NOT_FOUND = "DICOM_SERIES_NOT_FOUND"
    DICOM_SERIES_AMBIGUOUS = "DICOM_SERIES_AMBIGUOUS"
    DICOM_GEOMETRY_INVALID = "DICOM_GEOMETRY_INVALID"
    ROI_REQUIRED = "ROI_REQUIRED"
    ROI_EMPTY = "ROI_EMPTY"
    ROI_GEOMETRY_MISMATCH = "ROI_GEOMETRY_MISMATCH"
    EXTRACTION_PROFILE_UNKNOWN = "EXTRACTION_PROFILE_UNKNOWN"
    EXTRACTION_PROFILE_UNVERIFIED = "EXTRACTION_PROFILE_UNVERIFIED"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    BUNDLE_INTEGRITY_FAILURE = "BUNDLE_INTEGRITY_FAILURE"
    INFERENCE_FAILURE = "INFERENCE_FAILURE"
    REPORT_GENERATION_FAILURE = "REPORT_GENERATION_FAILURE"
    RADIOMICS_EXTRACTION_FAILURE = "RADIOMICS_EXTRACTION_FAILURE"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"


_STATUS_NAME_BY_CODE = {StatusCode[name]: StatusName[name] for name in StatusCode.__members__}
_RETRYABLE_CODES = {
    StatusCode.TOOL_TIMEOUT,
    StatusCode.REPORT_GENERATION_FAILURE,
    StatusCode.SERVICE_UNAVAILABLE,
    StatusCode.CAPACITY_EXCEEDED,
}


class StatusPayload(StrictContract):
    """响应的确定性状态摘要，编码、名称和严重度必须相互一致。"""

    code: StatusCode
    name: StatusName
    message: ShortText
    severity: Literal["info", "warning", "error"]
    retryable: bool

    @model_validator(mode="after")
    def validate_status_tuple(self) -> StatusPayload:
        """防止出现 ``code=2000`` 但名称或严重度声称失败的矛盾响应。"""

        if self.name != _STATUS_NAME_BY_CODE[self.code]:
            raise ValueError("status code and name do not match")
        expected_severity = (
            "info"
            if self.code == StatusCode.OK
            else "warning"
            if self.code == StatusCode.OK_WITH_WARNINGS
            else "error"
        )
        if self.severity != expected_severity:
            raise ValueError("status severity does not match status code")
        if self.retryable != (self.code in _RETRYABLE_CODES):
            raise ValueError("status retryable flag does not match status code")
        return self


class ErrorPayload(StrictContract):
    """可机器处理的错误明细；不允许自由文本列表或异常堆栈。"""

    field: Annotated[str | None, Field(max_length=256)] = None
    code: StatusName
    message: ShortText
    suggestion: Annotated[str | None, Field(max_length=256)] = None


class WarningPayload(StrictContract):
    """结构化警告；``review_required`` 明确是否提升人工复核优先级。"""

    code: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")]
    message: ShortText
    field: Annotated[str | None, Field(max_length=256)] = None
    review_required: bool


class ProvenancePayload(StrictContract):
    """不含患者数据的审计来源信息。"""

    service_version: Annotated[str, Field(min_length=1, max_length=32)]
    model_version: Sha256 | None
    model_schema_version: Annotated[str | None, Field(min_length=1, max_length=32)]
    model_feature_order_sha256: Sha256 | None
    input_sha256: Sha256
    created_at: UtcDateTime
    duration_ms: Annotated[int, Field(ge=0)]


ArtifactType = Literal[
    "case_qc",
    "dicom_extraction_permit",
    "ct_features",
    "ct_feature_package",
    "pathology_features",
    "prediction",
    "report",
]

_ARTIFACT_PREFIX_BY_TYPE: dict[str, str] = {
    "case_qc": "qc_",
    "dicom_extraction_permit": "qcpermit_",
    "ct_features": "ctf_",
    "ct_feature_package": "ctpkg_",
    "pathology_features": "pathf_",
    "prediction": "pred_",
    "report": "rpt_",
}


class ArtifactRef(StrictContract):
    """跨工具传递的短期不可变 artifact 引用，不携带原始特征向量。"""

    artifact_id: ArtifactId
    artifact_type: ArtifactType
    media_type: Annotated[str, Field(min_length=1, max_length=256)]
    content_sha256: Sha256
    created_at: UtcDateTime
    expires_at: UtcDateTime

    @model_validator(mode="after")
    def validate_artifact_identity(self) -> ArtifactRef:
        """校验 ID 前缀/类型一致，并保证有效期严格晚于创建时间。"""

        if not self.artifact_id.startswith(_ARTIFACT_PREFIX_BY_TYPE[self.artifact_type]):
            raise ValueError("artifact_id prefix does not match artifact_type")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        return self


class ToolResponse[DataT: StrictContract](StrictContract):
    """六个工具共同使用的响应外壳及成功/失败互斥规则。"""

    contract_version: Literal["1.1.0"] = CONTRACT_VERSION
    request_id: UUID4
    trace_id: UUID4
    status: StatusPayload
    data: DataT | None
    errors: Annotated[list[ErrorPayload], Field(max_length=20)] = Field(default_factory=list)
    warnings: Annotated[list[WarningPayload], Field(max_length=20)] = Field(default_factory=list)
    provenance: ProvenancePayload

    @model_validator(mode="after")
    def validate_result_exclusivity(self) -> ToolResponse[DataT]:
        """成功必须有 data 且无 errors；失败必须 data=null 且至少一个 error。"""

        success = self.status.code in {StatusCode.OK, StatusCode.OK_WITH_WARNINGS}
        if success and (self.data is None or self.errors):
            raise ValueError("successful response requires data and forbids errors")
        if not success and (self.data is not None or not self.errors):
            raise ValueError("failed response requires data=null and at least one error")
        if self.status.code == StatusCode.OK and self.warnings:
            raise ValueError("status OK cannot contain warnings; use OK_WITH_WARNINGS")
        if self.status.code == StatusCode.OK_WITH_WARNINGS and not self.warnings:
            raise ValueError("OK_WITH_WARNINGS requires at least one warning")
        return self
