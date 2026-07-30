"""稳定的 MCP 业务错误。

对外只暴露冻结的错误码、名称和安全消息；Python 异常类型、堆栈以及服务器路径只能
进入服务端日志，不能拼入工具响应。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final


class ErrorCode(IntEnum):
    """设计规格中冻结的状态码。"""

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
    MODEL_INCOMPATIBLE = 4228
    BUNDLE_INTEGRITY_FAILURE = 5001
    INFERENCE_FAILURE = 5002
    REPORT_GENERATION_FAILURE = 5003
    SERVICE_UNAVAILABLE = 5030
    CAPACITY_EXCEEDED = 5031


_RETRYABLE_CODES: Final[frozenset[ErrorCode]] = frozenset(
    {
        ErrorCode.TOOL_TIMEOUT,
        ErrorCode.REPORT_GENERATION_FAILURE,
        ErrorCode.SERVICE_UNAVAILABLE,
        ErrorCode.CAPACITY_EXCEEDED,
    }
)


@dataclass(slots=True, frozen=True)
class ContractError(Exception):
    """可安全映射为 MCP 结构化错误的业务异常。

    ``message`` 必须是面向调用者的固定描述，禁止包含绝对路径、原始患者数据或底层
    异常文本。内部异常应由调用层使用 ``logger.exception`` 单独记录。
    """

    code: int | ErrorCode
    name: str | None = None
    message: str = "The request could not be completed."
    field: str | None = None
    retryable: bool | None = None

    def __post_init__(self) -> None:
        """规范化名称和重试属性，同时保持异常对象不可变。"""

        try:
            known_code = ErrorCode(int(self.code))
        except ValueError as exc:
            raise ValueError("ContractError must use a registered error code.") from exc
        if self.name is not None and self.name != known_code.name:
            raise ValueError("Error name does not match the registered error code.")
        object.__setattr__(self, "code", known_code)
        object.__setattr__(self, "name", known_code.name)
        if self.retryable is None:
            object.__setattr__(self, "retryable", known_code in _RETRYABLE_CODES)
        Exception.__init__(self, self.message)

    def to_public_dict(self) -> dict[str, int | str | bool | None]:
        """生成可放入响应 ``errors`` 数组的安全对象。"""

        return {
            "code": int(self.code),
            "name": str(self.name),
            "message": self.message,
            "field": self.field,
            "retryable": bool(self.retryable),
        }


def contract_error(
    code: ErrorCode,
    message: str,
    *,
    field: str | None = None,
) -> ContractError:
    """使用注册状态码创建异常，避免调用方重复填写且误拼 ``name``。"""

    return ContractError(code=code, message=message, field=field)
