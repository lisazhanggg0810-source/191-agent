"""六个核心 MCP 工具的精确请求合同。

每个工具拥有独立的 ``input`` 模型，禁止用 ``dict[str, Any]`` 绕过字段白名单。
DICOM 提取工具不属于本模块；CT 合同中的 ``ct_package`` 仅保留设计规定的、
供未来已验证侧车 artifact 接入的受控引用接口。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import UUID4, Field, field_validator

from .common import CONTRACT_VERSION, StrictContract

CaseRef = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
QcArtifactId = Annotated[str, Field(pattern=r"^qc_[a-f0-9]{32}$")]
CtArtifactId = Annotated[str, Field(pattern=r"^ctf_[a-f0-9]{32}$")]
PathologyArtifactId = Annotated[str, Field(pattern=r"^pathf_[a-f0-9]{32}$")]
PredictionArtifactId = Annotated[str, Field(pattern=r"^pred_[a-f0-9]{32}$")]
FiniteNumber = int | float


class EmptyInput(StrictContract):
    """只读模型信息工具必须接收真正的空对象。"""


class BaseRequest(StrictContract):
    """所有工具共有的版本与追踪字段。"""

    contract_version: Literal["1.1.0"] = CONTRACT_VERSION
    request_id: UUID4
    trace_id: UUID4


class CaseRequest(BaseRequest):
    """需要访问脱敏病例包的工具请求基类。"""

    case_ref: CaseRef


class GetModelInfoRequest(BaseRequest):
    """模型信息请求；故意不声明 ``case_ref``，夹带该字段会被拒绝。"""

    input: EmptyInput


class CaseQCInput(StrictContract):
    """病例质控的来源选择策略，不允许请求改变服务端能力开关。"""

    ct_source_preference: Literal["precomputed"] = "precomputed"
    fallback_policy: Literal["precomputed_if_available"] = "precomputed_if_available"


class CaseQCRequest(CaseRequest):
    """病例数据完整性与隐私质控请求。"""

    input: CaseQCInput


class PrecomputedCTSource(StrictContract):
    """选择病例包中已获批准的预提取 CT 特征。"""

    mode: Literal["precomputed"]


class PrepareCTInput(StrictContract):
    """CT 准备工具的精确输入。"""

    qc_artifact_id: QcArtifactId
    source: PrecomputedCTSource


class PrepareCTRequest(CaseRequest):
    """1409 维 CT 特征准备请求。"""

    input: PrepareCTInput


class PreparePathologyInput(StrictContract):
    """病理准备工具只接受已通过质控的 artifact。"""

    qc_artifact_id: QcArtifactId


class PreparePathologyRequest(CaseRequest):
    """768 维患者级病理特征准备请求。"""

    input: PreparePathologyInput


class ClinicalInput(StrictContract):
    """与现有 ``PredictionService`` 一致的四项临床输入。

    ``Type`` 与 ``T`` 的训练词表依赖具体部署包，因此合同层只保证它们是
    有限数值，业务层再依据当前预处理器词表做精确类别校验。
    """

    age: Annotated[FiniteNumber, Field(ge=0, le=120)]
    male: Literal[0, 1]
    Type: FiniteNumber
    T: FiniteNumber

    @field_validator("age", "male", "Type", "T", mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value: object) -> object:
        """Python 的 bool 是 int 子类，但医学数值输入不能接受 true/false。"""

        if isinstance(value, bool):
            raise ValueError("clinical numeric fields must not be boolean")
        return value


class PredictMultimodalInput(StrictContract):
    """多模态预测只接受同一工作流产生的三个 artifact 和临床字段。"""

    qc_artifact_id: QcArtifactId
    ct_artifact_id: CtArtifactId
    pathology_artifact_id: PathologyArtifactId
    clinical: ClinicalInput


class PredictMultimodalRequest(CaseRequest):
    """五模型多模态融合预测请求。"""

    input: PredictMultimodalInput


class GenerateReportInput(StrictContract):
    """确定性报告仅引用 QC 与预测结果，不接收外部知识文本。"""

    qc_artifact_id: QcArtifactId
    prediction_artifact_id: PredictionArtifactId


class GenerateReportRequest(CaseRequest):
    """结构化科研辅助报告请求。"""

    input: GenerateReportInput
