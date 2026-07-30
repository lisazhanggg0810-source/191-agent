"""六个核心 MCP 工具的精确响应 data 与外壳。"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .common import ArtifactRef, Sha256, StrictContract, ToolResponse


class CTGroupCounts(StrictContract):
    """当前部署模型固定的四组 CT 特征数量。"""

    shape: Literal[14]
    original: Literal[93]
    wavelet: Literal[744]
    transformed: Literal[558]


class ModelInfoData(StrictContract):
    """只读模型事实；阈值和版本只能由已验证部署包提供。"""

    model_version: Sha256
    architecture: Annotated[str, Field(min_length=1, max_length=256)]
    model_schema_version: Annotated[str, Field(min_length=1, max_length=32)]
    model_feature_order_sha256: Sha256
    pathology_feature_count: Literal[768]
    ct_feature_count: Literal[1409]
    ct_group_counts: CTGroupCounts
    clinical_features: tuple[
        Literal["male"],
        Literal["age"],
        Literal["Type"],
        Literal["T"],
    ]
    member_count: Literal[5]
    threshold: Annotated[float, Field(ge=0, le=1)]
    integrity_verified: Literal[True]
    independent_test_claim: Literal[False]
    intended_use: Literal["research_assistance_only"]

    @model_validator(mode="after")
    def validate_clinical_order(self) -> ModelInfoData:
        """防止元数据把四项临床字段漏报、重复或改变既定集合。"""

        if self.clinical_features != ("male", "age", "Type", "T"):
            raise ValueError("clinical_features must be exactly male, age, Type, T")
        return self


class ModalityAvailability(StrictContract):
    """三种正式模型输入的存在状态。"""

    ct: Literal["present", "absent"]
    pathology: Literal["present", "absent"]
    clinical: Literal["present", "absent"]


class CTSourceAvailability(StrictContract):
    """质控阶段观察到的 CT 来源状态，不代表模型兼容性。"""

    precomputed: Literal["present", "absent"]


class CaseQCData(StrictContract):
    """病例 QC 摘要及下游必须持有的短期 artifact。"""

    artifact: ArtifactRef
    case_ref: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
    input_mode: Literal["precomputed"]
    ct_source_preference: Literal["precomputed"]
    ct_source_selected: Literal["precomputed"]
    fallback_policy: Literal["precomputed_if_available", "fail_closed"]
    passed: Literal[True]
    modalities: ModalityAvailability
    ct_sources: CTSourceAvailability
    privacy_check: Literal["passed"]
    files_checked: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_qc_artifact(self) -> CaseQCData:
        """成功 QC 必须返回 case_qc artifact，且正式三模态均已就绪。"""

        if self.artifact.artifact_type != "case_qc":
            raise ValueError("QC data requires a case_qc artifact")
        if set(self.modalities.model_dump().values()) != {"present"}:
            raise ValueError("passed QC requires all model modalities to be present")
        return self


class CompatibilityData(StrictContract):
    """核心 V1 预提取 CT 或已验签侧车 CT 的服务端兼容性结论。"""

    status: Literal["validated", "unvalidated"]
    model_compatible: bool
    decision: Literal["allow_prediction", "block_prediction"]
    basis: Literal[
        "approved_precomputed_case_package",
    ]
    blocking_reasons: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_no_blockers(self) -> CompatibilityData:
        """允许预测的 artifact 不能同时携带阻断原因。"""

        if self.decision == "allow_prediction" and self.blocking_reasons:
            raise ValueError("validated compatible CT cannot contain blocking reasons")
        if self.decision == "allow_prediction" and not self.model_compatible:
            raise ValueError("allow_prediction requires model_compatible=True")
        return self


class CTFeatureData(StrictContract):
    """已验证 1409 维 CT 特征的摘要，不暴露原始向量。"""

    artifact: ArtifactRef
    input_mode: Literal["precomputed"]
    extraction_performed: Literal[False]
    source_type: Literal["precomputed_features"]
    feature_count: Literal[1409]
    group_counts: CTGroupCounts
    ct_feature_order_sha256: Sha256
    compatibility: CompatibilityData

    @model_validator(mode="after")
    def validate_ct_artifact(self) -> CTFeatureData:
        """CT data 只能引用核心服务生成的 ctf artifact。"""

        if self.artifact.artifact_type != "ct_features":
            raise ValueError("CT feature data requires a ct_features artifact")
        if self.compatibility.basis != "approved_precomputed_case_package":
            raise ValueError("CT features must come from an approved precomputed case package")
        return self


class PathologyFeatureData(StrictContract):
    """已验证 768 维患者级病理特征摘要。"""

    artifact: ArtifactRef
    input_mode: Literal["precomputed"]
    extraction_performed: Literal[False]
    source_type: Literal["precomputed_features"]
    feature_count: Literal[768]
    pathology_feature_order_sha256: Sha256
    heatmap_status: Literal["not_available_in_v1"]

    @model_validator(mode="after")
    def validate_pathology_artifact(self) -> PathologyFeatureData:
        """病理 data 只能引用 pathf artifact。"""

        if self.artifact.artifact_type != "pathology_features":
            raise ValueError("pathology data requires a pathology_features artifact")
        return self


class PerformanceReference(StrictContract):
    """开发集性能来源，不得冒充独立外部测试。"""

    metric: Literal["oof_roc_auc"]
    value: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def validate_frozen_metric(self) -> PerformanceReference:
        """性能展示值冻结为设计书中的开发集 OOF ROC-AUC。"""

        if self.value != 0.7749:
            raise ValueError("OOF ROC-AUC reference must be exactly 0.7749")
        return self


class PredictionData(StrictContract):
    """五模型融合后的确定性预测摘要与人工复核信息。"""

    artifact: ArtifactRef
    positive_probability: Annotated[float, Field(ge=0, le=1)]
    threshold: Annotated[float, Field(ge=0, le=1)]
    predicted_class: Literal[0, 1]
    decision_margin: Annotated[float, Field(ge=-1, le=1)]
    absolute_threshold_distance: Annotated[float, Field(ge=0, le=1)]
    decision_proximity: Literal["near_threshold", "intermediate", "far_from_threshold"]
    human_review_required: Literal[True]
    review_priority: Literal["routine", "elevated"]
    review_reasons: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    member_count: Literal[5]
    ct_source_used: Literal["precomputed"]
    fallback_used: Literal[False]
    fallback_reason: None
    model_version: Sha256
    independent_test_claim: Literal[False]
    performance_reference: PerformanceReference

    @model_validator(mode="after")
    def validate_prediction_math(self) -> PredictionData:
        """复算阈值距离、分类和区间，拦截内部拼装出的矛盾结果。"""

        expected_margin = self.positive_probability - self.threshold
        if not math.isclose(self.decision_margin, expected_margin, abs_tol=1e-9):
            raise ValueError("decision_margin must equal probability minus threshold")
        if not math.isclose(
            self.absolute_threshold_distance,
            abs(expected_margin),
            abs_tol=1e-9,
        ):
            raise ValueError("absolute_threshold_distance is inconsistent")
        expected_class = 1 if self.positive_probability >= self.threshold else 0
        if self.predicted_class != expected_class:
            raise ValueError("predicted_class is inconsistent with deployment threshold")
        distance = self.absolute_threshold_distance
        expected_proximity = (
            "near_threshold"
            if distance <= 0.05
            else "intermediate"
            if distance <= 0.15
            else "far_from_threshold"
        )
        if self.decision_proximity != expected_proximity:
            raise ValueError("decision_proximity is inconsistent with threshold distance")
        if distance <= 0.05 and self.review_priority != "elevated":
            raise ValueError("near-threshold prediction requires elevated review")
        if self.artifact.artifact_type != "prediction":
            raise ValueError("prediction data requires a prediction artifact")
        return self


class ReportData(StrictContract):
    """确定性 HTML 报告摘要；V1 不伪造 Resource URI、热图或归因。"""

    artifact: ArtifactRef
    report_format: Literal["html"]
    report_resource_available: Literal[False]
    sections: tuple[
        Literal[
            "case_summary",
            "input_quality",
            "model_score",
            "limitations",
            "expert_review",
            "safety_statement",
        ],
        ...,
    ]
    heatmap_status: Literal["not_available_in_v1"]
    feature_attribution_status: Literal["not_available_in_v1"]
    ct_source_used: Literal["precomputed"]
    fallback_disclosed: Literal[False]
    safety_statement: Annotated[str, Field(min_length=1, max_length=2000)]

    @model_validator(mode="after")
    def validate_report_shape(self) -> ReportData:
        """固定章节顺序，并确保报告引用类型正确。"""

        expected = (
            "case_summary",
            "input_quality",
            "model_score",
            "limitations",
            "expert_review",
            "safety_statement",
        )
        if self.sections != expected:
            raise ValueError("report sections must use the frozen V1 order")
        if self.artifact.artifact_type != "report":
            raise ValueError("report data requires a report artifact")
        return self


class GetModelInfoResponse(ToolResponse[ModelInfoData]):
    """``crc_lnm_get_model_info`` 的独立响应合同。"""

    tool_name: Literal["crc_lnm_get_model_info"]


class CaseQCResponse(ToolResponse[CaseQCData]):
    """``crc_lnm_case_data_qc`` 的独立响应合同。"""

    tool_name: Literal["crc_lnm_case_data_qc"]


class PrepareCTResponse(ToolResponse[CTFeatureData]):
    """``crc_lnm_prepare_ct_features`` 的独立响应合同。"""

    tool_name: Literal["crc_lnm_prepare_ct_features"]


class PreparePathologyResponse(ToolResponse[PathologyFeatureData]):
    """``crc_lnm_prepare_pathology_features`` 的独立响应合同。"""

    tool_name: Literal["crc_lnm_prepare_pathology_features"]


class PredictMultimodalResponse(ToolResponse[PredictionData]):
    """``crc_lnm_predict_multimodal`` 的独立响应合同。"""

    tool_name: Literal["crc_lnm_predict_multimodal"]


class GenerateReportResponse(ToolResponse[ReportData]):
    """``crc_lnm_generate_report`` 的独立响应合同。"""

    tool_name: Literal["crc_lnm_generate_report"]
