"""Read-only model information service."""

from __future__ import annotations

from wei_multimodal.mcp_server.contracts import ModelInfoData
from wei_multimodal.mcp_server.contracts.outputs import CTGroupCounts
from wei_multimodal.mcp_server.errors import ContractError, ErrorCode

from .common import RuntimeDependencies


def get_model_info(runtime: RuntimeDependencies) -> ModelInfoData:
    """Return facts read from the integrity-checked deployment bundle.

    No case is opened and no prediction is executed.  The clinical field order in
    the public contract is the historical source-table order, while the underlying
    predictor still owns the actual preprocessing order.
    """

    info = runtime.prediction_service.get_info()
    expected_groups = {"shape": 14, "original": 93, "wavelet": 744, "transformed": 558}
    if (
        info.pathology_feature_count != 768
        or info.ct_feature_count != 1409
        or info.ct_group_counts != expected_groups
        or info.member_count != 5
        or not info.integrity_verified
    ):
        raise ContractError(
            ErrorCode.BUNDLE_INTEGRITY_FAILURE,
            message="Loaded deployment bundle does not match the frozen core contract.",
        )
    return ModelInfoData(
        model_version=info.model_version,
        architecture=info.architecture,
        model_schema_version=info.schema_version,
        model_feature_order_sha256=info.feature_order_sha256,
        pathology_feature_count=768,
        ct_feature_count=1409,
        ct_group_counts=CTGroupCounts(shape=14, original=93, wavelet=744, transformed=558),
        clinical_features=("male", "age", "Type", "T"),
        member_count=5,
        threshold=info.threshold,
        integrity_verified=True,
        independent_test_claim=False,
        intended_use="research_assistance_only",
    )


__all__ = ["get_model_info"]
