"""CRC-LNM 核心 MCP 的版本化 JSON 合同与 Schema 导出入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import (
    CONTRACT_VERSION,
    ArtifactRef,
    ErrorPayload,
    ProvenancePayload,
    StatusCode,
    StatusName,
    StatusPayload,
    StrictContract,
    ToolResponse,
    WarningPayload,
)
from .inputs import (
    CaseQCRequest,
    GenerateReportRequest,
    GetModelInfoRequest,
    PredictMultimodalRequest,
    PrepareCTRequest,
    PreparePathologyRequest,
)
from .outputs import (
    CaseQCData,
    CaseQCResponse,
    CTFeatureData,
    GenerateReportResponse,
    GetModelInfoResponse,
    ModelInfoData,
    PathologyFeatureData,
    PredictionData,
    PredictMultimodalResponse,
    PrepareCTResponse,
    PreparePathologyResponse,
    ReportData,
)

# 文件名与设计规格保持稳定；此映射中刻意没有 DICOM 提取工具。
CONTRACT_MODELS: dict[str, type[StrictContract]] = {
    "model-info-request": GetModelInfoRequest,
    "model-info-response": GetModelInfoResponse,
    "case-qc-request": CaseQCRequest,
    "case-qc-response": CaseQCResponse,
    "ct-features-request": PrepareCTRequest,
    "ct-features-response": PrepareCTResponse,
    "pathology-features-request": PreparePathologyRequest,
    "pathology-features-response": PreparePathologyResponse,
    "prediction-request": PredictMultimodalRequest,
    "prediction-response": PredictMultimodalResponse,
    "report-request": GenerateReportRequest,
    "report-response": GenerateReportResponse,
}


def build_contract_schemas() -> dict[str, dict[str, Any]]:
    """生成六套请求/响应的 Draft 2020-12 JSON Schema 文档。

    返回值可以被测试直接比较，也可由发布脚本写入 ``schemas/mcp/v1``。
    ``ref_template`` 固定，避免不同机器产生路径不同的 Schema。
    """

    documents: dict[str, dict[str, Any]] = {}
    for name, model in CONTRACT_MODELS.items():
        schema = model.model_json_schema(
            mode="validation",
            ref_template="#/$defs/{model}",
        )
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://crc-lnm.local/schemas/mcp/v1/{name}.schema.json"
        documents[name] = schema
    return documents


def write_contract_schemas(directory: Path) -> tuple[Path, ...]:
    """以确定性 UTF-8 JSON 写出版本化 Schema，供发布流程显式调用。"""

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in build_contract_schemas().items():
        destination = directory / f"{name}.schema.json"
        destination.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(destination)
    return tuple(written)


__all__ = [
    "CONTRACT_MODELS",
    "CONTRACT_VERSION",
    "ArtifactRef",
    "CaseQCData",
    "CaseQCRequest",
    "CaseQCResponse",
    "CTFeatureData",
    "ErrorPayload",
    "GenerateReportRequest",
    "GenerateReportResponse",
    "GetModelInfoRequest",
    "GetModelInfoResponse",
    "ModelInfoData",
    "PathologyFeatureData",
    "PredictMultimodalRequest",
    "PredictMultimodalResponse",
    "PredictionData",
    "PrepareCTRequest",
    "PrepareCTResponse",
    "PreparePathologyRequest",
    "PreparePathologyResponse",
    "ProvenancePayload",
    "ReportData",
    "StatusCode",
    "StatusName",
    "StatusPayload",
    "StrictContract",
    "ToolResponse",
    "WarningPayload",
    "build_contract_schemas",
    "write_contract_schemas",
]
