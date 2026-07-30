from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from starlette.testclient import TestClient

from wei_multimodal.case_packages import build_case_packages
from wei_multimodal.mcp_server.app import create_http_app
from wei_multimodal.mcp_server.artifact_store import ArtifactStore
from wei_multimodal.mcp_server.case_repository import CaseRepository
from wei_multimodal.mcp_server.contracts import (
    CaseQCRequest,
    GenerateReportRequest,
    PredictMultimodalRequest,
    PrepareCTRequest,
    PreparePathologyRequest,
)
from wei_multimodal.mcp_server.services import (
    RuntimeDependencies,
    case_data_qc,
    generate_report,
    predict_multimodal,
    prepare_ct_features,
    prepare_pathology_features,
)
from wei_multimodal.mcp_server.settings import load_mcp_settings
from wei_multimodal.service.prediction import PredictionService

RELEASE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROBABILITY = 0.6279473900794983
EXPECTED_THRESHOLD = 0.3529504342004657
EXPECTED_MODEL_VERSION = "003636d691d603908f8bd913b5ab7769a5e1479680c7f8317cda1d106e8bf73f"


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> RuntimeDependencies:
    prediction_service = PredictionService(
        RELEASE_ROOT / "models/deployment_bundle",
        device="cpu",
    )
    return RuntimeDependencies.create(
        prediction_service=prediction_service,
        artifact_store=ArtifactStore(tmp_path_factory.mktemp("artifacts")),
        case_repository=CaseRepository(RELEASE_ROOT / "demo/cases"),
        tool_timeout_seconds=120,
        max_concurrency=2,
    )


def test_real_bundle_has_five_verified_members(runtime: RuntimeDependencies) -> None:
    info = runtime.prediction_service.get_info()
    assert info.integrity_verified is True
    assert info.member_count == 5
    assert info.ct_feature_count == 1409
    assert info.pathology_feature_count == 768
    assert info.threshold == EXPECTED_THRESHOLD
    assert info.model_version == EXPECTED_MODEL_VERSION
    assert info.independent_test_claim is False


def test_complete_release_workflow_matches_golden_probability(
    runtime: RuntimeDependencies,
) -> None:
    trace_id = uuid4()
    case_ref = "demo_case_001"
    clinical = json.loads(
        (RELEASE_ROOT / "demo/cases/demo_case_001/clinical.json").read_text("utf-8")
    )
    qc = case_data_qc(
        CaseQCRequest(request_id=uuid4(), trace_id=trace_id, case_ref=case_ref, input={}),
        runtime,
    )
    ct = prepare_ct_features(
        PrepareCTRequest(
            request_id=uuid4(),
            trace_id=trace_id,
            case_ref=case_ref,
            input={
                "qc_artifact_id": qc.artifact.artifact_id,
                "source": {"mode": "precomputed"},
            },
        ),
        runtime,
    )
    pathology = prepare_pathology_features(
        PreparePathologyRequest(
            request_id=uuid4(),
            trace_id=trace_id,
            case_ref=case_ref,
            input={"qc_artifact_id": qc.artifact.artifact_id},
        ),
        runtime,
    )
    prediction = predict_multimodal(
        PredictMultimodalRequest(
            request_id=uuid4(),
            trace_id=trace_id,
            case_ref=case_ref,
            input={
                "qc_artifact_id": qc.artifact.artifact_id,
                "ct_artifact_id": ct.artifact.artifact_id,
                "pathology_artifact_id": pathology.artifact.artifact_id,
                "clinical": clinical,
            },
        ),
        runtime,
    )
    report = generate_report(
        GenerateReportRequest(
            request_id=uuid4(),
            trace_id=trace_id,
            case_ref=case_ref,
            input={
                "qc_artifact_id": qc.artifact.artifact_id,
                "prediction_artifact_id": prediction.artifact.artifact_id,
            },
        ),
        runtime,
    )

    assert qc.passed is True
    assert qc.ct_source_selected == "precomputed"
    assert ct.feature_count == 1409
    assert pathology.feature_count == 768
    # CPU BLAS/PyTorch builds can differ in the final float32 softmax digits while
    # preserving the verified weights, decision threshold, and predicted class.
    assert prediction.positive_probability == pytest.approx(EXPECTED_PROBABILITY, abs=2e-7)
    assert prediction.threshold == EXPECTED_THRESHOLD
    assert prediction.predicted_class == 1
    assert prediction.member_count == 5
    assert prediction.independent_test_claim is False
    assert report.heatmap_status == "not_available_in_v1"
    assert report.feature_attribution_status == "not_available_in_v1"


def test_real_jsonl_case_runs_the_complete_workflow(
    runtime: RuntimeDependencies,
    tmp_path: Path,
) -> None:
    case_root = tmp_path / "release_cases"
    assert build_case_packages(
        RELEASE_ROOT / "data/release_case_package_groups.jsonl",
        case_root,
    ) == 142
    jsonl_runtime = RuntimeDependencies.create(
        prediction_service=runtime.prediction_service,
        artifact_store=ArtifactStore(tmp_path / "jsonl_artifacts"),
        case_repository=CaseRepository(case_root),
        tool_timeout_seconds=120,
        max_concurrency=2,
    )
    trace_id = uuid4()
    case_ref = "PA1104647"
    clinical = json.loads((case_root / case_ref / "clinical.json").read_text("utf-8"))

    qc = case_data_qc(
        CaseQCRequest(request_id=uuid4(), trace_id=trace_id, case_ref=case_ref, input={}),
        jsonl_runtime,
    )
    ct = prepare_ct_features(
        PrepareCTRequest(
            request_id=uuid4(),
            trace_id=trace_id,
            case_ref=case_ref,
            input={"qc_artifact_id": qc.artifact.artifact_id, "source": {"mode": "precomputed"}},
        ),
        jsonl_runtime,
    )
    pathology = prepare_pathology_features(
        PreparePathologyRequest(
            request_id=uuid4(),
            trace_id=trace_id,
            case_ref=case_ref,
            input={"qc_artifact_id": qc.artifact.artifact_id},
        ),
        jsonl_runtime,
    )
    prediction = predict_multimodal(
        PredictMultimodalRequest(
            request_id=uuid4(),
            trace_id=trace_id,
            case_ref=case_ref,
            input={
                "qc_artifact_id": qc.artifact.artifact_id,
                "ct_artifact_id": ct.artifact.artifact_id,
                "pathology_artifact_id": pathology.artifact.artifact_id,
                "clinical": clinical,
            },
        ),
        jsonl_runtime,
    )

    assert qc.passed is True
    assert ct.feature_count == 1409
    assert pathology.feature_count == 768
    assert 0 <= prediction.positive_probability <= 1
    assert prediction.human_review_required is True


def test_real_http_app_requires_bearer_and_becomes_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token = "release-test-token-" + "a" * 32
    monkeypatch.setenv("CRC_LNM_MCP_BEARER_TOKEN", token)
    settings = load_mcp_settings(RELEASE_ROOT / "configs/mcp.yaml")
    settings = settings.model_copy(update={"artifact_root": tmp_path / "artifacts"})
    app = create_http_app(settings)

    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        assert client.get("/health/live").json() == {"status": "live"}
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert ready.json()["model_version"] == EXPECTED_MODEL_VERSION
        assert client.get("/mcp").status_code == 401
        authorized = client.post(
            "/mcp",
            headers={
                "Authorization": f"Bearer {token}",
                "accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "release-test", "version": "1.0"},
                },
            },
        )
        assert authorized.status_code == 200
