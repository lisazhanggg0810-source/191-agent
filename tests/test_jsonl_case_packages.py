from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from wei_multimodal.mcp_server.case_repository import CaseRepository

RELEASE_ROOT = Path(__file__).resolve().parents[1]
JSONL_SOURCE = RELEASE_ROOT / "data/release_case_package_groups.jsonl"
BUILDER_SOURCE = RELEASE_ROOT / "scripts/build_case_packages.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("case_package_builder", BUILDER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_jsonl_builds_allowlisted_case_packages(tmp_path: Path) -> None:
    builder = _load_builder()
    case_root = tmp_path / "cases"

    assert builder.build_case_packages(JSONL_SOURCE, case_root) == 142
    assert len([path for path in case_root.iterdir() if path.is_dir()]) == 142

    repository = CaseRepository(case_root)
    schema = json.loads(
        (RELEASE_ROOT / "models/deployment_bundle/schema.json").read_text("utf-8")
    )
    expected_ct = set(
        schema["ct_shape"]
        + schema["ct_original"]
        + schema["ct_wavelet"]
        + schema["ct_transformed"]
    )
    expected_pathology = set(schema["pathology_features"])

    for case_ref in sorted(path.name for path in case_root.iterdir() if path.is_dir()):
        case = repository.load(case_ref)
        assert case.manifest.input_mode == "precomputed"
        assert set(repository.read_ct_features(case)) == expected_ct
        assert set(repository.read_pathology_features(case)) == expected_pathology
        assert set(repository.read_clinical(case)) == {"age", "male", "Type", "T"}
