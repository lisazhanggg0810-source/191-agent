"""Trusted build-time conversion from release JSONL to MCP case packages.

The MCP runtime only reads allowlisted case directories. This module converts
the repository-owned JSONL release file into that immutable directory contract;
it never accepts a caller-supplied JSONL path through an MCP tool.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

BUILDER_SCHEMA_VERSION = "1.0.0"
CACHE_METADATA_FILENAME = ".case-package-cache.json"
EXPECTED_FILES = {
    "ct_features": "ct_features.json",
    "pathology_features": "pathology_features.json",
    "clinical": "clinical.json",
}
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "case_ref",
    "research_id",
    "input_mode",
    "files",
    "sha256",
}
CLINICAL_FIELDS = frozenset({"age", "male", "Type", "T"})
EXPECTED_FEATURE_COUNTS = {"ct_features": 1409, "pathology_features": 768}
CASE_REF_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write one deterministic UTF-8 JSON object and return its SHA-256."""

    encoded = _canonical_json_bytes(payload)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_object(value: object, name: str, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: {name} must be a JSON object")
    return value


def _validate_number(value: object, field_name: str, line_number: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"line {line_number}: {field_name} values must be numbers")
    if not math.isfinite(float(value)):
        raise ValueError(f"line {line_number}: {field_name} values must be finite")


def _validate_feature_mapping(
    payload: dict[str, Any],
    field_name: str,
    line_number: int,
) -> None:
    expected_count = EXPECTED_FEATURE_COUNTS[field_name]
    if len(payload) != expected_count:
        raise ValueError(
            f"line {line_number}: {field_name} must contain exactly {expected_count} features"
        )
    if any(not isinstance(name, str) or not name for name in payload):
        raise ValueError(
            f"line {line_number}: {field_name} feature names must be non-empty strings"
        )
    for value in payload.values():
        _validate_number(value, field_name, line_number)


def _validate_clinical(payload: dict[str, Any], line_number: int) -> None:
    if set(payload) != CLINICAL_FIELDS:
        raise ValueError(f"line {line_number}: clinical fields must be age, male, Type and T")
    for field_name, value in payload.items():
        _validate_number(value, f"clinical.{field_name}", line_number)
    if not 0 <= float(payload["age"]) <= 120:
        raise ValueError(f"line {line_number}: clinical.age must be between 0 and 120")
    if payload["male"] not in (0, 1):
        raise ValueError(f"line {line_number}: clinical.male must be 0 or 1")


def _validate_manifest(manifest: dict[str, Any], case_ref: str, line_number: int) -> str:
    if set(manifest) != REQUIRED_MANIFEST_FIELDS:
        raise ValueError(f"line {line_number}: manifest fields do not match schema 1.0.0")
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError(f"line {line_number}: unsupported manifest schema version")
    if manifest.get("case_ref") != case_ref:
        raise ValueError(f"line {line_number}: manifest case_ref does not match record")
    if manifest.get("input_mode") != "precomputed":
        raise ValueError(f"line {line_number}: only precomputed case records are supported")
    if manifest.get("files") != EXPECTED_FILES:
        raise ValueError(f"line {line_number}: manifest file names are not supported")
    research_id = manifest.get("research_id")
    if not isinstance(research_id, str) or not CASE_REF_PATTERN.fullmatch(research_id):
        raise ValueError(f"line {line_number}: manifest research_id is invalid")
    source_digests = manifest.get("sha256")
    if not isinstance(source_digests, dict) or set(source_digests) != set(EXPECTED_FILES.values()):
        raise ValueError(f"line {line_number}: manifest SHA-256 fields are invalid")
    if any(
        not isinstance(digest, str)
        or not SHA256_PATTERN.fullmatch(digest)
        or len(set(digest)) == 1
        for digest in source_digests.values()
    ):
        raise ValueError(f"line {line_number}: manifest SHA-256 values are invalid")
    return research_id


def _validate_record(record: dict[str, Any], line_number: int) -> tuple[str, dict[str, Any]]:
    case_ref = record.get("case_ref")
    if not isinstance(case_ref, str) or not CASE_REF_PATTERN.fullmatch(case_ref):
        raise ValueError(f"line {line_number}: case_ref must match the MCP allowlist pattern")

    manifest = _require_object(record.get("manifest"), "manifest", line_number)
    research_id = _validate_manifest(manifest, case_ref, line_number)
    ct_features = _require_object(record.get("ct_features"), "ct_features", line_number)
    pathology_features = _require_object(
        record.get("pathology_features"), "pathology_features", line_number
    )
    clinical = _require_object(record.get("clinical"), "clinical", line_number)
    _validate_feature_mapping(ct_features, "ct_features", line_number)
    _validate_feature_mapping(pathology_features, "pathology_features", line_number)
    _validate_clinical(clinical, line_number)
    return case_ref, {
        "research_id": research_id,
        "ct_features": ct_features,
        "pathology_features": pathology_features,
        "clinical": clinical,
    }


def _build_into_directory(input_path: Path, output_root: Path) -> int:
    case_refs: set[str] = set()
    count = 0
    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            record = _require_object(parsed, "record", line_number)
            case_ref, package = _validate_record(record, line_number)
            if case_ref in case_refs:
                raise ValueError(f"line {line_number}: duplicate case_ref {case_ref}")
            case_refs.add(case_ref)

            case_directory = output_root / case_ref
            case_directory.mkdir()
            digests = {
                EXPECTED_FILES[name]: _write_json(
                    case_directory / EXPECTED_FILES[name], package[name]
                )
                for name in EXPECTED_FILES
            }
            _write_json(
                case_directory / "manifest.json",
                {
                    "schema_version": "1.0.0",
                    "case_ref": case_ref,
                    "research_id": package["research_id"],
                    "input_mode": "precomputed",
                    "files": EXPECTED_FILES,
                    "sha256": digests,
                },
            )
            count += 1
    if count == 0:
        raise ValueError("input JSONL contains no case records")
    return count


def build_case_packages(input_path: Path, output_root: Path) -> int:
    """Atomically create allowlisted case directories from a trusted JSONL file."""

    source = input_path.resolve()
    destination = output_root.resolve()
    if not source.is_file():
        raise ValueError("input JSONL file is unavailable")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("output directory must be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = _sha256_file(source)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-",
        dir=destination.parent,
    ) as temp:
        stage = Path(temp) / destination.name
        stage.mkdir()
        count = _build_into_directory(source, stage)
        if _sha256_file(source) != source_digest:
            raise ValueError("input JSONL changed while case packages were being built")
        _write_json(
            stage / CACHE_METADATA_FILENAME,
            {
                "builder_schema_version": BUILDER_SCHEMA_VERSION,
                "case_count": count,
                "source_sha256": source_digest,
            },
        )
        if destination.exists():
            destination.rmdir()
        os.replace(stage, destination)
    return count


def ensure_case_packages(input_path: Path, output_root: Path) -> int:
    """Build a local cache once, rejecting stale or manually altered caches."""

    source = input_path.resolve()
    destination = output_root.resolve()
    if not source.is_file():
        raise ValueError("configured case-package JSONL file is unavailable")
    marker = destination / CACHE_METADATA_FILENAME
    if destination.exists() and any(destination.iterdir()):
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("existing case-package cache has no valid metadata") from exc
        cached_count = metadata.get("case_count")
        if (
            metadata.get("builder_schema_version") == BUILDER_SCHEMA_VERSION
            and metadata.get("source_sha256") == _sha256_file(source)
            and isinstance(cached_count, int)
            and cached_count > 0
        ):
            return cached_count
        raise ValueError("existing case-package cache does not match the configured JSONL")
    return build_case_packages(source, destination)


__all__ = ["build_case_packages", "ensure_case_packages"]
