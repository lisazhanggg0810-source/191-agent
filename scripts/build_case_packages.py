"""Build allowlisted precomputed case packages from the release JSONL file.

The MCP service intentionally accepts only a ``case_ref`` and loads immutable
case files from its configured root.  This utility is therefore a build-time
adapter: it turns each inline JSONL record into that existing file contract
without introducing an upload or caller-supplied-path API at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


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
CASE_REF_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write one deterministic UTF-8 JSON object and return its SHA-256."""

    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _require_object(value: object, name: str, line_number: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number}: {name} must be a JSON object")
    return value


def _validate_record(record: dict[str, Any], line_number: int) -> tuple[str, dict[str, Any]]:
    case_ref = record.get("case_ref")
    if not isinstance(case_ref, str) or not CASE_REF_PATTERN.fullmatch(case_ref):
        raise ValueError(f"line {line_number}: case_ref must match the MCP allowlist pattern")

    manifest = _require_object(record.get("manifest"), "manifest", line_number)
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
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in source_digests.values()
    ):
        raise ValueError(f"line {line_number}: manifest SHA-256 values are invalid")

    payloads = {
        "ct_features": _require_object(record.get("ct_features"), "ct_features", line_number),
        "pathology_features": _require_object(
            record.get("pathology_features"), "pathology_features", line_number
        ),
        "clinical": _require_object(record.get("clinical"), "clinical", line_number),
    }
    return case_ref, {"manifest": manifest, **payloads}


def build_case_packages(input_path: Path, output_root: Path) -> int:
    """Create case directories from a JSONL source that has not been staged yet."""

    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output directory must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

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
                EXPECTED_FILES[name]: _write_json(case_directory / EXPECTED_FILES[name], package[name])
                for name in EXPECTED_FILES
            }
            manifest = {
                "schema_version": "1.0.0",
                "case_ref": case_ref,
                "research_id": package["manifest"]["research_id"],
                "input_mode": "precomputed",
                "files": EXPECTED_FILES,
                "sha256": digests,
            }
            _write_json(case_directory / "manifest.json", manifest)
            count += 1
    if count == 0:
        raise ValueError("input JSONL contains no case records")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create allowlisted precomputed MCP case packages from JSONL."
    )
    parser.add_argument("--input", type=Path, required=True, help="source JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="empty case-root directory")
    args = parser.parse_args()
    count = build_case_packages(args.input, args.output)
    print(f"Built {count} case packages in {args.output}")


if __name__ == "__main__":
    main()
