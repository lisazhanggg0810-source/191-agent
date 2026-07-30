"""Build allowlisted precomputed case packages from the release JSONL file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_case_packages(input_path: Path, output_root: Path) -> int:
    """Load the in-tree builder for direct source-tree execution."""

    source_root = str(PROJECT_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from wei_multimodal.case_packages import build_case_packages as build

    return build(input_path, output_root)


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
