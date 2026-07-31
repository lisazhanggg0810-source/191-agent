# Verification Report

Verification date: 2026-07-31 (Asia/Shanghai)

## Completed Evidence

| Check | Result |
| --- | --- |
| `pytest -q --basetemp .test-tmp-103b` | 19 passed |
| `ruff check src tests --no-cache` | Passed |
| `mypy --no-incremental` | Passed, 47 source files |
| `uv lock --locked` | Passed with uv 0.7.19 |
| Contract schemas | 12 generated files match the active Pydantic contracts |
| Wheel and sdist | Built successfully; both include the model bundle and trusted JSONL |
| Isolated wheel runtime | Built the verified case cache and loaded the model from wheel assets |
| Default STDIO protocol | No application arguments; `initialize` and `list_tools` returned exactly six MCP tools |
| PyPI 1.0.3 | Wheel and sdist uploaded; online sizes and SHA-256 digests match local artifacts |
| Refreshed PyPI runtime | `uvx` installed 1.0.3 with the CPU backend and returned exactly six MCP tools |

The wheel artifacts produced during verification were approximately 17.5 MB each before
cleanup. The standalone smoke test used an unrelated working directory and a wheel-only
`PYTHONPATH`; it did not read repository `configs/`, `models/`, or `data/` paths.

## Not Executed Here

Docker is not installed in this environment, so no local image build or container runtime
claim is made. The locked Dockerfile was reviewed and uses `uv sync --locked` plus the
checked-in `uv.lock`.

The package has been published to PyPI, but ModelScope hosted deployment and manual tool
tests have not been executed from this environment. Nexent connection, agent debugging,
publication, and post-publication question validation remain external acceptance steps.
