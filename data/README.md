# Release Case Packages

`release_case_package_groups.jsonl` is converted during the Docker build into
the immutable per-case files read by the MCP service. The runtime accepts only
allowlisted `case_ref` values; it does not accept JSONL paths or raw image files
from an MCP caller.
