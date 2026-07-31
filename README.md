# CRC-LNM Multimodal Research Assistant MCP

This MCP server provides a six-tool, research-assistance workflow for allowlisted,
deidentified CRC-LNM cases. It accepts only precomputed 1409-dimensional CT features,
768-dimensional pathology features, and four clinical values. It does not accept raw
imaging files, file paths, or external feature vectors.

## ModelScope Hosted STDIO Configuration

Publish the wheel to PyPI before selecting ModelScope hosted STDIO deployment. The
first and only server configuration is valid JSON without comments and uses the PyPI
package rather than repository files or local paths.

```json
{
  "mcpServers": {
    "crc-lnm-research-assistant": {
      "command": "uvx",
      "args": [
        "crc-lnm-medical-agent"
      ],
      "env": {
        "UV_TORCH_BACKEND": "cpu"
      }
    }
  }
}
```

The console entry point defaults to STDIO, allowing ModelScope to use its minimal
allowlisted `uvx crc-lnm-medical-agent` command without application arguments. The
`UV_TORCH_BACKEND=cpu` environment setting keeps the hosted Linux installation on the
CPU PyTorch index instead of resolving CUDA runtime packages. The published wheel
contains the immutable model bundle and trusted release JSONL.
On first launch it creates a verified case-package cache and transient artifacts in a
writable system cache directory. No local path argument is required. Set
`CRC_LNM_MCP_RUNTIME_ROOT` only when an operator needs a different writable cache
location.

## Verification Order

1. Build and inspect the wheel, then run the console entry point from an unrelated
   working directory.
2. Publish the verified wheel to PyPI and start it with the exact `uvx` command above.
3. Let ModelScope complete `list_tools`, then manually test each required tool.
4. Obtain the ModelScope URL, add it as a Nexent custom MCP service, enable the six
   tools, debug the agent, and verify a post-publication question.

`docs/PLATFORM_DEPLOYMENT.md` covers the separate authenticated Streamable HTTP
container path. `使用说明.md` documents the local release workflow and constraints.
