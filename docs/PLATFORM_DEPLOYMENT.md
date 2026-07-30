# Platform Deployment

## ModelScope Hosted STDIO

ModelScope hosted STDIO deployment installs a published PyPI package. Use the exact
configuration in `configs/modelscope-mcp.json`; it is deliberately limited to `uvx`
and the published package name:

```json
{
  "mcpServers": {
    "crc-lnm-research-assistant": {
      "command": "uvx",
      "args": [
        "crc-lnm-medical-agent@latest",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

Do not add comments, local paths, repository commands, secrets, or a second server
configuration. The wheel includes the deployment bundle and release JSONL. Its default
startup builds the verified case cache in a writable system cache location, so the
platform does not need repository-level `configs/` or `data/` files.

Passing ModelScope `list_tools` proves only that the service starts and exposes tools.
Test the required tools manually before using the generated URL in Nexent.

## Streamable HTTP Container

The container path is separate from hosted STDIO. Build from the release root; the
Docker build uses `uv sync --locked`, includes the published JSON Schemas, and pins
the dependency graph in `uv.lock`.

Set `CRC_LNM_MCP_BEARER_TOKEN` to a secret of at least 32 bytes. The service exposes
`/mcp`, `/health/live`, and `/health/ready` on port 8000. Configure the `Authorization`
header only on a controlled platform connection.

## Nexent Completion Steps

After ModelScope produces a service URL, add it in Nexent as a custom MCP service,
enable the six `crc_lnm_*` tools, complete connection and tool tests, debug the agent,
publish it, and validate a post-publication question. Network reachability from Nexent
to the ModelScope URL must be tested in the target environment.
