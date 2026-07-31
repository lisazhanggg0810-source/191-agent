"""Command-line entry point for stdio and Streamable HTTP transports."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from wei_multimodal.mcp_server.app import create_http_app, create_mcp, load_default_settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CRC-LNM core MCP service.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the non-secret MCP YAML configuration.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
    )
    return parser


def main() -> None:
    """Load validated settings and run the selected official MCP transport."""

    arguments = _parser().parse_args()
    settings = load_default_settings(arguments.config)
    if arguments.transport == "stdio":
        mcp, _ = create_mcp(settings)
        mcp.run(transport="stdio")
        return
    app = create_http_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
