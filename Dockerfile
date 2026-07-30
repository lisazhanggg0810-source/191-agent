# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.7.19 AS uv

FROM python:3.13-slim AS case_builder

WORKDIR /build
COPY scripts/build_case_packages.py /build/build_case_packages.py
COPY src /build/src
COPY data/release_case_package_groups.jsonl /build/release_case_package_groups.jsonl
RUN python /build/build_case_packages.py \
    --input /build/release_case_package_groups.jsonl \
    --output /build/cases

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CRC_LNM_MCP_CONFIG=/app/configs/mcp.yaml \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra mcp --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --extra mcp

COPY configs /app/configs
COPY models /app/models
COPY demo /app/demo
COPY schemas /app/schemas
COPY --from=case_builder /build/cases/ /app/demo/cases/

RUN addgroup --system mcp \
    && adduser --system --ingroup mcp --home /nonexistent --no-create-home mcp \
    && mkdir -p /data/artifacts \
    && chown -R mcp:mcp /data \
    && chmod -R a-w /app/models /app/demo

USER mcp:mcp
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-m", "wei_multimodal.mcp_server.healthcheck"]

ENTRYPOINT ["python", "-m", "wei_multimodal.mcp_server"]
CMD ["--transport", "streamable-http"]
