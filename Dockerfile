# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS case_builder

WORKDIR /build
COPY scripts/build_case_packages.py /build/build_case_packages.py
COPY data/release_case_package_groups.jsonl /build/release_case_package_groups.jsonl
RUN python /build/build_case_packages.py \
    --input /build/release_case_package_groups.jsonl \
    --output /build/cases

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CRC_LNM_MCP_CONFIG=/app/configs/mcp.yaml

WORKDIR /app

COPY pyproject.toml /build/pyproject.toml
COPY src /build/src
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.9,<3" \
    && python -m pip install "/build" \
    && rm -rf /build

COPY configs /app/configs
COPY models /app/models
COPY demo /app/demo
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
