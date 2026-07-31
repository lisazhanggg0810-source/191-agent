# 平台部署说明

## stdio 托管

平台的“服务配置”必须使用 `configs/modelscope-mcp.json` 的完整 JSON。根字段必须为 `mcpServers`：

```json
{
  "mcpServers": {
    "crc-lnm-research-assistant": {
      "command": "uv",
      "args": [
        "run",
        "--extra",
        "mcp",
        "crc-lnm-mcp",
        "--config",
        "configs/mcp.local.yaml",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

不要在该文本框填写 URL、Docker 命令、环境变量或密钥。首次启动会从仓库内的 `data/release_case_package_groups.jsonl` 建立 `artifacts_local/cases` 缓存；该位置必须可写。

## HTTP 容器

容器端口为 `8000`，MCP 路径为 `/mcp`，健康检查为 `/health/live` 和 `/health/ready`。生产配置要求通过环境变量 `CRC_LNM_MCP_BEARER_TOKEN` 注入至少 32 字节的随机 secret；不要把 token 提交到仓库、YAML、截图或系统提示词。

平台能够设置 Header 时，向 MCP 请求注入 `Authorization: Bearer <secret>`。无法注入 Header 时，只能在服务完全位于受控隔离网络且没有公网入口时使用 `configs/mcp.nexent-internal.yaml`。

## 智能体

创建智能体后只添加六个 `crc_lnm_*` 工具，并将 `AGENT_PROMPT.md` 内容设为系统提示词。提示词已明确阻止把 NIfTI、WSI、文件路径、任务编号或不同维度的上游特征发送给推理流程。
