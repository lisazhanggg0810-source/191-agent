# 平台部署说明

## stdio 托管

在 ModelScope MCP 广场选择“从 GitHub 仓库快速创建”时，推荐填写 README 完整地址 `https://github.com/lisazhanggg0810-source/191-agent/blob/main/README.md`。不要使用带 `/tree/main` 的分支页面地址；平台前后端对这类地址的规范化结果可能不同，导致页面一直停留在“解析中”。平台会从 `README.md` 的 JSON 代码块解析服务配置；该代码块必须与 `configs/modelscope-mcp.json` 保持一致，且根字段必须为 `mcpServers`：

```json
{
  "mcpServers": {
    "crc-lnm-research-assistant": {
      "command": "uvx",
      "args": [
        "--no-progress",
        "--torch-backend",
        "cpu",
        "--from",
        "crc-lnm-medical-agent==1.0.6",
        "crc-lnm-medical-agent"
      ]
    }
  }
}
```

该配置通过 `uvx --from` 从 PyPI 显式安装并启动 `crc-lnm-medical-agent==1.0.6`，控制台入口默认使用 stdio。固定版本和入口名可以避免托管平台误判 `uvx 包名@latest`；`--torch-backend cpu` 使 PyTorch 使用 CPU 包源，并避免平台要求另行填写环境变量。发布包已经包含不可变的模型资源和病例 JSONL；首次启动会在平台可写的运行时目录建立缓存，不依赖仓库相对路径。不要在配置中加入 URL、Docker 命令、Bearer token、密钥或本地文件路径。

正常解析通常在一分钟内完成。若超过 60 秒仍停留在“解析中”，取消并刷新页面后使用上述 README 完整地址重试；若仍未完成，点击“进入自定义创建”，选择 STDIO，并粘贴 `configs/modelscope-mcp.json` 的完整内容。自定义创建会绕过 GitHub 自动解析，后续仍使用同一个 PyPI 包执行部署检测。

## HTTP 容器

容器端口为 `8000`，MCP 路径为 `/mcp`，健康检查为 `/health/live` 和 `/health/ready`。生产配置要求通过环境变量 `CRC_LNM_MCP_BEARER_TOKEN` 注入至少 32 字节的随机 secret；不要把 token 提交到仓库、YAML、截图或系统提示词。

平台能够设置 Header 时，向 MCP 请求注入 `Authorization: Bearer <secret>`。无法注入 Header 时，只能在服务完全位于受控隔离网络且没有公网入口时使用 `configs/mcp.nexent-internal.yaml`。

## 智能体

创建智能体后只添加六个 `crc_lnm_*` 工具，并将 `AGENT_PROMPT.md` 内容设为系统提示词。提示词已明确阻止把 NIfTI、WSI、文件路径、任务编号或不同维度的上游特征发送给推理流程。
