# 结直肠癌淋巴结转移多模态科研辅助 MCP

这是一个可由 MCP 客户端调用的只读科研辅助服务。它对服务端白名单中的脱敏病例执行固定工作流：病例质控、CT 特征准备、病理特征准备、五成员集成推理和 HTML 科研辅助报告生成。

> 仅用于科研辅助和病例复核，不替代病理诊断、治疗建议或临床最终决策。

## GitHub 快速创建

截图中的创建失败并非模型或 Docker 构建失败，而是平台把 README 中错误的根 JSON 当作“服务配置”解析。平台要求根字段为 `mcpServers`，而旧示例的根字段是环境变量名，因此校验失败。

在“从 GitHub 仓库快速创建”的“服务配置”框内，粘贴 [configs/modelscope-mcp.json](configs/modelscope-mcp.json) 的完整内容：

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

该配置以标准输入输出（stdio）运行。托管平台负责启动该进程并转发 MCP 消息，所以这里不填 URL、端口、Docker 命令或 Bearer token。不要把环境变量字段定义粘贴到“服务配置”框。

## 功能边界

- 输入：预提取 CT 特征 1409 维、病理特征 768 维，以及 `age`、`male`、`Type`、`T` 四项临床变量。
- 模型：5 个固定成员，部署阈值为 `0.3529504342004657`。
- 输出：风险概率、阈值关系、复核优先级、审计信息和转义 HTML 报告。
- MCP 仅暴露 6 个只读工具。

本版本不接受原始 DICOM、ROI、NIfTI、WSI、patch 或客户端文件路径；不运行 ResNet；不生成热图或特征因果归因。

## 六工具工作流

1. `crc_lnm_get_model_info`
2. `crc_lnm_case_data_qc`
3. `crc_lnm_prepare_ct_features`
4. `crc_lnm_prepare_pathology_features`
5. `crc_lnm_predict_multimodal`
6. `crc_lnm_generate_report`

每个病例使用一个新的 UUIDv4 `trace_id`，并在该病例的所有后续工具调用中保持不变。任一步失败即停止；artifact 不能跨病例、跨 trace 或跨服务重启复用。

## 本地验证

需要 Python 3.12 或 3.13 和 `uv`。在项目根目录执行：

```powershell
uv sync --extra mcp --extra validation
uv run --extra mcp --extra validation pytest -q
uv run --extra mcp crc-lnm-mcp --config configs/mcp.local.yaml --transport stdio
```

本地 stdio 配置不启用 Bearer 校验，仅用于同一受控主机上的 MCP 客户端或托管平台子进程。它会在 `artifacts_local/` 写入短期运行 artifact；该目录不属于发布内容。

## Docker / HTTP 部署

外部 HTTP 部署使用 `/mcp` 路径和 `8000` 端口：

```powershell
docker build -t crc-lnm-mcp:competition .
docker run --rm -p 8000:8000 `
  -e CRC_LNM_MCP_BEARER_TOKEN="<由平台 Secret 注入的至少 32 字节随机值>" `
  -e CRC_LNM_MCP_ALLOWED_HOSTS="<平台服务名>:*,localhost:*,127.0.0.1:*" `
  -e CRC_LNM_MCP_ALLOWED_ORIGINS="http://nexent:3000" `
  crc-lnm-mcp:competition
```

健康检查为 `GET /health/live` 和 `GET /health/ready`。HTTP 客户端必须配置 `Authorization: Bearer <同一 Secret>`。若平台无法注入 Header，只能在确认服务完全隔离且无公网入口后使用 `configs/mcp.nexent-internal.yaml`。

## 项目结构

- `configs/modelscope-mcp.json`：可直接导入的标准 `mcpServers` 服务配置。
- `configs/mcp.local.yaml`：GitHub 托管与本地 stdio 运行配置。
- `configs/mcp.yaml`：Docker/HTTP 的安全默认配置。
- `src/`：MCP、模型加载、输入校验和报告代码。
- `models/deployment_bundle/`：经完整性校验的五成员模型。
- `data/` 与 `scripts/build_case_packages.py`：Docker 构建期的白名单病例包数据及转换器。
- `demo/cases/demo_case_001/`：无患者信息的合成黄金回归病例。
- `docs/`：部署、智能体提示词、模型卡、测试报告与提交清单。

完整平台操作见 [使用说明](使用说明.md) 和 [平台容器部署说明](docs/PLATFORM_DEPLOYMENT.md)。
