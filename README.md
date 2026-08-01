# CRC-LNM 多模态科研辅助 MCP

这是一个只读的多模态科研辅助 MCP 服务。它仅处理服务端白名单病例的预计算特征：1409 维 CT、768 维病理特征和四项临床字段，并输出复核优先级与科研辅助报告。

原始 NIfTI、DICOM、WSI、patch、文件路径和不同维度的外部特征均不是可推理输入。给定 JSONL 会在构建期或本地 stdio 首次启动时转换为受控病例包；MCP 调用只使用 `case_ref`。

完整的本地测试、iData JSONL 使用方式、截图问题说明、托管配置和 Docker 部署步骤见 [使用说明](使用说明.md)。智能体系统提示词见 [docs/AGENT_PROMPT.md](docs/AGENT_PROMPT.md)。

## ModelScope 从 GitHub 快速创建

在 ModelScope MCP 广场选择“从 GitHub 仓库快速创建”时，推荐直接填写 README 完整地址 `https://github.com/lisazhanggg0810-source/191-agent/blob/main/README.md`。不要填写带 `/tree/main` 的分支页面地址；这类地址可能因平台前后端规范化结果不同而使页面停留在“解析中”。平台会从 `README.md` 的 JSON 代码块解析服务配置，因此以下内容必须保持为合法 JSON：

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

该入口通过 `uvx --from` 从 PyPI 显式安装并启动 `crc-lnm-medical-agent==1.0.6`，控制台入口默认使用 stdio。固定版本和入口名可以避免托管平台误判 `uvx 包名@latest`；`--torch-backend cpu` 使 PyTorch 使用 CPU 包源，并避免平台要求另行填写环境变量。不要在上述配置中加入仓库相对路径、Bearer token 或本地文件路径。

正常解析通常在一分钟内完成。若页面超过 60 秒仍停留在“解析中”，请取消后刷新创建页并改用上述 README 完整地址；仍未完成时，点击“进入自定义创建”，选择 STDIO，并直接粘贴同一份 `configs/modelscope-mcp.json` 配置。自定义创建不依赖 GitHub 自动解析。
