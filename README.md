# CRC-LNM 多模态科研辅助 MCP

这是一个只读的多模态科研辅助 MCP 服务。它仅处理服务端白名单病例的预计算特征：1409 维 CT、768 维病理特征和四项临床字段，并输出复核优先级与科研辅助报告。

原始 NIfTI、DICOM、WSI、patch、文件路径和不同维度的外部特征均不是可推理输入。给定 JSONL 会在构建期或本地 stdio 首次启动时转换为受控病例包；MCP 调用只使用 `case_ref`。

完整的本地测试、iData JSONL 使用方式、截图问题说明、托管配置和 Docker 部署步骤见 [使用说明](使用说明.md)。智能体系统提示词见 [docs/AGENT_PROMPT.md](docs/AGENT_PROMPT.md)。

## ModelScope 从 GitHub 快速创建

在 ModelScope MCP 广场选择“从 GitHub 仓库快速创建”时，请填写仓库根地址 `https://github.com/lisazhanggg0810-source/191-agent`。平台会从根目录 `README.md` 的 JSON 代码块解析服务配置，因此以下内容必须保持为合法 JSON：

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

该入口从 PyPI 安装 `crc-lnm-medical-agent`，当前发布版本为 `1.0.6`；控制台入口默认使用 stdio。不要在上述配置中加入仓库相对路径、Bearer token 或本地文件路径。
