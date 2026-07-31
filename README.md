# CRC-LNM 多模态科研辅助 MCP

这是一个只读的多模态科研辅助 MCP 服务。它仅处理服务端白名单病例的预计算特征：1409 维 CT、768 维病理特征和四项临床字段，并输出复核优先级与科研辅助报告。

原始 NIfTI、DICOM、WSI、patch、文件路径和不同维度的外部特征均不是可推理输入。给定 JSONL 会在构建期或本地 stdio 首次启动时转换为受控病例包；MCP 调用只使用 `case_ref`。

完整的本地测试、iData JSONL 使用方式、截图问题说明、托管配置和 Docker 部署步骤见 [使用说明](使用说明.md)。智能体系统提示词见 [docs/AGENT_PROMPT.md](docs/AGENT_PROMPT.md)。
