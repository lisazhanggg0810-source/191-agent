"""结直肠癌淋巴结转移 MCP 的核心基础设施。

本包只封装确定性的本地能力。原始 DICOM/WSI 提取、外部网络访问和聊天模型判断
不属于核心 V1 服务的职责。
"""

from wei_multimodal.mcp_server.errors import ContractError, ErrorCode

__all__ = ["ContractError", "ErrorCode"]
