# Nexent 平台容器部署说明

## 1. 上传前准备

1. 在 OpenLab 项目中进入操作桌面。
2. 通过“集成工程 > 文件共享 > 上传文档”把完整 `competition_release` 目录或压缩包传入操作桌面。
3. 不要上传训练数据、真实患者信息、平台密码、API key、Bearer token、私钥或本地缓存。
4. 确认目录中存在 `Dockerfile`、`pyproject.toml`、`src/`、`models/`、`demo/` 和 `configs/`。

## 2. 容器配置

推荐镜像名：`crc-lnm-mcp:competition`，服务端口：`8000`，MCP 路径：`/mcp`。

安全默认环境变量：

```text
CRC_LNM_MCP_CONFIG=/app/configs/mcp.yaml
CRC_LNM_MCP_BEARER_TOKEN=<平台Secret>
CRC_LNM_MCP_ALLOWED_HOSTS=<平台分配服务名>:*,localhost:*,127.0.0.1:*
CRC_LNM_MCP_ALLOWED_ORIGINS=http://nexent:3000
```

Bearer token 至少使用 32 字节随机值，只能通过平台 Secret 注入，不得写进 URL、YAML、提示词、截图或录屏。

容器启动后，先确认：

```text
/health/live  -> HTTP 200, {"status":"live"}
/health/ready -> HTTP 200, {"status":"ready","model_version":"..."}
```

`ready` 表示模型 manifest、权重、预处理器和 5 个成员已经加载并通过校验。

## 3. Nexent 添加 MCP

1. 登录 Nexent，进入“MCP工具”。
2. 选择“添加 MCP 服务 > 自定义”，切换到平台提供的容器部署方式。
3. 使用本目录作为 Docker 构建上下文，容器端口填写 `8000`。
4. 服务 URL 填写 `http://<平台分配服务名>:8000/mcp`。
5. 如果界面支持 Header，配置 `Authorization: Bearer <secret>`。
6. 保存、启动并启用服务。
7. 确认仅出现 README 中列出的 6 个 `crc_lnm_*` 工具。

平台版本的容器表单字段可能与指导截图略有差异，应以现场界面为准，但不得改变 `/mcp`、端口 8000 和六工具口径。

## 4. 无 Header 能力时的受限方案

只有同时满足以下条件时，才可把 `CRC_LNM_MCP_CONFIG` 改为 `/app/configs/mcp.nexent-internal.yaml`：

- Nexent 不能向 MCP 请求注入 Authorization Header。
- 容器只在比赛平台隔离网络中可达。
- 平台对用户、项目和服务已有访问控制。
- 服务没有公网入口，也没有从不受信网络进入的反向代理。

任何条件不满足都必须继续使用 Bearer 安全默认配置，并联系平台管理员解决 Header 注入。

## 5. 创建智能体

1. 在“智能体开发 > 新建”中选择 `Qwen3-32B`。
2. 如需建设知识库，平台模型使用 `Qwen3-embedding-8B` 和 `bge-reranker-v2-m3`；知识库只能提供医学依据，不能生成病例概率。
3. 添加本 MCP 的六个工具。
4. 将 `AGENT_PROMPT.md` 内容作为系统提示词。
5. 调试黄金病例、错误输入和跨 trace 拒绝场景。
6. 确认输出含科研辅助声明后再发布。

## 6. 演示与提交安全

- 录屏只使用 `demo_case_001`。
- 遮挡服务域名中的凭据、Header、平台账号和内部地址。
- 不展示 2177 维原始特征、模型权重内容或服务器路径。
- 不把合成病例结果描述为临床准确率。
- Docker 守护进程和 Nexent 页面中的实际构建、启用、发布需要在 OpenLab 操作桌面完成。
