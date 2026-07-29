# Nexent 同网络 MCP 服务

`compose.nexent.yaml` 使用项目根目录作为完整 Docker 构建上下文，并把服务加入
Nexent 官方 Compose 默认网络 `nexent_network`。容器仅向该网络暴露端口 8000，
不会默认发布到宿主机或公网。

## 启动

先把平台 Secret 放入当前终端的环境变量，不要写入仓库：

```powershell
$env:CRC_LNM_MCP_BEARER_TOKEN = '<平台 Secret>'
docker compose -f compose.nexent.yaml up -d --build
```

如果 Nexent 使用了自定义 Docker 网络，同时设置：

```powershell
$env:NEXENT_DOCKER_NETWORK = '<Nexent 所在 Docker 网络名>'
```

## Nexent URL 配置

- 服务 DNS 名：`crc-lnm-mcp`
- MCP URL：`http://crc-lnm-mcp:8000/mcp`
- 健康检查：`GET http://crc-lnm-mcp:8000/health/ready`，期望 HTTP 200
- 请求 Header：`Authorization: Bearer <平台 Secret>`

容器中的 `CRC_LNM_MCP_BEARER_TOKEN` 与 Nexent 请求 Header 使用的 Secret 必须完全相同。
在 Nexent 的 URL 类型 MCP 表单中，将同一 Secret 填入 Authorization Token；如果表单提供
任意 Header 键值配置，则把键设为 `Authorization`、值设为 `Bearer <平台 Secret>`。

## 验证

```powershell
docker compose -f compose.nexent.yaml ps
docker inspect --format '{{json .State.Health}}' crc-lnm-mcp
```

状态应为 `healthy`。未携带 Authorization 的 `/mcp` 请求应返回 HTTP 401；携带正确
Bearer Header 的 MCP `initialize` 请求应成功。
