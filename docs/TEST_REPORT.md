# 测试报告

测试日期：2026-07-25（Asia/Shanghai）

2026-07-29 更新：新增 GitHub 快速创建的 `mcpServers` 配置测试，并移除发布版中
不参与运行的 DICOM 交换配置。下列“发布包门禁”按当前源代码维护；完整 pytest
执行需要安装 `validation` 可选依赖。

## 1. 结论

比赛发布包已通过本地源码、真实模型、HTTP MCP 和 Docker 容器验收，可以上传到 OpenLab/Nexent 进行平台侧配置。平台账号登录、容器上传、Header 能力确认、智能体提示词粘贴和最终发布仍需在比赛平台人工完成。

## 2. 研发源码门禁

| 检查 | 结果 |
|---|---|
| pytest | 84 passed |
| Ruff | All checks passed |
| mypy strict | 61 source files, no issues |

研发版修复包括 sidecar 调用中未定义 `request`、过时的 `ct_package` 测试、4 个 mypy 错误、22 个 Ruff 错误，以及平台 Host/Origin 环境覆盖能力。

## 3. 比赛发布包门禁

| 检查 | 结果 |
|---|---|
| pytest 发布回归 | 10 个测试函数（安装 `validation` 依赖后执行） |
| Ruff | All checks passed |
| mypy strict | 44 source files, no issues |
| CT 请求 schema | 只允许 `source.mode=precomputed` |
| MCP 工具数 | 固定 6 个 |
| sidecar 客户端 | 不存在 |
| UTF-8/结构审计 | JSON、YAML、Markdown 和 Python 源码结构校验通过 |
| 凭据/绝对路径扫描 | 0 命中 |

`starlette.testclient` 当前会产生一条 `httpx` 接口弃用警告。这是测试依赖接口警告，
不影响服务运行；后续依赖升级时应迁移到推荐客户端。

## 4. 真实模型和黄金病例

- 模型 manifest SHA-256：`003636d691d603908f8bd913b5ab7769a5e1479680c7f8317cda1d106e8bf73f`。
- manifest 实际文件哈希与声明值一致。
- 成员数：5。
- CT 特征：1409。
- 病理特征：768。
- 部署阈值：`0.3529504342004657`。
- `independent_test_claim=false`。
- `demo_case_001` 概率基准：`0.6279473900794983`，绝对容差 `2e-7`。模型使用
  float32 CPU 推理；不同合法 CPU BLAS/PyTorch 构建可能在 softmax 的末位产生约
  `1e-7` 的差异，分类、阈值和模型版本必须保持一致。
- 预测类别：1。
- QC、CT、病理、预测、HTML 报告完整链路通过。

该病例为不含患者信息的零值合成工程回归病例，其概率不代表临床性能。

## 5. Docker 构建与运行

第一次构建发现 PyPI 默认 torch 会拉取 CUDA 13 依赖；Dockerfile 已改为从 PyTorch CPU wheel 源安装 `torch 2.13.0+cpu`。修复后镜像成功构建。

- 镜像：`crc-lnm-mcp:competition`。
- 镜像 ID：`sha256:86cd9214bb2f64f4a0b8d782080a66cb4c819ad5012b4a6fb8a758fffad9e5c3`。
- 镜像大小：431,072,891 bytes。
- 容器用户：`mcp:mcp`，运行 UID 100。
- `torch.cuda.is_available()`：`False`。
- 模型 manifest 权限：`root:root 555`。
- `/health/ready`：HTTP 200，返回正确模型版本。
- 无 Authorization 的 `/mcp`：HTTP 401。
- 正确 Bearer 的 MCP initialize：HTTP 200。
- 临时验收容器在测试后已停止并自动删除。

## 6. 平台侧待办

1. 把 `competition_release` 上传到 OpenLab 操作桌面。
2. 在 Nexent 容器模式构建或导入镜像。
3. 确认平台能设置 Authorization Header；不能设置时按部署文档评估隔离内网备用配置。
4. 配置服务名、端口 8000 和 `/mcp`。
5. 确认 Nexent 页面仅显示 6 个工具。
6. 选择 Qwen3-32B，粘贴 `AGENT_PROMPT.md`。
7. 使用 `demo_case_001` 完成录屏和最终平台验收。
