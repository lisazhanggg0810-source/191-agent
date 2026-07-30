# ModelScope MCP Hosted 部署操作手册

本文档记录如何将 `191-agent` 构建为自包含 Python 包，发布到 PyPI，并通过 `uvx` 部署为 ModelScope Hosted MCP。

## 项目信息

- GitHub：<https://github.com/lisazhanggg0810-source/191-agent>
- 本地仓库：`C:\Users\64666\Documents\体后\191-agent`
- PyPI 包名：`crc-lnm-medical-agent`
- 发布版本：`1.0.1`
- 命令行入口：`crc-lnm-mcp`
- ModelScope 内部传输：`stdio`
- ModelScope 对外传输：`streamable_http`

## 当前执行状态

- [x] 创建干净的 `191-agent` 仓库并把项目放到根目录
- [x] 把托管 YAML、5 个模型成员和演示病例放入 Python 包
- [x] 删除托管配置中的固定 artifact 路径
- [x] 默认从包内读取配置和资源
- [x] artifact 默认写入系统临时目录
- [x] 把运行依赖移入正式依赖
- [x] ModelScope 配置改为固定版本 `uvx`
- [x] Ruff 检查通过
- [x] Mypy 检查通过（45 个源文件）
- [x] Pytest 通过（11 项）
- [x] 构建 wheel 和源码包
- [x] 隔离环境验证 5 个模型和 6 个工具
- [x] 提交并推送 GitHub
- [ ] 发布 PyPI
- [ ] 创建并验收 ModelScope Hosted MCP

## 一、准备环境

### 进入仓库

```powershell
Set-Location "C:\Users\64666\Documents\体后\191-agent"
```

### 检查 Git

```powershell
git --version
git remote -v
git status
```

远程地址应为：

```text
https://github.com/lisazhanggg0810-source/191-agent.git
```

如果找不到 Git：

```powershell
winget install --id Git.Git -e --source winget
```

### 检查 uv 和 Python

```powershell
uv --version
uv python install 3.12
uv python pin 3.12
uv run python --version
```

应使用 `Python 3.12.x`。uv 管理的 Python 与电脑已有 Python 可以共存。

## 二、包内资源结构

运行资源位于：

```text
src/wei_multimodal/resources/
├── __init__.py
├── configs/
│   └── mcp.hosted.yaml
├── demo/
│   └── cases/demo_case_001/
└── models/
    └── deployment_bundle/
        ├── manifest.json
        ├── preprocessing.json
        ├── preprocessing.npz
        ├── schema.json
        ├── threshold.json
        └── seed_*/model_state.pt
```

验收命令：

```powershell
(Get-ChildItem "src\wei_multimodal\resources\models\deployment_bundle" -Recurse -File).Count
(Get-ChildItem "src\wei_multimodal\resources\demo\cases" -Recurse -File).Count
Test-Path "src\wei_multimodal\resources\configs\mcp.hosted.yaml"
```

预期结果：

```text
16
4
True
```

托管 YAML 中不能存在：

```yaml
artifact_root: ../artifacts_local
```

检查命令：

```powershell
Select-String -Path "src\wei_multimodal\resources\configs\mcp.hosted.yaml" -Pattern "^artifact_root:"
```

正确情况是没有输出。

## 三、关键代码配置

### 临时 artifact 目录

`src/wei_multimodal/mcp_server/settings.py` 使用：

```python
def _default_artifact_root() -> Path:
    return Path(tempfile.gettempdir()) / "crc_lnm_artifacts"
```

Docker 配置显式提供 `/data/artifacts` 时仍使用 Docker 路径；PyPI 托管配置未提供路径时使用系统临时目录。

### 默认包内配置

`src/wei_multimodal/mcp_server/app.py` 使用 `importlib.resources` 定位：

```text
wei_multimodal/resources/configs/mcp.hosted.yaml
```

ModelScope 启动命令不再传递仓库相对的 `--config`。

### PyPI 正式依赖

运行所需的以下依赖必须位于 `[project].dependencies`：

```text
mcp
numpy
pandas
pydantic
pyyaml
scikit-learn
imbalanced-learn
jinja2
torch
starlette
uvicorn
```

`validation` 组只保留 pytest、ruff、mypy 等开发检查工具。

### wheel 资源规则

`pyproject.toml` 的 `[tool.setuptools.package-data]` 必须覆盖：

```text
configs/*.yaml
models/deployment_bundle/*
models/deployment_bundle/*/*
demo/cases/*/*.json
```

## 四、ModelScope 服务配置

`configs/modelscope-mcp.json` 内容：

```json
{
  "mcpServers": {
    "crc-lnm-research-assistant": {
      "command": "uvx",
      "args": [
        "--index",
        "https://download.pytorch.org/whl/cpu",
        "--from",
        "crc-lnm-medical-agent==1.0.1",
        "crc-lnm-mcp",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

必须保留 PyTorch CPU 索引，否则 Linux 环境可能下载体积很大的 CUDA 依赖。

ModelScope stdio 托管不填写以下 Docker 环境变量：

```text
CRC_LNM_MCP_CONFIG
CRC_LNM_MCP_PORT
CRC_LNM_MCP_BEARER_TOKEN
CRC_LNM_MCP_ALLOWED_HOSTS
CRC_LNM_MCP_ALLOWED_ORIGINS
```

## 五、本地测试和构建

### 同步环境

```powershell
uv sync --python 3.12 --extra validation
```

### 运行质量检查

```powershell
uv run ruff check src tests
uv run mypy
uv run pytest -q
```

本次实际结果：

```text
Ruff: All checks passed
Mypy: Success, 45 source files
Pytest: 11 passed
```

### 构建

```powershell
uv build --no-sources
```

产物：

```text
dist/crc_lnm_medical_agent-1.0.1-py3-none-any.whl
dist/crc_lnm_medical_agent-1.0.1.tar.gz
```

本次 wheel 大小约 14.3 MB，包含：

- 1 个 `mcp.hosted.yaml`
- 5 个 `model_state.pt`
- 4 个病例 JSON
- 1 个报告模板

### 隔离验收

```powershell
$wheel = (Get-ChildItem "dist\*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$smokeDir = Join-Path $env:TEMP "crc-lnm-wheel-smoke-1.0.1"
New-Item -ItemType Directory -Force $smokeDir | Out-Null
Push-Location $smokeDir
uv run --no-project --python 3.12 --index "https://download.pytorch.org/whl/cpu" --with $wheel -- python -c "import asyncio; from wei_multimodal.mcp_server.app import load_default_settings, create_mcp, _build_runtime; s=load_default_settings(); r=_build_runtime(s); m,_=create_mcp(s); tools=asyncio.run(m.list_tools()); assert r.prediction_service.get_info().member_count == 5; assert len(tools) == 6; print('PASS: wheel=1.0.1 models=5 tools=6')"
Pop-Location
```

本次实际结果：

```text
PASS: wheel=1.0.1 models=5 tools=6
```

包从 uv 缓存的 `site-packages` 加载，配置同样来自安装包，artifact 路径为：

```text
C:\Users\64666\AppData\Local\Temp\crc_lnm_artifacts
```

## 六、提交并推送 GitHub

确认修改：

```powershell
git status
git diff --check
```

提交并推送：

```powershell
git add .
git commit -m "Package runtime assets for hosted MCP"
git push origin main
```

`dist/`、`.venv/`、`build/` 和 Python 缓存受 `.gitignore` 保护，不提交到 GitHub；`uv.lock` 应提交。

## 七、发布 PyPI

### 创建账号和 Token

1. 打开 <https://pypi.org/account/register/>；
2. 注册并验证邮箱；
3. 开启两步验证并保存恢复码；
4. 第一次发布时创建账号范围 API Token；
5. 首次发布成功后，换成项目范围 Token 或 GitHub Trusted Publisher。

不要把 Token 写入代码、JSON、GitHub、截图或聊天。

### 安全发布

在自己的 PowerShell 中执行：

```powershell
Set-Location "C:\Users\64666\Documents\体后\191-agent"
$secureToken = Read-Host "请粘贴 PyPI Token" -AsSecureString
$env:UV_PUBLISH_TOKEN = [System.Net.NetworkCredential]::new("", $secureToken).Password
uv publish
Remove-Item Env:UV_PUBLISH_TOKEN
Remove-Variable secureToken
```

发布成功后检查：

<https://pypi.org/project/crc-lnm-medical-agent/1.0.1/>

PyPI 同一版本不能覆盖。发布后如果代码发生变化，必须升级到 `1.0.2`。

### 测试 PyPI 包

在仓库外的空目录执行：

```powershell
uvx --refresh --index "https://download.pytorch.org/whl/cpu" --from "crc-lnm-medical-agent==1.0.1" crc-lnm-mcp --transport stdio
```

如果服务保持运行且没有缺文件、缺模块或权限错误，说明入口正常。按 `Ctrl+C` 停止。

## 八、创建 ModelScope Hosted MCP

打开：<https://modelscope.cn/mcp/servers/create?template=customize>

填写：

- 来源地址：`https://github.com/lisazhanggg0810-source/191-agent`
- 托管类型：`可托管部署`
- 服务配置：粘贴 `configs/modelscope-mcp.json`
- 环境变量：全部留空
- 云端输出协议：`streamable_http`
- README：使用仓库根目录 `README.md`

如果创建后需要单独部署，同样选择 `streamable_http`。

最终验收：

```json
{
  "is_hosted": true,
  "operational_urls": [
    {
      "transport_type": "streamable_http",
      "url": ".../mcp"
    }
  ]
}
```

## 九、常见错误

| 错误 | 原因 | 处理方法 |
|---|---|---|
| `git` 无法识别 | Git 未安装或终端未刷新 PATH | 安装 Git，关闭并重新打开终端 |
| `artifact_root:` 无法识别 | 把 YAML 内容输入了 PowerShell | 使用记事本编辑 YAML |
| `No module named mcp` 或 `torch` | 运行依赖仍在可选组 | 移入 `[project].dependencies` |
| 找不到 `configs/mcp.yaml` | 仍依赖仓库配置路径 | 使用包内 `mcp.hosted.yaml` |
| 找不到 `model_state.pt` | wheel 没有打包模型 | 检查 `package-data` 和 wheel 内容 |
| `Permission denied` | artifact 写入只读安装目录 | 使用系统临时目录 |
| 下载巨大 CUDA 包 | 缺少 CPU 索引 | 在 `uvx` 中加入 PyTorch CPU `--index` |
| PyPI 文件已存在 | 同版本不可覆盖 | 升级版本号后重建 |
| ModelScope 降级为 Local | 云端启动失败或超时 | 先确保空目录 `uvx` 验收通过，再查看平台日志 |

## 十、最终验收清单

- [x] 本地静态检查和测试全部通过
- [x] wheel 包含托管配置、5 个模型成员和病例
- [x] 本地隔离测试输出 `models=5 tools=6`
- [x] GitHub `main` 已包含本次修改
- [ ] PyPI 存在 `crc-lnm-medical-agent==1.0.1`
- [ ] 从 PyPI 的空目录 `uvx` 可以启动
- [ ] ModelScope 显示 Hosted
- [ ] `is_hosted` 为 `true`
- [ ] `operational_urls` 存在 `streamable_http` 的 `/mcp` 地址
