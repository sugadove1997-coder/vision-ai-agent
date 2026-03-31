# 基于多模态 AI Agent 的创意设计协作助手

毕业设计方向：**文本需求 → 初稿图像 → 多轮迭代优化** 的闭环；后端采用 **LangChain / LangGraph 图结构**（意图识别、需求分析、主动追问、流程规划），图像侧 **文生图**走 **Hugging Face 免费推理**（Flux / SDXL / SD1.5）；**图生图**仅 **本机 Diffusers（MPS/CUDA，8010，见 `docs/LOCAL_IMG2IMG.md`）**；对话过长时在 Agent 侧做 **摘要压缩**。

## 架构概览

| 模块 | 说明 |
|------|------|
| `agent/` | Python + FastAPI + **LangGraph**：`compress → intent → clarify \| plan` |
| `server/image.ts` | Node：**文生图** `/api/image`（HF）；**图生图** `/api/image/refine`（仅本地 `8010`） |
| `agent/local_img2img_service.py` | 本机 **Diffusers img2img**（默认 SD1.5，`npm run local-img2img`） |
| `src/App.tsx` | React：多轮对话、**上传参考图/待改底图**、工作流追踪、自动触发生图/图生图迭代 |

## 环境要求

- **Node.js** 18+
- **Python 3.10+**（推荐；用于 LangGraph Agent）
- **Hugging Face 免费账号 + Token**（**文生图**走官方免费额度；**图生图**仅本机 Diffusers，不耗 HF 图生图 API）

## 快速开始

### 1. 前端与 SD 服务

```bash
npm install
cp .env.example .env
# 编辑 .env：填写 HUGGINGFACE_TOKEN；国内网络可设 HF_PROXY
npm run dev:all
```

浏览器访问：<http://localhost:3000>（仅前端+生图，**无** LangGraph 时需用 `dev:thesis`）。

### 2. LangGraph Agent（完整论文功能）

```bash
cd agent
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-img2img.txt
cd ..
```

> **图生图本机推理**：`requirements-img2img.txt` 体积较大（含 PyTorch）；安装后 `npm run dev:thesis` 会同时启动 **8010** 本地 img2img。日志会写入 **`agent/logs/local-img2img.log`**，可在项目根另开终端执行 **`tail -f agent/logs/local_img2img.log`** 实时查看（不必找「是哪一个终端标签」）。详见 **`docs/LOCAL_IMG2IMG.md`**。
>
> 若 `/api/image/refine` 报 **timeout**：本地首次拉模型很慢，可在 `.env` 设 `LOCAL_IMG2IMG_TIMEOUT_MS=900000`（15 分钟）后重启 `npm run image-server`。

在项目根 `.env` 中配置**文本推理**（LangGraph）：

**免费组合（推荐答辩）：** 与「HF 免费生图」搭配，只再配 **Gemini** 即可。

```env
GEMINI_API_KEY=AIza...   # https://aistudio.google.com 免费申请
# 可选：GEMINI_AGENT_MODEL=gemini-2.0-flash
```

若已配置 **豆包方舟**（`DOUBAO_API_KEY` + `DOUBAO_MODEL`），默认会优先走方舟；想强制用 Gemini 时设 `AGENT_LLM_PROVIDER=gemini`。

### 本机 Ollama（最小可用、零云 API Key）

1. 安装并启动 [Ollama](https://ollama.com)，拉一个**小模型**（显存紧可用 3B 级）：

```bash
ollama pull llama3.2
# 或中文稍好：ollama pull qwen2.5:3b
```

2. 确认服务在本机：`curl http://127.0.0.1:11434/api/tags`

3. 项目根 `.env`：

```env
AGENT_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
# OLLAMA_BASE_URL=http://127.0.0.1:11434
```

4. `pip install -r agent/requirements.txt` 后执行 `npm run dev:thesis`。

> 生图仍用 `HUGGINGFACE_TOKEN`（Ollama 只管 LangGraph 的**文字**意图/规划）。

### 怎么查看本机「显存」

| 系统 | 做法 |
|------|------|
| **Windows** | 任务管理器 → **性能** → 选 **GPU** → 看「**专用 GPU 内存**」 |
| **NVIDIA（Linux）** | 终端执行 `nvidia-smi`，看 **Memory-Usage** |
| **macOS Apple 芯片** | 没有独立显存条，GPU 与 **统一内存** 共用；**关于本机** 看总内存；可选终端：`system_profiler SPDisplaysDataType` 看显卡信息 |
| **macOS Intel + AMD 独显** | 同上 **关于本机** / **系统信息** → 图形卡 |

或使用显式 OpenAI 兼容变量：

```env
LLM_API_KEY=...
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_MODEL=ep-你的接入点ID
```

```env
# 可选：超过多少条消息触发摘要（默认 10），保留最近几条（默认 4）
# AGENT_COMPRESS_AFTER=10
# AGENT_KEEP_RECENT=4
```

一键启动（生图服务 + Agent :8008 + Vite :3000）：

```bash
npm run dev:thesis
```

## API 说明

- `POST /api/agent/invoke`（经 Vite 代理）→ FastAPI `POST /invoke`  
  Body: `{ "messages": [{"role":"user"|"assistant","content":"..."}], "has_canvas_image": boolean }`  
  返回: `assistant_message`, `action`（`none` \| `generate` \| `refine` \| `chat`）, `sd_prompt`, `trace`

## 许可证

示例代码可随毕业设计说明使用；第三方模型与 API 遵循各平台条款。
