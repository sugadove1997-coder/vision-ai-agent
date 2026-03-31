# 本地 Diffusers 图生图（Mac M 系列 / CUDA）

## 运行前（原因）

- **为什么**：Hugging Face 免费 **serverless 图生图** 常返回 404 / 限流；你本机为 **Apple M3 Pro + 36GB 内存 + MPS**，足够在本地跑 **SD1.5 img2img**（`runwayml/stable-diffusion-v1-5`），延迟与稳定性更好。
- **做什么**：安装依赖 → 启动 `8010` 端口微服务 → `npm run dev:thesis` 会连同该服务一起启动（`I` 进程）。

## 「local-img2img 终端」到底在哪看？

脚本里说的 **local-img2img 终端** = **正在跑 `npm run local-img2img`（或 `dev:thesis` 里负责 8010 的那一路）的那个窗口**。

| 你怎么启动的 | 去哪看 |
|--------------|--------|
| **单独** `npm run local-img2img` | Cursor / VS Code：**终端（Terminal）**面板里，选跑这条命令的那一个标签页；启动时会打印几行说明。 |
| **`npm run dev:thesis`**（四条一起） | 同一个终端里会混排多路输出，**图生图**相关行前面有 **`[I]`** 前缀（`I` = img2img）。向上翻找 `[I]` 和 `[local_img2img]`。 |
| **不想找窗口** | 在项目根**再开一个终端**，执行：`tail -f agent/logs/local-img2img.log` —— 日志会**实时追加**（与屏幕上是同一份内容）。按 `Ctrl+C` 只停止查看，**不会**关掉 8010 服务。 |

日志文件路径：**`vision-ai-agent/agent/logs/local-img2img.log`**（已 `.gitignore`，不会提交）。

## 安装（只需一次）

```bash
cd agent
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-img2img.txt
```

首次图生图会从 Hub **下载约数 GB 权重**（需网络；公开模型一般不需 Token）。

## 环境变量（可选，写在项目根 `.env`）

| 变量 | 说明 |
|------|------|
| `LOCAL_IMG2IMG_URL` | 默认 `http://127.0.0.1:8010`。 |
| `LOCAL_IMG2IMG_MODEL` | 默认 `runwayml/stable-diffusion-v1-5`。 |
| `HF_HOME` | 可选。未设置时服务会把 Hub 缓存放到 **`agent/.hf_hub_cache`**，避免部分终端对 `~/.cache/huggingface` 报 **Operation not permitted**。若要用系统全局缓存可显式设 `HF_HOME`。 |

## 进度日志（判断是否卡住）

服务会把进度打到 **本窗口 + `agent/logs/local-img2img.log`**（由启动脚本 `tee` 写入），每行带 **`HH:MM:SS` 时间戳**，例如：

- `收到 POST /img2img … pipeline_loaded=…`：请求已进服务。
- **加载阶段**：每 **15s** 打一行 `…仍在加载 pipeline（已 Ns）`——说明还在 `from_pretrained` / 下载，**不是卡死**。
- `权重已载入内存，正在 .to(device)…` → `加载完成…（总耗时约 Xs）`：模型就绪。
- `去噪 [####----------] step 3/28 …（请求已 Xs）`：逐步去噪；Diffusers 可能还会打自己的 **tqdm** 进度条。
- `推理成功 … 本请求总耗时 Xs`：已返回 PNG。

**Node `image-server`** 在等待 8010 时，默认 **每 10s** 在终端打 `[img2img] 仍在等待 8010…`（可用 `.env` 里 `IMG2IMG_WAIT_LOG_MS` 调整）。

**浏览器**：图生图时画布遮罩与对话区会显示 **已等待秒数**，并提示去看哪两个终端。

## 运行后（如何确认成功）

1. 终端出现：`[local_img2img] 加载完成，可在 mps 上推理。`（或 `cuda` / `cpu`）
2. 浏览器图生图一次后，生图服务日志有：`[img2img] 本地 Diffusers 成功，PNG 字节数: ...`
3. 若本地未启动，Node 生图服务日志为连接失败，`/api/image/refine` 返回 **503**（无 Hugging Face 云端回退）。

## 单独启动本地图生图（不调 thesis 全家桶时）

```bash
npm run local-img2img
```

## 用 curl 测 8010 时「怎么知道还在生图」？

`curl` 会**一直等到整张 PNG 返回**才结束，所以**发请求的这一个窗口**在几分钟内可能**没有任何新输出**，容易误以为卡死。

- **以运行 `npm run local-img2img` 的终端为准**：应陆续出现 `[local_img2img] 收到请求`、`去噪 [###---] step …` 等（见上文「进度日志」）。
- 项目里提供了带**等待秒数**的脚本（在**第二个终端**跑，避免干等无反馈）：

```bash
bash scripts/test-8010-img2img.sh
# 首次特别慢可：CURL_MAX_TIME=900 bash scripts/test-8010-img2img.sh
# 本终端打印间隔（秒）：TEST_8010_TICK_SEC=5 bash scripts/test-8010-img2img.sh
```

**若你坚持用一条 `curl` 手测（与脚本 POST 体相同）**：在 `curl` 结束之前，**这个终端不会有任何输出**，这是 HTTP 特性，不是卡死。  
要看「等待了多久」只能：① 另开一个终端跑 `local-img2img` 看服务端日志；② 或像脚本那样**再起一个后台进程**定时 `echo`；③ 或直接用上面的脚本。

**手搓 curl + 本终端每 10s 一行日志（二合一复制）**：

```bash
( while sleep 10; do echo "[wait] $(date +%H:%M:%S) 仍在等 8010…"; done ) &
W=$!
# …此处粘贴你的 B64= 与 curl -m 420 …
kill $W 2>/dev/null
```

---

**答辩话术**：对话意图仍由 **Ollama 本地 LLM** 完成；**文生图**可走 HF 免费额度；**图生图**在演示机上用 **Diffusers + MPS** 本地推理，避免依赖不稳定云端 img2img。
