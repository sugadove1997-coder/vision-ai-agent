/**
 * 使用官方 @huggingface/inference SDK 生图，避免 router 路径问题
 * 若需代理：在 .env 中设置 HF_PROXY 或 HTTPS_PROXY，如 HF_PROXY=http://127.0.0.1:7890
 *
 * 默认模型选型：**免费使用**向 —— 仅需 https://huggingface.co 免费账号 + Token（勾选 Inference），
 * 走 Hugging Face 官方 serverless 免费额度（具体限额以 Hub 当前政策为准），不依赖付费第三方推理商。
 */
import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const PORT = Number(process.env.PROXY_PORT) || 3001;
const HF_TOKEN = process.env.HUGGINGFACE_TOKEN;
const PROXY_URL = process.env.HF_PROXY || process.env.HTTPS_PROXY || process.env.HTTP_PROXY;

if (!HF_TOKEN) {
  console.error('请在 .env 中设置 HUGGINGFACE_TOKEN');
  process.exit(1);
}

// 配置代理：在加载 SDK 前替换全局 fetch，使请求走代理
if (PROXY_URL) {
  const { ProxyAgent, fetch: undiciFetch } = await import('undici');
  const agent = new ProxyAgent(PROXY_URL);
  (globalThis as any).fetch = (input: any, init?: any) =>
    undiciFetch(input, { ...init, dispatcher: agent });
  try {
    console.log('[image] 已启用代理:', new URL(PROXY_URL).origin);
  } catch {
    console.log('[image] 已启用代理');
  }
}

const { InferenceClient } = await import('@huggingface/inference');
const hf = new InferenceClient(HF_TOKEN);

async function hfImageToBuffer(image: Blob | string): Promise<Buffer> {
  if (typeof image === 'string') return Buffer.from(image, 'base64');
  return Buffer.from(await image.arrayBuffer());
}

/** 使用 undici 直连本地 8010，避免 HF_PROXY 把 127.0.0.1 误走代理 */
async function tryLocalDiffusersImg2Img(
  prompt: string,
  imageBuf: Buffer,
  strength: number,
  width: number,
  height: number,
  signal?: AbortSignal,
  diag?: { httpStatus?: number; body?: string; catchMessage?: string }
): Promise<Buffer | null> {
  const base = (process.env.LOCAL_IMG2IMG_URL || 'http://127.0.0.1:8010').replace(/\/$/, '');
  const url = `${base}/img2img`;
  try {
    const waitLogMs = Math.max(
      5000,
      Math.min(60_000, Number(process.env.IMG2IMG_WAIT_LOG_MS) || 10_000)
    );
    console.log('[img2img] 仅本地 Diffusers →', url, `（等待期间每 ${waitLogMs / 1000}s 打印一次进度）`);
    const { fetch: undiciDirect } = await import('undici');
    const wait0 = Date.now();
    const tick = setInterval(() => {
      const s = Math.round((Date.now() - wait0) / 1000);
      console.log(
        `[img2img] 仍在等待 8010 返回 PNG… 已等待 ${s}s | 请看运行 local-img2img 的终端：[去噪 step] / Hub 下载进度`
      );
    }, waitLogMs);
    let res: Awaited<ReturnType<typeof undiciDirect>>;
    try {
      res = await undiciDirect(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          image_base64: imageBuf.toString('base64'),
          strength,
          width,
          height,
        }),
        signal,
      });
    } finally {
      clearInterval(tick);
    }
    if (!res.ok) {
      const t = await res.text();
      if (diag) {
        diag.httpStatus = res.status;
        diag.body = t;
      }
      console.warn('[img2img] 本地服务 HTTP', res.status, t.slice(0, 240));
      return null;
    }
    const ab = await res.arrayBuffer();
    const out = Buffer.from(ab);
    console.log('[img2img] 本地 Diffusers 成功，PNG 字节数:', out.length);
    return out;
  } catch (e: any) {
    if (diag) diag.catchMessage = e?.message || String(e);
    console.warn('[img2img] 本地图生图失败（无云端回退）:', e?.message || e);
    return null;
  }
}

const app = express();
app.use(express.json({ limit: '20mb' }));

app.post('/api/image', async (req, res) => {
  const { prompt, width = 1024, height = 576 } = req.body || {};
  if (!prompt || typeof prompt !== 'string') {
    res.status(400).json({ error: '缺少 prompt' });
    return;
  }
  const w = Number(width) || 1024;
  const h = Number(height) || 576;
  // 文生图：优先 HF 上常见「免费推理」模型；固定 provider 避免 auto 跳到收费/第三方
  const models = [
    'black-forest-labs/FLUX.1-schnell',
    'stabilityai/stable-diffusion-xl-base-1.0',
    'runwayml/stable-diffusion-v1-5',
  ];
  let lastError: any = null;
  for (const model of models) {
    try {
      const blob = await hf.textToImage({
        provider: 'hf-inference',
        model,
        inputs: prompt,
        parameters: { width: w, height: h },
      });
      res.setHeader('Content-Type', 'image/png');
      res.send(await hfImageToBuffer(blob));
      return;
    } catch (e: any) {
      lastError = e;
      console.warn(`[image] ${model} 失败:`, e?.message || e);
    }
  }
  const e = lastError;
  if (e) {
    const msg = e?.message || e?.toString?.() || '生图失败';
    let detail: string | null = null;
    if (e?.response && typeof e.response?.text === 'function') {
      try {
        detail = await e.response.text();
      } catch (_) {}
    }
    if (!detail && e?.cause?.message) detail = e.cause.message;
    if (!detail && e?.cause?.code) detail = `[${e.cause.code}] ${e.cause.message || ''}`.trim();
    console.error('[image]', msg, detail || '');
    const isNetwork = /fetch failed|ECONNREFUSED|ENOTFOUND|ETIMEDOUT|network/i.test(msg + (detail || ''));
    res.status(500).json({
      error: msg,
      detail,
      hint: isNetwork
        ? '无法连接 Hugging Face。请在 .env 中设置代理后重启生图服务，例如：HF_PROXY=http://127.0.0.1:7890（改为你的代理地址）'
        : '请确认 Token 在 https://huggingface.co/settings/tokens 已勾选 Inference；部分模型需在模型页先点「Agree」接受协议。免费额度用尽时会报错，可次日再试或换模型。',
    });
  } else {
    res.status(500).json({ error: '生图失败', detail: '无可用模型' });
  }
});

/** Stable Diffusion 图生图（迭代优化：基于用户反馈在保留构图的前提下调整） */
app.post('/api/image/refine', async (req, res) => {
  const {
    prompt,
    imageBase64,
    strength = 0.65,
    width = 1024,
    height = 576,
  } = req.body || {};
  if (!prompt || typeof prompt !== 'string') {
    res.status(400).json({ error: '缺少 prompt' });
    return;
  }
  if (!imageBase64 || typeof imageBase64 !== 'string') {
    res.status(400).json({ error: '缺少 imageBase64' });
    return;
  }
  const b64 = imageBase64.replace(/^data:image\/\w+;base64,/, '');
  let buf: Buffer;
  try {
    buf = Buffer.from(b64, 'base64');
  } catch {
    res.status(400).json({ error: 'imageBase64 无效' });
    return;
  }
  const s = Math.min(0.95, Math.max(0.1, Number(strength) || 0.65));
  const w = Number(width) || 1024;
  const h = Number(height) || 576;
  const localImg2imgTimeoutMs = Math.min(
    1_200_000,
    Math.max(120_000, Number(process.env.LOCAL_IMG2IMG_TIMEOUT_MS) || 600_000)
  );
  const localSignal =
    typeof AbortSignal !== 'undefined' && 'timeout' in AbortSignal
      ? AbortSignal.timeout(localImg2imgTimeoutMs)
      : undefined;

  const localDiag: { httpStatus?: number; body?: string; catchMessage?: string } = {};
  const localOut = await tryLocalDiffusersImg2Img(prompt, buf, s, w, h, localSignal, localDiag);
  if (localOut) {
    res.setHeader('Content-Type', 'image/png');
    res.send(localOut);
    return;
  }

  let msg = '本地图生图失败';
  let detail: string | undefined = localDiag.body || localDiag.catchMessage;
  if (localDiag.body) {
    try {
      const j = JSON.parse(localDiag.body) as { detail?: unknown };
      if (typeof j.detail === 'string') msg = j.detail;
    } catch {
      /* keep msg */
    }
  } else if (localDiag.catchMessage) {
    msg = localDiag.catchMessage;
  } else {
    msg = `无法连接本地图生图服务 ${process.env.LOCAL_IMG2IMG_URL || 'http://127.0.0.1:8010'}（ECONNREFUSED 等）`;
  }

  const errBlob = `${msg} ${detail || ''}`;
  const isTimeout =
    /aborted due to timeout|TimeoutError|ETIMEDOUT|timed out/i.test(errBlob) ||
    localDiag.catchMessage?.includes('timeout');
  const isNetwork = /fetch failed|ECONNRESET|ETIMEDOUT|ECONNREFUSED|ENOTFOUND|network|socket|SSL|TLS|UND_ERR|Proxy/i.test(
    errBlob
  );
  const localHubPermDenied =
    !!localDiag.body &&
    /Operation not permitted|Errno 1/i.test(localDiag.body) &&
    /huggingface|\.locks/i.test(localDiag.body);

  console.error('[img2img]', msg, detail || '');
  const status =
    localDiag.httpStatus && localDiag.httpStatus >= 400 && localDiag.httpStatus < 600
      ? localDiag.httpStatus
      : 503;
  res.status(status).json({
    error: msg,
    detail: detail?.slice(0, 2000),
    hint: localHubPermDenied
      ? '无法写入 Hugging Face 缓存（Operation not permitted）。请重启 local-img2img（默认缓存目录 agent/.hf_hub_cache），或在 .env 设置可写的 HF_HOME。'
      : isTimeout
        ? `图生图超时。首次下载/加载模型可能很慢：在 .env 增大 LOCAL_IMG2IMG_TIMEOUT_MS（当前上限 ${localImg2imgTimeoutMs}ms）后重启 image-server。`
        : isNetwork
          ? '连不上本机 8010。请先 `pip install -r agent/requirements-img2img.txt`，再 `npm run local-img2img`（或与 dev:thesis 一并启动）。'
          : '图生图仅走本地 Diffusers。请确认 `npm run local-img2img` 已运行且无报错；详见 README / docs/LOCAL_IMG2IMG.md。',
  });
});

// 豆包（火山方舟）Chat Completions：https://ark.cn-beijing.volces.com/api/v3/chat/completions
const DOUBAO_API_KEY = process.env.DOUBAO_API_KEY?.trim();
const DOUBAO_MODEL = process.env.DOUBAO_MODEL?.trim(); // 必填：推理接入点 ID（ep- 开头），在控制台创建接入点后获得
const VOLC_ARK_URL = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions';
const VOLC_ARK_ENDPOINT_CONSOLE = 'https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint';

// 免费文案生成：优先豆包，否则 Hugging Face（先 chatCompletion 再 textGeneration）
const CHAT_MODELS = ['microsoft/Phi-3-mini-4k-instruct', 'HuggingFaceH4/zephyr-7b-beta', 'google/gemma-2-2b-it'];
const TEXT_MODELS = ['google/flan-t5-large', 'google/flan-t5-base'];
app.post('/api/chat', async (req, res) => {
  const { prompt } = req.body || {};
  if (!prompt || typeof prompt !== 'string') {
    res.status(400).json({ error: '缺少 prompt' });
    return;
  }

  // 1) 若配置了豆包，仅用豆包（不再走 Hugging Face）
  if (DOUBAO_API_KEY) {
    if (!DOUBAO_MODEL) {
      res.status(400).json({
        error: '未配置 DOUBAO_MODEL',
        hint: `请在 .env 中设置 DOUBAO_MODEL 为你的推理接入点 ID（ep- 开头）。在火山方舟控制台创建接入点后即可获得：${VOLC_ARK_ENDPOINT_CONSOLE}`,
      });
      return;
    }
    try {
      const response = await fetch(VOLC_ARK_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${DOUBAO_API_KEY}`,
        },
        body: JSON.stringify({
          model: DOUBAO_MODEL,
          messages: [{ role: 'user', content: prompt }],
          max_tokens: 1024,
        }),
      });
      const data = await response.json().catch(() => ({}));
      const text = data?.choices?.[0]?.message?.content ?? '';
      if (text.trim()) {
        res.json({ text: text.trim() });
        return;
      }
      const errMsg = data?.error?.message || data?.message || data?.error?.code || `HTTP ${response.status}`;
      console.error('[chat] 豆包:', errMsg, data?.error || '');
      res.status(response.ok ? 500 : response.status).json({
        error: errMsg,
        detail: data?.error ? JSON.stringify(data.error) : undefined,
        hint: `若报「模型不存在」或「无权限」，请用推理接入点 ID（ep- 开头）作为 DOUBAO_MODEL，在控制台创建：${VOLC_ARK_ENDPOINT_CONSOLE}`,
      });
      return;
    } catch (e: any) {
      console.error('[chat] 豆包请求异常:', e?.message || e);
      res.status(500).json({
        error: e?.message || '豆包请求失败',
        hint: '请检查网络或代理，确认 DOUBAO_API_KEY、DOUBAO_MODEL 正确。',
      });
      return;
    }
  }

  let lastError: any = null;
  for (const model of CHAT_MODELS) {
    try {
      const out = await hf.chatCompletion({
        model,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 1024,
      });
      const text = out.choices?.[0]?.message?.content ?? '';
      if (text.trim()) {
        res.json({ text: text.trim() });
        return;
      }
    } catch (e: any) {
      lastError = e;
      console.warn(`[chat] ${model}:`, e?.message || e);
    }
  }

  for (const model of TEXT_MODELS) {
    try {
      const out = await hf.textGeneration({
        model,
        inputs: prompt,
        parameters: { max_new_tokens: 1024 },
      });
      const text = out.generated_text ?? '';
      if (text.trim()) {
        res.json({ text: text.trim() });
        return;
      }
    } catch (e: any) {
      lastError = e;
      console.warn(`[chat] ${model}:`, e?.message || e);
    }
  }

  const e = lastError;
  const msg = e?.message || e?.toString?.() || '文案生成失败';
  let detail: string | null = null;
  if (e?.cause?.message) detail = e.cause.message;
  if (e?.response && typeof e.response?.text === 'function') {
    try {
      detail = await e.response.text();
    } catch (_) {}
  }
  console.error('[chat]', msg, detail || '');
  res.status(500).json({
    error: msg,
    detail,
    hint: DOUBAO_API_KEY
      ? '豆包调用失败，请检查 DOUBAO_API_KEY、DOUBAO_MODEL 及火山方舟控制台接入点。'
      : '请确认代理已开启且 HUGGINGFACE_TOKEN 有效；或在 .env 中配置 DOUBAO_API_KEY 使用豆包。',
  });
});

app.listen(PORT, () => {
  console.log(`生图服务 http://localhost:${PORT}/api/image`);
  console.log(
    `[image] 图生图 /api/image/refine 仅本地 Diffusers → ${process.env.LOCAL_IMG2IMG_URL || 'http://127.0.0.1:8010'}`
  );
  console.log(`文案服务 http://localhost:${PORT}/api/chat` + (DOUBAO_API_KEY ? '（豆包已启用）' : ''));
});
