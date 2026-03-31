import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Image as ImageIcon,
  Sparkles,
  Download,
  Loader2,
  AlertCircle,
  Bot,
  User,
  GitBranch,
  ChevronDown,
  ChevronUp,
  Upload,
  X,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import Markdown from 'react-markdown';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { GoogleGenAI } from '@google/genai';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

type ChatRole = 'user' | 'assistant' | 'system';

interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
}

const PRESETS = [
  { label: '活动海报', text: '我想做一张活动海报，主题是春季新品发布，偏扁平插画风格。' },
  { label: '产品插画', text: '需要一张产品插画，主体是一杯咖啡，暖色调，适合电商详情页。' },
  { label: '场景概念', text: '游戏场景概念图：废土中的小型避难所，黄昏，电影级光影。' },
];

function stripDataUrlPrefix(b64OrDataUrl: string) {
  return b64OrDataUrl.replace(/^data:image\/\w+;base64,/, '');
}

/** 缩小底图，减轻 HF 图生图请求体与代理超时（长截图尤其明显） */
function downscaleDataUrlForRefine(dataUrl: string, maxEdge = 896): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      try {
        const w = img.naturalWidth;
        const h = img.naturalHeight;
        const scale = Math.min(1, maxEdge / Math.max(w, h));
        if (scale >= 1) {
          resolve(dataUrl);
          return;
        }
        const tw = Math.max(1, Math.round(w * scale));
        const th = Math.max(1, Math.round(h * scale));
        const canvas = document.createElement('canvas');
        canvas.width = tw;
        canvas.height = th;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          resolve(dataUrl);
          return;
        }
        ctx.drawImage(img, 0, 0, tw, th);
        resolve(canvas.toDataURL('image/jpeg', 0.88));
      } catch (err) {
        reject(err);
      }
    };
    img.onerror = () => reject(new Error('无法加载图片用于压缩'));
    img.src = dataUrl;
  });
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        '你好，我是**基于多模态 AI Agent 的创意设计协作助手**。请用自然语言描述你的海报、插画或场景概念需求；我会进行意图识别与需求分析，必要时主动追问，并调用 **Hugging Face 免费推理（Flux / SDXL 等）** 生成或迭代优化画面。',
    },
  ]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  /** 文生图 vs 本地图生图（用于界面提示） */
  const [drawMode, setDrawMode] = useState<'generate' | 'refine' | null>(null);
  const [drawElapsedSec, setDrawElapsedSec] = useState(0);
  const drawStartRef = useRef<number>(0);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  /** 用户上传的参考图 / 待修改底图（无 SD 结果时也可用于图生图迭代） */
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastTrace, setLastTrace] = useState<string[]>([]);
  const [showTrace, setShowTrace] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /** 画布展示：优先生成结果，否则显示上传图 */
  const canvasImage = generatedImage ?? uploadedImage;
  /** Agent / 图生图：有生成图则迭代生成图，否则用上传图 */
  const refineBaseImage = generatedImage ?? uploadedImage;
  const hasCanvasForAgent = !!(generatedImage || uploadedImage);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isThinking]);

  useEffect(() => {
    if (!isDrawing) {
      setDrawElapsedSec(0);
      return;
    }
    drawStartRef.current = Date.now();
    const id = window.setInterval(() => {
      setDrawElapsedSec(Math.floor((Date.now() - drawStartRef.current) / 1000));
    }, 500);
    return () => window.clearInterval(id);
  }, [isDrawing]);

  const hfConfigured = !!process.env.HUGGINGFACE_TOKEN;

  const runTextToImage = async (prompt: string) => {
    if (hfConfigured) {
      const res = await fetch('/api/image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, width: 1024, height: 576 }),
      });
      if (!res.ok) {
        const t = await res.text();
        let msg = t || `HTTP ${res.status}`;
        try {
          const j = JSON.parse(t);
          if (j?.error) msg = j.detail ? `${j.error}: ${j.detail}` : j.error;
          if (j?.hint) msg += `\n\n${j.hint}`;
        } catch {
          /* ignore */
        }
        throw new Error(msg);
      }
      const blob = await res.blob();
      return await new Promise<string>((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result as string);
        r.onerror = reject;
        r.readAsDataURL(blob);
      });
    }
    const key = process.env.GEMINI_API_KEY?.trim();
    if (!key) throw new Error('请配置 HUGGINGFACE_TOKEN（推荐，免费 Hub 推理）或 GEMINI_API_KEY');
    const ai = new GoogleGenAI({ apiKey: key });
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: `Create a high-quality creative image: ${prompt}. Aspect 16:9.`,
      config: { imageConfig: { aspectRatio: '16:9' } },
    });
    for (const part of response.candidates?.[0]?.content?.parts || []) {
      if (part.inlineData?.data) {
        return `data:image/png;base64,${part.inlineData.data}`;
      }
    }
    throw new Error('未能生成图片');
  };

  const runImageRefine = async (prompt: string, imageDataUrl: string) => {
    const scaled = await downscaleDataUrlForRefine(imageDataUrl, 896);
    const res = await fetch('/api/image/refine', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        imageBase64: stripDataUrlPrefix(scaled),
        strength: 0.62,
        width: 1024,
        height: 576,
      }),
    });
    if (!res.ok) {
      const t = await res.text();
      try {
        const j = JSON.parse(t);
        let m = j.error || t;
        if (j.detail) m += `\n${j.detail}`;
        if (j.hint) m += `\n\n${j.hint}`;
        throw new Error(m);
      } catch (e: any) {
        if (e?.message && !e.message.startsWith('{')) throw e;
        throw new Error(t);
      }
    }
    const blob = await res.blob();
    return await new Promise<string>((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result as string);
      r.onerror = reject;
      r.readAsDataURL(blob);
    });
  };

  const sendMessage = async (text?: string) => {
    const raw = (text ?? input).trim();
    if (!raw || isThinking) return;
    setInput('');
    setError(null);
    const userMsg: ChatMessage = {
      id: Math.random().toString(36).slice(2),
      role: 'user',
      content: raw,
    };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setIsThinking(true);
    setLastTrace([]);

    try {
      const payload = {
        messages: nextMessages
          .filter((m) => m.role !== 'system')
          .map((m) => ({ role: m.role, content: m.content })),
        has_canvas_image: hasCanvasForAgent,
      };

      const agentRes = await fetch('/api/agent/invoke', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!agentRes.ok) {
        const errText = await agentRes.text();
        if (agentRes.status === 404 || agentRes.status === 502) {
          throw new Error(
            '无法连接 LangGraph 服务。请运行 npm run dev:thesis，或单独启动：cd agent && .venv/bin/python -m uvicorn main:app --port 8008'
          );
        }
        let msg = errText || `Agent HTTP ${agentRes.status}`;
        try {
          const j = JSON.parse(errText);
          if (j?.detail != null) {
            msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
          }
        } catch {
          /* 非 JSON（如 HTML 错误页） */
        }
        throw new Error(msg.slice(0, 800) + '（详见终端 [A] Agent 日志）');
      }

      const data = await agentRes.json();
      setLastTrace(data.trace || []);

      const assistantMsg: ChatMessage = {
        id: Math.random().toString(36).slice(2),
        role: 'assistant',
        content: data.assistant_message || '（无回复）',
      };
      setMessages((prev) => [...prev, assistantMsg]);

      const action = data.action as string;
      const sdPrompt = (data.sd_prompt || '').trim();
      if ((action === 'generate' || action === 'refine') && sdPrompt) {
        setIsDrawing(true);
        setDrawMode(action === 'refine' && refineBaseImage ? 'refine' : 'generate');
        try {
          if (action === 'refine' && refineBaseImage) {
            const url = await runImageRefine(sdPrompt, refineBaseImage);
            setGeneratedImage(url);
          } else {
            const url = await runTextToImage(sdPrompt);
            setGeneratedImage(url);
          }
        } catch (imgErr: any) {
          setError(imgErr?.message || '生图失败');
        } finally {
          setIsDrawing(false);
          setDrawMode(null);
        }
      }
    } catch (e: any) {
      console.error(e);
      setError(e?.message || '请求失败');
    } finally {
      setIsThinking(false);
    }
  };

  const downloadImage = () => {
    if (!canvasImage) return;
    const a = document.createElement('a');
    a.href = canvasImage;
    a.download = `creative-agent-${Date.now()}.png`;
    a.click();
  };

  const onPickImageFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file || !file.type.startsWith('image/')) {
      setError('请选择图片文件（PNG / JPEG / WebP 等）');
      return;
    }
    const maxMb = 12;
    if (file.size > maxMb * 1024 * 1024) {
      setError(`图片请小于 ${maxMb}MB`);
      return;
    }
    setError(null);
    const r = new FileReader();
    r.onload = () => {
      const dataUrl = r.result as string;
      setUploadedImage(dataUrl);
    };
    r.onerror = () => setError('读取图片失败');
    r.readAsDataURL(file);
  };

  return (
    <div className="min-h-screen bg-[#0a051a] text-white font-sans selection:bg-purple-500/30">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-900/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-900/20 blur-[120px] rounded-full" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8">
        <header className="text-center mb-8">
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-4"
          >
            <GitBranch className="w-4 h-4 text-cyan-400" />
            <span className="text-sm text-cyan-200/90">LangGraph 工作流 · 意图识别 / 需求分析 / SD 调度</span>
          </motion.div>
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight mb-2 bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
            多模态 AI Agent 创意设计协作助手
          </h1>
          <p className="text-sm text-gray-400 max-w-2xl mx-auto">
            闭环：文本需求 → 结构化意图与追问 → 初稿图像 → 多轮迭代（图生图）。对话过长时由 Agent 侧自动摘要压缩上下文。
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* 对话 */}
          <div className="lg:col-span-5 flex flex-col min-h-[480px]">
            <div className="flex-1 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl overflow-hidden flex flex-col">
              <div className="p-4 border-b border-white/10 flex items-center justify-between">
                <h2 className="text-sm font-semibold text-purple-200 flex items-center gap-2">
                  <Bot className="w-4 h-4" />
                  协作对话
                </h2>
                {!hfConfigured && (
                  <span className="text-[10px] text-amber-400/90">未检测到 HF Token，生图将尝试 Gemini</span>
                )}
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-4 max-h-[420px]">
                <AnimatePresence initial={false}>
                  {messages.map((m) => (
                    <motion.div
                      key={m.id}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={cn(
                        'flex gap-3',
                        m.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                      )}
                    >
                      <div
                        className={cn(
                          'shrink-0 w-8 h-8 rounded-xl flex items-center justify-center',
                          m.role === 'user' ? 'bg-indigo-600/40' : 'bg-purple-600/40'
                        )}
                      >
                        {m.role === 'user' ? (
                          <User className="w-4 h-4" />
                        ) : (
                          <Bot className="w-4 h-4" />
                        )}
                      </div>
                      <div
                        className={cn(
                          'rounded-2xl px-4 py-3 text-sm max-w-[85%] border',
                          m.role === 'user'
                            ? 'bg-indigo-500/15 border-indigo-500/20 text-gray-100'
                            : 'bg-white/[0.06] border-white/10 text-gray-200'
                        )}
                      >
                        {m.role === 'assistant' ? (
                          <div className="prose prose-invert prose-sm max-w-none prose-p:my-1 prose-headings:text-purple-200">
                            <Markdown>{m.content}</Markdown>
                          </div>
                        ) : (
                          <p className="whitespace-pre-wrap">{m.content}</p>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
                {(isThinking || isDrawing) && (
                  <div className="flex flex-col gap-1 text-xs text-gray-500 pl-11">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-purple-400 shrink-0" />
                      {isDrawing ? (
                        drawMode === 'refine' ? (
                          <span>
                            本地图生图（8010 Diffusers）进行中… 已等待{' '}
                            <strong className="text-cyan-300 tabular-nums">{drawElapsedSec}</strong>s
                          </span>
                        ) : (
                          <span>
                            文生图中… 已等待{' '}
                            <strong className="text-purple-300 tabular-nums">{drawElapsedSec}</strong>s
                          </span>
                        )
                      ) : (
                        <span>LangGraph 推理中…</span>
                      )}
                    </div>
                    {isDrawing && drawMode === 'refine' && (
                      <p className="text-[10px] text-gray-600 leading-relaxed max-w-md">
                        看进度：终端里跑 <code className="text-gray-500">local-img2img</code> 的窗口，或{' '}
                        <code className="text-gray-500">dev:thesis</code> 里带{' '}
                        <code className="text-gray-500">[I]</code> 的行。也可另开终端：{' '}
                        <code className="text-gray-500">tail -f agent/logs/local-img2img.log</code>
                        。image-server 终端每 ~10s 会打等待日志。
                      </p>
                    )}
                  </div>
                )}
                <div ref={bottomRef} />
              </div>

              <div className="p-3 border-t border-white/10 space-y-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={onPickImageFile}
                />
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    disabled={isThinking}
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center gap-1.5 text-[11px] px-2.5 py-1.5 rounded-xl bg-white/5 border border-white/15 hover:bg-white/10 text-gray-300"
                  >
                    <Upload className="w-3.5 h-3.5" />
                    上传参考图 / 待改底图
                  </button>
                  {uploadedImage && (
                    <div className="relative inline-flex items-center gap-1 rounded-lg border border-white/15 bg-black/20 p-0.5 pr-6">
                      <img
                        src={uploadedImage}
                        alt="上传预览"
                        className="h-10 w-14 object-cover rounded-md"
                      />
                      <button
                        type="button"
                        disabled={isThinking}
                        onClick={() => setUploadedImage(null)}
                        className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-red-500/90 border border-white/20 flex items-center justify-center hover:bg-red-500"
                        aria-label="移除上传图"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                  <span className="text-[10px] text-gray-500">
                    提出修改前可先上传图片；有画布时 Agent 会走图生图迭代
                  </span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {PRESETS.map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      disabled={isThinking}
                      onClick={() => sendMessage(p.text)}
                      className="text-[10px] px-2 py-1 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 text-gray-400"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
                <div className="flex gap-2">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    placeholder="描述需求；或上传图片后提出修改意见（如：把背景改成夜景）…"
                    rows={2}
                    className="flex-1 bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500/40 resize-none"
                  />
                  <button
                    type="button"
                    disabled={isThinking || !input.trim()}
                    onClick={() => sendMessage()}
                    className="self-end px-4 py-3 rounded-xl bg-gradient-to-br from-purple-600 to-indigo-600 disabled:opacity-40"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>

            {lastTrace.length > 0 && (
              <div className="mt-3 rounded-2xl bg-white/[0.03] border border-white/10 p-3">
                <button
                  type="button"
                  onClick={() => setShowTrace(!showTrace)}
                  className="flex items-center gap-2 text-xs text-gray-400 w-full"
                >
                  {showTrace ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  工作流追踪（论文演示）
                </button>
                {showTrace && (
                  <ul className="mt-2 text-[10px] font-mono text-cyan-200/80 space-y-1">
                    {lastTrace.map((t, i) => (
                      <li key={i}>• {t}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>

          {/* 画布 */}
          <div className="lg:col-span-7 space-y-4">
            <div className="relative aspect-video rounded-[2rem] overflow-hidden bg-white/5 border border-white/10 backdrop-blur-xl">
              {!canvasImage && !isDrawing && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600 px-6 text-center">
                  <ImageIcon className="w-14 h-14 mb-3 opacity-20" />
                  <p className="text-sm">生成结果将显示于此</p>
                  <p className="text-xs text-gray-600 mt-1">或先在左侧「上传参考图」，再描述要如何修改</p>
                </div>
              )}
              {isDrawing && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50 backdrop-blur-sm z-10 px-6 text-center">
                  <Loader2 className="w-10 h-10 animate-spin text-purple-400 mb-3" />
                  <p className="text-sm text-purple-200">
                    {drawMode === 'refine'
                      ? '本地图生图（8010）推理中'
                      : '文生图（HF / Gemini）推理中'}
                  </p>
                  <p className="text-lg font-mono text-cyan-300 mt-2 tabular-nums">{drawElapsedSec}s</p>
                  {drawMode === 'refine' && (
                    <p className="text-[10px] text-gray-500 mt-2 max-w-sm">
                      详情：local-img2img 终端或 tail -f agent/logs/local-img2img.log；image-server 约每 10s 一行
                    </p>
                  )}
                </div>
              )}
              {canvasImage && (
                <img
                  src={canvasImage}
                  alt={generatedImage ? '生成结果' : '上传参考'}
                  className="w-full h-full object-cover"
                />
              )}
            </div>
            {canvasImage && (
              <div className="flex gap-3 justify-end">
                <button
                  type="button"
                  onClick={downloadImage}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/10 border border-white/15 text-sm"
                >
                  <Download className="w-4 h-4" />
                  下载
                </button>
              </div>
            )}

            {error && (
              <div className="flex gap-2 p-4 rounded-2xl bg-red-500/10 border border-red-500/25 text-red-300 text-sm whitespace-pre-wrap">
                <AlertCircle className="w-5 h-5 shrink-0" />
                {error}
              </div>
            )}

            <div className="rounded-2xl bg-white/[0.04] border border-white/10 p-4 text-xs text-gray-500 space-y-2">
              <p className="flex items-center gap-2 text-gray-400 font-medium">
                <Sparkles className="w-4 h-4 text-purple-400" />
                开题对齐说明
              </p>
              <ul className="list-disc pl-4 space-y-1">
                <li>LangChain / LangGraph 图结构：压缩 → 意图 → 追问或规划</li>
                <li>Stable Diffusion：Hugging Face 免费推理（Flux / SDXL / SD1.5 文生图；图生图默认 SDXL Base）</li>
                <li>多轮修改：可上传参考图或先生成，再在画布上走 refine；全新主题走文生图</li>
              </ul>
            </div>
          </div>
        </div>

        <footer className="mt-12 pt-6 border-t border-white/5 text-center text-gray-600 text-xs">
          <p>毕业设计 · 基于多模态 AI Agent 的创意设计协作助手 · 天津工业大学</p>
        </footer>
      </div>
    </div>
  );
}
