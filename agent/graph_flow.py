"""
基于 LangGraph 的创意设计协作工作流（开题：意图识别 → 需求分析 → 流程规划 / 主动追问）
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# 多轮对话压缩阈值：超过后摘要早期轮次，保留最近对话
COMPRESS_AFTER = int(os.getenv("AGENT_COMPRESS_AFTER", "10"))
KEEP_RECENT = int(os.getenv("AGENT_KEEP_RECENT", "4"))


class AgentState(TypedDict, total=False):
    messages: list[dict[str, str]]  # {role, content}
    has_canvas_image: bool
    # 运行时填充
    trace: list[str]
    summary_prefix: str
    intent: str
    need_clarify: bool
    missing_slots: list[str]
    clarify_question: str
    assistant_message: str
    action: Literal["none", "generate", "refine", "chat"]
    sd_prompt: str


def _msgs_to_lc(messages: list[dict[str, str]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        r, c = m.get("role", "user"), m.get("content", "")
        if r == "system":
            out.append(SystemMessage(content=c))
        elif r == "assistant":
            out.append(AIMessage(content=c))
        else:
            out.append(HumanMessage(content=c))
    return out


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no json in llm output")
    return json.loads(m.group())


def _latest_user_text(state: AgentState) -> str:
    for m in reversed(state.get("messages") or []):
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return str(m["content"]).strip()[:800]
    return ""


def _wants_new_image(text: str) -> bool:
    """用户是否在要「新画一张」—— 避免被误判成 general_chat"""
    t = text.lower()
    keys = (
        "生成",
        "画一张",
        "画一个",
        "来张",
        "来一幅",
        "做一张",
        "做个",
        "海报",
        "插画",
        "概念图",
        "帮我画",
        "给我画",
        "出一张",
        "文生图",
    )
    return any(k in text for k in keys) or "draw" in t or "generate an image" in t


def _sd_prompt_sanitize_for_user(last_user: str, sd_prompt: str) -> str:
    """减弱历史污染：本轮明确要蓝色时，去掉明显矛盾的绿色/改色碎片（勿全局替换英文 green，避免茎叶描述被误伤）"""
    lu = last_user
    out = sd_prompt
    if "蓝" in lu or "blue" in lu.lower():
        out = re.sub(r"[，,]\s*绿色[^，,]*", "", out)
        out = re.sub(r"[，,]\s*原图颜色改变[^，,]*", "", out)
    out = re.sub(r"\s*,\s*,+", ", ", out)
    return out.strip(" ,")


def _fallback_sd_prompt_refine(last_user: str) -> str:
    """iterative_refine 时 LLM 给了 none 或 sd_prompt 被清空时的英文兜底"""
    lu = last_user
    parts: list[str] = []
    if "背景" in lu or "底" in lu or "background" in lu.lower():
        if "绿" in lu or "green" in lu.lower():
            parts.append("change background to green, green backdrop, keep main subject unchanged")
        elif "蓝" in lu or "blue" in lu.lower():
            parts.append("change background to blue, blue backdrop, keep main subject unchanged")
        elif "红" in lu or "red" in lu.lower():
            parts.append("change background to red, red backdrop, keep main subject unchanged")
        elif "黑" in lu or "dark" in lu.lower():
            parts.append("dark background, keep main subject")
        elif "白" in lu or "white" in lu.lower():
            parts.append("white or light background, keep main subject")
    if "调" in lu or "色" in lu or "更亮" in lu or "更暗" in lu or "对比" in lu:
        parts.append("adjust colors and lighting as described, preserve composition")
    if not parts:
        parts.append(f"edit image as user requested: {lu[:160]}")
    parts.append("high quality, coherent, seamless")
    return ", ".join(parts)


def _fallback_sd_prompt_creative(last_user: str) -> str:
    """JSON 解析失败时的兜底：必须遵守颜色/主体，禁止写死红番茄"""
    lu = last_user
    parts: list[str] = []
    if "番茄" in lu or "tomato" in lu.lower():
        if "蓝" in lu or "blue" in lu.lower():
            parts.append(
                "single blue tomato, cyan and sapphire colored tomato fruit, surreal botanical, "
                "clearly blue skin not red, studio food photography, soft light, 8k, sharp focus"
            )
        elif "绿" in lu:
            parts.append(
                "green tomato, unripe green tomato on vine, botanical, natural light, 8k, sharp focus"
            )
        else:
            parts.append(
                "ripe red tomato on vine, food photography, natural colors, 8k, sharp focus"
            )
    else:
        parts.append(f"single main subject, highly detailed, centered composition: {lu}")
        parts.append("professional illustration or product render, 8k, sharp focus, clean background")
    return ", ".join(parts)


def _build_ollama() -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise RuntimeError("本机 Ollama 模式请安装：pip install langchain-ollama") from e
    om = os.getenv("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2"
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    return ChatOllama(model=om, base_url=base, temperature=0.3)


def _build_gemini(gemini_key: str) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise RuntimeError("使用 Gemini 请安装：pip install langchain-google-genai") from e
    g_model = os.getenv("GEMINI_AGENT_MODEL", "gemini-2.0-flash")
    return ChatGoogleGenerativeAI(
        model=g_model,
        google_api_key=gemini_key,
        temperature=0.3,
    )


def _build_openai_compat() -> ChatOpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DOUBAO_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://ark.cn-beijing.volces.com/api/v3"
    )
    model = os.getenv("LLM_MODEL") or os.getenv("DOUBAO_MODEL") or os.getenv("OPENAI_MODEL")
    if not api_key:
        raise RuntimeError("OpenAI 兼容接口缺少 LLM_API_KEY / DOUBAO_API_KEY")
    if not model:
        raise RuntimeError("需配置 LLM_MODEL 或 DOUBAO_MODEL（如方舟 ep-xxx）")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        temperature=0.3,
    )


def build_llm() -> BaseChatModel:
    """
    文本推理（意图 / 规划 / 摘要）优先级：
    1) AGENT_LLM_PROVIDER=ollama → 本机 Ollama（零 API Key）
    2) AGENT_LLM_PROVIDER=gemini → Gemini
    3) AGENT_LLM_PROVIDER=openai|ark|doubao → 方舟/OpenAI 兼容
    4) 自动：已配方舟/OpenAI → 兼容接口；否则有 GEMINI_API_KEY → Gemini；
       否则若配置了 OLLAMA_MODEL → Ollama
    """
    force = os.getenv("AGENT_LLM_PROVIDER", "").strip().lower()
    gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DOUBAO_API_KEY") or os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("LLM_MODEL") or os.getenv("DOUBAO_MODEL") or os.getenv("OPENAI_MODEL")
    has_openai_stack = bool(api_key and openai_model)
    ollama_model_env = os.getenv("OLLAMA_MODEL", "").strip()

    if force == "ollama":
        return _build_ollama()

    if force == "gemini":
        if not gemini_key:
            raise RuntimeError("AGENT_LLM_PROVIDER=gemini 但未配置 GEMINI_API_KEY")
        return _build_gemini(gemini_key)

    if force in ("openai", "ark", "doubao"):
        return _build_openai_compat()

    if has_openai_stack:
        return _build_openai_compat()

    if gemini_key:
        return _build_gemini(gemini_key)

    if ollama_model_env:
        return _build_ollama()

    raise RuntimeError(
        "请配置其一：AGENT_LLM_PROVIDER=ollama（本机 Ollama）+ 安装模型；"
        "或 GEMINI_API_KEY；或 LLM_API_KEY+LLM_MODEL / DOUBAO_API_KEY+DOUBAO_MODEL"
    )


def node_compress(state: AgentState) -> AgentState:
    trace = list(state.get("trace") or [])
    messages = list(state["messages"])
    if len(messages) <= COMPRESS_AFTER:
        trace.append("compress:skip")
        return {"trace": trace}

    # 摘要除最近 KEEP_RECENT 条以外的用户/助手内容
    head = messages[:-KEEP_RECENT]
    recent = messages[-KEEP_RECENT:]
    llm = build_llm()
    flat = "\n".join(f"{m['role']}: {m['content']}" for m in head)
    prompt = (
        "请将以下多轮对话压缩为一段中文摘要，保留用户的核心需求、风格、用途与已确认的信息，"
        "省略寒暄。**若后面轮次推翻了前面的修改意见（例如先改绿再要蓝），以最新一轮为准。**"
        "只输出摘要正文，不要标题。\n\n"
        + flat
    )
    summary = llm.invoke([HumanMessage(content=prompt)]).content
    if not isinstance(summary, str):
        summary = str(summary)
    summary_prefix = f"[历史对话摘要]\n{summary.strip()}\n\n"
    new_messages = [{"role": "system", "content": summary_prefix}] + recent
    trace.append("compress:applied")
    return {"messages": new_messages, "trace": trace, "summary_prefix": summary_prefix}


def node_intent(state: AgentState) -> AgentState:
    trace = list(state.get("trace") or [])
    llm = build_llm()
    sys = """你是创意设计助手的意图与需求分析模块。根据完整对话判断：
1) intent 只能是: creative_new（用户要新做海报/插画/概念图、或「生成/画一张/做一张」某画面）, iterative_refine（用户要改**当前已有**图：调色、换背景、修细节等）, general_chat（纯闲聊、问能力、与出图无关）
2) 用户说「生成一个蓝色番茄」「画一张赛博朋克城市」「来张海报」等——**一律 creative_new**，不是 general_chat。
3) 若 creative_new 但缺少「用途/风格/主体内容」中任意一项，need_clarify=true，并列出 missing_slots（若用户已明确主体+至少一个风格或用途，则不要过度追问）。
4) iterative_refine 需要用户已有成图；若 has_canvas_image 为 false，need_clarify=true，missing_slots 含「请先生成或上传一张图」

必须只输出一个 JSON 对象，不要 markdown：
{"intent":"...","need_clarify":true/false,"missing_slots":["..."],"reason":"..."}"""
    msgs = [SystemMessage(content=sys)] + _msgs_to_lc(state["messages"])
    raw = llm.invoke(msgs).content
    if not isinstance(raw, str):
        raw = str(raw)
    try:
        data = _extract_json(raw)
    except Exception:
        data = {
            "intent": "general_chat",
            "need_clarify": False,
            "missing_slots": [],
            "reason": "parse_fail",
        }
    intent = str(data.get("intent", "general_chat"))
    need_clarify = bool(data.get("need_clarify", False))
    missing = data.get("missing_slots") or []
    if not isinstance(missing, list):
        missing = []
    has_img = bool(state.get("has_canvas_image"))
    if intent == "iterative_refine" and not has_img:
        need_clarify = True
        if "请先生成或上传一张图" not in missing:
            missing = list(missing) + ["请先生成或上传一张图"]
    # 小模型常把「生成xxx」判成 general_chat，这里强制纠正
    last_u = _latest_user_text(state)
    if intent == "general_chat" and _wants_new_image(last_u):
        intent = "creative_new"
        need_clarify = False
        missing = []
    trace.append(f"intent:{intent}")
    return {
        "intent": intent,
        "need_clarify": need_clarify,
        "missing_slots": missing,
        "trace": trace,
    }


def node_clarify(state: AgentState) -> AgentState:
    trace = list(state.get("trace") or [])
    llm = build_llm()
    slots = state.get("missing_slots") or []
    sys = (
        "你是友好的创意设计助手。用户的信息不足。请根据 missing_slots 用 1～2 句中文主动追问，"
        "语气专业亲切，一次只问最关键的点。\n"
        f"missing_slots: {json.dumps(slots, ensure_ascii=False)}"
    )
    msgs = [SystemMessage(content=sys)] + _msgs_to_lc(state["messages"])
    q = llm.invoke(msgs).content
    if not isinstance(q, str):
        q = str(q)
    trace.append("clarify")
    return {
        "clarify_question": q.strip(),
        "assistant_message": q.strip(),
        "action": "none",
        "sd_prompt": "",
        "trace": trace,
    }


def node_plan(state: AgentState) -> AgentState:
    trace = list(state.get("trace") or [])
    llm = build_llm()
    intent = state.get("intent", "general_chat")
    has_img = bool(state.get("has_canvas_image"))
    last_user = _latest_user_text(state)
    sys = f"""你是 Stable Diffusion / SDXL 提示词工程师。

**本轮最高优先级**——用户刚刚这句话（可能含颜色、数量、风格）：
「{last_user}」

当前 intent={intent}，has_canvas_image={has_img}。

**sd_prompt 硬性规则**：
1) **整段只用英文**，逗号分隔标签；**禁止**在 sd_prompt 里写中文。
2) **必须**把本轮要画的主体译成英文关键词（如 tomato / blue tomato / cyberpunk city）；**颜色必须与本轮一致**（要蓝色就写 blue / cyan，不要默认红色）。
3) creative_new 时：**禁止**把历史对话里旧的「改绿色、原图颜色改变」等迭代指令抄进 sd_prompt，除非**本轮用户原话**里再次出现这些词。
4) 禁止只输出空洞的 "detailed illustration, 8k" 而没有具体主体。

输出 JSON（不要 markdown，不要代码块）：
{{
  "assistant_message": "给用户的中文回复，1～2 句，说明正如何生成；不要留空",
  "sd_prompt": "仅英文关键词与短语，逗号分隔",
  "action": "generate" | "refine" | "chat" | "none"
}}

规则：
- creative_new -> action 必须为 generate，sd_prompt 写完整画面英文描述
- iterative_refine 且 has_canvas_image=true -> **action 必须是 refine**，禁止 none、禁止 chat；sd_prompt 写要改的英文（如 green background）
- general_chat -> action=chat，sd_prompt 为空字符串
"""
    msgs = [SystemMessage(content=sys)] + _msgs_to_lc(state["messages"])
    raw = llm.invoke(msgs).content
    if not isinstance(raw, str):
        raw = str(raw)
    try:
        data = _extract_json(raw)
    except Exception:
        data = {
            "assistant_message": "好的，已根据你这句话整理英文生图提示并准备生成。",
            "sd_prompt": (
                _fallback_sd_prompt_creative(last_user)
                if intent == "creative_new"
                else _fallback_sd_prompt_refine(last_user)
            ),
            "action": (
                "generate"
                if intent == "creative_new"
                else ("refine" if intent == "iterative_refine" and has_img else "none")
            ),
        }
    action = str(data.get("action", "none"))
    if action not in ("none", "generate", "refine", "chat"):
        action = "none"
    # intent 与 action 对齐：新创作不应变成纯 chat
    if intent == "creative_new" and action == "chat":
        action = "generate"
    # 迭代 + 已有画布：必须出 refine，否则前端 action=none 不会调 /api/image/refine
    if intent == "iterative_refine" and has_img and action in ("generate", "none", "chat"):
        action = "refine"
    sd_prompt = str(data.get("sd_prompt", "")).strip()
    # 去掉模型误写入的中文，避免 SD / HF 指令混乱
    if action in ("generate", "refine"):
        sd_prompt = re.sub(r"[\u4e00-\u9fff]+", " ", sd_prompt)
        sd_prompt = re.sub(r"\s+", " ", sd_prompt).strip(" ,.")
    sd_prompt = _sd_prompt_sanitize_for_user(last_user, sd_prompt)
    # 解析成功但小模型仍可能泛泛而谈：用英文兜底补强（不再拼中文进 sd_prompt）
    generic = len(sd_prompt) < 28 or sd_prompt.lower() in (
        "high quality detailed artwork, professional composition",
        "high quality, professional composition",
    )
    if last_user and action in ("generate", "refine") and generic:
        fb = (
            _fallback_sd_prompt_creative(last_user)
            if action == "generate"
            else _fallback_sd_prompt_refine(last_user)
        )
        sd_prompt = f"{fb}, {sd_prompt}".strip(", ")
    elif last_user and action in ("generate", "refine"):
        # 确保颜色/主体线索在英文里（简单关键词注入）
        boost = []
        if "蓝" in last_user or "blue" in last_user.lower():
            boost.append("blue color theme as requested")
        if "红" in last_user or "red" in last_user.lower():
            boost.append("red color theme as requested")
        if "绿" in last_user:
            boost.append("green color theme as requested")
        if boost and not any(b.split()[0] in sd_prompt.lower() for b in boost):
            sd_prompt = f"{sd_prompt}, {', '.join(boost)}".strip(", ")

    assistant_message = str(data.get("assistant_message", "")).strip()
    # 去掉小模型泄漏的模板前缀
    assistant_message = re.sub(
        r"^给用户的中文回复[：:]\s*", "", assistant_message, flags=re.IGNORECASE
    ).strip()
    if intent == "iterative_refine" and has_img and action == "refine" and not sd_prompt.strip():
        sd_prompt = _fallback_sd_prompt_refine(last_user)
    if not assistant_message:
        if action == "generate":
            assistant_message = f"好的，正在根据你的描述生成画面：「{last_user[:100]}」。"
        elif action == "refine":
            assistant_message = "正在根据你的修改意见对当前画面做图生图调整，请稍候。"
        elif action == "chat":
            assistant_message = "我在这里，可以继续描述你的设计需求，或上传参考图后再说想怎么改。"
    elif action == "refine" and intent == "iterative_refine" and has_img:
        # 曾出现「说在画但 action=none」已纠正为 refine 时，统一成可信文案
        if "等待" in assistant_message and "分钟" in assistant_message and len(assistant_message) > 80:
            assistant_message = "正在按你的描述调整当前图片（图生图），请稍候片刻。"

    trace.append(f"plan:{action}")
    return {
        "assistant_message": assistant_message,
        "sd_prompt": sd_prompt,
        "action": action,
        "trace": trace,
    }


def route_after_intent(state: AgentState) -> Literal["clarify", "plan"]:
    return "clarify" if state.get("need_clarify") else "plan"


def build_app_graph():
    g = StateGraph(AgentState)
    g.add_node("compress", node_compress)
    g.add_node("intent", node_intent)
    g.add_node("clarify", node_clarify)
    g.add_node("plan", node_plan)

    g.add_edge(START, "compress")
    g.add_edge("compress", "intent")
    g.add_conditional_edges(
        "intent",
        route_after_intent,
        {"clarify": "clarify", "plan": "plan"},
    )
    g.add_edge("clarify", END)
    g.add_edge("plan", END)
    return g.compile()
