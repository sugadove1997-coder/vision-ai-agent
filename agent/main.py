"""
FastAPI 入口：多模态创意设计协作 Agent（LangGraph）
"""
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录 .env
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph_flow import build_app_graph

app = FastAPI(title="创意设计协作 Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_app_graph()
    return _graph


class TurnRequest(BaseModel):
    messages: list[dict[str, str]] = Field(default_factory=list)
    has_canvas_image: bool = False


class TurnResponse(BaseModel):
    assistant_message: str
    action: str
    sd_prompt: str
    trace: list[str]
    compressed: bool = False


@app.get("/health")
def health():
    return {"ok": True, "service": "langgraph-creative-agent"}


@app.post("/invoke", response_model=TurnResponse)
def invoke_turn(body: TurnRequest):
    if not body.messages:
        return TurnResponse(
            assistant_message="请先描述你的设计需求（例如：一张咖啡品牌海报，插画风格）。",
            action="none",
            sd_prompt="",
            trace=["empty_messages"],
        )

    g = get_graph()
    initial: dict = {
        "messages": body.messages,
        "has_canvas_image": body.has_canvas_image,
        "trace": [],
    }
    try:
        out = g.invoke(initial)
    except Exception as e:
        traceback.print_exc()
        om = os.getenv("OLLAMA_MODEL", "llama3.2")
        hint = (
            f"常见原因：1) Ollama 未运行或未拉取模型，终端执行 `ollama serve`（或打开 Ollama 应用）后 `ollama pull {om}`；"
            f"2) 本机 11434 端口被挡；3) 若不用 Ollama，请在 .env 改 AGENT_LLM_PROVIDER / 配置 GEMINI_API_KEY。"
        )
        return TurnResponse(
            assistant_message=f"**Agent 执行失败**\n\n`{type(e).__name__}: {e}`\n\n{hint}",
            action="none",
            sd_prompt="",
            trace=["error", type(e).__name__],
            compressed=False,
        )

    trace = list(out.get("trace") or [])
    compressed = any(t.startswith("compress:applied") for t in trace)
    return TurnResponse(
        assistant_message=out.get("assistant_message") or out.get("clarify_question") or "",
        action=out.get("action") or "none",
        sd_prompt=out.get("sd_prompt") or "",
        trace=trace,
        compressed=compressed,
    )
