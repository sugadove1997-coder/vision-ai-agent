"""
本地 Diffusers 图生图微服务（FastAPI）

**原因**：Hugging Face 免费线上 image-to-image 常 404/限流；本机 M 系列 GPU 可用 MPS 加速，
在 Apple Silicon + 大内存上跑 SD1.5 img2img 延迟可控、且不耗 HF 图生图额度。

启动（论文演示可与 agent 同开）：
  cd agent && .venv/bin/python -m uvicorn local_img2img_service:app --host 127.0.0.1 --port 8010

环境变量：
  LOCAL_IMG2IMG_MODEL  默认 runwayml/stable-diffusion-v1-5（显存/内存友好）
"""
from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image

_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_ROOT, "..", ".env"))

# 部分环境（如 IDE 内置终端沙箱）对 ~/.cache/huggingface 写入会 Errno 1 Operation not permitted。
# 未显式设置 HF_HOME 时，使用项目内目录，避免 Diffusers 拉锁文件失败。
_hf_home_default = os.path.join(_ROOT, ".hf_hub_cache")
if not os.environ.get("HF_HOME", "").strip():
    os.makedirs(_hf_home_default, exist_ok=True)
    os.environ["HF_HOME"] = _hf_home_default
    print(
        f"[{time.strftime('%H:%M:%S')}] [local_img2img] HF_HOME 未设置，已使用可写目录：{_hf_home_default}",
        file=sys.stderr,
        flush=True,
    )

app = FastAPI(title="Local Diffusers Img2Img", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipe = None
_DEVICE: str = "cpu"
_MODEL_ID = os.getenv("LOCAL_IMG2IMG_MODEL", "runwayml/stable-diffusion-v1-5").strip()
_pipeline_loading = False


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] [local_img2img] {msg}", file=sys.stderr, flush=True)


def _progress_bar(done: int, total: int, width: int = 22) -> str:
    total = max(int(total), 1)
    done = max(0, min(int(done), total))
    filled = min(width, int(round(width * done / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_pipeline():
    """懒加载：首次请求再下载权重（仅第一次较慢）。"""
    global _pipe, _DEVICE, _pipeline_loading
    if _pipe is not None:
        return _pipe, _DEVICE
    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
    except ImportError as e:
        raise RuntimeError(
            "未安装 diffusers/torch。请执行：cd agent && pip install -r requirements-img2img.txt"
        ) from e

    _DEVICE = _pick_device()
    dtype = torch.float16 if _DEVICE in ("mps", "cuda") else torch.float32
    _log(f"加载模型开始 model={_MODEL_ID} device={_DEVICE} dtype={dtype}")
    _log(
        "from_pretrained 中：首次会从 Hub 下载/校验权重，下方可能有 tqdm；"
        "本行每 15s 心跳表示仍在加载（非卡死）。"
    )

    _pipeline_loading = True
    load_t0 = time.perf_counter()

    def _load_heartbeat() -> None:
        while _pipeline_loading:
            time.sleep(15)
            if _pipeline_loading:
                elapsed = int(time.perf_counter() - load_t0)
                _log(f"…仍在加载 pipeline（已 {elapsed}s），请耐心等待")

    hb = threading.Thread(target=_load_heartbeat, daemon=True)
    hb.start()
    try:
        # low_cpu_mem_usage 默认 True 时易在 .to(mps/cuda) 后触发
        # 「Tensor on device cpu is not on the expected device meta」
        _pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            _MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=False,
        )
        _log("权重已载入内存，正在 .to(device)…")
        _pipe = _pipe.to(_DEVICE)
        try:
            _pipe.enable_attention_slicing()
        except Exception:
            pass
        _log(f"加载完成，可在 {_DEVICE} 上推理（总耗时约 {time.perf_counter() - load_t0:.1f}s）")
    finally:
        _pipeline_loading = False

    return _pipe, _DEVICE


class Img2ImgRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    image_base64: str = Field(..., min_length=8)
    strength: float = Field(0.65, ge=0.05, le=0.99)
    width: Optional[int] = Field(None, ge=64, le=1024)
    height: Optional[int] = Field(None, ge=64, le=1024)
    num_inference_steps: int = Field(28, ge=8, le=50)
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0)


@app.get("/health")
def health():
    return {"ok": True, "service": "local-diffusers-img2img", "model": _MODEL_ID}


@app.post("/img2img")
def img2img(body: Img2ImgRequest):
    req_t0 = time.perf_counter()
    _log(
        f"收到 POST /img2img steps={body.num_inference_steps} strength={body.strength:.2f} "
        f"prompt_len={len(body.prompt)} pipeline_loaded={_pipe is not None}"
    )
    raw = body.image_base64.strip()
    if "base64," in raw:
        raw = raw.split("base64,", 1)[-1]
    try:
        image_bytes = base64.b64decode(raw, validate=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}") from e
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}") from e

    tw = body.width or im.width
    th = body.height or im.height
    tw = max(64, min(1024, tw - tw % 8))
    th = max(64, min(1024, th - th % 8))
    # SD1.5 在 MPS 上过大分辨率易 OOM/变慢，默认压到 768 边
    if "stable-diffusion-v1" in _MODEL_ID.lower():
        cap = 768
        if tw > cap or th > cap:
            scale = cap / max(tw, th)
            tw = max(64, min(cap, int(tw * scale) - int(tw * scale) % 8))
            th = max(64, min(cap, int(th * scale) - int(th * scale) % 8))
    if (tw, th) != im.size:
        im = im.resize((tw, th), Image.Resampling.LANCZOS)

    _log(f"图像已解码并 resize → {tw}x{th}（耗时 {time.perf_counter() - req_t0:.2f}s）")

    try:
        _log("获取 pipeline（首次会触发下载/加载）…")
        pipe, device = get_pipeline()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    steps_cfg = body.num_inference_steps
    last_reported = [-1]

    def _on_step(step: int, timestep, latents) -> None:
        # diffusers 回调：step 为当前步序号（通常从 0 递增）
        if step == last_reported[0]:
            return
        last_reported[0] = step
        # img2img 实际去噪步数常 ≤ num_inference_steps，仍以配置步数做进度条总长（观感稳定）
        bar = _progress_bar(step + 1, steps_cfg)
        ts = int(timestep) if timestep is not None else -1
        _log(f"去噪 {bar} step {step + 1}/{steps_cfg} timestep={ts}（请求已 {time.perf_counter() - req_t0:.1f}s）")

    try:
        _log(f"开始 pipe 推理：{tw}x{th} device={device}（diffusers 可能还会打 tqdm 进度条）")
        call_kw = dict(
            prompt=body.prompt,
            image=im,
            strength=body.strength,
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
        )
        try:
            out = pipe(
                **call_kw,
                callback=_on_step,
                callback_steps=1,
                progress_bar=True,
            ).images[0]
        except TypeError:
            try:
                out = pipe(
                    **call_kw,
                    callback=_on_step,
                    callback_steps=1,
                ).images[0]
            except TypeError:
                _log("当前 diffusers 不接受 callback/progress_bar，改为默认推理（无逐步日志）")
                out = pipe(**call_kw).images[0]
    except Exception as e:
        _log(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    data = buf.getvalue()
    _log(
        f"推理成功 device={device} size={tw}x{th} PNG={len(data)} bytes "
        f"（本请求总耗时 {time.perf_counter() - req_t0:.1f}s）"
    )
    return Response(content=data, media_type="image/png")
