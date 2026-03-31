#!/usr/bin/env bash
# 启动 8010 图生图服务，并把 stderr/stdout 追加写入日志文件，方便「另开终端 tail -f」查看。
set -euo pipefail
set -o pipefail

AGENT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$AGENT_ROOT"
mkdir -p logs
LOG="$AGENT_ROOT/logs/local-img2img.log"

{
  echo ""
  echo "======== 本地图生图 8010 ========="
  echo "• 看进度：本窗口往下滚（带 [HH:MM:SS]、[去噪]、加载心跳）"
  echo "• 或另开**一个**终端，在项目根目录执行："
  echo "    tail -f agent/logs/local-img2img.log"
  echo "• 若用 npm run dev:thesis：找终端里带前缀 [I] 的行（I = img2img）"
  echo "=================================="
  echo ""
} | tee -a "$LOG"

"$AGENT_ROOT/.venv/bin/python" -m uvicorn local_img2img_service:app --host 127.0.0.1 --port 8010 2>&1 | tee -a "$LOG"
