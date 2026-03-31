#!/usr/bin/env bash
# 测试 8010 本地图生图是否返回 PNG（首次可能很慢：下载+加载模型）
#
# 与「手搓 curl」测试一致：灰 64² 底图、prompt=a small red circle on gray、
# strength=0.55、256²、num_inference_steps=12、curl -m 420（可用 CURL_MAX_TIME 覆盖）。
#
# 用法：
#   终端 A：npm run local-img2img   ← 时间戳 / 15s 加载心跳 / 去噪进度条
#   终端 B：bash scripts/test-8010-img2img.sh   ← 每 10s 一行「已等待 Ns」（可滚动回看）
#
# 单条 curl 在响应返回前**不会**打印任何进度；要看本窗口日志请用本脚本，或另开端口跑 while sleep。

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/.tmp_img2img_test.png"
URL="${LOCAL_IMG2IMG_URL:-http://127.0.0.1:8010}"
TICK_SEC="${TEST_8010_TICK_SEC:-10}"

echo "==> 健康检查: ${URL}/health"
curl -sS -m 10 "${URL}/health" | head -c 200 || true
echo ""
echo ""
echo "==> 细节进度：终端 A（local_img2img）。"
echo "==> 本终端每 ${TICK_SEC}s 打印一行等待时间（纯 curl 期间这里是静默的）。"
echo ""

B64="$(python3 -c "
import base64, io
from PIL import Image
im = Image.new('RGB', (64, 64), (200, 200, 200))
b = io.BytesIO()
im.save(b, 'PNG')
print(base64.b64encode(b.getvalue()).decode())
")"

rm -f "$OUT"

SPIN_SEC=0
(
  while true; do
    sleep "$TICK_SEC"
    SPIN_SEC=$((SPIN_SEC + TICK_SEC))
    echo "[test-8010-img2img] 已等待 ${SPIN_SEC}s（去噪/下载请看 local-img2img 终端）"
  done
) &
SPIN_PID=$!
trap 'kill "$SPIN_PID" 2>/dev/null || true' EXIT

# 默认 420s，与常见手搓 curl -m 420 一致；首次加载可 export CURL_MAX_TIME=900
MAX_TIME="${CURL_MAX_TIME:-420}"
set +e
curl -sS -o "$OUT" -w "\n\nimg2img_http=%{http_code} time_total=%{time_total}s bytes=%{size_download}\n" \
  -m "$MAX_TIME" -X POST "${URL}/img2img" \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"a small red circle on gray\",\"image_base64\":\"${B64}\",\"strength\":0.55,\"width\":256,\"height\":256,\"num_inference_steps\":12}"
CURL_RC=$?
set -e

kill "$SPIN_PID" 2>/dev/null || true
wait "$SPIN_PID" 2>/dev/null || true
trap - EXIT
printf '\n'

if [[ "$CURL_RC" -ne 0 ]]; then
  echo "curl 退出码: $CURL_RC（28 多为超时，可 export CURL_MAX_TIME=900 再试）"
  exit "$CURL_RC"
fi

if file "$OUT" 2>/dev/null | grep -q PNG; then
  echo "OK: 已写入 PNG → $OUT"
  file "$OUT" 2>/dev/null || true
  ls -la "$OUT"
else
  echo "失败: 输出不是 PNG，内容预览:"
  head -c 400 "$OUT" || true
  exit 1
fi
