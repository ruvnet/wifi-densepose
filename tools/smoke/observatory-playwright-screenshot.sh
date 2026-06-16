#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-http://127.0.0.1:3000/observatory.html}"
OUT="${2:-/tmp/ruview-observatory-playwright.png}"

if [ -z "${PLAYWRIGHT_BROWSERS_PATH:-}" ] && [ -d /run/media/deck/SDCARD/agents/playwright-browsers ]; then
  export PLAYWRIGHT_BROWSERS_PATH=/run/media/deck/SDCARD/agents/playwright-browsers
fi

npx playwright screenshot \
  --browser=chromium \
  --viewport-size=1280,800 \
  "$BASE" \
  "$OUT"

python3 - "$OUT" <<'PY'
import sys
from PIL import Image, ImageStat

p = sys.argv[1]
im = Image.open(p).convert("RGB")
stat = ImageStat.Stat(im)
print(f"file={p}")
print(f"size={im.size[0]}x{im.size[1]}")
print("stddev=" + ",".join(f"{x:.2f}" for x in stat.stddev))
print("nonblank=" + str(max(stat.stddev) > 5).lower())
PY
