#!/usr/bin/env bash
set -euo pipefail

# 使用 TuShare 下载全市场日频数据。
# 依赖：pip install tushare
# 运行前先设置 token：
#   export TUSHARE_TOKEN="YOUR_TOKEN"

# : "${TUSHARE_TOKEN:?请先 export TUSHARE_TOKEN=... (或在 shell 环境里设置 TUSHARE_PRO_TOKEN)}"
export TUSHARE_TOKEN=""

OUTPUT_DIR="${OUTPUT_DIR:-/home/rainjoe/ws/qlibC/raw_data/day}"
START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-}"
FORMAT="${FORMAT:-parquet}"
ADJUST="${ADJUST:-qfq}"
LIMIT="${LIMIT:-}"

cmd=(
  python3 ./data/download_all_stocks_daily_tushare.py
  --output-dir "$OUTPUT_DIR"
  --start-date "$START_DATE"
  --format "$FORMAT"
  --adjust "$ADJUST"
)

if [[ -n "$END_DATE" ]]; then
  cmd+=(--end-date "$END_DATE")
fi
if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

"${cmd[@]}"