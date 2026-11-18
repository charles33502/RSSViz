#!/usr/bin/env bash
# 用法： ./run.sh [twins 資料夾] [port]
# 啟動後在瀏覽器打開 http://localhost:${PORT}
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(realpath "${1:-$HOME/robosimstudio/examples/live/twins}")"
PORT="${2:-8502}"
exec "$HOME/miniconda3/envs/expviz/bin/streamlit" run "$DIR/app.py" \
  --server.port "$PORT" --server.address "${RSSVIZ_ADDR:-localhost}" --server.headless true --browser.gatherUsageStats false \
  -- --root "$ROOT"
