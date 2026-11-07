#!/usr/bin/env sh
export PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export HF_HUB_OFFLINE=0
echo "Demo đang chạy tại http://localhost:8501"
python -m streamlit run "$(dirname "$0")/app.py"
