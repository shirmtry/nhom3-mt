$env:PROJECT_ROOT = (Resolve-Path "$PSScriptRoot\..\..").Path
$env:HF_HUB_OFFLINE = "0"
Write-Host "🎬 Demo đang chạy tại http://localhost:8501"
python -m streamlit run "$PSScriptRoot\app.py"
