# FE_AGAL

FE_AGAL is the isolated French-to-English evaluation pipeline for this repository.

```powershell
.\venv\Scripts\python.exe FE_AGAL\data_splits.py
.\venv\Scripts\python.exe FE_AGAL\run_all.py --dry-run
.\venv\Scripts\python.exe -m pytest FE_AGAL\metrics\test_metrics.py -q
.\venv\Scripts\python.exe FE_AGAL\run_all.py
.\venv\Scripts\python.exe FE_AGAL\analyze.py
```

All new files stay under `FE_AGAL/`. The original source and model are imported, not copied or modified.
