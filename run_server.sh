#\!/bin/bash
python3 -m uvicorn model2.fast_api:app --host 0.0.0.0 --port 8000 --reload
