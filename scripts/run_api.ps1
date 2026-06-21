# Run FastAPI backend from project root
Set-Location $PSScriptRoot\..
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
