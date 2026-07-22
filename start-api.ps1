Set-Location "$PSScriptRoot\archive\v1"
$env:PYTHONPATH = "."
Write-Host "Demarrage API ITBwifi sur http://localhost:8000 ..." -ForegroundColor Cyan
.venv\Scripts\uvicorn.exe src.api.main:app --host 0.0.0.0 --port 8000
