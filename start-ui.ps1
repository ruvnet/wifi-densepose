Set-Location "$PSScriptRoot\ui"
Write-Host "Demarrage UI ITBwifi sur http://localhost:3000 ..." -ForegroundColor Green
C:\jarvis\python\python.exe -m http.server 3000
