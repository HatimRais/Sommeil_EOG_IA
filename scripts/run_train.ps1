# Lance l'entrainement CNN NPU avec sortie en direct dans ce terminal.
# Usage : depuis la racine du projet
#   .\scripts\run_train.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Error "Environnement .venv introuvable. Executez d'abord: python -m venv .venv"
}

New-Item -ItemType Directory -Force -Path "data\processed" | Out-Null

$env:PYTHONUNBUFFERED = "1"
Write-Host "=== Entrainement Sommeil_EOG_IA (train_npu.py) ===" -ForegroundColor Cyan
Write-Host "Ctrl+C pour interrompre. Le cache accelere les prochains runs." -ForegroundColor DarkGray
Write-Host ""

.\.venv\Scripts\python.exe -u src\train_npu.py
