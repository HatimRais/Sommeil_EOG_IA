# Prépare les 3 fichiers à uploader sur Kaggle Dataset.
# Usage (depuis la racine du projet) :
#   .\scripts\prepare_kaggle_dataset.ps1
#
# Ensuite : kaggle.com → Datasets → New Dataset → glisser le dossier kaggle_upload/

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$src = "data\processed"
$dst = "kaggle_upload"
$files = @(
    @{ Src = Join-Path $src "sleep_edf_corpus.npz"; Dst = "sleep_edf_corpus.npz" },
    @{ Src = Join-Path $src "sleep_edf_corpus_meta.json"; Dst = "sleep_edf_corpus_meta.json" },
    @{ Src = Join-Path $src "subject_split.json"; Dst = "subject_split.json" }
)
$cnnKeras = Join-Path "models" "sleep_model_cnn_best.keras"
$cnnNpz = Join-Path "models" "sleep_model_cnn_best_weights.npz"
if ((Test-Path $cnnKeras) -and -not (Test-Path $cnnNpz)) {
    Write-Host "Export weights.npz pour Kaggle..." -ForegroundColor DarkGray
    .\.venv\Scripts\python.exe scripts\export_cnn_weights_npz.py
}
$checkpoints = @(
    @{ Src = Join-Path "models" "sleep_model_v1_best.keras"; Dst = "sleep_model_v1_best.keras" },
    @{ Src = $cnnNpz; Dst = "sleep_model_cnn_best_weights.npz" }
)

New-Item -ItemType Directory -Force -Path $dst | Out-Null

foreach ($f in $files) {
    if (-not (Test-Path $f.Src)) {
        Write-Error "Manquant : $($f.Src) - lancez d abord train_npu.py (phase cache)."
    }
    Copy-Item $f.Src (Join-Path $dst $f.Dst) -Force
    $gb = [math]::Round((Get-Item (Join-Path $dst $f.Dst)).Length / 1GB, 2)
    Write-Host "OK  $($f.Dst)  ($gb GB)" -ForegroundColor Green
}

foreach ($ck in $checkpoints) {
    if (Test-Path $ck.Src) {
        Copy-Item $ck.Src (Join-Path $dst $ck.Dst) -Force
        $mb = [math]::Round((Get-Item (Join-Path $dst $ck.Dst)).Length / 1MB, 1)
        Write-Host "OK  $($ck.Dst)  ($mb MB)" -ForegroundColor Green
    } else {
        Write-Host "SKIP $($ck.Dst)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Dossier pret : $((Resolve-Path $dst).Path)" -ForegroundColor Cyan
Write-Host "1. Creer un dataset sur kaggle.com/datasets"
Write-Host "2. Uploader le contenu de kaggle_upload/ (zip ou fichiers)"
Write-Host "3. Noter le slug (ex. sommeil-eog-corpus) dans le notebook"
