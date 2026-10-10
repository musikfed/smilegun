$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = '1'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'uv was not found. Install uv and run this script again.' -ForegroundColor Red
    exit 1
}

if (-not (Test-Path '.venv')) {
    uv venv --python 3.12
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

uv sync --python 3.12
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run python generate_sound.py --ensure
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ''
Write-Host 'SmileGun v2.1.1: http://127.0.0.1:5000' -ForegroundColor Green
Write-Host 'Camera and microphone are owned by the browser.' -ForegroundColor Cyan
Write-Host 'Press Ctrl+C to stop.' -ForegroundColor Yellow
uv run python app.py
