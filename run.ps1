$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# Без этого Python буферизует вывод блоками, и отчёт диагностики появляется
# в консоли IDE с задержкой (или не появляется до остановки процесса).
$env:PYTHONUNBUFFERED = '1'

if (-not (Test-Path '.venv')) {
    uv venv --python 3.12
}

uv pip install -r requirements.txt

if (-not (Test-Path 'static\blaster.mp3')) {
    uv run python generate_sound.py
}

Write-Host ''
Write-Host 'SmileGun: http://127.0.0.1:5000' -ForegroundColor Green
Write-Host 'Для остановки: Ctrl+C' -ForegroundColor Yellow
uv run python app.py
