$ErrorActionPreference = "Stop"

$ai = $env:MANGA_AI_DIR
if ([string]::IsNullOrWhiteSpace($ai)) { $ai = "D:\AI" }
$venv = "$ai\venv"
$python = "$venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found: $venv. Run .\setup_windows.ps1 first."
}

Write-Host "Installing pre-built llama-cpp-python CPU wheel..."
& $python -m pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
& $python -m pip check

Write-Host "Qwen runtime installed."
Write-Host "Qwen is still disabled in the translator by default: USE_QWEN_EDITOR = False"
Write-Host "To enable it, change that setting in manga_translator_v10_4.py after testing NLLB-only output."
