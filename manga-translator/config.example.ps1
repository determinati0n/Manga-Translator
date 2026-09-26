# Optional local configuration for one PowerShell session.
# The application already defaults to D:\AI and D:\manhva.
# Run this file before launching the translator if you use other paths.

$env:MANGA_AI_DIR = "D:\AI"
$env:MANGA_TRANSLATOR_HOME = "D:\manhva"
$env:HF_HOME = "$env:MANGA_AI_DIR\huggingface"
$env:HF_HUB_CACHE = "$env:MANGA_AI_DIR\huggingface\hub"
$env:TORCH_HOME = "$env:MANGA_AI_DIR\torch"
$env:YOLO_CONFIG_DIR = "$env:MANGA_AI_DIR\ultralytics"
$env:PIP_CACHE_DIR = "$env:MANGA_AI_DIR\pip-cache"

Write-Host "Configuration loaded for this PowerShell session."
Write-Host "AI: $env:MANGA_AI_DIR"
Write-Host "Work: $env:MANGA_TRANSLATOR_HOME"
