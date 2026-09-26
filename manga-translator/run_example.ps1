$ErrorActionPreference = "Stop"
$python = "D:\AI\venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Run .\setup_windows.ps1 first." }

& $python .\manga_translator_v10_4.py `
  --input "D:\manhva\input.pdf" `
  --output "D:\manhva\translated.pdf" `
  --pages 4
