$ErrorActionPreference = "Stop"

# ============================================================
# Manga Translator — Windows CPU setup
# ============================================================
# Heavy caches/models live on D: by default. Edit these two lines if needed.
$env:MANGA_AI_DIR = "D:\AI"
$env:MANGA_TRANSLATOR_HOME = "D:\manhva"
$env:HF_HOME = "$env:MANGA_AI_DIR\huggingface"
$env:HF_HUB_CACHE = "$env:MANGA_AI_DIR\huggingface\hub"
$env:TORCH_HOME = "$env:MANGA_AI_DIR\torch"
$env:YOLO_CONFIG_DIR = "$env:MANGA_AI_DIR\ultralytics"
$env:PIP_CACHE_DIR = "$env:MANGA_AI_DIR\pip-cache"

# Persist cache locations for future PowerShell sessions.
[Environment]::SetEnvironmentVariable("MANGA_AI_DIR", $env:MANGA_AI_DIR, "User")
[Environment]::SetEnvironmentVariable("MANGA_TRANSLATOR_HOME", $env:MANGA_TRANSLATOR_HOME, "User")
[Environment]::SetEnvironmentVariable("HF_HOME", $env:HF_HOME, "User")
[Environment]::SetEnvironmentVariable("HF_HUB_CACHE", $env:HF_HUB_CACHE, "User")
[Environment]::SetEnvironmentVariable("TORCH_HOME", $env:TORCH_HOME, "User")
[Environment]::SetEnvironmentVariable("YOLO_CONFIG_DIR", $env:YOLO_CONFIG_DIR, "User")
[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", $env:PIP_CACHE_DIR, "User")

New-Item -ItemType Directory -Force -Path `
    $env:MANGA_AI_DIR, `
    $env:MANGA_TRANSLATOR_HOME, `
    $env:HF_HOME, `
    $env:HF_HUB_CACHE, `
    $env:TORCH_HOME, `
    $env:YOLO_CONFIG_DIR, `
    $env:PIP_CACHE_DIR | Out-Null

$venv = "$env:MANGA_AI_DIR\venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    py -3.12 -m venv $venv
}

$python = "$venv\Scripts\python.exe"
& $python -m pip install --upgrade pip

Write-Host "`n[1/3] Installing PyTorch CPU..."
& $python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "`n[2/3] Installing PaddlePaddle CPU..."
& $python -m pip install --upgrade paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

Write-Host "`n[3/3] Installing project dependencies..."
& $python -m pip install -r .\requirements.txt

Write-Host "`nVerifying core imports..."
& $python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
& $python -c "import paddle; print('paddle:', paddle.__version__); paddle.utils.run_check()"
& $python -c "import transformers, ultralytics, paddleocr, fitz, cv2; print('core imports: OK')"
& $python -m pip check

Write-Host "`nEnvironment prepared successfully."
Write-Host "Activate with:"
Write-Host "  & $venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Optional Qwen setup:"
Write-Host "  .\setup_qwen_windows.ps1"
