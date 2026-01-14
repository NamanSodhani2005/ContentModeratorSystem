param(
    [string]$PytorchDir = "C:\\src\\pytorch",
    [string]$VenvPython = "",
    [string]$CudaArch = "12.0",
    [int]$MaxJobs = 8
)

$ErrorActionPreference = "Stop"

if (-not $PSScriptRoot) {
    Write-Error "Script root not available. Run this script from a file."
    exit 1
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $VenvPython) {
    $VenvPython = Join-Path $RepoRoot ".venv\\Scripts\\python.exe"
}

function Require-Command($Name, $Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "$Name not found. $Hint"
        exit 1
    }
}

if (-not (Test-Path $VenvPython)) {
    Write-Error "Venv python not found at: $VenvPython"
    exit 1
}

Require-Command git "Install Git and make sure it's in PATH."
Require-Command cl "Open 'x64 Native Tools Command Prompt for VS 2022' and re-run this script."
Require-Command nvcc "Install CUDA Toolkit 12.4+ and ensure nvcc is in PATH."

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    & $VenvPython -m pip install cmake | Out-Host
}
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    & $VenvPython -m pip install ninja | Out-Host
}

if (-not (Test-Path $PytorchDir)) {
    Write-Host "Cloning PyTorch into $PytorchDir..."
    git clone --recursive https://github.com/pytorch/pytorch.git $PytorchDir
} else {
    Write-Host "Using existing PyTorch repo at $PytorchDir"
    git -C $PytorchDir submodule update --init --recursive
}

Write-Host "Installing build dependencies into venv..."
& $VenvPython -m pip install -U pip setuptools wheel | Out-Host
& $VenvPython -m pip install numpy typing-extensions pyyaml sympy | Out-Host
& $VenvPython -m pip install -r (Join-Path $PytorchDir "requirements.txt") | Out-Host

$env:TORCH_CUDA_ARCH_LIST = $CudaArch
$env:USE_CUDA = "1"
$env:USE_CUDNN = "1"
$env:CMAKE_GENERATOR = "Ninja"
$env:MAX_JOBS = "$MaxJobs"

Write-Host "Building PyTorch (this can take a long time)..."
Push-Location $PytorchDir
& $VenvPython setup.py develop
Pop-Location

Write-Host "Build complete. Verify CUDA:"
& $VenvPython -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
