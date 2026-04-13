# PowerShell setup for MoodPlay environments on Windows.
# Usage: powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Resolve-CondaExe {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }
    $fallback = Join-Path $env:USERPROFILE "miniconda3\condabin\conda.bat"
    if (Test-Path $fallback) {
        return $fallback
    }
    throw "Conda executable not found. Install Miniconda/Anaconda or open a shell with conda initialized."
}

$condaExe = Resolve-CondaExe

function Ensure-Env {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvName,
        [Parameter(Mandatory = $true)]
        [string]$EnvFile
    )
    $envExists = & $condaExe env list | Select-String -Pattern "^$EnvName\s"
    if ($envExists) {
        Write-Host "Updating conda env: $EnvName"
        & $condaExe env update -f $EnvFile
    } else {
        Write-Host "Creating conda env: $EnvName"
        & $condaExe env create -f $EnvFile
    }
}

Write-Host "=== Setting up vidcolor env ==="
Ensure-Env -EnvName "vidcolor" -EnvFile "environment.yml"

Write-Host "Installing CUDA torch in vidcolor"
& $condaExe run -n vidcolor python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing vidcolor requirements"
& $condaExe run -n vidcolor python -m pip install -r requirements.txt

Write-Host "Installing SAM2 (Segment Anything 2)"
& $condaExe run -n vidcolor python -m pip install -U git+https://github.com/facebookresearch/segment-anything-2.git

Write-Host "Installing CoTracker"
& $condaExe run -n vidcolor python -m pip install -U git+https://github.com/facebookresearch/co-tracker.git

Write-Host "=== Setting up gdino310 env ==="
Ensure-Env -EnvName "gdino310" -EnvFile "environment-gdino310.yml"

Write-Host "Installing CUDA torch in gdino310"
& $condaExe run -n gdino310 python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing gdino requirements"
& $condaExe run -n gdino310 python -m pip install -r requirements-gdino310.txt

Write-Host "Installing GroundingDINO"
& $condaExe run -n gdino310 python -m pip install -U git+https://github.com/IDEA-Research/GroundingDINO.git

Write-Host "Pinning transformers for GroundingDINO"
& $condaExe run -n gdino310 python -m pip uninstall -y transformers
& $condaExe run -n gdino310 python -m pip install "transformers==4.26.1"

Write-Host "=== Done ==="
Write-Host "Next steps:"
Write-Host "1) Place checkpoints under models/checkpoints (see README)."
Write-Host "2) Put GroundingDINO_SwinB.cfg.py in configs/perception/grounding_dino/."
Write-Host "3) Verify configs/perception/grounding_dino.yaml DLL paths for your machine."
Write-Host "4) Run: conda activate vidcolor; streamlit run app.py"
