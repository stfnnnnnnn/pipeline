# PowerShell setup for MoodPlay environments on Windows.
# Usage: powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Resolve-CondaExe {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        return $env:CONDA_EXE
    }

    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.CommandType -eq "Application" -and (Test-Path $cmd.Source)) {
        return $cmd.Source
    }

    $condaExeCmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($condaExeCmd -and $condaExeCmd.Source -and (Test-Path $condaExeCmd.Source)) {
        return $condaExeCmd.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\condabin\conda.bat"),
        (Join-Path $env:USERPROFILE "anaconda3\condabin\conda.bat"),
        "C:\ProgramData\miniconda3\condabin\conda.bat",
        "C:\ProgramData\anaconda3\condabin\conda.bat"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Conda executable not found. Install Miniconda/Anaconda or open a shell with conda initialized."
}

$condaExe = Resolve-CondaExe

function Invoke-Conda {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args,
        [switch]$CaptureOutput
    )

    if ($CaptureOutput) {
        $output = & $condaExe @Args 2>&1
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "Conda command failed (exit $exitCode): conda $($Args -join ' ')`n$($output -join "`n")"
        }
        return $output
    }

    & $condaExe @Args
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Conda command failed (exit $exitCode): conda $($Args -join ' ')"
    }
}

function Ensure-Env {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvName,
        [Parameter(Mandatory = $true)]
        [string]$EnvFile
    )

    $envListRaw = Invoke-Conda -Args @("env", "list", "--json") -CaptureOutput
    $envList = ($envListRaw -join "`n") | ConvertFrom-Json

    $envExists = $false
    foreach ($envPath in $envList.envs) {
        if ((Split-Path -Leaf $envPath).ToLowerInvariant() -eq $EnvName.ToLowerInvariant()) {
            $envExists = $true
            break
        }
    }

    if ($envExists) {
        Write-Host "Updating conda env: $EnvName"
        Invoke-Conda -Args @("env", "update", "--name", $EnvName, "--file", $EnvFile, "--environment-spec", "environment.yml", "--prune")
    } else {
        Write-Host "Creating conda env: $EnvName"
        Invoke-Conda -Args @("env", "create", "--name", $EnvName, "--file", $EnvFile, "--environment-spec", "environment.yml")
    }
}

function Get-DriverCudaVersion {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        return $null
    }

    try {
        $out = & $smi.Source 2>$null | Out-String
        if ($out -match "CUDA Version:\s*([0-9]+\.[0-9]+)") {
            return [double]$Matches[1]
        }
    } catch {
        return $null
    }

    return $null
}

function Test-NvidiaGpuAvailable {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        return $false
    }

    try {
        $out = & $smi.Source -L 2>$null | Out-String
        return ($out -match "GPU\s+[0-9]+")
    } catch {
        return $false
    }
}

function Test-TorchCudaKernel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvName
    )

    $probe = "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); n=torch.cuda.device_count(); print('device_count', n); print('gpu', torch.cuda.get_device_name(0) if n else 'none'); print('cap', torch.cuda.get_device_capability(0) if n else 'none'); x=torch.randn(128, device='cuda') if n else None; print('cuda_kernel_ok', float((x*x).sum().item()) if n else 'n/a')"
    try {
        $output = Invoke-Conda -Args @("run", "-n", $EnvName, "python", "-c", $probe) -CaptureOutput
        $text = $output -join "`n"
        if ($text -match "avail True" -and $text -match "cuda_kernel_ok") {
            return $true
        }
        return $false
    } catch {
        return $false
    }
}

function Test-TorchImport {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvName
    )

    $probe = "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
    try {
        $output = Invoke-Conda -Args @("run", "-n", $EnvName, "python", "-c", $probe) -CaptureOutput
        $text = $output -join "`n"
        return ($text -match "torch")
    } catch {
        return $false
    }
}

function Install-TorchCpuFallback {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvName
    )

    Write-Host "Installing CPU torch in $EnvName as compatibility fallback..."
    Invoke-Conda -Args @(
        "run", "-n", $EnvName, "python", "-m", "pip", "install", "-U",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    )

    if (-not (Test-TorchImport -EnvName $EnvName)) {
        throw "CPU torch install failed in env '$EnvName'."
    }

    Write-Host "CPU torch install verified in $EnvName."
}

function Install-TorchWithCudaFallback {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvName,
        [switch]$AllowCpuFallback = $true
    )

    $hasNvidiaGpu = Test-NvidiaGpuAvailable
    if (-not $hasNvidiaGpu) {
        if ($AllowCpuFallback) {
            Write-Host "No NVIDIA GPU detected on this machine. Skipping cu121/cu128 and installing CPU torch."
            Install-TorchCpuFallback -EnvName $EnvName
            return
        }
        throw "No NVIDIA GPU detected, and CPU fallback is disabled."
    }

    $driverCuda = Get-DriverCudaVersion
    $preferCu128 = $false
    if ($driverCuda -ne $null -and $driverCuda -ge 12.8) {
        $preferCu128 = $true
    }

    if ($driverCuda -ne $null) {
        if ($preferCu128) {
            Write-Host "Detected NVIDIA driver CUDA $driverCuda. Preferring cu128 wheels first."
        } else {
            Write-Host "Detected NVIDIA driver CUDA $driverCuda. Preferring cu121 wheels first."
        }
    } else {
        Write-Host "NVIDIA GPU detected but driver CUDA version could not be parsed. Trying cu121 then cu128."
    }

    if ($preferCu128) {
        $attempts = @(
            @{ Name = "cu128 stable"; Args = @("--index-url", "https://download.pytorch.org/whl/cu128") },
            @{ Name = "cu121 stable"; Args = @("--index-url", "https://download.pytorch.org/whl/cu121") }
        )
    } else {
        $attempts = @(
            @{ Name = "cu121 stable"; Args = @("--index-url", "https://download.pytorch.org/whl/cu121") },
            @{ Name = "cu128 stable"; Args = @("--index-url", "https://download.pytorch.org/whl/cu128") }
        )
    }

    foreach ($attempt in $attempts) {
        Write-Host "Installing torch in $EnvName using $($attempt.Name)..."
        try {
            Invoke-Conda -Args (
                @("run", "-n", $EnvName, "python", "-m", "pip", "install", "-U", "torch", "torchvision", "torchaudio") +
                $attempt.Args
            )
            if (Test-TorchCudaKernel -EnvName $EnvName) {
                Write-Host "Torch CUDA install verified in $EnvName with $($attempt.Name)."
                return
            }
            Write-Host "Torch installed in $EnvName but CUDA kernel probe failed for $($attempt.Name)."
        } catch {
            Write-Host "Torch install attempt failed for $EnvName using $($attempt.Name). Trying next fallback..."
        }
    }

    if ($AllowCpuFallback) {
        Write-Host "Unable to verify GPU torch in '$EnvName' with cu121/cu128. Falling back to CPU torch."
        Install-TorchCpuFallback -EnvName $EnvName
        return
    }

    throw "Unable to install a GPU-compatible torch build in env '$EnvName' using cu121/cu128 wheels, and CPU fallback is disabled."
}

function Test-GroundingDinoCustomOps {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvName
    )

    $probe = "import groundingdino._C as C; print('gdino_custom_ops_ok', hasattr(C, 'ms_deform_attn_forward'))"
    try {
        $output = Invoke-Conda -Args @("run", "-n", $EnvName, "python", "-c", $probe) -CaptureOutput
        $text = $output -join "`n"
        return ($text -match "gdino_custom_ops_ok True")
    } catch {
        return $false
    }
}

function Install-GroundingDinoWithCustomOps {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvName
    )

    Write-Host "Installing GroundingDINO build tools in $EnvName"
    Invoke-Conda -Args @("run", "-n", $EnvName, "python", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "ninja")

    Write-Host "Installing GroundingDINO from source in $EnvName"
    $srcZip = "https://github.com/IDEA-Research/GroundingDINO/archive/refs/heads/main.zip"
    $prevCudaVisible = $env:CUDA_VISIBLE_DEVICES
    # Hide CUDA during package build so setup.py avoids failing custom-op compilation on newer GPUs.
    $env:CUDA_VISIBLE_DEVICES = "-1"
    try {
        Invoke-Conda -Args @("run", "-n", $EnvName, "python", "-m", "pip", "install", "-U", "--force-reinstall", "--no-cache-dir", "--no-build-isolation", $srcZip)
    } catch {
        Write-Host "GroundingDINO install with --no-build-isolation failed, retrying standard source install..."
        Invoke-Conda -Args @("run", "-n", $EnvName, "python", "-m", "pip", "install", "-U", "--force-reinstall", "--no-cache-dir", $srcZip)
    } finally {
        if ($null -eq $prevCudaVisible) {
            Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
        } else {
            $env:CUDA_VISIBLE_DEVICES = $prevCudaVisible
        }
    }

    if (-not (Test-GroundingDinoCustomOps -EnvName $EnvName)) {
        Write-Host "GroundingDINO custom C++ ops are not available in env '$EnvName'."
        Write-Host "Pipeline will use PyTorch attention fallback in gdino_worker (GPU-capable but slower)."
    }
}

Write-Host "=== Setting up vidcolor env ==="
Ensure-Env -EnvName "vidcolor" -EnvFile "environment.yml"

Write-Host "Installing CUDA torch in vidcolor"
Install-TorchWithCudaFallback -EnvName "vidcolor"

Write-Host "Installing vidcolor requirements"
Invoke-Conda -Args @("run", "-n", "vidcolor", "python", "-m", "pip", "install", "-r", "requirements.txt")

Write-Host "Installing YOLO compatibility packages in vidcolor (retained intentionally)"
Invoke-Conda -Args @("run", "-n", "vidcolor", "python", "-m", "pip", "install", "-U", "ultralytics")

Write-Host "Installing SAM2 (Segment Anything 2)"
Invoke-Conda -Args @("run", "-n", "vidcolor", "python", "-m", "pip", "install", "-U", "git+https://github.com/facebookresearch/segment-anything-2.git")

Write-Host "Installing CoTracker"
Invoke-Conda -Args @("run", "-n", "vidcolor", "python", "-m", "pip", "install", "-U", "git+https://github.com/facebookresearch/co-tracker.git")

Write-Host "Ensuring base XMem dependencies in vidcolor (torch torchvision opencv-python pillow tqdm)"
$xmemBaseProbe = "import importlib.util as iu; pairs=[('torch','torch'),('torchvision','torchvision'),('cv2','opencv-python'),('PIL','pillow'),('tqdm','tqdm')]; miss=[p for m,p in pairs if iu.find_spec(m) is None]; print(' '.join(miss))"
$xmemMissingRaw = Invoke-Conda -Args @("run", "-n", "vidcolor", "python", "-c", $xmemBaseProbe) -CaptureOutput
$xmemMissing = ($xmemMissingRaw -join " ").Trim()
if ($xmemMissing) {
    $xmemPackages = $xmemMissing.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
    Write-Host "Installing missing XMem base packages: $($xmemPackages -join ', ')"
    Invoke-Conda -Args (@("run", "-n", "vidcolor", "python", "-m", "pip", "install", "-U") + $xmemPackages)
} else {
    Write-Host "Base XMem dependencies already present in vidcolor."
}

if (Test-Path "XMem\requirements.txt") {
    Write-Host "Installing XMem runtime dependencies in vidcolor"
    Invoke-Conda -Args @("run", "-n", "vidcolor", "python", "-m", "pip", "install", "-r", "XMem\requirements.txt")
} else {
    Write-Host "XMem requirements file not found at XMem\\requirements.txt, skipping XMem dependency install."
}

Write-Host "=== Setting up gdino310 env ==="
Ensure-Env -EnvName "gdino310" -EnvFile "environment-gdino310.yml"

Write-Host "Installing CUDA torch in gdino310"
Install-TorchWithCudaFallback -EnvName "gdino310"

Write-Host "Installing gdino requirements"
Invoke-Conda -Args @("run", "-n", "gdino310", "python", "-m", "pip", "install", "-r", "requirements-gdino310.txt")

Write-Host "Installing GroundingDINO"
Install-GroundingDinoWithCustomOps -EnvName "gdino310"

Write-Host "Pinning transformers for GroundingDINO"
Invoke-Conda -Args @("run", "-n", "gdino310", "python", "-m", "pip", "uninstall", "-y", "transformers")
Invoke-Conda -Args @("run", "-n", "gdino310", "python", "-m", "pip", "install", "transformers==4.26.1")

Write-Host "=== Done ==="
Write-Host "Next steps:"
Write-Host "1) Place checkpoints under models/checkpoints (see README)."
Write-Host "2) Put GroundingDINO_SwinB.cfg.py in configs/perception/grounding_dino/."
Write-Host "3) Optional: set DLL paths in configs/perception/grounding_dino.yaml only if auto-detection fails."
Write-Host "4) Run: conda activate vidcolor; streamlit run app.py"
