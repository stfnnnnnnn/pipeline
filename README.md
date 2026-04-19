# MoodPlay Video Colorizer

This repository contains a Streamlit app and a multi-stage perception pipeline:

1. Phase A (Keyframes): GroundingDINO + SAM2 masks
2. Phase B: CoTracker point tracking
3. Phase C: SAM2 masks on all frames
4. Optional: XMem propagation for temporal consistency

## Executive summary of recent platform changes

- Detection stack pivot: default detector migrated from legacy YOLOv26-style flow to GroundingDINO open-vocabulary detection.
- Segmentation quality upgrades: depth-aware fusion, strict semantic depth hierarchy, and accessory-safe negative-point handling to reduce mask fragmentation and crossfire.
- Tracking policy update: background classes are treated as large-stuff context and are excluded from foreground tracking paths where appropriate.
- Runtime architecture: GroundingDINO now runs from a dedicated `gdino310` conda environment through subprocess isolation for Windows stability.
- Compatibility posture: YOLO-related installation remains available in setup automation for legacy experiments and rollback workflows.

## Component roles in current pipeline

- GroundingDINO (`scripts/gdino_worker.py`, `src/perception/grounding_dino_detector.py`): open-vocabulary box proposal generation.
- SAM2 (`src/perception/sam_segmenter.py`): per-box mask generation and multimask candidate selection.
- CoTracker (`src/perception/cotracker_wrapper.py`): persistent motion-aware track support and temporal continuity.
- XMem (`src/perception/xmem_wrapper.py`): optional long-range temporal propagation.
- Orchestrator (`src/perception/segmentation.py`): fusion, association, depth-aware subtraction, and end-to-end phase execution.

Two conda environments are required:
- `vidcolor` for the main app (SAM2/CoTracker)
- `gdino310` for GroundingDINO (separate env for Windows DLL stability)

## 1) Create the environments

### Quick setup (Windows)
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

`setup_windows.ps1` is the primary installer for this repository. It creates/updates `vidcolor` and `gdino310`, installs GroundingDINO, installs XMem dependencies in `vidcolor`, and keeps YOLO compatibility installation.

Optional wrapper script is also available:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

### A. Main app env (vidcolor)
```powershell
conda env create -f environment.yml
conda activate vidcolor

# Install CUDA-enabled torch
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining deps
python -m pip install -r requirements.txt

# Install XMem runtime dependencies
python -m pip install -r XMem/requirements.txt

# Ensure base XMem dependencies are present
python -m pip install torch torchvision opencv-python pillow tqdm
```

Install YOLO compatibility packages (optional runtime path, retained intentionally):

```powershell
python -m pip install -U ultralytics
```

Verify CUDA:
```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### B. GroundingDINO env (gdino310)
```powershell
conda env create -f environment-gdino310.yml
conda activate gdino310

# Install CUDA-enabled torch
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install GroundingDINO from source
python -m pip install -U git+https://github.com/IDEA-Research/GroundingDINO.git
```

Pin transformers to avoid BERT API breakages:
```powershell
python -m pip uninstall -y transformers
python -m pip install "transformers==4.26.1"
```

Sanity checks:
```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import groundingdino._C as C; print('groundingdino _C ok')"
```

## 2) Model checkpoints

Place checkpoints in the following paths:

- SAM2 Hiera-L: models/checkpoints/sam2/sam2_hiera_large.pt
- CoTracker3: models/checkpoints/cotracker/cotracker3.pth
- GroundingDINO SwinB: models/checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth

GroundingDINO config file:
- configs/perception/grounding_dino/GroundingDINO_SwinB.cfg.py

## 3) GroundingDINO Windows DLL configuration

Open configs/perception/grounding_dino.yaml and set:

- `conda_exe` to your conda.bat
- `torch_lib_dir` to the gdino310 torch lib directory
- `cuda_bin_dir` to your CUDA bin path

Default values are already set for a typical Windows layout:
```
C:\Users\LENOVO\miniconda3\condabin\conda.bat
C:\Users\LENOVO\miniconda3\envs\gdino310\Lib\site-packages\torch\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
```

## 4) Run the app

```powershell
conda activate vidcolor
streamlit run app.py
```

## 5) Key configs to tune

- CoTracker settings:
	- configs/perception/cotracker3.yaml
- GroundingDINO prompts and thresholds:
	- configs/perception/grounding_dino.yaml

## Notes

- GroundingDINO runs in `gdino310` via `conda run` from the main app.
- Torch must be CUDA-enabled in both envs.
- XMem dependencies are installed in `vidcolor` from `XMem/requirements.txt` plus base packages (`torch`, `torchvision`, `opencv-python`, `pillow`, `tqdm`).
- YOLO installation is retained in setup scripts for backward compatibility, but GroundingDINO is the default detector in the current pipeline.
- If `conda` is not found by subprocess, update `conda_exe` in
	configs/perception/grounding_dino.yaml.
