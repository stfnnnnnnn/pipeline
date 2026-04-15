# MoodPlay Video Colorizer

This repository contains a Streamlit app and a multi-stage perception pipeline:

1. Phase A (Keyframes): YOLO26s + SAM2 masks
2. Phase B: CoTracker point tracking
3. Phase C: SAM2 masks on all frames
4. Optional: GroundingDINO for scene labels (buildings/trees/road/sky)

Two conda environments are required:
- `vidcolor` for the main app (YOLO/SAM2/CoTracker)
- `gdino310` for GroundingDINO (separate env for Windows DLL stability)

## 1) Create the environments

### Quick setup (Windows)
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

### A. Main app env (vidcolor)
```powershell
conda env create -f environment.yml
conda activate vidcolor

# Install CUDA-enabled torch
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining deps
python -m pip install -r requirements.txt
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

- YOLO26s: models/checkpoints/yolo/yolo26s.pt
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

- YOLO labels and masking:
	- configs/perception/yolo26_s.yaml
- CoTracker settings:
	- configs/perception/cotracker3.yaml
- GroundingDINO prompts and thresholds:
	- configs/perception/grounding_dino.yaml

## Notes

- GroundingDINO runs in `gdino310` via `conda run` from the main app.
- Torch must be CUDA-enabled in both envs.
- If `conda` is not found by subprocess, update `conda_exe` in
	configs/perception/grounding_dino.yaml.
