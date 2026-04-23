# PEFT LoRA End-to-End Documentation (MoodPlay Pipeline)

## 1) Scope and Goal

This document captures all essential PEFT LoRA details for this repository:

- installation prerequisites and dependency stack
- repository path map and where artifacts live
- complete code changes that were added/updated
- how to prepare data, launch training, and validate outputs
- how to run from the interactive console with training hooks
- troubleshooting and verification checklist

This is written for the current Windows workspace layout.


## 2) Repository Path Map

Primary repository root:

- `C:\Users\Administrator\Documents\pipeline`

Secondary workspace root (contains model assets/checkpoints in this workspace):

- `C:\Users\Administrator\Desktop\models`

Important training paths (relative to repository root):

- LoRA dataset root: `data/lora_training`
- LoRA config: `configs/training/lora_moods.yaml`
- LoRA trainer wrapper: `scripts/train_lora_mood.py`
- Local PEFT LoRA trainer: `scripts/train_text_to_image_lora.py`
- Dataset setup/import helper: `scripts/setup_lora_datasets.py`
- Dataset validator: `scripts/validate_lora_dataset.py`
- Interactive console: `scripts/lora_console.py`
- LoRA test script: `scripts/test_lora.py`
- App mood registry (active integration): `src/diffusion/style_lora_adapter.py`

LoRA output location:

- `models/loras/<mood>/pytorch_lora_weights.safetensors`


## 3) PEFT LoRA Installation and Environment

### 3.1 Recommended setup flow

Use repository bootstrap script (Windows):

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

Then activate training env:

```powershell
conda activate vidcolor
```

Install/refresh Python packages:

```powershell
python -m pip install -r requirements.txt
```

### 3.2 Core dependencies relevant to PEFT LoRA

From `requirements.txt`, essential LoRA stack includes:

- `torch` / `torchvision` / `torchaudio` (installed via setup script using CUDA wheel strategy)
- `diffusers==0.30.0`
- `transformers==4.44.0`
- `accelerate==0.34.0`
- `peft==0.12.0`
- `datasets==2.20.0` (added)
- `safetensors==0.4.5`
- `Pillow==10.4.0`
- `pyyaml==6.0.2`
- `tensorboard`

### 3.3 Accelerate setup

Run once per environment:

```powershell
accelerate config
```

Recommended for this local training flow:

- single machine
- single GPU
- fp16 mixed precision (if supported)

### 3.4 Verification commands

Verify GPU torch in `vidcolor`:

```powershell
conda run -n vidcolor python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Verify trainer CLI is available:

```powershell
conda run -n vidcolor python scripts/train_text_to_image_lora.py --help
```


## 4) Mood Naming and Dataset Convention

Active canonical moods for LoRA training are now exactly:

- `neo_tokyo`
- `neutral_realistic`
- `warm_sunny`

Dataset root layout:

```text
data/lora_training/
  neo_tokyo/
	 neo_tokyo_00001.png
	 neo_tokyo_00001.txt
	 ...
  neutral_realistic/
	 neutral_realistic_00001.png
	 neutral_realistic_00001.txt
	 ...
  warm_sunny/
	 warm_sunny_00001.png
	 warm_sunny_00001.txt
	 ...
```

Sidecar caption rule:

- If `<image>.txt` exists, caption is read from it.
- If missing, the trainer falls back to image stem text.


## 5) Complete Code Changes (Detailed)

### 5.1 New: local PEFT LoRA trainer

File: `scripts/train_text_to_image_lora.py`

Implemented:

1. Local image-folder dataset class (`LocalLoraImageDataset`) using image + optional sidecar caption.
2. UNet-only PEFT LoRA adapter injection via `peft.LoraConfig`.
3. Diffusers SD1.5 component loading (`tokenizer`, `text_encoder`, `vae`, `unet`, `scheduler`).
4. LoRA target modules configured as:
	- `to_k`, `to_q`, `to_v`, `to_out.0`
5. Training loop with:
	- `Accelerator`
	- gradient accumulation
	- optional gradient checkpointing
	- optional 8-bit Adam fallback logic
	- periodic checkpoint save
6. Final LoRA export using:
	- `StableDiffusionPipeline.save_lora_weights(...)`
	- output file: `pytorch_lora_weights.safetensors`
7. Run metadata output:
	- `models/loras/<mood>/run_config.json`

### 5.2 Updated: training wrapper

File: `scripts/train_lora_mood.py`

Changes:

1. Default training script now points to local trainer:
	- `scripts/train_text_to_image_lora.py`
2. Relative paths are normalized against repository root.
3. Explicit existence checks were added for config and trainer script.
4. Additional parameters now forwarded from config:
	- `dataloader_num_workers`
	- `checkpointing_steps`
	- `lr_warmup_steps`
	- optional `center_crop`, `random_flip`
5. Launch command uses `accelerate launch` and writes to mood-specific output dir.

### 5.3 Updated: dataset setup/import script

File: `scripts/setup_lora_datasets.py`

Changes:

1. Canonical mood templates aligned to 3 active moods:
	- `neo_tokyo`, `neutral_realistic`, `warm_sunny`
2. Added dataset import pipeline:
	- `--source_dir`
	- `--mood`
	- `--move` (optional move instead of copy)
3. Existing normalization retained:
	- deterministic renaming per mood
	- sidecar `.txt` alignment preservation
4. Auto-caption generation retained with mood trigger tokens.

### 5.4 Updated: interactive console with training hook logic

File: `scripts/lora_console.py`

Changes:

1. Canonical moods fixed to current training set.
2. Added `discover_moods()` from `data/lora_training` directories.
3. Added safe mood picker `pick_mood()`.
4. Added `train_hook(...)` that:
	- validates dataset before training
	- invokes mood training command
5. Menu updated:
	- train one mood (hooked)
	- train all moods sequentially (hooked)
6. Option 1 supports source-folder import and move/copy behavior.

### 5.5 Updated: app integration mood registry

File: `src/diffusion/style_lora_adapter.py`

Changes:

1. Active mood registry aligned to three moods only.
2. LoRA paths normalized to:
	- `models/loras/neo_tokyo/pytorch_lora_weights.safetensors`
	- `models/loras/neutral_realistic/pytorch_lora_weights.safetensors`
	- `models/loras/warm_sunny/pytorch_lora_weights.safetensors`

### 5.6 Updated: legacy mood manager consistency

File: `scripts/mood_manager.py`

Changes:

1. `neutral_realistic` path corrected.
2. Mood registry aligned with active three moods.
3. Logging cleanup for lint compatibility.

### 5.7 Updated: training configuration

File: `configs/training/lora_moods.yaml`

Key active fields:

- `base_model: runwayml/stable-diffusion-v1-5`
- `dataset_root: data/lora_training`
- `output_root: models/loras`
- `mood_name: neo_tokyo` (default; override with CLI)
- `resolution: 512`
- `train_batch_size: 1`
- `gradient_accumulation_steps: 4`
- `learning_rate: 1e-4`
- `max_train_steps: 1000`
- `rank: 16`
- `mixed_precision: fp16`
- `use_8bit_adam: true`
- `gradient_checkpointing: true`
- `dataloader_num_workers: 2`
- `checkpointing_steps: 200`
- `center_crop: true`
- `random_flip: false`


## 6) How to Run (Complete Procedures)

### 6.1 Fast start with console (recommended)

```powershell
conda activate vidcolor
python scripts/lora_console.py
```

Suggested flow in menu:

1. Option 1: setup/import dataset and captions
2. Option 2: validate dataset
3. Option 3: train one mood (hooked) OR option 4 train all moods (hooked)
4. Option 5: test generated LoRA

### 6.2 Manual CLI flow (single mood)

1. Import images into a mood folder:

```powershell
python scripts/setup_lora_datasets.py --source_dir "C:\path\to\images" --mood neo_tokyo
```

2. Validate:

```powershell
python scripts/validate_lora_dataset.py
```

3. Train:

```powershell
python scripts/train_lora_mood.py --mood neo_tokyo
```

4. Test:

```powershell
python scripts/test_lora.py --lora_path models/loras/neo_tokyo/pytorch_lora_weights.safetensors --trigger mdply_neo_tokyo --out_dir lora_tests
```

### 6.3 Manual CLI flow (all 3 moods)

```powershell
python scripts/validate_lora_dataset.py
python scripts/train_lora_mood.py --mood neo_tokyo
python scripts/train_lora_mood.py --mood neutral_realistic
python scripts/train_lora_mood.py --mood warm_sunny
```


## 7) Training Outputs and Artifacts

For each mood `<mood>`:

- final weights:
  - `models/loras/<mood>/pytorch_lora_weights.safetensors`
- checkpoint weights:
  - `models/loras/<mood>/checkpoint-<step>/pytorch_lora_weights.safetensors`
- run metadata:
  - `models/loras/<mood>/run_config.json`
- logs:
  - `models/loras/<mood>/logs/...`


## 8) Integration into Inference/App

The mood-to-LoRA mapping used by the app is in:

- `src/diffusion/style_lora_adapter.py`

At runtime, selecting a mood:

1. unloads previous LoRA if needed
2. loads corresponding mood LoRA from `models/loras/<mood>/...`
3. injects trigger token into prompt formatting
4. applies cross-attention scale from manager (`0.65`)


## 9) Troubleshooting and Operational Notes

### 9.1 Common training issues

1. CUDA OOM
- reduce `resolution` (for example 384)
- keep `train_batch_size=1`
- keep `gradient_checkpointing=true`
- reduce `rank` from 16 to 8

2. bitsandbytes unavailable on Windows
- set `use_8bit_adam: false`
- trainer automatically falls back to AdamW if import fails

3. Missing captions
- run `python scripts/setup_lora_datasets.py` to auto-generate
- verify with `python scripts/validate_lora_dataset.py`

4. Wrong mood paths at inference
- ensure `models/loras/<mood>/pytorch_lora_weights.safetensors` exists
- confirm mood keys in app registry match trained moods

### 9.2 Multi-root path caveat

This workspace has multiple roots. Keep LoRA outputs under repository-relative `models/loras` for reliable app loading. If you store files externally, update mood registry paths accordingly.

### 9.3 IDE unresolved-import warning vs runtime

If the editor uses a different Python interpreter, you may still see unresolved import warnings (`yaml`, `torch`, etc.). Runtime command execution in `vidcolor` is the source of truth for training.


## 10) Validation Checklist (Before Long Runs)

Run this checklist each time:

1. `conda activate vidcolor`
2. `python -m pip install -r requirements.txt`
3. `accelerate config` (done at least once)
4. `python scripts/validate_lora_dataset.py`
5. confirm each mood folder has images and captions
6. confirm free disk space for checkpoints/logs
7. start training via console option 3 or 4


## 11) Minimal Command Cheat Sheet

```powershell
# setup
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
conda activate vidcolor
python -m pip install -r requirements.txt
accelerate config

# import + normalize + caption
python scripts/setup_lora_datasets.py --source_dir "C:\path\to\images" --mood neo_tokyo

# validate
python scripts/validate_lora_dataset.py

# train one mood
python scripts/train_lora_mood.py --mood neo_tokyo

# train all moods
python scripts/train_lora_mood.py --mood neo_tokyo
python scripts/train_lora_mood.py --mood neutral_realistic
python scripts/train_lora_mood.py --mood warm_sunny

# test
python scripts/test_lora.py --lora_path models/loras/neo_tokyo/pytorch_lora_weights.safetensors --trigger mdply_neo_tokyo --out_dir lora_tests
```


## 12) Summary

PEFT LoRA is now fully integrated in-repo with:

1. local trainer implementation
2. wrapper-driven accelerative launch
3. mood-consistent data and app registry
4. interactive console hooks for one-click single/all-mood training
5. complete artifact path consistency under `models/loras`


## 13) April 23, 2026 Fix: Windows DataLoader Pickling Crash

### 13.1 Error observed during actual training

When running from `scripts/lora_console.py` (train sequence: `neo_tokyo`, `neutral_realistic`, `warm_sunny`), validation passed but training failed on the first mood with:

- `AttributeError: Can't pickle local object 'LocalLoraImageDataset.__init__.<locals>.<lambda>'`
- followed by upstream `subprocess.CalledProcessError` in:
	- `scripts/train_text_to_image_lora.py`
	- `scripts/train_lora_mood.py`
	- `scripts/lora_console.py`

### 13.2 Root cause

In `scripts/train_text_to_image_lora.py`, the transform pipeline used a local lambda as a no-op when `random_flip` was disabled:

- `transforms.Lambda(lambda img: img)`

On Windows, dataloader worker processes use spawn semantics and require picklable objects. A local lambda is not picklable, so worker startup crashed when `dataloader_num_workers` was greater than `0`.

### 13.3 Applied code change

The transform construction in `LocalLoraImageDataset` was updated to remove the lambda entirely:

1. Build `transform_ops` with `Resize` and crop.
2. Append `RandomHorizontalFlip()` only when `random_flip` is enabled.
3. Append `ToTensor()` and `Normalize()`.
4. Create `transforms.Compose(transform_ops)`.

This preserves training behavior while keeping the pipeline picklable on Windows.

### 13.4 Operational impact

- Existing config value `dataloader_num_workers: 2` is now valid for this code path on Windows.
- No change to dataset semantics or caption behavior.
- If worker issues reappear in a different environment, temporary fallback remains:
	- set `dataloader_num_workers: 0` in `configs/training/lora_moods.yaml`.

### 13.5 Additional non-blocking warnings seen

- Hugging Face Xet warning (`hf_xet` not installed): download fell back to regular HTTP.
- `--use_8bit_adam` warning: bitsandbytes unavailable, trainer fell back to AdamW as designed.

