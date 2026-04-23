# PEFT LoRA Training Guide (Windows, MoodPlay)

This project now includes a local PEFT LoRA trainer at:
- `scripts/train_text_to_image_lora.py`

You can train with only an image folder and optional captions.

## 1) Environment setup

From repo root (`pipeline`):

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
conda activate vidcolor
python -m pip install -r requirements.txt
```

If this is your first Accelerate run:

```powershell
accelerate config
```

Suggested choices:
- single machine
- 1 GPU
- fp16 mixed precision

## 2) Dataset layout expected by training

Default root:
- `data/lora_training`

Mood subfolders (already wired to app):
- `neo_tokyo`
- `neutral_realistic`
- `warm_sunny`

Each image can have a sidecar caption file with the same stem:
- `neo_tokyo_00001.png`
- `neo_tokyo_00001.txt`

If sidecar caption is missing, the trainer will fall back to image stem text.

## 3) Import your raw folder into a mood dataset

Copy raw images into `neo_tokyo` and auto-generate captions:

```powershell
python scripts/setup_lora_datasets.py --source_dir "C:\path\to\your\images" --mood neo_tokyo
```

Move instead of copy:

```powershell
python scripts/setup_lora_datasets.py --source_dir "C:\path\to\your\images" --mood neo_tokyo --move
```

Validate dataset:

```powershell
python scripts/validate_lora_dataset.py
```

You want `missing captions=0` for the target mood before training.

## 4) Training config

Edit:
- `configs/training/lora_moods.yaml`

Key fields:
- `mood_name`: dataset subfolder to train
- `max_train_steps`: total updates
- `learning_rate`: LoRA LR
- `rank`: LoRA rank
- `use_8bit_adam`: true if bitsandbytes works on your setup
- `gradient_checkpointing`: memory saver
- `checkpointing_steps`: save interval

Good starting values on 8-16GB VRAM:
- `resolution: 512`
- `train_batch_size: 1`
- `gradient_accumulation_steps: 4`
- `max_train_steps: 800` to `1500`
- `rank: 16`

## 5) Start PEFT LoRA training

From repo root:

```powershell
python scripts/train_lora_mood.py --mood neo_tokyo
```

Or override trainer path explicitly:

```powershell
python scripts/train_lora_mood.py --mood neo_tokyo --train_script scripts/train_text_to_image_lora.py
```

Outputs:
- final: `models/loras/<mood>/pytorch_lora_weights.safetensors`
- periodic checkpoints: `models/loras/<mood>/checkpoint-<step>/pytorch_lora_weights.safetensors`

## 6) Quick qualitative test

```powershell
python scripts/test_lora.py --lora_path models/loras/neo_tokyo/pytorch_lora_weights.safetensors --trigger mdply_neo_tokyo --out_dir lora_tests
```

## 7) Use in app

The app mood registry reads these paths/triggers from:
- `src/diffusion/style_lora_adapter.py`

Ensure your trained mood folder exists and trigger token matches captions.

## 8) Troubleshooting

### OOM / CUDA out of memory
- lower `resolution` to 384
- keep `train_batch_size: 1`
- keep `gradient_checkpointing: true`
- reduce `rank` from 16 to 8

### `bitsandbytes` error on Windows
- set `use_8bit_adam: false`
- trainer falls back to standard AdamW

### Slow dataloader on Windows
- set `dataloader_num_workers: 0` or `1`

### Missing LoRA file after training
- check `max_train_steps > 0`
- inspect console for save logs
- check `models/loras/<mood>/`
