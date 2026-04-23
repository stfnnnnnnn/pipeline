#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import yaml


def main():
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training/lora_moods.yaml")
    ap.add_argument("--train_script", default="scripts/train_text_to_image_lora.py")
    ap.add_argument("--mood", default=None, help="Override mood_name from config")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    train_script_path = Path(args.train_script)
    if not train_script_path.is_absolute():
        train_script_path = repo_root / train_script_path
    if not train_script_path.exists():
        raise FileNotFoundError(
            f"LoRA train script not found: {train_script_path}. "
            "Use --train_script with a valid path."
        )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mood = args.mood or cfg["mood_name"]

    data_root = Path(cfg.get("dataset_root", "data/lora_training"))
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    data_dir = data_root / mood
    out_root = Path(cfg["output_root"])
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_dir = out_root / mood

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate", "launch",
        "--mixed_precision", str(cfg["mixed_precision"]),
        str(train_script_path),
        "--pretrained_model_name_or_path", str(cfg["base_model"]),
        "--train_data_dir", str(data_dir),
        "--output_dir", str(out_dir),
        "--resolution", str(cfg["resolution"]),
        "--train_batch_size", str(cfg["train_batch_size"]),
        "--gradient_accumulation_steps", str(cfg["gradient_accumulation_steps"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--lr_scheduler", "constant",
        "--max_train_steps", str(cfg["max_train_steps"]),
        "--rank", str(cfg["rank"]),
        "--seed", str(cfg["seed"]),
        "--dataloader_num_workers", str(cfg.get("dataloader_num_workers", 0)),
        "--checkpointing_steps", str(cfg.get("checkpointing_steps", 200)),
        "--lr_warmup_steps", str(cfg.get("lr_warmup_steps", 0)),
    ]

    if cfg.get("use_8bit_adam", False):
        cmd.append("--use_8bit_adam")
    if cfg.get("gradient_checkpointing", False):
        cmd.append("--gradient_checkpointing")
    if cfg.get("center_crop", False):
        cmd.append("--center_crop")
    if cfg.get("random_flip", False):
        cmd.append("--random_flip")

    print("Running:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()