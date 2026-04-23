#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training/lora_moods.yaml")
    ap.add_argument("--train_script", default="train_text_to_image_lora.py")
    ap.add_argument("--mood", default=None, help="Override mood_name from config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mood = args.mood or cfg["mood_name"]

    data_root = Path(cfg.get("dataset_root", "data/lora_training"))
    data_dir = data_root / mood
    out_dir = Path(cfg["output_root"]) / mood

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate", "launch",
        "--mixed_precision", str(cfg["mixed_precision"]),
        args.train_script,
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
    ]

    if cfg.get("use_8bit_adam", False):
        cmd.append("--use_8bit_adam")
    if cfg.get("gradient_checkpointing", False):
        cmd.append("--gradient_checkpointing")

    print("Running:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()