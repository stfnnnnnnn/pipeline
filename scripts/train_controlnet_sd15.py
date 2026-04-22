#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/training/controlnet_sd15_poc.yaml"))
    ap.add_argument("--train-script", type=str, default="train_controlnet.py")  # from diffusers/examples/controlnet
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    cmd = [
        "python", args.train_script,
        "--pretrained_model_name_or_path", cfg["model_name_or_path"],
        "--output_dir", cfg["output_dir"],
        "--train_data_dir", ".",  # not used by JSONL mode, kept for compatibility
        "--resolution", str(cfg["resolution"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--max_train_steps", str(cfg["max_train_steps"]),
        "--train_batch_size", str(cfg["train_batch_size"]),
        "--gradient_accumulation_steps", str(cfg["gradient_accumulation_steps"]),
        "--checkpointing_steps", str(cfg["checkpointing_steps"]),
        "--validation_steps", str(cfg["validation_steps"]),
        "--mixed_precision", str(cfg["mixed_precision"]),
        "--seed", str(cfg["seed"]),
    ]

    # custom args supported in many forks; harmless if script supports them
    if cfg.get("use_8bit_adam", False):
        cmd.append("--use_8bit_adam")
    if cfg.get("gradient_checkpointing", False):
        cmd.append("--gradient_checkpointing")

    # JSONL-based datasets are typically passed via --train_data_dir with metadata in folder
    # If your script supports explicit metadata:
    # cmd += ["--dataset_name", "json", "--dataset_config_name", str(cfg["dataset_jsonl"])]
    # For many local scripts, place train.jsonl in data root and they auto-read metadata.
    print("Running:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()