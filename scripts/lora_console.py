#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    while True:
        print("\n=== MoodPlay LoRA Console ===")
        print("1) Setup dataset folders + auto-caption")
        print("2) Validate dataset")
        print("3) Train LoRA mood")
        print("4) Test LoRA in isolation")
        print("5) Run LoRA unit test")
        print("0) Exit")
        choice = input("Select: ").strip()

        if choice == "1":
            run(["python", "scripts/setup_lora_datasets.py"])

        elif choice == "2":
            run(["python", "scripts/validate_lora_dataset.py"])

        elif choice == "3":
            mood = input("Mood (neo_tokyo/warm_sunny/pastel_filmic/neutral_realistic): ").strip()
            train_script = input("Path to train_text_to_image_lora.py: ").strip()
            cmd = ["python", "scripts/train_lora_mood.py", "--mood", mood]
            if train_script:
                cmd += ["--train_script", train_script]
            run(cmd)

        elif choice == "4":
            lora_path = input("LoRA .safetensors path: ").strip()
            trigger = input("Trigger token (e.g. mdply_neo_tokyo): ").strip()
            out_dir = input("Output dir [lora_tests]: ").strip() or "lora_tests"
            run([
                "python", "scripts/test_lora.py",
                "--lora_path", lora_path,
                "--trigger", trigger,
                "--out_dir", out_dir
            ])

        elif choice == "5":
            run(["pytest", "tests/unit/test_lora_manager.py", "-q"])

        elif choice == "0":
            print("Bye.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()