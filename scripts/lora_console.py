#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


DATASET_ROOT = Path("data/lora_training")
CANONICAL_MOODS = ("neo_tokyo", "neutral_realistic", "warm_sunny")


def discover_moods() -> list[str]:
    if not DATASET_ROOT.exists():
        return list(CANONICAL_MOODS)

    found = sorted([p.name for p in DATASET_ROOT.iterdir() if p.is_dir()])
    ordered = [m for m in CANONICAL_MOODS if m in found]
    return ordered if ordered else list(CANONICAL_MOODS)


def mood_prompt_text(moods: list[str]) -> str:
    return "/".join(moods)


def pick_mood(moods: list[str]) -> str | None:
    print("\nAvailable moods:")
    for idx, mood in enumerate(moods, start=1):
        print(f"{idx}) {mood}")
    raw = input("Select mood number (or Enter to cancel): ").strip()
    if not raw:
        return None
    try:
        i = int(raw)
    except ValueError:
        print("Invalid number.")
        return None
    if i < 1 or i > len(moods):
        print("Out of range.")
        return None
    return moods[i - 1]


def run(cmd: list[str]) -> None:
    print("\n[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_hook(mood: str, train_script: str | None = None) -> None:
    # Always validate dataset before a training launch.
    run(["python", "scripts/validate_lora_dataset.py"])

    cmd = ["python", "scripts/train_lora_mood.py", "--mood", mood]
    if train_script:
        cmd += ["--train_script", train_script]
    run(cmd)


def main():
    while True:
        moods = discover_moods()
        print("\n=== MoodPlay LoRA Console ===")
        print("1) Setup dataset folders + auto-caption")
        print("2) Validate dataset")
        print("3) Train one mood (hooked)")
        print("4) Train all moods (hooked)")
        print("5) Test LoRA in isolation")
        print("6) Run LoRA unit test")
        print("0) Exit")
        choice = input("Select: ").strip()

        if choice == "1":
            src = input("Optional source folder with images [Enter to skip]: ").strip()
            cmd = ["python", "scripts/setup_lora_datasets.py"]
            if src:
                mood = input(f"Target mood ({mood_prompt_text(moods)}) [neo_tokyo]: ").strip() or "neo_tokyo"
                move_files = input("Move instead of copy? [y/N]: ").strip().lower() == "y"
                cmd += ["--source_dir", src, "--mood", mood]
                if move_files:
                    cmd.append("--move")
            run(cmd)

        elif choice == "2":
            run(["python", "scripts/validate_lora_dataset.py"])

        elif choice == "3":
            mood = pick_mood(moods)
            if not mood:
                continue
            train_script = input("Train script path [scripts/train_text_to_image_lora.py]: ").strip() or None
            train_hook(mood, train_script)

        elif choice == "4":
            train_script = input("Train script path [scripts/train_text_to_image_lora.py]: ").strip() or None
            print("\nTraining sequence:", ", ".join(moods))
            confirm = input("Proceed? [y/N]: ").strip().lower() == "y"
            if not confirm:
                continue
            run(["python", "scripts/validate_lora_dataset.py"])
            for mood in moods:
                cmd = ["python", "scripts/train_lora_mood.py", "--mood", mood]
                if train_script:
                    cmd += ["--train_script", train_script]
                run(cmd)

        elif choice == "5":
            lora_path = input("LoRA .safetensors path: ").strip()
            trigger = input("Trigger token (e.g. mdply_neo_tokyo): ").strip()
            out_dir = input("Output dir [lora_tests]: ").strip() or "lora_tests"
            run([
                "python", "scripts/test_lora.py",
                "--lora_path", lora_path,
                "--trigger", trigger,
                "--out_dir", out_dir
            ])

        elif choice == "6":
            run(["pytest", "tests/unit/test_lora_manager.py", "-q"])

        elif choice == "0":
            print("Bye.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()