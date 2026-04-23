#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path("data/lora_training")
VALID = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    if not ROOT.exists():
        print(f"[error] Missing dataset root: {ROOT}")
        return

    total_images = 0
    total_missing_caps = 0

    for mood_dir in sorted([p for p in ROOT.iterdir() if p.is_dir()]):
        imgs = [p for p in mood_dir.iterdir() if p.suffix.lower() in VALID]
        missing = [p for p in imgs if not p.with_suffix(".txt").exists()]
        total_images += len(imgs)
        total_missing_caps += len(missing)
        print(
            f"{mood_dir.name:18} images={len(imgs):4d} "
            f"captions={len(imgs)-len(missing):4d} missing_caps={len(missing):4d}"
        )

    print("-" * 72)
    print(f"TOTAL images={total_images}, missing captions={total_missing_caps}")


if __name__ == "__main__":
    main()