#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

MOOD_TEMPLATES = {
    "warm_sunny": {
        "trigger": "mdply_warm_sunny",
        "caption": "a city street scene, warm golden hour, sunlight",
    },
    "neo_tokyo": {
        "trigger": "mdply_neo_tokyo",
        "caption": "a city street scene, cyberpunk, neon lights, rainy",
    },
    "sunday_blues": {
        "trigger": "mdply_sunday_blues",
        "caption": "a city street scene, moody blue cinematic palette, low contrast",
    },
    "neutral_realistic": {
        "trigger": "mdply_neutral_real",
        "caption": "a city street scene, realistic daylight, documentary photography",
    },
}

ROOT_DIR = Path("data/lora_training")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def ensure_layout() -> None:
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    for mood in MOOD_TEMPLATES:
        p = ROOT_DIR / mood
        p.mkdir(parents=True, exist_ok=True)
        print(f"[ok] {p}")


def write_sidecar_captions() -> None:
    for mood, spec in MOOD_TEMPLATES.items():
        p = ROOT_DIR / mood
        if not p.exists():
            continue
        for img in p.iterdir():
            if img.suffix.lower() not in VALID_EXTS:
                continue
            txt = img.with_suffix(".txt")
            if txt.exists():
                continue
            txt.write_text(f"{spec['trigger']}, {spec['caption']}\n", encoding="utf-8")
            print(f"[caption] {txt}")


if __name__ == "__main__":
    print("=== MoodPlay LoRA Dataset Setup ===")
    ensure_layout()
    write_sidecar_captions()
    print("Done.")