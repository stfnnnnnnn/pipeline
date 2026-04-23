#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

MOOD_TEMPLATES = {
    "neo_tokyo": {
        "trigger": "mdply_neo_tokyo",
        "caption": "a city street scene, cyberpunk, neon lights, rainy",
    },
    "neutral_realistic": {
        "trigger": "mdply_neutral_real",
        "caption": "a city street scene, realistic daylight, documentary photography",
    },
    "warm_sunny": {
        "trigger": "mdply_warm_sunny",
        "caption": "a city street scene, warm golden hour, sunlight",
    },
}

ROOT_DIR = Path("data/lora_training")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
NAME_DIGITS = 5


def ensure_layout() -> None:
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    for mood in MOOD_TEMPLATES:
        p = ROOT_DIR / mood
        p.mkdir(parents=True, exist_ok=True)
        print(f"[ok] {p}")


def iter_mood_dirs() -> list[Path]:
    if not ROOT_DIR.exists():
        return []
    return sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def normalize_filenames() -> None:
    """Normalize image filenames per mood folder and keep sidecar captions aligned."""
    for mood_dir in iter_mood_dirs():
        mood = mood_dir.name
        images = sorted(
            [p for p in mood_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS],
            key=lambda p: p.name.lower(),
        )
        if not images:
            continue

        rename_plan: list[tuple[Path, Path]] = []
        for idx, src in enumerate(images, start=1):
            target_name = f"{mood}_{idx:0{NAME_DIGITS}d}{src.suffix}"
            dst = mood_dir / target_name
            if src.name != dst.name:
                rename_plan.append((src, dst))

        if not rename_plan:
            continue

        temp_plan: list[tuple[Path, Path]] = []
        for i, (src, dst) in enumerate(rename_plan, start=1):
            tmp = mood_dir / f"__rename_tmp__{mood}_{i:0{NAME_DIGITS}d}{src.suffix}"
            while tmp.exists():
                i += 1
                tmp = mood_dir / f"__rename_tmp__{mood}_{i:0{NAME_DIGITS}d}{src.suffix}"

            src.rename(tmp)
            src_txt = src.with_suffix(".txt")
            if src_txt.exists():
                src_txt.rename(tmp.with_suffix(".txt"))
            temp_plan.append((tmp, dst))

        for tmp, dst in temp_plan:
            if dst.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {dst}")
            tmp.rename(dst)
            tmp_txt = tmp.with_suffix(".txt")
            if tmp_txt.exists():
                tmp_txt.rename(dst.with_suffix(".txt"))
            print(f"[rename] {mood_dir.name}/{tmp.name} -> {dst.name}")


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


def import_from_source(source_dir: Path, mood: str, move_files: bool) -> None:
    if mood not in MOOD_TEMPLATES:
        valid = ", ".join(MOOD_TEMPLATES.keys())
        raise ValueError(f"Invalid mood '{mood}'. Choose one of: {valid}")
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    target_dir = ROOT_DIR / mood
    target_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS],
        key=lambda p: p.name.lower(),
    )
    if not files:
        print(f"[warn] no supported images found in {source_dir}")
        return

    next_index = 1
    for src in files:
        while True:
            dst = target_dir / f"{mood}_import_{next_index:0{NAME_DIGITS}d}{src.suffix.lower()}"
            next_index += 1
            if not dst.exists():
                break

        if move_files:
            shutil.move(str(src), str(dst))
            print(f"[move] {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"[copy] {src} -> {dst}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", type=Path, default=None, help="Optional folder with raw images to import")
    ap.add_argument("--mood", type=str, default="neo_tokyo", help="Target mood folder when importing")
    ap.add_argument("--move", action="store_true", help="Move files from source_dir instead of copying")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=== MoodPlay LoRA Dataset Setup ===")
    ensure_layout()
    if args.source_dir is not None:
        import_from_source(args.source_dir, args.mood, args.move)
    normalize_filenames()
    write_sidecar_captions()
    print("Done.")