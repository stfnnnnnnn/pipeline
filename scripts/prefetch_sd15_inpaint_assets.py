#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict


def import_or_raise(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing required module '{module_name}'. Install requirements in vidcolor first."
        ) from exc


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    yaml = import_or_raise("yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in config: {path}")
    return raw


def prefetch_repo(repo_id: str, cache_dir: str | None = None) -> str:
    huggingface_hub = import_or_raise("huggingface_hub")
    snapshot_download = huggingface_hub.snapshot_download

    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    return snapshot_download(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch SD1.5 inpainting and ControlNet model assets into local Hugging Face cache."
    )
    parser.add_argument(
        "--config",
        default="configs/model/sd15_controlnet.yaml",
        help="Path to SD1.5 ControlNet config file",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache directory (defaults to HF_HOME/cache).",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("Expected 'model' section in config.")

    base_model = model_cfg.get("base_inpaint_model")
    controlnet_model = model_cfg.get("controlnet_model")

    if not base_model or not controlnet_model:
        raise ValueError(
            "Config must define model.base_inpaint_model and model.controlnet_model."
        )

    print(f"[prefetch] base model: {base_model}")
    base_path = prefetch_repo(str(base_model), cache_dir=args.cache_dir)
    print(f"[prefetch] cached at: {base_path}")

    print(f"[prefetch] controlnet model: {controlnet_model}")
    controlnet_path = prefetch_repo(str(controlnet_model), cache_dir=args.cache_dir)
    print(f"[prefetch] cached at: {controlnet_path}")

    print("[prefetch] done")


if __name__ == "__main__":
    main()
