"""
GroundingDINO detector wrapper (runs in a separate conda env via subprocess).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _resolve_repo_path(path_value: str, *, label: str, must_exist: bool) -> str:
    p = Path(str(path_value).strip()).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()

    if must_exist and not p.exists():
        raise RuntimeError(f"GroundingDINO {label} not found: {p}")

    return str(p)


def _resolve_optional_dir(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    p = Path(str(path_value).strip()).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    else:
        p = p.resolve()
    if p.exists() and p.is_dir():
        return str(p)
    return None


def _resolve_conda_exe(configured: Optional[str]) -> str:
    candidates: List[str] = []

    if configured and str(configured).strip():
        candidates.append(str(configured).strip())

    env_conda = str(os.environ.get("CONDA_EXE", "")).strip()
    if env_conda:
        candidates.append(env_conda)

    for name in ("conda", "conda.exe", "conda.bat"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    user_home = Path.home()
    for p in (
        user_home / "miniconda3" / "condabin" / "conda.bat",
        user_home / "anaconda3" / "condabin" / "conda.bat",
        Path("C:/ProgramData/miniconda3/condabin/conda.bat"),
        Path("C:/ProgramData/anaconda3/condabin/conda.bat"),
    ):
        candidates.append(str(p))

    seen = set()
    ordered: List[str] = []
    for cand in candidates:
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cand)

    for cand in ordered:
        if shutil.which(cand):
            return cand
        p = Path(cand).expanduser()
        if p.exists():
            return str(p)

    if configured and str(configured).strip() and Path(str(configured).strip()).suffix.lower() in {".bat", ".exe"}:
        raise RuntimeError(
            f"GroundingDINO conda executable not found: {configured}. "
            "Update 'conda_exe' in configs/perception/grounding_dino.yaml or set CONDA_EXE."
        )

    return "conda"


def _is_cuda_runtime_error(error_text: str) -> bool:
    text = str(error_text or "").lower()
    markers = (
        "torch not compiled with cuda enabled",
        "cuda is not available",
        "cuda unavailable",
        "invalid device ordinal",
        "no cuda gpus are available",
        "found no nvidia driver",
        "cuda driver version is insufficient",
    )
    return any(m in text for m in markers)


@dataclass
class GroundingDinoConfig:
    enabled: bool
    env_name: str
    conda_exe: Optional[str]
    script_path: str
    model_config: str
    model_weights: str
    device: str
    box_threshold: float
    text_threshold: float
    prompts: List[str]
    environment_presets: Dict[str, Dict]
    default_preset: Optional[str]
    default_clutter: str
    max_detections: int
    torch_lib_dir: Optional[str]
    cuda_bin_dir: Optional[str]


class GroundingDinoDetector:
    def __init__(self, cfg: GroundingDinoConfig) -> None:
        self.cfg = cfg
        self._force_cpu = False

    @staticmethod
    def from_config(config_path: str, cfg: Optional[Dict] = None) -> Optional["GroundingDinoDetector"]:
        data = cfg if cfg is not None else _load_yaml(config_path)
        if not data or not bool(data.get("enabled", False)):
            return None

        prompts = data.get("prompts", [])
        if not isinstance(prompts, list):
            prompts = []
        prompts = [str(p).strip() for p in prompts if str(p).strip()]

        config = GroundingDinoConfig(
            enabled=True,
            env_name=str(data.get("env_name", "gdino310")),
            conda_exe=data.get("conda_exe", None),
            script_path=str(data.get("script_path", "scripts/gdino_worker.py")),
            model_config=str(data.get("model_config", "")),
            model_weights=str(data.get("model_weights", "")),
            device=str(data.get("device", "cuda")),
            box_threshold=float(data.get("box_threshold", 0.35)),
            text_threshold=float(data.get("text_threshold", 0.25)),
            prompts=prompts,
            environment_presets=data.get("environment_presets", {}) if isinstance(data.get("environment_presets", {}), dict) else {},
            default_preset=str(data.get("default_preset", "")).strip() or None,
            default_clutter=str(data.get("default_clutter", "medium")).strip().lower() or "medium",
            max_detections=int(data.get("max_detections", 50)),
            torch_lib_dir=data.get("torch_lib_dir", None),
            cuda_bin_dir=data.get("cuda_bin_dir", None),
        )
        return GroundingDinoDetector(config)

    def detect(
        self,
        image_path: str,
        *,
        prompts_override: Optional[List[str]] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        preset: Optional[str] = None,
        clutter_level: Optional[str] = None,
    ) -> List[Dict]:
        prompts = self._resolve_prompts(prompts_override=prompts_override, preset=preset)
        if not prompts:
            return []

        resolved_box = self._resolve_threshold(
            key="box_threshold",
            fallback=float(self.cfg.box_threshold),
            override=box_threshold,
            preset=preset,
            clutter_level=clutter_level,
        )
        resolved_text = self._resolve_threshold(
            key="text_threshold",
            fallback=float(self.cfg.text_threshold),
            override=text_threshold,
            preset=preset,
            clutter_level=clutter_level,
        )
        resolved_max = int(max_detections if max_detections is not None else self.cfg.max_detections)

        prompt = ". ".join(prompts)
        if not prompt.endswith("."):
            prompt += "."

        conda_exe = _resolve_conda_exe(self.cfg.conda_exe)
        resolved_script = _resolve_repo_path(self.cfg.script_path, label="worker script", must_exist=True)
        resolved_model_config = _resolve_repo_path(self.cfg.model_config, label="model config", must_exist=True)
        resolved_model_weights = _resolve_repo_path(self.cfg.model_weights, label="model weights", must_exist=True)
        resolved_image = _resolve_repo_path(image_path, label="input image", must_exist=True)
        torch_lib_dir = _resolve_optional_dir(self.cfg.torch_lib_dir)
        cuda_bin_dir = _resolve_optional_dir(self.cfg.cuda_bin_dir)

        run_args = [
            "run",
            "-n",
            self.cfg.env_name,
            "python",
            resolved_script,
            "--image",
            resolved_image,
            "--config",
            resolved_model_config,
            "--weights",
            resolved_model_weights,
            "--prompt",
            prompt,
            "--box-threshold",
            str(resolved_box),
            "--text-threshold",
            str(resolved_text),
            "--max-detections",
            str(resolved_max),
        ]

        requested_device = str(self.cfg.device or "cpu").strip() or "cpu"
        run_device = "cpu" if (self._force_cpu and requested_device.lower().startswith("cuda")) else requested_device

        def _invoke_worker(device_arg: str) -> subprocess.CompletedProcess:
            if Path(conda_exe).suffix.lower() == ".bat":
                cmd = ["cmd", "/c", conda_exe, *run_args]
            else:
                cmd = [conda_exe, *run_args]

            if device_arg:
                cmd += ["--device", device_arg]
            if torch_lib_dir:
                cmd += ["--torch-lib-dir", torch_lib_dir]
            if cuda_bin_dir:
                cmd += ["--cuda-bin-dir", cuda_bin_dir]

            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(PROJECT_ROOT),
            )

        try:
            proc = _invoke_worker(run_device)
        except OSError as exc:
            raise RuntimeError(f"GroundingDINO launch failed: {exc}") from exc

        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
            if run_device.lower().startswith("cuda") and _is_cuda_runtime_error(err):
                try:
                    cpu_proc = _invoke_worker("cpu")
                except OSError as exc:
                    raise RuntimeError(f"GroundingDINO launch failed: {exc}") from exc

                if cpu_proc.returncode == 0:
                    self._force_cpu = True
                    proc = cpu_proc
                else:
                    cpu_err = cpu_proc.stderr.strip() or cpu_proc.stdout.strip() or f"exit code {cpu_proc.returncode}"
                    raise RuntimeError(
                        "GroundingDINO failed on CUDA, then failed on CPU fallback. "
                        f"CUDA error: {err} | CPU error: {cpu_err}"
                    )
            else:
                raise RuntimeError(f"GroundingDINO failed: {err}")

        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            return []
        try:
            data = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"GroundingDINO output parse error: {exc}") from exc

        out: List[Dict] = []
        for item in data:
            try:
                bbox = np.array(item["bbox"], dtype=np.float32)
                label = str(item.get("label", "object")).strip()
                score = float(item.get("score", 1.0))
                if bbox.shape != (4,):
                    continue
                out.append({"bbox_xyxy": bbox, "label": label, "score": score})
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _resolve_prompts(self, prompts_override: Optional[List[str]], preset: Optional[str]) -> List[str]:
        if prompts_override:
            return self._normalize_prompts(prompts_override)

        preset_name = str(preset).strip() if preset else self.cfg.default_preset
        if preset_name and preset_name in self.cfg.environment_presets:
            entry = self.cfg.environment_presets[preset_name]
            if isinstance(entry, dict):
                preset_prompts = entry.get("prompts", [])
                if isinstance(preset_prompts, list):
                    p = self._normalize_prompts(preset_prompts)
                    if p:
                        return p

        return self._normalize_prompts(self.cfg.prompts)

    def _resolve_threshold(
        self,
        *,
        key: str,
        fallback: float,
        override: Optional[float],
        preset: Optional[str],
        clutter_level: Optional[str],
    ) -> float:
        if override is not None:
            return float(override)

        preset_name = str(preset).strip() if preset else self.cfg.default_preset
        clutter = str(clutter_level or self.cfg.default_clutter).strip().lower() or "medium"

        if preset_name and preset_name in self.cfg.environment_presets:
            entry = self.cfg.environment_presets.get(preset_name, {})
            if isinstance(entry, dict):
                v = entry.get(key, None)
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict):
                    if clutter in v and isinstance(v[clutter], (int, float)):
                        return float(v[clutter])
                    if "medium" in v and isinstance(v["medium"], (int, float)):
                        return float(v["medium"])

        return float(fallback)

    @staticmethod
    def _normalize_prompts(prompts: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for p in prompts:
            s = " ".join(str(p).strip().lower().split())
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out
