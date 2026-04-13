"""
GroundingDINO detector wrapper (runs in a separate conda env via subprocess).
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml


def _load_yaml(path: str) -> Dict:
	p = Path(path)
	if not p.exists():
		return {}
	with open(p, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	return data if isinstance(data, dict) else {}


@dataclass
class GroundingDinoConfig:
	enabled: bool
	env_name: str
	conda_exe: Optional[str]
	script_path: str
	model_config: str
	model_weights: str
	device: str
	allow_cpu_fallback: bool
	box_threshold: float
	text_threshold: float
	prompts: List[str]
	max_detections: int
	torch_lib_dir: Optional[str]
	cuda_bin_dir: Optional[str]


class GroundingDinoDetector:
	def __init__(self, cfg: GroundingDinoConfig) -> None:
		self.cfg = cfg
		self._repo_root = Path(__file__).resolve().parents[2]
		self._resolved_conda_exe = self._resolve_conda_exe(self.cfg.conda_exe)
		self._resolved_torch_lib_dir = self._resolve_torch_lib_dir(self.cfg.torch_lib_dir)
		self._resolved_cuda_bin_dir = self._resolve_cuda_bin_dir(self.cfg.cuda_bin_dir)

	def _resolve_repo_path(self, raw: str) -> str:
		p = Path(str(raw))
		if p.is_absolute():
			return str(p)
		return str(self._repo_root / p)

	@staticmethod
	def _resolve_conda_exe(configured: Optional[str]) -> str:
		if configured:
			candidate = Path(str(configured))
			if candidate.exists():
				return str(candidate)

		from_env = os.environ.get("CONDA_EXE")
		if from_env:
			candidate = Path(from_env)
			if candidate.exists():
				if candidate.suffix.lower() == ".exe":
					bat = candidate.parent.parent / "condabin" / "conda.bat"
					if bat.exists():
						return str(bat)
				return str(candidate)

		fallbacks = [
			Path(r"C:\ProgramData\miniconda3\condabin\conda.bat"),
			Path(r"C:\ProgramData\anaconda3\condabin\conda.bat"),
			Path.home() / "miniconda3" / "condabin" / "conda.bat",
			Path.home() / "anaconda3" / "condabin" / "conda.bat",
		]
		for candidate in fallbacks:
			if candidate.exists():
				return str(candidate)
		return "conda"

	@staticmethod
	def _existing_path_or_none(path: Optional[str]) -> Optional[str]:
		if not path:
			return None
		p = Path(str(path))
		return str(p) if p.exists() else None

	def _resolve_torch_lib_dir(self, configured: Optional[str]) -> Optional[str]:
		existing = self._existing_path_or_none(configured)
		if existing:
			return existing

		conda_p = Path(self._resolved_conda_exe)
		base_prefix = conda_p.parent.parent if conda_p.name.lower().startswith("conda") else None
		candidates: List[Path] = []
		if base_prefix is not None:
			candidates.append(base_prefix / "envs" / self.cfg.env_name / "Lib" / "site-packages" / "torch" / "lib")

		candidates += [
			Path(r"C:\ProgramData\miniconda3") / "envs" / self.cfg.env_name / "Lib" / "site-packages" / "torch" / "lib",
			Path(r"C:\ProgramData\anaconda3") / "envs" / self.cfg.env_name / "Lib" / "site-packages" / "torch" / "lib",
			Path.home() / "miniconda3" / "envs" / self.cfg.env_name / "Lib" / "site-packages" / "torch" / "lib",
			Path.home() / "anaconda3" / "envs" / self.cfg.env_name / "Lib" / "site-packages" / "torch" / "lib",
		]
		for candidate in candidates:
			if candidate.exists():
				return str(candidate)
		return None

	def _resolve_cuda_bin_dir(self, configured: Optional[str]) -> Optional[str]:
		existing = self._existing_path_or_none(configured)
		if existing:
			return existing

		cuda_env_keys = [k for k in os.environ.keys() if k.upper().startswith("CUDA_PATH")]
		for key in sorted(cuda_env_keys):
			base = os.environ.get(key)
			if not base:
				continue
			candidate = Path(base) / "bin"
			if candidate.exists():
				return str(candidate)

		cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
		if cuda_root.exists():
			versions = sorted([p for p in cuda_root.glob("v*") if p.is_dir()])
			for v in reversed(versions):
				candidate = v / "bin"
				if candidate.exists():
					return str(candidate)
		return None

	@staticmethod
	def _is_cuda_incompatible_error(text: str) -> bool:
		msg = (text or "").lower()
		return (
			"no kernel image is available for execution on the device" in msg
			or "not compatible with the current pytorch installation" in msg
			or ("sm_" in msg and "not compatible" in msg)
		)

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
			allow_cpu_fallback=bool(data.get("allow_cpu_fallback", False)),
			box_threshold=float(data.get("box_threshold", 0.35)),
			text_threshold=float(data.get("text_threshold", 0.25)),
			prompts=prompts,
			max_detections=int(data.get("max_detections", 50)),
			torch_lib_dir=data.get("torch_lib_dir", None),
			cuda_bin_dir=data.get("cuda_bin_dir", None),
		)
		return GroundingDinoDetector(config)

	def detect(self, image_path: str) -> List[Dict]:
		if not self.cfg.prompts:
			return []

		prompt = ". ".join(self.cfg.prompts)
		if not prompt.endswith("."):
			prompt += "."

		conda_exe = self._resolved_conda_exe
		script_path = self._resolve_repo_path(self.cfg.script_path)
		model_config = self._resolve_repo_path(self.cfg.model_config)
		model_weights = self._resolve_repo_path(self.cfg.model_weights)
		img_path = self._resolve_repo_path(image_path)
		torch_lib_dir = self._resolved_torch_lib_dir
		cuda_bin_dir = self._resolved_cuda_bin_dir

		for label, p in (("script_path", script_path), ("model_config", model_config), ("model_weights", model_weights), ("image", img_path)):
			if not Path(p).exists():
				raise RuntimeError(f"GroundingDINO config error: {label} does not exist: {p}")

		run_cmd = [
			conda_exe,
			"run",
			"-n",
			self.cfg.env_name,
			"python",
			script_path,
			"--image",
			img_path,
			"--config",
			model_config,
			"--weights",
			model_weights,
			"--prompt",
			prompt,
			"--box-threshold",
			str(self.cfg.box_threshold),
			"--text-threshold",
			str(self.cfg.text_threshold),
			"--max-detections",
			str(self.cfg.max_detections),
		]

		def _build_cmd(device_value: Optional[str]) -> List[str]:
			if conda_exe.lower().endswith(".bat"):
				cmd = ["cmd", "/c", *run_cmd]
			else:
				cmd = list(run_cmd)

			if device_value:
				cmd += ["--device", str(device_value)]
			if torch_lib_dir:
				cmd += ["--torch-lib-dir", torch_lib_dir]
			if cuda_bin_dir:
				cmd += ["--cuda-bin-dir", cuda_bin_dir]
			return cmd

		device_requested = str(self.cfg.device or "")
		proc = subprocess.run(_build_cmd(device_requested), capture_output=True, text=True, cwd=str(self._repo_root))
		if proc.returncode != 0:
			err = proc.stderr.strip() or proc.stdout.strip()
			custom_ops_missing = (
				"failed to load custom c++ ops" in err.lower()
				or "nameerror: name '_c' is not defined" in err.lower()
			)
			if custom_ops_missing:
				raise RuntimeError(
					"GroundingDINO custom C++ ops are missing in env "
					f"'{self.cfg.env_name}'. Reinstall GroundingDINO from source after installing a "
					"GPU-compatible torch wheel (cu121/cu128), then verify with: "
					"python -c \"import groundingdino._C as C; print(hasattr(C, 'ms_deform_attn_forward'))\""
				)
			# Optional CPU fallback when CUDA kernels are incompatible in this env.
			if (
				device_requested.startswith("cuda")
				and self.cfg.allow_cpu_fallback
				and self._is_cuda_incompatible_error(err)
			):
				proc = subprocess.run(_build_cmd("cpu"), capture_output=True, text=True, cwd=str(self._repo_root))
				if proc.returncode != 0:
					err2 = proc.stderr.strip() or proc.stdout.strip()
					raise RuntimeError(f"GroundingDINO failed on CUDA and CPU fallback: {err2}")
			elif device_requested.startswith("cuda") and self._is_cuda_incompatible_error(err):
				raise RuntimeError(
					"GroundingDINO CUDA build is incompatible with this GPU. "
					"Reinstall torch/torchvision/torchaudio in env "
					f"'{self.cfg.env_name}' using cu128 or nightly cu128 wheels. "
					"Set allow_cpu_fallback=true in grounding_dino.yaml if CPU fallback is acceptable."
				)
			else:
				raise RuntimeError(f"GroundingDINO failed: {err}")

		lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
		if not lines:
			return []
		try:
			data = json.loads(lines[-1])
		except json.JSONDecodeError as exc:
			raise RuntimeError(f"GroundingDINO output parse error: {exc}")

		out: List[Dict] = []
		for item in data:
			try:
				bbox = np.array(item["bbox"], dtype=np.float32)
				label = str(item.get("label", "object")).strip()
				if bbox.shape != (4,):
					continue
				out.append({"bbox_xyxy": bbox, "label": label})
			except Exception:
				continue
		return out
