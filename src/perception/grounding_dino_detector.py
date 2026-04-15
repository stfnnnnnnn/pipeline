"""
GroundingDINO detector wrapper (runs in a separate conda env via subprocess).
"""
from __future__ import annotations

import json
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
	box_threshold: float
	text_threshold: float
	prompts: List[str]
	max_detections: int
	torch_lib_dir: Optional[str]
	cuda_bin_dir: Optional[str]


class GroundingDinoDetector:
	def __init__(self, cfg: GroundingDinoConfig) -> None:
		self.cfg = cfg

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

		conda_exe = str(self.cfg.conda_exe) if self.cfg.conda_exe else "conda"
		run_cmd = [
			conda_exe,
			"run",
			"-n",
			self.cfg.env_name,
			"python",
			self.cfg.script_path,
			"--image",
			image_path,
			"--config",
			self.cfg.model_config,
			"--weights",
			self.cfg.model_weights,
			"--prompt",
			prompt,
			"--box-threshold",
			str(self.cfg.box_threshold),
			"--text-threshold",
			str(self.cfg.text_threshold),
			"--max-detections",
			str(self.cfg.max_detections),
		]

		if conda_exe.lower().endswith(".bat"):
			cmd = ["cmd", "/c", *run_cmd]
		else:
			cmd = run_cmd

		if self.cfg.device:
			cmd += ["--device", self.cfg.device]
		if self.cfg.torch_lib_dir:
			cmd += ["--torch-lib-dir", str(self.cfg.torch_lib_dir)]
		if self.cfg.cuda_bin_dir:
			cmd += ["--cuda-bin-dir", str(self.cfg.cuda_bin_dir)]

		proc = subprocess.run(cmd, capture_output=True, text=True)
		if proc.returncode != 0:
			raise RuntimeError(f"GroundingDINO failed: {proc.stderr.strip()}")

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
				score = float(item.get("score", 1.0))
				if bbox.shape != (4,):
					continue
				out.append({"bbox_xyxy": bbox, "label": label, "score": score})
			except Exception:
				continue
		return out
