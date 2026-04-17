"""
XMem VOS wrapper used to propagate keyframe instance masks across a video.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch


@dataclass
class XMemRuntimeConfig:
    enabled: bool = False
    repo_path: str = "XMem"
    model_path: str = "XMem/saves/XMem.pth"
    device: str = "cuda"
    mem_every: int = 5
    deep_update_every: int = -1
    max_mid_term_frames: int = 10
    min_mid_term_frames: int = 5
    max_long_term_elements: int = 10000
    num_prototypes: int = 128
    top_k: int = 30
    disable_long_term: bool = False


class XMemMaskPropagator:
    def __init__(self, cfg: XMemRuntimeConfig) -> None:
        self.cfg = cfg

    def is_ready(self) -> bool:
        if not bool(self.cfg.enabled):
            return False
        model = Path(self.cfg.model_path)
        repo = Path(self.cfg.repo_path)
        return model.exists() and repo.exists()

    def propagate(
        self,
        frames_bgr: List[np.ndarray],
        keyframe_label_maps: Dict[int, np.ndarray],
        all_instance_ids: Optional[List[int]] = None,
    ) -> Dict[int, np.ndarray]:
        if not frames_bgr:
            return {}
        if not self.is_ready() or not keyframe_label_maps:
            return {}

        xmem_network, xmem_core = self._load_xmem_modules(Path(self.cfg.repo_path))
        use_cuda = self.cfg.device.startswith("cuda") and torch.cuda.is_available()
        device = self.cfg.device if use_cuda else "cpu"
        map_location = None if use_cuda else torch.device("cpu")

        config = {
            "mem_every": int(self.cfg.mem_every),
            "deep_update_every": int(self.cfg.deep_update_every),
            "enable_long_term": not bool(self.cfg.disable_long_term),
            "max_mid_term_frames": int(self.cfg.max_mid_term_frames),
            "min_mid_term_frames": int(self.cfg.min_mid_term_frames),
            "max_long_term_elements": int(self.cfg.max_long_term_elements),
            "num_prototypes": int(self.cfg.num_prototypes),
            "top_k": int(self.cfg.top_k),
            "disable_long_term": bool(self.cfg.disable_long_term),
        }

        network = xmem_network(config, str(self.cfg.model_path), map_location=map_location).to(device).eval()
        processor = xmem_core(network, config=config)

        labels = self._collect_labels(keyframe_label_maps, all_instance_ids)
        if not labels:
            return {}
        processor.set_all_labels(labels)

        out: Dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for ti, frame in enumerate(frames_bgr):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0

                mask_t = None
                valid_labels = None
                if ti in keyframe_label_maps:
                    label_map = keyframe_label_maps[ti].astype(np.int32)
                    one_hot = np.stack([(label_map == lid).astype(np.float32) for lid in labels], axis=0)
                    mask_t = torch.from_numpy(one_hot).to(device)
                    valid_labels = [lid for lid in labels if int(np.any(label_map == lid)) == 1]

                with torch.cuda.amp.autocast(enabled=use_cuda):
                    prob = processor.step(rgb_t, mask_t, valid_labels, end=(ti == len(frames_bgr) - 1))

                if prob is None:
                    continue

                idx_map = torch.max(prob, dim=0).indices.detach().cpu().numpy().astype(np.int32)
                lbl = np.zeros_like(idx_map, dtype=np.int32)
                for idx, lid in enumerate(labels, start=1):
                    lbl[idx_map == idx] = int(lid)
                out[ti] = lbl

        return out

    @staticmethod
    def _collect_labels(keyframe_label_maps: Dict[int, np.ndarray], all_instance_ids: Optional[List[int]]) -> List[int]:
        if all_instance_ids:
            return [int(x) for x in sorted(set(all_instance_ids))]

        labels = set()
        for m in keyframe_label_maps.values():
            u = np.unique(m)
            labels.update(int(x) for x in u if int(x) > 0)
        return sorted(labels)

    @staticmethod
    def _load_xmem_modules(repo_path: Path):
        if not repo_path.exists():
            raise RuntimeError(f"XMem repo path not found: {repo_path}")

        repo_str = str(repo_path.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            network_mod = importlib.import_module("model.network")
            infer_mod = importlib.import_module("inference.inference_core")
            return network_mod.XMem, infer_mod.InferenceCore
        except Exception as exc:
            raise RuntimeError(f"Failed to import XMem modules from {repo_path}: {exc}") from exc
