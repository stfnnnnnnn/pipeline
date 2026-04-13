"""
CoTracker wrapper for MoodPlay.
Tracks sampled points across all video frames and returns per-instance tracks.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

COTRACKER_IMPORT_ERROR = None
CoTrackerPredictor = None
try:
    from cotracker.predictor import CoTrackerPredictor as _CoTrackerPredictor  # type: ignore
    CoTrackerPredictor = _CoTrackerPredictor
except Exception as e:
    COTRACKER_IMPORT_ERROR = e
    try:
        from cotracker.models.core.cotracker.predictor import CoTrackerPredictor as _CoTrackerPredictor  # type: ignore
        CoTrackerPredictor = _CoTrackerPredictor
        COTRACKER_IMPORT_ERROR = None
    except Exception as e2:
        COTRACKER_IMPORT_ERROR = e2


class CoTrackerPersistentTracker:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        v2: Optional[bool] = None,
        offline: bool = True,
        window_len: Optional[int] = None,
        use_half: Optional[bool] = None,
        fallback_cpu_on_oom: bool = True,
        max_frames_per_chunk: Optional[int] = None,
    ) -> None:
        if CoTrackerPredictor is None:
            raise ImportError(f"CoTracker import failed. Install cotracker. Error: {COTRACKER_IMPORT_ERROR}")

        name = Path(checkpoint_path).name.lower()
        if v2 is None:
            if "cotracker2" in name or "v2" in name:
                v2 = True
            elif "cotracker3" in name or "v3" in name:
                v2 = False

        if use_half is None:
            use_half = device.startswith("cuda")

        self.model = self._build_predictor(
            checkpoint_path=checkpoint_path,
            offline=offline,
            v2=v2,
            window_len=window_len,
        )

        try:
            self.model = self.model.to(device)
        except Exception:
            pass

        self.device = device
        self.use_half = bool(use_half)
        self.fallback_cpu_on_oom = fallback_cpu_on_oom
        self.max_frames_per_chunk = max_frames_per_chunk

    @staticmethod
    def _build_predictor(
        checkpoint_path: str,
        offline: bool,
        v2: Optional[bool],
        window_len: Optional[int],
    ):
        sig = inspect.signature(CoTrackerPredictor)
        supports_window_len = "window_len" in sig.parameters

        def _infer_expected_window_len(err_msg: str) -> Optional[int]:
            if "time_emb" not in err_msg:
                return None
            matches = re.findall(r"torch\.Size\(\[1,\s*(\d+),\s*\d+\]\)", err_msg)
            if len(matches) >= 2:
                try:
                    # First match corresponds to checkpoint tensor shape in this error format.
                    return int(matches[0])
                except Exception:
                    return None
            return None

        def _construct(window_len_override: Optional[int]):
            kwargs = {}

            if "checkpoint" in sig.parameters:
                kwargs["checkpoint"] = checkpoint_path
            if "offline" in sig.parameters:
                kwargs["offline"] = bool(offline)
            if "v2" in sig.parameters and v2 is not None:
                kwargs["v2"] = bool(v2)
            if supports_window_len and window_len_override is not None:
                kwargs["window_len"] = int(window_len_override)

            try:
                if kwargs:
                    return CoTrackerPredictor(**kwargs)
                return CoTrackerPredictor(checkpoint_path)
            except TypeError:
                try:
                    return CoTrackerPredictor(checkpoint=checkpoint_path)
                except TypeError:
                    return CoTrackerPredictor(checkpoint_path)

        candidate_windows: List[Optional[int]] = []
        if supports_window_len:
            if window_len is not None:
                candidate_windows.append(int(window_len))
            # Try library default first when no explicit override is required.
            candidate_windows.append(None)
            for common in (60, 16, 8):
                if common not in candidate_windows:
                    candidate_windows.append(common)
        else:
            candidate_windows = [None]

        attempted: List[Optional[int]] = []
        idx = 0
        last_error: Optional[Exception] = None
        while idx < len(candidate_windows):
            win = candidate_windows[idx]
            attempted.append(win)
            try:
                return _construct(win)
            except RuntimeError as exc:
                last_error = exc
                if supports_window_len:
                    inferred = _infer_expected_window_len(str(exc))
                    if inferred is not None and inferred not in candidate_windows:
                        candidate_windows.insert(idx + 1, inferred)
            idx += 1

        attempted_str = ", ".join(["default" if x is None else str(x) for x in attempted])
        raise RuntimeError(
            "Failed to initialize CoTracker predictor for checkpoint "
            f"'{checkpoint_path}'. Tried window_len values: {attempted_str}. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def sample_points_from_mask(mask: np.ndarray, k: int = 24) -> np.ndarray:
        if mask is None:
            return np.zeros((0, 2), dtype=np.float32)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(xs) <= k:
            idx = np.arange(len(xs))
        else:
            idx = np.linspace(0, len(xs) - 1, num=k, dtype=int)
        return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

    def track_points(
        self,
        frames_bgr: List[np.ndarray],
        instance_seed_points: Dict[int, np.ndarray],
        instance_seed_frames: Optional[Dict[int, int]] = None,
    ) -> Dict[int, np.ndarray]:
        if len(frames_bgr) == 0:
            return {}

        if self.max_frames_per_chunk and len(frames_bgr) > self.max_frames_per_chunk:
            if instance_seed_frames and any(int(v) > 0 for v in instance_seed_frames.values()):
                return self._track_points_batch(frames_bgr, instance_seed_points, instance_seed_frames)
            return self._track_points_chunked(frames_bgr, instance_seed_points)

        return self._track_points_batch(frames_bgr, instance_seed_points, instance_seed_frames)

    def _track_points_batch(
        self,
        frames_bgr: List[np.ndarray],
        instance_seed_points: Dict[int, np.ndarray],
        instance_seed_frames: Optional[Dict[int, int]] = None,
    ) -> Dict[int, np.ndarray]:
        global_points: List[np.ndarray] = []
        global_t0: List[np.ndarray] = []
        slices: Dict[int, Tuple[int, int]] = {}
        start = 0
        for iid, pts in instance_seed_points.items():
            if pts.shape[0] == 0:
                continue
            global_points.append(pts)
            seed_t = 0
            if instance_seed_frames is not None:
                seed_t = int(instance_seed_frames.get(iid, 0))
            seed_t = max(0, min(seed_t, len(frames_bgr) - 1))
            global_t0.append(np.full((pts.shape[0], 1), float(seed_t), dtype=np.float32))
            end = start + pts.shape[0]
            slices[iid] = (start, end)
            start = end

        if start == 0:
            return {}

        video_np = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr], axis=0)
        video_t = torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0  # [1,T,3,H,W]

        pts0 = np.concatenate(global_points, axis=0)  # [N,2]
        t0 = np.concatenate(global_t0, axis=0) if global_t0 else np.zeros((pts0.shape[0], 1), dtype=np.float32)
        queries = np.concatenate([t0, pts0], axis=1)  # [N,3] => [t,x,y]
        queries_t = torch.from_numpy(queries).unsqueeze(0).float()  # [1,N,3]

        video_t = video_t.to(self.device)
        queries_t = queries_t.to(self.device)

        with torch.inference_mode():
            try:
                if self.use_half and self.device.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(video_t, queries=queries_t)
                else:
                    out = self.model(video_t, queries=queries_t)
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc) or not self.fallback_cpu_on_oom:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = self.model.to("cpu")
                self.device = "cpu"
                video_t = video_t.to("cpu")
                queries_t = queries_t.to("cpu")
                out = self.model(video_t, queries=queries_t)

        tracks = out[0] if isinstance(out, (tuple, list)) else out
        tracks_np = tracks.detach().cpu().numpy()[0]  # [T,N,2]

        tracked_by_instance: Dict[int, np.ndarray] = {}
        for iid, (s, e) in slices.items():
            tracked_by_instance[iid] = tracks_np[:, s:e, :]
        return tracked_by_instance

    def _track_points_chunked(
        self,
        frames_bgr: List[np.ndarray],
        instance_seed_points: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        max_len = int(self.max_frames_per_chunk or 0)
        if max_len <= 0:
            return self._track_points_batch(frames_bgr, instance_seed_points)

        current_points = {
            iid: pts.copy() for iid, pts in instance_seed_points.items() if pts is not None and pts.size > 0
        }
        tracked_chunks: Dict[int, List[np.ndarray]] = {}

        for start in range(0, len(frames_bgr), max_len):
            chunk_frames = frames_bgr[start : start + max_len]
            if not chunk_frames:
                continue
            chunk_tracks = self._track_points_batch(chunk_frames, current_points)
            if not chunk_tracks:
                break
            for iid, tr in chunk_tracks.items():
                tracked_chunks.setdefault(iid, []).append(tr)
                current_points[iid] = tr[-1]

        stitched: Dict[int, np.ndarray] = {}
        for iid, parts in tracked_chunks.items():
            if parts:
                stitched[iid] = np.concatenate(parts, axis=0)
        return stitched

    @staticmethod
    def _is_cuda_oom(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda" in msg and "memory" in msg