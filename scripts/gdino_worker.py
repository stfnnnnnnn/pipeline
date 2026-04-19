"""
GroundingDINO worker (runs inside the gdino310 env).
Outputs JSON list of detections to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import torch

try:
	import torchvision.ops as ops
except Exception:
	ops = None


def _add_dll_dirs(torch_lib_dir: str | None, cuda_bin_dir: str | None) -> None:
	if not hasattr(os, "add_dll_directory"):
		return
	if torch_lib_dir:
		os.add_dll_directory(torch_lib_dir)
	if cuda_bin_dir:
		os.add_dll_directory(cuda_bin_dir)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
	xyxy = boxes.copy()
	xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]
	xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]
	return xyxy


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
	xyxy = boxes.copy()
	xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
	xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
	xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
	xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
	return xyxy


def _normalize_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
	if boxes.size == 0:
		return boxes
	max_val = float(np.max(boxes))
	if max_val <= 1.5:
		scale = np.array([w, h, w, h], dtype=np.float32)
		boxes = boxes * scale
		return _cxcywh_to_xyxy(boxes)

	if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
		return _xywh_to_xyxy(boxes)

	return boxes


def _clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
	if boxes.size == 0:
		return boxes
	boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
	boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
	boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
	boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
	return boxes


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--image", required=True)
	parser.add_argument("--config", required=True)
	parser.add_argument("--weights", required=True)
	parser.add_argument("--prompt", required=True)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--box-threshold", type=float, default=0.35)
	parser.add_argument("--text-threshold", type=float, default=0.25)
	parser.add_argument("--max-detections", type=int, default=50)
	parser.add_argument("--torch-lib-dir", default=None)
	parser.add_argument("--cuda-bin-dir", default=None)
	args = parser.parse_args()

	_add_dll_dirs(args.torch_lib_dir, args.cuda_bin_dir)

	from groundingdino.util.inference import load_model, load_image, predict

	model = load_model(args.config, args.weights)
	image_source, image = load_image(args.image)
	boxes, logits, phrases = predict(
		model=model,
		image=image,
		caption=args.prompt,
		box_threshold=args.box_threshold,
		text_threshold=args.text_threshold,
		device=args.device,
	)

	if hasattr(boxes, "detach"):
		boxes = boxes.detach().cpu().numpy()
	if hasattr(logits, "detach"):
		logits = logits.detach().cpu().numpy()

	h, w = image_source.shape[:2]
	boxes = _normalize_boxes(np.array(boxes, dtype=np.float32), w, h)
	boxes = _clip_boxes(boxes, w, h)

	scores = np.array(logits, dtype=np.float32)
	if scores.ndim > 1:
		scores = np.max(scores, axis=tuple(range(1, scores.ndim)))
	scores = scores.reshape(-1)
	if scores.shape[0] != boxes.shape[0]:
		scores = np.ones((boxes.shape[0],), dtype=np.float32)

	if ops is not None and boxes.shape[0] > 0:
		boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
		scores_tensor = torch.as_tensor(scores, dtype=torch.float32)
		keep_idx = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.45).cpu().numpy().astype(np.int64)
		boxes = boxes[keep_idx]
		scores = scores[keep_idx]
		phrases = [phrases[int(i)] for i in keep_idx if int(i) < len(phrases)]

	logits = scores

	results: List[dict] = []
	for box, score, phrase in zip(boxes, logits, phrases):
		if len(results) >= args.max_detections:
			break
		results.append(
			{
				"bbox": [float(x) for x in box.tolist()],
				"score": float(score),
				"label": str(phrase).strip(),
			}
		)

	print(json.dumps(results))


if __name__ == "__main__":
	main()
