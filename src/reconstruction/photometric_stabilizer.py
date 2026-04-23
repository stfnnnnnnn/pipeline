from __future__ import annotations
import cv2
import numpy as np


def luminance_lock_rgb(source_gray_rgb: np.ndarray, generated_rgb: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(source_gray_rgb, cv2.COLOR_RGB2LAB)
    gen_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB)
    out = gen_lab.copy()
    out[:, :, 0] = src_lab[:, :, 0]
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)