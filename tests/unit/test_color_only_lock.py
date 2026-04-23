import numpy as np
import cv2
from src.diffusion.color_only_diffusion import InstanceColorizer


def test_luminance_lock_preserves_l_channel():
    h, w = 64, 64
    src = np.zeros((h, w, 3), dtype=np.uint8)
    src[:] = [120, 120, 120]  # gray

    gen = np.zeros((h, w, 3), dtype=np.uint8)
    gen[:] = [200, 50, 50]  # colored

    out = InstanceColorizer.luminance_lock(src, gen)

    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
    out_lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)

    assert np.allclose(src_lab[:, :, 0], out_lab[:, :, 0])