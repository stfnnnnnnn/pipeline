from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)


@dataclass
class ColorOnlyConfig:
    base_model: str = "runwayml/stable-diffusion-inpainting"
    controlnet_model: str = "lllyasviel/sd-controlnet-canny"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16
    canny_low: int = 100
    canny_high: int = 200
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    strength: float = 0.70


class InstanceColorizer:
    def __init__(self, cfg: Optional[ColorOnlyConfig] = None) -> None:
        self.cfg = cfg or ColorOnlyConfig()

        controlnet = ControlNetModel.from_pretrained(
            self.cfg.controlnet_model,
            torch_dtype=self.cfg.torch_dtype,
        )

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.cfg.base_model,
            controlnet=controlnet,
            torch_dtype=self.cfg.torch_dtype,
            safety_checker=None,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if self.cfg.device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.cfg.device)

    @staticmethod
    def to_gray_rgb(rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def canny_control(self, gray_rgb: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray_rgb, self.cfg.canny_low, self.cfg.canny_high)
        return np.stack([edges, edges, edges], axis=2)

    @staticmethod
    def luminance_lock(original_gray_rgb: np.ndarray, generated_rgb: np.ndarray) -> np.ndarray:
        src_lab = cv2.cvtColor(original_gray_rgb, cv2.COLOR_RGB2LAB)
        gen_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB)
        out_lab = gen_lab.copy()
        out_lab[:, :, 0] = src_lab[:, :, 0]
        return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)

    def colorize_instance(
        self,
        original_rgb: np.ndarray,
        mask_u8: np.ndarray,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        o = overrides or {}
        steps = int(o.get("num_inference_steps", self.cfg.num_inference_steps))
        gs = float(o.get("guidance_scale", self.cfg.guidance_scale))
        strength = float(o.get("strength", self.cfg.strength))

        gray_rgb = self.to_gray_rgb(original_rgb)
        control_np = self.canny_control(gray_rgb)

        pil_img = Image.fromarray(gray_rgb)
        pil_mask = Image.fromarray(mask_u8.astype(np.uint8))
        pil_control = Image.fromarray(control_np)

        out_pil = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_img,
            mask_image=pil_mask,
            control_image=pil_control,
            num_inference_steps=steps,
            guidance_scale=gs,
            strength=strength,
        ).images[0]

        generated_rgb = np.array(out_pil)
        locked_rgb = self.luminance_lock(gray_rgb, generated_rgb)

        return {
            "gray_rgb": gray_rgb,
            "control_rgb": control_np,
            "generated_rgb": generated_rgb,
            "locked_rgb": locked_rgb,
        }