#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_path", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--out_dir", default="lora_tests")
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")

    pipe.load_lora_weights(args.lora_path)

    prompts = [
        f"{args.trigger}, a highly detailed city street, daylight",
        f"{args.trigger}, a vintage red sports car parked in an empty parking lot",
        f"{args.trigger}, a simple coffee mug on a wooden table",
    ]
    scales = [0.5, 0.65, 0.8, 1.0]

    for p in prompts:
        for s in scales:
            image = pipe(
                prompt=p,
                num_inference_steps=args.steps,
                guidance_scale=7.5,
                cross_attention_kwargs={"scale": s},
                generator=torch.Generator("cuda").manual_seed(42),
            ).images[0]
            name = p.replace(",", "").replace(" ", "_")[:60]
            image.save(out / f"{name}_scale_{s:.2f}.png")

    print(f"[done] saved in {out}")


if __name__ == "__main__":
    main()