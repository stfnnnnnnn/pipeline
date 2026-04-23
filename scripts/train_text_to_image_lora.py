#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LOGGER = get_logger(__name__, log_level="INFO")


class LocalLoraImageDataset(Dataset):
    def __init__(
        self,
        root: Path,
        tokenizer: CLIPTokenizer,
        resolution: int,
        center_crop: bool,
        random_flip: bool,
    ) -> None:
        self.root = root
        self.tokenizer = tokenizer

        self.image_paths = sorted(
            [p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS],
            key=lambda p: p.name.lower(),
        )
        if not self.image_paths:
            raise ValueError(f"No training images found under: {self.root}")

        crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        transform_ops = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            crop,
        ]
        if random_flip:
            transform_ops.append(transforms.RandomHorizontalFlip())
        transform_ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_transform = transforms.Compose(transform_ops)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_caption(self, image_path: Path) -> str:
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()
            if text:
                return text
        # Fall back to the stem if sidecar caption is missing.
        return image_path.stem.replace("_", " ")

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_transform(image)

        caption = self._read_caption(image_path)
        tokenized = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
        }


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    input_ids = torch.stack([e["input_ids"] for e in examples])
    return {
        "pixel_values": pixel_values.contiguous().float(),
        "input_ids": input_ids,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PEFT LoRA training for SD1.5 with local image folder + .txt captions")
    ap.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    ap.add_argument("--train_data_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)

    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--lr_scheduler", type=str, default="constant")
    ap.add_argument("--lr_warmup_steps", type=int, default=0)
    ap.add_argument("--max_train_steps", type=int, default=1000)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--dataloader_num_workers", type=int, default=0)

    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpointing_steps", type=int, default=200)

    ap.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--use_8bit_adam", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--center_crop", action="store_true")
    ap.add_argument("--random_flip", action="store_true")

    return ap.parse_args()


def save_lora_weights(accelerator: Accelerator, unet: UNet2DConditionModel, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_unet = accelerator.unwrap_model(unet)
    lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(save_dir),
        unet_lora_layers=lora_state_dict,
        safe_serialization=True,
    )


def pick_optimizer(use_8bit_adam: bool):
    if use_8bit_adam:
        try:
            bnb = importlib.import_module("bitsandbytes")

            return bnb.optim.AdamW8bit, True
        except ImportError:
            LOGGER.warning("--use_8bit_adam requested but bitsandbytes is unavailable. Falling back to AdamW.")
    return torch.optim.AdamW, False


def main() -> None:
    args = parse_args()

    if args.resolution % 8 != 0:
        raise ValueError("--resolution must be divisible by 8")

    args.train_data_dir = args.train_data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.train_data_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {args.train_data_dir}")

    project_config = ProjectConfiguration(
        project_dir=str(args.output_dir),
        logging_dir=str(args.output_dir / "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)

    LOGGER.info("Loading model components from: %s", args.pretrained_model_name_or_path)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer_cls, is_8bit = pick_optimizer(args.use_8bit_adam)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = optimizer_cls(trainable_params, lr=args.learning_rate)

    train_dataset = LocalLoraImageDataset(
        root=args.train_data_dir,
        tokenizer=tokenizer,
        resolution=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("lora_train")
        run_info = {
            "model": args.pretrained_model_name_or_path,
            "train_data_dir": str(args.train_data_dir),
            "output_dir": str(args.output_dir),
            "resolution": args.resolution,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_train_steps": args.max_train_steps,
            "rank": args.rank,
            "seed": args.seed,
            "mixed_precision": args.mixed_precision,
            "optimizer": "AdamW8bit" if is_8bit else "AdamW",
        }
        (args.output_dir / "run_config.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    LOGGER.info("Examples: %d", len(train_dataset))
    LOGGER.info("Epochs: %d", num_train_epochs)
    LOGGER.info("Per-device batch size: %d", args.train_batch_size)
    LOGGER.info("Total train batch size (w. parallel and accumulation): %d", total_batch_size)
    LOGGER.info("Max train steps: %d", args.max_train_steps)

    global_step = 0

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                input_ids = batch["input_ids"]

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    ckpt_dir = args.output_dir / f"checkpoint-{global_step}"
                    save_lora_weights(accelerator, unet, ckpt_dir)
                    LOGGER.info("Saved checkpoint LoRA weights to: %s", ckpt_dir)

                if global_step >= args.max_train_steps:
                    break

            if accelerator.is_local_main_process and step % 10 == 0:
                LOGGER.info(
                    "epoch=%d step=%d global_step=%d loss=%.6f",
                    epoch,
                    step,
                    global_step,
                    loss.detach().item(),
                )

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_weights(accelerator, unet, args.output_dir)
        LOGGER.info("Saved final LoRA weights to: %s", args.output_dir)
        LOGGER.info("Expected weight file: %s", args.output_dir / "pytorch_lora_weights.safetensors")

    accelerator.end_training()


if __name__ == "__main__":
    main()
