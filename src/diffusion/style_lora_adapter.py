from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


class MoodStyleManager:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.current_mood: Optional[str] = None
        self.lora_scale: float = 0.65
        self.mood_registry: Dict[str, Dict[str, str]] = {
            "Warm Sunny Day": {
                "trigger": "mdply_warm_sunny",
                "path": "models/loras/warm_sunny/pytorch_lora_weights.safetensors",
            },
            "Neo Tokyo Neon": {
                "trigger": "mdply_neo_tokyo",
                "path": "models/loras/neo_tokyo/pytorch_lora_weights.safetensors",
            },
            "Sunday Blues": {
                "trigger": "mdply_sunday_blues",
                "path": "models/loras/sunday_blues/pytorch_lora_weights.safetensors",
            },
            "Neutral Realistic": {
                "trigger": "mdply_neutral_real",
                "path": "models/loras/neutral_realistic/pytorch_lora_weights.safetensors",
            },
        }

    def load_mood(self, mood_selection: str) -> None:
        if mood_selection not in self.mood_registry:
            self.unload_mood()
            return
        if self.current_mood == mood_selection:
            return

        self.unload_mood()
        lora_path = Path(self.mood_registry[mood_selection]["path"])
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights missing: {lora_path}")

        self.pipeline.load_lora_weights(str(lora_path))
        self.current_mood = mood_selection

    def unload_mood(self) -> None:
        if self.current_mood is not None:
            self.pipeline.unload_lora_weights()
            self.current_mood = None

    def format_prompt(self, base_prompt: str) -> str:
        if self.current_mood is None:
            return base_prompt
        trigger = self.mood_registry[self.current_mood]["trigger"]
        return f"{trigger}, {base_prompt}"

    def get_cross_attention_kwargs(self) -> dict:
        if self.current_mood is None:
            return {}
        return {"scale": self.lora_scale}