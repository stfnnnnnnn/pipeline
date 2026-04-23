from pathlib import Path
import logging

class MoodStyleManager:
    def __init__(self, pipeline):
        """
        Initializes the Mood Manager.
        :param pipeline: The initialized StableDiffusionControlNetPipeline
        """
        self.pipeline = pipeline
        self.current_mood = None
        self.lora_scale = 0.65  # Weight to prevent structural collapse
        
        # Registry mapping UI dropdown names to trigger words and file paths
        self.mood_registry = {
            "Neo Tokyo Neon": {
                "trigger": "mdply_neo_tokyo",
                "path": "models/loras/neo_tokyo/pytorch_lora_weights.safetensors"
            },
            "Neutral Realistic": {
                "trigger": "mdply_neutral_real",
                "path": "models/loras/neutral_realistic/pytorch_lora_weights.safetensors"
            },
            "Warm Sunny Day": {
                "trigger": "mdply_warm_sunny",
                "path": "models/loras/warm_sunny/pytorch_lora_weights.safetensors"
            }
        }

    def load_mood(self, mood_selection: str):
        """Swaps the LoRA dynamically based on user selection."""
        if mood_selection not in self.mood_registry:
            logging.warning("Mood '%s' not found. Defaulting to base model.", mood_selection)
            self.unload_mood()
            return

        # Don't reload if it's already active
        if self.current_mood == mood_selection:
            return

        # Cleanly unload any existing LoRA to prevent weight crossover
        self.unload_mood()

        lora_path = Path(self.mood_registry[mood_selection]["path"])
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights missing at {lora_path}")

        logging.info("Loading Global Mood Template: %s", mood_selection)
        self.pipeline.load_lora_weights(str(lora_path))
        self.current_mood = mood_selection

    def unload_mood(self):
        """Removes LoRA weights from the U-Net to return to the base SD1.5 model."""
        if self.current_mood is not None:
            logging.info("Unloading current mood template...")
            self.pipeline.unload_lora_weights()
            self.current_mood = None

    def format_prompt(self, base_prompt: str) -> str:
        """Appends the active mood trigger word to the user's global prompt."""
        if self.current_mood is None:
            return base_prompt
        
        trigger_word = self.mood_registry[self.current_mood]["trigger"]
        # Prepend the trigger word for maximum attention weight
        return f"{trigger_word}, {base_prompt}"

    def get_cross_attention_kwargs(self) -> dict:
        """Returns the specific cross-attention arguments required for inference."""
        if self.current_mood is None:
            return {}
        # Forces the LoRA to apply at 65% strength
        return {"scale": self.lora_scale}