from pathlib import Path
import tempfile

from src.diffusion.style_lora_adapter import MoodStyleManager


class DummyPipe:
    def __init__(self):
        self.loaded_path = None
        self.unloaded = False

    def load_lora_weights(self, p):
        self.loaded_path = p

    def unload_lora_weights(self):
        self.unloaded = True


def test_mood_manager_load_unload_prompt_kwargs():
    pipe = DummyPipe()
    mgr = MoodStyleManager(pipe)

    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "pytorch_lora_weights.safetensors"
        fake.write_text("x", encoding="utf-8")
        mgr.mood_registry = {
            "Neo Tokyo Neon": {"trigger": "mdply_neo_tokyo", "path": str(fake)}
        }

        mgr.load_mood("Neo Tokyo Neon")
        assert mgr.current_mood == "Neo Tokyo Neon"
        assert pipe.loaded_path == str(fake)

        p = mgr.format_prompt("a city street")
        assert p.startswith("mdply_neo_tokyo")

        kwargs = mgr.get_cross_attention_kwargs()
        assert "scale" in kwargs and abs(kwargs["scale"] - 0.65) < 1e-6

        mgr.unload_mood()
        assert mgr.current_mood is None
        assert pipe.unloaded is True