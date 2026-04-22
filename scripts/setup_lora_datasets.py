import os
from pathlib import Path

# Define the mood templates and their specific trigger tokens
MOOD_TEMPLATES = {
    "warm_sunny": {"trigger": "mdply_warm_sunny", "caption": "a city street scene, warm golden hour, sunlight"},
    "neo_tokyo": {"trigger": "mdply_neo_tokyo", "caption": "a city street scene, cyberpunk, neon lights, rainy"},
    "pastel_filmic": {"trigger": "mdply_pastel_filmic", "caption": "a city street scene, wes anderson style, pastel colors, low contrast"},
    "neutral_real": {"trigger": "mdply_neutral_real", "caption": "a city street scene, realistic daylight, documentary photography"}
}

ROOT_DIR = Path("lora_training_data")

def setup_directories_and_captions():
    # 1. Create root and sub-folders
    for mood in MOOD_TEMPLATES.keys():
        mood_path = ROOT_DIR / mood
        mood_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {mood_path}")

    # 2. Scan folders and auto-generate .txt captions for any image found
    valid_extensions = {".jpg", ".jpeg", ".png"}
    
    for mood, data in MOOD_TEMPLATES.items():
        mood_path = ROOT_DIR / mood
        images = [f for f in mood_path.iterdir() if f.suffix.lower() in valid_extensions]
        
        for img_path in images:
            txt_path = img_path.with_suffix(".txt")
            # Only write if caption doesn't already exist
            if not txt_path.exists():
                caption = f"{data['trigger']}, {data['caption']}"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                print(f"Created caption for {img_path.name}: {caption}")

if __name__ == "__main__":
    print("--- MoodPlay Dataset Manager ---")
    setup_directories_and_captions()
    print("Setup complete. Drop your 512x512 images into the folders and run this script again to auto-caption.")