@echo off
echo Starting MoodPlay PEFT LoRA Training...
echo Hardware Profile: RTX 5060 Ti (VRAM Optimized)

:: Define which mood you are currently training (Change this variable to train different moods)
set MOOD_NAME=neo_tokyo
set OUTPUT_DIR=models/loras/%MOOD_NAME%
set DATA_DIR=lora_training_data/%MOOD_NAME%

:: Ensure output directory exists
mkdir "%OUTPUT_DIR%" 2>nul

:: Launch HuggingFace Accelerate with memory-saving flags
accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py ^
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" ^
  --train_data_dir="%DATA_DIR%" ^
  --output_dir="%OUTPUT_DIR%" ^
  --resolution=512 ^
  --train_batch_size=1 ^
  --gradient_accumulation_steps=4 ^
  --mixed_precision="fp16" ^
  --use_8bit_adam ^
  --gradient_checkpointing ^
  --rank=16 ^
  --learning_rate=1e-4 ^
  --lr_scheduler="constant" ^
  --max_train_steps=1000 ^
  --seed=42

echo Training for %MOOD_NAME% complete!
pause