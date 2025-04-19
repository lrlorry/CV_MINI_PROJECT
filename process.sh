#!/bin/bash
# process_all.sh - Script to run all processing commands
# Usage: bash process_all.sh

echo "Starting image processing..."

# Create output directories
mkdir -p results/finetuned

# Common parameters
SKETCH="sketch.jpg"
DEPTH="depth.png"
BLOCK_SIZE=256
OVERLAP=64

# Process with regular model
echo "Processing with regular model (final_model.pth)..."

echo "[1/8] Running original color mode..."
python3 process.py --mode process --model models/final_model.pth --sketch $SKETCH --depth $DEPTH \
  --output final_model_result --block_size $BLOCK_SIZE --overlap $OVERLAP --color_mode original --use_lab

echo "[2/8] Running HSV color mode..."
python3 process.py --mode process --model models/final_model.pth --sketch $SKETCH --depth $DEPTH \
  --output final_model_result --block_size $BLOCK_SIZE --overlap $OVERLAP --color_mode hsv --use_lab

echo "[3/8] Running palette color mode..."
python3 process.py --mode process --model models/final_model.pth --sketch $SKETCH --depth $DEPTH \
  --output final_model_result --block_size $BLOCK_SIZE --overlap $OVERLAP --color_mode palette --use_lab

echo "[4/8] Running quantized color mode..."
python3 process.py --mode process --model models/final_model.pth --sketch $SKETCH --depth $DEPTH \
  --output final_model_result --block_size $BLOCK_SIZE --overlap $OVERLAP --color_mode quantized --use_lab

# Process with finetuned model
echo "Processing with finetuned model (finetuned_model.pth)..."

echo "[5/8] Running original color mode (finetuned)..."
python3 process.py --mode process --model models/finetuned_model.pth --sketch $SKETCH --depth $DEPTH \
  --output finetune_model_result --block_size $BLOCK_SIZE --overlap $OVERLAP --color_mode original --use_lab

echo "[6/8] Running HSV color mode (finetuned)..."
python3 process.py --mode process --model models/finetuned_model.pth --sketch $SKETCH --depth $DEPTH \
  --output finetune_model_result --block_size $BLOCK_SIZE --overlap $OVERLAP --color_mode hsv --use_lab

echo "[7/8] Running palette color mode (finetuned)..."
python3 process.py --mode process --model models/finetuned_model.pth --sketch $SKETCH --depth $DEPTH \
  --output finetune_model_result --block_size $BLOCK_SIZE --overlap $OVERLAP --color_mode palette --use_lab

echo "[8/8] Running quantized color mode (finetuned)..."
python3 process.py --mode process --model models/finetuned_model.pth --sketch $SKETCH --depth $DEPTH \
  --output finetune_model_result --block_size $BLOCK_SIZE --overlap $OVERLAP --color_mode quantized --use_lab

echo "All processing complete!"
echo "Results saved to 'final_model_result' and 'finetune_model_result' directories."