#!/bin/bash
echo "Running LLM inference..."

echo "Running Zero-Shot inference..."
python ../scripts/zero-shot_inference.py || { echo "Error: zero-shot_inference.py failed."; exit 1; }

echo "Running Few-Shot inference..."
python ../scripts/few-shot_inference.py || { echo "Error: few-shot_inference.py failed."; exit 1; }

echo "LLM inference completed successfully."
