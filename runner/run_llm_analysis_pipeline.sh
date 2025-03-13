#!/bin/bash
echo "Running LLM analysis pipeline..."

echo "Processing LLM responses..."
python ../scripts/process_llm_responses.py || { echo "Error: process_llm_responses.py failed."; exit 1; }

echo "Analyzing LLM outputs..."
python ../scripts/llm_output_analysis.py || { echo "Error: llm_output_analysis.py failed."; exit 1; }

echo "LLM analysis pipeline completed successfully."
