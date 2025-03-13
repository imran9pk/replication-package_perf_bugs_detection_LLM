@echo off
echo Running LLM inference...

echo Running Zero-Shot inference...
python ../scripts/zero-shot_inference.py || exit /b 1

echo Running Few-Shot inference...
python ../scripts/few-shot_inference.py || exit /b 1

echo LLM inference completed successfully.
