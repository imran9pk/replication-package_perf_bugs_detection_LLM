@echo off
echo Running LLM analysis pipeline...

echo Processing LLM responses...
python ../scripts/process_llm_responses.py || exit /b 1

echo Analyzing LLM outputs...
python ../scripts/llm_output_analysis.py || exit /b 1

echo LLM analysis pipeline completed successfully.
