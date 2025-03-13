#!/bin/bash
echo "Generating results..."

echo "Computing RQ1 & RQ2 results..."
python ../scripts/rq1_rq2_results.py || { echo "Error: rq1_rq2_results.py failed."; exit 1; }

echo "Computing RQ3 results..."
python ../scripts/rq3_results.py || { echo "Error: rq3_results.py failed."; exit 1; }

echo "Generating figures..."
python ../scripts/figures.py || { echo "Error: figures.py failed."; exit 1; }

echo "Results generation completed successfully."
