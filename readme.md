# Replication Package for Performance Bug Detection using LLMs and ML Models

This repository contains the replication package for the study on performance bug detection using Large Language Models (LLMs) and Machine Learning (ML) models. It includes all necessary scripts and instructions to replicate the experiments and analyses presented in the study.


## 1. Requirements

- **Python 3.10.5**  
- All dependencies are listed in **`requirements.txt`**

### Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Hugging Face API Key

- The LLM inference scripts require a **Hugging Face API** token.  
- Set the token in the ```config.json``` file before running inference.

```
{
  "HUGGINGFACE_TOKEN": "Your_token"
}
```

The token should be replaced with your actual Hugging Face API token.


## 2. Directory Structure and Description

This package is structured as follows:

```plaintext
replication-package_per_bug_detection_LLM/
│── data/                      # Contains input and output data
│── projects/                   # Contains source code of analyzed projects
│── scripts/                    # Contains Python scripts for LLM and ML processing
│── runner/                     # Contains batch and shell scripts for execution
│── config.json                 # Stores API key for Hugging Face
│── requirements.txt            # Lists dependencies
│── README.md                   # This file
```
### Key Directories
- ```scripts/```: Contains all Python scripts for inference, processing, analysis, and ML.
- ```runner/```: Contains execution scripts for running different parts of the pipeline.

## 3. Running the Experiments

### Step 1: Run LLM Inference

#### Option 1: Run Zero-Shot and Few-Shot Separately

To run zero-shot inference, execute:
```bash
python scripts/zero-shot_inference.py
```

To run few-shot inference, execute:
```bash
python scripts/few-shot_inference.py
```

#### Option 2: Run Both Together

On Windows:
```
runner\run_llm_inference.bat
```

On Linux/macOS:
```
bash runner/run_llm_inference.sh
```

**Outputs**: Results are stored in ```data/output/llm_inference/```.

### Step 2: Running ML Models for Performance Bug Detection
To execute ML models:
```
python scripts/ml_model_runner.py
```
**Output**: ML performance metrics will be stored in ```data/output/with_without_anti_pattern_metrics_ML_Models.csv```.

### Step 3: Process and Analyze LLM Outputs
After running LLM inference, process and analyze the results.
On Windows:
```
runner\run_llm_analysis_pipeline.bat
```

On Linux/macOS:
```
bash runner/run_llm_analysis_pipeline.sh
```
**Outputs:**
- ```combined_LLM_outputs_cleaned.csv``` → processed and combined output of LLM inference
- ```memory_errors_files.csv``` → list of files that could not be inferred due to possible GPU memory limitation
- ```data/output/summary_llms.csv``` → Summary of LLM performance
- ```data/output/project_wise_metrics.csv``` → Project-wise evaluation results

### Step 4: Generate Results
This step generates results for **RQ1**, **RQ2**, and **RQ3** and creates necessary visualizations.

On Windows:
```
runner\generate_results.bat
```

On Linux/macOS:
```
bash runner/generate_results.sh
```
**Outputs:**
- ```data/output/rq1_zeroshot_results.csv``` → RQ1 results
- ```data/output/rq2_fewshot_results.csv``` → RQ2 results
- ```data/output/rq3_combined_matrix.csv``` → RQ3 results
- ```data/output/figures/``` → Generated plots

## 4. Viewing Results
- LLM Performance summary: ```data/output/summary_llms.csv```
- Project-wise metrics for LLM: ```data/output/project_wise_metrics.csv```
- ML model results: ```data/output/with_without_anti_pattern_metrics_ML_Models.csv```
- RQ1 results: ```data/output/rq1_zeroshot_results.csv```
- RQ2 results: ```data/output/rq2_fewshot_results.csv```
- RQ3 results: ```data/output/rq3_combined_matrix.csv```
- Plots and figures: ```data/output/figures/```