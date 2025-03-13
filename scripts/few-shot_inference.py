# %%
import os

# Limit library threading based on allocated SLURM CPUs
# os.environ["OMP_NUM_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
from transformers import logging as transformers_logging
import time
import argparse
from utils import read_file, read_huggingface_token, load_tokenizer_and_model, classification_CausalLM 
from utils import prepare_two_shot_prompt, get_two_shot_data, combine_csv_files, setup_experiment

# Set up verbose logging for transformers library
transformers_logging.set_verbosity_info()

# Setup Experiment
experiment = setup_experiment("fewShot")

# Unpack values from experiment setup
args = experiment["args"]
output_dir = experiment["output_dir"]
src_dir = experiment["src_dir"]
batch_dir = experiment["batch_dir"]
test_csv = experiment["test_csv"]
train_csv = experiment["train_csv"]
combined_output_csv = experiment["combined_output_csv"]
model_name = experiment["model_name"]

# Read Hugging Face Token
huggingface_token = read_huggingface_token()

# Load Model & Tokenizer
tokenizer, model = load_tokenizer_and_model(model_name, huggingface_token)

if model is None:
    raise RuntimeError("Failed to load the model. Exiting.")

print("Starting Few-Shot Inference...")

# Load Data
print("Loading test and train datasets...")
test_data = pd.read_csv(test_csv)
train_data = pd.read_csv(train_csv)
two_shot_batches = get_two_shot_data(test_data, train_data)

# Process in batches
batch_size = 5  # Number of rows per chunk
batch_number = 0  # Start with the specific batch
max_batches = None  # Set this to any number you want to stop at, or None to process all batches
total_time = 0
start_row = batch_number * batch_size
total_rows = len(two_shot_batches)
print(f"Total number of files: {len(two_shot_batches)}")


# Loop through the dataset in chunks
for start_idx in range(start_row, total_rows, batch_size):
    
    #Start the time for each batch
    batch_start_time = time.time()

    if max_batches is not None and batch_number >= max_batches:
        print(f"Stopping at batch {batch_number} as max_batches limit is reached.")
        break  # Exit the loop if the batch limit is reached

    print(f"Processing batch {batch_number} (rows {start_idx} to {start_idx + batch_size - 1})...")

    # Define the end index for the current batch
    end_idx = min(start_idx + batch_size, total_rows)
    batch = pd.DataFrame(two_shot_batches).iloc[start_idx:end_idx].copy()

    # Prepare a list to store the results for the current batch
    responses = []
    predictions = []
    errors = []
    token_sizes = []

    for idx, row in batch.iterrows():
        project_name = row['Project_name']
        github_path = row['github_path']
        pos_example_path = row['positive_example']
        neg_example_path = row['negative_example']

        print(f"Processing file {project_name} at path {github_path}...")

        # Process file content and examples
        file_content, status_file = read_file(src_dir, project_name, github_path)
        pos_example, status_pos = read_file(src_dir, project_name, pos_example_path)
        neg_example, status_neg = read_file(src_dir, project_name, neg_example_path)

                
        if status_file == -1 or status_pos == -1 or status_neg == -1:
            responses.append(None)
            predictions.append(None)
            errors.append("Failed to read or File not found")
            print(f"Failed to read or File not found for one of: {github_path}, {pos_example_path}, {neg_example_path}")
            continue

        # Prepare the two-shot prompt
        twoshot_prompt = prepare_two_shot_prompt(file_content, pos_example, neg_example)

        # Perform Classification by inference
        response, prediction, error_message, token_size = classification_CausalLM(model, tokenizer, twoshot_prompt)
        responses.append(response)
        predictions.append(prediction)
        errors.append(error_message)
        token_sizes.append(token_size)
        print(f"Inference completed for {github_path}. Prediction: {prediction}, Error: {error_message}")
        
    # Add the predictions and texts to the batch DataFrame
    batch['codellama_response'] = responses
    batch['codellama_prediction'] = predictions
    batch['errors'] = errors
    batch['token_sizes'] = token_sizes

    # Save the current batch to a separate CSV file
    batch_file_name = os.path.join(batch_dir, f'batch_{batch_number}.csv')
    batch.to_csv(batch_file_name, index=False)
    print(f"Batch {batch_number} saved as {batch_file_name}")

    #End the time for each batch
    batch_end_time = time.time() - batch_start_time
    print(f"Batch {batch_number} completed in {batch_end_time:.2f} seconds.")
    total_time += batch_end_time

    # Increment batch number for the next iteration
    batch_number += 1

# At the end, combine all batch CSVs into a single final CSV file
csv_files = [os.path.join(batch_dir, f'batch_{i}.csv') for i in range(batch_number)]
combine_csv_files(output_dir, combined_output_csv, csv_files)

print(f"All batches combined and saved as {combined_output_csv}")
print("Job completed.")
print(f"Total time taken for all batches: {total_time:.2f} seconds")
print("Few-shot inference complete. Results saved.")
