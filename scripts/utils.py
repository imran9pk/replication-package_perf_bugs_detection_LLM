import os
import re
import torch
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# Function to parse arguments and set up directories
def setup_experiment(experiment_type):
    
    # Argument Parsing
    parser = argparse.ArgumentParser(description=f"Run {experiment_type.capitalize()} Performance Bug Classification")
    parser.add_argument("--comments", choices=["with", "without"], required=True, help="Use source files with or without comments")
    parser.add_argument("--model", choices=["qwen", "llama", "artigenz"], required=True, help="Choose the model to use")
    args = parser.parse_args()

    # Define directories based on comments (with/without)
    input_dir = "../data/input"
    output_dir = "../data/output/llm_inference"
    src_dir = "../projects/src_files_raw" if args.comments == "with" else "../projects/src_files_no_comments"
    batch_dir = os.path.join(output_dir, f"{args.model}/{args.comments}Comments_{experiment_type}")

    # Ensure directories exist
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Define dataset paths
    test_csv = os.path.join(input_dir, "test_data_per_project.csv")
    train_csv = os.path.join(input_dir, "train_data_per_project.csv") if experiment_type == "fewShot" else None
    combined_output_csv = os.path.join(output_dir, f"{args.model}_{args.comments}Comments_{experiment_type}.csv")

    # Define model mappings
    model_name = {
        "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "llama": "codellama/CodeLlama-7b-Instruct-hf",
        "artigenz": "Artigenz/Artigenz-Coder-DS-6.7B"
    }[args.model]

    return {
        "args": args,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "src_dir": src_dir,
        "batch_dir": batch_dir,
        "test_csv": test_csv,
        "train_csv": train_csv,
        "combined_output_csv": combined_output_csv,
        "model_name": model_name
    }

# Function to read Hugging Face token from a `.env` file
def read_huggingface_token():
    token_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    token = None
    
    if os.path.exists(token_file_path):
        with open(token_file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("HUGGINGFACE_TOKEN="):
                    token = line.strip().split("=")[1].strip()
                    break
    
    if not token:
        token = os.getenv("HUGGINGFACE_TOKEN")  # Fallback to environment variable if file is missing

    if not token:
        raise ValueError("Hugging Face token not found. Please add it to a `.env` file or set the environment variable.")
    
    print("Hugging Face token loaded successfully.")
    return token


def read_file(src_dir, project_name, github_path, verbose=False):

    # Just adding this do handle Project name case in CSV file
    if (project_name == "Openolat"):
      project_name = "OpenOLAT"
    if(project_name == "Exoplayer"):
      project_name = "ExoPlayer"

    content = None
    status = -1

    # Construct the initial project directory path
    project_dir = os.path.join(src_dir, project_name)
    
    # Check if the original project directory exists, if not try variations
    if not os.path.exists(project_dir):
        
        # Try variations with lowercase and uppercase first letter
        project_dir_lower = os.path.join(src_dir, project_name[0].lower() + project_name[1:])
        project_dir_upper = os.path.join(src_dir, project_name[0].upper() + project_name[1:])
        
        # Update project_dir if one of the variations exists
        if os.path.exists(project_dir_lower):
            project_dir = project_dir_lower
        elif os.path.exists(project_dir_upper):
            project_dir = project_dir_upper
        else:
            if verbose:
                print(f"Project directory not found for any case variation: {project_name}")
            return None, -1  # Return if no directory variation is found

    # Construct the full file path using the identified project directory
    github_path = github_path.lstrip('/')  # Clean up leading slashes
    file_path = os.path.join(project_dir, github_path)
    file_path = os.path.normpath(file_path)

    # Check if the file exists and try to read content
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            status = 1  # Set status to success if content is read successfully
        except UnicodeDecodeError:
            if verbose:
                print(f"Error reading file with UTF-8, retrying with Latin-1 encoding: {file_path}")
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                status = 1  # Set status to success if content is read successfully
            except Exception as e:
                if verbose:
                    print(f"Failed to read file with Latin-1 encoding as well: {file_path}. Error: {e}")
    else:
        if verbose:
            print(f"File not found at path: {file_path}")

    # Return content and status
    return content, status

def classification_CausalLM(model, tokenizer, prompt):
    response = None
    prediction = None
    error_message = None
    token_size = 0

    try:
        # Tokenize prompt
        print("Tokenizing prompt...")
        tokenized_prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(tokenized_prompt, return_tensors="pt", padding=True)
        
        # Print tokens length
        token_size = len(inputs['input_ids'][0])

        # Move inputs to the model's device
        inputs = {key: val.to(model.device) for key, val in inputs.items()}
        print("Inputs moved to model device successfully.:" + str(model.device))
        
        # Generate output with max_new_tokens=10
        output = model.generate(
            inputs['input_ids'], 
            attention_mask = inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=10,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1
        )

        # Decode the output tokens and convert to lowercase
        response = tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip().lower()
            
        # Split the response if "[/INST]" is present, to remove prompt text
        if "[/inst]" in response:
            response = response.split("[/inst]")[-1].strip()

        if re.search(r'\byes\b', response):
            prediction = 1
        elif re.search(r'\bno\b', response):
            prediction = 0

    except torch.cuda.OutOfMemoryError as e:
        print(f"Out of Memory error during pipeline execution: {e}")
        error_message = "Out of Memory"
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        error_message = str(e)
    
    # Return the predictions and any error message (if an error occurred)
    return response, prediction, error_message, token_size

def load_tokenizer_and_model(model_name, huggingface_token):

    # Check CUDA availability
    print("Checking CUDA availability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA availability: {device.upper()}")
    
    try:
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Tokenizer initialized successfully.")

        # Initialize model
        print("Initializing model...")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=huggingface_token)
        model.to(device)
        print(f"Model initialized successfully on {device.upper()}.")
    
    except Exception as e:
        print(f"Error during model or tokenizer loading: {e}")

    return tokenizer, model

def prepare_zero_shot_prompt(file_content):

    zero_shot_prompt = [
        {
            "role": "system",
            "content": "You are an AI binary performance bug classifier that identifies whether " 
                        "the provided code contains perforamnce bug or not."
                        "Provide response only in following format: performance bug: <YES or NO>"
                        "Do not include anything else in response."
        },
        {
            "role": "user",
            "content": "Does the following code snippet contains any performance bug?" 
                        f"{file_content}"
        },
    ]
    return zero_shot_prompt

def prepare_two_shot_prompt(file_content, positive_example, negative_example):
    two_shot_prompt = [
        {
            "role": "system",
            "content": "You are an AI binary classifier tasked with identifying whether a code file contains performance bugs. "
                       "Respond only in the following format: performance bug: <YES or NO>. Do not include anything else in the response. "
        },
        # Positive example (buggy file)
        {
            "role": "user",
            "content": "Does the following code snippet contain any performance bug?\n"
                       f"{positive_example}\n"
        },
        {
            "role": "assistant",
            "content": "performance bug: YES"
        },
        # Negative example (non-buggy file)
        {
            "role": "user",
            "content": "Does the following code snippet contain any performance bug?\n"
                       f"{negative_example}\n"
        },
        {
            "role": "assistant",
            "content": "performance bug: NO"
        },
        # Actual input to classify
        {
            "role": "user",
            "content": "Does the following code snippet contain any performance bug?\n"
                       f"{file_content}\n"
        },
    ]
    return two_shot_prompt

#Prepare project-wise data for two-shot classification.
#Each file in the test dataset is paired with a positive and a negative example from the training dataset.  
def get_two_shot_data(test_data, train_data):
    
    two_shot_batches = []

    # Process each project
    for project in test_data['Project_name'].unique():
        print(f"Processing project: {project}")

        # Filter test and train data for the current project
        project_test_data = test_data[test_data['Project_name'] == project]
        project_train_data = train_data[train_data['Project_name'] == project]

        # Sort training data by token_size_codellama
        project_train_data = project_train_data.sort_values(by='token_size_codellama')

        # Ensure the training set has positive and negative examples
        positive_example_rows = project_train_data[project_train_data['label'] == 1]
        negative_example_rows = project_train_data[project_train_data['label'] == 0]

        if positive_example_rows.empty or negative_example_rows.empty:
            print(f"Skipping project {project} due to insufficient examples in training set.")
            continue

        # Select the smallest positive and negative examples
        positive_example = positive_example_rows.iloc[0]['github_path']
        negative_example = negative_example_rows.iloc[0]['github_path']

        # Add each test file along with examples
        for _, test_row in project_test_data.iterrows():
            file = test_row['File']
            github_path = test_row['github_path']
            label = test_row['label']

            two_shot_batches.append({
                "Project_name": project,
                "File": file,
                "github_path": github_path,
                "label": label,
                "positive_example": positive_example,
                "negative_example": negative_example
            })

    return two_shot_batches

# Load all batch CSVs and combine them into one DataFrame
def combine_csv_files(output_dir, combined_output_csv, csv_files):
    print("Combining all batch CSVs into a single final CSV file...")
    if csv_files:
        combined_df = pd.concat([pd.read_csv(f) for f in csv_files])
        os.makedirs(output_dir, exist_ok=True)
        combined_df.to_csv(combined_output_csv, index=False)
        print(f"All batches combined and saved as {combined_output_csv}")
    else:
        print("No batch files found. Skipping final combination.")
