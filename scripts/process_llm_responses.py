# %%
import os
import glob
import pandas as pd
import re

# %%
# Match patterns in the response to determine if the code has bugs
def interpret_response(response):
    response = str(response).lower()
    bug_patterns = [
        r'potential',
        r'contains',  
        r'possible',
        r'has a few', 
        r'several',
        r'small performance',   
        r'minor issues',
        r'there are a few',
        r'there is a',
    ]

    for pattern in bug_patterns:
        if re.search(pattern, response):
            return 1
    return 0

# %%
input_folder = "../data/output/llm_inference"
output_folder = "../data/output"
combined_output_file = os.path.join(output_folder, 'combined_LLM_outputs_cleaned.csv')
memory_errors_file = os.path.join(output_folder, 'memory_errors_files.csv')
model_prefixes = ("artigenz", "codellama", "qwen")

# %%
#1- This script reads all the CSV files in the input folder and combines them into one CSV file
# It also adds rquired columns to the start of the dataframe for model, comments included and prompting
#2- Also, interprets the response using string patterns where predictions are missing and adds a new column for it
#3- Lastly, filters out rows, from combined dataset, that have out of memory errors also saves error files to a separate CSV file

csv_files = glob.glob(f"{input_folder}/*.csv")
dataframes = []
memory_errors = set()

# Iterate through the list of CSV files
for file in csv_files:
    file_name = os.path.basename(file)
    
    # Check if the file starts with any of the valid prefixes
    if file_name.startswith(model_prefixes):
        df = pd.read_csv(file)
        
        # Extract information from the file name
        parts = file_name.split('_')
        model = parts[0]
        comments_included = "withComments" in file_name
        prompting = "zeroShot" if "zeroShot" in file_name else "fewShot" if "fewShot" in file_name else None
        
        # Add new columns to the start of dataframe
        df.insert(0, 'Model', model)
        df.insert(1, 'CommentsIncluded', comments_included)
        df.insert(2, 'Prompting', prompting)
        
        # Append the dataframe to the list
        dataframes.append(df)

# Combine all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Filter rows that has out of memory errors and place them in memory_errors set
filtered_rows = combined_df[combined_df['errors'].notna() & (combined_df['errors'].astype(str).str.strip() != "")]
for _, row in filtered_rows.iterrows():
    memory_errors.add((row['Project_name'], row['github_path']))


# Remove rows corresponding to memory_errors from each group
combined_df = combined_df[
    ~combined_df.apply(
        lambda row: (row['Project_name'], row['github_path']) in memory_errors,
        axis=1
    )
]

# Add the Interpreted Column to interpret the missing predictions
combined_df['Interpreted'] = combined_df.apply(
    lambda row: None if not pd.isna(row['errors']) else (
        interpret_response(row['codellama_response'])
        if pd.isna(row['codellama_prediction']) else None
    ),
    axis=1
    )

# Convert the unique combinations set to a DataFrame
output_errors_df = pd.DataFrame(list(memory_errors), columns=['Project_name', 'github_path'])
output_errors_df = output_errors_df.sort_values(by=['Project_name'])
output_errors_df.to_csv(memory_errors_file, index=False)
print(f"Collected error files in {memory_errors_file}.")

# Save the combined dataframe to a new CSV file
combined_df.to_csv(combined_output_file, index=False)
print(f"Processed Combined LLMs output save to: {combined_output_file}")

# %%



