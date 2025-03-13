# %%
import pandas as pd
import os

# %%
# output_folder = r"D:\Sync\PerforamnceBugPrediction\kalifano_outputs_combined\Organized\output"
output_folder = "../data/output"

llm_metrics_file = os.path.join(output_folder, 'project_wise_metrics.csv')
ml_metrics_file = os.path.join(output_folder, 'with_without_anti_pattern_metrics_ML_Models.csv')
combined_f1_scores_file = os.path.join(output_folder, 'combined_f1_scores_pivoted.csv') 

wilcoxon_results_file = os.path.join(output_folder, 'wilcoxon_results.csv')
p_values_file = os.path.join(output_folder, 'p-values.csv')
cles_file = os.path.join(output_folder, 'cles.csv')

# %%
def rename_column(column):
    if isinstance(column, tuple):
        # If the column is a tuple (e.g., LLM configurations), use the first element (Model name)
        return column[0]
    elif isinstance(column, str) and "with anti-pattern metrics" in column:
        # If the column contains "with anti-pattern metrics", extract the first part (e.g., "DT")
        return column.split(" ")[0]
    else:
        # Return the column as-is for anything else
        return column
    
llm_data = pd.read_csv(llm_metrics_file)
# Filter the LLM data based on conditions - The best configuration is "withComments" and "fewShot"
filtered_llm_data = llm_data[(llm_data['CommentsIncluded'] == True) & (llm_data['Prompting'] == 'fewShot')].copy()

# Add a column to represent configurations as tuples
filtered_llm_data['Configuration'] = list(zip(filtered_llm_data['Model'], 
                                              filtered_llm_data['CommentsIncluded'], 
                                              filtered_llm_data['Prompting']))

# Pivot LLM data to have one column for each configuration
llm_pivot = filtered_llm_data.pivot(index='Project', columns='Configuration', values='F1 Score')


# Load the ML CSV file
ml_data = pd.read_csv(ml_metrics_file)
# Filter the ML data based on valid algorithm metrics
valid_algorithms = [
    "CNB with anti-pattern metrics",
    "LR with anti-pattern metrics",
    "DT with anti-pattern metrics",
    "MLP with anti-pattern metrics",
    "SVM with anti-pattern metrics",
    "RF with anti-pattern metrics"
]
filtered_ml_data = ml_data[ml_data['algorithm_metric'].isin(valid_algorithms)].copy()

# Pivot ML data to have one column for each algorithm metric
ml_pivot = filtered_ml_data.pivot(index='project', columns='algorithm_metric', values='f1')

# Rename index and columns for consistency
ml_pivot.index.name = 'Project'
ml_pivot.columns.name = None

# Combine the LLM and ML pivoted data
combined_data = pd.concat([llm_pivot, ml_pivot], axis=1)

# Apply the renaming function  and save to csv
combined_data.columns = [rename_column(col) for col in combined_data.columns]
combined_data.to_csv(combined_f1_scores_file)
print("Combined F1 scores (pivoted) saved as: "+combined_f1_scores_file)

# %%
from scipy.stats import wilcoxon
import pandas as pd
import numpy as np

# Perform statistical tests
def perform_statistical_tests(input_csv):
    data = pd.read_csv(input_csv, index_col=0)
    
    # Separate columns for LLMs and ML algorithms
    llm_columns = ['artigenz', 'codellama', 'qwen']
    ml_columns = ['CNB', 'DT', 'LR', 'MLP', 'RF', 'SVM']

    # Prepare for storing results
    comparisons = []
    p_values = []
    cles_values = []

    # Iterate over all LLM and ML combinations
    for llm_col in llm_columns:
        for ml_col in ml_columns:
            # Extract scores for the current pair of LLM and ML
            llm_scores = data[llm_col]
            ml_scores = data[ml_col]

            # Drop NaN values (only consider paired non-NaN values)
            paired_scores = data[[llm_col, ml_col]].dropna()
            llm_scores_filtered = paired_scores[llm_col]
            ml_scores_filtered = paired_scores[ml_col]

            if len(llm_scores_filtered) < 2:  # Skip if not enough data
                comparisons.append((llm_col, ml_col))
                p_values.append(None)
                cles_values.append(None)
                continue

            # Perform Wilcoxon test
            try:
                stat, p_value = wilcoxon(llm_scores_filtered, ml_scores_filtered, alternative='two-sided')
                p_values.append(round(p_value, 3))  # Round to 3 decimal places
            except ValueError:
                # If Wilcoxon cannot be performed
                p_values.append(None)

            # Compute CLES
            n = len(llm_scores_filtered)
            cles = sum([
                1 if llm_scores_filtered.iloc[i] > ml_scores_filtered.iloc[i]
                else 0.5 if llm_scores_filtered.iloc[i] == ml_scores_filtered.iloc[i]
                else 0
                for i in range(n)
            ]) / n
            cles_values.append(round(cles, 2))  # Round to 3 decimal places

            comparisons.append((llm_col, ml_col))

    # Create DataFrame for results
    results_df = pd.DataFrame(comparisons, columns=['LLM', 'ML Algo'])
    results_df['P-Value'] = p_values
    results_df['CLES'] = cles_values

    # Replace the CLES values with None where there's no significant difference (p > 0.05) 
    results_df['CLES'] = results_df.apply(lambda row: "--" if row['P-Value'] > 0.05 else row['CLES'], axis=1)

    # Save results
    # results_df.to_csv(wilcoxon_results_file ,index=False)

    # Pivot P-values and CLES into matrix format for readability
    p_value_matrix = results_df.pivot(index='ML Algo', columns='LLM', values='P-Value')
    # p_value_matrix.to_csv(p_values_file)

    cles_matrix = results_df.pivot(index='ML Algo', columns='LLM', values='CLES')
    # cles_matrix.to_csv(cles_file)

    #combine cles and p_values where each cell contains the p-value and cles value
    # the format is p-value (cles)
    combined_matrix = p_value_matrix.copy()
    for column in cles_matrix.columns:
        # if the p-value is 0, replace it with "0" to avoid scientific notation
        combined_matrix[column] = combined_matrix[column].apply(lambda x: "0" if x == 0 else x)
        combined_matrix[column] = combined_matrix[column].astype(str) + " (" + cles_matrix[column].astype(str) + ")"
    combined_matrix.to_csv(os.path.join(output_folder, 'rq3_combined_matrix.csv'))


    # print("Wilcoxon test results saved to " + wilcoxon_results_file)
    # print("P-value matrix saved to "+ p_values_file)
    # print("CLES matrix saved to " + cles_file)

    return results_df, p_value_matrix, cles_matrix

# Perform statistical tests on the given file
results_df, p_value_matrix, cles_matrix = perform_statistical_tests(combined_f1_scores_file)



