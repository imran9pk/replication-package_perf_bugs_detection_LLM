import pandas as pd
import os

output_folder = "../data/output"
summary_table_path = os.path.join(output_folder, 'summary_llms.csv')

# Define output file paths
zeroshot_output_path = os.path.join(output_folder, "rq1_zeroshot_results.csv")
fewshot_output_path = os.path.join(output_folder, "rq2_fewshot_results.csv")

summary_df = pd.read_csv(summary_table_path)

# Filter results for zero-shot and few-shot configurations
zeroshot_df = summary_df[summary_df['Prompting'].str.contains("zeroShot", case=False)]
fewshot_df = summary_df[summary_df['Prompting'].str.contains("fewshot", case=False)]

# Group results by Model and With/Without Comments for both configurations
zeroshot_grouped = zeroshot_df.groupby(['Model', 'CommentsIncluded']).agg({
    'Precision': 'mean',
    'Recall': 'mean',
    'F1': 'mean'
}).reset_index()

fewshot_grouped = fewshot_df.groupby(['Model', 'CommentsIncluded']).agg({
    'Precision': 'mean',
    'Recall': 'mean',
    'F1': 'mean'
}).reset_index()

# Sort within each model by F1 score in descending order
zeroshot_grouped = zeroshot_grouped.sort_values(by=['Model', 'F1'], ascending=[True, False])
fewshot_grouped = fewshot_grouped.sort_values(by=['Model', 'F1'], ascending=[True, False])

# Save the sorted grouped results to CSV files
zeroshot_grouped.to_csv(zeroshot_output_path, index=False)
fewshot_grouped.to_csv(fewshot_output_path, index=False)

print(f"Zero-shot results saved to {zeroshot_output_path}")
print(f"Few-shot results saved to {fewshot_output_path}")
