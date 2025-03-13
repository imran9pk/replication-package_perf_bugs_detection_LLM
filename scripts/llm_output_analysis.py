# %%
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# %%
def get_projects_with_zero_positive_labels(ml_test_data_cleaned):
    test_df = pd.read_csv(ml_test_data_cleaned)

    projects = test_df['Project_name'].unique()
    projects_with_zero_positive_labels = []

    for project in projects:
        test_proj = test_df[test_df['Project_name'] == project]
        if all(test_proj['label'] == 0):
            projects_with_zero_positive_labels.append(project.strip().lower())
    return projects_with_zero_positive_labels


# %%
input_folder = "../data/input"
output_folder = "../data/output"

ml_test_data_cleaned = os.path.join(input_folder, 'test_data_per_project.csv')
combined_LLM_outputs_path = os.path.join(output_folder,'combined_LLM_outputs_cleaned.csv')

output_summary_path = os.path.join(output_folder,'summary_llms.csv')
output_metrics_path = os.path.join(output_folder,'project_wise_metrics.csv')

combined_LLM_outputs_df = pd.read_csv(combined_LLM_outputs_path)

# Fill missing predictions with interpreted values
if 'Interpreted' in combined_LLM_outputs_df.columns:
        combined_LLM_outputs_df['codellama_prediction'] = combined_LLM_outputs_df['codellama_prediction'].fillna(combined_LLM_outputs_df['Interpreted'])


#count unique projects in the combined_LLM_outputs_df
projects = combined_LLM_outputs_df['Project_name'].unique()
print(f"Number of unique projects in the combined LLM outputs: {len(projects)}")

# Filter out projects with zero positive labels in ML Test set
skipped_projects = get_projects_with_zero_positive_labels(ml_test_data_cleaned)
combined_LLM_outputs_df['Project_name'] = combined_LLM_outputs_df['Project_name'].str.strip().str.lower()

combined_LLM_outputs_df = combined_LLM_outputs_df[~combined_LLM_outputs_df['Project_name'].isin(skipped_projects)]
print(f"Skipped projects with zero positive labels in ML Test set: {skipped_projects}")

projects = combined_LLM_outputs_df['Project_name'].unique()
print(f"Number of unique projects in the combined LLM outputs: {len(projects)}")

# Initialize a summary list
summary = []
# Process the combined dataframe grouped by Model, CommentsIncluded, and Prompting
grouped_by_configs = combined_LLM_outputs_df.groupby(['Model', 'CommentsIncluded', 'Prompting'])
for (model_name, comments_included, prompting), group_df in grouped_by_configs:

    # Metrics
    total_inferences = len(group_df)
    responses = group_df['codellama_response'].notna().sum()
    yes_no = group_df['codellama_prediction'].notna().sum()
    heuristic = group_df['Interpreted'].notna().sum() if 'Interpreted' in group_df.columns else 0

    # Ground truth label counts
    label_counts = group_df['label'].value_counts().to_dict()
    gt_1 = label_counts.get(1, 0)
    gt_0 = label_counts.get(0, 0)

    # Prediction counts
    prediction_counts = group_df['codellama_prediction'].value_counts().to_dict()
    pred_1 = prediction_counts.get(1, 0)
    pred_0 = prediction_counts.get(0, 0)

    # Precision, Recall, F1, and Confusion Matrix
    precision = recall = f1 = tn = fp = fn = tp = None

    filtered_df = group_df.dropna(subset=['codellama_prediction'])
    labels = filtered_df['label']
    predictions = filtered_df['codellama_prediction']

    if not filtered_df.empty:
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    # Append summary information
    summary.append({
        'Model': model_name,
        'CommentsIncluded': comments_included,  # Boolean
        'Prompting': prompting,                # fewShot or zeroShot
        'Total Inferences': total_inferences,
        'Responses': responses,
        'Yes/No': yes_no,
        'Heuristic': heuristic,
        'GT_true': gt_1,
        'GT_false': gt_0,
        'Predicted_true': pred_1,
        'Predicted_false': pred_0,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    })

# Initialize project_wise_metrics list
project_wise_metrics = []
grouped_by_configs_projects = combined_LLM_outputs_df.groupby(['Model', 'CommentsIncluded', 'Prompting', 'Project_name'])
for (model_name, comments_included, prompting, project_name), group_df in grouped_by_configs_projects:
    # Drop rows with missing predictions
    filtered_group = group_df.dropna(subset=['codellama_prediction'])

    if not filtered_group.empty:
        # Extract labels and predictions
        labels = filtered_group['label']
        predictions = filtered_group['codellama_prediction']

        # Calculate metrics
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

        # Append results to project_wise_metrics
        project_wise_metrics.append({
            'Project': project_name,
            'Model': model_name,
            'CommentsIncluded': True if comments_included else False,
            'Prompting': prompting,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })


# Save the summary as a CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv(output_summary_path, index=False)
print(f"Summary table has been saved to {output_summary_path}")

# Save project_wise_metrics to a csv
metrics_df = pd.DataFrame(project_wise_metrics)
metrics_df.to_csv(output_metrics_path, index=False)
print(f"Project-wise F1 scores saved to {output_metrics_path}")




