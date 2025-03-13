import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plots_path = r"../data/output/figures"
csv_path = r"../data/output/project_wise_metrics.csv"

df = pd.read_csv(csv_path, usecols=["Project", "Model", "CommentsIncluded", "Prompting", "F1 Score"])

# Filter for CommentsIncluded = TRUE and Prompting = zeroShot and fewShot
df_filtered = df[(df["CommentsIncluded"] == True) & (df["Prompting"] == "zeroShot")]
fewshot_filtered = df[(df["CommentsIncluded"] == True) & (df["Prompting"] == "fewShot")]

# Identify projects where all F1 Scores are zero
projects_to_drop = df_filtered.groupby("Project")["F1 Score"].sum()
projects_to_drop = projects_to_drop[projects_to_drop == 0].index  # Projects where sum of F1 scores is 0

# Remove those projects
df_filtered = df_filtered[~df_filtered["Project"].isin(projects_to_drop)]

# Sort projects by highest F1 Score in descending order
sorted_projects = df_filtered.sort_values(by="F1 Score", ascending=False)["Project"].unique()

# Convert Project column to categorical type with sorted order
df_filtered["Project"] = pd.Categorical(df_filtered["Project"], categories=sorted_projects, ordered=True)

# Custom legend mapping
custom_legend_labels = {
    "artigenz": "Artigenz Coder 6.7b",
    "codellama": "CodeLlama 7b instruct",
    "qwen": "Qwen Coder 7b instruct"
}

# Replace model names dynamically
df_filtered["Model"] = df_filtered["Model"].map(lambda x: custom_legend_labels.get(x, x))

# Plot: Grouped Bar Chart Comparing F1 Scores Across Projects for Different Models
plt.figure(figsize=(12, 6))  # Adjusted figure size
sns.barplot(data=df_filtered, x="Project", y="F1 Score", hue="Model", dodge=True)

# Improve readability
plt.xticks(rotation=45, ha="right", fontsize=12, rotation_mode="anchor")  # Rotate labels for better visibility
plt.ylabel("F1 Score", fontsize=14)
plt.xlabel("Project", fontsize=14)

# Adjust legend position to avoid overlapping with bars
plt.legend(title="Model", bbox_to_anchor=(1, 1), loc="upper right", fontsize=12, frameon=True)

# Reduce excessive whitespace
plt.tight_layout()
plt.savefig(os.path.join(plots_path,'f1-zeroshot-commetns.pdf'), format='pdf',bbox_inches='tight')



# Plotting for fewshot
df_filtered = fewshot_filtered

# Identify projects where all F1 Scores are zero
projects_to_drop = df_filtered.groupby("Project")["F1 Score"].sum()
projects_to_drop = projects_to_drop[projects_to_drop == 0].index  # Projects where sum of F1 scores is 0

# Remove those projects
df_filtered = df_filtered[~df_filtered["Project"].isin(projects_to_drop)]

# Sort projects by highest F1 Score in descending order
sorted_projects = df_filtered.sort_values(by="F1 Score", ascending=False)["Project"].unique()

# Convert Project column to categorical type with sorted order
df_filtered["Project"] = pd.Categorical(df_filtered["Project"], categories=sorted_projects, ordered=True)

# Custom legend mapping
custom_legend_labels = {
    "artigenz": "Artigenz Coder 6.7b",
    "codellama": "CodeLlama 7b instruct",
    "qwen": "Qwen Coder 7b instruct"
}

# Replace model names dynamically
df_filtered["Model"] = df_filtered["Model"].map(lambda x: custom_legend_labels.get(x, x))

# Plot: Grouped Bar Chart Comparing F1 Scores Across Projects for Different Models
plt.figure(figsize=(12, 6))  # Adjusted figure size
sns.barplot(data=df_filtered, x="Project", y="F1 Score", hue="Model", dodge=True)

# Improve readability
plt.xticks(rotation=45, ha="right", fontsize=12, rotation_mode="anchor")  # Rotate labels for better visibility
plt.ylabel("F1 Score", fontsize=14)
plt.xlabel("Project", fontsize=14)

# Adjust legend position to avoid overlapping with bars
plt.legend(title="Model", bbox_to_anchor=(1, 1), loc="upper right", fontsize=12, frameon=True)

# Reduce excessive whitespace
plt.tight_layout()
plt.savefig(os.path.join(plots_path,'f1-fewshot-commetns.pdf'), format='pdf',bbox_inches='tight')