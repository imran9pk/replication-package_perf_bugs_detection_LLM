# %%
# %%
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# %%
def get_correlated_features(dataset):
    correlation_matrix = dataset.corr()
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                colname = correlation_matrix.columns[min(i,j)]
                #print('(', correlation_matrix.columns[min(i,j)], ',', correlation_matrix.columns[max(i,j)], ')')
                correlated_features.add(colname)
    return correlated_features

def train_test_model(train_x, train_y, test_x, test_y, algorithm, param_dist):
    sc = MinMaxScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)
    clf = algorithm(**param_dist)
    
    clf.fit(train_x, train_y)

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(test_x)
        prob_pos = probs[:,1]
    else:  # use decision function
        prob_pos = clf.decision_function(test_x)
        prob_pos =  (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    
    
    fpr, tpr, thresholds = metrics.roc_curve(test_y, prob_pos, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    predictions = clf.predict(test_x)
    accuracy = metrics.accuracy_score(test_y, predictions)
    precision = metrics.precision_score(test_y, predictions, zero_division=0)
    recall = metrics.recall_score(test_y, predictions, zero_division=0)
    f1 = metrics.f1_score(test_y, predictions, zero_division=0)
    
    # Compute confusion matrix and extract TP, FP, TN, FN
    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()

    # return auc
    # Return all metrics
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def train_test_evaluation(current_proj_dataset, train_data_proj, test_data_proj, algorithm, param_dist):

    # Get rid of correlated features based on the whole dataset
    correlated_features = get_correlated_features(current_proj_dataset)
    train_data_proj = train_data_proj.drop(labels=correlated_features, axis=1)
    test_data_proj = test_data_proj.drop(labels=correlated_features, axis=1)

    # Separate features (X) and labels (y)
    train_x = train_data_proj.drop(['label'], axis=1)
    train_y = train_data_proj['label']
    test_x = test_data_proj.drop(['label'], axis=1)
    test_y = test_data_proj['label']

    #initialize metrics scores
    metrics_scores = {
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'tp': [],
        'fp': [],
        'tn': [],
        'fn': []
    }

    # Train and evaluate using train_test_model
    metrics_scores = train_test_model(train_x, train_y, test_x, test_y, algorithm, param_dist)

    return metrics_scores

# %%
proposed_features = ['num_if_inloop', 'num_loop_inif', 'num_nested_loop', 'num_nested_loop_incrit',
              'synchronization', 'thread', 'io_in_loop', 'database_in_loop', 'collection_in_loop',
              'io', 'database', 'collection', 'recursive']

algorithms = ['CNB', 'LR', 'DT', 'MLP', 'SVM', 'RF']
# algorithms = ['RF']

# %%
algo = {}
algo['CNB'] = ComplementNB
algo['LR'] = LogisticRegression
algo['DT'] = DecisionTreeClassifier
algo['MLP'] = MLPClassifier
algo['SVM'] = LinearSVC
algo['RF'] = RandomForestClassifier

algo_param_dist = {}
algo_param_dist['CNB'] = {}
algo_param_dist['LR'] = {}
algo_param_dist['DT'] = {}
algo_param_dist['SVM'] = {}
algo_param_dist['MLP'] = {}
algo_param_dist['CNB'] = {'alpha':0.001}
algo_param_dist['LR'] = {'max_iter': 100, 'class_weight': 'balanced'}
algo_param_dist['DT'] = { 'criterion':'entropy','max_features':0.3, 'max_depth': 6,
                          'min_samples_leaf':1, 'min_samples_split':2,
                          'random_state':0, 'class_weight':'balanced'}
algo_param_dist['MLP'] = {'max_iter':200,
                          'hidden_layer_sizes':7,
                          'shuffle': False,
                          'learning_rate': 'adaptive'}
algo_param_dist['SVM'] = {'max_iter':1000, 'class_weight':'balanced'}
algo_param_dist['RF'] = {'n_estimators':500, 'max_samples': 0.2, 'n_jobs': 2, 
              'criterion':'entropy','max_features':0.2,
              'min_samples_leaf':1, 'min_samples_split':2,
              'random_state':0, 'class_weight':'balanced_subsample',
              'verbose':0 }

# %%
results_df = pd.DataFrame(columns=['project', 'algorithm_metric','auc', 'accuracy', 'precision', 'recall', 'f1', 'tp', 'fp', 'tn', 'fn'])

skipped_projs = []
# temp_file_path = 'incremental_results_Test.csv'
# data_dir = 'data/experiment_dataset-preprocessed'

output_folder = "../data/output"
input_folder = "../data/input"

train_split_file = os.path.join(input_folder, 'train_data_per_project.csv')
test_split_file = os.path.join(input_folder, 'test_data_per_project.csv')
combined_dataset = os.path.join(input_folder, 'combined_dataset-processed.csv')

output_file = os.path.join(output_folder, 'with_without_anti_pattern_metrics_ML_Models.csv')


# Load the train and test split files
train_split = pd.read_csv(train_split_file)
test_split = pd.read_csv(test_split_file)
dataset = pd.read_csv(combined_dataset)

projects = test_split['Project_name'].unique()
projects = [x.strip().lower() for x in projects]
projects.sort()

# N=100

# %%
for project in projects:
    
    print(f"Processing project: {project}")

    # filter rows for the current project
    train_proj = train_split[train_split['Project_name'].str.strip().str.lower() == project]
    test_proj = test_split[test_split['Project_name'].str.strip().str.lower() == project]
    current_proj_dataset = dataset[dataset['Project_name'].str.strip().str.lower() == project]

    l1,l2 = len(train_proj), len(test_proj)
    if(l1 == 0 and l2 == 0):
        print(f"Project {project} skipped: No split available")
        continue

    # Skip evaluation if any split has no positive labels
    if all(train_proj['label'] == 0) or all(test_proj['label'] == 0):
        skipped_projs.append(project)
        print(f"Project {project} skipped: Insufficient positive labels in train/test splits")
        continue

    current_proj_dataset.loc[:, 'github_path'] = current_proj_dataset['github_path'].str.strip().str.lower()
    test_proj.loc[:, 'github_path'] = test_proj['github_path'].str.strip().str.lower()
    train_proj.loc[:, 'github_path'] = train_proj['github_path'].str.strip().str.lower()

    train_data_proj = current_proj_dataset[current_proj_dataset['github_path'].isin(train_proj['github_path'])]
    test_data_proj = current_proj_dataset[current_proj_dataset['github_path'].isin(test_proj['github_path'])]

    cols_drop = ['Project_name', 'File','github_path','token_size', 'file_available','token_size_codellama']
    train_data_proj = train_data_proj.drop(cols_drop, axis=1, errors='ignore')
    test_data_proj = test_data_proj.drop(cols_drop, axis=1, errors='ignore')
    current_proj_dataset = current_proj_dataset.drop(cols_drop, axis=1, errors='ignore')

    for algorithm in algorithms:
        print(f"Processing with {algorithm}")
        all_metrics_scores = train_test_evaluation(current_proj_dataset, train_data_proj, test_data_proj, algo[algorithm], algo_param_dist[algorithm])            

        row = {
            'project': project,
            'auc': np.mean(all_metrics_scores['auc']) if all_metrics_scores['auc'] else np.nan,
            'accuracy': np.mean(all_metrics_scores['accuracy']) if all_metrics_scores['accuracy'] else 0,
            'precision': np.mean(all_metrics_scores['precision']) if all_metrics_scores['precision'] else 0,
            'recall': np.mean(all_metrics_scores['recall']) if all_metrics_scores['recall'] else 0,
            'f1': np.mean(all_metrics_scores['f1']) if all_metrics_scores['f1'] else 0,
            'algorithm_metric': algorithm + ' with anti-pattern metrics',
            'tp': np.sum(all_metrics_scores['tp']),  
            'fp': np.sum(all_metrics_scores['fp']),  
            'tn': np.sum(all_metrics_scores['tn']),  
            'fn': np.sum(all_metrics_scores['fn'])   
        }
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        
        remain_dataset = current_proj_dataset.drop(labels=proposed_features, axis=1)
        defect_metrics_scores = train_test_evaluation(remain_dataset, train_data_proj, test_data_proj, algo[algorithm], algo_param_dist[algorithm])
        row = {
            'project': project,
            'auc': np.mean(defect_metrics_scores['auc']) if defect_metrics_scores['auc'] else np.nan,
            'accuracy': np.mean(defect_metrics_scores['accuracy']) if defect_metrics_scores['accuracy'] else 0,
            'precision': np.mean(defect_metrics_scores['precision']) if defect_metrics_scores['precision'] else 0,
            'recall': np.mean(defect_metrics_scores['recall']) if defect_metrics_scores['recall'] else 0,
            'f1': np.mean(defect_metrics_scores['f1']) if defect_metrics_scores['f1'] else 0,
            'algorithm_metric': algorithm + ' without_anti-pattern_metrics',
            'tp': np.sum(defect_metrics_scores['tp']),
            'fp': np.sum(defect_metrics_scores['fp']),
            'tn': np.sum(defect_metrics_scores['tn']),
            'fn': np.sum(defect_metrics_scores['fn'])
        }
        
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        
        print(f"******Project {project} Processed******")
# %%

results_df.to_csv(output_file, index=False)

# save skipped projects to a file
with open(os.path.join(output_folder,'skipped_projects_ML.txt'), 'w') as f:
    for item in skipped_projs:
        f.write("%s\n" % item)