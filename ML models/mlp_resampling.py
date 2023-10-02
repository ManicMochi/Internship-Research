import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE


def compute_scores(model, X, y):
    scores_accuracy = cross_val_score(model, X, y, cv=9, scoring='accuracy')
    scores_f1_micro = cross_val_score(model, X, y, cv=9, scoring='f1_micro')
    scores_f1_macro = cross_val_score(model, X, y, cv=9, scoring='f1_macro')
    scores_precision_micro = cross_val_score(model, X, y, cv=9, scoring='precision_micro')
    scores_precision_macro = cross_val_score(model, X, y, cv=9, scoring='precision_macro')
    scores_recall_micro = cross_val_score(model, X, y, cv=9, scoring='recall_micro')
    scores_recall_macro = cross_val_score(model, X, y, cv=9, scoring='recall_macro')
    
    mean_accuracy = np.mean(scores_accuracy)
    var_accuracy = np.var(scores_accuracy)
    mean_f1_micro = np.mean(scores_f1_micro)
    var_f1_micro = np.var(scores_f1_micro)
    mean_f1_macro = np.mean(scores_f1_macro)
    var_f1_macro = np.var(scores_f1_macro)
    mean_precision_micro = np.mean(scores_precision_micro)
    var_precision_micro = np.var(scores_precision_micro)
    mean_precision_macro = np.mean(scores_precision_macro)
    var_precision_macro = np.var(scores_precision_macro)
    mean_recall_micro = np.mean(scores_recall_micro)
    var_recall_micro = np.var(scores_recall_micro)
    mean_recall_macro = np.mean(scores_recall_macro)
    var_recall_macro = np.var(scores_recall_macro)

    return (mean_accuracy, var_accuracy, 
            mean_f1_micro, var_f1_micro, mean_f1_macro, var_f1_macro,
            mean_precision_micro, var_precision_micro, mean_precision_macro, var_precision_macro,
            mean_recall_micro, var_recall_micro, mean_recall_macro, var_recall_macro)

#noresampling
#PC call
#df = pd.read_csv('C:/Users/chris/Downloads/VSCode/Internship_objects/datasets/creditcard.csv')

#Laptop call
df = pd.read_csv('C:/Users/chris/Documents/GitHub/Internship-Research/datasets/page_blocks.csv')

X = df.iloc[:, :-1]  
y = df.iloc[:, -1]  

scaler = StandardScaler()  
X = scaler.fit_transform(X)  

model = MLPClassifier(max_iter=1000)
model.fit(X, y)

#ADASYN resampling
X_ADASYN = X.copy()
y_ADASYN = y.copy()

class_counts = y.value_counts().to_dict()
min_instances_per_class = 100
sampling_strategy_dict = {key: min_instances_per_class for key, value in class_counts.items() if value < min_instances_per_class}
adasyn = ADASYN(sampling_strategy=sampling_strategy_dict, n_neighbors=4, random_state=42)
X_ADASYN, y_ADASYN = adasyn.fit_resample(X_ADASYN, y_ADASYN)

model_ADASYN = MLPClassifier()
model_ADASYN.fit(X_ADASYN, y_ADASYN)

#SMOTE resampling
X_SMOTE = X.copy()
y_SMOTE = y.copy()

smote = SMOTE(random_state=42)

X_SMOTE, y_SMOTE  = smote.fit_resample(X_SMOTE, y_SMOTE)

model_SMOTE = MLPClassifier(max_iter=1000)
model_SMOTE.fit(X_SMOTE, y_SMOTE)

#borderline SMOTE resampling
X_BSMOTE = X.copy()
y_BSMOTE = y.copy()

bsmote = BorderlineSMOTE()
X_BSMOTE, y_BSMOTE = bsmote.fit_resample(X_BSMOTE, y_BSMOTE)

model_BSMOTE = MLPClassifier(max_iter=1000)
model_BSMOTE.fit(X_BSMOTE, y_BSMOTE)

#SMOTEENN resampling
X_SMOTEENN = X.copy()
y_SMOTEENN = y.copy()

smoteenn = SMOTEENN()
X_SMOTEENN, y_SMOTEENN = smoteenn.fit_resample(X_SMOTEENN, y_SMOTEENN)

model_SMOTEENN = MLPClassifier(max_iter=1000)
model_SMOTEENN.fit(X_SMOTEENN, y_SMOTEENN)

#ClusterCentroids  resampling
X_CLUSTER = X.copy()
y_CLUSTER = y.copy()

cluster_centroids = ClusterCentroids(sampling_strategy='auto', random_state=42)
X_CLUSTER, y_CLUSTER = cluster_centroids.fit_resample(X_CLUSTER, y_CLUSTER)

model_CLUSTER = MLPClassifier(max_iter= 1000)
model_CLUSTER.fit(X_CLUSTER, y_CLUSTER)

#collecting scores
mean_accuracy, var_accuracy, mean_f1_micro, var_f1_micro, mean_f1_macro, var_f1_macro, mean_precision_micro, var_precision_micro, mean_precision_macro, var_precision_macro, mean_recall_micro, var_recall_micro, mean_recall_macro, var_recall_macro = compute_scores(model, X, y)
mean_accuracy_ADASYN, var_accuracy_ADASYN, mean_f1_micro_ADASYN, var_f1_micro_ADASYN, mean_f1_macro_ADASYN, var_f1_macro_ADASYN, mean_precision_micro_ADASYN, var_precision_micro_ADASYN, mean_precision_macro_ADASYN, var_precision_macro_ADASYN, mean_recall_micro_ADASYN, var_recall_micro_ADASYN, mean_recall_macro_ADASYN, var_recall_macro_ADASYN = compute_scores(model_ADASYN, X_ADASYN, y_ADASYN)
mean_accuracy_SMOTE, var_accuracy_SMOTE, mean_f1_micro_SMOTE, var_f1_micro_SMOTE, mean_f1_macro_SMOTE, var_f1_macro_SMOTE, mean_precision_micro_SMOTE, var_precision_micro_SMOTE, mean_precision_macro_SMOTE, var_precision_macro_SMOTE, mean_recall_micro_SMOTE, var_recall_micro_SMOTE, mean_recall_macro_SMOTE, var_recall_macro_SMOTE = compute_scores(model_SMOTE, X_SMOTE, y_SMOTE)
mean_accuracy_BSMOTE, var_accuracy_BSMOTE, mean_f1_micro_BSMOTE, var_f1_micro_BSMOTE, mean_f1_macro_BSMOTE, var_f1_macro_BSMOTE, mean_precision_micro_BSMOTE, var_precision_micro_BSMOTE, mean_precision_macro_BSMOTE, var_precision_macro_BSMOTE, mean_recall_micro_BSMOTE, var_recall_micro_BSMOTE, mean_recall_macro_BSMOTE, var_recall_macro_BSMOTE = compute_scores(model_BSMOTE, X_BSMOTE, y_BSMOTE)
mean_accuracy_SMOTEENN, var_accuracy_SMOTEENN, mean_f1_micro_SMOTEENN, var_f1_micro_SMOTEENN, mean_f1_macro_SMOTEENN, var_f1_macro_SMOTEENN, mean_precision_micro_SMOTEENN, var_precision_micro_SMOTEENN, mean_precision_macro_SMOTEENN, var_precision_macro_SMOTEENN, mean_recall_micro_SMOTEENN, var_recall_micro_SMOTEENN, mean_recall_macro_SMOTEENN, var_recall_macro_SMOTEENN = compute_scores(model_SMOTEENN, X_SMOTEENN, y_SMOTEENN)
mean_accuracy_CLUSTER, var_accuracy_CLUSTER, mean_f1_micro_CLUSTER, var_f1_micro_CLUSTER, mean_f1_macro_CLUSTER, var_f1_macro_CLUSTER, mean_precision_micro_CLUSTER, var_precision_micro_CLUSTER, mean_precision_macro_CLUSTER, var_precision_macro_CLUSTER, mean_recall_micro_CLUSTER, var_recall_micro_CLUSTER, mean_recall_macro_CLUSTER, var_recall_macro_CLUSTER = compute_scores(model_CLUSTER, X_CLUSTER, y_CLUSTER)

def print_scores(resampling_name, mean_accuracy, var_accuracy, mean_f1_micro, var_f1_micro, mean_f1_macro, var_f1_macro, mean_precision_micro, var_precision_micro, mean_precision_macro, var_precision_macro, mean_recall_micro, var_recall_micro, mean_recall_macro, var_recall_macro):
    print(resampling_name)
    print("Accuracy Mean:", mean_accuracy, "(Variance:", var_accuracy, ")")
    print("F1 Micro Mean:", mean_f1_micro, "(Variance:", var_f1_micro, ")")
    print("F1 Macro Mean:", mean_f1_macro, "(Variance:", var_f1_macro, ")")
    print("Mean Precicion Micro:", mean_precision_micro, "(Variance:", var_precision_micro, ")")
    print("Mean Precision Macro:", mean_precision_macro, "(Variance:", var_precision_macro, ")")
    print("Mean Recall Micro:", mean_recall_micro, "(Variance:", var_recall_micro, ")")
    print("Mean Recall Macro:", mean_recall_macro, "(Variance:", var_recall_macro, ")")

def all_scores():
    print()
    print("Multilayer Preceptron Model:")
    print()
    print_scores("No resampling technique", mean_accuracy, var_accuracy, mean_f1_micro, var_f1_micro, mean_f1_macro, var_f1_macro, mean_precision_micro, var_precision_micro, mean_precision_macro, var_precision_macro, mean_recall_micro, var_recall_micro, mean_recall_macro, var_recall_macro)
    print()
    print_scores("ADASYN", mean_accuracy_ADASYN, var_accuracy_ADASYN, mean_f1_micro_ADASYN, var_f1_micro_ADASYN, mean_f1_macro_ADASYN, var_f1_macro_ADASYN, mean_precision_micro_ADASYN, var_precision_micro_ADASYN, mean_precision_macro_ADASYN, var_precision_macro_ADASYN, mean_recall_micro_ADASYN, var_recall_micro_ADASYN, mean_recall_macro_ADASYN, var_recall_macro_ADASYN)
    print()
    print_scores("SMOTE", mean_accuracy_SMOTE, var_accuracy_SMOTE, mean_f1_micro_SMOTE, var_f1_micro_SMOTE, mean_f1_macro_SMOTE, var_f1_macro_SMOTE, mean_precision_micro_SMOTE, var_precision_micro_SMOTE, mean_precision_macro_SMOTE, var_precision_macro_SMOTE, mean_recall_micro_SMOTE, var_recall_micro_SMOTE, mean_recall_macro_SMOTE, var_recall_macro_SMOTE)
    print()
    print_scores("BSMOTE", mean_accuracy_BSMOTE, var_accuracy_BSMOTE, mean_f1_micro_BSMOTE, var_f1_micro_BSMOTE, mean_f1_macro_BSMOTE, var_f1_macro_BSMOTE, mean_precision_micro_BSMOTE, var_precision_micro_BSMOTE, mean_precision_macro_BSMOTE, var_precision_macro_BSMOTE, mean_recall_micro_BSMOTE, var_recall_micro_BSMOTE, mean_recall_macro_BSMOTE, var_recall_macro_BSMOTE)
    print()
    print_scores("SMOTEENN", mean_accuracy_SMOTEENN, var_accuracy_SMOTEENN, mean_f1_micro_SMOTEENN, var_f1_micro_SMOTEENN, mean_f1_macro_SMOTEENN, var_f1_macro_SMOTEENN, mean_precision_micro_SMOTEENN, var_precision_micro_SMOTEENN, mean_precision_macro_SMOTEENN, var_precision_macro_SMOTEENN, mean_recall_micro_SMOTEENN, var_recall_micro_SMOTEENN, mean_recall_macro_SMOTEENN, var_recall_macro_SMOTEENN)
    print()
    print_scores("Cluster Centroids", mean_accuracy_CLUSTER, var_accuracy_CLUSTER, mean_f1_micro_CLUSTER, var_f1_micro_CLUSTER, mean_f1_macro_CLUSTER, var_f1_macro_CLUSTER, mean_precision_micro_CLUSTER, var_precision_micro_CLUSTER, mean_precision_macro_CLUSTER, var_precision_macro_CLUSTER, mean_recall_micro_CLUSTER, var_recall_micro_CLUSTER, mean_recall_macro_CLUSTER, var_recall_macro_CLUSTER)

all_scores()

def plot_tsne_with_labels_and_save(X, y, title, save_filename):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y, cmap='viridis', marker='o')
    plt.title(title)
    plt.colorbar(label='Class Labels')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(save_filename)  # Save the figure as an image

# Example usage
plot_tsne_with_labels_and_save(X, y, "Multilayer Preceptron - Page Blocks No Resampling", "MLP_no_resampling.png")
plot_tsne_with_labels_and_save(X_ADASYN, y_ADASYN, "Multilayer Preceptron - Page Blocks ADASYN", "MLP_adasyn.png")
plot_tsne_with_labels_and_save(X_SMOTE, y_SMOTE, "Multilayer Preceptron - Page Blocks SMOTE", "MLP_smote.png")
plot_tsne_with_labels_and_save(X_BSMOTE, y_BSMOTE, "Multilayer Preceptron - Page Blocks BSMOTE", "MLP_bsmote.png")
plot_tsne_with_labels_and_save(X_SMOTEENN, y_SMOTEENN, "Multilayer Preceptron - Page Blocks SMOTEENN", "MLP_smoteenn.png")
plot_tsne_with_labels_and_save(X_CLUSTER, y_CLUSTER, "Multilayer Preceptron - Page Blocks CLUSTER", "MLP_cluster.png")