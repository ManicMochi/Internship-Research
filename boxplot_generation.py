import matplotlib.pyplot as plt
import numpy as np

# Define the data for different resampling techniques
techniques = ["No resampling", "ADASYN", "SMOTE", "BSMOTE", "SMOTEENN", "Cluster Centroids"]

# Create a dictionary to hold accuracy data for each metric
metrics = {
    "Accuracy": [0.7294946768630979, 0.7118758434547908, 0.7300785634118967, 0.7251402918069584, 0.9179894179894179, 0.65625],
    "F1 Micro": [0.6886339781076622, 0.7207977207977208, 0.71986531986532, 0.7197530864197531, 0.911111111111111, 0.6770833333333334],
    "F1 Macro": [0.7033554880647973, 0.7227670014603386, 0.7436036414279545, 0.7345841335244132, 0.882281981482188, 0.6811134906360538],
    "Mean Precision Micro": [0.7061028639976008, 0.7061778377567851, 0.7348484848484849, 0.7424242424242424, 0.8957671957671958, 0.6631944444444444],
    "Mean Precision Macro": [0.7120674772184802, 0.7313206653425992, 0.7234515496980536, 0.7438231318770676, 0.9181437389770724, 0.6750425591111866],
    "Mean Recall Micro": [0.7180236917079024, 0.7033288349077823, 0.7249719416386083, 0.7373176206509541, 0.9185185185185184, 0.6840277777777778],
    "Mean Recall Macro": [0.7067001537110233, 0.6911231884057971, 0.7324330259112867, 0.7325428194993413, 0.8962962962962963, 0.6666666666666666],
}

# Create positions for the boxes
positions = np.arange(1, len(techniques) + 1)

# Create a box plot for each metric
plt.figure(figsize=(12, 8))
for metric, data in metrics.items():
    plt.boxplot(data, vert=False, labels=techniques, positions=positions)
    positions += len(techniques) + 1  # Separate the box plots horizontally

plt.title("Box Plots for Multilayer Perceptron Model Metrics by Resampling Technique")
plt.xlabel("Metric Value")
plt.xticks(np.arange(1, len(techniques) * len(metrics) + 1), techniques * len(metrics), rotation=45)
plt.legend(metrics.keys())
plt.show()
