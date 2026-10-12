#ex3

import pyod.utils.data
import pyod.models.knn
import numpy as np
from sklearn.metrics import balanced_accuracy_score

import matplotlib.pyplot
x_train, x_test, y_train, y_test = pyod.utils.data.generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)

z_scores = np.abs((x_train[:, 0] - np.mean(x_train[:, 0])) / np.std(x_train[:, 0]))

threshold = np.quantile(z_scores, 1 - 0.1)

y_pred = (z_scores > threshold).astype(int)

balanced_accuracy = balanced_accuracy_score(y_train, y_pred)
print(f"Z-score threshold: {threshold:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")

