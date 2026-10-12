#ex4

import numpy as np
from sklearn.metrics import balanced_accuracy_score

rng = np.random.default_rng(42)
contamination = 0.1
n_samples = 1000
n_outliers = int(n_samples * contamination)
n_inliers = n_samples - n_outliers

mu = np.array([2.0, -1.0])
covariance = np.array([[2.0, 0.8],[0.8, 1.0],])
L = np.linalg.cholesky(covariance)

x_inliers = rng.standard_normal((n_inliers, 2))
inliers = x_inliers @ L.T + mu

x_outliers = rng.standard_normal((n_outliers, 2)) + np.array([5.0, 5.0])
outliers = x_outliers @ L.T + mu

data = np.vstack((inliers, outliers))
y_true = np.concatenate((np.zeros(n_inliers), np.ones(n_outliers)))

z_scores = (data - data.mean(axis=0)) / data.std(axis=0)
anomaly_scores = np.linalg.norm(z_scores, axis=1)

threshold = np.quantile(anomaly_scores, 1 - contamination)
y_pred = (anomaly_scores >= threshold).astype(int)

print(f"Z-score threshold: {threshold:.4f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")