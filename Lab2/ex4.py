from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyod.models.combination import average, maximization
from pyod.models.knn import KNN
from pyod.utils.utility import standardizer


data = loadmat(Path(__file__).with_name("cardio.mat"))
X = data["X"]
y = data["y"].ravel().astype(int)
contamination = np.mean(y == 1)

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

n_neighbors_values = range(30, 121, 10)
train_scores = []
test_scores = []

for n_neighbors in n_neighbors_values:
	model = KNN(n_neighbors=n_neighbors, contamination=contamination)
	model.fit(X_train)
	train_scores.append(model.decision_scores_)
	test_scores.append(model.decision_function(X_test))

	train_predictions = model.labels_
	test_predictions = model.predict(X_test)
	print(
		f"KNN (n_neighbors={n_neighbors}): "
		f"train BA={balanced_accuracy_score(y_train, train_predictions):.4f}, "
		f"test BA={balanced_accuracy_score(y_test, test_predictions):.4f}"
	)

train_scores = np.column_stack(train_scores)
test_scores = np.column_stack(test_scores)
train_scores, test_scores = standardizer(train_scores, test_scores)

for name, combine in (("Average", average), ("Maximization", maximization)):
	combined_train = combine(train_scores)
	combined_test = combine(test_scores)
	threshold = np.quantile(combined_train, 1 - contamination)
	train_predictions = (combined_train > threshold).astype(int)
	test_predictions = (combined_test > threshold).astype(int)
	print(
		f"{name}: "
		f"train BA={balanced_accuracy_score(y_train, train_predictions):.4f}, "
		f"test BA={balanced_accuracy_score(y_test, test_predictions):.4f}"
	)
