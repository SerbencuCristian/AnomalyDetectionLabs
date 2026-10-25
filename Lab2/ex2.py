import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.utils.data import generate_data_clusters
from sklearn.metrics import balanced_accuracy_score


x_train, x_test, y_train, y_test = generate_data_clusters(
	n_train=400,
	n_test=200,
	n_features=2,
	n_clusters=2,
	contamination=0.1,
	random_state=42,
)
n_neighbors_values = [5, 10, 20, 40]
point_colors = {0: "tab:blue", 1: "tab:red"}




for n_neighbors in n_neighbors_values:
	model = KNN(n_neighbors=n_neighbors, contamination=0.1)
	model.fit(x_train)

	train_predictions = model.predict(x_train)
	test_predictions = model.predict(x_test)
	train_score = balanced_accuracy_score(y_train, train_predictions)
	test_score = balanced_accuracy_score(y_test, test_predictions)
	print(
		f"n_neighbors={n_neighbors}: "
		f"train balanced accuracy={train_score:.3f}, "
		f"test balanced accuracy={test_score:.3f}"
	)

	figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
	panels = [
		(axes[0, 0], x_train, y_train, "Training data: ground truth"),
		(axes[0, 1], x_train, train_predictions, "Training data: predictions"),
		(axes[1, 0], x_test, y_test, "Test data: ground truth"),
		(axes[1, 1], x_test, test_predictions, "Test data: predictions"),
	]

	for axis, data, labels, title in panels:
		for label, color in point_colors.items():
			selected = labels == label
			axis.scatter(
				data[selected, 0], data[selected, 1],
				color=color, s=24,
				label="Outlier" if label else "Inlier",
			)
		axis.set_title(title)
		axis.set_xlabel("Feature 1")
		axis.set_ylabel("Feature 2")

	axes[0, 0].legend()
	figure.suptitle(f"KNN anomaly detection (n_neighbors={n_neighbors})")

plt.show()
