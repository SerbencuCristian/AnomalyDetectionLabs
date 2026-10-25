import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF


X, _ = make_blobs(
	n_samples=[200, 100],
	centers=[(-10, -10), (10, 10)],
	cluster_std=[2, 6],
	random_state=42,
)

n_neighbors_values = [5, 20, 40]
colors = {0: "tab:blue", 1: "tab:red"}

for n_neighbors in n_neighbors_values:
	models = [
		("KNN", KNN(n_neighbors=n_neighbors, contamination=0.07)),
		("LOF", LOF(n_neighbors=n_neighbors, contamination=0.07, novelty=True)),
	]
	figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

	for axis, (name, model) in zip(axes, models):
		model.fit(X)
		for label, color in colors.items():
			points = X[model.labels_ == label]
			axis.scatter(
				points[:, 0], points[:, 1], color=color,
				label="Outlier" if label else "Inlier", s=24,
			)
		axis.set_title(f"{name} (n_neighbors={n_neighbors})")
		axis.set_xlabel("Feature 1")
		axis.set_ylabel("Feature 2")

	axes[0].legend()

plt.show()
