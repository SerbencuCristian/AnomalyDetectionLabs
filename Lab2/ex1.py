import numpy as np
import matplotlib.pyplot as plt


MEANS = (-2.0, 0.0, 2.0)
NOISE_VARIANCES = (0.05, 0.5, 2.0)
POINTS_PER_TYPE = 40
TOP_COUNT = 5

POINT_TYPES = [
	("Regular", 0.8, 1.0, "tab:blue"),
	("High variance on x", 4.0, 1.0, "tab:orange"),
	("High variance on y", 0.8, 4.0, "tab:green"),
	("High variance on both", 4.0, 4.0, "tab:red"),
]


def calculate_leverage(features):
	design_matrix = np.column_stack((np.ones(features.shape[0]), features))
	orthogonal_basis, _ = np.linalg.qr(design_matrix, mode="reduced")
	return np.sum(orthogonal_basis**2, axis=1)


def generate_data(rng, dimensions, mean, noise_variance):
	features_by_type = []
	responses_by_type = []
	labels = []

	coefficients = np.arange(1, dimensions + 1, dtype=float)
	for point_type, feature_scale, noise_scale, _ in POINT_TYPES:
		features = rng.normal(
			loc=0.0,
			scale=feature_scale,
			size=(POINTS_PER_TYPE, dimensions),
		)
		noise = rng.normal(
			loc=mean,
			scale=np.sqrt(noise_variance) * noise_scale,
			size=POINTS_PER_TYPE,
		)
		responses = features @ coefficients + 1.0 + noise

		features_by_type.append(features)
		responses_by_type.append(responses)
		labels.extend([point_type] * POINTS_PER_TYPE)

	features = np.vstack(features_by_type)
	responses = np.concatenate(responses_by_type)
	labels = np.asarray(labels)
	leverage = calculate_leverage(features)
	return features, responses, labels, leverage


def plot_experiment(dimensions, seed):
	rng = np.random.default_rng(seed)
	figure, axes = plt.subplots(
		len(NOISE_VARIANCES), len(MEANS), figsize=(15, 12),
		constrained_layout=True,
		subplot_kw={"projection": "3d"} if dimensions == 2 else {},
	)

	for row, noise_variance in enumerate(NOISE_VARIANCES):
		for column, mean in enumerate(MEANS):
			features, responses, labels, leverage = generate_data(
				rng, dimensions, mean, noise_variance
			)
			top_indices = np.argsort(leverage)[-TOP_COUNT:]
			axis = axes[row, column]

			for point_type, _, _, color in POINT_TYPES:
				selected = labels == point_type
				if dimensions == 1:
					axis.scatter(
						features[selected, 0], responses[selected],
						color=color, alpha=0.65, s=24, label=point_type,
					)
				else:
					axis.scatter(
						features[selected, 0], features[selected, 1],
						responses[selected], color=color, alpha=0.65,
						s=24, label=point_type,
					)

			if dimensions == 1:
				axis.scatter(
					features[top_indices, 0], responses[top_indices],
					marker="*", color="black", s=150, zorder=5,
					label=f"Top {TOP_COUNT} leverage" if row == column == 0 else None,
				)
			else:
				axis.scatter(
					features[top_indices, 0], features[top_indices, 1],
					responses[top_indices], marker="*", color="black", s=150,
					label=f"Top {TOP_COUNT} leverage" if row == column == 0 else None,
				)

			axis.set_title(f"$\\mu$={mean:g}, $\\sigma^2$={noise_variance:g}")
			axis.set_xlabel("$x_1$")
			if dimensions == 1:
				axis.set_ylabel("$y$")
			else:
				axis.set_ylabel("$x_2$")
				axis.set_zlabel("$y$")

	figure.suptitle(f"Leverage scores with {dimensions} predictor(s)")
	figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3)


def main():
	plot_experiment(dimensions=1, seed=2026)
	plot_experiment(dimensions=2, seed=2027)
	plt.show()


if __name__ == "__main__":
	main()
