
#ex1.py
import pyod.utils.data
import pyod.models.knn

import matplotlib.pyplot

x_train, x_test, y_train, y_test = pyod.utils.data.generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)

matplotlib.pyplot.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1],
            color="blue", label="Normal")
matplotlib.pyplot.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1],
            color="red", label="Outlier")

matplotlib.pyplot.xlabel("Feature 1")
matplotlib.pyplot.ylabel("Feature 2")
matplotlib.pyplot.title("Training Samples")
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

