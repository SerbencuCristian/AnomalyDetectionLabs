#ex2

import pyod.utils.data
import pyod.models.knn

x_train, x_test, y_train, y_test = pyod.utils.data.generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

model = pyod.models.knn.KNN(contamination=0.1)
model.fit(x_train)

for data, labels, name in ((x_train, y_train, "Training"), (x_test, y_test, "Testing")):
	predictions = model.predict(data)
	tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
	score = balanced_accuracy_score(labels, predictions)
	print(f"{name}: TN={tn}, TP={tp}, FN={fn}, FP={fp}, balanced accuracy={score:.3f}")

test_scores = model.decision_function(x_test)
fpr, tpr, _ = roc_curve(y_test, test_scores)
plt.plot(fpr, tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve (testing data)")
plt.show()

changed_model = pyod.models.knn.KNN(contamination=0.2)
changed_model.fit(x_train)
changed_predictions = changed_model.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, changed_predictions, labels=[0, 1]).ravel()
score = balanced_accuracy_score(y_test, changed_predictions)
print(f"Testing (contamination=0.2): TN={tn}, TP={tp}, FN={fn}, FP={fp}, balanced accuracy={score:.3f}")