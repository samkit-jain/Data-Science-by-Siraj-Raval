from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
P = [[190, 70, 43]]

classifiers = {
	'Decision Tree': DecisionTreeClassifier(),
	'K-Neighbors': KNeighborsClassifier(),
	'SVM': SVC(),
}

accuracy_list = []

for k, v in classifiers.items():
	print("----" + k + "----")
	clf = v
	clf = clf.fit(X, Y)

	prediction = clf.predict(P)
	total_prediction = clf.predict(X)

	accuracy = accuracy_score(Y, total_prediction) # or use clf.score(X, Y)
	accuracy = accuracy * 100

	accuracy_list.append(accuracy)

	print("Prediction for", P[0], "-", prediction[0])
	print("Accuracy over whole dataset -", round(accuracy, 2), "%\n")

	# using same dict for storing accuracy
	classifiers[k] = accuracy

max_accuracy = max(classifiers.values())
max_clf = [k for k, v in classifiers.items() if v == max_accuracy][0]

print("Highest accuracy of", max_accuracy, "% attained by", max_clf)