#Exercise 3.10 - Classification on Scikit-Learn's diabetes data
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

names = [ "SVM", "Naive Bayes", "LDA",
"QDA", "Decision Tree", "Random Forest",
"Nearest Neighbors", "Neural Networks"]
classifiers = [
SVC(),
GaussianNB(),
LinearDiscriminantAnalysis(),
QuadraticDiscriminantAnalysis(reg_param=0.1),
DecisionTreeClassifier(),
RandomForestClassifier(),
KNeighborsClassifier(),
MLPClassifier(alpha=1, max_iter=1000)]

# load_diabetes returns a continuous target (disease progression score),
# so it must be converted into classes before it can be used for classification.
# Here we split it into 2 classes based on the median value:
# 1 = above median (higher progression), 0 = below median (lower progression)
X, y = load_diabetes(return_X_y=True)
y = (y > np.median(y)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name +": " + str(score))
    