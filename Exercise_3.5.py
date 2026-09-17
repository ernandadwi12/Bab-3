#Example 3.7 Naive Bayes Iris - Modified for Exercise 3.5
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

X, y = load_iris(return_X_y=True)

# Train the model
clf = GaussianNB()
clf.fit(X, y)

# Save the model to a file
filename = 'naivebayes_model.sav'
pickle.dump(clf, open(filename, 'wb'))
print("Model saved to", filename)

# Load the model from the file
loaded_model = pickle.load(open(filename, 'rb'))
print("Model loaded from", filename)

# Make a prediction with the loaded model
p = loaded_model.predict([[5.0, 3.4, 1.5, 0.4]])
print(p)
