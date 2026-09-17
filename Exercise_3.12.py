#Exercise 3.12 - Modified Example 3.18: Multiple Linear Regression on Linnerud dataset
from sklearn import linear_model
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
import numpy as np

# Load the Linnerud dataset
# X = physiological measurements (Chins, Situps, Jumps)
# y = exercise measurements (Weight, Waist, Pulse)
linnerud = load_linnerud()
x = linnerud.data
y = linnerud.target

print("Feature names:", linnerud.feature_names)
print("Target names:", linnerud.target_names)
print("X shape:", x.shape)
print("y shape:", y.shape)

# Split into train/test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Fit multiple linear regression model (multi-output, since y has 3 columns)
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

print('\nCoefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)

# Predict on the test set
pred = reg.predict(x_test)
print('\nPredictions on test set: \n', pred)
print('Actual values: \n', y_test)

# R^2 score of the model
score = reg.score(x_test, y_test)
print('\nR^2 score: \n', score)

# Predict for a new sample (e.g. Chins=5, Situps=162, Jumps=60)
new_sample = [[5, 162, 60]]
new_pred = reg.predict(new_sample)
print('\nPrediction for new sample', new_sample, ':\n', new_pred)