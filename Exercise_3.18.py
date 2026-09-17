import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    explained_variance_score,
    r2_score,
)

housing = fetch_california_housing(as_frame=True)
df = housing.frame
df = df.rename(columns={"MedHouseVal": "MEDV"})

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

X_train = df_train.drop(columns=["MEDV"])
y_train = df_train["MEDV"]
X_test = df_test.drop(columns=["MEDV"])
y_test = df_test["MEDV"]

automl = AutoML()
automl.fit(
    X_train=X_train,
    y_train=y_train,
    task="regression",
    time_budget=30,
)

print()
print("Finished training the pipeline!")
print(f"Best estimator: {automl.best_estimator}")
print(f"Best config: {automl.best_config}")
print()

print("Here are the results from our best model")
print("predicting MEDV")
print()

try:
    importances = automl.feature_importances_
    feat_table = pd.DataFrame({
        "Feature Name": X_train.columns,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)
    print(feat_table.to_string(index=False))
except Exception:
    print("(Feature importance not available for this model type.)")

print()
print("***********************************************")
print("Advanced scoring metrics for the trained regression model on this")
print("particular dataset:")
print()

y_pred = automl.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

diffs = y_pred - y_test.values
positive_diffs = diffs[diffs > 0]
negative_diffs = diffs[diffs < 0]

print("Here is the overall RMSE for these predictions:")
print(rmse)
print()
print("Here is the average of the predictions:")
print(np.mean(y_pred))
print()
print("Here is the average actual value on this validation set:")
print(np.mean(y_test))
print()
print("Here is the median prediction:")
print(np.median(y_pred))
print()
print("Here is the median actual value:")
print(np.median(y_test))
print()
print("Here is the mean absolute error:")
print(mae)
print()
print("Here is the median absolute error (robust to outliers):")
print(medae)
print()
print("Here is the explained variance:")
print(explained_var)
print()
print("Here is the R-squared value:")
print(r2)
print()
print("Count of positive differences (prediction > actual):")
print(len(positive_diffs))
print("Count of negative differences:")
print(len(negative_diffs))
print("Average positive difference:")
print(positive_diffs.mean() if len(positive_diffs) > 0 else float("nan"))
print("Average negative difference:")
print(negative_diffs.mean() if len(negative_diffs) > 0 else float("nan"))
print()
print("***********************************************")