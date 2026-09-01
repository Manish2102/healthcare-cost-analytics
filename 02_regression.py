"""
STEP 2: Regression — predict a NUMBER (medical cost) from patient features.

Regression = predicting a continuous value. Classification (next script) =
predicting a category. That's the entire conceptual difference; everything
else (train/test split, evaluation, avoiding overfitting) is shared.

We train two models on purpose, not one:
  - Linear Regression: the simplest possible model. If a complex model can't
    beat this by much, the complex model isn't earning its complexity.
  - Random Forest: an ensemble of decision trees. Usually stronger on
    tabular data with non-linear relationships (like "smoker" interacting
    with "age" and "bmi" here).

Comparing them is the actual skill — not just running one model and
reporting its score.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).parent
MODELS = HERE / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(HERE / "data" / "insurance.csv")

# Features (X) vs target (y). "charges" is what we're trying to predict,
# so it can NEVER appear in X — that would be leaking the answer to the model.
X = df.drop(columns=["charges"])
y = df["charges"]

# sex, smoker, region are text categories — models need numbers, not strings.
# OneHotEncoder turns e.g. region=[northeast,northwest,southeast,southwest]
# into 4 separate 0/1 columns. ColumnTransformer applies that ONLY to the
# categorical columns and leaves numeric columns (age, bmi, children) as-is.
CATEGORICAL = ["sex", "smoker", "region"]
NUMERIC = ["age", "bmi", "children"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), CATEGORICAL),
], remainder="passthrough")  # numeric columns pass through untouched

# Split BEFORE training. The model must never see the test set during
# fitting — that's the only way the test score tells us anything about
# how it'll perform on a real, unseen patient. random_state fixes the
# shuffle so results are reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n{name}")
    print(f"  MAE  (avg $ off by):      ${mae:,.2f}")
    print(f"  RMSE (penalizes big misses more): ${rmse:,.2f}")
    print(f"  R^2  (variance explained, 1.0=perfect): {r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

results = {}

print("=" * 60)
print("MODEL 1: Linear Regression (the baseline)")
print("=" * 60)
lin_pipeline = Pipeline([
    ("prep", preprocess),
    ("model", LinearRegression()),
])
lin_pipeline.fit(X_train, y_train)
results["linear_regression"] = evaluate("Linear Regression", lin_pipeline, X_test, y_test)

print()
print("=" * 60)
print("MODEL 2: Random Forest Regressor")
print("=" * 60)
rf_pipeline = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)),
])
rf_pipeline.fit(X_train, y_train)
results["random_forest"] = evaluate("Random Forest", rf_pipeline, X_test, y_test)

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
best_name = min(results, key=lambda k: results[k]["rmse"])
print(f"Lower RMSE wins: {best_name} (RMSE ${results[best_name]['rmse']:,.2f})")
print("Save this number — it's the '$X average prediction error' claim for the resume.")

# Save whichever pipeline is best — the WHOLE pipeline (encoder + model),
# so the API later can hand it raw patient data and get a prediction back
# without re-implementing the encoding logic.
best_pipeline = lin_pipeline if best_name == "linear_regression" else rf_pipeline
joblib.dump(best_pipeline, MODELS / "regression_model.joblib")
print(f"\nSaved best model to models/regression_model.joblib")
