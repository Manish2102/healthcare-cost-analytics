"""
STEP 3: Classification — predict a CATEGORY (high-cost risk: yes/no).

The insurance dataset has no "risk" label — real claims systems do (a claim
either exceeded a cost threshold or it didn't), so we derive one: anyone in
the top 25% of charges is "high-cost". This is a completely standard way to
turn a regression problem into a classification one when you care about a
decision (flag this patient for review) rather than an exact number.

Why two models again? Logistic Regression gives clean, interpretable
probabilities and coefficients. Random Forest usually classifies better but
is more of a black box. In a real risk-flagging system you'd want to be able
to explain WHY someone got flagged — that's a genuine trade-off, not just
"pick whichever scores higher."
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

HERE = Path(__file__).parent
MODELS = HERE / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(HERE / "data" / "insurance.csv")

# Derive the label: top 25% of charges = high-cost-risk = 1, else 0.
threshold = df["charges"].quantile(0.75)
df["high_cost"] = (df["charges"] > threshold).astype(int)
print(f"High-cost threshold: ${threshold:,.2f}")
print(f"Class balance: {df['high_cost'].value_counts(normalize=True).round(3).to_dict()}")
print("(Roughly 75/25 split — imbalanced enough that plain accuracy alone would be")
print(" misleading: a model that always predicts 'not high-cost' would already score ~75%.")
print(" That's exactly why we also look at precision/recall/F1 below, not just accuracy.)")

X = df.drop(columns=["charges", "high_cost"])
y = df["high_cost"]

CATEGORICAL = ["sex", "smoker", "region"]
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), CATEGORICAL),
], remainder="passthrough")

# stratify=y keeps the same 75/25 class balance in both train and test sets —
# without it, a random split could accidentally put almost all high-cost
# cases in one side.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"\n{name}")
    print(f"  Accuracy:  {accuracy_score(y_test, preds):.3f}  (% predicted correctly)")
    print(f"  Precision: {precision_score(y_test, preds):.3f}  (of flagged patients, % actually high-cost)")
    print(f"  Recall:    {recall_score(y_test, preds):.3f}  (of actual high-cost patients, % we caught)")
    print(f"  F1:        {f1_score(y_test, preds):.3f}  (balance of precision & recall)")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, probs):.3f}  (ranking quality, 0.5=random, 1.0=perfect)")
    print(f"  Confusion matrix [[TN FP][FN TP]]:\n{confusion_matrix(y_test, preds)}")
    return f1_score(y_test, preds)

print()
print("=" * 60)
print("MODEL 1: Logistic Regression")
print("=" * 60)
log_pipeline = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=1000)),
])
log_pipeline.fit(X_train, y_train)
log_f1 = evaluate("Logistic Regression", log_pipeline, X_test, y_test)

print()
print("=" * 60)
print("MODEL 2: Random Forest Classifier")
print("=" * 60)
rf_pipeline = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42, max_depth=6)),
])
rf_pipeline.fit(X_train, y_train)
rf_f1 = evaluate("Random Forest", rf_pipeline, X_test, y_test)

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
best_pipeline, best_name, best_f1 = (
    (rf_pipeline, "random_forest", rf_f1) if rf_f1 >= log_f1
    else (log_pipeline, "logistic_regression", log_f1)
)
print(f"Higher F1 wins: {best_name} (F1={best_f1:.3f})")

joblib.dump(best_pipeline, MODELS / "classification_model.joblib")
joblib.dump(threshold, MODELS / "high_cost_threshold.joblib")
print(f"\nSaved best model to models/classification_model.joblib")
