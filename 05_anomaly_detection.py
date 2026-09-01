"""
STEP 5: Anomaly detection — which claims cost way more than their profile predicts?

This is the exact technique the Amgen JD calls out under "Analytics & Decision
Engines: anomaly detection (claims, pricing, Gross-to-Net)". The idea:

  1. Use the regression model from step 2 to predict "expected" cost for
     every patient based on their profile.
  2. Compare actual cost to expected cost. A patient costing 5x what their
     age/bmi/smoker-status would predict is an anomaly worth a human
     looking at (fraud? data error? a genuinely unusual medical case?).

We ALSO run Isolation Forest, a model built specifically for anomaly
detection, so we can compare "anomaly = big residual" (simple, explainable)
against "anomaly = what a dedicated algorithm flags" (more general, less
explainable). Showing both and comparing is the actual point — a single
anomaly-detection number in isolation proves nothing.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import IsolationForest

HERE = Path(__file__).parent
MODELS = HERE / "models"

df = pd.read_csv(HERE / "data" / "insurance.csv")

# ---- Method 1: residual-based (uses our own regression model) ----
regression_pipeline = joblib.load(MODELS / "regression_model.joblib")
X = df.drop(columns=["charges"])
df["predicted_charges"] = regression_pipeline.predict(X)
df["residual"] = df["charges"] - df["predicted_charges"]
df["residual_ratio"] = df["charges"] / df["predicted_charges"]

# Flag anything costing >2x what the model expected.
df["anomaly_residual"] = (df["residual_ratio"] > 2.0).astype(int)

print("=" * 60)
print("METHOD 1: Residual-based anomaly detection")
print("=" * 60)
print(f"Flagged {df['anomaly_residual'].sum()} of {len(df)} patients "
      f"({df['anomaly_residual'].mean()*100:.1f}%) as costing >2x their predicted amount.")
print("\nTop 5 biggest anomalies:")
print(
    df.sort_values("residual_ratio", ascending=False)
    [["age", "sex", "bmi", "smoker", "charges", "predicted_charges", "residual_ratio"]]
    .head(5)
    .round(2)
    .to_string(index=False)
)

# ---- Method 2: Isolation Forest (a model built for this) ----
# Isolation Forest works by randomly splitting the data repeatedly; outliers
# get isolated in fewer splits than normal points, because they're "further"
# from everything else. contamination=0.05 tells it to expect ~5% anomalies.
print()
print("=" * 60)
print("METHOD 2: Isolation Forest")
print("=" * 60)
numeric_features = df[["age", "bmi", "children", "charges"]]
iso = IsolationForest(contamination=0.05, random_state=42)
df["anomaly_isoforest"] = (iso.fit_predict(numeric_features) == -1).astype(int)
print(f"Flagged {df['anomaly_isoforest'].sum()} of {len(df)} patients "
      f"({df['anomaly_isoforest'].mean()*100:.1f}%) as anomalies.")

# ---- Compare the two methods ----
agreement = (df["anomaly_residual"] == df["anomaly_isoforest"]).mean()
both_flagged = ((df["anomaly_residual"] == 1) & (df["anomaly_isoforest"] == 1)).sum()
print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"The two methods agree on {agreement*100:.1f}% of all patients.")
print(f"{both_flagged} patients were flagged by BOTH methods — these are the")
print("highest-confidence anomalies, worth reviewing first in a real system.")

joblib.dump(iso, MODELS / "anomaly_model.joblib")
print(f"\nSaved Isolation Forest model to models/anomaly_model.joblib")
