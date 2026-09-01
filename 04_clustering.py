"""
STEP 4: Clustering — find groups WITHOUT being told what the groups are.

This is the key conceptual difference from steps 2 and 3: regression and
classification are "supervised" — we had a known right answer (charges,
high_cost) to train against. Clustering is "unsupervised" — there is no
label. We're asking K-Means to look at patient features alone and find
natural groupings, then we go look at what those groups actually mean.

K-Means needs to be told how many clusters (k) to find. Since we don't know
the "right" k in advance, we try several and use the elbow method + silhouette
score to justify the choice — this is the actual skill, not just calling
KMeans(n_clusters=3) because 3 felt like a nice number.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

HERE = Path(__file__).parent
MODELS = HERE / "models"
OUT = HERE / "outputs"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

df = pd.read_csv(HERE / "data" / "insurance.csv")
X = df.drop(columns=["charges"])  # cluster on patient profile, not on cost itself

CATEGORICAL = ["sex", "smoker", "region"]
# K-Means is distance-based, so unlike the tree models above, features on
# different scales (age: 18-64, bmi: ~15-53) would distort distances unless
# we standardize them first (mean=0, std=1).
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), CATEGORICAL),
    ("num", StandardScaler(), ["age", "bmi", "children"]),
])
X_processed = preprocess.fit_transform(X)

print("=" * 60)
print("FINDING k: elbow method + silhouette score")
print("=" * 60)
inertias, silhouettes = [], []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_processed)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_processed, labels)
    silhouettes.append(sil)
    print(f"  k={k}: inertia={km.inertia_:,.0f}  silhouette={sil:.3f}  "
          f"(higher silhouette = more distinct clusters)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_title("Elbow method (look for the bend)")
axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia")
axes[1].plot(list(k_range), silhouettes, marker="o", color="orange")
axes[1].set_title("Silhouette score (higher = better)")
axes[1].set_xlabel("k")
plt.tight_layout()
plt.savefig(OUT / "clustering_k_selection.png", dpi=120)
plt.close()
print(f"\nSaved: outputs/clustering_k_selection.png")

best_k = k_range[int(np.argmax(silhouettes))]
print(f"\nBest k by silhouette score: {best_k}")

final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = final_km.fit_predict(X_processed)

print()
print("=" * 60)
print(f"CLUSTER PROFILES (k={best_k})")
print("=" * 60)
profile = df.groupby("cluster").agg(
    count=("charges", "size"),
    avg_age=("age", "mean"),
    avg_bmi=("bmi", "mean"),
    pct_smoker=("smoker", lambda s: (s == "yes").mean()),
    avg_charges=("charges", "mean"),
).round(2)
print(profile)
print("\nRead this as: does each cluster tell a coherent story (e.g. 'older smokers,")
print("high cost' vs 'young non-smokers, low cost')? If yes, the clustering found")
print("something real. If the clusters look arbitrary, k or the features need rework.")

joblib.dump({"preprocess": preprocess, "kmeans": final_km}, MODELS / "clustering_model.joblib")
print(f"\nSaved model to models/clustering_model.joblib")
