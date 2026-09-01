"""
STEP 1: Exploratory Data Analysis (EDA)

Before training any model, look at the data. The goal here is to answer:
  - What does each column actually mean, and what range/type is it?
  - Is the target (charges) skewed? (Regression models care about this.)
  - Which features actually correlate with charges? (Tells us what the model
    will likely lean on, and lets us sanity-check its predictions later.)

Nothing here trains a model. This is purely "understand the data" work —
skipping it is the #1 reason people build models that quietly do the wrong thing.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files instead of popping up a window
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(Path(__file__).parent / "data" / "insurance.csv")

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print(df.head())
print()
print("=" * 60)
print("DTYPES & MISSING VALUES")
print("=" * 60)
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

print()
print("=" * 60)
print("NUMERIC SUMMARY")
print("=" * 60)
print(df.describe())

# charges is our regression target. Real-world cost/price data is almost
# always right-skewed (most people cost little, a few cost a LOT) — this
# matters because it can violate assumptions of plain linear regression.
print()
print("=" * 60)
print("TARGET (charges) SKEW:", round(df["charges"].skew(), 3))
print("=" * 60)
print("(0 = symmetric, >1 = strongly right-skewed. This tells us later whether")
print(" to log-transform the target before fitting a linear model.)")

import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["charges"], kde=True, ax=axes[0])
axes[0].set_title("charges (raw)")
sns.histplot(np.log1p(df["charges"]), kde=True, ax=axes[1])
axes[1].set_title("log(charges) — closer to normal?")
plt.tight_layout()
plt.savefig(OUT / "charges_distribution.png", dpi=120)
plt.close()
print(f"\nSaved: outputs/charges_distribution.png")

# Correlation with charges — only numeric columns participate directly;
# categoricals (sex, smoker, region) need encoding first, which we'll do
# properly in the modeling scripts. Here we just eyeball smoker vs non-smoker
# since it's famously the single biggest driver of medical cost.
print()
print("=" * 60)
print("NUMERIC CORRELATION WITH charges")
print("=" * 60)
print(df.corr(numeric_only=True)["charges"].sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="smoker", y="charges", ax=ax)
ax.set_title("charges by smoker status")
plt.tight_layout()
plt.savefig(OUT / "charges_by_smoker.png", dpi=120)
plt.close()
print(f"Saved: outputs/charges_by_smoker.png")

print()
print("=" * 60)
print("MEAN charges BY smoker STATUS")
print("=" * 60)
print(df.groupby("smoker")["charges"].mean().round(2))
