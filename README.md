# Healthcare Cost Analytics

A single dataset, four classical ML techniques: **regression**, **classification**,
**clustering**, and **anomaly detection**. Built to actually learn each one, not just
to have a project to list.

## Dataset

[`data/insurance.csv`](data/insurance.csv) — the public "Medical Cost Personal Datasets"
(1,338 rows: age, sex, bmi, children, smoker, region, charges). Originally compiled for
the book *Machine Learning with R* (Brett Lantz); this copy is from
[stedy/Machine-Learning-with-R-datasets](https://github.com/stedy/Machine-Learning-with-R-datasets).
It's a well-known public dataset used here for learning — not real claims/pricing data.

## What each script teaches

| Script | Technique | Question it answers |
| --- | --- | --- |
| `01_eda.py` | Exploratory analysis | What actually drives medical cost in this data? |
| `02_regression.py` | Linear Regression + Random Forest | Given a patient's profile, predict their cost |
| `03_classification.py` | Logistic Regression + Random Forest | Is this patient high-cost-risk (yes/no)? |
| `04_clustering.py` | K-Means | Are there natural patient segments, with no labels given? |
| `05_anomaly_detection.py` | Isolation Forest | Which claims cost far more than their profile predicts? |
| `06_api.py` | FastAPI | Serve the trained models behind real endpoints |

Run them in order — each one saves its trained model to `models/` and its plots to `outputs/`,
which the later scripts and the API load back in.

## Setup

```bash
cd 01-healthcare-cost-analytics
../.venv/Scripts/python.exe 01_eda.py
../.venv/Scripts/python.exe 02_regression.py
../.venv/Scripts/python.exe 03_classification.py
../.venv/Scripts/python.exe 04_clustering.py
../.venv/Scripts/python.exe 05_anomaly_detection.py
../.venv/Scripts/python.exe -m uvicorn 06_api:app --reload
```

## Findings

- **Smoking status dominates cost**: non-smokers average $8,434, smokers average $32,050
  (~4x) — by far the single biggest driver, ahead of age or BMI individually.
- **Regression**: Random Forest (R²=0.873, MAE=$2,524) beat plain Linear Regression
  (R²=0.784, MAE=$4,181) — the cost relationship isn't purely linear, likely because
  smoking interacts with age/BMI rather than adding independently.
- **Classification**: flagging the top-25%-cost patients as "high-risk," Random Forest
  hit 92.2% accuracy / F1=0.817, catching ~70% of actual high-cost patients while
  rarely false-flagging (precision 0.98). Logistic Regression had a *higher* ROC-AUC
  (0.864 vs 0.837) despite lower F1 — a reminder that F1 is threshold-specific while
  ROC-AUC measures ranking quality across all thresholds.
- **Clustering**: best silhouette score was only 0.214 (k=3) — a real but soft
  age-based grouping (young/mid/older), not sharply separated segments. Reported
  honestly rather than overstated.
- **Anomaly detection**: two independent methods (residual-based vs Isolation Forest)
  each flagged ~5% of patients, but only overlapped on 1 of 127 total flagged patients
  (~0.8% true overlap) — the naive "90.6% agreement" figure is misleading because it's
  inflated by both methods correctly agreeing on the 95% of *normal* patients.
- **Deployed** all four models behind a FastAPI service (`/predict/cost`, `/predict/risk`,
  `/segment`, `/predict/anomaly`) — verified end-to-end with real requests.
