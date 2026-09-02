# Vendor Reliability Scoring — Oil & Gas Procurement

A machine learning project that scores vendor reliability in oil & gas
procurement using a blend of **operational, financial, and behavioral**
signals — built at the intersection of psychology, economics, and
procurement domain experience.

## Why this project

Procurement teams usually score vendors on hard numbers: on-time delivery,
defect rates, payment compliance. But two vendors can look identical on
paper and still behave very differently — one is slow to respond to RFQs,
haggles on every quote, or triggers frequent disputes. These "soft"
behavioral patterns are exactly what psychology and behavioral economics
are built to treat as measurable, predictive signals rather than noise.

This project asks: **do lightweight behavioral features meaningfully sharpen
a vendor reliability score built mostly on operational data?**

## What it does

1. **`generate_data.py`** — generates a synthetic dataset of 450 vendors.
   Each vendor has a hidden ("latent") reliability trait — similar to how
   psychometrics models unobservable traits like conscientiousness through
   multiple noisy indicators — which drives observable operational,
   financial, and behavioral features.
2. **`modeling.py`** — trains and compares Linear Regression, Random
   Forest, and Gradient Boosting models to predict a vendor's composite
   reliability score from its raw features, and extracts feature
   importance.
3. **`app.py`** — an interactive Streamlit dashboard with a custom industrial
   theme (IBM Plex typography, a steel/rust/patina color system, a live gauge)
   to explore the vendor portfolio, filter by tier/category/country, and
   score a new or hypothetical vendor.

## A deliberate design choice: the non-linear "quality shock"

The composite reliability score isn't a purely linear formula. Once a
vendor's defect rate crosses a ~15% tolerance threshold, the score takes a
disproportionately steep hit — modeled after **prospect theory's loss
aversion**: trust in a vendor doesn't erode gradually as quality problems
mount, it drops off a cliff past a point. This gives the dataset genuine
non-linearity, which is exactly where the tree-based models
(Random Forest, Gradient Boosting) earn a real edge over the linear
baseline — a small but honest example of applying a psychological concept
to how the data itself is generated, not just to the write-up.

## Running it

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python generate_data.py

# 2. Train models and produce the feature importance plot
python modeling.py

# 3. Launch the dashboard
streamlit run app.py
```

## Data

All data is fully synthetic. No confidential vendor, tender, or company
information is used — the schema is modeled after realistic oil & gas
procurement fields (RFQ/PO cycles, vendor tiers, delivery/quality metrics)
but every value is generated, not real.

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, Matplotlib.
