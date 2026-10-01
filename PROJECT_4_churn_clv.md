# PROJECT 4: Customer Churn & Lifetime Value Prediction (Data Science)

> Agent brief. Read fully before acting. Work in the repo root. Follow "Working rules" at the bottom.

## 1. Business problem
Which customers are about to lapse, and is a retention offer worth its cost? The output is a **targeting rule** backed by expected value, not just a model score.

What makes this project different from a standard classification project: (a) churn is not given, it must be **defined from behavior**; (b) validation is **temporal**; (c) the final decision uses **business costs**.

## 2. Data (downloaded manually into `data/raw/`)
UCI "Online Retail II": `online_retail_II.xlsx` with two sheets (2009-12 to 2010-12 and 2010-12 to 2011-12). Columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country.

Cleaning rules (log row counts removed at each step):
- Drop rows with missing Customer ID (cannot be tracked).
- Cancellations: Invoice starting with `C`. Keep them in a separate table (return behavior is a feature), exclude from revenue.
- Remove Quantity <= 0 and Price <= 0 from purchases; remove non-product codes (POST, D, M, BANK CHARGES, DOT, etc.); document each.
- Revenue = Quantity x Price. Combine both sheets, dedupe exact duplicates, convert to Parquet in `data/processed/`.

## 3. Repo structure (create exactly)
```
data/raw/  data/processed/
notebooks/
  01_eda.ipynb
  02_churn_definition_and_features.ipynb
  03_modeling.ipynb
  04_business_rule.ipynb
  05_clv_bgnbd.ipynb          # optional stretch
src/
  config.yaml                 # cutoffs, windows, business assumptions
  data.py  features.py  labels.py  split.py  train.py  evaluate.py  policy.py
reports/
  figures/  metrics.json  model_comparison.md  business_rule.md
app/streamlit_app.py          # optional
tests/                        # leakage and split tests
requirements.txt  README.md
```

## 4. Tasks

### 4.1 EDA
Transactions per customer, repeat rate, inter-purchase gaps distribution, seasonality (strong Q4), wholesale vs retail customers (heavy tails), country mix. The inter-purchase gap distribution **justifies the churn window**; show it.

### 4.2 Churn definition (behavioral, snapshot design)
- Choose monthly **snapshot dates** (cutoffs). For each snapshot `t`, the customer set = customers with at least one purchase before `t`.
- Label: `churn = 1` if the customer makes **no purchase in (t, t + H]**, with H = 90 days by default (from config; justify with the gap distribution, and run 60/120 as sensitivity).
- Features use **only data strictly before `t`**.

### 4.3 Features (RFM and beyond), computed per (customer, snapshot)
- Recency (days since last purchase), Frequency (orders in last 90/180/365 days and lifetime), Monetary (revenue in the same windows, average order value)
- Tenure (days since first purchase), average and std of inter-order gap, recency divided by average gap (overdue ratio), number of distinct products/categories, return/cancellation rate, share of Q4 purchases, trend features (orders last 90d vs previous 90d), country group, wholesale-like flag (large average quantity).
- Write `tests/` that assert no feature uses data at or after the snapshot.

### 4.4 Time-based split (no random split)
- Train on earlier snapshots, validate on a middle block, test on the **latest** snapshots.
- Leave a **gap of H days** between the last training snapshot and the first test snapshot so training labels do not peek into the test period.
- Note the last usable snapshot = last data date minus H.
- Group awareness: the same customer appears in multiple snapshots; never evaluate on data that leaks labels forward. Report how many test customers were also in train.

### 4.5 Modeling
- Baseline: logistic regression (scaled, class_weight tuned, L2) with a sklearn Pipeline.
- Challenger: gradient boosting (LightGBM or XGBoost; HistGradientBoosting as fallback). Tune with **time-series-aware CV** (expanding window), not shuffled CV.
- Metrics: PR-AUC (primary), ROC-AUC, Brier score, calibration curve, lift/gain by decile, precision/recall at top-k. Calibrate probabilities (isotonic or Platt) on the validation block, since the business rule needs true probabilities.
- Compare in `reports/model_comparison.md`, including a naive "recency only" heuristic baseline. Report performance drift across test snapshots.

### 4.6 Explainability
SHAP for the gradient boosting model: global summary, dependence plots for recency/frequency/overdue ratio, and 3 individual customer explanations. Translate the top drivers into plain business language.

### 4.7 Business rule (the core deliverable)
Expected value of treating customer *i*:
`EV_i = P(churn_i) x P(retained | offer) x expected_margin_i_over_horizon - offer_cost`
(Also include the incentive cost paid to customers who would have stayed anyway.)
- Put every assumption in `src/config.yaml` and **label them as assumptions** (offer acceptance/save rate, discount %, gross margin %, contact cost). Do not present them as findings.
- Expected margin per customer: from historical spend in the horizon (or BG/NBD-based CLV if 05 is done).
- Rule: treat customers where EV > 0, ranked by EV, subject to an optional budget cap. Compute total expected profit versus: treat nobody, treat everyone, treat by simple recency rule.
- **Sensitivity analysis**: heatmap of profit versus save rate x offer cost; find the break-even save rate. State plainly that true incremental impact needs an A/B test (uplift), and propose the test design.
- Write `reports/business_rule.md` (one page) with the rule in one sentence and the profit table.

### 4.8 Optional stretch
- BG/NBD + Gamma-Gamma (`lifetimes` or `pymc-marketing`) for CLV; compare P(alive) to the classifier's churn probability.
- `app/streamlit_app.py`: upload or select a customer, show churn probability, SHAP explanation, EV and treat/don't-treat decision, with sliders for the cost assumptions.

## 5. Acceptance criteria
- Pipeline reproducible: `python -m src.train` regenerates features, model, `reports/metrics.json` and figures from raw data.
- Leakage tests pass. Split diagram or table in README showing train/gap/test periods.
- README leads with: the churn definition, the primary metric result, and **the business rule in one sentence**, then how to run.
- Every reported number comes from executed code.

## 6. Working rules
- Ask before installing packages outside `requirements.txt`. Never commit raw data.
- Never invent results; if a step cannot run, leave a marked TODO.
- Plan first, then run phase by phase (4.1 -> 4.7), verifying each before continuing. Set random seeds.
- Keep code modular in `src/`; notebooks call `src/` functions instead of duplicating logic.
