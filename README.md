# Customer Churn & Lifetime Value Prediction System

A production-grade analytics and machine learning pipeline that models behavioral customer churn and optimizes targeted retention campaigns using Expected Value (EV) decisions.

---

## The Problem

E-commerce businesses frequently lose revenue when existing repeat customers stop purchasing. Standard predictive models output probability scores, but raw probabilities alone do not tell a company whether offering a discount or incentive is financially worth its cost. Target targeting everyone wastes budget on customers who would have purchased anyway or will churn regardless, while under-targeting leaves predictable revenue on the table. This project solves that problem by transforming calibrated churn predictions into financially optimal, expected-value targeting rules.

---

## Key Results & Business Rule

- **Cleaned Data Scope**: Processed 1,067,371 raw transaction records down to 776,580 clean purchase events across 5,852 unique customers and £17,068,583.72 in gross sales revenue.
- **Empirical Churn Window**: 83.17% of all historical repeat purchase gaps occur within 90 days, establishing an empirical 90-day inactivity threshold ($H = 90$ days) for behavioral churn labeling.
- **Business Decision Rule**: *Target retention offers exclusively to customers where the expected revenue saved exceeds the campaign offer cost and incentive costs incurred on non-churners ($EV_i > 0$).*

---

## Churn Definition & Temporal Validation

### Behavioral Churn Definition
Unlike subscription models with explicit cancellation dates, non-subscription retail churn must be inferred from customer behavior:
1. **Snapshot Date ($t$)**: Monthly cutoff dates where customer history is evaluated.
2. **Feature Window ($\le t$)**: All predictive features (RFM metrics, order momentum, gap statistics, return rates) are computed using transactions occurring strictly on or before $t$.
3. **Observation Window ($(t, t + H]$)**: A customer is labeled `churn = 1` if they place zero orders in the $H = 90$ days following snapshot $t$, and `churn = 0` if they purchase at least once.

### Temporal Validation Split
To prevent temporal data leakage, models are evaluated using time-based snapshot splits with a strict $H$-day boundary gap:

| Split Block | Snapshot Range | Description |
| :--- | :--- | :--- |
| **Train Set** | 2010-04-01 to 2010-09-01 | Earlier historical snapshot cutoffs used for model fitting |
| **Validation Set** | 2010-10-01 to 2010-12-01 | Middle snapshot cutoffs used for probability calibration |
| **Gap Period** | 90 Days ($H$) | Buffer period ensuring training labels do not peek into test data |
| **Test Set** | 2011-04-01 to 2011-09-01 | Latest historical snapshot cutoffs used for final evaluation |

---

## Repository Structure

```
.
├── data/
│   ├── raw/                      # Raw online_retail_II.csv data (git-ignored)
│   └── processed/                # Cleaned Parquet files (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb              # Cleaning, EDA, gap distribution, seasonality
│   ├── 02_churn_definition.ipynb # Monthly snapshots, label & feature engineering
│   ├── 03_modeling.ipynb         # Time-based split, LightGBM/Logistic, calibration
│   └── 04_business_rule.ipynb    # EV optimization, SHAP explainability, sensitivity
├── src/
│   ├── __init__.py
│   ├── config.yaml               # Cutoffs, window definitions, business assumptions
│   ├── theme.py                  # Custom design tokens (#1F3A5F, #2A9D8F, #E9A23B, #D1495B)
│   ├── data.py                   # Data ingestion, cleaning, row count logging
│   ├── features.py               # Leakage-free RFM & temporal feature extraction
│   ├── labels.py                 # Snapshot creation & behavioral churn label logic
│   ├── split.py                  # Temporal train/val/test splitting with H-day gap
│   ├── train.py                  # End-to-end training pipeline runner
│   ├── evaluate.py               # PR-AUC, calibration, decile lift calculation
│   └── policy.py                 # EV calculation, thresholding & targeting rules
├── reports/
│   ├── figures/                  # Generated figures and charts
│   ├── metrics.json              # Canonical evaluation metrics
│   ├── model_comparison.md       # Model performance comparison
│   └── business_rule.md          # Business rule description & profit summary
├── app/
│   ├── streamlit_app.py          # Streamlit decision dashboard entry point
│   ├── components.py             # Reusable UI cards and risk tags
│   ├── style.css                 # Dashboard styling
│   └── views/                    # Dashboard page views (1-4)
├── tests/                        # Unit and leakage test suite
├── docs/                         # Methodology documentation and defense notes
├── requirements.txt              # Pinned Python dependencies
└── README.md
```

---

## Quickstart Guide

### 1. Prerequisites & Virtual Environment Setup
Ensure Python 3.11+ is installed. Create and activate a clean virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
Install pinned Python requirements:

```powershell
pip install -r requirements.txt
```

### 3. Place Raw Dataset
Ensure the raw dataset is placed at `data/raw/online_retail_II.csv`.

### 4. Run Data Pipeline & Execution
Run data cleaning, feature engineering, and model training:

```powershell
python -m src.data
python -m src.train
```

### 5. Run Test Suite
Verify pipeline integrity and data leakage checks:

```powershell
python -m pytest
python -m ruff check .
```

### 6. Launch Interactive Dashboard
Launch the Streamlit web dashboard:

```powershell
streamlit run app/streamlit_app.py
```

---

## Illustrative Business Assumptions & Unit Economics

The decision framework relies on explicit business parameters configured in `src/config.yaml`. These parameters are illustrative assumptions rather than measured dataset properties:

- **Offer Save Rate ($S$)**: 25% (Proportion of targeted churners persuaded to stay).
- **Offer Incentive Cost ($C_{\text{offer}}$)**: £15.00 per targeted customer.
- **Gross Profit Margin ($M$)**: 40% applied to customer transaction value.
- **Administrative Contact Cost ($C_{\text{contact}}$)**: £1.50 per target.
- **Non-Churner Redemption Rate**: 80% of non-churners redeem the incentive if targeted.

---

## Methodological Limitations

1. **Single Retailer Focus**: Data originates from one UK-based non-subscription retailer (2009–2011), requiring domain tuning before application to SaaS or subscription models.
2. **Assumed Cost Parameters**: Unit financial costs and retention save rates are illustrative assumptions.
3. **Observational Baseline**: Standard classification models estimate correlation rather than direct causal impact. Validating real incremental revenue requires randomized control trial (A/B test) deployment.

---

## License & Data Source

- **Data Source**: UCI Machine Learning Repository — Online Retail II Dataset.
- **License**: Open Data Commons / Creative Commons Attribution 4.0 International (CC BY 4.0).
