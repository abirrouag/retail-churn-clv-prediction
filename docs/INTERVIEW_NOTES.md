# Portfolio Defense Notes: Customer Churn & Lifetime Value

> **Confidential Interview Cheat-Sheet**: Maintained at each project checkpoint. Designed to prepare the author for live technical interviews, hiring manager deep-dives, and code defenses.

---

## Glossary of Core Terms
- **RFM (Recency, Frequency, Monetary)**: Classical behavioral features measuring days since last purchase (Recency), total order counts (Frequency), and aggregate monetary spend (Monetary).
- **PR-AUC (Precision-Recall Area Under Curve)**: Primary classification metric evaluating precision across all recall thresholds. Superior to ROC-AUC when class distributions are imbalanced (churners are minority).
- **Probability Calibration**: Mapping raw classifier probability scores (which are often uncalibrated or over-confident) to true empirical probabilities using Isotonic or Sigmoid/Platt scaling.
- **Lift / Gain**: Comparative metric showing how many times better a model-driven targeting strategy performs relative to random targeting or a simple baseline.
- **SHAP (SHapley Additive exPlanations)**: Game-theoretic approach to feature attribution, breaking down individual predictions into additive feature contributions.
- **Data Leakage**: The invalid inclusion of future information (data at or after snapshot date $t$) into feature engineering or training labels.
- **Expanding-Window Time-Series CV**: Cross-validation split strategy for temporal data where training sets sequentially expand over historical time windows without shuffling.
- **Expected Value (EV) Decision Policy**: Business logic mapping calibrated churn probabilities and monetary unit economics to a net profit/loss figure per customer ($EV_i$).

---

## Phase 0: Setup & Technical Foundation

### 60-Second Explanation
In Phase 0, we established a reproducible project framework matching enterprise software standards. We initialized Git, established a clean folder structure separating source modules (`src/`), notebooks (`notebooks/`), reports (`reports/`), and Streamlit app views (`app/`), and configured strict `.gitignore` rules to keep raw data and temporary files out of version control. We defined centralized design system tokens in `src/theme.py` and `.streamlit/config.toml` to guarantee visual consistency across figures and dashboards. Finally, we placed and validated the 1.067M-row UCI Online Retail II raw dataset.

### Key Decisions & Alternatives Rejected
- **Decision**: Centralize configuration in `src/config.yaml` rather than using hardcoded magic numbers in python modules.
  - *Alternative Rejected*: Defining cutoffs and parameters directly inside scripts. Config-driven design ensures single-point modification and reproducibility.
- **Decision**: Standardize chart palette on Navy (`#1F3A5F`), Teal (`#2A9D8F`), Amber (`#E9A23B`), and Coral (`#D1495B`) with strict semantic usage (Coral reserved strictly for churn/risk).
  - *Alternative Rejected*: Using default Seaborn/Matplotlib color cycles or unconstrained color palettes.
- **Decision**: Ingest CSV data at `data/raw/online_retail_II.csv`.
  - *Alternative Rejected*: Downloading unofficial external files or downloading substitute datasets.

### 5 Likely Interview Questions & Honest Answers

1. **Q: Why did you place business assumptions (offer cost, save rate) in `config.yaml` rather than deriving them from the dataset?**
   *Answer*: The dataset contains transaction history but lacks internal financial margins, offer redemption rates, or marketing contact costs. Treating assumed business parameters as measured findings would be dishonest. Isolating them in `config.yaml` allows stakeholders to run scenario analysis without altering core model features.

2. **Q: Why use Parquet instead of CSV for intermediate data storage in `data/processed/`?**
   *Answer*: Parquet is a columnar format offering faster read/write I/O, compression, and built-in type retention (e.g., datetime parsing preserved), eliminating parsing overhead across scripts and notebooks.

3. **Q: Why is random K-Fold cross-validation inappropriate for this churn project?**
   *Answer*: Customer transactions have temporal structure and multi-row customer repetition. Shuffling data across time creates severe temporal data leakage—predicting past behavior using future information. Expanding-window CV respects time order.

4. **Q: How do you enforce consistency between Matplotlib notebook charts and the Streamlit dashboard?**
   *Answer*: Both consume design tokens exported from `src/theme.py`. Matplotlib rcParams and Plotly layout templates are configured programmatically upon module import, matching `.streamlit/config.toml`.

5. **Q: What is the benefit of keeping `src/` modules decoupled from `notebooks/`?**
   *Answer*: Notebooks serve as narrative, analytical reports while heavy engineering logic (data cleaning, feature calculation, model training) lives in clean, testable Python functions in `src/`. This avoids code duplication and allows automated execution via `python -m src.train`.

### Key Numbers to Remember
- **Total Raw Transaction Rows**: 1,067,371 rows.
- **Raw File Size**: 94.85 MB.
- **Raw Columns**: 8 (`Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`).

### Open Weaknesses & Limitations to Admit Openly
- **Single Retailer**: Data comes from a single UK-based online retailer (2009–2011), limiting generalizability to subscription or SaaS businesses without domain adaptation.
- **Assumed Unit Economics**: Margin percentages and offer save rates are illustrative assumptions.

---

## Phase 1: Data Cleaning & Exploratory Data Analysis

### 60-Second Explanation
In Phase 1, we constructed `src/data.py` to systematically ingest, clean, and audit 1,067,371 raw transaction records into a production-grade Parquet dataset of 776,580 purchase records and 18,744 return records. We logged row counts at every step: dropping missing Customer IDs, separating cancellations (`C` invoices), filtering non-product fees (`POST`, `DOT`, `M`, `BANK CHARGES`), and removing exact duplicate rows. In `notebooks/01_eda.ipynb`, we evaluated transaction frequency, seasonality, and the inter-purchase gap distribution. We proved that 83.17% of all repeat orders occur within 90 days, providing strong empirical justification for our 90-day behavioral churn window ($H = 90$).

### Key Decisions & Alternatives Rejected
- **Decision**: Separate cancellations (`Invoice` starting with `C`) into an independent returns table (`online_retail_returns.parquet`) rather than discarding them or subtracting quantities directly in-place.
  - *Alternative Rejected*: Deleting return rows entirely. Keeping them in a dedicated table allows us to build return rate and return monetary share features per customer in Phase 2.
- **Decision**: Filter out non-product stock codes (`POST`, `DOT`, `M`, `BANK CHARGES`, `ADJUST`, `C2`, etc.).
  - *Alternative Rejected*: Including postage and fee charges in customer monetary spend. Non-product fees distort pure product purchasing behavior and average order values.
- **Decision**: Justify the 90-day churn window empirically using the empirical cumulative distribution function (ECDF) of inter-purchase gaps.
  - *Alternative Rejected*: Picking an arbitrary cutoff without data-backed evidence.

### 5 Likely Interview Questions & Honest Answers

1. **Q: How did you handle cancellations in the dataset?**
   *Answer*: Invoices starting with `C` represent returns. We extracted them into `data/processed/online_retail_returns.parquet` to prevent negative quantities from corrupting sales revenue, while preserving them to engineer customer return frequency and return monetary ratios.

2. **Q: Why did you filter out non-product stock codes like `POST` or `BANK CHARGES`?**
   *Answer*: Non-product entries represent shipping fees, manual adjustments, or bank charges rather than merchandise purchases. Including them would artificially inflate order frequency and distort product preferences.

3. **Q: How did you empirically justify setting the churn cutoff to $H = 90$ days?**
   *Answer*: We analyzed 30,743 sequential inter-purchase gaps among repeat buyers. The median gap is 25 days, the 75th percentile is 62 days, and 83.17% of all repeat orders happen within 90 days. A customer inactive for 90 days has an 83%+ empirical probability of having lapsed.

4. **Q: What major seasonal patterns did you discover, and why do they matter for modeling?**
   *Answer*: Gross monthly revenue surges by over 100% in Q4 (October–November exceeding £1.1M/month vs ~£550k in Q1–Q3). If a model ignores seasonality, it will misinterpret post-holiday order drops in Q1 as permanent customer churn.

5. **Q: What is the geographic mix, and how does it influence feature engineering?**
   *Answer*: The UK dominates sales, generating 83.71% of total revenue (£14.29M out of £17.07M). We must include explicit domestic vs international country flags as model features.

### Key Numbers to Remember
- **Raw Rows**: 1,067,371
- **Dropped Missing Customer ID**: 243,007 rows (22.77% of raw dataset)
- **Extracted Cancellations/Returns**: 18,744 rows
- **Filtered Quantity/Price $\le 0$**: 71 rows
- **Filtered Non-Product Codes**: 2,914 rows
- **Removed Duplicate Rows**: 26,055 rows
- **Final Cleaned Purchases**: 776,580 rows $\times$ 9 columns
- **Total Unique Customers**: 5,852
- **Total Gross Revenue**: £17,068,583.72
- **Repeat Buyer Rate**: 72.35% (4,234 of 5,852 customers)
- **Total Inter-Purchase Gaps Analyzed**: 30,743 gaps
- **Median Inter-Purchase Gap**: 25.0 days (75th percentile: 62.0 days)
- **Proportion of Gaps $\le 90$ Days**: 83.17%
- **UK Revenue Share**: 83.71% (£14.29M)

### Open Weaknesses & Limitations to Admit Openly
- **High Missing Customer ID Rate**: 22.77% of raw transactions (243k rows) lacked Customer IDs (guest checkout) and had to be dropped because individual behavior could not be tracked.
- **Short 24-Month Data Period**: 2 years allows seeing two Q4 seasonal cycles, but longer historical data would strengthen multi-year trend modeling.
