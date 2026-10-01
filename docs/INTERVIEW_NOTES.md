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
