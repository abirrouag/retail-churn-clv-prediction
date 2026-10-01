"""Data ingestion, cleaning, and preprocessing pipeline module."""

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

NON_PRODUCT_CODES = [
    "POST",
    "DOT",
    "M",
    "C2",
    "D",
    "S",
    "BANK CHARGES",
    "ADJUST",
    "ADJUST2",
    "AMAZONFEE",
    "CRUK",
    "TEST001",
    "TEST",
    "PADS",
]


def load_config(config_path: str = "src/config.yaml") -> dict:
    """Load system configuration yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(filepath: str = "data/raw/online_retail_II.csv") -> pd.DataFrame:
    """Load raw transaction dataset from CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at {path}")
    logger.info(f"Loading raw data from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Raw data loaded: {len(df):,} rows x {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw transactions and return (cleaned_purchases, returns_df).

    Logging row counts at every cleaning step.
    """
    initial_rows = len(df)
    logger.info(f"[Step 0] Initial raw row count: {initial_rows:,}")

    # 1. Drop missing Customer ID
    df_valid_cust = df.dropna(subset=["Customer ID"]).copy()
    dropped_cust = initial_rows - len(df_valid_cust)
    logger.info(
        f"[Step 1] Dropped missing Customer ID: {dropped_cust:,} rows removed -> {len(df_valid_cust):,} remaining"
    )

    # Clean data types
    df_valid_cust["Customer ID"] = df_valid_cust["Customer ID"].astype(int).astype(str)
    df_valid_cust["InvoiceDate"] = pd.to_datetime(df_valid_cust["InvoiceDate"])
    df_valid_cust["Invoice"] = df_valid_cust["Invoice"].astype(str).str.strip()
    df_valid_cust["StockCode"] = (
        df_valid_cust["StockCode"].astype(str).str.strip().str.upper()
    )

    # 2. Separate cancellations (Invoices starting with 'C')
    is_cancel = df_valid_cust["Invoice"].str.startswith("C")
    returns_df = df_valid_cust[is_cancel].copy()
    purchases_df = df_valid_cust[~is_cancel].copy()
    logger.info(
        f"[Step 2] Separated cancellations: {len(returns_df):,} return records extracted -> {len(purchases_df):,} purchase candidates remaining"
    )

    # 3. Filter Quantity > 0 and Price > 0
    valid_price_qty = purchases_df[
        (purchases_df["Quantity"] > 0) & (purchases_df["Price"] > 0)
    ].copy()
    dropped_qty_price = len(purchases_df) - len(valid_price_qty)
    logger.info(
        f"[Step 3] Filtered Quantity <= 0 or Price <= 0: {dropped_qty_price:,} rows removed -> {len(valid_price_qty):,} remaining"
    )

    # 4. Remove non-product stock codes
    valid_products = valid_price_qty[
        ~valid_price_qty["StockCode"].isin(NON_PRODUCT_CODES)
    ].copy()
    dropped_non_product = len(valid_price_qty) - len(valid_products)
    logger.info(
        f"[Step 4] Filtered non-product stock codes ({', '.join(NON_PRODUCT_CODES[:5])}...): {dropped_non_product:,} rows removed -> {len(valid_products):,} remaining"
    )

    # 5. Compute Revenue = Quantity * Price
    valid_products["Revenue"] = valid_products["Quantity"] * valid_products["Price"]

    # 6. Deduplicate exact duplicate rows
    deduped = valid_products.drop_duplicates().copy()
    dropped_dups = len(valid_products) - len(deduped)
    logger.info(
        f"[Step 6] Removed exact duplicate rows: {dropped_dups:,} rows removed -> {len(deduped):,} final cleaned rows"
    )

    return deduped, returns_df


def save_processed_data(
    purchases_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    purchases_output: str = "data/processed/online_retail_cleaned.parquet",
    returns_output: str = "data/processed/online_retail_returns.parquet",
) -> None:
    """Save cleaned dataframes to Parquet format."""
    p_path = Path(purchases_output)
    r_path = Path(returns_output)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving cleaned purchases to {p_path}...")
    purchases_df.to_parquet(p_path, index=False)
    logger.info(f"Saving returns table to {r_path}...")
    returns_df.to_parquet(r_path, index=False)
    logger.info("Processed data saved successfully.")


def run_pipeline(
    config_path: str = "src/config.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end data cleaning pipeline."""
    cfg = load_config(config_path)
    raw_df = load_raw_data(cfg["data"]["raw_path"])
    purchases_df, returns_df = clean_data(raw_df)
    save_processed_data(
        purchases_df,
        returns_df,
        purchases_output=cfg["data"]["processed_path"],
        returns_output=cfg["data"]["returns_path"],
    )
    return purchases_df, returns_df


if __name__ == "__main__":
    run_pipeline()
