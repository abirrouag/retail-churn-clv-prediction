"""Unit tests for data cleaning and preprocessing pipeline."""

import pandas as pd

from src.data import clean_data


def test_clean_data_pipeline():
    """Test data cleaning steps on a hand-crafted dummy dataframe."""
    raw_dummy = pd.DataFrame(
        [
            # Valid transaction 1
            {
                "Invoice": "500001",
                "StockCode": "85048",
                "Description": "ITEM A",
                "Quantity": 2,
                "InvoiceDate": "2010-01-01 10:00:00",
                "Price": 5.0,
                "Customer ID": 12345.0,
                "Country": "United Kingdom",
            },
            # Duplicate of transaction 1
            {
                "Invoice": "500001",
                "StockCode": "85048",
                "Description": "ITEM A",
                "Quantity": 2,
                "InvoiceDate": "2010-01-01 10:00:00",
                "Price": 5.0,
                "Customer ID": 12345.0,
                "Country": "United Kingdom",
            },
            # Missing Customer ID (should be dropped)
            {
                "Invoice": "500002",
                "StockCode": "85048",
                "Description": "ITEM A",
                "Quantity": 1,
                "InvoiceDate": "2010-01-01 11:00:00",
                "Price": 5.0,
                "Customer ID": None,
                "Country": "United Kingdom",
            },
            # Cancellation (should go to returns_df)
            {
                "Invoice": "C500003",
                "StockCode": "85048",
                "Description": "ITEM A",
                "Quantity": -2,
                "InvoiceDate": "2010-01-02 10:00:00",
                "Price": 5.0,
                "Customer ID": 12345.0,
                "Country": "United Kingdom",
            },
            # Invalid Quantity <= 0 purchase
            {
                "Invoice": "500004",
                "StockCode": "85048",
                "Description": "ITEM A",
                "Quantity": 0,
                "InvoiceDate": "2010-01-03 10:00:00",
                "Price": 5.0,
                "Customer ID": 12345.0,
                "Country": "United Kingdom",
            },
            # Non-product code (POST)
            {
                "Invoice": "500005",
                "StockCode": "POST",
                "Description": "POSTAGE",
                "Quantity": 1,
                "InvoiceDate": "2010-01-04 10:00:00",
                "Price": 15.0,
                "Customer ID": 12345.0,
                "Country": "United Kingdom",
            },
            # Valid transaction 2
            {
                "Invoice": "500006",
                "StockCode": "22423",
                "Description": "ITEM B",
                "Quantity": 4,
                "InvoiceDate": "2010-01-05 10:00:00",
                "Price": 10.0,
                "Customer ID": 67890.0,
                "Country": "France",
            },
        ]
    )

    purchases_df, returns_df = clean_data(raw_dummy)

    # 1. Missing Customer ID dropped (row 500002 removed)
    assert "500002" not in purchases_df["Invoice"].values

    # 2. Cancellation separated (C500003 in returns_df, not purchases_df)
    assert len(returns_df) == 1
    assert returns_df.iloc[0]["Invoice"] == "C500003"
    assert "C500003" not in purchases_df["Invoice"].values

    # 3. Non-product code filtered out (500005 with POST removed)
    assert "POST" not in purchases_df["StockCode"].values

    # 4. Revenue calculated correctly (2 * 5.0 = 10.0 and 4 * 10.0 = 40.0)
    assert "Revenue" in purchases_df.columns
    assert set(purchases_df["Revenue"].values) == {10.0, 40.0}

    # 5. Exact duplicates removed (only 1 row for 500001)
    assert len(purchases_df) == 2
    assert set(purchases_df["Customer ID"].values) == {"12345", "67890"}
