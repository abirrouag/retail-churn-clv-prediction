"""Basic setup and environment verification tests."""

import os

import yaml

from src.theme import COLOR_CORAL, COLOR_NAVY


def test_raw_data_exists():
    """Verify raw dataset exists in data/raw/."""
    path = os.path.join("data", "raw", "online_retail_II.csv")
    assert os.path.exists(path), f"Raw data file {path} not found"


def test_config_loadable():
    """Verify config.yaml is valid YAML and contains required sections."""
    path = os.path.join("src", "config.yaml")
    assert os.path.exists(path), f"Config file {path} not found"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    assert "churn_definition" in cfg
    assert "business_assumptions" in cfg
    assert cfg["churn_definition"]["horizon_days"] == 90


def test_theme_colors():
    """Verify theme color tokens are valid hex codes."""
    assert COLOR_NAVY.startswith("#") and len(COLOR_NAVY) == 7
    assert COLOR_CORAL.startswith("#") and len(COLOR_CORAL) == 7
