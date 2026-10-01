"""Design system tokens, color palettes, and plotting utilities for Matplotlib and Plotly."""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

# Color Tokens
COLOR_NAVY = "#1F3A5F"
COLOR_TEAL = "#2A9D8F"
COLOR_AMBER = "#E9A23B"
COLOR_CORAL = "#D1495B"
COLOR_INK = "#1B2430"
COLOR_SLATE = "#5B6675"
COLOR_MIST = "#F4F6F9"
COLOR_CARD = "#FFFFFF"
COLOR_LINE = "#E1E6ED"

PALETTE = {
    "navy": COLOR_NAVY,
    "teal": COLOR_TEAL,
    "amber": COLOR_AMBER,
    "coral": COLOR_CORAL,
    "ink": COLOR_INK,
    "slate": COLOR_SLATE,
    "mist": COLOR_MIST,
    "card": COLOR_CARD,
    "line": COLOR_LINE,
}

# Risk Tier Mapping
RISK_TIERS = {
    "Low": COLOR_TEAL,
    "Medium": COLOR_AMBER,
    "High": COLOR_CORAL,
}

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def set_matplotlib_theme():
    """Apply global design system settings to Matplotlib."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 11,
            "figure.facecolor": COLOR_CARD,
            "axes.facecolor": COLOR_CARD,
            "axes.edgecolor": COLOR_LINE,
            "axes.linewidth": 1.0,
            "axes.labelcolor": COLOR_INK,
            "axes.labelsize": 11,
            "axes.labelweight": "medium",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": COLOR_NAVY,
            "xtick.color": COLOR_SLATE,
            "ytick.color": COLOR_SLATE,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": COLOR_LINE,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "legend.facecolor": COLOR_CARD,
            "legend.edgecolor": COLOR_LINE,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "savefig.facecolor": COLOR_CARD,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )


def create_plotly_template():
    """Create a Plotly layout template matching the design system."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": FONT_FAMILY, "color": COLOR_INK, "size": 12},
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_CARD,
        title={"font": {"color": COLOR_NAVY, "size": 16, "family": FONT_FAMILY}},
        xaxis={
            "gridcolor": COLOR_LINE,
            "linecolor": COLOR_LINE,
            "tickfont": {"color": COLOR_SLATE},
            "titlefont": {"color": COLOR_INK},
            "zerolinecolor": COLOR_LINE,
        },
        yaxis={
            "gridcolor": COLOR_LINE,
            "linecolor": COLOR_LINE,
            "tickfont": {"color": COLOR_SLATE},
            "titlefont": {"color": COLOR_INK},
            "zerolinecolor": COLOR_LINE,
        },
        colorway=[COLOR_NAVY, COLOR_TEAL, COLOR_AMBER, COLOR_CORAL, COLOR_SLATE],
    )
    pio.templates["custom_theme"] = template
    pio.templates.default = "custom_theme"


def save_figure(fig, filepath, caption=None):
    """Save a Matplotlib figure consistently with a footer caption."""
    if caption:
        fig.text(
            0.01, 0.01, caption, color=COLOR_SLATE, fontsize=8, ha="left", va="bottom"
        )
    fig.savefig(filepath, dpi=160, bbox_inches="tight")
