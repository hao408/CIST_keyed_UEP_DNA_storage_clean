"""Nature-style plotting defaults for manuscript figures.

This module centralizes visual choices only. It does not compute or alter any
experimental result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


MM_TO_INCH = 1.0 / 25.4
SINGLE_COL = 85 * MM_TO_INCH
DOUBLE_COL = 178 * MM_TO_INCH


METHOD_COLORS = {
    "No RS": "#D5D8DC",
    "Uniform RS-4": "#7A7F87",
    "Uniform RS-4/5": "#AEB4BC",
    "Static UEP": "#3F6FA8",
    "Random UEP": "#A36B91",
    "Keyed UEP": "#D55E00",
    "Uniform RS-8": "#2E8B57",
}

ATTACKER_COLORS = {
    "Rule-based": "#5B7FA6",
    "Logistic regression": "#D98C3A",
    "Random forest": "#4F9A6A",
}

CONTAINER_COLORS = {
    "Variable length": "#8C8C8C",
    "Deterministic padding": "#C9A04F",
    "Tail-only dummy": "#9A6E9E",
    "Full-container masked": "#2E8B73",
}


def apply_nature_style() -> None:
    """Apply clean, publication-oriented matplotlib defaults."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "grid.linewidth": 0.45,
            "grid.color": "#E6E8EB",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def save_figure(fig, stem: str | Path, dpi: int = 600) -> None:
    """Save a matplotlib figure as editable PDF and high-resolution PNG."""

    stem = Path(stem)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=dpi)


def clean_axis(ax, grid: bool = True, axis: str = "y") -> None:
    """Apply shared axis cleanup."""

    if grid:
        ax.grid(axis=axis, color="#E6E8EB", linewidth=0.45)
        ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

