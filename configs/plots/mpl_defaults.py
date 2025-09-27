# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: jupytext,text_representation,kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
# ---

# %%
"""
Matplotlib defaults for crypto-vol-quickstart.

Usage (top of any notebook):
    from configs.plots.mpl_defaults import use_mpl_defaults, format_date_axis
    use_mpl_defaults()

Then (optional):
    ax = plt.gca()
    format_date_axis(ax)
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# %%
def use_mpl_defaults(dpi: int = 150, base_font: float = 11.0) -> None:
    """Set sane plotting defaults (no custom colors). Call once per notebook."""
    rc = mpl.rcParams
    rc["figure.dpi"] = dpi
    rc["savefig.dpi"] = dpi
    rc["savefig.bbox"] = "tight"

    rc["figure.figsize"] = (11, 4.5)

    rc["font.size"] = base_font
    rc["axes.titlesize"] = base_font + 1
    rc["axes.labelsize"] = base_font
    rc["xtick.labelsize"] = base_font - 2
    rc["ytick.labelsize"] = base_font - 2
    rc["legend.fontsize"] = base_font - 2
    rc["legend.frameon"] = False

    rc["axes.grid"] = True
    rc["grid.linestyle"] = "--"
    rc["grid.alpha"] = 0.3

    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False

    rc["lines.linewidth"] = 1.25

# %%
def format_date_axis(ax: mpl.axes.Axes | None = None) -> mpl.axes.Axes:
    """Apply concise date formatting to the x-axis of a time series plot."""
    ax = ax or plt.gca()
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")
    return ax
