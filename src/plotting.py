"""Shared chart style and saving (Member B)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import FIG_DIR  # noqa: E402

# Colours carry the same meaning in every chart.
BLUE = "#2f6f9f"     # training / main series
ORANGE = "#d9822b"   # test
RED = "#b23a48"      # noise / risk
GREY = "#8a94a6"     # comparison (benchmark)
LIGHT = "#c9ced6"

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")
    return path
