"""EDS-2026-0083 revision: the complete two-panel Figure 5 at 300 dpi.

Panel (a) is the submitted divergence map, drawn by make_fig5.panel_a unchanged
(purity filter, Sava-upper-Danube corridor, asymmetric diverging norm, L20).
Panel (b) replaces the contaminated-only dose-response with the contaminated and
matched-null band divergences of the worked case (worked_case_null_bands.csv),
95 % intervals over 30 replicates, fonts matched to panel (a).
Requires figures/overpred_pleniusculus_L20.csv (submitted analysis output).
Output: figures/rev_eds/Fig5.png and Fig5.pdf (300 dpi; PDF without a creation date).
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
LEVELS = (3, 10, 20)
COLS = ("#9ecae1", "#3182bd", "#08306b")
NAMES = ["0\u20130.1", "0.1\u20130.3", "0.3\u20130.5", "0.5\u20130.7", "0.7\u20131.0"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def panel_b(ax):
    w = pd.read_csv(REPO / "reports/rev_eds/worked_case_null_bands.csv")
    for lvl, col, off in zip(LEVELS, COLS, (-0.22, 0.0, 0.22)):
        g = w[w["level"] == lvl].sort_values("band_index")
        x = g["band_index"].to_numpy() + off
        ax.errorbar(x - 0.05, g["sub_mean"], yerr=[g["sub_mean"] - g["sub_lo"], g["sub_hi"] - g["sub_mean"]],
                    fmt="o", color=col, ms=4.5, capsize=2, lw=1.1, label=f"L{lvl} contaminated")
        ax.errorbar(x + 0.05, g["null_mean"], yerr=[g["null_mean"] - g["null_lo"], g["null_hi"] - g["null_mean"]],
                    fmt="s", mfc="white", color=col, ms=4.5, capsize=2, lw=1.1, label=f"L{lvl} null")
    t = w[w["level"] == LEVELS[0]].sort_values("band_index")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(t["band_index"])
    ax.set_xticklabels([f"{n}\n(n = {int(k):,})" for n, k in zip(NAMES, t["n_subc"])], fontsize=8.5)
    ax.set_xlabel("benchmark-suitability band")
    ax.set_ylabel("divergence from the benchmark")
    ax.set_title("(b)", fontsize=12, loc="left", fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")


def main():
    src = REPO / "figures/overpred_pleniusculus_L20.csv"
    if not src.exists():
        sys.exit(f"missing {src.relative_to(REPO)}: regenerate it with run_overpred_ci.py before this figure")
    mf = load_module("make_fig5", REPO / "make_fig5.py")
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.8, 5.8), constrained_layout=True)
    mf.panel_a(ax_a, 20)
    panel_b(ax_b)
    out = REPO / "figures/rev_eds"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "Fig5.png", dpi=300)
    fig.savefig(out / "Fig5.pdf", dpi=300, metadata={"CreationDate": None})
    print("wrote figures/rev_eds/Fig5.png and Fig5.pdf")


if __name__ == "__main__":
    main()
