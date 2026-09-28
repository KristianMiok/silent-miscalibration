"""EDS-2026-0083 revision: figures on calibrated members and nulls (all from committed CSVs).

fig1_coverage          Protocol B coverage (fixed-map Platt) by level and track; units as
                       lines (Okabe-Ito colours, one marker per unit), nominal 0.95 dashed,
                       range of null-refit coverage over units shaded as the empirical
                       reference
fig2_excess_divergence excess mean signed divergence over the null by benchmark quintile
                       (calibrated), per track; faint unit traces, bold level means
fig3_conformal         coverage before (circles) and after (squares) LOBO split conformal
                       calibration (calibrated), per track; unit colours as in fig1
fig4_width_inflation   width inflation factor by level and track (calibrated); bars are
                       medians over all eight units, uncorrected conditions (1.00) included
fig5b_worked_case_null worked case: contaminated and null divergence by band with 95%
                       replicate intervals and N per band (panel b; panel a unchanged)
fig6_odonata_null      Odonata replicate: contaminated and null divergence by band, on all
                       records and on records never used in a fit (if the CSV exists)
figS_cv_vs_coverage    cross-validated AUC of the contaminated model against coverage,
                       Protocol A and Protocol B (calibrated); gate 0.70, nominal 0.95,
                       miscoverage threshold 0.90, silent region shaded
Outputs: figures/rev_eds/<name>.pdf and .png (300 dpi).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
R = REPO / "reports/rev_eds"
FIG = REPO / "figures/rev_eds"
TRACKS = [("local_only", "local-only"), ("upstream_only", "upstream-only"), ("combined", "combined")]
LEVELS = [3, 10, 20]
LEVEL_COLS = ["#9ecae1", "#3182bd", "#08306b"]
TRACK_COLS = {"local_only": "#1b9e77", "upstream_only": "#d95f02", "combined": "#7570b3"}
UNITS = {
    "Astacus_astacus": r"$\it{A.\,astacus}$",
    "Austropotamobius_fulcisianus_pooled": r"$\it{A.\,fulcisianus}$",
    "Austropotamobius_torrentium_pooled": r"$\it{A.\,torrentium}$",
    "Faxonius_limosus_alien": r"$\it{F.\,limosus}$",
    "Pacifastacus_leniusculus_alien": r"$\it{P.\,leniusculus}$",
    "Pontastacus_leptodactylus_pooled": r"$\it{P.\,leptodactylus}$",
    "Procambarus_clarkii_alien": r"$\it{P.\,clarkii}$ (non-native)",
    "Procambarus_clarkii_native": r"$\it{P.\,clarkii}$ (native range)",
}
UNIT_COLS = dict(zip(UNITS, ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#000000", "#56B4E9", "#F0E442"]))
UNIT_MARK = dict(zip(UNITS, ["o", "s", "^", "D", "v", "P", "X", "*"]))
BAND_NAMES = ["0\u20130.1", "0.1\u20130.3", "0.3\u20130.5", "0.5\u20130.7", "0.7\u20131.0"]
NOMINAL, THRESH, GATE = 0.95, 0.90, 0.70
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})  # byte-stable
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/rev_eds/{name}.pdf and .png")


def panel_platt():
    pan = pd.read_csv(R / "protocol_b_platt_panel.csv")
    return pan[pan["method"] == "platt"]


def level_axis(ax, ticks):
    ax.set_xticks(ticks)
    ax.set_xticklabels(["L3", "L10", "L20"])
    ax.set_xlabel("contamination level")


def band_plot(ax, d, legend_loc):
    """d columns: level, band_index, n, c, c_lo, c_hi (contaminated), z, z_lo, z_hi (null)."""
    for lvl, col, off in zip(LEVELS, LEVEL_COLS, (-0.22, 0.0, 0.22)):
        g = d[d["level"] == lvl].sort_values("band_index")
        x = g["band_index"].to_numpy() + off
        ax.errorbar(x - 0.05, g["c"], yerr=[g["c"] - g["c_lo"], g["c_hi"] - g["c"]], fmt="o", color=col,
                    ms=3.5, capsize=1.5, lw=0.9, label=f"L{lvl} contaminated")
        ax.errorbar(x + 0.05, g["z"], yerr=[g["z"] - g["z_lo"], g["z_hi"] - g["z"]], fmt="s", mfc="white",
                    color=col, ms=3.5, capsize=1.5, lw=0.9, label=f"L{lvl} null")
    t = d[d["level"] == LEVELS[0]].sort_values("band_index")
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xlim(-0.5, 4.5)
    ax.set_xticks(t["band_index"])
    ax.set_xticklabels([f"{BAND_NAMES[int(i)]}\n(n = {int(k):,})" for i, k in zip(t["band_index"], t["n"])],
                       fontsize=6)
    ax.set_xlabel("benchmark-suitability band")
    if legend_loc:
        ax.legend(frameon=False, ncol=2, fontsize=6, loc=legend_loc)


def fig1():
    pan = panel_platt()
    nul = pd.read_csv(R / "null_coverage.csv", keep_default_na=False, na_values=[""])  # "null" is a pandas NA token
    nul = nul[(nul["method"] == "platt") & (nul["source"] == "null")]
    assert len(nul) == 72, f"expected 72 null conditions, found {len(nul)}"
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.6), sharey=True)
    for ax, (t, tl), letter in zip(axes, TRACKS, "abc"):
        n = nul[nul["track"] == t].groupby("level")["coverage"].agg(["min", "max"])
        ax.fill_between(n.index, n["min"], n["max"], color="0.55", alpha=0.4, lw=0, zorder=0,
                        label="null refits (range over units)")
        for ed, lab in UNITS.items():
            d = pan[(pan["entity_dir"] == ed) & (pan["track"] == t)].sort_values("level")
            ax.plot(d["level"], d["coverage"], "-", marker=UNIT_MARK[ed], color=UNIT_COLS[ed], mec="0.25",
                    mew=0.4, ms=3.5, lw=1, label=lab)
        ax.axhline(NOMINAL, ls="--", color="0.2", lw=0.8, label="nominal 0.95")
        level_axis(ax, LEVELS)
        ax.set_ylim(0.40, 1.01)
        ax.set_title(f"({letter}) {tl}", loc="left")
        print(f"fig1 {tl}: null-refit coverage range " +
              ", ".join(f"L{lvl} {r['min']:.3f}-{r['max']:.3f}" for lvl, r in n.iterrows()))
    axes[0].set_ylabel("coverage of the clean benchmark")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig1_coverage")


def fig2():
    b = pd.read_csv(R / "protocol_b_platt_bins.csv")
    b = b[b["method"] == "platt"]
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.5), sharey=True)
    for ax, (t, tl), letter in zip(axes, TRACKS, "abc"):
        for lvl, col in zip(LEVELS, LEVEL_COLS):
            d = b[(b["track"] == t) & (b["level"] == lvl)]
            for _, g in d.groupby("entity_dir"):
                g = g.sort_values("bin")
                ax.plot(g["bin"] + 1, g["excess"], color=col, alpha=0.3, lw=0.7)
            m = d.groupby("bin")["excess"].mean()
            ax.plot(m.index + 1, m.values, "-o", color=col, lw=1.8, ms=3.5, label=f"L{lvl}")
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_title(f"({letter}) {tl}", loc="left")
    axes[1].set_xlabel("benchmark-suitability quintile (1 = lowest)")
    axes[0].set_ylabel("excess divergence over the null")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig2_excess_divergence")


def fig3():
    pan = panel_platt()
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.5), sharey=True)
    for ax, (t, tl), letter in zip(axes, TRACKS, "abc"):
        d = pan[pan["track"] == t]
        for i, lvl in enumerate(LEVELS):
            dl = d[d["level"] == lvl]
            for _, r in dl.iterrows():
                c = UNIT_COLS[r["entity_dir"]]
                ax.plot([i - 0.14, i + 0.14], [r["coverage_uncorrected"], r["coverage_conformal"]], color="0.8", lw=0.5)
                ax.plot(i - 0.14, r["coverage_uncorrected"], "o", color=c, mec="0.25", mew=0.3, ms=3)
                ax.plot(i + 0.14, r["coverage_conformal"], "s", color=c, mec="0.25", mew=0.3, ms=3)
            ax.plot(i - 0.14, dl["coverage_uncorrected"].mean(), "o", mfc="white", mec="k", ms=6, mew=1.2)
            ax.plot(i + 0.14, dl["coverage_conformal"].mean(), "s", color="k", ms=5.5)
        ax.axhline(NOMINAL, ls="--", color="0.2", lw=0.8)
        level_axis(ax, range(3))
        ax.set_ylim(0.40, 1.01)
        ax.set_title(f"({letter}) {tl}", loc="left")
    axes[0].set_ylabel("coverage of the clean benchmark")
    h = [Line2D([], [], marker="o", ls="", color="0.4", ms=4, label="uncorrected"),
         Line2D([], [], marker="s", ls="", color="0.4", ms=4, label="after LOBO conformal"),
         Line2D([], [], marker="o", ls="", mfc="white", mec="k", ms=6, label="mean, uncorrected"),
         Line2D([], [], marker="s", ls="", color="k", ms=5.5, label="mean, after conformal")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig3_conformal")


def fig4():
    pan = panel_platt()
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for off, (t, tl) in zip((-0.24, 0.0, 0.24), TRACKS):
        for i, lvl in enumerate(LEVELS):
            v = pan[(pan["track"] == t) & (pan["level"] == lvl)]["width_inflation_factor"].to_numpy()
            x = i + off + rng.uniform(-0.06, 0.06, len(v))
            ax.scatter(x, v, s=9, color=TRACK_COLS[t], alpha=0.8, label=tl if i == 0 else None)
            ax.plot([i + off - 0.09, i + off + 0.09], [np.median(v)] * 2, color=TRACK_COLS[t], lw=2)
    ax.axhline(1, color="0.3", lw=0.6, ls=":")
    level_axis(ax, range(3))
    ax.set_ylabel("width inflation factor")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig4_width_inflation")
    w = pan["width_inflation_factor"]
    corr = pan[w > 1.0005]
    print("fig4 medians: all eight units per level | corrected conditions only, pooled over levels")
    for t, tl in TRACKS:
        a_ = pan[pan["track"] == t]
        c_ = corr[corr["track"] == t]
        print(f"  {tl:13s} " + " ".join(f"L{l} {a_[a_['level'] == l]['width_inflation_factor'].median():.2f}"
                                        for l in LEVELS) +
              f" | corrected n = {len(c_)}, median {c_['width_inflation_factor'].median():.2f}")
    print(f"  all tracks: corrected n = {len(corr)}, median {corr['width_inflation_factor'].median():.2f}, "
          f"range {corr['width_inflation_factor'].min():.2f}-{w.max():.2f}")


def fig5b():
    w = pd.read_csv(R / "worked_case_null_bands.csv").rename(columns={
        "sub_mean": "c", "sub_lo": "c_lo", "sub_hi": "c_hi", "null_mean": "z", "null_lo": "z_lo",
        "null_hi": "z_hi", "n_subc": "n"})
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    band_plot(ax, w, "upper right")
    ax.set_ylabel("divergence from the benchmark")
    ax.set_title("(b)", loc="left")
    fig.tight_layout()
    save(fig, "fig5b_worked_case_null")


def fig6():
    p = R / "odonata_null_bands.csv"
    if not p.exists():
        print("fig6 skipped: reports/rev_eds/odonata_null_bands.csv not found")
        return
    o = pd.read_csv(p).rename(columns={
        "cont_mean": "c", "cont_lo": "c_lo", "cont_hi": "c_hi", "null_mean": "z", "null_lo": "z_lo",
        "null_hi": "z_hi"})
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9), sharey=True)
    for ax, (dom, title), loc in zip(axes, [("full", "(a) all records"),
                                           ("held_out", "(b) records never used in a fit")], ("best", None)):
        band_plot(ax, o[o["domain"] == dom], loc)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("divergence from the benchmark")
    fig.tight_layout()
    save(fig, "fig6_odonata_null")


def figS():
    a = pd.read_csv(R / "item1_protocol_a_join.csv")
    a["competent"] = a["competent"].astype(bool)
    bo = pd.read_csv(R / "protocol_b_platt_oos.csv")
    bo = bo[(bo["method"] == "platt") & (bo["level"] > 0)]
    bj = panel_platt().merge(bo[["entity_dir", "track", "level", "auc"]], on=["entity_dir", "track", "level"])
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0), sharey=True)
    for lvl, col in zip(LEVELS, LEVEL_COLS):
        c = a[a["competent"] & (a["level"] == lvl)]
        axes[0].scatter(c["auc_cont_oos"], c["coverage_uncorrected"], s=10, color=col, zorder=3)
        d = bj[bj["level"] == lvl]
        axes[1].scatter(d["auc"], d["coverage"], s=12, color=col, zorder=3)
    x = a[~a["competent"]]
    axes[0].scatter(x["auc_cont_oos"], x["coverage_uncorrected"], s=12, marker="x", color="0.5", zorder=3)
    for ax, title in zip(axes, ["(a) Protocol A, 30-replicate ensembles", "(b) Protocol B, calibrated consensus"]):
        ax.set_ylim(0.35, 1.03)
        x0, x1 = ax.get_xlim()
        ax.fill_between([GATE, x1], 0.35, THRESH, color="0.9", lw=0, zorder=0)
        ax.set_xlim(x0, x1)
        ax.axvline(GATE, color="0.3", lw=0.7, ls=":")
        ax.axhline(NOMINAL, color="0.2", lw=0.8, ls="--")
        ax.axhline(THRESH, color="0.2", lw=0.6)
        ax.set_title(title, loc="left")
        ax.set_xlabel("cross-validated AUC of the contaminated model")
    axes[0].set_ylabel("coverage of the clean benchmark")
    h = [Line2D([], [], marker="o", ls="", color=c, ms=4, label=f"L{l}") for l, c in zip(LEVELS, LEVEL_COLS)]
    h += [Line2D([], [], marker="x", ls="", color="0.5", ms=4, label="excluded (clean AUC < 0.70)"),
          Patch(facecolor="0.9", label="silent: passes the gate, coverage < 0.90"),
          Line2D([], [], color="0.2", lw=0.8, ls="--", label="nominal 0.95"),
          Line2D([], [], color="0.2", lw=0.6, label="miscoverage threshold 0.90"),
          Line2D([], [], color="0.3", lw=0.7, ls=":", label="gate: AUC 0.70")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "figS_cv_vs_coverage")
    for proto, d, auc, cov in (("Protocol A (competent)", a[a["competent"]], "auc_cont_oos", "coverage_uncorrected"),
                               ("Protocol B (platt)", bj, "auc", "coverage")):
        s = [f"{int(((d['level'] == l) & (d[auc] >= GATE) & (d[cov] < THRESH)).sum())}/"
             f"{int(((d['level'] == l) & (d[cov] < THRESH)).sum())}" for l in LEVELS]
        print(f"figS {proto}: silent/miscovered L3/L10/L20 = {' '.join(s)}")


def main():
    for f in (fig1, fig2, fig3, fig4, fig5b, fig6, figS):
        f()


if __name__ == "__main__":
    main()
