"""EDS-2026-0083 revision, item 1 (Protocol A): out-of-sample performance.

Reviewer 3 argues the base models may simply be weak and that ordinary
cross-validation would have flagged them. For Protocol A this is answerable
from existing outputs: the companion paper's Grid B stores basin-blocked
5-fold CV metrics (AUC, TSS, Brier, Boyce) for every replicate, fitted from
the same frozen seeds from which the Protocol A ensembles were regenerated.

Clean reference: axis 'benchmark', level 0.
Contaminated: axis 'lowacc', levels 3/10/20. Folds are assigned on the
contaminated presence set, so held-out folds contain contaminated records,
i.e. what a practitioner's CV would report.

Output: reports/rev_eds/item1_protocol_a_oos.csv, one row per
entity x algorithm x track x level: replicate mean/sd, replicate and fold
counts, presence and segment counts, deltas against the clean reference.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data/results/grid_b_merged/grid_b_results_raw_merged.parquet"
OUT = REPO / "reports/rev_eds/item1_protocol_a_oos.csv"

PANEL = [
    "Procambarus clarkii (alien)", "Pacifastacus leniusculus (alien)",
    "Faxonius limosus (alien)", "Astacus astacus",
    "Procambarus clarkii (native)", "Pontastacus leptodactylus (pooled)",
    "Austropotamobius torrentium (pooled)", "Austropotamobius fulcisianus (pooled)",
]
ALGOS = ["random_forest", "xgboost"]
LEVELS = [3, 10, 20]
METRICS = ["auc", "tss", "brier", "boyce"]
KEYS = ["entity", "algorithm", "track"]


def main():
    g = pd.read_parquet(SRC)
    g = g[g["entity"].isin(PANEL) & g["algorithm"].isin(ALGOS)]
    clean = g[(g["axis"] == "benchmark") & (g["level"] == 0)]
    cont = g[(g["axis"] == "lowacc") & g["level"].isin(LEVELS)]
    d = pd.concat([clean, cont], ignore_index=True)
    ok = d[d["status"] == "ok"]
    print(f"replicate rows: {len(d)}, ok: {len(ok)}, error: {len(d) - len(ok)}")

    grp = ok.groupby(KEYS + ["axis", "level"])
    # list-like .agg on a column-selected groupby raises IndexError under the
    # pandas version in this environment, so mean and std are computed apart
    agg = pd.concat([grp[METRICS].mean().add_suffix("_mean"),
                     grp[METRICS].std().add_suffix("_std")], axis=1)
    agg["n_reps_ok"] = grp.size()
    agg["min_folds"] = grp["n_folds_completed"].min()
    agg["benchmark_presence_n"] = grp["benchmark_presence_n"].median()
    agg["segments"] = grp["accessible_area_segment_count"].median()
    agg = agg.reset_index()

    ref = agg[agg["axis"] == "benchmark"][KEYS + [f"{m}_mean" for m in METRICS]]
    ref = ref.rename(columns={f"{m}_mean": f"{m}_clean" for m in METRICS})
    agg = agg.merge(ref, on=KEYS, how="left")
    for m in METRICS:
        agg[f"delta_{m}"] = agg[f"{m}_mean"] - agg[f"{m}_clean"]

    expected = len(PANEL) * len(ALGOS) * 3 * (1 + len(LEVELS))
    print(f"cells: {len(agg)} of {expected} expected; min reps ok per cell: {agg['n_reps_ok'].min()}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.round(4).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}\n")

    cols = ["auc_mean", "delta_auc", "tss_mean", "delta_tss", "brier_mean", "delta_brier"]
    print(agg.groupby(["algorithm", "track", "level"])[cols].mean().round(3).to_string())

    c = agg[agg["axis"] == "lowacc"]
    b = agg[agg["axis"] == "benchmark"]
    print(f"\nclean AUC across entity x algorithm x track: {b['auc_mean'].min():.3f} to {b['auc_mean'].max():.3f}")
    print(f"contaminated cells, delta AUC: {c['delta_auc'].min():+.3f} to {c['delta_auc'].max():+.3f}")
    print(f"contaminated cells, delta TSS: {c['delta_tss'].min():+.3f} to {c['delta_tss'].max():+.3f}")


if __name__ == "__main__":
    main()
