"""EDS-2026-0083 revision: memorization control for Protocol B (pre-registered).

The evaluation surface (target-group sub-catchments in occupied basins) also
holds training sites: the background draw, identical in the clean and the
contaminated fit (same seed and size), and in contaminated cells the
sub-catchments of the contaminant records themselves. Tree members nearly
memorise training points, so part of the contaminated-minus-clean divergence
could sit mechanically on those sites.

For every contaminated Protocol B cell, from the submitted member surfaces:
coverage of the four-member interval (2.5-97.5 percentiles) against the clean
benchmark (mean of the four clean members) and the median signed divergence
of the consensus point, on four site sets:
  all            every surface site (must reproduce panel_summary_protocol_b.csv)
  no_contam      without this cell's contaminant sites
  no_pool        without any low-accuracy-pool site of the taxon
  never_trained  without contaminant and background sites
plus the fraction of sites where the consensus exceeds the benchmark in five
equal-count benchmark bins, for 'all' and 'never_trained'.

Pre-registered criteria (fixed before the run): the contamination effect is
not a memorization artefact if (1) over cells with coverage < 0.90 on 'all',
the median ratio of coverage deficits (0.95 - coverage) never_trained/all is
>= 0.75, and (2) at L10 and L20 the panel-mean gradient in the fraction
over-predicted (lowest minus highest benchmark bin) on 'never_trained' is
>= 0.75 of that on 'all'. Exit code 1 only if 'all' fails to reproduce the
submitted coverage.

Outputs: reports/rev_eds/memorization_control_protocol_b.csv (per cell)
         reports/rev_eds/memorization_control_protocol_b_bins.csv (per cell x bin)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SURF = REPO / "data/replicate_surfaces_protocol_b"
WORK = REPO / "data/rev_eds/protocol_b_cv"
SUMM = REPO / "figures/panel_summary_protocol_b.csv"
OUT = REPO / "reports/rev_eds/memorization_control_protocol_b.csv"
OUT_BINS = REPO / "reports/rev_eds/memorization_control_protocol_b_bins.csv"
REPS = ["rep_00", "rep_01", "rep_02", "rep_03"]
N_BINS = 5
RATIO_MIN = 0.75


def load_members(cell_dir):
    parts = []
    for r in REPS:
        d = pd.read_parquet(cell_dir / f"{r}.parquet")
        parts.append(pd.Series(d["predicted_probability"].to_numpy(dtype=float),
                               index=d["subc_id"].astype(str).to_numpy(), name=r))
    return pd.concat(parts, axis=1)


def interval_stats(C, bench):
    lo = np.percentile(C, 2.5, axis=1)
    hi = np.percentile(C, 97.5, axis=1)
    return {"n": int(len(bench)),
            "coverage": float(np.mean((bench >= lo) & (bench <= hi))),
            "msd_mean": float(np.median(C.mean(axis=1) - bench)),
            "msd_median": float(np.median(np.median(C, axis=1) - bench))}


def bin_fracs(C, bench):
    q = pd.qcut(pd.Series(bench).rank(method="first"), N_BINS, labels=False)
    df = pd.DataFrame({"bin": q.to_numpy(), "over_mean": C.mean(axis=1) > bench,
                       "over_median": np.median(C, axis=1) > bench})
    g = df.groupby("bin")
    return pd.DataFrame({"n": g.size(), "frac_over_mean": g["over_mean"].mean(),
                         "frac_over_median": g["over_median"].mean()}).reset_index()


def main():
    rows, brows = [], []
    for cdir in sorted(SURF.glob("*__lowacc__L*")):
        edir, _, track, _, lvl = cdir.name.split("__")
        level = int(lvl[1:])
        C = load_members(cdir)
        B = load_members(SURF / f"{edir}__consensus__{track}__benchmark__L0")
        idx = C.index.intersection(B.index)
        Cv = C.loc[idx].to_numpy()
        bench = B.loc[idx].mean(axis=1).to_numpy()
        sites = pd.read_parquet(WORK / cdir.name / "training_sites.parquet")
        sid = sites["subc_id"].astype(str)
        contam = set(sid[sites["role"] == "contaminant"])
        backg = set(sid[sites["role"] == "background"])
        pool = set(pd.read_parquet(WORK / f"{edir}__pool_sites.parquet")["subc_id"].astype(str))
        ids = pd.Index(idx)
        masks = {"all": np.ones(len(ids), dtype=bool),
                 "no_contam": ~ids.isin(list(contam)),
                 "no_pool": ~ids.isin(list(pool)),
                 "never_trained": ~ids.isin(list(contam | backg))}
        row = {"entity_dir": edir, "track": track, "level": level}
        for name, m in masks.items():
            for k, v in interval_stats(Cv[m], bench[m]).items():
                row[f"{name}_{k}"] = v
        rows.append(row)
        for name in ("all", "never_trained"):
            m = masks[name]
            brows.append(bin_fracs(Cv[m], bench[m]).assign(entity_dir=edir, track=track,
                                                            level=level, mask=name))
    res = pd.DataFrame(rows)
    bins = pd.concat(brows, ignore_index=True)

    summ = pd.read_csv(SUMM)
    summ["level"] = pd.to_numeric(summ["level"].astype(str).str.lstrip("L"))
    chk = res.merge(summ[["entity_dir", "track", "level", "coverage", "median_signed_diff"]],
                    on=["entity_dir", "track", "level"], how="left")
    d_cov = float((chk["all_coverage"] - chk["coverage"]).abs().max())
    d_mean = float((chk["all_msd_mean"] - chk["median_signed_diff"]).abs().max())
    d_med = float((chk["all_msd_median"] - chk["median_signed_diff"]).abs().max())
    point = "mean" if d_mean <= d_med else "median"
    print(f"cells: {len(res)}; reproduction of submitted coverage: max |diff| {d_cov:.2e}")
    print(f"submitted median signed diff matches the {point} consensus "
          f"(max |diff|: mean {d_mean:.2e}, median {d_med:.2e})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.round(5).to_csv(OUT, index=False)
    bins.round(5).to_csv(OUT_BINS, index=False)

    cov_cols = [f"{m}_coverage" for m in ("all", "no_contam", "no_pool", "never_trained")]
    s = res.groupby(["track", "level"])[cov_cols].mean()
    s["never_trained_share"] = (res["never_trained_n"] / res["all_n"]).groupby(
        [res["track"], res["level"]]).mean()
    print("\nmean coverage by site set:")
    print(s.round(3).to_string())

    mis = res[res["all_coverage"] < 0.90]
    ratio = float(((0.95 - mis["never_trained_coverage"]) / (0.95 - mis["all_coverage"])).median())
    still = int((mis["never_trained_coverage"] < 0.90).sum())
    cov_pass = ratio >= RATIO_MIN
    print(f"\ncells below 0.90 on all sites: {len(mis)}; still below 0.90 on never-trained sites: {still}; "
          f"median deficit ratio {ratio:.2f} -> criterion 1 {'PASS' if cov_pass else 'FAIL'}")

    col = f"frac_over_{point}"
    g = bins.pivot_table(index=["mask", "level"], columns="bin", values=col, aggfunc="mean")
    print(f"\nfraction over-predicted ({point} consensus) by benchmark bin, panel mean:")
    print(g.round(3).to_string())
    grad = g[0] - g[N_BINS - 1]
    ratios = []
    for lvl in (10, 20):
        r = float(grad.loc[("never_trained", lvl)] / grad.loc[("all", lvl)])
        ratios.append(r)
        print(f"L{lvl}: gradient all {grad.loc[('all', lvl)]:+.3f}, "
              f"never-trained {grad.loc[('never_trained', lvl)]:+.3f}, ratio {r:.2f}")
    grad_pass = all(r >= RATIO_MIN for r in ratios)
    print(f"criterion 2 {'PASS' if grad_pass else 'FAIL'}")
    sys.exit(0 if d_cov < 1e-6 else 1)


if __name__ == "__main__":
    main()
