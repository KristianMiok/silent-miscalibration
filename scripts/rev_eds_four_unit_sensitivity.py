"""EDS-2026-0083 revision: sensitivity excluding the four units with > 30 % upstream missingness.

A. fulcisianus, P. leptodactylus and P. clarkii (both ranges) are the units whose upstream
predictors exceed 30 % missingness, so that Protocol A's screen leaves a single upstream
predictor and their upstream-only Protocol A configurations fail the competence gate.
From the committed CSVs, the headline Protocol B results (calibrated members) and the
Protocol A null-adjusted silent counts are recomputed on the other four units
(A. astacus, A. torrentium, F. limosus, P. leniusculus). Excess divergence per condition
is the mean over the equal-count benchmark quintiles.

Pre-registered reading (fixed and committed before the run):
  the headline results hold without the four units if, on the local and combined tracks
  of the remaining units, (1) mean coverage declines with level and is below 0.90 at L20,
  and (2) the mean excess divergence over the null is positive at every level and rises
  with level;
  P. clarkii in its native range shows the same profile if its mean excess divergence
  (local and combined) is positive at every level and rises with level.
Output: reports/rev_eds/four_unit_sensitivity.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
R = REPO / "reports/rev_eds"
OUT = R / "four_unit_sensitivity.csv"
EXCLUDED = ["Austropotamobius_fulcisianus_pooled", "Pontastacus_leptodactylus_pooled",
            "Procambarus_clarkii_alien", "Procambarus_clarkii_native"]
NATIVE = "Procambarus_clarkii_native"
LEVELS = (3, 10, 20)
TRACKS = ("local_only", "upstream_only", "combined")
SCORED = ("local_only", "combined")
THRESH, GATE = 0.90, 0.70


def rising_positive(v):
    v = np.asarray(v, float)
    return bool((v > 0).all() and (np.diff(v) > 0).all())


def main():
    pan = pd.read_csv(R / "protocol_b_platt_panel.csv")
    pan = pan[pan["method"] == "platt"]
    oos = pd.read_csv(R / "protocol_b_platt_oos.csv")
    oos = oos[(oos["method"] == "platt") & (oos["level"] > 0)]
    bins = pd.read_csv(R / "protocol_b_platt_bins.csv")
    bins = bins[bins["method"] == "platt"]
    exc = bins.groupby(["entity_dir", "track", "level"])["excess"].mean().rename("excess_div").reset_index()
    pan = (pan.merge(oos[["entity_dir", "track", "level", "auc"]], on=["entity_dir", "track", "level"])
              .merge(exc, on=["entity_dir", "track", "level"]))
    pa = pd.read_csv(R / "protocol_a_null_panel.csv")

    rows = []
    for subset in ("all eight", "four retained"):
        b = pan if subset == "all eight" else pan[~pan["entity_dir"].isin(EXCLUDED)]
        a = pa if subset == "all eight" else pa[~pa["entity_dir"].isin(EXCLUDED)]
        for t in TRACKS + ("local+combined",):
            tr = SCORED if t == "local+combined" else (t,)
            bt, at = b[b["track"].isin(tr)], a[a["track"].isin(tr)]
            for lvl in LEVELS:
                bl, al = bt[bt["level"] == lvl], at[at["level"] == lvl]
                mis = bl["coverage"] < THRESH
                rows.append({"subset": subset, "track": t, "level": lvl, "n_units": bl["entity_dir"].nunique(),
                             "b_mean_coverage": bl["coverage"].mean(), "b_miscovered": int(mis.sum()),
                             "b_silent": int((mis & (bl["auc"] >= GATE)).sum()),
                             "b_excess_divergence": bl["excess_div"].mean(), "a_configs": len(al),
                             "a_silent_adj": int(al["silent_adj"].sum()),
                             "a_miscovered_adj": int(al["miscovered_adj"].sum())})
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT, index=False)
    print(res.round(3).to_string(index=False))

    allm = res[(res["subset"] == "all eight") & (res["track"] == "local+combined")]["b_excess_divergence"]
    print("\ncheck, all eight units, local+combined excess divergence: " +
          " / ".join(f"{v:+.3f}" for v in allm) + " (text: 0.009 / 0.024 / 0.039)")
    r = res[(res["subset"] == "four retained") & (res["track"] == "local+combined")].set_index("level")
    cov, ex = r["b_mean_coverage"].to_numpy(), r["b_excess_divergence"].to_numpy()
    c1 = bool((np.diff(cov) < 0).all() and cov[-1] < THRESH)
    c2 = rising_positive(ex)
    print("\npre-registered reading, four retained units, local+combined:")
    print(f"  (1) coverage {' / '.join(f'{v:.3f}' for v in cov)} -> {'PASS' if c1 else 'FAIL'}")
    print(f"  (2) excess divergence {' / '.join(f'{v:+.3f}' for v in ex)} -> {'PASS' if c2 else 'FAIL'}")
    print(f"  -> headline results {'hold' if c1 and c2 else 'do not hold'} without the four units")

    print(f"\n{NATIVE}, Protocol B calibrated: coverage and excess divergence by quintile")
    nat = []
    for lvl in LEVELS:
        for t in SCORED:
            c = pan[(pan["entity_dir"] == NATIVE) & (pan["track"] == t) & (pan["level"] == lvl)]
            g = bins[(bins["entity_dir"] == NATIVE) & (bins["track"] == t) & (bins["level"] == lvl)].sort_values("bin")
            print(f"  {t:10s} L{lvl:>2}: coverage {c['coverage'].iloc[0]:.3f}; quintiles "
                  + " ".join(f"{v:+.3f}" for v in g["excess"]) + f"; mean {g['excess'].mean():+.3f}")
        nat.append(pan[(pan["entity_dir"] == NATIVE) & pan["track"].isin(SCORED) & (pan["level"] == lvl)]["excess_div"].mean())
    print(f"  mean excess, local+combined: {' / '.join(f'{v:+.3f}' for v in nat)} -> "
          f"{'same profile' if rising_positive(nat) else 'not the same profile'}")


if __name__ == "__main__":
    main()
