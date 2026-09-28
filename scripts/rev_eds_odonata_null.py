"""EDS-2026-0083 revision: matched-perturbation null for the Odonata cross-taxon replicate (Fig. S3).

Source analysis: filter_bias/run_odonata_overpred.py (commit 458cf94), German Odonata,
target Pyrrhosoma nymphula, bio_1-bio_19 at record points, random forest (300 trees,
min_samples_leaf 2). Its contamination design is additive: every high-accuracy target
presence is kept and low-accuracy presences are added so that they make up L % of the
training presences (L = 3, 10, 20; 30 replicates), with a fixed target-group background.

  benchmark          High + background, seed 0 (as submitted)
  contaminated(L, r) High + Low[sel] + background, seed 1000 + r (as submitted; the
                     selection stream default_rng(L) is reproduced draw for draw)
  null(L, r)         High + High[dup] + background, seed 1000 + r: same presences,
                     background, seed and training size, but the added records are
                     duplicates of high-accuracy presences instead of low-accuracy ones,
                     as in the Protocol B panel null (item 3c); dup from default_rng(100 + L)

Divergence = prediction - benchmark, averaged within benchmark-suitability bands, on two
evaluation domains:
  full      every record in the domain of the submitted analysis
  held_out  records never used in any fit: no target record (High or Low), no background
            record, and no record whose climate vector equals one of theirs (identical
            predictors give identical predictions), so the result is not carried by the
            injected points themselves
Excess = contaminated - null, paired by replicate (shared seed); 95 % interval over the
30 paired differences.

Pre-registered reading (fixed and committed before the run), on each domain:
  (1) edge band [0.1, 0.3): excess > 0 at L3, L10 and L20, rising with level, and the
      95 % paired interval excluding zero at L10 and L20;
  (2) shape: excess in the core band [0.7, 1.0) below the edge-band excess at every level.
A criterion is not evaluated on a band with fewer than 200 records. If (1) holds on the
full domain but not on held_out, the supplement reports the replicate as carried by the
injected points and the cross-taxon claim is narrowed accordingly.
Gate: the regenerated contaminated arm on the full domain against
filter_bias/reports/odonata_overpred_band_ci.csv.
Outputs: reports/rev_eds/odonata_null_reps.csv (per replicate),
         reports/rev_eds/odonata_null_bands.csv (summary).
"""
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

REPO = Path(__file__).resolve().parent.parent
FB = REPO.parent / "filter_bias"
ANNOT = FB / "reports/odonata_de_annotated.csv"
SUBMITTED = FB / "reports/odonata_overpred_band_ci.csv"
OUT = REPO / "reports/rev_eds"
ENV = [f"bio_{i}" for i in range(1, 20)]
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 5.5, 15.5, 47.0, 55.5
BANDS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0001]
TARGET = "Pyrrhosoma nymphula"
LEVELS = (3, 10, 20)
REPS = 30
NBG = 10000
N_TREES = 300
MIN_N = 200


def rf_predict(X_tr, y_tr, X_dom, seed):
    rf = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=seed, min_samples_leaf=2)
    rf.fit(X_tr, y_tr)
    return rf.predict_proba(X_dom)[:, 1]


def git(*args):
    return subprocess.run(["git", "-C", str(FB), *args], capture_output=True, text=True).stdout.strip()


def q025(s):
    return float(np.percentile(s, 2.5))


def q975(s):
    return float(np.percentile(s, 97.5))


def main():
    for p in (ANNOT, SUBMITTED):
        if not p.exists():
            sys.exit(f"missing {p}")
    dirty = git("status", "--porcelain", "run_odonata_overpred.py")
    print(f"{ANNOT.name} md5 {hashlib.md5(ANNOT.read_bytes()).hexdigest()}; filter_bias HEAD "
          f"{git('rev-parse', '--short', 'HEAD')}{' (run_odonata_overpred.py modified)' if dirty else ''}; "
          f"scikit-learn {sklearn.__version__}")

    df = pd.read_csv(ANNOT)
    inbox = (df["decimalLongitude"].between(LON_MIN, LON_MAX) & df["decimalLatitude"].between(LAT_MIN, LAT_MAX))
    df = df[inbox].dropna(subset=ENV).reset_index(drop=True)
    tgt_s = df["species"] == TARGET
    X_high = df.loc[tgt_s & (df["accuracy_class"] == "High"), ENV].to_numpy()
    X_low = df.loc[tgt_s & (df["accuracy_class"] == "Low"), ENV].to_numpy()
    bg_pool = df.loc[~tgt_s, ENV]
    bg = bg_pool.sample(n=min(NBG, len(bg_pool)), random_state=0)
    X_bg = bg.to_numpy()
    X_dom = df[ENV].to_numpy()
    n_high = len(X_high)

    key = pd.util.hash_pandas_object(df[ENV], index=False).to_numpy()
    used = tgt_s.to_numpy().copy()
    used[bg.index.to_numpy()] = True
    held = ~np.isin(key, np.unique(key[used]))
    domains = {"full": np.ones(len(df), dtype=bool), "held_out": held}
    print(f"{len(df)} records, {len(np.unique(key))} distinct climate vectors; target {n_high} High, "
          f"{len(X_low)} Low; background {len(X_bg)}; held-out domain {int(held.sum())} records "
          f"({100 * held.mean():.1f} %)")

    p_bench = rf_predict(np.vstack([X_high, X_bg]), np.r_[np.ones(n_high), np.zeros(len(X_bg))], X_dom, seed=0)
    band = pd.cut(pd.Series(p_bench), BANDS, right=False, include_lowest=True)
    cats = [str(c) for c in band.cat.categories]
    codes = band.cat.codes.to_numpy()
    assert (codes >= 0).all()
    K = len(cats)
    for name, m in domains.items():
        print(f"  {name:8s} n per band: " +
              ", ".join(f"{c} {int(k)}" for c, k in zip(cats, np.bincount(codes[m], minlength=K))))

    rows = []
    for L in LEVELS:
        t0 = time.time()
        f = L / 100.0
        n_low = min(int(round(f / (1 - f) * n_high)), len(X_low))
        rng = np.random.default_rng(L)
        rng_null = np.random.default_rng(100 + L)
        y = np.r_[np.ones(n_high + n_low), np.zeros(len(X_bg))]
        for rep in range(REPS):
            sel = rng.choice(len(X_low), size=n_low, replace=False)
            dup = rng_null.choice(n_high, size=n_low, replace=False)
            for arm, X_add in (("contaminated", X_low[sel]), ("null", X_high[dup])):
                d = rf_predict(np.vstack([X_high, X_add, X_bg]), y, X_dom, seed=1000 + rep) - p_bench
                for name, m in domains.items():
                    s = np.bincount(codes[m], weights=d[m], minlength=K)
                    n = np.bincount(codes[m], minlength=K)
                    for k in range(K):
                        if n[k]:
                            rows.append({"domain": name, "level": L, "rep": rep, "arm": arm, "band_index": k,
                                         "band": cats[k], "n": int(n[k]), "mean_div": s[k] / n[k]})
        print(f"  L{L}: {n_low} records added per replicate, {time.time() - t0:.0f} s", flush=True)

    reps = pd.DataFrame(rows)
    reps.to_csv(OUT / "odonata_null_reps.csv", index=False, float_format="%.6f")
    w = reps.pivot_table(index=["domain", "level", "band_index", "band", "n", "rep"], columns="arm",
                         values="mean_div").reset_index()
    w["excess"] = w["contaminated"] - w["null"]
    summ = (w.groupby(["domain", "level", "band_index", "band", "n"])
            .agg(cont_mean=("contaminated", "mean"), cont_lo=("contaminated", q025), cont_hi=("contaminated", q975),
                 null_mean=("null", "mean"), null_lo=("null", q025), null_hi=("null", q975),
                 excess_mean=("excess", "mean"), excess_lo=("excess", q025), excess_hi=("excess", q975))
            .reset_index())
    summ.round(5).to_csv(OUT / "odonata_null_bands.csv", index=False)

    sub = pd.read_csv(SUBMITTED)
    g = summ[summ["domain"] == "full"].merge(sub, on=["level", "band"], suffixes=("", "_sub"))
    print(f"\ngate vs submitted {SUBMITTED.name}: {len(g)} of {len(sub)} bands matched, n equal "
          f"{bool((g['n'] == g['n_sub']).all())}; max |diff| mean {(g['cont_mean'].round(4) - g['mean_div']).abs().max():.1e}, "
          f"2.5 % {(g['cont_lo'].round(4) - g['lo2.5']).abs().max():.1e}, "
          f"97.5 % {(g['cont_hi'].round(4) - g['hi97.5']).abs().max():.1e}")

    for name in domains:
        s = summ[summ["domain"] == name]
        print(f"\n{name}: contaminated, null and paired excess (mean [2.5, 97.5 %]) by band")
        for _, r in s.iterrows():
            print(f"  L{int(r['level']):>2} {r['band']:12s} n={int(r['n']):>6}  cont {r['cont_mean']:+.4f}  "
                  f"null {r['null_mean']:+.4f}  excess {r['excess_mean']:+.4f} "
                  f"[{r['excess_lo']:+.4f}, {r['excess_hi']:+.4f}]")

    edge, core = cats[1], cats[4]
    print(f"\npre-registered reading (edge {edge}, core {core}):")
    for name in domains:
        s = summ[summ["domain"] == name]
        e = s[s["band"] == edge].set_index("level").reindex(LEVELS)
        c = s[s["band"] == core].set_index("level").reindex(LEVELS)
        if e["n"].isna().any() or e["n"].min() < MIN_N:
            print(f"  {name}: (1) and (2) not evaluable (edge band n < {MIN_N})")
            continue
        em = e["excess_mean"].to_numpy()
        c1 = bool((em > 0).all() and (np.diff(em) > 0).all() and (e.loc[[10, 20], "excess_lo"] > 0).all())
        print(f"  {name}: (1) edge excess L3/L10/L20 = " + "/".join(f"{v:+.4f}" for v in em) +
              f", L10/L20 lower bounds " + "/".join(f"{v:+.4f}" for v in e.loc[[10, 20], "excess_lo"]) +
              f" -> {'PASS' if c1 else 'FAIL'}")
        if c["n"].isna().any() or c["n"].min() < MIN_N:
            print(f"  {name}: (2) not evaluable (core band n < {MIN_N})")
        else:
            cm = c["excess_mean"].to_numpy()
            print(f"  {name}: (2) core excess L3/L10/L20 = " + "/".join(f"{v:+.4f}" for v in cm) +
                  f" -> {'PASS' if (cm < em).all() else 'FAIL'}")


if __name__ == "__main__":
    main()
