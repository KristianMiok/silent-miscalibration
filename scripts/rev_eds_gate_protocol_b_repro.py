"""EDS-2026-0083 revision: reproduction gate for Protocol B (pre-registered).

Every new Protocol B analysis (cross-validation, member calibration) refits
models, so it must run on the upstream code that produced the submitted
surfaces. The upstream (sdm-robustness) is not pinned anywhere in this
repository; the candidate is tag v1.0 (8bbfb1d), installed editable from a
git worktree at ../sdm-robustness-v1.0.

Test: regenerate two submitted cells (Astacus astacus, combined track: the
clean benchmark L0 and lowacc L10) with scripts/run_protocol_b.py's own
functions and compare every member surface with the saved one by subc_id.

Pass criterion, fixed before the run: identical subc_id sets and
max |difference| < 1e-6 for all eight member surfaces. Exit code 1 on failure.

Output: reports/rev_eds/gate_protocol_b_repro.csv
"""
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
MASTER = Path(os.environ.get("TS_MASTER_CSV") or
              REPO.parent / "sdm-robustness/data/raw/combined_data_true_master.csv")
OUT = REPO / "reports/rev_eds/gate_protocol_b_repro.csv"
ENTITY = "Astacus astacus"
TRACK = "combined"
LEVELS = [0, 10]
TOL = 1e-6


def load_runner():
    spec = importlib.util.spec_from_file_location("run_protocol_b", REPO / "scripts/run_protocol_b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    import pygam
    import sklearn
    import xgboost
    import sdm_robustness
    from sdm_robustness.pipeline.core import clean_predictors, get_track_columns
    from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR
    from trustworthy_sdm.regen import assemble_inputs

    rpb = load_runner()
    up = Path(sdm_robustness.__file__).resolve().parent
    head = subprocess.run(["git", "-C", str(up), "log", "-1", "--format=%h%d"],
                          capture_output=True, text=True).stdout.strip()
    v = lambda m: getattr(m, "__version__", "?")
    print(f"upstream: {up} @ {head}")
    print(f"python {platform.python_version()}, numpy {v(np)}, pandas {v(pd)}, "
          f"sklearn {v(sklearn)}, xgboost {v(xgboost)}, pygam {v(pygam)}")
    print(f"master: {MASTER} md5 {md5(MASTER)}")

    t0 = time.time()
    inputs = assemble_inputs(entity=ENTITY, track="combined", axis=rpb.AXIS, master_csv=MASTER)
    print(f"inputs assembled in {time.time() - t0:.1f}s")
    full_cols = clean_predictors(inputs.benchmark, get_track_columns(inputs.benchmark, TRACK),
                                 missing_threshold_pct=rpb.MISSING_THRESHOLD_PCT)
    with open(REPO / "data/protocol_b_predictor_sets.json") as f:
        reduced_cols = json.load(f)[f"{ENTITY}|||{TRACK}"]["predictors"]
    edir = ENTITY_NAME_TO_DIR[ENTITY]

    rows = []
    for level in LEVELS:
        t1 = time.time()
        surf = rpb.consensus_surfaces_for_cell(inputs, TRACK, level, reduced_cols, full_cols, rpb.SEED)
        secs = time.time() - t1
        kind = "benchmark" if level == 0 else rpb.AXIS
        cell = REPO / "data/replicate_surfaces_protocol_b" / f"{edir}__consensus__{TRACK}__{kind}__L{level}"
        for algo, s in surf.items():
            saved = pd.read_parquet(cell / f"{rpb.ALGO_TO_REP[algo]}.parquet")
            saved = pd.Series(saved["predicted_probability"].values, index=saved["subc_id"].astype(str).values)
            new = pd.Series(np.asarray(s.values, dtype=float), index=pd.Index(s.index).astype(str))
            common = saved.index.intersection(new.index)
            diff = (new.loc[common] - saved.loc[common]).abs()
            rows.append({
                "entity": ENTITY, "track": TRACK, "level": level, "algorithm": algo,
                "n_saved": len(saved), "n_new": len(new), "n_common": len(common),
                "max_abs_diff": float(diff.max()) if len(common) else np.nan,
                "mean_abs_diff": float(diff.mean()) if len(common) else np.nan,
                "cell_fit_seconds": round(secs, 1),
            })
        print(f"L{level}: four members fitted in {secs:.1f}s")

    res = pd.DataFrame(rows)
    res["pass"] = ((res["n_saved"] == res["n_new"]) & (res["n_new"] == res["n_common"])
                   & (res["max_abs_diff"] < TOL))
    res["upstream"] = head
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(res.drop(columns=["entity", "track", "upstream"]).to_string(index=False))
    ok = bool(res["pass"].all())
    print(f"\nGATE {'PASS' if ok else 'FAIL'}; wrote {OUT.relative_to(REPO)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
