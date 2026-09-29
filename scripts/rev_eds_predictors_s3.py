"""EDS-2026-0083 revision: predictor table for supplement S3, straight from the pipeline.

One row per predictor column of the master table (Hydrography90m via GeoFRESH). Columns:
predictor, scale (local for l_*, upstream for u_*, as in
sdm_robustness.pipeline.core.get_track_columns), base_name (prefix stripped), the tracks
using it (local-only: l_*; upstream-only: u_*; combined: both), missing_share on the full
master table, and every column of the companion study's data/variable_domain_mapping.csv,
joined on the best-matching key. Per-unit retention under the Protocol A (30 %) and
Protocol B (70 %) missingness screens is not repeated here; it is the predictor counts of
table S1. Output: reports/rev_eds/predictors_s3.csv.
"""
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MASTER = Path(os.environ.get("TS_MASTER_CSV") or
              REPO.parent / "sdm-robustness/data/raw/combined_data_true_master.csv")
MAPPING = REPO.parent / "sdm-robustness-v1.0/data/variable_domain_mapping.csv"
OUT = REPO / "reports/rev_eds/predictors_s3.csv"


def main():
    m = pd.read_csv(MASTER, low_memory=False)
    preds = [c for c in m.columns if c.startswith(("l_", "u_"))]
    t = pd.DataFrame({"predictor": preds})
    t["scale"] = t["predictor"].str[0].map({"l": "local", "u": "upstream"})
    t["base_name"] = t["predictor"].str[2:]
    t["tracks"] = t["scale"].map({"local": "local-only, combined", "upstream": "upstream-only, combined"})
    t["missing_share_master"] = t["predictor"].map(lambda c: float(m[c].isna().mean()))

    mp = pd.read_csv(MAPPING)
    print(f"mapping file: {mp.shape}, columns {list(mp.columns)}")
    best_col, best_kind, best_hits = None, None, -1
    for c in mp.columns:
        vals = set(mp[c].astype(str))
        for kind, ref in (("predictor", set(t["predictor"])), ("base_name", set(t["base_name"]))):
            hits = len(vals & ref)
            if hits > best_hits:
                best_col, best_kind, best_hits = c, kind, hits
    print(f"join: mapping['{best_col}'] on {best_kind}, {best_hits} of {len(t)} predictors matched")
    mp = mp.rename(columns={best_col: best_kind}).drop_duplicates(subset=[best_kind])
    res = t.merge(mp, on=best_kind, how="left")
    res.sort_values(["scale", "predictor"]).round(4).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO)}: {len(res)} predictors "
          f"({int((res['scale'] == 'local').sum())} local, {int((res['scale'] == 'upstream').sum())} upstream)")
    unmatched = res[res[mp.columns.difference([best_kind])[0]].isna()] if len(mp.columns) > 1 else res.iloc[0:0]
    if len(unmatched):
        print(f"unmatched in mapping: {len(unmatched)}; first few: {unmatched['predictor'].head(6).tolist()}")
    print("\nmissing share by scale (min / median / max):")
    print(res.groupby("scale")["missing_share_master"].agg(["min", "median", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
