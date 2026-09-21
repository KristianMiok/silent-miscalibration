"""EDS-2026-0083 revision, item 4b: what travels with the low-accuracy flag.

Per taxon and accuracy class (and pooled), from the prepared tables:
Strahler order of the snapped sub-catchment (share first-order, share order
<= 2, median), snapping distance (median, 90th percentile, share beyond 200 m
and beyond 1 km), year of record (median, share from 2015 on), share
associated with a lake (hylak_id present), and the means of the ab_200m,
ab_500m and ab_1000m columns. Aggregates only.

Output: reports/rev_eds/item4b_metadata_by_class.csv
"""
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
MASTER = Path(os.environ.get("TS_MASTER_CSV") or
              REPO.parent / "sdm-robustness/data/raw/combined_data_true_master.csv")
OUT = REPO / "reports/rev_eds/item4b_metadata_by_class.csv"
SHORT = {"Procambarus clarkii (alien)": "Pcla_a", "Pacifastacus leniusculus (alien)": "Plen",
         "Faxonius limosus (alien)": "Flim", "Astacus astacus": "Aast",
         "Procambarus clarkii (native)": "Pcla_n", "Pontastacus leptodactylus (pooled)": "Plep",
         "Austropotamobius torrentium (pooled)": "Ator", "Austropotamobius fulcisianus (pooled)": "Aful"}


def load_runner():
    spec = importlib.util.spec_from_file_location("run_protocol_b", REPO / "scripts/run_protocol_b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def describe(df):
    s = pd.to_numeric(df["strahler"], errors="coerce")
    d = pd.to_numeric(df["distance_m"], errors="coerce")
    y = pd.to_numeric(df["Year_of_record"], errors="coerce")
    out = {"n": len(df),
           "strahler_1_share": float((s == 1).mean()),
           "strahler_le2_share": float((s <= 2).mean()),
           "strahler_median": float(s.median()),
           "snap_m_median": float(d.median()),
           "snap_m_p90": float(d.quantile(0.9)),
           "snap_gt_200m_share": float((d > 200).mean()),
           "snap_gt_1km_share": float((d > 1000).mean()),
           "year_median": float(y.median()),
           "year_ge_2015_share": float((y >= 2015).mean()),
           "lake_share": float(df["hylak_id"].notna().mean())}
    for c in ("ab_200m", "ab_500m", "ab_1000m"):
        out[f"{c}_mean"] = float(pd.to_numeric(df[c], errors="coerce").mean())
    return out


def main():
    warnings.simplefilter("ignore")
    rpb = load_runner()
    from trustworthy_sdm.regen import assemble_inputs
    rows, allb, allp = [], [], []
    for entity in rpb.ENTITIES:
        inp = assemble_inputs(entity=entity, track="combined", axis=rpb.AXIS, master_csv=MASTER)
        rows.append({"taxon": SHORT[entity], "class": "high", **describe(inp.benchmark)})
        rows.append({"taxon": SHORT[entity], "class": "low", **describe(inp.contamination_pool)})
        allb.append(inp.benchmark)
        allp.append(inp.contamination_pool)
    rows.append({"taxon": "ALL", "class": "high", **describe(pd.concat(allb, ignore_index=True))})
    rows.append({"taxon": "ALL", "class": "low", **describe(pd.concat(allp, ignore_index=True))})
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(OUT, index=False)
    t = res.assign(col=res["taxon"] + "_" + res["class"].str[0]).set_index("col").drop(columns=["taxon", "class"]).T
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
