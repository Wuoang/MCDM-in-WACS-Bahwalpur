#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROMETHEE II runner — step-by-step, paper-aligned, with numbered outputs and a for_paper subset.
- Inputs and outputs are configured at the TOP of this file (no CLI args).
- Saves ALL intermediate artifacts to OUTDIR with numbered filenames.
- Also creates OUTDIR/for_paper with only key results, numbered 01..

Method:
  1) Read PerformanceMatrix, CriteriaInfo, GlobalWeights from Excel.
  2) Validate and align criteria.
  3) For each criterion: compute pairwise preference indices using V-shape (for continuous)
     or Usual (for qualitative/ordinal) preference functions.
     - Here: All numeric criteria → V-shape auto-scaled by stdev.
     - Jobs, Acceptance (qualitative) → Usual (0 if equal, 1 if better).
  4) Aggregate weighted preference indices π(a,b) = Σ w_c * P_c(a,b).
  5) Compute Leaving flow (φ+), Entering flow (φ-), Net flow φ = φ+ - φ-.
  6) Rank scenarios by Net flow (descending).

Outputs in OUTDIR:
  01_PerformanceMatrix_Read.csv
  02_CriteriaInfo_Read.csv
  03_GlobalWeights_Read.csv
  04_GlobalWeights_Normalized.csv
  05_PairwisePreferences_PerCriterion.csv
  06_AggregatedPreferenceMatrix.csv
  07_LeavingEnteringNetFlows.csv
  08_PROMETHEEII_Ranking.csv
  09_README_mapping.txt

for_paper subset (OUTDIR/for_paper):
  01_PROMETHEEII_Ranking.csv
  02_PROMETHEEII_NetFlows.csv

Author: Your project
"""

# ============================== CONFIG ==============================
INPUT_XLSX = "scenario_performance_matrix.xlsx"

SHEET_PERFORMANCE = "PerformanceMatrix"
SHEET_CRITERIA    = "CriteriaInfo"
SHEET_WEIGHTS     = "GlobalWeights"

OUTDIR = "promethee2_outputs"

EPS = 1e-12

# For_paper subset selection
FOR_PAPER_SUFFIXES_IN_ORDER = [
    "PROMETHEEII_Ranking.csv",
    "LeavingEnteringNetFlows.csv",
]

# ============================== CODE ==============================
import os, math, glob, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd

# Global counter for numbering outputs
FILE_COUNTER = 1
def numbered_filename(base_name: str) -> str:
    global FILE_COUNTER
    fname = f"{FILE_COUNTER:02d}_{base_name}"
    FILE_COUNTER += 1
    return fname

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def read_inputs(xlsx_path: str):
    perf = pd.read_excel(xlsx_path, sheet_name=SHEET_PERFORMANCE)
    crit = pd.read_excel(xlsx_path, sheet_name=SHEET_CRITERIA)
    wts  = pd.read_excel(xlsx_path, sheet_name=SHEET_WEIGHTS)
    return perf, crit, wts

def validate_align(perf: pd.DataFrame, crit: pd.DataFrame, wts: pd.DataFrame):
    if "Scenario" not in perf.columns:
        raise ValueError("PerformanceMatrix must have a 'Scenario' column.")
    perf_criteria = [c for c in perf.columns if c != "Scenario"]
    crit_cols = crit["Criterion"].astype(str).tolist()
    wt_cols   = wts["Criterion"].astype(str).tolist()
    set_perf = set(perf_criteria)
    if set_perf != set(crit_cols) or set_perf != set(wt_cols):
        raise ValueError("Criteria mismatch across sheets.")
    crit = crit.set_index("Criterion").loc[perf_criteria].reset_index()
    wts  = wts.set_index("Criterion").loc[perf_criteria].reset_index()
    perf_num = perf.copy()
    for c in perf_criteria:
        perf_num[c] = pd.to_numeric(perf_num[c], errors="coerce")
    if perf_num[perf_criteria].isnull().any().any():
        raise ValueError("Missing / non-numeric performance values in PerformanceMatrix.")
    wts_num = wts.copy()
    wts_num["Weight"] = pd.to_numeric(wts_num["Weight"], errors="coerce")
    if wts_num["Weight"].isnull().any() or (wts_num["Weight"] < 0).any():
        raise ValueError("Invalid GlobalWeights.")
    sum_w = wts_num["Weight"].sum()
    if sum_w <= EPS: raise ValueError("GlobalWeights sum to zero or negative.")
    wts_norm = wts_num.copy()
    wts_norm["Weight"] = wts_norm["Weight"] / sum_w
    return perf_num, crit, wts_num, wts_norm, perf_criteria

def save_df(df: pd.DataFrame, name: str, outdir: str):
    df.to_csv(os.path.join(outdir, numbered_filename(name)), index=False)

def preference_vshape(a: float, b: float, s: float) -> float:
    """V-shape preference function with threshold s."""
    d = a - b
    if d <= 0: return 0.0
    return min(d/s, 1.0)

def preference_usual(a: float, b: float) -> float:
    return 1.0 if a > b else 0.0

def compute_preferences(perf: pd.DataFrame, crit: pd.DataFrame, wts_norm: pd.DataFrame, perf_criteria):
    scenarios = perf["Scenario"].tolist()
    n = len(scenarios)
    m = len(perf_criteria)
    weights = wts_norm.set_index("Criterion")["Weight"].to_dict()
    crit_types = crit.set_index("Criterion")["Type"].to_dict()
    # Stdev per criterion for V-shape scaling
    stdevs = {c: max(perf[c].std(ddof=0), EPS) for c in perf_criteria}

    # Build per-criterion preference matrices
    records = []
    pref_per_crit = {c: np.zeros((n,n)) for c in perf_criteria}
    for k, c in enumerate(perf_criteria):
        for i in range(n):
            for j in range(n):
                if i==j: continue
                a = perf.loc[i,c]; b = perf.loc[j,c]
                ctype = str(crit_types[c]).lower()
                if ctype == "benefit":
                    # Higher is better
                    if c in ["Jobs","Acceptance"]:
                        pref = preference_usual(a,b)
                    else:
                        pref = preference_vshape(a,b,stdevs[c])
                else: # cost criterion
                    if c in ["Jobs","Acceptance"]:
                        pref = preference_usual(-a,-b)
                    else:
                        pref = preference_vshape(b,a,stdevs[c]) # reverse for cost
                pref_per_crit[c][i,j] = pref
                records.append({"Criterion":c,"Scenario_i":scenarios[i],"Scenario_j":scenarios[j],"Preference":pref})
    df_records = pd.DataFrame(records)
    return pref_per_crit, df_records, weights, scenarios

def aggregate_preferences(pref_per_crit, weights, scenarios, perf_criteria):
    n = len(scenarios)
    agg = np.zeros((n,n))
    for c in perf_criteria:
        agg += weights[c] * pref_per_crit[c]
    # DataFrame
    rows = []
    for i in range(n):
        for j in range(n):
            if i==j: continue
            rows.append({"Scenario_i":scenarios[i],"Scenario_j":scenarios[j],"Pi":agg[i,j]})
    df = pd.DataFrame(rows)
    return agg, df

def compute_flows(agg: np.ndarray, scenarios: list):
    n = len(scenarios)
    leaving = agg.sum(axis=1) / (n-1)
    entering = agg.sum(axis=0) / (n-1)
    net = leaving - entering
    df = pd.DataFrame({
        "Scenario": scenarios,
        "LeavingFlow": leaving,
        "EnteringFlow": entering,
        "NetFlow": net
    })
    return df

def rank_scenarios(df_flows: pd.DataFrame):
    df_rank = df_flows.sort_values("NetFlow", ascending=False).reset_index(drop=True)
    df_rank.insert(0, "Rank", np.arange(1, len(df_rank)+1))
    return df_rank

def write_readme_mapping(outdir: str):
    lines = [
        "PROMETHEE II OUTPUT → STEP MAPPING",
        "01_PerformanceMatrix_Read.csv            : Raw performance matrix",
        "02_CriteriaInfo_Read.csv                 : Criterion types",
        "03_GlobalWeights_Read.csv                : Raw weights",
        "04_GlobalWeights_Normalized.csv          : Normalized weights",
        "05_PairwisePreferences_PerCriterion.csv  : All pairwise preference indices per criterion",
        "06_AggregatedPreferenceMatrix.csv        : Weighted aggregated preferences Pi(a,b)",
        "07_LeavingEnteringNetFlows.csv           : Leaving, entering, and net flows per scenario",
        "08_PROMETHEEII_Ranking.csv               : Final ranking by net flow",
    ]
    with open(os.path.join(outdir, numbered_filename("README_mapping.txt")), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _numeric_prefix(path: Path) -> int:
    m = re.match(r"^(\d+)_", path.name)
    return int(m.group(1)) if m else -1

def _find_by_suffix(outdir: str, suffix: str) -> Path | None:
    candidates = [Path(p) for p in glob.glob(os.path.join(outdir, f"*{suffix}"))]
    if not candidates: return None
    with_num = [p for p in candidates if _numeric_prefix(p) >= 0]
    if with_num:
        return sorted(with_num, key=_numeric_prefix, reverse=True)[0]
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def make_for_paper_subset(outdir: str):
    paper_dir = os.path.join(outdir, "for_paper")
    os.makedirs(paper_dir, exist_ok=True)
    for old in Path(paper_dir).glob("??_*.csv"):
        try: old.unlink()
        except Exception: pass
    copied = []
    # Ranking
    src_rank = _find_by_suffix(outdir, "PROMETHEEII_Ranking.csv")
    if src_rank:
        dst = Path(paper_dir)/"01_PROMETHEEII_Ranking.csv"
        shutil.copyfile(src_rank, dst)
        copied.append(dst.name)
    # Flows
    src_flows = _find_by_suffix(outdir, "LeavingEnteringNetFlows.csv")
    if src_flows:
        dst = Path(paper_dir)/"02_PROMETHEEII_NetFlows.csv"
        shutil.copyfile(src_flows, dst)
        copied.append(dst.name)
    readme = [
        "This folder contains the key PROMETHEE II results for the paper:",
        "01: Ranking by Net Flow",
        "02: Leaving, Entering, Net flows per scenario",
        "",
        "Files included:",
        *[f" - {x}" for x in copied]
    ]
    with open(os.path.join(paper_dir, "README_for_paper.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))

def main():
    ensure_outdir(OUTDIR)
    perf, crit, wts = read_inputs(INPUT_XLSX)
    save_df(perf, "PerformanceMatrix_Read.csv", OUTDIR)
    save_df(crit, "CriteriaInfo_Read.csv", OUTDIR)
    save_df(wts,  "GlobalWeights_Read.csv", OUTDIR)
    perf_num, crit_aligned, wts_num, wts_norm, perf_criteria = validate_align(perf, crit, wts)
    save_df(wts_norm, "GlobalWeights_Normalized.csv", OUTDIR)
    # Compute preferences
    pref_per_crit, df_records, weights, scenarios = compute_preferences(perf_num, crit_aligned, wts_norm, perf_criteria)
    save_df(df_records, "PairwisePreferences_PerCriterion.csv", OUTDIR)
    agg, df_agg = aggregate_preferences(pref_per_crit, weights, scenarios, perf_criteria)
    save_df(df_agg, "AggregatedPreferenceMatrix.csv", OUTDIR)
    df_flows = compute_flows(agg, scenarios)
    save_df(df_flows, "LeavingEnteringNetFlows.csv", OUTDIR)
    df_rank = rank_scenarios(df_flows)
    save_df(df_rank, "PROMETHEEII_Ranking.csv", OUTDIR)
    write_readme_mapping(OUTDIR)
    make_for_paper_subset(OUTDIR)
    print("\nPROMETHEE II complete. See:", OUTDIR)
    print(df_rank.to_string(index=False))

if __name__ == "__main__":
    main()
