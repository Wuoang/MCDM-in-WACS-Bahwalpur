#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOPSIS runner — step-by-step, paper-aligned, with numbered outputs and a for_paper subset.
- Inputs and outputs are configured at the TOP of this file (no CLI args).
- Saves ALL intermediate artifacts to OUTDIR with numbered filenames.
- Also creates OUTDIR/for_paper with only key results, numbered 01..

Method:
  1) Read PerformanceMatrix, CriteriaInfo, GlobalWeights from Excel.
  2) Validate and align criteria.
  3) Min–max normalize each criterion to [0,1].
  4) Reverse cost criteria so that higher is always better.
  5) Apply global weights: weighted normalized matrix.
  6) Ideal (best) & Anti-ideal (worst) per column.
  7) Distances (Euclidean) to ideal and anti-ideal.
  8) Closeness Coefficient CC = D- / (D+ + D-).
  9) Rank scenarios by CC (descending).

Outputs in OUTDIR:
  01_PerformanceMatrix_Read.csv
  02_CriteriaInfo_Read.csv
  03_GlobalWeights_Read.csv
  04_GlobalWeights_Normalized.csv
  05_Normalization_Summary.csv
  06_NormalizedMatrix_BenefitOriented.csv
  07_WeightedNormalizedMatrix.csv
  08_Ideal_AntiIdeal.csv
  09_Distances.csv
  10_Closeness_Coefficients.csv
  11_TOPSIS_Ranking.csv
  12_README_mapping.txt

for_paper subset (OUTDIR/for_paper):
  01_TOPSIS_Ranking.csv
  02_TOPSIS_Closeness_Coefficients.csv
  03_TOPSIS_Distances.csv

Author: Amjaf=d Mehmood
"""

# ============================== CONFIG ==============================
INPUT_XLSX = "scenario_performance_matrix.xlsx"

SHEET_PERFORMANCE = "PerformanceMatrix"
SHEET_CRITERIA    = "CriteriaInfo"
SHEET_WEIGHTS     = "GlobalWeights"

OUTDIR = "topsis_outputs"

# Normalization behavior
EPS = 1e-12           # numerical tolerance
DEGENERATE_FILL = 1.0 # when max == min, fill normalized values with this constant

# for_paper subset selection by suffix
FOR_PAPER_SUFFIXES_IN_ORDER = [
    "TOPSIS_Ranking.csv",
    "Closeness_Coefficients.csv",
    "Distances.csv",
]

# ============================== CODE ==============================
import os, math, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict

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
    # Basic checks
    if "Scenario" not in perf.columns:
        raise ValueError("PerformanceMatrix must have a 'Scenario' column.")
    # Criteria list from performance (exclude 'Scenario')
    perf_criteria = [c for c in perf.columns if c != "Scenario"]

    # Criteria in info and weights
    crit_cols = crit["Criterion"].astype(str).tolist()
    wt_cols   = wts["Criterion"].astype(str).tolist()

    # Ensure sets match
    set_perf = set(perf_criteria)
    set_info = set(crit_cols)
    set_wts  = set(wt_cols)
    if set_perf != set_info or set_perf != set_wts:
        raise ValueError(f"Criteria mismatch across sheets.\n"
                         f"Performance: {sorted(set_perf)}\n"
                         f"CriteriaInfo: {sorted(set_info)}\n"
                         f"GlobalWeights: {sorted(set_wts)}")

    # Align order of crit & wts to perf
    crit = crit.set_index("Criterion").loc[perf_criteria].reset_index()
    wts  = wts.set_index("Criterion").loc[perf_criteria].reset_index()

    # Validate cost/benefit tags
    valid_types = {"Cost","Benefit","cost","benefit"}
    bad = crit[~crit["Type"].astype(str).isin(valid_types)]
    if not bad.empty:
        raise ValueError(f"Invalid Type entries in CriteriaInfo (must be 'Cost' or 'Benefit'):\n{bad}")

    # Convert performance to numeric
    perf_num = perf.copy()
    for c in perf_criteria:
        perf_num[c] = pd.to_numeric(perf_num[c], errors="coerce")

    if perf_num[perf_criteria].isnull().any().any():
        nulls = perf_num[perf_criteria].isnull().sum()
        raise ValueError(f"Missing / non-numeric performance values detected:\n{nulls[nulls>0]}")

    # Check weights
    wts_num = wts.copy()
    wts_num["Weight"] = pd.to_numeric(wts_num["Weight"], errors="coerce")
    if wts_num["Weight"].isnull().any():
        raise ValueError("GlobalWeights contains non-numeric or missing weights.")

    if (wts_num["Weight"] < 0).any():
        raise ValueError("GlobalWeights has negative weights, which is not allowed.")

    sum_w = wts_num["Weight"].sum()
    if sum_w <= EPS:
        raise ValueError("GlobalWeights sum to zero or negative.")

    # Normalize weights to sum 1
    wts_norm = wts_num.copy()
    wts_norm["Weight"] = wts_norm["Weight"] / sum_w

    return perf_num, crit, wts_num, wts_norm, perf_criteria

def save_df(df: pd.DataFrame, name: str, outdir: str):
    df.to_csv(os.path.join(outdir, numbered_filename(name)), index=False)

def minmax_normalize(values: np.ndarray):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=EPS):
        return np.full_like(values, fill_value=DEGENERATE_FILL, dtype=float), vmin, vmax
    z = (values - vmin) / (vmax - vmin)
    return z.astype(float), vmin, vmax

def benefit_orient_and_weight(perf: pd.DataFrame, crit: pd.DataFrame, wts_norm: pd.DataFrame, perf_criteria: List[str]):
    # Normalize each criterion to [0,1]; reverse costs
    norm = pd.DataFrame({"Scenario": perf["Scenario"]})
    rows_summary = []
    for c in perf_criteria:
        vals = perf[c].values.astype(float)
        z, vmin, vmax = minmax_normalize(vals)
        ctype = str(crit.loc[crit["Criterion"]==c, "Type"].values[0]).strip().lower()
        if ctype == "cost":
            z = 1.0 - z
        norm[c] = z
        rows_summary.append({
            "Criterion": c, "Type": "Cost" if ctype=="cost" else "Benefit",
            "Min": vmin, "Max": vmax, "Degenerate": (abs(vmax-vmin) <= EPS)
        })
    summary = pd.DataFrame(rows_summary)
    # Weighted normalized
    weights = wts_norm.set_index("Criterion")["Weight"].to_dict()
    wnorm = norm.copy()
    for c in perf_criteria:
        wnorm[c] = wnorm[c].astype(float) * float(weights[c])
    return norm, wnorm, summary

def ideal_antiideal(wnorm: pd.DataFrame, perf_criteria: List[str]):
    # Ideal best = column max; Anti-ideal worst = column min
    ideals = []
    for c in perf_criteria:
        ideals.append({"Criterion": c,
                       "Ideal_Best": float(wnorm[c].max()),
                       "AntiIdeal_Worst": float(wnorm[c].min())})
    return pd.DataFrame(ideals)

def distances_and_cc(wnorm: pd.DataFrame, perf_criteria: List[str], df_ideal: pd.DataFrame):
    # Extract vectors
    X = wnorm[perf_criteria].values.astype(float)  # (n_scenarios, n_criteria)
    v_plus  = df_ideal.set_index("Criterion")["Ideal_Best"].loc[perf_criteria].values.astype(float)
    v_minus = df_ideal.set_index("Criterion")["AntiIdeal_Worst"].loc[perf_criteria].values.astype(float)

    # Euclidean distances to ideal and anti-ideal
    D_plus  = np.sqrt(((X - v_plus)**2).sum(axis=1))
    D_minus = np.sqrt(((X - v_minus)**2).sum(axis=1))

    # Closeness coefficient
    CC = D_minus / (D_plus + D_minus + EPS)

    df_dist = pd.DataFrame({
        "Scenario": wnorm["Scenario"],
        "D_plus": D_plus,
        "D_minus": D_minus
    })
    df_cc = pd.DataFrame({
        "Scenario": wnorm["Scenario"],
        "CC": CC
    })
    # Ranking
    df_rank = df_cc.merge(df_dist, on="Scenario")
    df_rank = df_rank.sort_values("CC", ascending=False).reset_index(drop=True)
    df_rank.insert(0, "Rank", np.arange(1, len(df_rank)+1))
    return df_dist, df_cc, df_rank

def write_readme_mapping(outdir: str):
    lines = [
        "TOPSIS OUTPUT → STEP MAPPING",
        "01_PerformanceMatrix_Read.csv            : Raw performance matrix as read",
        "02_CriteriaInfo_Read.csv                 : Criterion types (Cost/Benefit) as read",
        "03_GlobalWeights_Read.csv                : Raw weights as read",
        "04_GlobalWeights_Normalized.csv          : Weights renormalized to sum 1",
        "05_Normalization_Summary.csv             : Min/Max per criterion + degenerate flags",
        "06_NormalizedMatrix_BenefitOriented.csv  : Min–max normalized; costs reversed so higher is better",
        "07_WeightedNormalizedMatrix.csv          : Normalized matrix multiplied by weights",
        "08_Ideal_AntiIdeal.csv                   : Ideal best and anti-ideal worst per criterion",
        "09_Distances.csv                         : Euclidean distances to ideal (+) and anti-ideal (-)",
        "10_Closeness_Coefficients.csv            : CC = D- / (D+ + D-)",
        "11_TOPSIS_Ranking.csv                    : Final ranking by CC (descending)",
        "",
        "for_paper subset contains only:",
        "  01_TOPSIS_Ranking.csv",
        "  02_TOPSIS_Closeness_Coefficients.csv",
        "  03_TOPSIS_Distances.csv",
    ]
    with open(os.path.join(outdir, numbered_filename("README_mapping.txt")), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _numeric_prefix(path: Path) -> int:
    m = re.match(r"^(\d+)_", path.name)
    return int(m.group(1)) if m else -1

def _find_by_suffix(outdir: str, suffix: str) -> Path | None:
    candidates = [Path(p) for p in glob.glob(os.path.join(outdir, f"*{suffix}"))]
    if not candidates:
        return None
    with_num = [p for p in candidates if _numeric_prefix(p) >= 0]
    if with_num:
        return sorted(with_num, key=_numeric_prefix, reverse=True)[0]
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def make_for_paper_subset(outdir: str):
    paper_dir = os.path.join(outdir, "for_paper")
    os.makedirs(paper_dir, exist_ok=True)
    # Clean existing numbered csvs
    for old in Path(paper_dir).glob("??_*.csv"):
        try: old.unlink()
        except Exception: pass

    copied = []
    for idx, suffix in enumerate(FOR_PAPER_SUFFIXES_IN_ORDER, start=1):
        src = _find_by_suffix(outdir, suffix)
        if src is None:
            copied.append(f"(missing) {idx:02d}_{suffix}")
            continue
        dst = Path(paper_dir) / f"{idx:02d}_{suffix if suffix.startswith('TOPSIS_') else 'TOPSIS_'+suffix}"
        # Normalize names to start with TOPSIS_ in for_paper
        if not suffix.startswith("TOPSIS_"):
            # rename to TOPSIS_Suffix
            newname = f"{idx:02d}_TOPSIS_{suffix}"
            dst = Path(paper_dir) / newname
        src = Path(src)
        import shutil
        shutil.copyfile(src, dst)
        copied.append(dst.name)

    # readme
    readme = [
        "This folder contains the key TOPSIS results for the paper (numbered):",
        "01: Ranking (by CC)",
        "02: Closeness coefficients (CC)",
        "03: Distances to ideal (+) and anti-ideal (-)",
        "",
        "Files included:",
        *[f" - {x}" for x in copied]
    ]
    with open(os.path.join(paper_dir, "README_for_paper.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))

def main():
    ensure_outdir(OUTDIR)
    # 1) Read
    perf, crit, wts = read_inputs(INPUT_XLSX)
    save_df(perf, "PerformanceMatrix_Read.csv", OUTDIR)
    save_df(crit, "CriteriaInfo_Read.csv", OUTDIR)
    save_df(wts,  "GlobalWeights_Read.csv", OUTDIR)

    # 2) Validate/align
    perf_num, crit_aligned, wts_num, wts_norm, perf_criteria = validate_align(perf, crit, wts)
    save_df(wts_norm, "GlobalWeights_Normalized.csv", OUTDIR)

    # 3) Normalize + reverse costs; 4) Weighted normalized
    norm, wnorm, summary = benefit_orient_and_weight(perf_num, crit_aligned, wts_norm, perf_criteria)
    save_df(summary, "Normalization_Summary.csv", OUTDIR)
    save_df(norm, "NormalizedMatrix_BenefitOriented.csv", OUTDIR)
    save_df(wnorm, "WeightedNormalizedMatrix.csv", OUTDIR)

    # 5) Ideal / Anti-ideal
    df_ideal = ideal_antiideal(wnorm, perf_criteria)
    save_df(df_ideal, "Ideal_AntiIdeal.csv", OUTDIR)

    # 6) Distances and CC
    df_dist, df_cc, df_rank = distances_and_cc(wnorm, perf_criteria, df_ideal)
    save_df(df_dist, "Distances.csv", OUTDIR)
    save_df(df_cc, "Closeness_Coefficients.csv", OUTDIR)
    save_df(df_rank, "TOPSIS_Ranking.csv", OUTDIR)

    # 7) Mapping README and for_paper subset
    write_readme_mapping(OUTDIR)
    make_for_paper_subset(OUTDIR)

    # Console summary
    print("\nTOPSIS complete. See:", OUTDIR)
    print(df_rank.to_string(index=False))

if __name__ == "__main__":
    main()
