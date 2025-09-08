#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuzzy AHP (TFN) per Yang et al. (2013, Nat Hazards 68:657–674)
Creates FULL step-by-step outputs and tags each output with the paper's equations.

USAGE
  python fuzzy_ahp_run_verbose.py "AHP_Public Health Expert.xlsx" \
      "AHP_LCA Expert.xlsx" "AHP_EfW Process Engineer.xlsx" \
      "AHP_Infrastructure Economist.xlsx" "AHP_Municipal Operations Manager.xlsx"

OUTPUTS (in ./fa_outputs/)
  <level>_matrix_fuzzy_long.csv                 # Aggregated TFN matrix (Table 2–6 analogs)
  <level>_matrix_fuzzy_wide_l.csv/.m.csv/.u.csv # Same matrix split by l/m/u
  <level>_row_sums_TFN.csv                      # Row-sum TFNs (pre Eq. 4)
  <level>_total_TFN.csv                         # Sum_{i,j} TFN (denominator in Eq. 4)
  <level>_Si_TFN.csv                            # Synthetic extents S_i = row_sum * total^{-1} (Eq. 4)
  <level>_V_matrix.csv                          # V(Si >= Sj) (Eq. 1 & Eq. 5)
  <level>_P_vector.csv                          # P_i = min_j V(Si >= Sj) (Eq. 5 → Eq. 6)
  <level>_weights_normalized.csv                # w (normalized P) (Eq. 6)
  results_pillars.csv                           # Final pillar weights (reported)
  results_env.csv / results_econ.csv / results_social.csv  # Within-pillar weights (reported)
  results_global.csv                             # Global weights via Eq. 8 (reported)
  README_mapping.txt                             # What maps to which equation/table

NOTES
- Expert aggregation: arithmetic mean of TFNs across experts (paper Eq. 3). Set EXPERT_TFN_AGGREGATION="geometric" to test variant.
- Pairwise sheets must be upper-triangle lists with columns: i, j, l, m, u
"""

import os, sys, math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import shutil
import glob, re
from pathlib import Path

# ------------ CONFIG ------------
EXPERT_TFN_AGGREGATION = "geometric"  # "arithmetic" (paper) | "geometric"
OUTDIR = "fa_outputs"

PILLARS = ["Environmental", "Economic", "Social"]
ENV_ITEMS = ["E_HH_DALYs", "E_NR_USD", "E_EQ_species_yr"]
ECON_ITEMS = ["EC_CAPEX", "EC_OPEX", "EC_Revenue"]
SOC_ITEMS = ["S_Jobs", "S_SocialAcceptance"]

SHEET_NAMES = {
    "pillars": "Pairwise_Pillars_Fuzzy",
    "env": "Pairwise_Env_Fuzzy",
    "econ": "Pairwise_Econ_Fuzzy",
    "soc": "Pairwise_Social_Fuzzy",
}
# Numerical tolerances
_TAU = 1e-12
# Global counter for numbering outputs
FILE_COUNTER = 1


def numbered_filename(base_name: str) -> str:
    """
    Prefix the file name with an incrementing counter like 01_, 02_, etc.
    """
    global FILE_COUNTER
    fname = f"{FILE_COUNTER:02d}_{base_name}"
    FILE_COUNTER += 1
    return fname


# ------------ TFN helpers ------------
def tfn_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def tfn_scalar_div(a, s):
    return (a[0] / s, a[1] / s, a[2] / s)


def tfn_mul(a, b):
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def tfn_inv(a):
    l, m, u = a
    return (1.0 / u, 1.0 / m, 1.0 / l)


def tfn_centroid(a):
    return (a[0] + a[1] + a[2]) / 3.0


def tfn_geom_mean(lst):
    l = np.prod([x[0] for x in lst]) ** (1 / len(lst))
    m = np.prod([x[1] for x in lst]) ** (1 / len(lst))
    u = np.prod([x[2] for x in lst]) ** (1 / len(lst))
    return (l, m, u)


def tfn_arith_mean(lst):
    l = float(np.mean([x[0] for x in lst]))
    m = float(np.mean([x[1] for x in lst]))
    u = float(np.mean([x[2] for x in lst]))
    return (l, m, u)


# Paper Eq. (1): degree of possibility V(M2 ≥ M1)


def degree_of_possibility(A, B) -> float:
    """
    V(A >= B) for TFNs A=(lA,mA,uA), B=(lB,mB,uB) per Chang (1996) style.
    Piecewise with small tolerances to avoid spurious 0/1 from roundoff.
    """
    lA, mA, uA = A
    lB, mB, uB = B

    # 1) Modal dominance
    if mA >= mB - _TAU:
        return 1.0

    # 2) Disjoint (A entirely left of B)
    if uA <= lB + _TAU:
        return 0.0

    # 3) Partial overlap
    num = lB - uA
    den = (mA - uA) - (
        mB - lB
    )  # NOTE: right spread of A minus left spread of B (mapped form)
    if abs(den) < _TAU:
        # Degenerate slopes: fall back to centroid ordering
        cA = (lA + mA + uA) / 3.0
        cB = (lB + mB + uB) / 3.0
        return 1.0 if cA >= cB else 0.0

    val = num / den
    # Clamp to [0,1]
    return 0.0 if val < 0.0 else (1.0 if val > 1.0 else float(val))


# ------------ Matrix builders ------------
def build_full_matrix(
    items: List[str], upper_df: pd.DataFrame
) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    n = len(items)
    idx = {it: i for i, it in enumerate(items)}
    mat = {(i, i): (1.0, 1.0, 1.0) for i in range(n)}

    # Track which pairs were explicitly provided
    provided = set()

    for _, r in upper_df.iterrows():
        ii, jj = str(r["i"]).strip(), str(r["j"]).strip()
        if ii not in idx or jj not in idx:
            continue
        i, j = idx[ii], idx[jj]
        # Require all three numbers
        if pd.isna(r.get("l")) or pd.isna(r.get("m")) or pd.isna(r.get("u")):
            raise ValueError(f"Missing TFN values for pair ({ii},{jj}) in sheet.")
        l, m, u = float(r["l"]), float(r["m"]), float(r["u"])
        # Basic TFN sanity
        if not (l <= m <= u):
            raise ValueError(
                f"Invalid TFN (l<=m<=u violated) for pair ({ii},{jj}): {(l,m,u)}"
            )
        mat[(i, j)] = (l, m, u)
        mat[(j, i)] = (1.0 / u, 1.0 / m, 1.0 / l)
        provided.add((min(i, j), max(i, j)))

    # Ensure *every* upper-triangle pair was provided
    missing = []
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) not in provided:
                missing.append((items[a], items[b]))
    if missing:
        raise ValueError(
            f"Upper-triangle pairs missing (don’t default to (1,1,1)): {missing}"
        )

    return mat


def aggregate_matrices(
    mats: List[Dict[Tuple[int, int], Tuple[float, float, float]]], mode="arithmetic"
):
    keys = mats[0].keys()
    agg = {}
    for k in keys:
        lst = [m[k] for m in mats]
        if mode == "geometric":
            agg[k] = tfn_geom_mean(lst)
        elif mode == "trimmed10":
            # 10% trimmed mean preserves spread better than full mean
            def tmean(vals):
                vals = np.sort(np.array(vals, dtype=float))
                t = max(1, int(0.1 * len(vals)))
                core = vals[t : len(vals) - t] if len(vals) > 2 * t else vals
                return float(core.mean())

            agg[k] = (
                tmean([x[0] for x in lst]),
                tmean([x[1] for x in lst]),
                tmean([x[2] for x in lst]),
            )
        else:
            agg[k] = tfn_arith_mean(lst)
    return agg


# ------------ Core FAHP per paper ------------
def save_matrix_long(
    level_name: str,
    items: List[str],
    mat: Dict[Tuple[int, int], Tuple[float, float, float]],
):
    rows = []
    for i, it in enumerate(items):
        for j, jt in enumerate(items):
            l, m, u = mat[(i, j)]
            rows.append({"row": it, "col": jt, "l": l, "m": m, "u": u})
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(OUTDIR, numbered_filename(f"{level_name}_matrix_fuzzy_long.csv")),
        index=False,
    )
    # split wide l/m/u
    n = len(items)
    L = pd.DataFrame(np.zeros((n, n)), index=items, columns=items)
    M = L.copy()
    U = L.copy()
    for r in rows:
        L.loc[r["row"], r["col"]] = r["l"]
        M.loc[r["row"], r["col"]] = r["m"]
        U.loc[r["row"], r["col"]] = r["u"]
    L.to_csv(
        os.path.join(OUTDIR, numbered_filename(f"{level_name}_matrix_fuzzy_wide_l.csv"))
    )
    M.to_csv(
        os.path.join(OUTDIR, numbered_filename(f"{level_name}_matrix_fuzzy_wide_m.csv"))
    )
    U.to_csv(
        os.path.join(OUTDIR, numbered_filename(f"{level_name}_matrix_fuzzy_wide_u.csv"))
    )


def fuzzy_ahp_weights_verbose(
    level_name: str,
    items: List[str],
    mat: Dict[Tuple[int, int], Tuple[float, float, float]],
) -> pd.Series:
    os.makedirs(OUTDIR, exist_ok=True)

    # --- Save aggregated fuzzy matrix (Tables 2–6 analog) ---
    save_matrix_long(level_name, items, mat)

    # --- Compute totals and S_i (Eq. 3 & Eq. 4) ---
    n = len(items)
    total = (0.0, 0.0, 0.0)
    for i in range(n):
        for j in range(n):
            total = tfn_add(total, mat[(i, j)])
    total_inv = tfn_inv(total)

    # Row sums
    row_sums = []
    for i in range(n):
        rs = (0.0, 0.0, 0.0)
        for j in range(n):
            rs = tfn_add(rs, mat[(i, j)])
        row_sums.append(rs)
    pd.DataFrame(
        [
            {
                "item": items[i],
                "l": row_sums[i][0],
                "m": row_sums[i][1],
                "u": row_sums[i][2],
            }
            for i in range(n)
        ]
    ).to_csv(
        os.path.join(OUTDIR, numbered_filename(f"{level_name}_row_sums_TFN.csv")),
        index=False,
    )
    pd.DataFrame([{"sum_l": total[0], "sum_m": total[1], "sum_u": total[2]}]).to_csv(
        os.path.join(OUTDIR, numbered_filename(f"{level_name}_total_TFN.csv")),
        index=False,
    )

    # Synthetic extents S_i
    S = []
    for i in range(n):
        S.append(tfn_mul(row_sums[i], total_inv))
    pd.DataFrame(
        [
            {"item": items[i], "S_l": S[i][0], "S_m": S[i][1], "S_u": S[i][2]}
            for i in range(n)
        ]
    ).to_csv(
        os.path.join(OUTDIR, numbered_filename(f"{level_name}_Si_TFN.csv")), index=False
    )

    # --- Degree of possibility V(Si >= Sj) (Eq. 1 & Eq. 5) ---
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            V[i, j] = 1.0 if i == j else degree_of_possibility(S[i], S[j])
    Vdf = pd.DataFrame(V, index=items, columns=items)
    Vdf.to_csv(os.path.join(OUTDIR, numbered_filename(f"{level_name}_V_matrix.csv")))

    # --- P vector & normalized weights (Eq. 5 → Eq. 6) ---
    P = np.zeros(n)
    for i in range(n):
        others = [V[i, j] for j in range(n) if j != i]
        P[i] = min(others) if others else 1.0
    pdf = pd.DataFrame({"item": items, "P": P})
    pdf.to_csv(
        os.path.join(OUTDIR, numbered_filename(f"{level_name}_P_vector.csv")),
        index=False,
    )

    if P.sum() <= 1e-12:
        w = np.ones(n) / n
    else:
        w = P / P.sum()
    wser = pd.Series(w, index=items)
    wser.to_csv(
        os.path.join(OUTDIR, numbered_filename(f"{level_name}_weights_normalized.csv")),
        header=["weight"],
    )
    return wser


def run_level(
    expert_files: List[str], sheet_name: str, items: List[str], level_name: str
) -> pd.Series:
    mats = []
    for path in expert_files:
        df = pd.read_excel(path, sheet_name=sheet_name)
        mats.append(build_full_matrix(items, df))
    agg_mat = aggregate_matrices(mats, mode=EXPERT_TFN_AGGREGATION)
    weights = fuzzy_ahp_weights_verbose(level_name, items, agg_mat)
    return weights


def write_readme_mapping():
    lines = []
    lines.append("OUTPUT → PAPER MAPPING (Yang et al., 2013)\n")
    lines.append(
        "- *_matrix_fuzzy_*              → Pairwise fuzzy judgment matrices (Tables 2–6 analogs, Section 3.2 & 3.3)"
    )
    lines.append(
        "- *_row_sums_TFN.csv            → Intermediate totals per row (used in Eq. 4)"
    )
    lines.append(
        "- *_total_TFN.csv               → Sum over all entries (denominator in Eq. 4)"
    )
    lines.append("- *_Si_TFN.csv                  → Synthetic extents S_i (Eq. 4)")
    lines.append(
        "- *_V_matrix.csv                → Degree of possibility V(Si ≥ Sj) (Eq. 1) used in Eq. 5"
    )
    lines.append(
        "- *_P_vector.csv                → P_i = min_j V(Si ≥ Sj) (Eq. 5) → single ranking (Eq. 6)"
    )
    lines.append(
        "- *_weights_normalized.csv      → Normalized weights per level (Eq. 6)"
    )
    lines.append("- results_global.csv            → Hierarchical synthesis (Eq. 8)")
    with open(os.path.join(OUTDIR, "README_mapping.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _numeric_prefix(path: Path) -> int:
    """
    Extract a leading NN_ numeric prefix from a file name, e.g. '07_results_env.csv' -> 7.
    Returns -1 if no numeric prefix is found.
    """
    m = re.match(r"^(\d+)_", path.name)
    return int(m.group(1)) if m else -1


def _find_by_suffix(outdir: str, suffix: str) -> Path | None:
    """
    Find a file in `outdir` whose name ends with `suffix`, e.g. '*results_env.csv'.
    If multiple exist, pick the one with the largest numeric prefix; if none have
    a numeric prefix, pick the most recently modified.
    """
    candidates = [Path(p) for p in glob.glob(os.path.join(outdir, f"*{suffix}"))]
    if not candidates:
        return None
    # Prefer largest numeric prefix (later step), fallback to mtime
    with_num = [p for p in candidates if _numeric_prefix(p) >= 0]
    if with_num:
        return sorted(with_num, key=_numeric_prefix, reverse=True)[0]
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def make_paper_subset():
    """
    Collect only the results typically reported in the paper,
    save them to fa_outputs/for_paper/ with numbered filenames
    to show order of the process. Works even when source files in
    fa_outputs/ are prefixed like '01_...csv'.
    """
    paper_dir = os.path.join(OUTDIR, "for_paper")
    os.makedirs(paper_dir, exist_ok=True)

    # Order matters (1 → 9)
    files_in_order = [
        "pillars_matrix_fuzzy_long.csv",
        "env_matrix_fuzzy_long.csv",
        "econ_matrix_fuzzy_long.csv",
        "soc_matrix_fuzzy_long.csv",
        "results_pillars.csv",
        "results_env.csv",
        "results_econ.csv",
        "results_social.csv",
        "results_global.csv",
    ]

    # Clean old copies in for_paper/ (keep README)
    for old in Path(paper_dir).glob("??_*.csv"):
        try:
            old.unlink()
        except Exception:
            pass

    copied = []
    for idx, suffix in enumerate(files_in_order, start=1):
        src_path = _find_by_suffix(OUTDIR, suffix)
        if src_path is None:
            # Not found — skip but note in README
            copied.append(f"(missing) {idx:02d}_{suffix}")
            continue
        dst_name = f"{idx:02d}_{suffix}"
        dst_path = Path(paper_dir) / dst_name
        shutil.copyfile(src_path, dst_path)
        copied.append(dst_name)

    # README (always rewritten)
    readme = [
        "This folder contains the key results typically presented in the paper,",
        "renamed with step numbers to indicate the order of generation.",
        "",
        "01–04 : Pairwise fuzzy judgment matrices (Tables-style)",
        "05–08 : Normalized weights at pillar and within-pillar level",
        "09    : Global weights (Eq. 8)",
        "",
        "Files included here:",
        *[f" - {x}" for x in copied],
        "",
        "Note: Sources were located by suffix matching in ../fa_outputs/*.csv.",
        "If a file is marked '(missing)', make sure the upstream step ran and wrote it.",
    ]
    with open(
        os.path.join(paper_dir, "README_for_paper.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(readme))


def main():
    # ========= INPUT FILES (same directory as this script) =========
    expert_files = [
        "inputs/AHP_Operations_Expert.xlsx",
        "inputs/AHP_Financial_Expert.xlsx",
        "inputs/AHP_PracticalFeasibility_Expert.xlsx",
        "inputs/AHP_EfW Process Engineer.xlsx",
        "inputs/AHP_LCA Expert.xlsx",
        "inputs/AHP_Municipal Operations Manager.xlsx",
        "inputs/AHP_Public Health Expert.xlsx",
        "inputs/AHP_Infrastructure Economist.xlsx",
    ]
    os.makedirs(OUTDIR, exist_ok=True)

    # Pillars (level_name: 'pillars')
    w_pillars = run_level(expert_files, SHEET_NAMES["pillars"], PILLARS, "pillars")
    w_pillars.sort_values(ascending=False).to_csv(
        os.path.join(OUTDIR, numbered_filename("results_pillars.csv")),
        header=["weight"],
    )

    # Environmental
    w_env = run_level(expert_files, SHEET_NAMES["env"], ENV_ITEMS, "env")
    w_env.sort_values(ascending=False).to_csv(
        os.path.join(OUTDIR, numbered_filename("results_env.csv")),
        header=["within_env_weight"],
    )

    # Economic
    w_econ = run_level(expert_files, SHEET_NAMES["econ"], ECON_ITEMS, "econ")
    w_econ.sort_values(ascending=False).to_csv(
        os.path.join(OUTDIR, numbered_filename("results_econ.csv")),
        header=["within_econ_weight"],
    )

    # Social
    w_soc = run_level(expert_files, SHEET_NAMES["soc"], SOC_ITEMS, "soc")
    w_soc.sort_values(ascending=False).to_csv(
        os.path.join(OUTDIR, numbered_filename("results_social.csv")),
        header=["within_soc_weight"],
    )

    # Global synthesis (Eq. 8)
    global_w = {}
    for c in ENV_ITEMS:
        global_w[c] = float(w_pillars["Environmental"] * w_env[c])
    for c in ECON_ITEMS:
        global_w[c] = float(w_pillars["Economic"] * w_econ[c])
    for c in SOC_ITEMS:
        global_w[c] = float(w_pillars["Social"] * w_soc[c])

    tot = sum(global_w.values())
    if tot <= 1e-12:
        raise ValueError("Global weights sum to zero; check inputs.")
    for k in list(global_w.keys()):
        global_w[k] = global_w[k] / tot

    df_global = pd.DataFrame(
        [{"criterion_id": k, "global_weight": v} for k, v in global_w.items()]
    ).sort_values("global_weight", ascending=False)
    df_global.to_csv(
        os.path.join(OUTDIR, numbered_filename("results_global.csv")), index=False
    )

    write_readme_mapping()
    make_paper_subset()

    # Console summary
    print(
        "\n== Aggregation mode:", EXPERT_TFN_AGGREGATION, "(paper uses 'arithmetic') =="
    )
    print("\nPillars:\n", w_pillars.sort_values(ascending=False))
    print("\nEnvironmental within:\n", w_env.sort_values(ascending=False))
    print("\nEconomic within:\n", w_econ.sort_values(ascending=False))
    print("\nSocial within:\n", w_soc.sort_values(ascending=False))
    print("\nGlobal weights (Eq. 8):\n", df_global)


if __name__ == "__main__":
    main()
