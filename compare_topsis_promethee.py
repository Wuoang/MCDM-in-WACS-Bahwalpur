#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script — consolidates TOPSIS and PROMETHEE II results into one table.
- Reads TOPSIS and PROMETHEE II output CSVs from their respective output folders.
- Merges CC, ranks, and net flows into one comparison table.
- Saves outputs to comparison_outputs/ with numbered filenames.
- Also creates comparison_outputs/for_paper with only the final comparison table.

Inputs are configured at the top (no CLI args).

Outputs:
  comparison_outputs/
    01_TOPSIS_Raw.csv
    02_PROMETHEEII_Raw.csv
    03_ComparisonTable.csv
    04_README_mapping.txt
  comparison_outputs/for_paper/
    01_ComparisonTable.csv
    README_for_paper.txt
"""

# ============================== CONFIG ==============================
TOPSIS_RANKING_CSV     = "topsis_outputs/11_TOPSIS_Ranking.csv"
PROMETHEE_RANKING_CSV  = "promethee2_outputs/08_PROMETHEEII_Ranking.csv"

OUTDIR = "comparison_outputs"

# ============================== CODE ==============================
import os, shutil
import pandas as pd
from pathlib import Path

FILE_COUNTER = 1
def numbered_filename(base_name: str) -> str:
    global FILE_COUNTER
    fname = f"{FILE_COUNTER:02d}_{base_name}"
    FILE_COUNTER += 1
    return fname

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def save_df(df: pd.DataFrame, name: str, outdir: str):
    df.to_csv(os.path.join(outdir, numbered_filename(name)), index=False)

def main():
    ensure_outdir(OUTDIR)
    # Read raw data
    df_topsis = pd.read_csv(TOPSIS_RANKING_CSV)
    df_prom = pd.read_csv(PROMETHEE_RANKING_CSV)

    save_df(df_topsis, "TOPSIS_Raw.csv", OUTDIR)
    save_df(df_prom, "PROMETHEEII_Raw.csv", OUTDIR)

    # Extract relevant cols
    df_topsis_sub = df_topsis[["Scenario","CC","Rank"]].rename(
        columns={"CC":"TOPSIS_CC","Rank":"TOPSIS_Rank"})
    df_prom_sub = df_prom[["Scenario","NetFlow","Rank"]].rename(
        columns={"NetFlow":"PROMETHEEII_NetFlow","Rank":"PROMETHEEII_Rank"})

    # Merge
    df_comp = df_topsis_sub.merge(df_prom_sub, on="Scenario", how="outer")

    # Save merged comparison
    save_df(df_comp, "ComparisonTable.csv", OUTDIR)

    # README
    readme = [
        "Comparison Outputs:",
        "01_TOPSIS_Raw.csv          : Raw ranking table from TOPSIS script",
        "02_PROMETHEEII_Raw.csv     : Raw ranking table from PROMETHEE II script",
        "03_ComparisonTable.csv     : Consolidated table with Scenario, CC, TOPSIS Rank, NetFlow, PROMETHEEII Rank",
        "04_README_mapping.txt      : This file",
        "for_paper/ subset          : Only the comparison table (01_ComparisonTable.csv)",
    ]
    with open(os.path.join(OUTDIR, numbered_filename("README_mapping.txt")), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))

    # For_paper subset
    paper_dir = os.path.join(OUTDIR, "for_paper")
    os.makedirs(paper_dir, exist_ok=True)
    for old in Path(paper_dir).glob("??_*.csv"):
        try: old.unlink()
        except Exception: pass
    dst = Path(paper_dir)/"01_ComparisonTable.csv"
    shutil.copyfile(os.path.join(OUTDIR,"03_ComparisonTable.csv"), dst)
    with open(os.path.join(paper_dir,"README_for_paper.txt"),"w",encoding="utf-8") as f:
        f.write("for_paper contains only the consolidated comparison table:\n01_ComparisonTable.csv\n")

    print("\nComparison complete. See:", OUTDIR)
    print(df_comp.to_string(index=False))

if __name__ == "__main__":
    main()
