# Multi-Criteria Decision-Making (MCDM) for Waste Management

This repository contains the full pipeline for evaluating Municipal Solid Waste Management (MSWM) scenarios using a hybrid **MCDM framework** that integrates:

1. **Fuzzy AHP (Analytic Hierarchy Process)**  
   - Derives global weights for criteria using expert judgments expressed as linguistic terms and converted into triangular fuzzy numbers (TFNs).  
   - Implements Chang’s extent analysis method.  

2. **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)**  
   - Ranks scenarios based on closeness to the ideal solution and distance from the anti-ideal.  

3. **PROMETHEE II (Preference Ranking Organization Method for Enrichment Evaluation)**  
   - Ranks scenarios based on pairwise dominance relations using V-shape and Usual preference functions.  

4. **Comparison Module**  
   - Consolidates TOPSIS and PROMETHEE II outputs into a single table for cross-method interpretation.  

---

## Repository Structure

- `calc_details.md` → Detailed methodology and formulae used in this framework. 
- `fuzzy_ahp_run_verbose.py` → Computes criteria weights (pillar, within-pillar, and global) using Fuzzy AHP.  
- `topsis_run_verbose.py` → Performs TOPSIS analysis with full step-by-step outputs.  
- `promethee2_run_verbose.py` → Performs PROMETHEE II analysis with detailed outputs.  
- `compare_topsis_promethee.py` → Generates a consolidated comparison table of scenario rankings.  
- `scenario_performance_matrix - Template.xlsx` → Template for scenario data, criteria info, and weights.
- `AHP_Template.xlsx` → Template for pairwise comparison for fuzzy AHP using triangular fuzzy numbers (TFN).  
---

## Outputs

Each script generates:
- A **results folder** with numbered, descriptive CSVs for every intermediate step.  
- A `for_paper/` subfolder with only the key results formatted for direct use in publications.  
