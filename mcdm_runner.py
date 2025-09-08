
"""
mcdm_runner.py — Recompute Fuzzy AHP + TOPSIS + PROMETHEE II
Usage:
    python mcdm_runner.py /path/to/MCDM_Waste_Management_Framework.xlsx

It reads the filled workbook, computes:
- Global criterion weights via Fuzzy AHP (Buckley geometric mean + centroid defuzzification)
- TOPSIS ranking
- PROMETHEE II ranking (V-shape with p=STD for continuous unless overridden; Usual for ordinal)
Outputs results into a "Results" sheet in the same workbook.
"""

import sys
import math
import numpy as np
import pandas as pd

def build_fuzzy_matrix(items, pairs_df):
    n = len(items)
    # Initialize with (1,1,1) on diagonal and None elsewhere
    L = np.ones((n,n), dtype=float)
    M = np.ones((n,n), dtype=float)
    U = np.ones((n,n), dtype=float)

    # Map item to index
    idx = {it:i for i,it in enumerate(items)}

    # Fill from upper-triangle list
    for _, r in pairs_df.dropna(subset=["i","j"]).iterrows():
        i, j = r["i"], r["j"]
        if i not in idx or j not in idx:
            continue
        a = idx[i]; b = idx[j]
        try:
            l = float(r["l"]); m = float(r["m"]); u = float(r["u"])
        except:
            # skip if any missing
            continue
        # Upper entry
        L[a,b], M[a,b], U[a,b] = l, m, u
        # Reciprocal for lower entry
        L[b,a], M[b,a], U[b,a] = 1.0/u, 1.0/m, 1.0/l

    return L, M, U

def fuzzy_geometric_mean(L, M, U):
    """Buckley method: geometric mean of each row for l, m, u"""
    n = L.shape[0]
    gl = np.prod(L, axis=1)**(1.0/n)
    gm = np.prod(M, axis=1)**(1.0/n)
    gu = np.prod(U, axis=1)**(1.0/n)
    return gl, gm, gu

def normalize_fuzzy_weights(gl, gm, gu):
    sum_l = np.sum(gl)
    sum_m = np.sum(gm)
    sum_u = np.sum(gu)
    wl = gl / sum_u  # conservative
    wm = gm / sum_m
    wu = gu / sum_l  # optimistic
    return wl, wm, wu

def centroid_defuzzify(wl, wm, wu):
    return (wl + wm + wu) / 3.0

def fuzzy_ahp_weights(items, pairs_df):
    L, M, U = build_fuzzy_matrix(items, pairs_df)
    gl, gm, gu = fuzzy_geometric_mean(L, M, U)
    wl, wm, wu = normalize_fuzzy_weights(gl, gm, gu)
    w_crisp = centroid_defuzzify(wl, wm, wu)
    # Normalize crisp weights to sum 1
    w_crisp = w_crisp / np.sum(w_crisp)
    return pd.Series(w_crisp, index=items)

def minmax_normalize(values, kind):
    x = np.array(values, dtype=float)
    if len(x) == 0:
        return x
    xmin, xmax = np.min(x), np.max(x)
    if math.isclose(xmin, xmax):
        z = np.ones_like(x)  # no discrimination
    else:
        z = (x - xmin) / (xmax - xmin)
    if kind.lower() == "cost":
        z = 1.0 - z
    return z

def topsis(weighted_norm_df):
    # ideal best and worst
    ideal = weighted_norm_df.max(axis=0)
    anti = weighted_norm_df.min(axis=0)
    # distances
    d_plus = np.sqrt(((weighted_norm_df - ideal)**2).sum(axis=1))
    d_minus = np.sqrt(((weighted_norm_df - anti)**2).sum(axis=1))
    cc = d_minus / (d_plus + d_minus + 1e-12)
    return cc

def vshape_pref(d, p):
    if d <= 0:
        return 0.0
    if p <= 1e-12:
        return 1.0 if d > 0 else 0.0
    return min(d/p, 1.0)

def usual_pref(d):
    return 1.0 if d > 0 else 0.0

def promethee(normalized_df, weights, pref_setup):
    # pref_setup: dict crit -> (func, p_value or 'STD')
    alts = normalized_df.index.tolist()
    crits = normalized_df.columns.tolist()
    n = len(alts)

    # Compute pairwise aggregated preference pi(a,b)
    pi = np.zeros((n,n), dtype=float)

    for k, crit in enumerate(crits):
        w = weights[crit]
        vals = normalized_df[crit].values.astype(float)
        # Threshold p
        func, praw = pref_setup[crit]
        if isinstance(praw, str) and praw.upper() == "STD":
            p = float(np.std(vals, ddof=0))
        else:
            try:
                p = float(praw)
            except:
                p = 0.0

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = vals[i] - vals[j]  # benefit orientation assumed
                if func == "V-shape":
                    P = vshape_pref(d, p)
                else:
                    P = usual_pref(d)
                pi[i,j] += w * P

    # Flows
    leaving = pi.sum(axis=1) / (n - 1)
    entering = pi.sum(axis=0) / (n - 1)
    net = leaving - entering
    return leaving, entering, net

def main(xlsx_path):
    xl = pd.ExcelFile(xlsx_path)
    criteria = pd.read_excel(xl, "Criteria")
    scenarios = pd.read_excel(xl, "Scenarios")

    # Read pairwise sheets
    pair_pillars = pd.read_excel(xl, "Pairwise_Pillars_Fuzzy")
    pair_env = pd.read_excel(xl, "Pairwise_Env_Fuzzy")
    pair_econ = pd.read_excel(xl, "Pairwise_Econ_Fuzzy")
    pair_soc = pd.read_excel(xl, "Pairwise_Social_Fuzzy")

    # Build items per level
    pillars = ["Environmental", "Economic", "Social"]
    env_items = ["E_HH_DALYs", "E_NR_USD", "E_EQ_species_yr"]
    econ_items = ["EC_CAPEX", "EC_OPEX", "EC_Revenue"]
    soc_items = ["S_Jobs", "S_SocialAcceptance"]

    # Fuzzy AHP pillar weights
    w_pillars = fuzzy_ahp_weights(pillars, pair_pillars)

    # Fuzzy AHP weights within each pillar
    w_env = fuzzy_ahp_weights(env_items, pair_env)
    w_econ = fuzzy_ahp_weights(econ_items, pair_econ)
    w_soc = fuzzy_ahp_weights(soc_items, pair_soc)

    # Global weights
    global_w = {}
    for c in env_items:
        global_w[c] = float(w_pillars["Environmental"] * w_env[c])
    for c in econ_items:
        global_w[c] = float(w_pillars["Economic"] * w_econ[c])
    for c in soc_items:
        global_w[c] = float(w_pillars["Social"] * w_soc[c])

    # Normalize global weights to sum 1 (safety)
    total = sum(global_w.values())
    if total <= 0:
        raise ValueError("Global weights sum to zero; please fill pairwise fuzzy matrices.")
    for k in list(global_w.keys()):
        global_w[k] = global_w[k] / total

    # Extract scenario data matrix
    crit_ids = criteria["criterion_id"].tolist()
    mat = scenarios.set_index("scenario_id")[crit_ids].copy()
    # Coerce to float where possible
    for c in crit_ids:
        mat[c] = pd.to_numeric(mat[c], errors="coerce")

    # Min-max normalize per criterion (cost reversed)
    norm = pd.DataFrame(index=mat.index)
    for _, row in criteria.iterrows():
        cid = row["criterion_id"]
        ctype = str(row["type"]).lower()
        norm[cid] = minmax_normalize(mat[cid].values, ctype)

    # Weighted normalized for TOPSIS
    wvec = pd.Series(global_w)
    weighted_norm = norm[wvec.index] * wvec.values

    # TOPSIS
    cc = topsis(weighted_norm)
    topsis_df = pd.DataFrame({
        "scenario_id": weighted_norm.index,
        "topsis_cc": cc
    }).sort_values("topsis_cc", ascending=False)
    topsis_df["topsis_rank"] = range(1, len(topsis_df) + 1)

    # PROMETHEE setup
    pref_setup = {}
    for _, row in criteria.iterrows():
        cid = row["criterion_id"]
        func = str(row["promethee_pref_function"])
        pth = row["promethee_threshold"]
        pref_setup[cid] = (func, pth if pd.notna(pth) and pth != "" else "STD")

    # PROMETHEE uses normalized (benefit-oriented) values; aggregate with global weights
    leaving, entering, net = promethee(norm[wvec.index], wvec, pref_setup)
    promethee_df = pd.DataFrame({
        "scenario_id": norm.index,
        "phi_plus_leaving": leaving,
        "phi_minus_entering": entering,
        "phi_net": net
    }).sort_values("phi_net", ascending=False)
    promethee_df["promethee_rank"] = range(1, len(promethee_df) + 1)

    # Global weights report
    gw_df = pd.DataFrame({
        "criterion_id": list(wvec.index),
        "global_weight": [global_w[c] for c in wvec.index]
    }).sort_values("global_weight", ascending=False)

    # Pillar and within-pillar weights report
    pillars_df = pd.DataFrame({"pillar": w_pillars.index, "weight": w_pillars.values}).sort_values("weight", ascending=False)
    env_df = pd.DataFrame({"criterion": w_env.index, "within_env_weight": w_env.values}).sort_values("within_env_weight", ascending=False)
    econ_df = pd.DataFrame({"criterion": w_econ.index, "within_econ_weight": w_econ.values}).sort_values("within_econ_weight", ascending=False)
    soc_df = pd.DataFrame({"criterion": w_soc.index, "within_soc_weight": w_soc.values}).sort_values("within_soc_weight", ascending=False)

    # Write to workbook
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        gw_df.to_excel(writer, index=False, sheet_name="Results_GlobalWeights")
        topsis_df.to_excel(writer, index=False, sheet_name="Results_TOPSIS")
        promethee_df.to_excel(writer, index=False, sheet_name="Results_PROMETHEE")
        pillars_df.to_excel(writer, index=False, sheet_name="Results_PillarWeights")
        env_df.to_excel(writer, index=False, sheet_name="Results_EnvWeights")
        econ_df.to_excel(writer, index=False, sheet_name="Results_EconWeights")
        soc_df.to_excel(writer, index=False, sheet_name="Results_SocWeights")

    print("Done. Results written to sheets: Results_*")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mcdm_runner.py /path/to/MCDM_Waste_Management_Framework.xlsx")
        sys.exit(1)
    main(sys.argv[1])
