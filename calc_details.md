# Methodology and Calculation Details

This file provides the full methodological path used in this project — from Fuzzy AHP weighting to scenario ranking with TOPSIS and PROMETHEE II, and the final comparison.  

---

## 1. Fuzzy AHP (Analytic Hierarchy Process)

### Purpose
Derive **global weights** for sustainability criteria (Environmental, Economic, Social).  

### Process
1. **Expert judgments** → linguistic terms mapped to TFNs.  
2. **Aggregation (geometric mean):**  
   \[
   \tilde{a}_{ij}^{\,\text{agg}}=\left(\prod_{k=1}^{K} l^{(k)}\right)^{1/K}, \;\left(\prod_{k=1}^{K} m^{(k)}\right)^{1/K}, \;\left(\prod_{k=1}^{K} u^{(k)}\right)^{1/K}
   \]
3. **Synthetic extent (Chang’s method):**  
   \[
   \tilde{S}_i = \tilde{r}_i \otimes \tilde{T}^{-1}
   \]
4. **Degree of possibility (piecewise):**  
   \[
   V(\tilde{A}\ge \tilde{B}) = 
   \begin{cases}
   1, & m_A \ge m_B\\
   0, & u_A \le l_B\\
   \text{overlap fraction}, & \text{otherwise}
   \end{cases}
   \]
5. **P-vector:**  
   \[
   P_i = \min_{j\ne i} V(\tilde{S}_i \ge \tilde{S}_j)
   \]
6. **Weights:**  
   \[
   w_i = \frac{P_i}{\sum P_i}
   \]
7. **Global weights:** Pillar weight × within-pillar weight.

---

## 2. TOPSIS

### Logic
Rank scenarios by closeness to the ideal and distance from the anti-ideal.

### Steps
1. Normalize values (min–max to [0,1]).  
2. Invert cost criteria (1 – z).  
3. Weighted matrix:  
   \[
   y_{ic} = w_c \cdot z^{*}_{ic}
   \]
4. Ideal and anti-ideal:  
   \[
   v_c^+=\max y_{ic}, \quad v_c^-=\min y_{ic}
   \]
5. Distances:  
   \[
   D_i^+ = \sqrt{\sum (y_{ic}-v_c^+)^2}, \quad D_i^- = \sqrt{\sum (y_{ic}-v_c^-)^2}
   \]
6. Closeness coefficient:  
   \[
   CC_i = \frac{D_i^-}{D_i^+ + D_i^-}
   \]

---

## 3. PROMETHEE II

### Logic
Rank scenarios using pairwise dominance flows.

### Steps
1. **Preference functions:**  
   - **V-shape (continuous):** scaled by stdev.  
   - **Usual (binary):** for Jobs, Acceptance.  
2. **Aggregate preference index:**  
   \[
   \pi(a,b) = \sum_c w_c \, P_c(a,b)
   \]
3. **Flows:**  
   \[
   \phi^+(a)=\frac{1}{n-1}\sum_{b\ne a}\pi(a,b), \quad
   \phi^-(a)=\frac{1}{n-1}\sum_{b\ne a}\pi(b,a)
   \]
4. **Net flow:**  
   \[
   \phi(a) = \phi^+(a) - \phi^-(a)
   \]
5. Rank by descending \(\phi(a)\).

---

## 4. Comparison

- **TOPSIS outputs:** Closeness coefficients (CC) + rank.  
- **PROMETHEE II outputs:** Net flows (φ) + rank.  
- **Comparison table:** merges both → Scenario, CC, TOPSIS Rank, NetFlow, PROMETHEE II Rank.  
- Agreements at extremes (best/worst) = robust results.  
- Differences in mid-ranks reflect method logic (TOPSIS = closeness, PROMETHEE = dominance).

---
