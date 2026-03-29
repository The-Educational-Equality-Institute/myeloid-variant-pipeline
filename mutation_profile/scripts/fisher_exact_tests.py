#!/usr/bin/env python3
"""
Fisher's exact tests for mutual exclusivity of IDH2-SETBP1 and IDH2-PTPN11.

Tests pairwise co-occurrence patterns across multiple cohorts using 2x2 contingency
tables and scipy.stats.fisher_exact. Computes one-sided (depletion/mutual exclusivity)
and two-sided p-values, odds ratios, and expected vs observed counts.

Data sources:
  - IPSS-M (Bernard et al., NEJM Evidence 2022): N=2,957
  - MDS MSK 2020 (Papaemmanuil et al.): N=4,231
  - Beat AML (OHSU, Tyner et al.): N=785 unique patients
  - 8-study aggregate: N=11,789

Author: Automated analysis
Date: 2026-03-14
"""

import math
import json
from datetime import datetime
from scipy.stats import fisher_exact
import numpy as np


def make_2x2(n_total, n_a, n_b, n_ab):
    """
    Construct a 2x2 contingency table from marginals and overlap.

    Returns:
        table: [[a+b+, a+b-], [a-b+, a-b-]]
    """
    a_plus_b_plus = n_ab
    a_plus_b_minus = n_a - n_ab
    a_minus_b_plus = n_b - n_ab
    a_minus_b_minus = n_total - n_a - n_b + n_ab

    table = [[a_plus_b_plus, a_plus_b_minus],
             [a_minus_b_plus, a_minus_b_minus]]

    # Sanity checks
    assert a_plus_b_plus >= 0, f"Negative cell: a+b+ = {a_plus_b_plus}"
    assert a_plus_b_minus >= 0, f"Negative cell: a+b- = {a_plus_b_minus}"
    assert a_minus_b_plus >= 0, f"Negative cell: a-b+ = {a_minus_b_plus}"
    assert a_minus_b_minus >= 0, f"Negative cell: a-b- = {a_minus_b_minus}"
    assert sum(sum(row) for row in table) == n_total, "Table doesn't sum to N"

    return table


def run_fisher(table, pair_name=""):
    """
    Run Fisher's exact test on a 2x2 table.

    Returns dict with odds ratio and p-values for both alternatives.
    """
    table_np = np.array(table)

    # Two-sided
    odds_ratio_2s, p_two_sided = fisher_exact(table_np, alternative='two-sided')

    # One-sided: less = tests for depletion (mutual exclusivity)
    odds_ratio_less, p_less = fisher_exact(table_np, alternative='less')

    # One-sided: greater = tests for enrichment (co-occurrence)
    odds_ratio_greater, p_greater = fisher_exact(table_np, alternative='greater')

    return {
        'pair': pair_name,
        'table': table,
        'odds_ratio': odds_ratio_2s,
        'p_two_sided': p_two_sided,
        'p_less': p_less,       # mutual exclusivity (depletion)
        'p_greater': p_greater,  # co-occurrence (enrichment)
    }


def expected_count(n_total, n_a, n_b):
    """Expected co-occurrence count under independence."""
    return n_total * (n_a / n_total) * (n_b / n_total)


def format_table_md(table, gene_a, gene_b):
    """Format 2x2 table as markdown."""
    lines = []
    lines.append(f"|  | {gene_b}+ | {gene_b}- | **Row Total** |")
    lines.append("|---|---:|---:|---:|")
    row_a_plus = table[0][0] + table[0][1]
    row_a_minus = table[1][0] + table[1][1]
    lines.append(f"| **{gene_a}+** | {table[0][0]} | {table[0][1]} | {row_a_plus} |")
    lines.append(f"| **{gene_a}-** | {table[1][0]} | {table[1][1]} | {row_a_minus} |")
    col_b_plus = table[0][0] + table[1][0]
    col_b_minus = table[0][1] + table[1][1]
    total = sum(sum(row) for row in table)
    lines.append(f"| **Col Total** | {col_b_plus} | {col_b_minus} | **{total}** |")
    return "\n".join(lines)


def format_pval(p):
    """Format p-value with appropriate precision."""
    if p < 1e-10:
        return f"{p:.2e}"
    elif p < 0.001:
        return f"{p:.2e}"
    elif p < 0.01:
        return f"{p:.4f}"
    elif p < 0.1:
        return f"{p:.4f}"
    else:
        return f"{p:.4f}"


def format_or(odds_ratio):
    """Format odds ratio."""
    if odds_ratio == 0:
        return "0.00 (undefined, cell=0)"
    elif odds_ratio == float('inf'):
        return "inf"
    else:
        return f"{odds_ratio:.4f}"


# =============================================================================
# SECTION 1: IPSS-M (N=2,957)
# =============================================================================

print("=" * 70)
print("SECTION 1: IPSS-M Cohort (N=2,957)")
print("=" * 70)

N_IPSSM = 2957

# Gene counts
ipssm_genes = {
    'DNMT3A': 476,
    'IDH2': 127,
    'PTPN11': 52,
    'SETBP1': 98,
}

# Pairwise counts
ipssm_pairs = {
    ('DNMT3A', 'IDH2'): 24,
    ('DNMT3A', 'PTPN11'): 11,
    ('DNMT3A', 'SETBP1'): 14,
    ('IDH2', 'PTPN11'): 0,
    ('IDH2', 'SETBP1'): 1,
    ('PTPN11', 'SETBP1'): 8,
}

# Run all pairwise Fisher tests for IPSS-M
ipssm_results = {}
genes = ['DNMT3A', 'IDH2', 'PTPN11', 'SETBP1']

for i in range(len(genes)):
    for j in range(i + 1, len(genes)):
        ga, gb = genes[i], genes[j]
        n_a = ipssm_genes[ga]
        n_b = ipssm_genes[gb]
        n_ab = ipssm_pairs[(ga, gb)]

        table = make_2x2(N_IPSSM, n_a, n_b, n_ab)
        result = run_fisher(table, f"{ga}+{gb}")

        exp = expected_count(N_IPSSM, n_a, n_b)
        result['expected'] = exp
        result['observed'] = n_ab
        result['ratio'] = n_ab / exp if exp > 0 else float('inf')
        result['n_a'] = n_a
        result['n_b'] = n_b
        result['gene_a'] = ga
        result['gene_b'] = gb

        ipssm_results[(ga, gb)] = result

        print(f"\n{ga} vs {gb}:")
        print(f"  Table: {table}")
        print(f"  Expected: {exp:.2f}, Observed: {n_ab}, Ratio: {result['ratio']:.3f}")
        print(f"  OR = {format_or(result['odds_ratio'])}")
        print(f"  p (two-sided) = {format_pval(result['p_two_sided'])}")
        print(f"  p (less/depletion) = {format_pval(result['p_less'])}")
        print(f"  p (greater/enrichment) = {format_pval(result['p_greater'])}")


# =============================================================================
# SECTION 2: MDS MSK 2020 (N=4,231)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: MDS MSK 2020 (N=4,231)")
print("=" * 70)

N_MSK = 4231
msk_idh2 = 359
msk_setbp1 = 51
msk_idh2_setbp1 = 2

table_msk = make_2x2(N_MSK, msk_idh2, msk_setbp1, msk_idh2_setbp1)
result_msk = run_fisher(table_msk, "IDH2+SETBP1 (MDS MSK)")
exp_msk = expected_count(N_MSK, msk_idh2, msk_setbp1)

print(f"\nIDH2 vs SETBP1 (MDS MSK 2020):")
print(f"  Table: {table_msk}")
print(f"  Expected: {exp_msk:.2f}, Observed: {msk_idh2_setbp1}, Ratio: {msk_idh2_setbp1/exp_msk:.3f}")
print(f"  OR = {format_or(result_msk['odds_ratio'])}")
print(f"  p (two-sided) = {format_pval(result_msk['p_two_sided'])}")
print(f"  p (less/depletion) = {format_pval(result_msk['p_less'])}")


# =============================================================================
# SECTION 3: Beat AML (N=785)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Beat AML (N=785)")
print("=" * 70)

N_BEAT = 785
beat_idh2 = 100
beat_setbp1 = 10
beat_idh2_setbp1 = 2

table_beat = make_2x2(N_BEAT, beat_idh2, beat_setbp1, beat_idh2_setbp1)
result_beat = run_fisher(table_beat, "IDH2+SETBP1 (Beat AML)")
exp_beat = expected_count(N_BEAT, beat_idh2, beat_setbp1)

print(f"\nIDH2 vs SETBP1 (Beat AML):")
print(f"  Table: {table_beat}")
print(f"  Expected: {exp_beat:.2f}, Observed: {beat_idh2_setbp1}, Ratio: {beat_idh2_setbp1/exp_beat:.3f}")
print(f"  OR = {format_or(result_beat['odds_ratio'])}")
print(f"  p (two-sided) = {format_pval(result_beat['p_two_sided'])}")
print(f"  p (less/depletion) = {format_pval(result_beat['p_less'])}")


# =============================================================================
# SECTION 4: Combined/Meta analysis across cohorts for IDH2-SETBP1
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Cohort Comparison for IDH2-SETBP1")
print("=" * 70)

cohorts_idh2_setbp1 = [
    ("IPSS-M", N_IPSSM, 127, 98, 1),
    ("MDS MSK 2020", N_MSK, 359, 51, 2),
    ("Beat AML", N_BEAT, 100, 10, 2),
]

for name, n, na, nb, nab in cohorts_idh2_setbp1:
    t = make_2x2(n, na, nb, nab)
    r = run_fisher(t, f"IDH2+SETBP1 ({name})")
    e = expected_count(n, na, nb)
    print(f"\n{name}: Obs={nab}, Exp={e:.2f}, Ratio={nab/e:.3f}, "
          f"OR={format_or(r['odds_ratio'])}, "
          f"p(less)={format_pval(r['p_less'])}, p(2s)={format_pval(r['p_two_sided'])}")


# =============================================================================
# SECTION 5: Probability of zero quadruple-carriers under independence
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Quadruple Carrier Probability")
print("=" * 70)

# Aggregate frequencies from 8-study aggregate (N=11,789)
N_AGG = 11789

# From FINAL report: aggregate frequencies
# DNMT3A: 15.5%, IDH2: 6.2%, PTPN11: 3.5%, SETBP1: 2.4%
p_dnmt3a = 0.155
p_idh2 = 0.062
p_ptpn11 = 0.035
p_setbp1 = 0.024

p_all4_indep = p_dnmt3a * p_idh2 * p_ptpn11 * p_setbp1
lambda_indep = N_AGG * p_all4_indep
p_zero_indep = math.exp(-lambda_indep)

print(f"\nUnder full independence:")
print(f"  P(DNMT3A) = {p_dnmt3a}")
print(f"  P(IDH2) = {p_idh2}")
print(f"  P(PTPN11) = {p_ptpn11}")
print(f"  P(SETBP1) = {p_setbp1}")
print(f"  P(all 4) = {p_all4_indep:.6e}")
print(f"  Lambda = N * P(all 4) = {N_AGG} * {p_all4_indep:.6e} = {lambda_indep:.4f}")
print(f"  P(0 | Poisson) = e^(-{lambda_indep:.4f}) = {p_zero_indep:.4f}")

# With IDH2-SETBP1 mutual exclusivity correction (0.46x from aggregate)
me_correction = 0.46
p_all4_corrected = p_all4_indep * me_correction
lambda_corrected = N_AGG * p_all4_corrected
p_zero_corrected = math.exp(-lambda_corrected)

print(f"\nWith IDH2-SETBP1 mutual exclusivity correction (0.46x):")
print(f"  P(all 4, corrected) = {p_all4_corrected:.6e}")
print(f"  Lambda(corrected) = {lambda_corrected:.4f}")
print(f"  P(0 | Poisson) = e^(-{lambda_corrected:.4f}) = {p_zero_corrected:.4f}")

# Using IPSS-M Fisher p-value for more precise correction
ipssm_idh2_setbp1_or = ipssm_results[('IDH2', 'SETBP1')]['odds_ratio']
ipssm_ratio = ipssm_results[('IDH2', 'SETBP1')]['ratio']

print(f"\nUsing IPSS-M-derived OR for IDH2-SETBP1 ({ipssm_idh2_setbp1_or:.4f}):")
p_all4_ipssm_corrected = p_all4_indep * ipssm_ratio
lambda_ipssm_corrected = N_AGG * p_all4_ipssm_corrected
p_zero_ipssm_corrected = math.exp(-lambda_ipssm_corrected)
print(f"  P(all 4, IPSS-M corrected) = {p_all4_ipssm_corrected:.6e}")
print(f"  Lambda = {lambda_ipssm_corrected:.4f}")
print(f"  P(0 | Poisson) = {p_zero_ipssm_corrected:.4f}")

# Also add IDH2-PTPN11 mutual exclusivity (observed 0 in IPSS-M)
ipssm_idh2_ptpn11_ratio = ipssm_results[('IDH2', 'PTPN11')]['ratio']
print(f"\nIDH2-PTPN11 ratio in IPSS-M: {ipssm_idh2_ptpn11_ratio:.3f}")
print("  (Zero co-occurrences — ratio is 0, strongest possible exclusivity signal)")

# Combined correction: IDH2-SETBP1 AND IDH2-PTPN11 both depleted
# Under the most conservative model (using aggregate 0.46 for SETBP1):
# The quadruple requires both IDH2+SETBP1 and IDH2+PTPN11 in the same cell
# These are compounding constraints
print("\nNote: The quadruple requires BOTH IDH2+SETBP1 (depleted) AND IDH2+PTPN11 (zero in IPSS-M)")
print("These are independent constraints on the same gene (IDH2), making the quadruple")
print("even less likely than any single pairwise correction implies.")


# =============================================================================
# GENERATE MARKDOWN REPORT
# =============================================================================

report_lines = []

def add(s=""):
    report_lines.append(s)

add("# Fisher's Exact Test: Mutual Exclusivity Analysis")
add()
add(f"**Analysis date:** {datetime.now().strftime('%Y-%m-%d')}")
add(f"**Method:** Fisher's exact test (scipy.stats.fisher_exact)")
add(f"**Hypothesis:** One-sided test for depletion (mutual exclusivity); two-sided also reported")
add(f"**Significance threshold:** p < 0.05")
add()
add("---")
add()

# =============================================================================
# SECTION 1: Primary tests
# =============================================================================
add("## 1. Primary Tests: IDH2-SETBP1 and IDH2-PTPN11 in IPSS-M (N=2,957)")
add()
add("### 1a. IDH2 vs SETBP1")
add()

r = ipssm_results[('IDH2', 'SETBP1')]
add(format_table_md(r['table'], 'IDH2', 'SETBP1'))
add()
add(f"- **Observed co-occurrence:** {r['observed']}")
add(f"- **Expected under independence:** {r['expected']:.2f}")
add(f"- **Observed/Expected ratio:** {r['ratio']:.3f}")
add(f"- **Odds ratio:** {format_or(r['odds_ratio'])}")
add(f"- **p-value (one-sided, depletion):** {format_pval(r['p_less'])}")
add(f"- **p-value (two-sided):** {format_pval(r['p_two_sided'])}")
add()

if r['p_less'] < 0.05:
    add(f"**Interpretation:** Statistically significant mutual exclusivity (p = {format_pval(r['p_less'])}). "
        f"IDH2 and SETBP1 co-occur at {r['ratio']:.1%} of the rate expected under independence. "
        f"Only {r['observed']} patient(s) carry both mutations vs. {r['expected']:.1f} expected.")
else:
    add(f"**Interpretation:** Depletion observed (ratio {r['ratio']:.3f}) but does not reach significance "
        f"at the 0.05 level (p = {format_pval(r['p_less'])}).")
add()

add("### 1b. IDH2 vs PTPN11")
add()

r = ipssm_results[('IDH2', 'PTPN11')]
add(format_table_md(r['table'], 'IDH2', 'PTPN11'))
add()
add(f"- **Observed co-occurrence:** {r['observed']}")
add(f"- **Expected under independence:** {r['expected']:.2f}")
add(f"- **Observed/Expected ratio:** {r['ratio']:.3f}")
add(f"- **Odds ratio:** {format_or(r['odds_ratio'])}")
add(f"- **p-value (one-sided, depletion):** {format_pval(r['p_less'])}")
add(f"- **p-value (two-sided):** {format_pval(r['p_two_sided'])}")
add()

if r['p_less'] < 0.05:
    add(f"**Interpretation:** Statistically significant mutual exclusivity (p = {format_pval(r['p_less'])}). "
        f"Zero patients carry both IDH2 and PTPN11 in the IPSS-M cohort, vs. {r['expected']:.1f} expected. "
        f"The odds ratio of 0 reflects complete mutual exclusivity in this dataset.")
else:
    add(f"**Interpretation:** Depletion observed but does not reach significance (p = {format_pval(r['p_less'])}).")
add()

add("---")
add()

# =============================================================================
# SECTION 2: Cross-cohort IDH2-SETBP1
# =============================================================================
add("## 2. IDH2 vs SETBP1 Across Cohorts")
add()

# MDS MSK
add("### 2a. MDS MSK 2020 (N=4,231)")
add()
table_msk_fmt = make_2x2(N_MSK, msk_idh2, msk_setbp1, msk_idh2_setbp1)
r_msk = run_fisher(table_msk_fmt, "IDH2+SETBP1 (MDS MSK)")
exp_msk_val = expected_count(N_MSK, msk_idh2, msk_setbp1)

add(format_table_md(table_msk_fmt, 'IDH2', 'SETBP1'))
add()
add(f"- **Observed:** {msk_idh2_setbp1}, **Expected:** {exp_msk_val:.2f}, **Ratio:** {msk_idh2_setbp1/exp_msk_val:.3f}")
add(f"- **Odds ratio:** {format_or(r_msk['odds_ratio'])}")
add(f"- **p-value (depletion):** {format_pval(r_msk['p_less'])}")
add(f"- **p-value (two-sided):** {format_pval(r_msk['p_two_sided'])}")
add()

# Beat AML
add("### 2b. Beat AML / OHSU (N=785)")
add()
table_beat_fmt = make_2x2(N_BEAT, beat_idh2, beat_setbp1, beat_idh2_setbp1)
r_beat = run_fisher(table_beat_fmt, "IDH2+SETBP1 (Beat AML)")
exp_beat_val = expected_count(N_BEAT, beat_idh2, beat_setbp1)

add(format_table_md(table_beat_fmt, 'IDH2', 'SETBP1'))
add()
add(f"- **Observed:** {beat_idh2_setbp1}, **Expected:** {exp_beat_val:.2f}, **Ratio:** {beat_idh2_setbp1/exp_beat_val:.3f}")
add(f"- **Odds ratio:** {format_or(r_beat['odds_ratio'])}")
add(f"- **p-value (depletion):** {format_pval(r_beat['p_less'])}")
add(f"- **p-value (two-sided):** {format_pval(r_beat['p_two_sided'])}")
add()

# Summary table
add("### 2c. Cross-Cohort Summary: IDH2 vs SETBP1")
add()
add("| Cohort | N | IDH2 | SETBP1 | Obs | Exp | Ratio | OR | p (depletion) | p (two-sided) |")
add("|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for name, n, na, nb, nab in cohorts_idh2_setbp1:
    t = make_2x2(n, na, nb, nab)
    r_c = run_fisher(t, name)
    e = expected_count(n, na, nb)
    ratio = nab / e if e > 0 else 0
    add(f"| {name} | {n:,} | {na} | {nb} | {nab} | {e:.1f} | {ratio:.3f} | {format_or(r_c['odds_ratio'])} | {format_pval(r_c['p_less'])} | {format_pval(r_c['p_two_sided'])} |")

add()
add("**Consistent pattern:** IDH2 and SETBP1 show depletion across all MDS cohorts. The IPSS-M result "
    "is the most strongly depleted (ratio 0.24). The Beat AML cohort, with only 10 SETBP1-mutated patients, "
    "has too little power to detect depletion; the 2 observed co-occurrences vs 1.3 expected is within sampling noise.")
add()
add("---")
add()

# =============================================================================
# SECTION 3: Full pairwise matrix (IPSS-M)
# =============================================================================
add("## 3. Full Pairwise Matrix: All {DNMT3A, IDH2, PTPN11, SETBP1} Combinations in IPSS-M")
add()
add("| Gene A | Gene B | N(A) | N(B) | Observed | Expected | Ratio | OR | p (depletion) | p (enrichment) | p (two-sided) | Direction |")
add("|--------|--------|-----:|-----:|--------:|--------:|------:|---:|---:|---:|---:|---:|")

for i in range(len(genes)):
    for j in range(i + 1, len(genes)):
        ga, gb = genes[i], genes[j]
        r = ipssm_results[(ga, gb)]

        if r['ratio'] < 0.8:
            direction = "DEPLETED"
        elif r['ratio'] > 1.2:
            direction = "ENRICHED"
        else:
            direction = "neutral"

        sig_marker = ""
        if r['p_less'] < 0.05 and r['ratio'] < 1:
            sig_marker = " *"
        elif r['p_greater'] < 0.05 and r['ratio'] > 1:
            sig_marker = " *"

        add(f"| {ga} | {gb} | {r['n_a']} | {r['n_b']} | "
            f"{r['observed']} | {r['expected']:.1f} | {r['ratio']:.3f} | "
            f"{format_or(r['odds_ratio'])} | {format_pval(r['p_less'])} | "
            f"{format_pval(r['p_greater'])} | {format_pval(r['p_two_sided'])} | {direction}{sig_marker} |")

add()
add("\\* Statistically significant at p < 0.05 (one-sided test in the direction of observed deviation)")
add()

# Classification summary
add("### 3a. Classification")
add()
add("| Pattern | Gene Pairs |")
add("|---------|-----------|")

depleted = []
enriched = []
neutral = []

for i in range(len(genes)):
    for j in range(i + 1, len(genes)):
        ga, gb = genes[i], genes[j]
        r = ipssm_results[(ga, gb)]
        label = f"{ga}-{gb}"
        if r['p_less'] < 0.05 and r['ratio'] < 1:
            depleted.append(f"{label} (p={format_pval(r['p_less'])})")
        elif r['p_greater'] < 0.05 and r['ratio'] > 1:
            enriched.append(f"{label} (p={format_pval(r['p_greater'])})")
        else:
            neutral.append(f"{label} (ratio={r['ratio']:.2f})")

add(f"| **Significant mutual exclusivity** | {'; '.join(depleted) if depleted else 'None'} |")
add(f"| **Significant co-occurrence** | {'; '.join(enriched) if enriched else 'None'} |")
add(f"| **Not significant** | {'; '.join(neutral) if neutral else 'None'} |")

add()
add("---")
add()

# =============================================================================
# SECTION 4: Quadruple carrier probability
# =============================================================================
add("## 4. Probability of Zero Quadruple-Carriers Under Independence")
add()
add("Using aggregate mutation frequencies from 8 non-overlapping studies (N=11,789):")
add()
add("| Gene | Frequency |")
add("|------|-----------|")
add(f"| DNMT3A | {p_dnmt3a:.1%} |")
add(f"| IDH2 | {p_idh2:.1%} |")
add(f"| PTPN11 | {p_ptpn11:.1%} |")
add(f"| SETBP1 | {p_setbp1:.1%} |")
add()
add("### 4a. Under Full Independence")
add()
add("```")
add(f"P(all 4) = P(DNMT3A) x P(IDH2) x P(PTPN11) x P(SETBP1)")
add(f"         = {p_dnmt3a} x {p_idh2} x {p_ptpn11} x {p_setbp1}")
add(f"         = {p_all4_indep:.6e}")
add(f"")
add(f"Lambda   = N x P(all 4)")
add(f"         = {N_AGG:,} x {p_all4_indep:.6e}")
add(f"         = {lambda_indep:.4f}")
add(f"")
add(f"P(k=0)   = e^(-lambda)")
add(f"         = e^(-{lambda_indep:.4f})")
add(f"         = {p_zero_indep:.4f}")
add("```")
add()
add(f"Under full independence, we expect {lambda_indep:.3f} patients with all four mutations in 11,789. "
    f"The probability of observing zero is **{p_zero_indep:.1%}**. "
    f"Even without any mutual exclusivity, the quadruple is rare enough that zero observations is the most likely outcome.")
add()

add("### 4b. With IDH2-SETBP1 Mutual Exclusivity Correction")
add()
add("Applying the observed aggregate depletion factor of 0.46 for IDH2-SETBP1:")
add()
add("```")
add(f"P(all 4, corrected) = P(all 4) x 0.46")
add(f"                    = {p_all4_indep:.6e} x 0.46")
add(f"                    = {p_all4_corrected:.6e}")
add(f"")
add(f"Lambda(corrected)   = {N_AGG:,} x {p_all4_corrected:.6e}")
add(f"                    = {lambda_corrected:.4f}")
add(f"")
add(f"P(k=0, corrected)   = e^(-{lambda_corrected:.4f})")
add(f"                    = {p_zero_corrected:.4f}")
add("```")
add()
add(f"With the IDH2-SETBP1 exclusivity correction, the expected count drops to {lambda_corrected:.3f}, "
    f"and P(0) = **{p_zero_corrected:.1%}**.")
add()

add("### 4c. With IPSS-M-Derived Correction (Most Stringent)")
add()
add(f"Using the IPSS-M IDH2-SETBP1 observed/expected ratio of {ipssm_ratio:.3f}:")
add()
add("```")
add(f"P(all 4, IPSS-M corrected) = {p_all4_indep:.6e} x {ipssm_ratio:.3f}")
add(f"                           = {p_all4_ipssm_corrected:.6e}")
add(f"")
add(f"Lambda(IPSS-M corrected)   = {N_AGG:,} x {p_all4_ipssm_corrected:.6e}")
add(f"                           = {lambda_ipssm_corrected:.4f}")
add(f"")
add(f"P(k=0, IPSS-M corrected)   = e^(-{lambda_ipssm_corrected:.4f})")
add(f"                           = {p_zero_ipssm_corrected:.4f}")
add("```")
add()

add("### 4d. Summary of Quadruple Carrier Expectations")
add()
add("| Model | P(all 4) | Lambda (N=11,789) | P(k=0) |")
add("|-------|----------|-------------------|--------|")
add(f"| Full independence | {p_all4_indep:.2e} | {lambda_indep:.4f} | {p_zero_indep:.4f} |")
add(f"| Aggregate ME correction (0.46x) | {p_all4_corrected:.2e} | {lambda_corrected:.4f} | {p_zero_corrected:.4f} |")
add(f"| IPSS-M ME correction ({ipssm_ratio:.3f}x) | {p_all4_ipssm_corrected:.2e} | {lambda_ipssm_corrected:.4f} | {p_zero_ipssm_corrected:.4f} |")
add()
add("Under all models, observing zero quadruple-carriers in ~12,000 patients is the expected outcome "
    "(P > 90%). The zero observation is consistent with statistical expectation and does not by itself "
    "require a biological explanation beyond low individual gene frequencies. However, the complete "
    "absence of the three-gene intermediate DNMT3A+IDH2+SETBP1 (0 observed across all databases, "
    "despite ~4 expected under independence) provides stronger evidence of biological constraint.")
add()

add("---")
add()

# =============================================================================
# SECTION 5: Detailed contingency tables
# =============================================================================
add("## 5. All Contingency Tables (IPSS-M, N=2,957)")
add()
add("Provided for reproducibility and verification. Each table shows the 2x2 layout used for Fisher's exact test.")
add()

for i in range(len(genes)):
    for j in range(i + 1, len(genes)):
        ga, gb = genes[i], genes[j]
        r = ipssm_results[(ga, gb)]
        add(f"### {ga} vs {gb}")
        add()
        add(format_table_md(r['table'], ga, gb))
        add()
        add(f"Fisher's exact test: OR = {format_or(r['odds_ratio'])}, "
            f"p(depletion) = {format_pval(r['p_less'])}, "
            f"p(two-sided) = {format_pval(r['p_two_sided'])}")
        add()

add("---")
add()
add("## 6. Methodology Notes")
add()
add("1. **Fisher's exact test** was chosen over chi-squared because several cells have low expected counts (< 5).")
add("2. **One-sided test (alternative='less')** directly tests the hypothesis of depletion/mutual exclusivity, i.e., "
    "that the odds ratio is less than 1.")
add("3. **Two-sided test** is also reported for completeness and is appropriate when the direction of deviation is not hypothesized a priori.")
add("4. **Odds ratio** from Fisher's exact test is the conditional maximum likelihood estimate. An OR of 0 indicates "
    "complete mutual exclusivity in the sample (zero co-occurrences).")
add("5. **Expected counts** are computed as N x P(A) x P(B), equivalent to the expected value under the independence model.")
add("6. **Poisson approximation** (Section 4) treats the number of quadruple-carriers as Poisson(lambda), appropriate "
    "when lambda << N.")
add("7. All p-values are exact (computed from the hypergeometric distribution), not asymptotic approximations.")
add()
add("---")
add()
add("## 7. Data Sources")
add()
add("| Cohort | Reference | N |")
add("|--------|-----------|---|")
add("| IPSS-M | Bernard et al., NEJM Evidence 2022 (PMID 36160768) | 2,957 |")
add("| MDS MSK 2020 | Papaemmanuil et al., accessed via cBioPortal | 4,231 |")
add("| Beat AML | Tyner et al., accessed via Vizome/cBioPortal | 785 (unique patients) |")
add("| 8-study aggregate | Multiple sources, deduplicated | 11,789 |")
add()
add("Source data: `mutation_profile/results/ipssm_cooccurrence.json`, `FINAL_cooccurrence_report.md`")

# Write report
report_text = "\n".join(report_lines)

output_path = "Path(__file__).resolve().parents[2]/mutation_profile/results/fisher_exact_mutual_exclusivity.md"
with open(output_path, 'w') as f:
    f.write(report_text)

print(f"\n\nReport written to: {output_path}")
print(f"Report length: {len(report_text)} characters, {len(report_lines)} lines")
