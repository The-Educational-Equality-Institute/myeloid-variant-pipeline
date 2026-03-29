#!/usr/bin/env python3
"""
Compute formal mutual exclusivity and co-occurrence statistics for
DNMT3A, IDH2, PTPN11, SETBP1 pairwise combinations using IPSS-M dataset.

Also computes triple/quadruple probabilities and additional pairs
(IDH2 vs SETBP1, SETBP1 vs del7/del7q, PTPN11 vs SETBP1).

Output: mutation_profile/results/mutual_exclusivity_stats.json
"""

import json
import sys
from itertools import combinations
from pathlib import Path
from math import log, factorial, exp, comb

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.special import gammaln

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUT_PATH = PROJECT_ROOT / "mutation_profile" / "data" / "ipssm" / "df_mut.tsv"
CNA_PATH = PROJECT_ROOT / "mutation_profile" / "data" / "ipssm" / "df_cna.tsv"
OUT_PATH = PROJECT_ROOT / "mutation_profile" / "results" / "mutual_exclusivity_stats.json"


def load_data():
    """Load mutation and CNA binary matrices, binarize any values > 1."""
    df_mut = pd.read_csv(MUT_PATH, sep="\t", index_col=0)
    df_cna = pd.read_csv(CNA_PATH, sep="\t", index_col=0)
    # Binarize
    df_mut = df_mut.clip(upper=1)
    df_cna = df_cna.clip(upper=1)
    return df_mut, df_cna


def contingency_table(a: pd.Series, b: pd.Series):
    """Build 2x2 contingency table from two binary series."""
    both = int(((a == 1) & (b == 1)).sum())
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    neither = int(((a == 0) & (b == 0)).sum())
    return {"both": both, "only_A": only_a, "only_B": only_b, "neither": neither}


def compute_odds_ratio_ci(table: dict, n: int):
    """
    Compute odds ratio, 95% CI (Woolf logit method), and log-odds.
    Applies Haldane correction (+0.5) if any cell is zero.
    """
    a = table["both"]
    b = table["only_A"]
    c = table["only_B"]
    d = table["neither"]

    # Haldane correction if any cell is 0
    if a == 0 or b == 0 or c == 0 or d == 0:
        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_c, b_c, c_c, d_c = a, b, c, d

    odds_ratio = (a_c * d_c) / (b_c * c_c)
    log_or = log(odds_ratio)
    se_log_or = (1 / a_c + 1 / b_c + 1 / c_c + 1 / d_c) ** 0.5
    ci_lower = exp(log_or - 1.96 * se_log_or)
    ci_upper = exp(log_or + 1.96 * se_log_or)

    return odds_ratio, [round(ci_lower, 6), round(ci_upper, 6)], log_or


def expected_under_independence(a_series: pd.Series, b_series: pd.Series, n: int):
    """Expected co-occurrence count under independence."""
    p_a = a_series.sum() / n
    p_b = b_series.sum() / n
    return p_a * p_b * n


def benjamini_hochberg(pvalues: list):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    corrected = [0.0] * n
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        orig_idx, pval = indexed[i]
        adjusted = pval * n / rank
        cummin = min(cummin, adjusted)
        corrected[orig_idx] = min(cummin, 1.0)
    return corrected


def pairwise_test(a_series: pd.Series, b_series: pd.Series, gene_a: str, gene_b: str, n: int):
    """Run full pairwise analysis for two genes."""
    table = contingency_table(a_series, b_series)

    # Fisher's exact test (two-sided)
    contingency_matrix = np.array([
        [table["both"], table["only_A"]],
        [table["only_B"], table["neither"]]
    ])
    fisher_or, fisher_p = fisher_exact(contingency_matrix, alternative="two-sided")

    # Our computed OR with CI
    odds_ratio, ci, log_or = compute_odds_ratio_ci(table, n)
    expected = expected_under_independence(a_series, b_series, n)

    direction = "co-occurring" if odds_ratio > 1 else "mutually_exclusive"
    if odds_ratio == 1.0:
        direction = "independent"

    return {
        "contingency_table": table,
        "fisher_pvalue": round(fisher_p, 10),
        "fisher_odds_ratio": round(fisher_or, 6) if np.isfinite(fisher_or) else None,
        "odds_ratio": round(odds_ratio, 6),
        "odds_ratio_95ci": [round(ci[0], 6), round(ci[1], 6)],
        "log_odds": round(log_or, 6),
        "expected_under_independence": round(expected, 4),
        "observed": table["both"],
        "direction": direction,
    }


def compute_triple_probability(df: pd.DataFrame, genes: list, n: int):
    """
    Compute observed vs expected triple co-occurrence.
    Uses Poisson approximation for P(X=0) when expected is small.
    """
    mask = df[genes].sum(axis=1) == len(genes)
    observed = int(mask.sum())

    # Expected under full independence
    marginals = [df[g].sum() / n for g in genes]
    expected = n
    for p in marginals:
        expected *= p

    # P(X=k) under Poisson with lambda=expected
    if expected > 0:
        # P(X=0) = e^(-lambda)
        p_zero = exp(-expected)
        # P(X=observed) using Poisson PMF
        if observed == 0:
            p_observed = p_zero
        else:
            log_p = -expected + observed * log(expected) - gammaln(observed + 1)
            p_observed = exp(log_p)
    else:
        p_zero = 1.0
        p_observed = 1.0

    return {
        "genes": genes,
        "observed": observed,
        "expected_under_independence": round(expected, 6),
        "marginal_rates": {g: round(df[g].sum() / n, 6) for g in genes},
        "poisson_p_of_zero": round(p_zero, 10),
        "poisson_p_of_observed": round(p_observed, 10),
    }


def compute_quadruple_probability(df: pd.DataFrame, genes: list, n: int):
    """Compute observed vs expected quadruple co-occurrence."""
    mask = df[genes].sum(axis=1) == len(genes)
    observed = int(mask.sum())

    marginals = [df[g].sum() / n for g in genes]
    expected = n
    for p in marginals:
        expected *= p

    p_zero = exp(-expected) if expected > 0 else 1.0

    return {
        "genes": genes,
        "observed": observed,
        "expected_under_independence": round(expected, 6),
        "marginal_rates": {g: round(df[g].sum() / n, 6) for g in genes},
        "poisson_p_of_zero": round(p_zero, 10),
        "note": (
            f"Under independence, expected {round(expected, 4)} patients with all 4 mutations. "
            f"Observed {observed}. Poisson P(X=0) = {p_zero:.6e}."
        ),
    }


def discover_test(df: pd.DataFrame, genes: list, n: int):
    """
    Implement a simplified DISCOVER-style mutual exclusivity test.

    DISCOVER tests mutual exclusivity by comparing observed co-occurrence
    to expected under independence, using an exact test based on the
    multivariate hypergeometric distribution. Here we use a permutation-based
    approximation since scipy doesn't have DISCOVER built in.

    We also compute a simple binomial-based group exclusivity test.
    """
    # For each pair, check if co-occurrence is less than expected
    observed_cooccurrence = 0
    expected_cooccurrence = 0.0

    for g1, g2 in combinations(genes, 2):
        obs = int(((df[g1] == 1) & (df[g2] == 1)).sum())
        exp_val = (df[g1].sum() / n) * (df[g2].sum() / n) * n
        observed_cooccurrence += obs
        expected_cooccurrence += exp_val

    # Coverage: fraction of patients with at least one mutation
    any_mut = (df[genes].sum(axis=1) >= 1).sum()
    coverage = any_mut / n

    # Exclusivity: fraction of mutated patients with exactly one mutation
    exactly_one = (df[genes].sum(axis=1) == 1).sum()
    exclusivity = exactly_one / any_mut if any_mut > 0 else 0

    # Group mutual exclusivity: compare total pairwise co-occurrences to expected
    # Under independence, total co-occurrences ~ Poisson(expected_cooccurrence)
    if expected_cooccurrence > 0:
        # P(X <= observed) for Poisson - left tail for mutual exclusivity
        from scipy.stats import poisson
        p_me = poisson.cdf(observed_cooccurrence, expected_cooccurrence)
    else:
        p_me = 1.0

    return {
        "method": "simplified_DISCOVER_approximation",
        "genes": genes,
        "total_pairwise_cooccurrences_observed": observed_cooccurrence,
        "total_pairwise_cooccurrences_expected": round(expected_cooccurrence, 4),
        "coverage": round(coverage, 6),
        "exclusivity_among_mutated": round(exclusivity, 6),
        "n_patients_with_any": int(any_mut),
        "n_patients_exactly_one": int(exactly_one),
        "poisson_me_pvalue": round(p_me, 10),
        "direction": "mutually_exclusive" if observed_cooccurrence < expected_cooccurrence else "co-occurring",
    }


def main():
    df_mut, df_cna = load_data()
    n = len(df_mut)
    assert n == 2957, f"Expected 2957 patients, got {n}"

    genes = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]

    # ---- Pairwise tests for all 6 combinations ----
    pairwise_results = {}
    pair_labels = []
    pair_pvalues = []

    for g1, g2 in combinations(genes, 2):
        label = f"{g1}+{g2}"
        result = pairwise_test(df_mut[g1], df_mut[g2], g1, g2, n)
        pairwise_results[label] = result
        pair_labels.append(label)
        pair_pvalues.append(result["fisher_pvalue"])

    # ---- Additional pairs ----
    # IDH2 vs SETBP1 (already in the 6 combos, but we re-confirm)
    # SETBP1 vs del7/del7q
    # PTPN11 vs SETBP1 (already in the 6 combos)

    # SETBP1 vs del7
    label = "SETBP1+del7"
    result = pairwise_test(df_mut["SETBP1"], df_cna["del7"], "SETBP1", "del7", n)
    pairwise_results[label] = result
    pair_labels.append(label)
    pair_pvalues.append(result["fisher_pvalue"])

    # SETBP1 vs del7q
    label = "SETBP1+del7q"
    result = pairwise_test(df_mut["SETBP1"], df_cna["del7q"], "SETBP1", "del7q", n)
    pairwise_results[label] = result
    pair_labels.append(label)
    pair_pvalues.append(result["fisher_pvalue"])

    # SETBP1 vs del7_or_del7q (combined)
    del7_combined = ((df_cna["del7"] == 1) | (df_cna["del7q"] == 1)).astype(int)
    label = "SETBP1+del7_or_del7q"
    result = pairwise_test(df_mut["SETBP1"], del7_combined, "SETBP1", "del7_or_del7q", n)
    pairwise_results[label] = result
    pair_labels.append(label)
    pair_pvalues.append(result["fisher_pvalue"])

    # ---- Benjamini-Hochberg correction ----
    bh_corrected = benjamini_hochberg(pair_pvalues)
    for i, label in enumerate(pair_labels):
        pairwise_results[label]["bh_corrected_pvalue"] = round(bh_corrected[i], 10)

    # ---- Triple combinations ----
    triple_results = {}
    for triple in combinations(genes, 3):
        label = "+".join(triple)
        result = compute_triple_probability(df_mut, list(triple), n)
        triple_results[label] = result

    # ---- Quadruple combination ----
    quad_result = compute_quadruple_probability(df_mut, genes, n)

    # ---- DISCOVER-style group test ----
    discover_result = discover_test(df_mut, genes, n)

    # ---- Assemble output ----
    output = {
        "dataset": "IPSS-M",
        "n_patients": n,
        "genes_analyzed": genes,
        "marginal_mutation_counts": {g: int(df_mut[g].sum()) for g in genes},
        "marginal_mutation_rates": {g: round(df_mut[g].sum() / n, 6) for g in genes},
        "pairwise_tests": pairwise_results,
        "triple_probabilities": triple_results,
        "quadruple_probability": quad_result,
        "group_exclusivity_test": discover_result,
        "methodology": {
            "fisher_exact": "scipy.stats.fisher_exact, two-sided",
            "odds_ratio_ci": "Woolf logit method with Haldane +0.5 correction for zero cells",
            "bh_correction": "Benjamini-Hochberg FDR applied across all pairwise tests",
            "triple_quadruple": "Poisson approximation for rare event probability under independence",
            "group_test": "Simplified DISCOVER approximation using Poisson model for total pairwise co-occurrences",
        },
    }

    # ---- Write output ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # ---- Print summary ----
    print(f"Results written to {OUT_PATH}")
    print(f"\nDataset: IPSS-M, N = {n}")
    print(f"\nMarginal rates:")
    for g in genes:
        cnt = int(df_mut[g].sum())
        print(f"  {g}: {cnt} ({cnt/n*100:.1f}%)")

    print(f"\n{'Pair':<25} {'Obs':>4} {'Exp':>7} {'OR':>8} {'95% CI':>22} {'Fisher p':>12} {'BH p':>12} {'Direction'}")
    print("-" * 110)
    for label in pair_labels:
        r = pairwise_results[label]
        ci_str = f"[{r['odds_ratio_95ci'][0]:.3f}, {r['odds_ratio_95ci'][1]:.3f}]"
        print(f"  {label:<23} {r['observed']:>4} {r['expected_under_independence']:>7.2f} "
              f"{r['odds_ratio']:>8.3f} {ci_str:>22} {r['fisher_pvalue']:>12.2e} "
              f"{r['bh_corrected_pvalue']:>12.2e} {r['direction']}")

    print(f"\nTriple co-occurrences:")
    for label, r in triple_results.items():
        print(f"  {label}: observed={r['observed']}, expected={r['expected_under_independence']:.4f}, "
              f"P(X=0)={r['poisson_p_of_zero']:.6e}")

    print(f"\nQuadruple co-occurrence:")
    print(f"  {'+'.join(genes)}: observed={quad_result['observed']}, "
          f"expected={quad_result['expected_under_independence']:.6f}, "
          f"P(X=0)={quad_result['poisson_p_of_zero']:.6e}")

    print(f"\nGroup exclusivity test:")
    print(f"  Total pairwise co-occurrences: observed={discover_result['total_pairwise_cooccurrences_observed']}, "
          f"expected={discover_result['total_pairwise_cooccurrences_expected']:.2f}")
    print(f"  Poisson ME p-value: {discover_result['poisson_me_pvalue']:.6e}")
    print(f"  Coverage: {discover_result['coverage']:.4f} ({discover_result['n_patients_with_any']} patients)")
    print(f"  Exclusivity: {discover_result['exclusivity_among_mutated']:.4f}")


if __name__ == "__main__":
    main()
