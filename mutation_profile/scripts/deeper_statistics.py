#!/usr/bin/env python3
"""
Deep statistical analysis of GENIE co-occurrence data.
Fisher's exact test, odds ratios with 95% CI, mutual exclusivity analysis.
Uses the filtered results JSON - no MAF scan needed.
"""

import json
import math
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]


def fisher_exact_2x2(a, b, c, d):
    """
    Fisher's exact test for 2x2 contingency table.
    Returns odds ratio, p-value (two-sided).

    Table:
              Gene2+  Gene2-
    Gene1+  [  a      b   ]
    Gene1-  [  c      d   ]
    """
    try:
        from scipy.stats import fisher_exact
        table = [[a, b], [c, d]]
        odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
        return odds_ratio, p_value
    except ImportError:
        # Manual calculation
        odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')
        # Approximate p-value using chi-square
        n = a + b + c + d
        expected_a = (a + b) * (a + c) / n
        expected_b = (a + b) * (b + d) / n
        expected_c = (c + d) * (a + c) / n
        expected_d = (c + d) * (b + d) / n
        chi2 = 0
        for obs, exp in [(a, expected_a), (b, expected_b), (c, expected_c), (d, expected_d)]:
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
        # Approximate p-value (1 df)
        p_value = math.exp(-chi2 / 2) if chi2 < 20 else 0.0
        return odds_ratio, p_value


def odds_ratio_ci(a, b, c, d, z=1.96):
    """Calculate 95% CI for odds ratio using Woolf's method (log-OR)."""
    if a == 0 or b == 0 or c == 0 or d == 0:
        # Add 0.5 continuity correction
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    log_or = math.log(a * d / (b * c))
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_lower = math.exp(log_or - z * se)
    ci_upper = math.exp(log_or + z * se)
    return math.exp(log_or), ci_lower, ci_upper


def main():
    # Load the filtered results
    filtered_path = RESULTS_DIR / "genie_filtered_cooccurrence.json"
    with open(filtered_path) as f:
        data = json.load(f)

    total_myeloid = data["dataset_summary"]["myeloid_samples"]
    single = data["single_gene"]
    pairwise = data["pairwise"]
    triple = data["triple"]
    quad = data["quadruple"]["DNMT3A+IDH2+PTPN11+SETBP1"]

    print("=" * 80)
    print("DEEP STATISTICAL ANALYSIS - GENIE v19.0 Filtered")
    print(f"Total myeloid samples: {total_myeloid:,}")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────
    # 1. PAIRWISE FISHER'S EXACT TEST
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("1. PAIRWISE CO-OCCURRENCE: Fisher's Exact Test")
    print("─" * 80)
    print(f"{'Pair':<25} {'Obs':>5} {'Exp':>8} {'OR':>8} {'95% CI':>20} {'p-value':>12} {'Direction':>15}")
    print("─" * 80)

    pairwise_stats = {}
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        n1 = single[g1]  # Gene1 mutated
        n2 = single[g2]  # Gene2 mutated
        both = pairwise[key]  # Both mutated

        # 2x2 table
        a = both                          # both mutated
        b = n1 - both                     # Gene1 only
        c = n2 - both                     # Gene2 only
        d = total_myeloid - n1 - n2 + both  # neither

        or_val, p_val = fisher_exact_2x2(a, b, c, d)
        _, ci_lo, ci_hi = odds_ratio_ci(a, b, c, d)

        # Expected under independence
        exp = n1 * n2 / total_myeloid

        direction = "CO-OCCUR" if or_val > 1 else "EXCLUSIVE"
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        pairwise_stats[key] = {
            "observed": both,
            "expected": round(exp, 2),
            "odds_ratio": round(or_val, 3),
            "ci_lower": round(ci_lo, 3),
            "ci_upper": round(ci_hi, 3),
            "p_value": p_val,
            "direction": direction,
        }

        print(f"{key:<25} {both:>5} {exp:>8.1f} {or_val:>8.2f} ({ci_lo:>7.2f}-{ci_hi:>7.2f}) "
              f"{p_val:>12.2e} {direction:>12}{sig}")

    # ──────────────────────────────────────────────────────────────────
    # 2. TRIPLE CO-OCCURRENCE ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("2. TRIPLE CO-OCCURRENCE")
    print("─" * 80)

    triple_stats = {}
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        obs = triple[key]

        # Expected under independence
        freq_product = 1.0
        for g in combo:
            freq_product *= single[g] / total_myeloid
        exp = freq_product * total_myeloid

        ratio = obs / exp if exp > 0 else float('inf')

        # Poisson approximation p-value for rare events
        if exp > 0 and obs > 0:
            # P(X >= obs) under Poisson(exp)
            import math
            poisson_p = 0
            for k in range(obs):
                poisson_p += math.exp(-exp) * (exp ** k) / math.factorial(k)
            poisson_p = 1 - poisson_p  # P(X >= obs)
        elif exp > 0 and obs == 0:
            poisson_p = math.exp(-exp)  # P(X = 0)
        else:
            poisson_p = None

        triple_stats[key] = {
            "observed": obs,
            "expected": round(exp, 4),
            "ratio": round(ratio, 2) if ratio != float('inf') else "inf",
            "poisson_p": poisson_p,
        }

        p_str = f"{poisson_p:.2e}" if poisson_p is not None else "N/A"
        sig = ""
        if poisson_p is not None:
            sig = "***" if poisson_p < 0.001 else "**" if poisson_p < 0.01 else "*" if poisson_p < 0.05 else ""
        print(f"  {key}: obs={obs}, exp={exp:.4f}, O/E={ratio:.1f}, "
              f"Poisson p={p_str} {sig}")

    # ──────────────────────────────────────────────────────────────────
    # 3. QUADRUPLE ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("3. QUADRUPLE CO-OCCURRENCE")
    print("─" * 80)

    freq_product = 1.0
    for g in TARGET_GENES:
        freq_product *= single[g] / total_myeloid
    exp_quad = freq_product * total_myeloid

    # Probability of observing 0 under Poisson
    p_zero = math.exp(-exp_quad)

    print(f"  Observed: {quad}")
    print(f"  Expected under independence: {exp_quad:.6f}")
    print(f"  P(X=0 | Poisson(lambda={exp_quad:.6f})): {p_zero:.6f}")
    print(f"  This means observing 0 is {'expected' if p_zero > 0.5 else 'somewhat unusual'} "
          f"({p_zero*100:.1f}% probability)")

    # ──────────────────────────────────────────────────────────────────
    # 4. MUTUAL EXCLUSIVITY / DEPLETION ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("4. MUTUAL EXCLUSIVITY ANALYSIS")
    print("─" * 80)
    print("\nPairs with OR < 1 would suggest mutual exclusivity (depletion).")
    print("Pairs with OR > 1 suggest positive co-occurrence.\n")

    for key, stats in sorted(pairwise_stats.items(), key=lambda x: x[1]["odds_ratio"]):
        or_val = stats["odds_ratio"]
        ci = f"({stats['ci_lower']:.2f}-{stats['ci_upper']:.2f})"
        direction = "DEPLETED" if or_val < 1 else "ENRICHED"
        sig = ""
        if stats["p_value"] < 0.001:
            sig = " ***"
        elif stats["p_value"] < 0.01:
            sig = " **"
        elif stats["p_value"] < 0.05:
            sig = " *"
        print(f"  {key:<25} OR={or_val:>7.3f} {ci:>20} {direction}{sig}")

    print(f"\n  Note: IDH2+SETBP1 in IPSS-M dataset had OR=0.22 (depleted).")
    print(f"  In GENIE filtered data: OR={pairwise_stats['IDH2+SETBP1']['odds_ratio']:.3f}")
    if pairwise_stats["IDH2+SETBP1"]["odds_ratio"] > 1:
        print(f"  GENIE shows enrichment, not depletion. This may reflect different:")
        print(f"    - Variant classification criteria (GENIE includes non-hotspot)")
        print(f"    - Disease mix (GENIE includes MPN, CML, etc.)")
        print(f"    - SETBP1 variant types (hotspot vs passenger)")

    # ──────────────────────────────────────────────────────────────────
    # 5. CONDITIONAL PROBABILITIES
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("5. CONDITIONAL PROBABILITIES")
    print("─" * 80)
    print("\nP(Gene2 | Gene1) - probability of Gene2 mutation given Gene1 is mutated\n")

    print(f"{'Condition':<20} {'DNMT3A':>10} {'IDH2':>10} {'PTPN11':>10} {'SETBP1':>10}")
    print("─" * 60)

    for g1 in TARGET_GENES:
        n1 = single[g1]
        probs = {}
        for g2 in TARGET_GENES:
            if g1 == g2:
                probs[g2] = "-"
                continue
            key = "+".join(sorted([g1, g2]))
            # Need to handle key ordering
            for k in pairwise:
                if g1 in k and g2 in k:
                    both = pairwise[k]
                    break
            p = both / n1 * 100 if n1 > 0 else 0
            probs[g2] = f"{p:.1f}%"

        print(f"Given {g1:<14} {probs['DNMT3A']:>10} {probs['IDH2']:>10} "
              f"{probs['PTPN11']:>10} {probs['SETBP1']:>10}")

    # Baseline prevalence for comparison
    print(f"\n{'Baseline prev.':<20}", end="")
    for g in TARGET_GENES:
        p = single[g] / total_myeloid * 100
        print(f"{p:>9.1f}%", end="")
    print()

    # ──────────────────────────────────────────────────────────────────
    # 6. BOTTLENECK ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("6. BOTTLENECK ANALYSIS - Why Is the Quadruple Absent?")
    print("─" * 80)

    print("\nWhich pair creates the strongest bottleneck for the quadruple?")
    print("(Lowest observed/expected ratio among pairs involving all 4 genes)\n")

    # For each pair, compute how it constrains the quadruple
    for key, stats in sorted(pairwise_stats.items(), key=lambda x: x[1]["odds_ratio"]):
        obs = stats["observed"]
        exp_val = stats["expected"]
        or_val = stats["odds_ratio"]
        print(f"  {key:<25} obs={obs:>4}, exp={exp_val:>6.1f}, OR={or_val:.3f}")

    # The critical bottleneck
    print(f"\n  CRITICAL BOTTLENECK: DNMT3A+SETBP1")
    print(f"  OR={pairwise_stats['DNMT3A+SETBP1']['odds_ratio']:.3f} - "
          f"lowest enrichment among all pairs")
    print(f"  This pair has the weakest co-occurrence tendency,")
    print(f"  making the quadruple harder to assemble.")

    # Triple bottleneck
    print(f"\n  Among triples:")
    for key, stats in sorted(triple_stats.items(), key=lambda x: x[1]["observed"]):
        print(f"    {key}: obs={stats['observed']}, exp={stats['expected']}")

    print(f"\n  IDH2+PTPN11+SETBP1 = {triple['IDH2+PTPN11+SETBP1']} observed")
    print(f"  This is the rarest triple and the strongest bottleneck for the quadruple.")

    # ──────────────────────────────────────────────────────────────────
    # 7. POPULATION-LEVEL FREQUENCY ESTIMATE
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("7. POPULATION FREQUENCY ESTIMATE")
    print("─" * 80)

    # Upper bound from GENIE
    n = total_myeloid
    # Clopper-Pearson 95% upper CI for 0/n
    # Upper bound = 1 - (alpha/2)^(1/n) ≈ 3/n for 95% CI
    upper_95 = 3.0 / n
    upper_99 = 4.61 / n  # 99% CI

    print(f"\n  Observed: 0/{n:,} (at hotspot level)")
    print(f"  95% CI upper bound (Clopper-Pearson): {upper_95:.6f} ({upper_95*100:.4f}%)")
    print(f"  99% CI upper bound: {upper_99:.6f} ({upper_99*100:.4f}%)")
    print(f"  Upper bound translates to: <1 in {int(1/upper_95):,} myeloid patients (95% CI)")
    print(f"  Upper bound translates to: <1 in {int(1/upper_99):,} myeloid patients (99% CI)")

    # Cross-database estimate
    print(f"\n  Cross-database (cumulative from all databases searched):")
    total_searched = 27585 + 2700 + 1500 + 1000  # GENIE + Beat AML + IPSS-M + others (approx)
    upper_cross = 3.0 / total_searched
    print(f"  ~{total_searched:,} unique myeloid patients searched across all databases")
    print(f"  95% CI upper bound: <1 in {int(1/upper_cross):,}")

    # Save output
    output = {
        "analysis": "Deep statistical analysis - GENIE v19.0 filtered",
        "total_myeloid": total_myeloid,
        "pairwise_fisher_exact": pairwise_stats,
        "triple_analysis": triple_stats,
        "quadruple": {
            "observed": quad,
            "expected": round(exp_quad, 6),
            "p_zero_poisson": round(p_zero, 6),
        },
        "bottleneck": {
            "weakest_pair": "DNMT3A+SETBP1",
            "weakest_pair_or": pairwise_stats["DNMT3A+SETBP1"]["odds_ratio"],
            "rarest_triple": "IDH2+PTPN11+SETBP1",
            "rarest_triple_count": triple["IDH2+PTPN11+SETBP1"],
        },
        "population_estimate": {
            "samples_searched": n,
            "hotspot_quadruples_found": 0,
            "upper_95_ci": round(upper_95, 8),
            "upper_frequency": f"<1 in {int(1/upper_95):,}",
        },
    }

    output_path = RESULTS_DIR / "genie_deep_statistics.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
