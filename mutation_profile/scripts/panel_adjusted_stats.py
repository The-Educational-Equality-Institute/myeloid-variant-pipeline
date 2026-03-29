#!/usr/bin/env python3
"""
Panel-adjusted co-occurrence statistics for GENIE v19.0.

Problem: SETBP1 is only covered by 52/166 gene panels. The previous analysis
used all 27,585 myeloid samples as denominator, but many were never tested for
SETBP1. This inflates the denominator and dilutes SETBP1-involving statistics.

Solution: Identify which panels cover ALL 4 target genes (DNMT3A, IDH2, PTPN11,
SETBP1), restrict to samples sequenced on those panels, and recalculate all
statistics using this panel-adjusted denominator.

Filters: protein-altering variants only (Missense_Mutation, Nonsense_Mutation,
Frame_Shift_Del, Frame_Shift_Ins, Splice_Site, In_Frame_Del, In_Frame_Ins,
Nonstop_Mutation, Translation_Start_Site).
"""

import csv
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from scipy.stats import beta as beta_dist
from scipy.stats import fisher_exact

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
GENIE_RAW = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]

PROTEIN_ALTERING = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Nonstop_Mutation",
    "Translation_Start_Site",
}

# ONCOTREE codes for myeloid malignancies.
# Prefix-based matching with explicit exclusions for non-myeloid codes
# that share prefixes (MPNST = peripheral nerve sheath tumor,
# SMZL/SMMCL = lymphomas).
MYELOID_PREFIXES = ("AML", "MDS", "MPN", "CMML", "JMML", "CML", "TMN", "SM", "CEL", "HES")
MYELOID_EXACT = {"AMML", "APL", "APMF", "AUL", "MLADS", "MS"}
NON_MYELOID_EXACT = {"MPNST", "SMZL", "SMMCL"}


def is_myeloid(oncotree_code: str) -> bool:
    """Return True if the ONCOTREE_CODE represents a myeloid malignancy."""
    if not oncotree_code:
        return False
    code = oncotree_code.strip()
    if code in NON_MYELOID_EXACT:
        return False
    if code in MYELOID_EXACT:
        return True
    return code.startswith(MYELOID_PREFIXES)


# ---------------------------------------------------------------------------
# Step 1: Parse gene panel files
# ---------------------------------------------------------------------------
def parse_gene_panels() -> dict[str, set[str]]:
    """
    Read all data_gene_panel_*.txt files.
    Returns {panel_id: set_of_genes}.
    """
    panels = {}
    panel_files = sorted(GENIE_RAW.glob("data_gene_panel_*.txt"))
    for pf in panel_files:
        panel_id = None
        genes = set()
        with open(pf) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("stable_id:"):
                    panel_id = line.split(":", 1)[1].strip()
                elif line.startswith("gene_list:"):
                    # Tab-separated gene names after "gene_list:"
                    parts = line.split("\t")
                    genes = {g.strip() for g in parts[1:] if g.strip()}
        if panel_id:
            panels[panel_id] = genes
    return panels


# ---------------------------------------------------------------------------
# Step 2: Identify panels covering all 4 target genes
# ---------------------------------------------------------------------------
def find_covering_panels(panels: dict[str, set[str]]) -> set[str]:
    """Return panel IDs that cover ALL 4 target genes."""
    target_set = set(TARGET_GENES)
    covering = set()
    for panel_id, genes in panels.items():
        if target_set.issubset(genes):
            covering.add(panel_id)
    return covering


# ---------------------------------------------------------------------------
# Step 3: Parse clinical sample file
# ---------------------------------------------------------------------------
def parse_clinical_samples() -> dict[str, tuple[str, str]]:
    """
    Read data_clinical_sample.txt.
    Returns {sample_id: (oncotree_code, seq_assay_id)}.
    Skips comment/header lines starting with '#'.
    """
    clinical_path = GENIE_RAW / "data_clinical_sample.txt"
    samples = {}
    with open(clinical_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = None
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = row
                continue
            sample_id = row[header.index("SAMPLE_ID")]
            oncotree = row[header.index("ONCOTREE_CODE")]
            assay = row[header.index("SEQ_ASSAY_ID")]
            samples[sample_id] = (oncotree, assay)
    return samples


# ---------------------------------------------------------------------------
# Step 4: Build myeloid sample sets (adjusted and unadjusted)
# ---------------------------------------------------------------------------
def build_sample_sets(
    samples: dict[str, tuple[str, str]],
    covering_panels: set[str],
) -> tuple[set[str], set[str]]:
    """
    Returns:
      - all_myeloid: all myeloid sample IDs (unadjusted)
      - adjusted_myeloid: myeloid samples on panels covering all 4 genes
    """
    all_myeloid = set()
    adjusted_myeloid = set()
    for sample_id, (oncotree, assay) in samples.items():
        if is_myeloid(oncotree):
            all_myeloid.add(sample_id)
            if assay in covering_panels:
                adjusted_myeloid.add(sample_id)
    return all_myeloid, adjusted_myeloid


# ---------------------------------------------------------------------------
# Step 5: Scan MAF file for mutations in target genes
# ---------------------------------------------------------------------------
def scan_mutations(
    all_myeloid: set[str],
    adjusted_myeloid: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    Scan data_mutations_extended.txt for protein-altering mutations
    in the 4 target genes.

    Returns two dicts (unadjusted, adjusted), each mapping
    gene_name -> set of sample_ids with a mutation in that gene.
    """
    maf_path = GENIE_RAW / "data_mutations_extended.txt"

    target_set = set(TARGET_GENES)
    unadj_hits: dict[str, set[str]] = {g: set() for g in TARGET_GENES}
    adj_hits: dict[str, set[str]] = {g: set() for g in TARGET_GENES}

    with open(maf_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = None
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = row
                hugo_idx = header.index("Hugo_Symbol")
                vc_idx = header.index("Variant_Classification")
                sample_idx = header.index("Tumor_Sample_Barcode")
                continue

            gene = row[hugo_idx]
            if gene not in target_set:
                continue
            vc = row[vc_idx]
            if vc not in PROTEIN_ALTERING:
                continue
            sample_id = row[sample_idx]

            if sample_id in all_myeloid:
                unadj_hits[gene].add(sample_id)
            if sample_id in adjusted_myeloid:
                adj_hits[gene].add(sample_id)

    return unadj_hits, adj_hits


# ---------------------------------------------------------------------------
# Step 6: Compute co-occurrence counts
# ---------------------------------------------------------------------------
def compute_cooccurrence(hits: dict[str, set[str]]) -> dict:
    """
    From per-gene sample sets, compute single, pairwise, triple, and
    quadruple co-occurrence counts.
    """
    single = {g: len(s) for g, s in hits.items()}

    pairwise = {}
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        pairwise[key] = len(hits[g1] & hits[g2])

    triple = {}
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        s = hits[combo[0]]
        for g in combo[1:]:
            s = s & hits[g]
        triple[key] = len(s)

    quad_key = "+".join(TARGET_GENES)
    quad_set = hits[TARGET_GENES[0]]
    for g in TARGET_GENES[1:]:
        quad_set = quad_set & hits[g]
    quadruple = {quad_key: len(quad_set)}

    return {
        "single": single,
        "pairwise": pairwise,
        "triple": triple,
        "quadruple": quadruple,
    }


# ---------------------------------------------------------------------------
# Step 7: Fisher's exact test with OR + CI
# ---------------------------------------------------------------------------
def fisher_test(a: int, b: int, c: int, d: int) -> dict:
    """
    2x2 Fisher's exact test.
              Gene2+  Gene2-
    Gene1+  [  a       b  ]
    Gene1-  [  c       d  ]

    Returns dict with odds_ratio, ci_lower, ci_upper, p_value.
    """
    table = [[a, b], [c, d]]
    result = fisher_exact(table, alternative="two-sided")
    or_val = result.statistic
    p_val = result.pvalue

    # Woolf CI for log-OR
    aa, bb, cc, dd = a, b, c, d
    if aa == 0 or bb == 0 or cc == 0 or dd == 0:
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = math.log(aa * dd / (bb * cc))
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    ci_lower = math.exp(log_or - 1.96 * se)
    ci_upper = math.exp(log_or + 1.96 * se)

    return {
        "odds_ratio": round(or_val, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "p_value": p_val,
    }


# ---------------------------------------------------------------------------
# Step 8: Clopper-Pearson confidence interval
# ---------------------------------------------------------------------------
def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Upper bound of Clopper-Pearson (exact binomial) confidence interval.
    For k=0: upper = 1 - (alpha/2)^(1/n)
    General: uses scipy.stats.beta ppf.
    """
    if k == 0:
        return 1.0 - (alpha / 2) ** (1 / n)
    return beta_dist.ppf(1 - alpha / 2, k + 1, n - k)


# ---------------------------------------------------------------------------
# Step 9: Poisson p-value for rare triple/quad events
# ---------------------------------------------------------------------------
def poisson_pvalue_ge(obs: int, expected: float) -> float | None:
    """P(X >= obs) under Poisson(expected). For enrichment test."""
    if expected <= 0:
        return None
    if obs == 0:
        return 1.0  # P(X >= 0) = 1
    p_less = 0.0
    for k in range(obs):
        p_less += math.exp(-expected) * (expected**k) / math.factorial(k)
    return 1.0 - p_less


def poisson_pvalue_le(obs: int, expected: float) -> float | None:
    """P(X <= obs) under Poisson(expected). For depletion test."""
    if expected <= 0:
        return None
    p = 0.0
    for k in range(obs + 1):
        p += math.exp(-expected) * (expected**k) / math.factorial(k)
    return p


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(
    label: str,
    n_total: int,
    cooc: dict,
) -> dict:
    """Run full statistical suite on a set of co-occurrence counts."""
    single = cooc["single"]
    pairwise_counts = cooc["pairwise"]
    triple_counts = cooc["triple"]
    quad_counts = cooc["quadruple"]

    # -- Pairwise Fisher's exact --
    pairwise_stats = {}
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        both = pairwise_counts[key]
        n1 = single[g1]
        n2 = single[g2]
        a = both
        b = n1 - both
        c = n2 - both
        d = n_total - n1 - n2 + both
        expected = n1 * n2 / n_total

        stats = fisher_test(a, b, c, d)
        stats["observed"] = both
        stats["expected"] = round(expected, 2)
        stats["direction"] = "CO-OCCUR" if stats["odds_ratio"] > 1 else "EXCLUSIVE"
        pairwise_stats[key] = stats

    # -- Triple expected counts --
    triple_stats = {}
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        obs = triple_counts[key]
        freq_prod = 1.0
        for g in combo:
            freq_prod *= single[g] / n_total
        expected = freq_prod * n_total
        ratio = obs / expected if expected > 0 else float("inf")
        p_enrich = poisson_pvalue_ge(obs, expected)

        triple_stats[key] = {
            "observed": obs,
            "expected": round(expected, 4),
            "obs_exp_ratio": round(ratio, 2) if ratio != float("inf") else "inf",
            "poisson_p_enrichment": p_enrich,
        }

    # -- Quadruple --
    quad_key = "+".join(TARGET_GENES)
    quad_obs = quad_counts[quad_key]
    freq_prod = 1.0
    for g in TARGET_GENES:
        freq_prod *= single[g] / n_total
    quad_expected = freq_prod * n_total
    p_zero = math.exp(-quad_expected) if quad_expected > 0 else 1.0

    quad_stats = {
        "observed": quad_obs,
        "expected": round(quad_expected, 6),
        "p_zero_poisson": round(p_zero, 6),
    }

    # -- Clopper-Pearson upper bound for 0/n --
    upper_95 = clopper_pearson_upper(quad_obs, n_total, alpha=0.05)
    upper_99 = clopper_pearson_upper(quad_obs, n_total, alpha=0.01)

    pop_estimate = {
        "n": n_total,
        "observed_quadruples": quad_obs,
        "upper_95_ci": round(upper_95, 8),
        "upper_99_ci": round(upper_99, 8),
        "upper_95_frequency": f"<1 in {int(1 / upper_95):,}" if upper_95 > 0 else "undefined",
        "upper_99_frequency": f"<1 in {int(1 / upper_99):,}" if upper_99 > 0 else "undefined",
    }

    return {
        "label": label,
        "total_myeloid": n_total,
        "single_gene_counts": single,
        "single_gene_prevalence": {g: round(single[g] / n_total * 100, 2) for g in TARGET_GENES},
        "pairwise_fisher_exact": pairwise_stats,
        "triple_analysis": triple_stats,
        "quadruple": quad_stats,
        "population_estimate": pop_estimate,
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------
def print_results(results: dict, prefix: str = "") -> None:
    label = results["label"]
    n = results["total_myeloid"]
    single = results["single_gene_counts"]
    prev = results["single_gene_prevalence"]
    pw = results["pairwise_fisher_exact"]
    tri = results["triple_analysis"]
    quad = results["quadruple"]
    pop = results["population_estimate"]

    print(f"\n{'=' * 80}")
    print(f"{prefix}{label}")
    print(f"Total myeloid samples: {n:,}")
    print(f"{'=' * 80}")

    # Single gene
    print(f"\n  Single-gene counts:")
    for g in TARGET_GENES:
        print(f"    {g:<10} {single[g]:>6}  ({prev[g]:.2f}%)")

    # Pairwise
    print(f"\n  {'Pair':<25} {'Obs':>5} {'Exp':>8} {'OR':>8} {'95% CI':>20} {'p-value':>12} {'Dir':>12}")
    print(f"  {'-' * 90}")
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        s = pw[key]
        sig = "***" if s["p_value"] < 0.001 else "**" if s["p_value"] < 0.01 else "*" if s["p_value"] < 0.05 else ""
        print(
            f"  {key:<25} {s['observed']:>5} {s['expected']:>8.1f} {s['odds_ratio']:>8.3f} "
            f"({s['ci_lower']:>7.3f}-{s['ci_upper']:>7.3f}) "
            f"{s['p_value']:>12.2e} {s['direction']:>10}{sig}"
        )

    # Triples
    print(f"\n  Triple co-occurrence:")
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        s = tri[key]
        p_str = f"{s['poisson_p_enrichment']:.2e}" if s["poisson_p_enrichment"] is not None else "N/A"
        print(f"    {key}: obs={s['observed']}, exp={s['expected']}, O/E={s['obs_exp_ratio']}, p={p_str}")

    # Quadruple
    print(f"\n  Quadruple: obs={quad['observed']}, exp={quad['expected']}, P(X=0)={quad['p_zero_poisson']}")

    # Population estimate
    print(f"\n  Population frequency upper bound (Clopper-Pearson exact):")
    print(f"    95% CI: {pop['upper_95_ci']:.8f}  =>  {pop['upper_95_frequency']}")
    print(f"    99% CI: {pop['upper_99_ci']:.8f}  =>  {pop['upper_99_frequency']}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(unadj: dict, adj: dict) -> None:
    print(f"\n{'=' * 100}")
    print("SIDE-BY-SIDE COMPARISON: Unadjusted vs Panel-Adjusted")
    print(f"{'=' * 100}")

    n_u = unadj["total_myeloid"]
    n_a = adj["total_myeloid"]
    print(f"\n  Denominator:  unadjusted = {n_u:,}   |   adjusted = {n_a:,}   |   diff = {n_u - n_a:,} ({(n_u - n_a) / n_u * 100:.1f}% removed)")

    # Single gene prevalence
    print(f"\n  {'Gene':<12} {'Unadj count':>12} {'Unadj %':>10} {'Adj count':>12} {'Adj %':>10} {'Change':>10}")
    print(f"  {'-' * 66}")
    for g in TARGET_GENES:
        cu = unadj["single_gene_counts"][g]
        pu = unadj["single_gene_prevalence"][g]
        ca = adj["single_gene_counts"][g]
        pa = adj["single_gene_prevalence"][g]
        change = pa - pu
        print(f"  {g:<12} {cu:>12,} {pu:>9.2f}% {ca:>12,} {pa:>9.2f}% {change:>+9.2f}pp")

    # Pairwise
    print(f"\n  {'Pair':<25} {'Unadj OR':>10} {'Adj OR':>10} {'Unadj p':>12} {'Adj p':>12} {'OR change':>12}")
    print(f"  {'-' * 81}")
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        su = unadj["pairwise_fisher_exact"][key]
        sa = adj["pairwise_fisher_exact"][key]
        or_change = sa["odds_ratio"] - su["odds_ratio"]
        print(
            f"  {key:<25} {su['odds_ratio']:>10.3f} {sa['odds_ratio']:>10.3f} "
            f"{su['p_value']:>12.2e} {sa['p_value']:>12.2e} {or_change:>+11.3f}"
        )

    # Triple
    print(f"\n  {'Triple':<35} {'Unadj obs':>10} {'Unadj exp':>10} {'Adj obs':>10} {'Adj exp':>10}")
    print(f"  {'-' * 75}")
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        tu = unadj["triple_analysis"][key]
        ta = adj["triple_analysis"][key]
        print(
            f"  {key:<35} {tu['observed']:>10} {tu['expected']:>10} "
            f"{ta['observed']:>10} {ta['expected']:>10}"
        )

    # Quad
    qu = unadj["quadruple"]
    qa = adj["quadruple"]
    print(f"\n  Quadruple: unadj obs={qu['observed']} exp={qu['expected']} | adj obs={qa['observed']} exp={qa['expected']}")

    # Population estimate
    pu = unadj["population_estimate"]
    pa = adj["population_estimate"]
    print(f"\n  Population frequency upper bound (95% CI):")
    print(f"    Unadjusted: {pu['upper_95_frequency']}  (n={pu['n']:,})")
    print(f"    Adjusted:   {pa['upper_95_frequency']}  (n={pa['n']:,})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("PANEL-ADJUSTED CO-OCCURRENCE STATISTICS -- GENIE v19.0")
    print("=" * 80)

    # Step 1: Parse gene panels
    print("\n[1/6] Parsing gene panel files...")
    panels = parse_gene_panels()
    print(f"  Found {len(panels)} gene panels")

    # Step 2: Find covering panels
    print("\n[2/6] Identifying panels covering ALL 4 target genes...")
    covering = find_covering_panels(panels)
    print(f"  {len(covering)} / {len(panels)} panels cover DNMT3A + IDH2 + PTPN11 + SETBP1")

    # Report per-gene coverage
    for g in TARGET_GENES:
        count = sum(1 for p, genes in panels.items() if g in genes)
        print(f"    {g:<10} covered by {count}/{len(panels)} panels")

    # List covering panels
    print(f"\n  Covering panels ({len(covering)}):")
    for p in sorted(covering):
        print(f"    - {p} ({len(panels[p])} genes)")

    # Step 3: Parse clinical samples
    print("\n[3/6] Parsing clinical sample file...")
    samples = parse_clinical_samples()
    print(f"  Total samples in GENIE: {len(samples):,}")

    # Step 4: Build sample sets
    print("\n[4/6] Building myeloid sample sets...")
    all_myeloid, adjusted_myeloid = build_sample_sets(samples, covering)
    print(f"  All myeloid samples (unadjusted): {len(all_myeloid):,}")
    print(f"  Myeloid samples on 4-gene panels (adjusted): {len(adjusted_myeloid):,}")
    print(f"  Samples excluded by panel filter: {len(all_myeloid) - len(adjusted_myeloid):,} "
          f"({(len(all_myeloid) - len(adjusted_myeloid)) / len(all_myeloid) * 100:.1f}%)")

    # Show panel breakdown for adjusted samples
    panel_counts = defaultdict(int)
    for sid in adjusted_myeloid:
        _, assay = samples[sid]
        panel_counts[assay] += 1
    print(f"\n  Adjusted samples by panel:")
    for panel, count in sorted(panel_counts.items(), key=lambda x: -x[1]):
        print(f"    {panel:<35} {count:>6} samples")

    # Step 5: Scan MAF
    print("\n[5/6] Scanning MAF file for mutations in target genes (protein-altering only)...")
    unadj_hits, adj_hits = scan_mutations(all_myeloid, adjusted_myeloid)
    print("  Done.")
    for g in TARGET_GENES:
        print(f"    {g:<10} unadj={len(unadj_hits[g]):>5}  adj={len(adj_hits[g]):>5}")

    # Step 6: Compute statistics
    print("\n[6/6] Computing statistics...")
    unadj_cooc = compute_cooccurrence(unadj_hits)
    adj_cooc = compute_cooccurrence(adj_hits)

    unadj_results = run_analysis(
        "UNADJUSTED (all myeloid samples, n={:,})".format(len(all_myeloid)),
        len(all_myeloid),
        unadj_cooc,
    )
    adj_results = run_analysis(
        "PANEL-ADJUSTED (4-gene covered samples, n={:,})".format(len(adjusted_myeloid)),
        len(adjusted_myeloid),
        adj_cooc,
    )

    # Print results
    print_results(unadj_results, prefix="[A] ")
    print_results(adj_results, prefix="[B] ")
    print_comparison(unadj_results, adj_results)

    # Key finding summary
    print(f"\n{'=' * 80}")
    print("KEY FINDINGS")
    print(f"{'=' * 80}")
    setbp1_unadj_prev = unadj_results["single_gene_prevalence"]["SETBP1"]
    setbp1_adj_prev = adj_results["single_gene_prevalence"]["SETBP1"]
    print(f"\n  1. SETBP1 prevalence: {setbp1_unadj_prev:.2f}% (unadjusted) -> {setbp1_adj_prev:.2f}% (adjusted)")
    print(f"     Change: {setbp1_adj_prev - setbp1_unadj_prev:+.2f} percentage points")
    print(f"     The unadjusted denominator diluted SETBP1 prevalence by including")
    print(f"     {len(all_myeloid) - len(adjusted_myeloid):,} samples that were never tested for SETBP1.")

    # SETBP1-involving pairs
    print(f"\n  2. SETBP1-involving pair statistics (adjusted vs unadjusted OR):")
    for g in ["DNMT3A", "IDH2", "PTPN11"]:
        key = "+".join(sorted([g, "SETBP1"], key=TARGET_GENES.index))
        ou = unadj_results["pairwise_fisher_exact"][key]["odds_ratio"]
        oa = adj_results["pairwise_fisher_exact"][key]["odds_ratio"]
        print(f"     {key}: OR {ou:.3f} -> {oa:.3f} ({oa - ou:+.3f})")

    pop_a = adj_results["population_estimate"]
    print(f"\n  3. Population frequency upper bound (panel-adjusted):")
    print(f"     0/{len(adjusted_myeloid):,} quadruple carriers")
    print(f"     95% CI upper bound: {pop_a['upper_95_frequency']}")
    print(f"     99% CI upper bound: {pop_a['upper_99_frequency']}")

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "analysis": "Panel-adjusted co-occurrence statistics -- GENIE v19.0",
        "methodology": {
            "problem": "SETBP1 covered by only {}/{} panels; unadjusted denominator dilutes statistics".format(
                len(covering), len(panels)
            ),
            "solution": "Restrict to {} panels covering all 4 target genes".format(len(covering)),
            "filter": "Protein-altering variants only",
            "variant_classes": sorted(PROTEIN_ALTERING),
        },
        "panel_coverage": {
            "total_panels": len(panels),
            "panels_covering_all_4": len(covering),
            "covering_panel_ids": sorted(covering),
            "per_gene_coverage": {
                g: sum(1 for p, genes in panels.items() if g in genes)
                for g in TARGET_GENES
            },
        },
        "sample_counts": {
            "total_genie_samples": len(samples),
            "all_myeloid": len(all_myeloid),
            "adjusted_myeloid": len(adjusted_myeloid),
            "excluded_by_panel_filter": len(all_myeloid) - len(adjusted_myeloid),
        },
        "unadjusted": unadj_results,
        "adjusted": adj_results,
    }

    output_path = RESULTS_DIR / "panel_adjusted_statistics.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
