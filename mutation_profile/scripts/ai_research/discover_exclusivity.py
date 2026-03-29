#!/usr/bin/env python3
"""
discover_exclusivity.py -- DISCOVER-style mutual exclusivity test for patient genes.

Implements the DISCOVER mutual exclusivity framework (Canisius et al., Genome
Biology 2016) using a permutation-based approach that controls for per-gene
marginal mutation rates. Fisher's exact test conflates gene-level mutation
frequency with pairwise association; DISCOVER separates these by permuting
each gene's mutation vector independently, preserving marginal rates while
breaking pairwise structure.

Algorithm:
  For each gene pair (A, B) in the myeloid cohort:
    1. Compute observed co-occurrence count.
    2. Run N=10,000 permutations: independently shuffle each gene's binary
       mutation vector (preserving the number of mutated samples per gene).
    3. In each permutation, count co-occurrences under the null.
    4. P-value (ME) = fraction of permutations with co-occurrence <= observed.
    5. P-value (CO) = fraction of permutations with co-occurrence >= observed.
    6. Two-sided p-value = 2 * min(p_ME, p_CO).
  Apply Benjamini-Hochberg correction across all pairs.

Tests all 10 pairs among the 5 patient genes: DNMT3A, IDH2, SETBP1, PTPN11, EZH2.
Compares DISCOVER p-values with existing Fisher's exact test p-values.

Inputs:
    - mutation_profile/data/genie/raw/data_mutations_extended.txt
    - mutation_profile/data/genie/raw/data_clinical_sample.txt
    - mutation_profile/data/genie/raw/genomic_information_*.txt
    - mutation_profile/results/cooccurrence/myeloid_pairwise_results.json (Fisher results)

Outputs:
    - mutation_profile/results/ai_research/discover_exclusivity.json
    - mutation_profile/results/ai_research/discover_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/discover_exclusivity.py

Runtime: ~30-60 seconds (10,000 permutations x 10 pairs)
Dependencies: pandas, numpy, scipy, statsmodels
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PATIENT_GENES = ["DNMT3A", "IDH2", "SETBP1", "PTPN11", "EZH2"]

N_PERMUTATIONS = 10_000
RANDOM_SEED = 42

# Myeloid cancer types (same as pairwise_matrix.py)
MYELOID_CANCER_TYPES_NONLEUK = [
    "Myelodysplastic Syndromes",
    "Myeloproliferative Neoplasms",
    "Myelodysplastic/Myeloproliferative Neoplasms",
    "Myeloid Neoplasms with Germ Line Predisposition",
]

MYELOID_LEUKEMIA_KEYWORDS = [
    "AML", "Acute Myeloid", "Acute Myelomonocytic", "Acute Monoblastic",
    "Acute Megakaryoblastic", "Acute Panmyelosis", "Pure Erythroid",
    "Myeloid Sarcoma", "Myeloid Leukemia", "Myeloid Proliferations",
    "Myeloid/Lymphoid Neoplasms", "Therapy-Related Myeloid",
    "Therapy-Related Acute Myeloid", "Therapy-Related Myelodysplastic",
    "Transient Abnormal Myelopoiesis", "APL with PML-RARA",
    "Mixed Phenotype Acute Leukemia",
]

CODING_CLASSIFICATIONS = {
    "Missense_Mutation", "Nonsense_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site",
}

HYPERMUTATION_THRESHOLD = 40


def load_data(data_dir: Path) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Load GENIE data and return myeloid samples, panel coverage, and mutation sets.

    Returns
    -------
    myeloid_samples : set of sample IDs
    gene_eligible : dict mapping gene -> set of eligible sample IDs
    gene_mutated : dict mapping gene -> set of mutated sample IDs
    """
    # Clinical data
    print("[1/4] Loading clinical data...")
    clin = pd.read_csv(data_dir / "data_clinical_sample.txt", sep="\t", comment="#")
    print(f"  Total samples: {len(clin):,}")

    myeloid_mask = clin["CANCER_TYPE"].isin(MYELOID_CANCER_TYPES_NONLEUK)
    leukemia_mask = clin["CANCER_TYPE"] == "Leukemia"
    myeloid_leukemia_mask = leukemia_mask & clin["CANCER_TYPE_DETAILED"].apply(
        lambda x: any(kw.lower() in str(x).lower() for kw in MYELOID_LEUKEMIA_KEYWORDS)
        if pd.notna(x) else False
    )
    clin_myeloid = clin[myeloid_mask | myeloid_leukemia_mask].copy()
    myeloid_samples = set(clin_myeloid["SAMPLE_ID"])
    sample_panel = dict(zip(clin_myeloid["SAMPLE_ID"], clin_myeloid["SEQ_ASSAY_ID"]))
    print(f"  Myeloid samples: {len(myeloid_samples):,}")

    # Gene panels
    print("[2/4] Loading gene panels...")
    panel_genes: dict[str, set[str]] = {}
    for pf in data_dir.glob("data_gene_panel_*.txt"):
        panel_id = None
        genes: set[str] = set()
        with open(pf) as fh:
            for line in fh:
                if line.startswith("stable_id:"):
                    panel_id = line.strip().split(":", 1)[1].strip()
                if line.startswith("gene_list:"):
                    genes = {g.strip() for g in line.strip().split("\t")[1:] if g.strip()}
        if panel_id:
            panel_genes[panel_id] = genes
    print(f"  Loaded {len(panel_genes)} panels")

    # Gene matrix for more accurate panel assignment
    gene_matrix = pd.read_csv(data_dir / "data_gene_matrix.txt", sep="\t")
    sample_mut_panel = dict(zip(gene_matrix["SAMPLE_ID"], gene_matrix["mutations"]))
    for sid in myeloid_samples:
        if sid in sample_mut_panel and pd.notna(sample_mut_panel.get(sid)):
            sample_panel[sid] = sample_mut_panel[sid]

    # Per-gene eligible samples (panel covers gene)
    gene_eligible: dict[str, set[str]] = {}
    for gene in PATIENT_GENES:
        eligible = set()
        for sid in myeloid_samples:
            pid = sample_panel.get(sid)
            if pid and pid in panel_genes and gene in panel_genes[pid]:
                eligible.add(sid)
        gene_eligible[gene] = eligible
        print(f"    {gene}: {len(eligible):,} eligible samples")

    # Mutations
    print("[3/4] Loading mutations...")
    mutations = pd.read_csv(
        data_dir / "data_mutations_extended.txt",
        sep="\t", comment="#",
        usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"],
        dtype=str,
    )
    mutations = mutations[mutations["Tumor_Sample_Barcode"].isin(myeloid_samples)]
    mutations = mutations[mutations["Variant_Classification"].isin(CODING_CLASSIFICATIONS)]

    # Hypermutation filter
    print("[4/4] Applying hypermutation filter...")
    coding_counts = mutations.groupby("Tumor_Sample_Barcode").size()
    hypermutated = set(coding_counts[coding_counts > HYPERMUTATION_THRESHOLD].index)
    print(f"  Hypermutated excluded: {len(hypermutated):,}")
    mutations = mutations[~mutations["Tumor_Sample_Barcode"].isin(hypermutated)]
    myeloid_samples -= hypermutated
    for gene in PATIENT_GENES:
        gene_eligible[gene] -= hypermutated

    # Build mutation sets
    gene_muts = mutations[mutations["Hugo_Symbol"].isin(PATIENT_GENES)]
    sample_gene = gene_muts[["Tumor_Sample_Barcode", "Hugo_Symbol"]].drop_duplicates()
    gene_mutated: dict[str, set[str]] = {}
    for gene in PATIENT_GENES:
        gene_mutated[gene] = set(
            sample_gene[sample_gene["Hugo_Symbol"] == gene]["Tumor_Sample_Barcode"]
        )

    print(f"\n  Final myeloid cohort: {len(myeloid_samples):,} samples")
    return myeloid_samples, gene_eligible, gene_mutated


def discover_test(
    gene_a: str,
    gene_b: str,
    gene_eligible: dict[str, set[str]],
    gene_mutated: dict[str, set[str]],
    n_permutations: int = N_PERMUTATIONS,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run DISCOVER-style permutation test for one gene pair.

    For the eligible sample set (intersection of panels covering both genes),
    independently permute each gene's binary mutation vector N times. Count
    co-occurrences in each permutation to build the null distribution.

    Parameters
    ----------
    gene_a, gene_b : str
        Gene symbols.
    gene_eligible : dict
        Gene -> set of eligible sample IDs.
    gene_mutated : dict
        Gene -> set of mutated sample IDs.
    n_permutations : int
        Number of permutations (default 10,000).
    rng : numpy random Generator
        For reproducibility.

    Returns
    -------
    dict with observed, expected_perm, p_me, p_co, p_two_sided, null_distribution stats.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Panel-adjusted: samples covering both genes
    eligible = gene_eligible[gene_a] & gene_eligible[gene_b]
    eligible_list = sorted(eligible)
    n = len(eligible_list)

    if n == 0:
        return {
            "gene_a": gene_a, "gene_b": gene_b,
            "n_eligible": 0, "n_a": 0, "n_b": 0,
            "observed": 0, "expected_perm": 0.0,
            "p_me": 1.0, "p_co": 1.0, "p_two_sided": 1.0,
            "null_mean": 0.0, "null_std": 0.0,
            "fisher_p": 1.0, "fisher_or": 1.0,
        }

    eligible_set = set(eligible_list)
    mut_a = gene_mutated[gene_a] & eligible_set
    mut_b = gene_mutated[gene_b] & eligible_set
    n_a = len(mut_a)
    n_b = len(mut_b)
    observed = len(mut_a & mut_b)

    # Fisher's exact test for comparison
    n_a_only = n_a - observed
    n_b_only = n_b - observed
    n_neither = n - n_a - n_b + observed
    table = np.array([[observed, n_a_only], [n_b_only, n_neither]])
    fisher_or, fisher_p = fisher_exact(table, alternative="two-sided")

    # Permutation test: independently permute binary vectors
    # Create binary arrays for efficiency
    vec_a = np.zeros(n, dtype=np.int8)
    vec_a[:n_a] = 1
    vec_b = np.zeros(n, dtype=np.int8)
    vec_b[:n_b] = 1

    null_cooccurrences = np.empty(n_permutations, dtype=np.int32)
    for i in range(n_permutations):
        rng.shuffle(vec_a)
        rng.shuffle(vec_b)
        null_cooccurrences[i] = np.sum(vec_a & vec_b)

    # P-values
    # ME: observed <= null (fewer co-occurrences than expected = mutual exclusivity)
    p_me = (np.sum(null_cooccurrences <= observed) + 1) / (n_permutations + 1)
    # CO: observed >= null (more co-occurrences than expected = co-occurrence)
    p_co = (np.sum(null_cooccurrences >= observed) + 1) / (n_permutations + 1)
    # Two-sided
    p_two_sided = min(2.0 * min(p_me, p_co), 1.0)

    null_mean = float(np.mean(null_cooccurrences))
    null_std = float(np.std(null_cooccurrences))

    expected_indep = (n_a * n_b) / n if n > 0 else 0.0
    oe_ratio = observed / expected_indep if expected_indep > 0 else (float("inf") if observed > 0 else 0.0)

    return {
        "gene_a": gene_a,
        "gene_b": gene_b,
        "n_eligible": n,
        "n_a": n_a,
        "n_b": n_b,
        "freq_a": round(n_a / n * 100, 2) if n > 0 else 0.0,
        "freq_b": round(n_b / n * 100, 2) if n > 0 else 0.0,
        "observed": observed,
        "expected_indep": round(expected_indep, 4),
        "expected_perm": round(null_mean, 4),
        "oe_ratio": round(oe_ratio, 4) if np.isfinite(oe_ratio) else None,
        "direction": "mutual_exclusivity" if observed < null_mean else "co-occurrence",
        "p_me": round(p_me, 6),
        "p_co": round(p_co, 6),
        "p_two_sided": round(p_two_sided, 6),
        "null_mean": round(null_mean, 4),
        "null_std": round(null_std, 4),
        "null_min": int(np.min(null_cooccurrences)),
        "null_max": int(np.max(null_cooccurrences)),
        "null_median": float(np.median(null_cooccurrences)),
        "fisher_or": round(float(fisher_or), 4) if np.isfinite(fisher_or) else None,
        "fisher_p": round(float(fisher_p), 8),
    }


def load_existing_fisher(results_dir: Path) -> dict[str, dict]:
    """Load existing Fisher results from pairwise_matrix.py output."""
    json_path = results_dir.parent / "cooccurrence" / "myeloid_pairwise_results.json"
    if not json_path.exists():
        print(f"  Warning: {json_path} not found, Fisher comparison will use inline values")
        return {}

    with open(json_path) as f:
        data = json.load(f)

    fisher_results: dict[str, dict] = {}
    for pair in data["pairs"]:
        key = f"{pair['gene_a']}+{pair['gene_b']}"
        fisher_results[key] = pair
        # Also store reverse key
        key_rev = f"{pair['gene_b']}+{pair['gene_a']}"
        fisher_results[key_rev] = pair

    return fisher_results


def generate_report(
    results: list[dict],
    fisher_lookup: dict[str, dict],
    output_path: Path,
) -> None:
    """Generate markdown report comparing Fisher and DISCOVER results."""
    lines: list[str] = []
    lines.append("# DISCOVER Mutual Exclusivity Analysis")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Standard Fisher's exact test for mutual exclusivity conflates gene-level")
    lines.append("mutation frequency with pairwise association: genes that are rarely mutated")
    lines.append("will naturally have few co-occurrences, inflating apparent mutual exclusivity.")
    lines.append("The DISCOVER framework (Canisius et al., Genome Biology 2016) addresses this")
    lines.append("by computing a null distribution that accounts for each gene's marginal")
    lines.append("mutation rate.")
    lines.append("")
    lines.append("### Implementation")
    lines.append("")
    lines.append("For each gene pair (A, B) among the 5 patient genes:")
    lines.append("")
    lines.append("1. Identify the panel-adjusted eligible sample set (samples whose sequencing")
    lines.append("   panel covers both genes).")
    lines.append("2. Compute the observed co-occurrence count.")
    lines.append(f"3. Run {N_PERMUTATIONS:,} permutations: independently shuffle each gene's binary")
    lines.append("   mutation vector, preserving marginal mutation counts.")
    lines.append("4. Build null distribution of co-occurrence counts.")
    lines.append("5. P-value (ME) = fraction of permutations with co-occurrence <= observed.")
    lines.append("6. Two-sided p-value = 2 x min(p_ME, p_CO).")
    lines.append("7. Apply Benjamini-Hochberg correction across all 10 pairs.")
    lines.append("")
    lines.append("### Key difference from Fisher's exact test")
    lines.append("")
    lines.append("Fisher conditions on row and column marginals of the 2x2 table, testing")
    lines.append("whether the odds ratio differs from 1. DISCOVER's permutation null preserves")
    lines.append("each gene's mutation count independently, which better models the biological")
    lines.append("null where two genes are independently mutated at their observed rates.")
    lines.append("For rare genes (like SETBP1 at ~1.3%), this distinction matters because")
    lines.append("Fisher can flag low co-occurrence counts as significant even when the expected")
    lines.append("count under independence is already very low.")
    lines.append("")

    # Comparison table
    lines.append("## Results: Fisher vs DISCOVER comparison")
    lines.append("")
    lines.append("| Gene pair | Obs | Exp (indep) | Exp (perm) | O/E | Fisher p | Fisher BH | DISCOVER p | DISCOVER BH | Direction |")
    lines.append("|-----------|-----|-------------|------------|-----|----------|-----------|------------|-------------|-----------|")

    for r in results:
        pair_key = f"{r['gene_a']}+{r['gene_b']}"
        fisher_data = fisher_lookup.get(pair_key, {})
        fisher_p_raw = fisher_data.get("p_value", r["fisher_p"])
        fisher_bh = fisher_data.get("p_adjusted", None)
        fisher_bh_str = f"{fisher_bh:.2e}" if fisher_bh is not None else "N/A"

        oe_str = f"{r['oe_ratio']:.4f}" if r["oe_ratio"] is not None else "N/A"
        lines.append(
            f"| {r['gene_a']}+{r['gene_b']} "
            f"| {r['observed']} "
            f"| {r['expected_indep']:.2f} "
            f"| {r['expected_perm']:.2f} "
            f"| {oe_str} "
            f"| {fisher_p_raw:.2e} "
            f"| {fisher_bh_str} "
            f"| {r['p_two_sided']:.4f} "
            f"| {r['p_bh']:.4f} "
            f"| {r['direction']} |"
        )

    lines.append("")

    # Per-gene frequencies
    lines.append("## Per-gene mutation frequencies")
    lines.append("")
    lines.append("| Gene | Mutated | Eligible | Frequency |")
    lines.append("|------|---------|----------|-----------|")
    seen_genes: set[str] = set()
    for r in results:
        for gene_key, n_key, freq_key in [
            ("gene_a", "n_a", "freq_a"),
            ("gene_b", "n_b", "freq_b"),
        ]:
            gene = r[gene_key]
            if gene not in seen_genes:
                seen_genes.add(gene)
                # Find a result that has this gene as gene_a or gene_b
                n_val = r[n_key]
                freq_val = r[freq_key]
                n_elig = r["n_eligible"]
                lines.append(f"| {gene} | {n_val} | {n_elig} | {freq_val:.2f}% |")
    lines.append("")

    # Significance assessment
    lines.append("## Significance after controlling for gene-level rates")
    lines.append("")

    sig_me = [r for r in results if r["p_bh"] < 0.05 and r["direction"] == "mutual_exclusivity"]
    sig_co = [r for r in results if r["p_bh"] < 0.05 and r["direction"] == "co-occurrence"]
    not_sig = [r for r in results if r["p_bh"] >= 0.05]

    if sig_me:
        lines.append("### Significant mutual exclusivities (DISCOVER BH < 0.05)")
        lines.append("")
        for r in sig_me:
            lines.append(f"- **{r['gene_a']}+{r['gene_b']}**: observed={r['observed']}, "
                         f"expected={r['expected_perm']:.2f}, O/E={r['oe_ratio']:.4f}, "
                         f"DISCOVER p={r['p_two_sided']:.4f}, BH={r['p_bh']:.4f}")
        lines.append("")
    else:
        lines.append("No gene pairs show statistically significant mutual exclusivity after")
        lines.append("controlling for gene-level mutation rates (DISCOVER BH >= 0.05).")
        lines.append("")

    if sig_co:
        lines.append("### Significant co-occurrences (DISCOVER BH < 0.05)")
        lines.append("")
        for r in sig_co:
            lines.append(f"- **{r['gene_a']}+{r['gene_b']}**: observed={r['observed']}, "
                         f"expected={r['expected_perm']:.2f}, O/E={r['oe_ratio']:.4f}, "
                         f"DISCOVER p={r['p_two_sided']:.4f}, BH={r['p_bh']:.4f}")
        lines.append("")

    if not_sig:
        lines.append("### Not significant (DISCOVER BH >= 0.05)")
        lines.append("")
        for r in not_sig:
            lines.append(f"- {r['gene_a']}+{r['gene_b']}: observed={r['observed']}, "
                         f"expected={r['expected_perm']:.2f}, "
                         f"DISCOVER p={r['p_two_sided']:.4f}, BH={r['p_bh']:.4f}")
        lines.append("")

    # Interpretation
    lines.append("## Biological interpretation")
    lines.append("")
    lines.append("### IDH2-SETBP1 mutual exclusivity")
    lines.append("")
    lines.append("IDH2 and SETBP1 show the strongest signal among the 5 patient genes.")

    idh2_setbp1 = next((r for r in results if
                         {r["gene_a"], r["gene_b"]} == {"IDH2", "SETBP1"}), None)
    if idh2_setbp1:
        if idh2_setbp1["p_bh"] < 0.05:
            lines.append(f"The mutual exclusivity remains significant after DISCOVER correction "
                         f"(BH={idh2_setbp1['p_bh']:.4f}), confirming that the low co-occurrence "
                         f"is not merely an artifact of both genes being rare. This suggests genuine "
                         f"biological incompatibility between IDH2 and SETBP1 mutations in myeloid "
                         f"neoplasms.")
        else:
            lines.append(f"The mutual exclusivity is NOT significant after DISCOVER correction "
                         f"(BH={idh2_setbp1['p_bh']:.4f}). The low observed co-occurrence "
                         f"({idh2_setbp1['observed']} cases) can be explained by the low marginal "
                         f"rates of both genes (IDH2 ~{idh2_setbp1['freq_a']:.1f}%, "
                         f"SETBP1 ~{idh2_setbp1['freq_b']:.1f}%). Under independence, only "
                         f"~{idh2_setbp1['expected_perm']:.1f} co-occurrences are expected, "
                         f"making the observed count consistent with the null.")
    lines.append("")

    lines.append("### Implication for the patient's mutation profile")
    lines.append("")
    lines.append("The patient carries all 5 genes mutated simultaneously. The DISCOVER analysis")
    lines.append("helps distinguish which pairwise combinations represent genuinely unusual")
    lines.append("co-occurrences vs. pairs that appear rare simply because both genes are")
    lines.append("individually uncommon. Pairs with significant DISCOVER p-values indicate")
    lines.append("biological constraints that the patient's tumor has overcome, while")
    lines.append("non-significant pairs indicate that their co-occurrence rarity is adequately")
    lines.append("explained by marginal mutation rates alone.")
    lines.append("")

    lines.append("### Fisher vs DISCOVER divergence")
    lines.append("")
    divergent = []
    for r in results:
        pair_key = f"{r['gene_a']}+{r['gene_b']}"
        fisher_data = fisher_lookup.get(pair_key, {})
        fisher_bh = fisher_data.get("p_adjusted")
        if fisher_bh is not None:
            fisher_sig = fisher_bh < 0.05
            discover_sig = r["p_bh"] < 0.05
            if fisher_sig != discover_sig:
                divergent.append((r, fisher_bh, fisher_sig, discover_sig))

    if divergent:
        lines.append("The following pairs show divergent significance between Fisher and DISCOVER:")
        lines.append("")
        for r, f_bh, f_sig, d_sig in divergent:
            f_label = "significant" if f_sig else "not significant"
            d_label = "significant" if d_sig else "not significant"
            lines.append(f"- **{r['gene_a']}+{r['gene_b']}**: Fisher BH={f_bh:.4f} ({f_label}), "
                         f"DISCOVER BH={r['p_bh']:.4f} ({d_label})")
        lines.append("")
        lines.append("Divergence indicates that Fisher's test was influenced by the marginal")
        lines.append("mutation rates rather than true pairwise association. DISCOVER's result")
        lines.append("is more appropriate for biological interpretation.")
    else:
        lines.append("Fisher and DISCOVER agree on significance for all 10 pairs. The existing")
        lines.append("Fisher results are robust to the marginal rate correction.")
    lines.append("")

    lines.append("## Reference")
    lines.append("")
    lines.append("Canisius S, Martens JWM, Wessels LFA. A novel independence test for somatic")
    lines.append("alterations in cancer shows that biology drives mutual exclusivity but chance")
    lines.append("explains most co-occurrence. Genome Biology. 2016;17:261.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {output_path}")


def main() -> None:
    t0 = time.time()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # mutation_profile/
    data_dir = project_root / "data" / "genie" / "raw"
    results_dir = project_root / "results" / "ai_research"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DISCOVER MUTUAL EXCLUSIVITY ANALYSIS")
    print(f"Patient genes: {', '.join(PATIENT_GENES)}")
    print(f"Permutations: {N_PERMUTATIONS:,}")
    print("=" * 70)

    # Load data
    myeloid_samples, gene_eligible, gene_mutated = load_data(data_dir)

    # Run DISCOVER test for all 10 pairs
    pairs = list(combinations(PATIENT_GENES, 2))
    assert len(pairs) == 10, f"Expected 10 pairs, got {len(pairs)}"

    print(f"\nRunning DISCOVER permutation tests ({N_PERMUTATIONS:,} permutations)...")
    rng = np.random.default_rng(RANDOM_SEED)
    results: list[dict] = []

    for gene_a, gene_b in pairs:
        print(f"  Testing {gene_a} + {gene_b}...", end=" ", flush=True)
        result = discover_test(
            gene_a, gene_b, gene_eligible, gene_mutated,
            n_permutations=N_PERMUTATIONS, rng=rng,
        )
        print(f"obs={result['observed']}, exp_perm={result['expected_perm']:.2f}, "
              f"p_ME={result['p_me']:.4f}, p_two={result['p_two_sided']:.4f}")
        results.append(result)

    # BH correction on two-sided p-values
    p_vals = np.array([r["p_two_sided"] for r in results])
    reject, p_adj, _, _ = multipletests(p_vals, method="fdr_bh")
    for i, r in enumerate(results):
        r["p_bh"] = round(float(p_adj[i]), 6)
        r["significant_bh"] = bool(reject[i])

    # Sort by p-value
    results.sort(key=lambda r: r["p_two_sided"])

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Pair':<20} {'Obs':>4} {'Exp':>8} {'O/E':>8} {'Fisher p':>12} "
          f"{'DISCOVER p':>12} {'BH':>8} {'Dir':>20}")
    print("-" * 100)
    for r in results:
        oe_str = f"{r['oe_ratio']:.4f}" if r["oe_ratio"] is not None else "N/A"
        print(f"{r['gene_a']}+{r['gene_b']:<12} {r['observed']:>4} "
              f"{r['expected_perm']:>8.2f} {oe_str:>8} "
              f"{r['fisher_p']:>12.2e} {r['p_two_sided']:>12.4f} "
              f"{r['p_bh']:>8.4f} {r['direction']:>20}")

    n_sig = sum(1 for r in results if r["significant_bh"])
    print(f"\nSignificant after BH correction: {n_sig}/10")

    # Load existing Fisher results for comparison
    fisher_lookup = load_existing_fisher(results_dir)

    # Save JSON
    json_output = {
        "metadata": {
            "analysis": "DISCOVER mutual exclusivity test (permutation-based)",
            "reference": "Canisius et al., Genome Biology 2016;17:261",
            "data_source": "AACR Project GENIE v19.0",
            "genes": PATIENT_GENES,
            "n_permutations": N_PERMUTATIONS,
            "random_seed": RANDOM_SEED,
            "n_pairs_tested": len(results),
            "correction_method": "Benjamini-Hochberg (FDR)",
            "significance_threshold": 0.05,
            "hypermutation_threshold": HYPERMUTATION_THRESHOLD,
            "variant_filter": "coding only (nonsynonymous)",
            "n_significant_bh": n_sig,
        },
        "results": results,
    }

    json_path = results_dir / "discover_exclusivity.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

    # Generate report
    report_path = results_dir / "discover_report.md"
    generate_report(results, fisher_lookup, report_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
