#!/usr/bin/env python3
"""
pairwise_matrix.py -- Pairwise co-occurrence matrix for top 20 myeloid driver genes.

Computes Fisher's exact test for all 190 gene pairs with:
  - Myeloid neoplasm filtering (AML, MDS, MPN, MDS/MPN, CMML)
  - Coding mutation filtering (nonsynonymous variants only)
  - Hypermutation filtering (>40 coding mutations per sample)
  - Panel-aware denominators (only count samples whose panel covers both genes)
  - Benjamini-Hochberg correction across all 190 tests

Inputs:
    - mutation_profile/data/genie/raw/data_mutations_extended.txt
    - mutation_profile/data/genie/raw/data_clinical_sample.txt
    - mutation_profile/data/genie/raw/genomic_information_*.txt

Outputs:
    - mutation_profile/results/cooccurrence/myeloid_pairwise_matrix.tsv
    - mutation_profile/results/cooccurrence/myeloid_pairwise_results.tsv
    - mutation_profile/results/cooccurrence/myeloid_pairwise_results.json

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/pairwise_matrix.py

Runtime: ~25 seconds
Dependencies: pandas, numpy, scipy, statsmodels
"""

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
TOP20_GENES = [
    "DNMT3A", "TET2", "ASXL1", "SRSF2", "SF3B1",
    "RUNX1", "TP53", "FLT3", "NPM1", "NRAS",
    "KRAS", "IDH1", "IDH2", "STAG2", "EZH2",
    "U2AF1", "BCOR", "PTPN11", "SETBP1", "CBL",
]

# Myeloid cancer types in GENIE CANCER_TYPE field
MYELOID_CANCER_TYPES = [
    "Leukemia",                                    # includes all AML subtypes
    "Myelodysplastic Syndromes",                   # MDS
    "Myeloproliferative Neoplasms",                # MPN (PV, ET, PMF)
    "Myelodysplastic/Myeloproliferative Neoplasms",# MDS/MPN (CMML, aCML, etc.)
    "Myeloid Neoplasms with Germ Line Predisposition",
]

# Keep only the myeloid-lineage detailed subtypes within "Leukemia"
# (exclude B-ALL, T-ALL, ambiguous lineage that are lymphoid)
MYELOID_LEUKEMIA_KEYWORDS = [
    "AML", "Acute Myeloid", "Acute Myelomonocytic", "Acute Monoblastic",
    "Acute Megakaryoblastic", "Acute Panmyelosis", "Pure Erythroid",
    "Myeloid Sarcoma", "Myeloid Leukemia", "Myeloid Proliferations",
    "Myeloid/Lymphoid Neoplasms", "Therapy-Related Myeloid",
    "Therapy-Related Acute Myeloid", "Therapy-Related Myelodysplastic",
    "Transient Abnormal Myelopoiesis", "APL with PML-RARA",
    "Mixed Phenotype Acute Leukemia",  # include MPAL (has myeloid component)
]

# Coding (nonsynonymous) variant classifications
CODING_CLASSIFICATIONS = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Splice_Site",
    "Nonstop_Mutation",
    "Translation_Start_Site",
}

HYPERMUTATION_THRESHOLD = 40  # max coding mutations per sample


def main():
    t0 = time.time()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # mutation_profile/
    data_dir = project_root / "data" / "genie" / "raw"
    results_dir = project_root / "results" / "cooccurrence"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PAIRWISE CO-OCCURRENCE ANALYSIS — TOP 20 MYELOID DRIVER GENES")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load clinical sample data and filter to myeloid neoplasms
    # ------------------------------------------------------------------
    print("\n[1/7] Loading clinical sample data...")
    clin = pd.read_csv(data_dir / "data_clinical_sample.txt", sep="\t", comment="#")
    print(f"  Total samples in GENIE: {len(clin):,}")

    # Non-leukemia myeloid types: keep all samples
    myeloid_mask = clin["CANCER_TYPE"].isin([
        "Myelodysplastic Syndromes",
        "Myeloproliferative Neoplasms",
        "Myelodysplastic/Myeloproliferative Neoplasms",
        "Myeloid Neoplasms with Germ Line Predisposition",
    ])

    # Leukemia type: filter to myeloid-lineage subtypes only
    leukemia_mask = clin["CANCER_TYPE"] == "Leukemia"
    myeloid_leukemia_mask = leukemia_mask & clin["CANCER_TYPE_DETAILED"].apply(
        lambda x: any(kw.lower() in str(x).lower() for kw in MYELOID_LEUKEMIA_KEYWORDS)
        if pd.notna(x) else False
    )

    clin_myeloid = clin[myeloid_mask | myeloid_leukemia_mask].copy()
    myeloid_samples = set(clin_myeloid["SAMPLE_ID"])
    print(f"  Myeloid neoplasm samples: {len(myeloid_samples):,}")

    # Build sample->panel lookup
    sample_panel = dict(zip(clin_myeloid["SAMPLE_ID"], clin_myeloid["SEQ_ASSAY_ID"]))

    # ------------------------------------------------------------------
    # 2. Load gene panel definitions
    # ------------------------------------------------------------------
    print("\n[2/7] Loading gene panel definitions...")
    panel_genes = {}  # panel_id -> set of genes
    panel_files = list(data_dir.glob("data_gene_panel_*.txt"))
    for pf in panel_files:
        with open(pf) as fh:
            for line in fh:
                if line.startswith("gene_list:"):
                    panel_id = None
                    # re-read to get stable_id
                    break
            fh.seek(0)
            panel_id = None
            for line in fh:
                if line.startswith("stable_id:"):
                    panel_id = line.strip().split(":")[1].strip()
                if line.startswith("gene_list:"):
                    genes = set(line.strip().split("\t")[1:])
                    if panel_id:
                        panel_genes[panel_id] = genes
    print(f"  Loaded {len(panel_genes)} gene panels")

    # Also load gene_matrix for sample-level panel assignment (more accurate)
    gene_matrix = pd.read_csv(data_dir / "data_gene_matrix.txt", sep="\t")
    # gene_matrix has columns: SAMPLE_ID, mutations, cna, sv
    # The 'mutations' column gives the panel used for mutation calling
    sample_mut_panel = dict(zip(gene_matrix["SAMPLE_ID"], gene_matrix["mutations"]))

    # Use gene_matrix panel if available, fall back to clinical SEQ_ASSAY_ID
    for sid in myeloid_samples:
        if sid in sample_mut_panel and pd.notna(sample_mut_panel.get(sid)):
            sample_panel[sid] = sample_mut_panel[sid]

    # Precompute: for each gene, which myeloid samples have a panel covering it
    gene_eligible_samples = {}
    for gene in TOP20_GENES:
        eligible = set()
        for sid in myeloid_samples:
            pid = sample_panel.get(sid)
            if pid and pid in panel_genes and gene in panel_genes[pid]:
                eligible.add(sid)
        gene_eligible_samples[gene] = eligible
    for g in TOP20_GENES:
        print(f"    {g}: {len(gene_eligible_samples[g]):,} samples with panel coverage")

    # ------------------------------------------------------------------
    # 3. Load mutations, filter to myeloid + coding
    # ------------------------------------------------------------------
    print("\n[3/7] Loading mutation data (this may take a minute)...")
    mut_cols = ["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"]
    mutations = pd.read_csv(
        data_dir / "data_mutations_extended.txt",
        sep="\t",
        comment="#",
        usecols=mut_cols,
        dtype=str,
    )
    print(f"  Total mutations loaded: {len(mutations):,}")

    # Filter to myeloid samples
    mutations = mutations[mutations["Tumor_Sample_Barcode"].isin(myeloid_samples)]
    print(f"  Mutations in myeloid samples: {len(mutations):,}")

    # Filter to coding variants
    mutations = mutations[mutations["Variant_Classification"].isin(CODING_CLASSIFICATIONS)]
    print(f"  Coding mutations: {len(mutations):,}")

    # ------------------------------------------------------------------
    # 4. Hypermutation filter
    # ------------------------------------------------------------------
    print("\n[4/7] Applying hypermutation filter...")
    coding_counts = mutations.groupby("Tumor_Sample_Barcode").size()
    hypermutated = set(coding_counts[coding_counts > HYPERMUTATION_THRESHOLD].index)
    print(f"  Hypermutated samples (>{HYPERMUTATION_THRESHOLD} coding mutations): {len(hypermutated):,}")

    mutations = mutations[~mutations["Tumor_Sample_Barcode"].isin(hypermutated)]
    # Also remove hypermutated from eligible sets
    myeloid_samples -= hypermutated
    for gene in TOP20_GENES:
        gene_eligible_samples[gene] -= hypermutated
    print(f"  Remaining myeloid samples: {len(myeloid_samples):,}")
    print(f"  Remaining coding mutations: {len(mutations):,}")

    # ------------------------------------------------------------------
    # 5. Build per-sample mutation profile for top 20 genes
    # ------------------------------------------------------------------
    print("\n[5/7] Building per-sample mutation profiles...")
    # A sample is "mutated" for a gene if it has at least one coding mutation
    gene_mutations = mutations[mutations["Hugo_Symbol"].isin(TOP20_GENES)]
    # Deduplicate: one entry per (sample, gene)
    sample_gene = gene_mutations[["Tumor_Sample_Barcode", "Hugo_Symbol"]].drop_duplicates()
    # Build set of mutated samples per gene
    gene_mutated_samples = {}
    for gene in TOP20_GENES:
        gene_mutated_samples[gene] = set(
            sample_gene[sample_gene["Hugo_Symbol"] == gene]["Tumor_Sample_Barcode"]
        )

    print("  Per-gene mutation counts (panel-adjusted):")
    for g in TOP20_GENES:
        n_mut = len(gene_mutated_samples[g] & gene_eligible_samples[g])
        n_elig = len(gene_eligible_samples[g])
        freq = n_mut / n_elig * 100 if n_elig > 0 else 0
        print(f"    {g}: {n_mut:,}/{n_elig:,} ({freq:.1f}%)")

    # ------------------------------------------------------------------
    # 6. Pairwise Fisher's exact test for all 190 pairs
    # ------------------------------------------------------------------
    print("\n[6/7] Running pairwise Fisher's exact tests (190 pairs)...")
    pairs = list(combinations(TOP20_GENES, 2))
    assert len(pairs) == 190, f"Expected 190 pairs, got {len(pairs)}"

    results = []
    for gene_a, gene_b in pairs:
        # Panel adjustment: only samples whose panel covers BOTH genes
        eligible = gene_eligible_samples[gene_a] & gene_eligible_samples[gene_b]
        n_total = len(eligible)
        if n_total == 0:
            results.append({
                "gene_a": gene_a, "gene_b": gene_b,
                "n_both": 0, "n_a_only": 0, "n_b_only": 0, "n_neither": 0,
                "n_eligible": 0, "freq_a": np.nan, "freq_b": np.nan,
                "observed": 0, "expected": np.nan, "oe_ratio": np.nan,
                "log2_oe": np.nan, "odds_ratio": np.nan,
                "p_value": np.nan, "direction": "insufficient_data",
            })
            continue

        mut_a = gene_mutated_samples[gene_a] & eligible
        mut_b = gene_mutated_samples[gene_b] & eligible

        n_both = len(mut_a & mut_b)
        n_a_only = len(mut_a - mut_b)
        n_b_only = len(mut_b - mut_a)
        n_neither = n_total - n_both - n_a_only - n_b_only

        freq_a = len(mut_a) / n_total
        freq_b = len(mut_b) / n_total
        expected = freq_a * freq_b * n_total

        if expected > 0:
            oe_ratio = n_both / expected
            log2_oe = np.log2(oe_ratio) if oe_ratio > 0 else -np.inf
        else:
            oe_ratio = np.inf if n_both > 0 else np.nan
            log2_oe = np.inf if n_both > 0 else np.nan

        # 2x2 contingency table for Fisher's exact test
        #                gene_b_mut   gene_b_wt
        # gene_a_mut    [n_both,      n_a_only ]
        # gene_a_wt     [n_b_only,    n_neither]
        table = np.array([[n_both, n_a_only], [n_b_only, n_neither]])
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

        direction = "co-occurrence" if oe_ratio >= 1.0 else "mutual_exclusivity"

        results.append({
            "gene_a": gene_a,
            "gene_b": gene_b,
            "n_both": n_both,
            "n_a_only": n_a_only,
            "n_b_only": n_b_only,
            "n_neither": n_neither,
            "n_eligible": n_total,
            "freq_a": round(freq_a, 6),
            "freq_b": round(freq_b, 6),
            "observed": n_both,
            "expected": round(expected, 4),
            "oe_ratio": round(oe_ratio, 4) if np.isfinite(oe_ratio) else oe_ratio,
            "log2_oe": round(log2_oe, 4) if np.isfinite(log2_oe) else log2_oe,
            "odds_ratio": round(odds_ratio, 4) if np.isfinite(odds_ratio) else odds_ratio,
            "p_value": p_value,
            "direction": direction,
        })

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # BH correction
    # ------------------------------------------------------------------
    valid_mask = results_df["p_value"].notna()
    p_values = results_df.loc[valid_mask, "p_value"].values
    reject, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
    results_df.loc[valid_mask, "p_adjusted"] = p_adj
    results_df.loc[valid_mask, "significant"] = reject
    results_df.loc[~valid_mask, "p_adjusted"] = np.nan
    results_df.loc[~valid_mask, "significant"] = False

    n_sig = results_df["significant"].sum()
    n_cooc = ((results_df["direction"] == "co-occurrence") & results_df["significant"]).sum()
    n_excl = ((results_df["direction"] == "mutual_exclusivity") & results_df["significant"]).sum()

    print(f"\n  Total pairs tested: {len(results_df)}")
    print(f"  Significant after BH (FDR<0.05): {n_sig}")
    print(f"    Co-occurrences: {n_cooc}")
    print(f"    Mutual exclusivities: {n_excl}")

    # Top co-occurrences
    sig_cooc = results_df[
        (results_df["direction"] == "co-occurrence") & results_df["significant"]
    ].sort_values("oe_ratio", ascending=False)
    print("\n  Top 10 co-occurrences (by O/E ratio):")
    for _, r in sig_cooc.head(10).iterrows():
        print(f"    {r['gene_a']}+{r['gene_b']}: O/E={r['oe_ratio']:.2f}, "
              f"obs={r['observed']}, exp={r['expected']:.1f}, "
              f"p_adj={r['p_adjusted']:.2e}")

    # Top mutual exclusivities
    sig_excl = results_df[
        (results_df["direction"] == "mutual_exclusivity") & results_df["significant"]
    ].sort_values("oe_ratio", ascending=True)
    print("\n  Top 10 mutual exclusivities (by O/E ratio):")
    for _, r in sig_excl.head(10).iterrows():
        print(f"    {r['gene_a']}+{r['gene_b']}: O/E={r['oe_ratio']:.4f}, "
              f"obs={r['observed']}, exp={r['expected']:.1f}, "
              f"p_adj={r['p_adjusted']:.2e}")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    print("\n[7/7] Saving results...")

    # --- TSV: one row per pair ---
    tsv_path = results_dir / "myeloid_pairwise_results.tsv"
    results_df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6g")
    print(f"  Saved: {tsv_path}")

    # --- TSV: 20x20 matrix of log2(O/E) for heatmap ---
    matrix = pd.DataFrame(np.nan, index=TOP20_GENES, columns=TOP20_GENES)
    for _, r in results_df.iterrows():
        val = r["log2_oe"]
        matrix.loc[r["gene_a"], r["gene_b"]] = val
        matrix.loc[r["gene_b"], r["gene_a"]] = val
    # Diagonal = 0 (self-comparison)
    np.fill_diagonal(matrix.values, 0.0)
    matrix_path = results_dir / "myeloid_pairwise_matrix.tsv"
    matrix.to_csv(matrix_path, sep="\t", float_format="%.4f")
    print(f"  Saved: {matrix_path}")

    # --- JSON: full results with metadata ---
    json_results = {
        "metadata": {
            "analysis": "Pairwise co-occurrence of top 20 myeloid driver genes",
            "data_source": "AACR Project GENIE",
            "n_myeloid_samples": len(myeloid_samples),
            "n_hypermutated_excluded": len(hypermutated),
            "hypermutation_threshold": HYPERMUTATION_THRESHOLD,
            "n_pairs_tested": len(results_df),
            "n_significant_bh": int(n_sig),
            "n_significant_cooccurrence": int(n_cooc),
            "n_significant_mutual_exclusivity": int(n_excl),
            "genes": TOP20_GENES,
            "filters": {
                "cancer_types": MYELOID_CANCER_TYPES,
                "variant_classifications": sorted(CODING_CLASSIFICATIONS),
                "hypermutation_threshold": HYPERMUTATION_THRESHOLD,
                "correction_method": "Benjamini-Hochberg (FDR)",
                "significance_threshold": 0.05,
            },
            "gene_frequencies": {
                g: {
                    "n_mutated": len(gene_mutated_samples[g] & gene_eligible_samples[g]),
                    "n_eligible": len(gene_eligible_samples[g]),
                    "frequency": round(
                        len(gene_mutated_samples[g] & gene_eligible_samples[g])
                        / len(gene_eligible_samples[g]) * 100, 2
                    ) if len(gene_eligible_samples[g]) > 0 else 0,
                }
                for g in TOP20_GENES
            },
        },
        "pairs": [],
    }

    for _, r in results_df.iterrows():
        pair = {
            "gene_a": r["gene_a"],
            "gene_b": r["gene_b"],
            "contingency_table": {
                "both_mutated": int(r["n_both"]),
                "a_only": int(r["n_a_only"]),
                "b_only": int(r["n_b_only"]),
                "neither": int(r["n_neither"]),
            },
            "n_eligible_samples": int(r["n_eligible"]),
            "freq_a": float(r["freq_a"]) if pd.notna(r["freq_a"]) else None,
            "freq_b": float(r["freq_b"]) if pd.notna(r["freq_b"]) else None,
            "observed": int(r["observed"]),
            "expected": round(float(r["expected"]), 4) if pd.notna(r["expected"]) else None,
            "oe_ratio": round(float(r["oe_ratio"]), 4) if pd.notna(r["oe_ratio"]) and np.isfinite(r["oe_ratio"]) else None,
            "log2_oe": round(float(r["log2_oe"]), 4) if pd.notna(r["log2_oe"]) and np.isfinite(r["log2_oe"]) else None,
            "odds_ratio": round(float(r["odds_ratio"]), 4) if pd.notna(r["odds_ratio"]) and np.isfinite(r["odds_ratio"]) else None,
            "p_value": float(r["p_value"]) if pd.notna(r["p_value"]) else None,
            "p_adjusted": round(float(r["p_adjusted"]), 8) if pd.notna(r["p_adjusted"]) else None,
            "significant": bool(r["significant"]) if pd.notna(r["significant"]) else False,
            "direction": r["direction"],
        }
        json_results["pairs"].append(pair)

    json_path = results_dir / "myeloid_pairwise_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved: {json_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
