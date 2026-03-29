#!/usr/bin/env python3
"""
revolver_input.py -- Prepare binary mutation matrix and CCF matrix for REVOLVER analysis.

REVOLVER (Repeated Evolution in cancer) infers the temporal order of somatic
mutation acquisition across a cancer cohort.  It requires:
  1. A binary mutation matrix (patients x genes, 0/1)
  2. Optionally, a CCF (cancer cell fraction) matrix for phylogenetic ordering

This script builds both from GENIE v19.0 myeloid data using the standard
filtering pipeline (myeloid OncoTree codes, coding variants only,
hypermutation exclusion, panel-aware sample selection).

Inputs:
    - mutation_profile/data/genie/raw/data_mutations_extended.txt
    - mutation_profile/data/genie/raw/data_clinical_sample.txt
    - mutation_profile/data/genie/raw/data_gene_panel_*.txt
    - mutation_profile/data/genie/raw/data_gene_matrix.txt

Outputs:
    - mutation_profile/results/ai_research/revolver_input_matrix.csv
    - mutation_profile/results/ai_research/revolver_ccf_matrix.csv
    - mutation_profile/results/ai_research/revolver_summary.json

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/revolver_input.py

Runtime: ~30 seconds
Dependencies: pandas, numpy
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Add parent dir to path so we can import genie_loader
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from genie_loader import GENIEData, MYELOID_ONCOTREE_CODES, NONCODING_CLASSIFICATIONS, TARGET_GENES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The 34 target genes from CLAUDE.md
REVOLVER_GENES = TARGET_GENES

# Coding variant classifications (positive list, matching pairwise_matrix.py)
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

HYPERMUTATION_THRESHOLD = 20  # max coding mutations in target genes per sample

# Minimum number of mutated patients for a gene to be included in REVOLVER
MIN_GENE_FREQUENCY = 10


def main():
    t0 = time.time()

    project_root = SCRIPT_DIR.parent.parent  # mutation_profile/
    data_dir = project_root / "data" / "genie" / "raw"
    results_dir = project_root / "results" / "ai_research"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REVOLVER INPUT PREPARATION — GENIE v19.0 MYELOID COHORT")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data via GENIEData loader
    # ------------------------------------------------------------------
    print("\n[1/7] Loading GENIE data...")
    gd = GENIEData(data_dir)

    # ------------------------------------------------------------------
    # 2. Get myeloid samples
    # ------------------------------------------------------------------
    print("\n[2/7] Filtering to myeloid samples...")
    myeloid_samples = gd.get_myeloid_samples()
    print(f"  Myeloid samples (OncoTree): {len(myeloid_samples):,}")

    # ------------------------------------------------------------------
    # 3. Get coding mutations in myeloid samples
    # ------------------------------------------------------------------
    print("\n[3/7] Filtering to coding mutations...")
    coding_muts = gd.get_coding_mutations(samples=myeloid_samples)
    print(f"  Coding mutations in myeloid: {len(coding_muts):,}")

    # Also apply positive-list filter (matching pairwise_matrix.py approach)
    coding_muts = coding_muts[
        coding_muts["Variant_Classification"].isin(CODING_CLASSIFICATIONS)
    ]
    print(f"  After positive-list filter: {len(coding_muts):,}")

    # ------------------------------------------------------------------
    # 4. Hypermutation filter
    # ------------------------------------------------------------------
    print("\n[4/7] Applying hypermutation filter...")
    target_muts = coding_muts[coding_muts["Hugo_Symbol"].isin(REVOLVER_GENES)]
    counts_per_sample = target_muts.groupby("Tumor_Sample_Barcode").size()
    hypermutated = set(
        counts_per_sample[counts_per_sample > HYPERMUTATION_THRESHOLD].index
    )
    print(f"  Hypermutated samples (>{HYPERMUTATION_THRESHOLD} target-gene muts): {len(hypermutated):,}")

    myeloid_samples -= hypermutated
    coding_muts = coding_muts[
        ~coding_muts["Tumor_Sample_Barcode"].isin(hypermutated)
    ]
    print(f"  Remaining myeloid samples: {len(myeloid_samples):,}")

    # ------------------------------------------------------------------
    # 5. Panel-aware filtering: keep only samples covering ALL target genes
    # ------------------------------------------------------------------
    print("\n[5/7] Panel-aware sample selection...")
    covered_samples = gd.samples_covering_genes(REVOLVER_GENES, myeloid_only=True)
    covered_samples -= hypermutated
    print(f"  Samples covering all 34 target genes: {len(covered_samples):,}")

    # Also compute per-gene coverage for a relaxed approach
    gene_coverage: dict[str, set[str]] = {}
    for gene in REVOLVER_GENES:
        gene_coverage[gene] = gd.samples_covering_genes(
            [gene], myeloid_only=True
        ) - hypermutated

    # For REVOLVER, we use the full myeloid set (not restricted to all-34 coverage)
    # but only mark a gene as mutated if the sample's panel covers that gene.
    # This avoids both false negatives and false positives.
    analysis_samples = myeloid_samples
    print(f"  Using full myeloid cohort for REVOLVER: {len(analysis_samples):,}")

    # ------------------------------------------------------------------
    # 6. Build binary mutation matrix and CCF matrix
    # ------------------------------------------------------------------
    print("\n[6/7] Building binary mutation matrix and CCF matrix...")

    # Filter to target genes
    target_coding = coding_muts[
        (coding_muts["Hugo_Symbol"].isin(REVOLVER_GENES))
        & (coding_muts["Tumor_Sample_Barcode"].isin(analysis_samples))
    ].copy()
    print(f"  Target gene coding mutations: {len(target_coding):,}")

    # Compute VAF where depth data available
    target_coding["VAF"] = np.nan
    has_depth = (
        target_coding["t_alt_count"].notna()
        & target_coding["t_depth"].notna()
        & (target_coding["t_depth"] > 0)
    )
    target_coding.loc[has_depth, "VAF"] = (
        target_coding.loc[has_depth, "t_alt_count"].astype(float)
        / target_coding.loc[has_depth, "t_depth"].astype(float)
    )
    vaf_available = target_coding["VAF"].notna().sum()
    print(f"  Mutations with VAF data: {vaf_available:,} / {len(target_coding):,} "
          f"({vaf_available / len(target_coding) * 100:.1f}%)")

    # For each (sample, gene), take the MAX VAF mutation (dominant clone)
    # and mark as binary 1 if ANY coding mutation exists
    sample_gene_vaf = (
        target_coding.groupby(["Tumor_Sample_Barcode", "Hugo_Symbol"])["VAF"]
        .max()
        .reset_index()
    )
    sample_gene_binary = (
        target_coding[["Tumor_Sample_Barcode", "Hugo_Symbol"]]
        .drop_duplicates()
    )

    # Pivot to matrix form
    # Binary matrix
    sample_gene_binary["mutated"] = 1
    binary_wide = sample_gene_binary.pivot_table(
        index="Tumor_Sample_Barcode",
        columns="Hugo_Symbol",
        values="mutated",
        fill_value=0,
        aggfunc="max",
    )

    # Ensure all target genes are columns (add missing as 0)
    for gene in REVOLVER_GENES:
        if gene not in binary_wide.columns:
            binary_wide[gene] = 0
    binary_wide = binary_wide[REVOLVER_GENES]  # consistent column order

    # Add samples with NO target gene mutations as all-zero rows
    all_sample_ids = sorted(analysis_samples)
    missing_samples = set(all_sample_ids) - set(binary_wide.index)
    if missing_samples:
        zero_rows = pd.DataFrame(
            0,
            index=sorted(missing_samples),
            columns=REVOLVER_GENES,
        )
        binary_wide = pd.concat([binary_wide, zero_rows])
    binary_wide = binary_wide.sort_index()

    # Apply panel masking: if a sample's panel does NOT cover a gene,
    # set that cell to NA (not 0, because 0 means "tested and not mutated")
    # REVOLVER can handle NA values
    panel_mask_count = 0
    for gene in REVOLVER_GENES:
        covered = gene_coverage[gene]
        uncovered_in_matrix = set(binary_wide.index) - covered
        if uncovered_in_matrix:
            binary_wide.loc[sorted(uncovered_in_matrix), gene] = np.nan
            panel_mask_count += len(uncovered_in_matrix)

    # Remove samples that are ALL NA (no gene covered)
    all_na_mask = binary_wide.isna().all(axis=1)
    n_all_na = all_na_mask.sum()
    if n_all_na > 0:
        print(f"  Removing {n_all_na} samples with no panel coverage for any target gene")
        binary_wide = binary_wide[~all_na_mask]

    # Remove genes with fewer than MIN_GENE_FREQUENCY mutations
    gene_mut_counts = (binary_wide == 1).sum()
    low_freq_genes = gene_mut_counts[gene_mut_counts < MIN_GENE_FREQUENCY].index.tolist()
    kept_genes = [g for g in REVOLVER_GENES if g not in low_freq_genes]
    if low_freq_genes:
        print(f"  Removing {len(low_freq_genes)} genes with <{MIN_GENE_FREQUENCY} mutations: "
              f"{', '.join(low_freq_genes)}")
    binary_wide = binary_wide[kept_genes]

    # For REVOLVER: remove patients with 0 mutations (all 0 or all NA after filtering)
    # REVOLVER needs patients with at least 1 mutation
    has_mutation = (binary_wide == 1).any(axis=1)
    n_no_mut = (~has_mutation).sum()
    print(f"  Patients with no target gene mutations (excluded from REVOLVER): {n_no_mut:,}")
    binary_revolver = binary_wide[has_mutation].copy()

    print(f"\n  Final REVOLVER binary matrix: {binary_revolver.shape[0]:,} patients x "
          f"{binary_revolver.shape[1]} genes")

    # CCF matrix: VAF / 0.5 as rough CCF estimate (assumes diploid, heterozygous)
    # Cap at 1.0
    ccf_wide = sample_gene_vaf.pivot_table(
        index="Tumor_Sample_Barcode",
        columns="Hugo_Symbol",
        values="VAF",
        aggfunc="max",
    )
    # Compute CCF = min(VAF / 0.5, 1.0)
    ccf_wide = (ccf_wide / 0.5).clip(upper=1.0)

    # Align to same samples and genes as binary matrix
    ccf_wide = ccf_wide.reindex(
        index=binary_revolver.index,
        columns=kept_genes,
    )

    # Where binary is 0 (no mutation), CCF should be 0
    ccf_wide = ccf_wide.where(binary_revolver == 1, other=np.nan)
    # Where binary is 1 but no VAF data, set CCF to NA
    ccf_available = ccf_wide.notna().sum().sum()
    ccf_total = (binary_revolver == 1).sum().sum()
    print(f"  CCF values available: {ccf_available:,} / {ccf_total:,} mutation entries "
          f"({ccf_available / ccf_total * 100:.1f}%)" if ccf_total > 0 else "  No mutations found")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    print("\n[7/7] Saving outputs...")

    # Binary matrix CSV
    binary_path = results_dir / "revolver_input_matrix.csv"
    binary_revolver.index.name = "sample_id"
    binary_revolver.to_csv(binary_path)
    print(f"  Saved: {binary_path}")

    # CCF matrix CSV
    ccf_path = results_dir / "revolver_ccf_matrix.csv"
    ccf_wide.index.name = "sample_id"
    ccf_wide.to_csv(ccf_path)
    print(f"  Saved: {ccf_path}")

    # Summary statistics
    n_patients = binary_revolver.shape[0]
    n_genes = binary_revolver.shape[1]
    n_ones = (binary_revolver == 1).sum().sum()
    n_zeros = (binary_revolver == 0).sum().sum()
    n_na = binary_revolver.isna().sum().sum()
    total_cells = n_patients * n_genes
    sparsity = n_zeros / (n_zeros + n_ones) if (n_zeros + n_ones) > 0 else 0

    # Per-gene stats
    gene_stats = {}
    for gene in kept_genes:
        col = binary_revolver[gene]
        n_mut = (col == 1).sum()
        n_wt = (col == 0).sum()
        n_uncovered = col.isna().sum()
        freq = n_mut / (n_mut + n_wt) * 100 if (n_mut + n_wt) > 0 else 0
        gene_stats[gene] = {
            "n_mutated": int(n_mut),
            "n_wildtype": int(n_wt),
            "n_uncovered": int(n_uncovered),
            "frequency_pct": round(freq, 2),
        }

    # Mutation count distribution
    mut_counts = (binary_revolver == 1).sum(axis=1)
    mut_count_dist = mut_counts.value_counts().sort_index().to_dict()
    mut_count_dist = {int(k): int(v) for k, v in mut_count_dist.items()}

    # Mean CCF per gene (where available)
    mean_ccf = {}
    for gene in kept_genes:
        vals = ccf_wide[gene].dropna()
        if len(vals) > 0:
            mean_ccf[gene] = {
                "mean_ccf": round(vals.mean(), 3),
                "median_ccf": round(vals.median(), 3),
                "n_with_ccf": int(len(vals)),
            }

    summary = {
        "analysis": "REVOLVER input preparation",
        "data_source": "AACR Project GENIE v19.0",
        "n_patients": n_patients,
        "n_genes": n_genes,
        "genes_included": kept_genes,
        "genes_excluded_low_freq": low_freq_genes,
        "total_cells": total_cells,
        "n_mutated_cells": int(n_ones),
        "n_wildtype_cells": int(n_zeros),
        "n_uncovered_cells": int(n_na),
        "sparsity": round(sparsity, 4),
        "mean_mutations_per_patient": round(mut_counts.mean(), 2),
        "median_mutations_per_patient": int(mut_counts.median()),
        "mutation_count_distribution": mut_count_dist,
        "gene_statistics": gene_stats,
        "ccf_coverage": {
            "n_entries_with_ccf": int(ccf_available),
            "n_total_mutation_entries": int(ccf_total),
            "coverage_pct": round(ccf_available / ccf_total * 100, 1) if ccf_total > 0 else 0,
        },
        "mean_ccf_per_gene": mean_ccf,
        "filters": {
            "oncotree_codes": sorted(MYELOID_ONCOTREE_CODES),
            "coding_classifications": sorted(CODING_CLASSIFICATIONS),
            "hypermutation_threshold": HYPERMUTATION_THRESHOLD,
            "min_gene_frequency": MIN_GENE_FREQUENCY,
            "panel_aware_masking": True,
        },
        "output_files": {
            "binary_matrix": str(binary_path.name),
            "ccf_matrix": str(ccf_path.name),
        },
    }

    summary_path = results_dir / "revolver_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("REVOLVER INPUT SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Patients:                {n_patients:,}")
    print(f"  Genes:                   {n_genes}")
    print(f"  Mutated cells:           {n_ones:,}")
    print(f"  Wildtype cells:          {n_zeros:,}")
    print(f"  Uncovered (NA) cells:    {n_na:,}")
    print(f"  Sparsity (0s / tested):  {sparsity:.1%}")
    print(f"  Mean muts/patient:       {mut_counts.mean():.2f}")
    print(f"  Median muts/patient:     {int(mut_counts.median())}")
    print(f"  CCF coverage:            {ccf_available:,}/{ccf_total:,} "
          f"({ccf_available / ccf_total * 100:.1f}%)" if ccf_total > 0 else "  No mutations")
    print(f"\n  Mutation count distribution:")
    for n_muts, count in sorted(mut_count_dist.items()):
        print(f"    {n_muts} mutation(s): {count:,} patients")

    print(f"\n  Top 10 most frequently mutated genes:")
    sorted_genes = sorted(gene_stats.items(), key=lambda x: x[1]["n_mutated"], reverse=True)
    for gene, stats in sorted_genes[:10]:
        print(f"    {gene}: {stats['n_mutated']:,} ({stats['frequency_pct']:.1f}%)")

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
