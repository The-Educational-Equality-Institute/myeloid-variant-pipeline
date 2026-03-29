#!/usr/bin/env python3
"""
setbp1_cooccurrence.py -- SETBP1 co-occurrence analysis against 33 myeloid driver genes.

Analyzes SETBP1 mutation co-occurrence with 33 partner genes in AACR GENIE
myeloid cohort. Includes SKI-domain hotspot breakdown and DDX41 deep dive.

Uses same filters as four_gene_cooccurrence.py:
  - Myeloid malignancy oncotree codes
  - Coding variant classifications only
  - Hypermutation filter (>= 15 coding mutations per sample)
  - Panel adjustment: only count samples where BOTH genes in a pair are covered

Inputs:
    - mutation_profile/data/genie/raw/data_mutations_extended.txt
    - mutation_profile/data/genie/raw/data_clinical_sample.txt
    - mutation_profile/data/genie/raw/genomic_information_*.txt

Outputs:
    - mutation_profile/results/setbp1_makishima/setbp1_cooccurrence_matrix.json
    - mutation_profile/results/setbp1_makishima/setbp1_cooccurrence_matrix.tsv
    - mutation_profile/results/setbp1_makishima/setbp1_ski_domain_cooccurrence.json
    - mutation_profile/results/setbp1_makishima/ddx41_setbp1_deep_dive.json
    - mutation_profile/results/setbp1_makishima/summary_for_email.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/setbp1_cooccurrence.py

Runtime: ~12 seconds
Dependencies: pandas, numpy, scipy, statsmodels

Reference: Makishima et al., Nature Genetics 2013 -- SETBP1 SKI domain hotspots
"""

import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # mutation_profile/
DATA_DIR = PROJECT_ROOT / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results" / "setbp1_makishima"
LOG_DIR = PROJECT_ROOT / "logs"

# 33 partner genes
PARTNER_GENES = [
    "ASXL1", "TET2", "SRSF2", "SF3B1", "RUNX1", "TP53", "FLT3", "NPM1",
    "NRAS", "KRAS", "CBL", "EZH2", "U2AF1", "STAG2", "BCOR", "BCORL1",
    "DDX41", "DNMT3A", "IDH1", "IDH2", "PTPN11", "JAK2", "CALR", "MPL",
    "PHF6", "WT1", "CEBPA", "GATA2", "ZRSR2", "RAD21", "SMC1A", "SMC3",
    "CSF3R",
]

# All genes we need panel coverage for
ALL_GENES = ["SETBP1"] + PARTNER_GENES

# Myeloid cancer types from GENIE CANCER_TYPE field
MYELOID_CANCER_TYPES = [
    "Leukemia",
    "Myeloproliferative Neoplasms",
    "Myelodysplastic Syndromes",
    "Myelodysplastic/Myeloproliferative Neoplasms",
    "Myeloid Neoplasms with Germ Line Predisposition",
    "Histiocytosis",
]

# Coding variant classifications (exclude silent, intronic, UTR, etc.)
CODING_CLASSIFICATIONS = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Translation_Start_Site",
    "Nonstop_Mutation",
}

# Hypermutation threshold: samples with > this many coding mutations in TARGET
# genes are excluded (per CLAUDE.md project rules)
HYPERMUTATION_THRESHOLD = 20

# SETBP1 SKI domain hotspot positions (Makishima et al. 2013)
# The SKI homology domain degron spans residues ~858-871
# Key hotspots: D868, S869, G870, I871
SKI_DOMAIN_POSITIONS = set(range(858, 872))  # 858-871 inclusive
SKI_HOTSPOT_RESIDUES = {868, 869, 870, 871}  # D868, S869, G870, I871

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "setbp1_cooccurrence.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_clinical_samples() -> pd.DataFrame:
    """Load clinical sample data and filter to myeloid malignancies."""
    log.info("Loading clinical sample data...")
    clin = pd.read_csv(DATA_DIR / "data_clinical_sample.txt", sep="\t", comment="#")
    log.info(f"  Total samples in GENIE: {len(clin):,}")

    myeloid = clin[clin["CANCER_TYPE"].isin(MYELOID_CANCER_TYPES)].copy()
    log.info(f"  Myeloid samples: {len(myeloid):,}")
    log.info(f"  Myeloid patients: {myeloid['PATIENT_ID'].nunique():,}")
    for ct in MYELOID_CANCER_TYPES:
        n = (myeloid["CANCER_TYPE"] == ct).sum()
        if n > 0:
            log.info(f"    {ct}: {n:,}")
    return myeloid


def load_gene_matrix() -> pd.DataFrame:
    """Load gene matrix (sample -> panel mapping)."""
    log.info("Loading gene matrix...")
    gm = pd.read_csv(DATA_DIR / "data_gene_matrix.txt", sep="\t")
    log.info(f"  Samples with panel info: {len(gm):,}")
    return gm


def build_panel_gene_sets() -> dict:
    """Parse all gene panel files to build {panel_id: set(genes)} mapping."""
    log.info("Building panel-gene coverage map...")
    panel_genes = {}
    panel_dir = DATA_DIR

    for fpath in sorted(panel_dir.glob("data_gene_panel_*.txt")):
        panel_id = None
        genes = set()
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line.startswith("stable_id:"):
                    panel_id = line.split(":", 1)[1].strip()
                elif line.startswith("gene_list:"):
                    gene_str = line.split(":", 1)[1].strip()
                    genes = set(gene_str.split("\t"))
        if panel_id:
            panel_genes[panel_id] = genes

    log.info(f"  Parsed {len(panel_genes)} gene panels")
    return panel_genes


def load_mutations(sample_ids: set) -> pd.DataFrame:
    """Load mutation data, filter to myeloid samples and coding variants."""
    log.info("Loading mutations (this may take a minute)...")
    cols = [
        "Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode",
        "HGVSp_Short", "Protein_position", "Start_Position", "End_Position",
        "Chromosome",
    ]
    mut = pd.read_csv(
        DATA_DIR / "data_mutations_extended.txt",
        sep="\t", comment="#", usecols=cols, low_memory=False,
    )
    log.info(f"  Total mutations in GENIE: {len(mut):,}")

    # Filter to myeloid samples
    mut = mut[mut["Tumor_Sample_Barcode"].isin(sample_ids)].copy()
    log.info(f"  Mutations in myeloid samples: {len(mut):,}")

    # Filter to coding variants
    mut = mut[mut["Variant_Classification"].isin(CODING_CLASSIFICATIONS)].copy()
    log.info(f"  Coding mutations in myeloid: {len(mut):,}")

    return mut


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def apply_hypermutation_filter(
    mut: pd.DataFrame,
) -> tuple[pd.DataFrame, set]:
    """Remove samples with > HYPERMUTATION_THRESHOLD coding mutations in
    the 34 target genes. Per project rules, the count is restricted to
    the target gene set (ALL_GENES), not all genes genome-wide.
    Returns filtered mutations and set of excluded sample IDs.
    """
    target_muts = mut[mut["Hugo_Symbol"].isin(ALL_GENES)]
    counts = target_muts.groupby("Tumor_Sample_Barcode").size()
    hypermut = set(counts[counts > HYPERMUTATION_THRESHOLD].index)
    log.info(f"  Hypermutated samples (> {HYPERMUTATION_THRESHOLD} target-gene coding muts): {len(hypermut):,}")
    mut_filtered = mut[~mut["Tumor_Sample_Barcode"].isin(hypermut)].copy()
    log.info(f"  Mutations after hypermutation filter: {len(mut_filtered):,}")
    log.info(f"  Samples after filter: {mut_filtered['Tumor_Sample_Barcode'].nunique():,}")
    return mut_filtered, hypermut


def get_panel_adjusted_samples(
    gene_a: str,
    gene_b: str,
    sample_panel: dict,
    panel_genes: dict,
) -> set:
    """Return set of sample IDs where BOTH gene_a and gene_b are on the panel."""
    valid = set()
    for sample_id, panel_id in sample_panel.items():
        genes_on_panel = panel_genes.get(panel_id, set())
        if gene_a in genes_on_panel and gene_b in genes_on_panel:
            valid.add(sample_id)
    return valid


# ---------------------------------------------------------------------------
# Co-occurrence Statistics
# ---------------------------------------------------------------------------

def compute_cooccurrence(
    gene_a: str,
    gene_b: str,
    mutated_samples: dict,
    eligible_samples: set,
) -> dict:
    """Compute Fisher's exact test for co-occurrence of two genes.

    Args:
        gene_a, gene_b: gene names
        mutated_samples: {gene: set(sample_ids)} for all genes
        eligible_samples: set of samples where BOTH genes are on panel

    Returns:
        dict with contingency table, odds ratio, p-value, O/E ratio
    """
    n_total = len(eligible_samples)
    if n_total == 0:
        return {
            "gene_a": gene_a, "gene_b": gene_b,
            "n_eligible": 0, "error": "no eligible samples",
        }

    a_mut = mutated_samples.get(gene_a, set()) & eligible_samples
    b_mut = mutated_samples.get(gene_b, set()) & eligible_samples

    both = len(a_mut & b_mut)
    a_only = len(a_mut - b_mut)
    b_only = len(b_mut - a_mut)
    neither = n_total - both - a_only - b_only

    # Contingency table:
    #              gene_b_mut  gene_b_wt
    # gene_a_mut   both        a_only
    # gene_a_wt    b_only      neither
    table = np.array([[both, a_only], [b_only, neither]])

    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

    # Observed / Expected
    freq_a = (both + a_only) / n_total if n_total > 0 else 0
    freq_b = (both + b_only) / n_total if n_total > 0 else 0
    expected = freq_a * freq_b * n_total
    oe_ratio = both / expected if expected > 0 else float("inf")

    return {
        "gene_a": gene_a,
        "gene_b": gene_b,
        "n_eligible": n_total,
        "n_a": both + a_only,
        "n_b": both + b_only,
        "n_both": both,
        "n_a_only": a_only,
        "n_b_only": b_only,
        "n_neither": neither,
        "expected": round(expected, 3),
        "oe_ratio": round(oe_ratio, 3),
        "odds_ratio": round(odds_ratio, 4) if np.isfinite(odds_ratio) else "Inf",
        "p_value": p_value,
        "direction": "co-occurrence" if oe_ratio > 1 else "mutual exclusivity",
    }


def apply_bh_correction(results: list[dict]) -> list[dict]:
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    p_values = [r["p_value"] for r in results]
    if not p_values:
        return results

    reject, pvals_corrected, _, _ = multipletests(p_values, method="fdr_bh")

    for i, r in enumerate(results):
        r["p_bh"] = round(pvals_corrected[i], 6)
        r["significant_bh05"] = bool(reject[i])

    return results


# ---------------------------------------------------------------------------
# SETBP1 SKI Domain Analysis
# ---------------------------------------------------------------------------

def parse_protein_position(hgvsp: str) -> int | None:
    """Extract numeric protein position from HGVSp_Short like p.D868N."""
    if pd.isna(hgvsp) or not isinstance(hgvsp, str):
        return None
    # Remove 'p.' prefix
    s = hgvsp.replace("p.", "")
    # Extract digits
    digits = ""
    for c in s:
        if c.isdigit():
            digits += c
        elif digits:
            break
    return int(digits) if digits else None


def classify_setbp1_mutations(mut: pd.DataFrame) -> pd.DataFrame:
    """Add SKI domain classification to SETBP1 mutations."""
    setbp1 = mut[mut["Hugo_Symbol"] == "SETBP1"].copy()
    setbp1["protein_pos"] = setbp1["HGVSp_Short"].apply(parse_protein_position)
    setbp1["is_ski_domain"] = setbp1["protein_pos"].apply(
        lambda x: x is not None and x in SKI_DOMAIN_POSITIONS
    )
    setbp1["is_ski_hotspot"] = setbp1["protein_pos"].apply(
        lambda x: x is not None and x in SKI_HOTSPOT_RESIDUES
    )
    return setbp1


# ---------------------------------------------------------------------------
# DDX41 Deep Dive
# ---------------------------------------------------------------------------

def ddx41_deep_dive(
    mut: pd.DataFrame,
    myeloid_samples: pd.DataFrame,
    sample_panel: dict,
    panel_genes: dict,
    mutated_samples: dict,
    hypermut_excluded: set,
) -> dict:
    """Comprehensive DDX41 analysis: panel coverage, co-occurrence with
    SETBP1, and DDX41 co-occurrence with all other drivers."""

    # Panel coverage for DDX41
    ddx41_panels = {pid for pid, genes in panel_genes.items() if "DDX41" in genes}
    ddx41_eligible = {
        sid for sid, pid in sample_panel.items()
        if pid in ddx41_panels
    }
    ddx41_eligible -= hypermut_excluded

    log.info(f"DDX41 deep dive:")
    log.info(f"  Panels covering DDX41: {len(ddx41_panels)}")
    log.info(f"  Myeloid samples with DDX41 coverage: {len(ddx41_eligible):,}")

    # DDX41 + SETBP1 patient details
    ddx41_mut_samples = mutated_samples.get("DDX41", set())
    setbp1_mut_samples = mutated_samples.get("SETBP1", set())

    # Patients with both DDX41 and SETBP1 coverage
    both_covered = get_panel_adjusted_samples("DDX41", "SETBP1", sample_panel, panel_genes)
    both_covered -= hypermut_excluded

    both_mutated = ddx41_mut_samples & setbp1_mut_samples & both_covered

    log.info(f"  DDX41+SETBP1 co-mutated patients: {len(both_mutated)}")

    # Details for co-mutated patients
    patient_details = []
    for sample_id in sorted(both_mutated):
        sample_muts = mut[mut["Tumor_Sample_Barcode"] == sample_id]
        patient_row = myeloid_samples[myeloid_samples["SAMPLE_ID"] == sample_id]

        cancer_type = patient_row["CANCER_TYPE"].iloc[0] if len(patient_row) > 0 else "Unknown"
        cancer_detail = patient_row["CANCER_TYPE_DETAILED"].iloc[0] if len(patient_row) > 0 else "Unknown"
        oncotree = patient_row["ONCOTREE_CODE"].iloc[0] if len(patient_row) > 0 else "Unknown"

        ddx41_muts = sample_muts[sample_muts["Hugo_Symbol"] == "DDX41"]
        setbp1_muts = sample_muts[sample_muts["Hugo_Symbol"] == "SETBP1"]
        other_muts = sample_muts[
            ~sample_muts["Hugo_Symbol"].isin(["DDX41", "SETBP1"])
        ]["Hugo_Symbol"].unique().tolist()

        patient_details.append({
            "sample_id": sample_id,
            "cancer_type": cancer_type,
            "cancer_type_detailed": cancer_detail,
            "oncotree_code": oncotree,
            "ddx41_mutations": ddx41_muts["HGVSp_Short"].tolist(),
            "ddx41_classifications": ddx41_muts["Variant_Classification"].tolist(),
            "setbp1_mutations": setbp1_muts["HGVSp_Short"].tolist(),
            "setbp1_classifications": setbp1_muts["Variant_Classification"].tolist(),
            "other_mutated_genes": sorted(other_muts),
            "total_coding_mutations": len(sample_muts),
        })

    # DDX41 co-occurrence with all other drivers (not just SETBP1)
    ddx41_vs_all = []
    for partner in ALL_GENES:
        if partner == "DDX41":
            continue
        eligible = get_panel_adjusted_samples("DDX41", partner, sample_panel, panel_genes)
        eligible -= hypermut_excluded
        result = compute_cooccurrence("DDX41", partner, mutated_samples, eligible)
        ddx41_vs_all.append(result)

    ddx41_vs_all = apply_bh_correction(ddx41_vs_all)
    ddx41_vs_all.sort(key=lambda x: x["p_value"])

    return {
        "ddx41_panel_coverage": {
            "n_panels_with_ddx41": len(ddx41_panels),
            "panel_ids": sorted(ddx41_panels),
            "n_myeloid_samples_covered": len(ddx41_eligible),
        },
        "ddx41_setbp1_overlap": {
            "n_both_covered": len(both_covered),
            "n_ddx41_mutated_in_covered": len(ddx41_mut_samples & both_covered),
            "n_setbp1_mutated_in_covered": len(setbp1_mut_samples & both_covered),
            "n_co_mutated": len(both_mutated),
            "patient_details": patient_details,
        },
        "ddx41_cooccurrence_all_drivers": ddx41_vs_all,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_json(data: dict | list, path: Path) -> None:
    """Save data as formatted JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"  Saved: {path}")


def save_tsv(results: list[dict], path: Path) -> None:
    """Save co-occurrence results as TSV for Excel/R import."""
    df = pd.DataFrame(results)
    col_order = [
        "gene_a", "gene_b", "n_eligible", "n_a", "n_b", "n_both",
        "expected", "oe_ratio", "odds_ratio", "p_value", "p_bh",
        "significant_bh05", "direction",
        "n_a_only", "n_b_only", "n_neither",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]
    df.to_csv(path, sep="\t", index=False)
    log.info(f"  Saved: {path}")


def generate_summary(
    results: list[dict],
    ski_results: list[dict],
    ddx41_dive: dict,
    setbp1_n: int,
    total_myeloid: int,
    ski_n: int,
    non_ski_n: int,
) -> str:
    """Generate human-readable summary for email."""
    lines = []
    lines.append("# SETBP1 Co-occurrence Analysis — GENIE Myeloid Cohort")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Cohort Summary")
    lines.append(f"- Total myeloid samples (post-hypermutation filter): {total_myeloid:,}")
    lines.append(f"- SETBP1-mutated samples: {setbp1_n:,}")
    lines.append(f"- SETBP1 SKI-domain hotspot samples: {ski_n:,}")
    lines.append(f"- SETBP1 non-SKI-domain samples: {non_ski_n:,}")
    lines.append("")

    # Significant co-occurrences
    sig_co = [r for r in results if r.get("significant_bh05") and r["direction"] == "co-occurrence"]
    sig_ex = [r for r in results if r.get("significant_bh05") and r["direction"] == "mutual exclusivity"]

    lines.append("## Significant Co-occurrences (BH q < 0.05)")
    if sig_co:
        lines.append("")
        lines.append("| Partner | Co-mutated | O/E | p-value | BH q |")
        lines.append("|---------|-----------|-----|---------|------|")
        for r in sorted(sig_co, key=lambda x: x["p_value"]):
            lines.append(
                f"| {r['gene_b']} | {r['n_both']}/{r['n_eligible']:,} "
                f"| {r['oe_ratio']:.2f} | {r['p_value']:.2e} | {r['p_bh']:.4f} |"
            )
    else:
        lines.append("None found.")
    lines.append("")

    lines.append("## Significant Mutual Exclusivities (BH q < 0.05)")
    if sig_ex:
        lines.append("")
        lines.append("| Partner | Co-mutated | O/E | p-value | BH q |")
        lines.append("|---------|-----------|-----|---------|------|")
        for r in sorted(sig_ex, key=lambda x: x["p_value"]):
            lines.append(
                f"| {r['gene_b']} | {r['n_both']}/{r['n_eligible']:,} "
                f"| {r['oe_ratio']:.2f} | {r['p_value']:.2e} | {r['p_bh']:.4f} |"
            )
    else:
        lines.append("None found.")
    lines.append("")

    # SKI domain results
    lines.append("## SKI Domain Hotspot Co-occurrence")
    ski_sig = [r for r in ski_results if r.get("significant_bh05")]
    if ski_sig:
        lines.append("")
        lines.append("| Partner | Co-mutated | O/E | p-value | BH q | Direction |")
        lines.append("|---------|-----------|-----|---------|------|-----------|")
        for r in sorted(ski_sig, key=lambda x: x["p_value"]):
            lines.append(
                f"| {r['gene_b']} | {r['n_both']}/{r['n_eligible']:,} "
                f"| {r['oe_ratio']:.2f} | {r['p_value']:.2e} | {r['p_bh']:.4f} "
                f"| {r['direction']} |"
            )
    else:
        lines.append("No significant associations after BH correction for SKI-domain only.")
    lines.append("")

    # DDX41 deep dive
    ddx41_setbp1 = ddx41_dive["ddx41_setbp1_overlap"]
    lines.append("## DDX41 Deep Dive")
    lines.append(f"- Panels covering DDX41: {ddx41_dive['ddx41_panel_coverage']['n_panels_with_ddx41']}")
    lines.append(f"- Myeloid samples with DDX41 coverage: {ddx41_dive['ddx41_panel_coverage']['n_myeloid_samples_covered']:,}")
    lines.append(f"- DDX41+SETBP1 both covered: {ddx41_setbp1['n_both_covered']:,}")
    lines.append(f"- DDX41 mutated (in covered): {ddx41_setbp1['n_ddx41_mutated_in_covered']}")
    lines.append(f"- SETBP1 mutated (in covered): {ddx41_setbp1['n_setbp1_mutated_in_covered']}")
    lines.append(f"- Co-mutated: {ddx41_setbp1['n_co_mutated']}")
    lines.append("")

    if ddx41_setbp1["patient_details"]:
        lines.append("### DDX41+SETBP1 Patient Details")
        lines.append("")
        for pt in ddx41_setbp1["patient_details"]:
            lines.append(f"- **{pt['sample_id']}**: {pt['cancer_type_detailed']} ({pt['oncotree_code']})")
            lines.append(f"  - DDX41: {', '.join(pt['ddx41_mutations'])} ({', '.join(pt['ddx41_classifications'])})")
            lines.append(f"  - SETBP1: {', '.join(pt['setbp1_mutations'])} ({', '.join(pt['setbp1_classifications'])})")
            lines.append(f"  - Other genes: {', '.join(pt['other_mutated_genes'][:10])}")
        lines.append("")

    # Top DDX41 associations
    lines.append("### DDX41 Co-occurrence with All Drivers (top 10 by p-value)")
    lines.append("")
    lines.append("| Partner | Co-mutated | O/E | p-value | BH q | Direction |")
    lines.append("|---------|-----------|-----|---------|------|-----------|")
    for r in ddx41_dive["ddx41_cooccurrence_all_drivers"][:10]:
        lines.append(
            f"| {r['gene_b']} | {r['n_both']}/{r['n_eligible']:,} "
            f"| {r['oe_ratio']:.2f} | {r['p_value']:.2e} | {r['p_bh']:.4f} "
            f"| {r['direction']} |"
        )
    lines.append("")

    # Full results table (all 33)
    lines.append("## Full SETBP1 Co-occurrence Results (all 33 partners)")
    lines.append("")
    lines.append("| Partner | Eligible | SETBP1 | Partner | Both | Expected | O/E | p-value | BH q | Sig |")
    lines.append("|---------|----------|--------|---------|------|----------|-----|---------|------|-----|")
    for r in sorted(results, key=lambda x: x["p_value"]):
        sig_mark = "*" if r.get("significant_bh05") else ""
        lines.append(
            f"| {r['gene_b']} | {r['n_eligible']:,} | {r['n_a']} | {r['n_b']} "
            f"| {r['n_both']} | {r['expected']:.1f} | {r['oe_ratio']:.2f} "
            f"| {r['p_value']:.2e} | {r['p_bh']:.4f} | {sig_mark} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 70)
    log.info("SETBP1 Co-occurrence Analysis — GENIE Myeloid Cohort")
    log.info("=" * 70)

    # 1. Load data
    myeloid_samples = load_clinical_samples()
    gene_matrix = load_gene_matrix()
    panel_genes = build_panel_gene_sets()

    # Merge to get panel assignment for myeloid samples
    myeloid_with_panel = myeloid_samples.merge(
        gene_matrix[["SAMPLE_ID", "mutations"]], on="SAMPLE_ID", how="inner"
    )
    log.info(f"Myeloid samples with panel info (all): {len(myeloid_with_panel):,}")

    # Patient-level deduplication: keep one sample per patient.
    # Prefer samples on panels with the most target-gene coverage, then first
    # sample alphabetically as tiebreaker (deterministic).
    def _panel_target_coverage(panel_id):
        return len(panel_genes.get(panel_id, set()) & set(ALL_GENES))

    myeloid_with_panel["_target_cov"] = myeloid_with_panel["mutations"].apply(
        _panel_target_coverage
    )
    myeloid_with_panel = myeloid_with_panel.sort_values(
        ["PATIENT_ID", "_target_cov", "SAMPLE_ID"],
        ascending=[True, False, True],
    )
    myeloid_with_panel = myeloid_with_panel.drop_duplicates(
        subset="PATIENT_ID", keep="first"
    )
    myeloid_with_panel = myeloid_with_panel.drop(columns=["_target_cov"])
    myeloid_sample_ids = set(myeloid_with_panel["SAMPLE_ID"])
    log.info(f"Myeloid patients (1 sample each): {len(myeloid_sample_ids):,}")

    # Build sample -> panel lookup (only myeloid samples)
    sample_panel = dict(
        zip(myeloid_with_panel["SAMPLE_ID"], myeloid_with_panel["mutations"])
    )

    # Only keep samples where SETBP1 is on the panel
    setbp1_eligible = {
        sid for sid, pid in sample_panel.items()
        if "SETBP1" in panel_genes.get(pid, set())
    }
    log.info(f"Myeloid samples with SETBP1 on panel: {len(setbp1_eligible):,}")

    # 2. Load and filter mutations
    mut = load_mutations(myeloid_sample_ids)
    mut, hypermut_excluded = apply_hypermutation_filter(mut)

    # Remove hypermutated from our eligible pools
    setbp1_eligible -= hypermut_excluded
    clean_sample_panel = {
        sid: pid for sid, pid in sample_panel.items()
        if sid not in hypermut_excluded
    }

    # 3. Build mutated-sample sets per gene
    log.info("Building per-gene mutation sets...")
    relevant_genes = set(ALL_GENES)
    gene_muts = mut[mut["Hugo_Symbol"].isin(relevant_genes)]
    mutated_samples = {}
    for gene in ALL_GENES:
        samples_with_gene = set(
            gene_muts.loc[gene_muts["Hugo_Symbol"] == gene, "Tumor_Sample_Barcode"]
        )
        mutated_samples[gene] = samples_with_gene
        if gene == "SETBP1":
            log.info(f"  SETBP1 mutated (coding, post-filter): {len(samples_with_gene)}")

    # -----------------------------------------------------------------------
    # 4. SETBP1 vs all 33 partners — panel-adjusted Fisher's exact
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 50)
    log.info("SETBP1 co-occurrence with 33 partners")
    log.info("=" * 50)

    results = []
    for partner in PARTNER_GENES:
        eligible = get_panel_adjusted_samples(
            "SETBP1", partner, clean_sample_panel, panel_genes
        )
        eligible -= hypermut_excluded
        result = compute_cooccurrence("SETBP1", partner, mutated_samples, eligible)
        results.append(result)

    results = apply_bh_correction(results)
    results.sort(key=lambda x: x["p_value"])

    log.info("\nTop results by p-value:")
    for r in results[:10]:
        log.info(
            f"  {r['gene_b']:8s}: both={r['n_both']:3d}, "
            f"O/E={r['oe_ratio']:.2f}, p={r['p_value']:.2e}, "
            f"BH={r['p_bh']:.4f} {'***' if r.get('significant_bh05') else ''}"
        )

    # -----------------------------------------------------------------------
    # 5. SKI Domain Analysis
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 50)
    log.info("SKI Domain Hotspot Analysis (positions 858-871)")
    log.info("=" * 50)

    setbp1_classified = classify_setbp1_mutations(mut)
    ski_samples = set(
        setbp1_classified.loc[
            setbp1_classified["is_ski_domain"], "Tumor_Sample_Barcode"
        ]
    )
    non_ski_samples = set(
        setbp1_classified.loc[
            ~setbp1_classified["is_ski_domain"], "Tumor_Sample_Barcode"
        ]
    ) - ski_samples  # Patients with ONLY non-SKI mutations

    log.info(f"  SKI-domain SETBP1 patients: {len(ski_samples)}")
    log.info(f"  Non-SKI SETBP1 patients (exclusive): {len(non_ski_samples)}")

    # Show SKI domain mutation distribution
    ski_muts = setbp1_classified[setbp1_classified["is_ski_domain"]]
    log.info(f"  SKI domain mutations breakdown:")
    for pos_label, count in ski_muts["HGVSp_Short"].value_counts().head(10).items():
        log.info(f"    {pos_label}: {count}")

    # Run co-occurrence for SKI-domain mutated SETBP1 only
    mutated_samples_ski = mutated_samples.copy()
    mutated_samples_ski["SETBP1"] = ski_samples

    ski_results = []
    for partner in PARTNER_GENES:
        eligible = get_panel_adjusted_samples(
            "SETBP1", partner, clean_sample_panel, panel_genes
        )
        eligible -= hypermut_excluded
        result = compute_cooccurrence(
            "SETBP1", partner, mutated_samples_ski, eligible
        )
        ski_results.append(result)

    ski_results = apply_bh_correction(ski_results)
    ski_results.sort(key=lambda x: x["p_value"])

    log.info("\nSKI-domain top results:")
    for r in ski_results[:10]:
        log.info(
            f"  {r['gene_b']:8s}: both={r['n_both']:3d}, "
            f"O/E={r['oe_ratio']:.2f}, p={r['p_value']:.2e}, "
            f"BH={r['p_bh']:.4f} {'***' if r.get('significant_bh05') else ''}"
        )

    # Non-SKI results for comparison
    mutated_samples_nonski = mutated_samples.copy()
    mutated_samples_nonski["SETBP1"] = non_ski_samples

    nonski_results = []
    for partner in PARTNER_GENES:
        eligible = get_panel_adjusted_samples(
            "SETBP1", partner, clean_sample_panel, panel_genes
        )
        eligible -= hypermut_excluded
        result = compute_cooccurrence(
            "SETBP1", partner, mutated_samples_nonski, eligible
        )
        nonski_results.append(result)

    nonski_results = apply_bh_correction(nonski_results)

    # -----------------------------------------------------------------------
    # 6. DDX41 Deep Dive
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 50)
    log.info("DDX41 Deep Dive")
    log.info("=" * 50)

    ddx41_dive = ddx41_deep_dive(
        mut, myeloid_samples, clean_sample_panel, panel_genes,
        mutated_samples, hypermut_excluded,
    )

    # -----------------------------------------------------------------------
    # 7. Save Results
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 50)
    log.info("Saving results")
    log.info("=" * 50)

    # Main co-occurrence matrix
    save_json(results, RESULTS_DIR / "setbp1_cooccurrence_matrix.json")
    save_tsv(results, RESULTS_DIR / "setbp1_cooccurrence_matrix.tsv")

    # SKI domain results
    ski_output = {
        "ski_domain_positions": "858-871",
        "ski_hotspot_residues": "D868, S869, G870, I871",
        "n_ski_domain_patients": len(ski_samples),
        "n_non_ski_patients": len(non_ski_samples),
        "ski_domain_cooccurrence": ski_results,
        "non_ski_domain_cooccurrence": nonski_results,
    }
    save_json(ski_output, RESULTS_DIR / "setbp1_ski_domain_cooccurrence.json")

    # DDX41 deep dive
    save_json(ddx41_dive, RESULTS_DIR / "ddx41_setbp1_deep_dive.json")

    # Summary
    setbp1_n = len(mutated_samples["SETBP1"])
    total_myeloid_clean = len(set(clean_sample_panel.keys()) - hypermut_excluded)
    summary = generate_summary(
        results, ski_results, ddx41_dive,
        setbp1_n, total_myeloid_clean,
        len(ski_samples), len(non_ski_samples),
    )
    with open(RESULTS_DIR / "summary_for_email.md", "w") as f:
        f.write(summary)
    log.info(f"  Saved: {RESULTS_DIR / 'summary_for_email.md'}")

    log.info("")
    log.info("=" * 70)
    log.info("DONE")
    log.info("=" * 70)

    return results, ski_results, ddx41_dive


if __name__ == "__main__":
    main()
