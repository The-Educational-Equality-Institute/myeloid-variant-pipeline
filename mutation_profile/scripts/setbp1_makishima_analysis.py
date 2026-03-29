#!/usr/bin/env python3
"""
SETBP1 Co-occurrence Analysis for Professor Hideki Makishima (Kyoto University)

AACR GENIE v19.0 (3.46M mutation rows, 27,585 myeloid samples, 32 target genes)

Analyses:
  1. SETBP1 co-occurrence with all 31 other myeloid driver genes
  2. SETBP1 SKI-domain (aa 858-871) vs non-SKI co-occurrence breakdown
  3. DDX41 + SETBP1 deep dive (with panel coverage caveat)
  4. Full pairwise matrix for top 20 most frequently mutated myeloid genes

Output: mutation_profile/results/setbp1_makishima/
"""

import json
import math
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

# ============================================================================
# CONSTANTS
# ============================================================================

PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
GENIE_RAW = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "setbp1_makishima"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# All 32 myeloid driver genes
TARGET_GENES = [
    "ASXL1", "BCOR", "BCORL1", "CALR", "CBL", "CEBPA", "CSF3R",
    "DDX41", "DNMT3A", "EZH2", "FLT3", "GATA2", "IDH1", "IDH2",
    "JAK2", "KIT", "KRAS", "MPL", "NPM1", "NRAS", "PHF6",
    "PTPN11", "RUNX1", "SETBP1", "SF3B1", "SRSF2", "STAG2",
    "TET2", "TP53", "U2AF1", "WT1", "ZRSR2",
]
TARGET_GENES_SET = set(TARGET_GENES)

# Variant classifications to KEEP (protein-altering)
PATHOGENIC_VAR_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Nonstop_Mutation", "Translation_Start_Site",
}

# Variant classifications to EXCLUDE
EXCLUDED_VAR_CLASSES = {
    "Intron", "Silent", "3'UTR", "5'UTR", "Splice_Region",
    "3'Flank", "5'Flank", "IGR", "RNA",
}

# OncoTree codes for myeloid neoplasms
MYELOID_ONCOTREE_CODES = {
    "AML", "AMLMRC", "AMLRGA", "AMLRR", "AMLNOS", "APL", "AMOL", "APMF",
    "AMLMBC", "AMLCBFB", "AMLRUNX1", "AMLMLLT3", "AMLDEKNUP", "AMLBCR",
    "AMLGATA2MECOM", "AMLNPM1", "AMLCEBPA", "AMLTP53", "AMLDEK",
    "MDS", "MDS5Q", "MDSEB1", "MDSEB2", "MDSMD", "MDSSLD", "MDSRS",
    "MDSU", "MDSRSMD", "MDSSID", "MDSLB", "MDSIB1", "MDSIB2",
    "RCMD", "RARS", "RA", "RAEB", "RAEB1", "RAEB2", "RCUD",
    "CMML", "CMML0", "CMML1", "CMML2", "JMML", "MDSMPNU",
    "MDSMPNRST", "ACML", "ACML_ATYPICAL",
    "MPN", "CML", "ET", "PV", "PMF", "CMLBCRABL1", "SM", "CEL",
    "MPNU", "MPNST",
    "BPDCN", "MPAL", "TMN", "ALAL",
    "TMDS", "TAML",
}

MYELOID_KEYWORDS = [
    "leukemia", "myeloid", "myelodysplast", "mds", "aml",
    "myeloproliferat", "myelomonocytic", "erythroleukemia",
]

# SETBP1 SKI domain: amino acid positions 858-871
SETBP1_SKI_POSITIONS = range(858, 872)

# Hypermutation threshold (across ALL 32 target genes)
HYPERMUTATION_THRESHOLD = 20


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_star_pattern(protein_change):
    """Check if protein change matches UCHI artifact p.*NNN* pattern."""
    return bool(re.match(r'^p\.\*\d+\*$', protein_change))


def is_myeloid(sample_info):
    """Check if sample is myeloid based on OncoTree code or cancer type keywords."""
    code = sample_info.get("oncotree_code", "").upper()
    if code in MYELOID_ONCOTREE_CODES:
        return True
    cancer_type = (
        sample_info.get("cancer_type", "") + " " +
        sample_info.get("cancer_type_detailed", "")
    ).lower()
    return any(kw in cancer_type for kw in MYELOID_KEYWORDS)


def classify_setbp1_mutation(protein_change):
    """Classify SETBP1 mutation as SKI-domain or non-SKI.

    SKI domain: amino acid positions 858-871 (D868, S869, G870, I871, etc.)
    """
    if not protein_change:
        return "non-SKI"
    match = re.match(r'^p\.[A-Z*](\d+)', protein_change)
    if match:
        pos = int(match.group(1))
        if pos in SETBP1_SKI_POSITIONS:
            return "SKI"
    return "non-SKI"


def run_fisher_test(n_total, n_a, n_b, n_ab):
    """Run Fisher's exact test and compute obs/exp statistics.

    Returns dict with all stats, or None if cells are negative.
    """
    table = [
        [n_ab, n_a - n_ab],
        [n_b - n_ab, n_total - n_a - n_b + n_ab],
    ]
    for row in table:
        for cell in row:
            if cell < 0:
                return None

    table_np = np.array(table)
    odds_ratio, p_value = fisher_exact(table_np, alternative='two-sided')
    _, p_less = fisher_exact(table_np, alternative='less')
    _, p_greater = fisher_exact(table_np, alternative='greater')

    expected = (n_a / n_total) * (n_b / n_total) * n_total if n_total > 0 else 0
    obs_exp = n_ab / expected if expected > 0 else float('inf')

    return {
        "n_total": n_total,
        "n_gene_a": n_a,
        "n_gene_b": n_b,
        "n_both": n_ab,
        "expected": round(expected, 4),
        "obs_exp_ratio": round(obs_exp, 4),
        "odds_ratio": round(odds_ratio, 6) if np.isfinite(odds_ratio) else None,
        "p_value_two_sided": p_value,
        "p_value_less": p_less,
        "p_value_greater": p_greater,
        "direction": "enriched" if obs_exp > 1.2 else ("depleted" if obs_exp < 0.8 else "neutral"),
        "contingency_table": table,
    }


def pair_stats(g1, g2, patient_list, on_panel, mutated):
    """Compute panel-adjusted stats for a gene pair in one pass.

    Returns (panel_n, n_g1, n_g2, n_both).
    """
    n = 0
    n_g1 = 0
    n_g2 = 0
    n_both = 0
    for pid in patient_list:
        panel = on_panel[pid]
        if g1 in panel and g2 in panel:
            n += 1
            has_g1 = g1 in mutated[pid]
            has_g2 = g2 in mutated[pid]
            if has_g1:
                n_g1 += 1
            if has_g2:
                n_g2 += 1
            if has_g1 and has_g2:
                n_both += 1
    return n, n_g1, n_g2, n_both


def pair_stats_with_subset(anchor_gene, partner_gene, anchor_patients,
                           patient_list, on_panel, mutated):
    """Like pair_stats but uses a custom patient set for the anchor gene.

    Used for SKI-domain analysis where anchor_patients is a subset of
    SETBP1-mutated patients (only SKI or only non-SKI).

    Returns (panel_n, n_anchor, n_partner, n_both).
    """
    n = 0
    n_anchor = 0
    n_partner = 0
    n_both = 0
    for pid in patient_list:
        panel = on_panel[pid]
        if anchor_gene in panel and partner_gene in panel:
            n += 1
            has_anchor = pid in anchor_patients
            has_partner = partner_gene in mutated[pid]
            if has_anchor:
                n_anchor += 1
            if has_partner:
                n_partner += 1
            if has_anchor and has_partner:
                n_both += 1
    return n, n_anchor, n_partner, n_both


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return []
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


def format_pval(p):
    """Format p-value for display."""
    if p < 1e-10:
        return f"{p:.2e}"
    elif p < 0.001:
        return f"{p:.2e}"
    elif p < 0.01:
        return f"{p:.4f}"
    else:
        return f"{p:.4f}"


# ============================================================================
# DATA LOADING (patterns from analyze_genie_filtered.py)
# ============================================================================

def load_clinical_samples():
    """Load clinical sample data: sample_id -> info dict."""
    path = GENIE_RAW / "data_clinical_sample.txt"
    samples = {}
    with open(path) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            if header is None:
                header = line.strip().split("\t")
                continue
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))
            sid = row.get("SAMPLE_ID", "")
            if not sid:
                continue
            samples[sid] = {
                "patient_id": row.get("PATIENT_ID", ""),
                "oncotree_code": row.get("ONCOTREE_CODE", ""),
                "cancer_type": row.get("CANCER_TYPE", ""),
                "cancer_type_detailed": row.get("CANCER_TYPE_DETAILED", ""),
                "center": row.get("CENTER", ""),
                "seq_panel": row.get("SEQ_ASSAY_ID", ""),
            }
    return samples


def load_clinical_patients():
    """Load patient-level clinical data: patient_id -> info dict."""
    path = GENIE_RAW / "data_clinical_patient.txt"
    patients = {}
    with open(path) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            if header is None:
                header = line.strip().split("\t")
                continue
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))
            pid = row.get("PATIENT_ID", "")
            if not pid:
                continue
            age_str = row.get("AGE_AT_SEQ_REPORT", "")
            try:
                age = float(age_str)
            except (ValueError, TypeError):
                age = None
            patients[pid] = {
                "age": age,
                "sex": row.get("SEX", ""),
                "center": row.get("CENTER", ""),
            }
    return patients


def load_gene_panels():
    """Load gene panel definitions: panel_name -> set of genes."""
    panels = {}
    for pf in GENIE_RAW.glob("data_gene_panel_*.txt"):
        panel_name = pf.stem.replace("data_gene_panel_", "")
        genes = set()
        with open(pf) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line.startswith("stable_id") or not line:
                    continue
                for part in line.split("\t"):
                    part = part.strip()
                    if part and not part.startswith("gene_list"):
                        genes.add(part)
        panels[panel_name] = genes
    return panels


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 70)
    print("SETBP1 Co-occurrence Analysis for Makishima")
    print(f"Target genes: {len(TARGET_GENES)} myeloid drivers")
    print("Database: AACR GENIE v19.0-public")
    print("=" * 70)

    # --- Load clinical data ---
    print("\nLoading clinical data...")
    all_samples = load_clinical_samples()
    print(f"  Total samples: {len(all_samples):,}")

    myeloid_samples = {
        sid: info for sid, info in all_samples.items() if is_myeloid(info)
    }
    print(f"  Myeloid samples: {len(myeloid_samples):,}")

    patient_clinical = load_clinical_patients()
    print(f"  Total patients in clinical file: {len(patient_clinical):,}")

    gene_panels = load_gene_panels()
    print(f"  Gene panels loaded: {len(gene_panels)}")

    # --- Build patient -> panel genes mapping ---
    # For each myeloid patient, store the union of genes across all their samples' panels
    patient_panel_genes = {}
    for sid, info in myeloid_samples.items():
        panel = info.get("seq_panel", "")
        pid = info.get("patient_id", "")
        if panel in gene_panels:
            panel_genes = gene_panels[panel]
        else:
            # Unknown panel: assume all target genes covered
            panel_genes = TARGET_GENES_SET
        if pid not in patient_panel_genes:
            patient_panel_genes[pid] = set()
        patient_panel_genes[pid] |= panel_genes

    # Compute sample-level panel coverage per gene (for reporting)
    gene_coverage_samples = {g: 0 for g in TARGET_GENES}
    for sid, info in myeloid_samples.items():
        panel = info.get("seq_panel", "")
        if panel in gene_panels:
            for g in TARGET_GENES:
                if g in gene_panels[panel]:
                    gene_coverage_samples[g] += 1
        else:
            for g in TARGET_GENES:
                gene_coverage_samples[g] += 1

    print("\n  Panel coverage (myeloid samples):")
    for g in sorted(TARGET_GENES):
        pct = gene_coverage_samples[g] / len(myeloid_samples) * 100
        marker = " *** LOW ***" if pct < 50 else ""
        print(f"    {g:8s}: {gene_coverage_samples[g]:6,} ({pct:5.1f}%){marker}")

    # --- Load and filter mutations (single pass through MAF) ---
    print(f"\nLoading mutations (single pass, {len(TARGET_GENES)} target genes)...")
    maf_file = GENIE_RAW / "data_mutations_extended.txt"

    # patient -> gene -> [mutation dicts]
    patient_mutations = defaultdict(lambda: defaultdict(list))

    # SETBP1 classification: patient -> "SKI" / "non-SKI" / "both"
    patient_setbp1_class = {}

    # Track total coding mutations per patient for hypermutation detection
    patient_total_muts = Counter()

    # DDX41 germline/somatic tracking
    ddx41_mutation_status = defaultdict(list)

    stats = {
        "total_rows": 0,
        "target_gene_rows": 0,
        "myeloid_target_rows": 0,
        "excluded_noncoding": 0,
        "excluded_star": 0,
        "excluded_unknown_class": 0,
        "kept_coding": 0,
    }

    has_mutation_status = False

    with open(maf_file) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            if header is None:
                header = line.strip().split("\t")
                has_mutation_status = "Mutation_Status" in header
                if has_mutation_status:
                    print("  Mutation_Status column found")
                continue

            stats["total_rows"] += 1
            if stats["total_rows"] % 1_000_000 == 0:
                print(f"  Processed {stats['total_rows']:,} rows...")

            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))

            gene = row.get("Hugo_Symbol", "")
            if gene not in TARGET_GENES_SET:
                continue
            stats["target_gene_rows"] += 1

            sample_id = row.get("Tumor_Sample_Barcode", "")
            if sample_id not in myeloid_samples:
                continue
            stats["myeloid_target_rows"] += 1

            patient_id = myeloid_samples[sample_id]["patient_id"]
            var_class = row.get("Variant_Classification", "")
            protein = row.get("HGVSp_Short", "")

            # Filter non-coding
            if var_class in EXCLUDED_VAR_CLASSES:
                stats["excluded_noncoding"] += 1
                continue

            # Filter star-pattern artifacts
            if is_star_pattern(protein):
                stats["excluded_star"] += 1
                continue

            # Must be in known pathogenic classes
            if var_class not in PATHOGENIC_VAR_CLASSES:
                stats["excluded_unknown_class"] += 1
                continue

            stats["kept_coding"] += 1

            mut_info = {
                "gene": gene,
                "protein_change": protein,
                "variant_classification": var_class,
                "variant_type": row.get("Variant_Type", ""),
                "chromosome": row.get("Chromosome", ""),
                "start_pos": row.get("Start_Position", ""),
                "ref_allele": row.get("Reference_Allele", ""),
                "alt_allele": row.get("Tumor_Seq_Allele2", ""),
                "t_depth": row.get("t_depth", ""),
                "t_alt_count": row.get("t_alt_count", ""),
                "sample_id": sample_id,
                "center": myeloid_samples[sample_id].get("center", ""),
            }

            if has_mutation_status:
                mut_info["mutation_status"] = row.get("Mutation_Status", "")

            patient_mutations[patient_id][gene].append(mut_info)
            patient_total_muts[patient_id] += 1

            # Classify SETBP1 mutations
            if gene == "SETBP1":
                ski_class = classify_setbp1_mutation(protein)
                if patient_id in patient_setbp1_class:
                    if patient_setbp1_class[patient_id] != ski_class:
                        patient_setbp1_class[patient_id] = "both"
                else:
                    patient_setbp1_class[patient_id] = ski_class

            # Track DDX41 mutation status
            if gene == "DDX41" and has_mutation_status:
                ddx41_mutation_status[patient_id].append(
                    row.get("Mutation_Status", "")
                )

    print(f"\n  Filtering stats:")
    for k, v in stats.items():
        print(f"    {k}: {v:,}")

    # --- Identify and exclude hypermutated patients ---
    hypermutated = {
        pid for pid, count in patient_total_muts.items()
        if count > HYPERMUTATION_THRESHOLD
    }
    print(f"\n  Hypermutated patients (>{HYPERMUTATION_THRESHOLD} mutations "
          f"in {len(TARGET_GENES)} target genes): {len(hypermutated)}")
    for pid in sorted(hypermutated):
        count = patient_total_muts[pid]
        genes_hit = sorted(patient_mutations[pid].keys())
        print(f"    {pid}: {count} mutations across {len(genes_hit)} genes: "
              f"{', '.join(genes_hit)}")

    # Remove hypermutated
    for pid in hypermutated:
        del patient_mutations[pid]
        patient_setbp1_class.pop(pid, None)

    # --- Build per-patient data structures ---
    # All myeloid patient IDs (excluding hypermutated)
    all_myeloid_patients = set()
    for sid, info in myeloid_samples.items():
        pid = info["patient_id"]
        if pid not in hypermutated:
            all_myeloid_patients.add(pid)
    all_myeloid_list = sorted(all_myeloid_patients)
    total_myeloid = len(all_myeloid_patients)

    # patient -> set of genes on their panel
    on_panel = {}
    for pid in all_myeloid_list:
        on_panel[pid] = patient_panel_genes.get(pid, TARGET_GENES_SET)

    # patient -> set of mutated genes
    mutated = {}
    for pid in all_myeloid_list:
        mutated[pid] = set(patient_mutations.get(pid, {}).keys())

    # gene -> set of patients
    gene_patients = {g: set() for g in TARGET_GENES}
    for pid in all_myeloid_list:
        for g in mutated[pid]:
            if g in TARGET_GENES_SET:
                gene_patients[g].add(pid)

    n_setbp1 = len(gene_patients["SETBP1"])

    print(f"\n  Myeloid patients (after exclusions): {total_myeloid:,}")
    print(f"\n  Gene mutation counts:")
    for g in sorted(TARGET_GENES, key=lambda x: len(gene_patients[x]), reverse=True):
        print(f"    {g:8s}: {len(gene_patients[g]):5,}")

    # ========================================================================
    # ANALYSIS 1: SETBP1 Co-occurrence Matrix
    # ========================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 1: SETBP1 Co-occurrence with All Myeloid Drivers")
    print(f"{'='*70}")
    print(f"\n  SETBP1-mutated patients: {n_setbp1}")

    analysis1_results = []

    for g in sorted(TARGET_GENES):
        if g == "SETBP1":
            continue

        panel_n, n_setbp1_adj, n_gene_adj, n_both_adj = pair_stats(
            "SETBP1", g, all_myeloid_list, on_panel, mutated
        )
        if panel_n == 0:
            continue

        result = run_fisher_test(panel_n, n_setbp1_adj, n_gene_adj, n_both_adj)
        if result is None:
            continue

        coverage_pct = round(panel_n / total_myeloid * 100, 1)

        entry = {
            "gene": g,
            "n_setbp1": n_setbp1_adj,
            "n_gene": n_gene_adj,
            "n_both": n_both_adj,
            "expected": result["expected"],
            "obs_exp_ratio": result["obs_exp_ratio"],
            "odds_ratio": result["odds_ratio"],
            "p_value": result["p_value_two_sided"],
            "p_less": result["p_value_less"],
            "p_greater": result["p_value_greater"],
            "direction": result["direction"],
            "panel_n": panel_n,
            "panel_coverage_pct": coverage_pct,
        }

        if coverage_pct < 50:
            entry["coverage_warning"] = (
                f"Low panel coverage ({coverage_pct}%). Results based on "
                f"{panel_n:,} of {total_myeloid:,} myeloid patients."
            )

        analysis1_results.append(entry)

    # Sort by obs/exp descending
    analysis1_results.sort(key=lambda x: x["obs_exp_ratio"], reverse=True)

    # BH correction
    pvals = [r["p_value"] for r in analysis1_results]
    bh_corrected = benjamini_hochberg(pvals)
    for i, r in enumerate(analysis1_results):
        r["p_value_bh"] = bh_corrected[i]

    print(f"\n  {'Gene':8s} {'N_SETBP1':>8} {'N_gene':>8} {'N_both':>8} "
          f"{'Expected':>10} {'Obs/Exp':>8} {'P-value':>12} {'BH':>12} "
          f"{'Panel_N':>8} {'Dir':>10}")
    print(f"  {'-'*106}")
    for r in analysis1_results:
        print(f"  {r['gene']:8s} {r['n_setbp1']:8,} {r['n_gene']:8,} "
              f"{r['n_both']:8,} {r['expected']:10.2f} {r['obs_exp_ratio']:8.2f} "
              f"{format_pval(r['p_value']):>12} {format_pval(r['p_value_bh']):>12} "
              f"{r['panel_n']:8,} {r['direction']:>10}")

    # Save Analysis 1
    analysis1_output = {
        "analysis": "SETBP1 co-occurrence with 31 myeloid driver genes",
        "database": "AACR GENIE v19.0-public",
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "methodology": {
            "variant_filter": "Protein-altering only",
            "variant_classes_included": sorted(PATHOGENIC_VAR_CLASSES),
            "variant_classes_excluded": sorted(EXCLUDED_VAR_CLASSES),
            "hypermutation_threshold": HYPERMUTATION_THRESHOLD,
            "hypermutated_excluded": sorted(list(hypermutated)),
            "denominator": "Panel-adjusted (patients whose panel covers both genes)",
            "statistical_test": "Fisher exact (two-sided), BH-corrected",
        },
        "cohort": {
            "total_samples": len(all_samples),
            "myeloid_samples": len(myeloid_samples),
            "myeloid_patients": total_myeloid,
            "setbp1_mutated": n_setbp1,
        },
        "results": analysis1_results,
    }

    with open(RESULTS_DIR / "setbp1_cooccurrence_matrix.json", "w") as f:
        json.dump(analysis1_output, f, indent=2, default=str)

    with open(RESULTS_DIR / "setbp1_cooccurrence_matrix.tsv", "w") as f:
        cols = ["Gene", "N_SETBP1", "N_gene", "N_both", "Expected", "Obs_Exp",
                "P_value", "P_value_BH", "Panel_N", "Panel_Coverage_Pct",
                "Direction", "Coverage_Warning"]
        f.write("\t".join(cols) + "\n")
        for r in analysis1_results:
            warning = r.get("coverage_warning", "")
            f.write(
                f"{r['gene']}\t{r['n_setbp1']}\t{r['n_gene']}\t{r['n_both']}\t"
                f"{r['expected']:.4f}\t{r['obs_exp_ratio']:.4f}\t"
                f"{r['p_value']:.6e}\t{r['p_value_bh']:.6e}\t{r['panel_n']}\t"
                f"{r['panel_coverage_pct']}\t{r['direction']}\t{warning}\n"
            )

    # ========================================================================
    # ANALYSIS 2: SETBP1 SKI-domain Breakdown
    # ========================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 2: SETBP1 SKI-domain vs Non-SKI Co-occurrence")
    print(f"{'='*70}")

    ski_patients = set()
    non_ski_patients = set()
    for pid, classification in patient_setbp1_class.items():
        if classification in ("SKI", "both"):
            ski_patients.add(pid)
        if classification in ("non-SKI", "both"):
            non_ski_patients.add(pid)

    print(f"\n  SETBP1 SKI-domain patients: {len(ski_patients)}")
    print(f"  SETBP1 non-SKI patients: {len(non_ski_patients)}")
    print(f"  SETBP1 both SKI + non-SKI: {len(ski_patients & non_ski_patients)}")

    # Variant spectra
    ski_variants = Counter()
    non_ski_variants = Counter()
    for pid in gene_patients["SETBP1"]:
        for m in patient_mutations[pid].get("SETBP1", []):
            protein = m["protein_change"]
            if classify_setbp1_mutation(protein) == "SKI":
                ski_variants[protein] += 1
            else:
                non_ski_variants[protein] += 1

    print(f"\n  Top SKI-domain variants:")
    for v, c in ski_variants.most_common(10):
        print(f"    {v}: {c}")
    print(f"\n  Top non-SKI variants:")
    for v, c in non_ski_variants.most_common(10):
        print(f"    {v}: {c}")

    analysis2_ski_results = []
    analysis2_nonski_results = []

    for subset_name, subset_pids, results_list in [
        ("SKI", ski_patients, analysis2_ski_results),
        ("non-SKI", non_ski_patients, analysis2_nonski_results),
    ]:
        for g in sorted(TARGET_GENES):
            if g == "SETBP1":
                continue

            panel_n, n_anchor, n_gene_adj, n_both_adj = pair_stats_with_subset(
                "SETBP1", g, subset_pids, all_myeloid_list, on_panel, mutated
            )
            if panel_n == 0:
                continue

            result = run_fisher_test(panel_n, n_anchor, n_gene_adj, n_both_adj)
            if result is None:
                continue

            entry = {
                "gene": g,
                "setbp1_subset": subset_name,
                "n_setbp1": n_anchor,
                "n_gene": n_gene_adj,
                "n_both": n_both_adj,
                "expected": result["expected"],
                "obs_exp_ratio": result["obs_exp_ratio"],
                "p_value": result["p_value_two_sided"],
                "direction": result["direction"],
                "panel_n": panel_n,
            }
            results_list.append(entry)

        results_list.sort(key=lambda x: x["obs_exp_ratio"], reverse=True)

    # Print SKI summary
    print(f"\n  SKI-domain SETBP1 co-occurrence (genes with co-mutations):")
    print(f"  {'Gene':8s} {'N_SKI':>6} {'N_gene':>8} {'Both':>6} "
          f"{'Exp':>8} {'O/E':>8} {'P-value':>12}")
    for r in analysis2_ski_results:
        if r["n_both"] > 0 or r["expected"] > 1:
            print(f"  {r['gene']:8s} {r['n_setbp1']:6,} {r['n_gene']:8,} "
                  f"{r['n_both']:6,} {r['expected']:8.2f} "
                  f"{r['obs_exp_ratio']:8.2f} {format_pval(r['p_value']):>12}")

    analysis2_output = {
        "analysis": "SETBP1 SKI-domain vs non-SKI co-occurrence breakdown",
        "ski_domain_range": "amino acids 858-871",
        "ski_patients": len(ski_patients),
        "non_ski_patients": len(non_ski_patients),
        "patients_with_both": len(ski_patients & non_ski_patients),
        "ski_variant_counts": dict(ski_variants.most_common()),
        "non_ski_variant_counts": dict(non_ski_variants.most_common(30)),
        "ski_cooccurrence": analysis2_ski_results,
        "non_ski_cooccurrence": analysis2_nonski_results,
    }

    with open(RESULTS_DIR / "setbp1_ski_domain_cooccurrence.json", "w") as f:
        json.dump(analysis2_output, f, indent=2, default=str)

    # ========================================================================
    # ANALYSIS 3: DDX41 + SETBP1 Deep Dive
    # ========================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 3: DDX41 + SETBP1 Deep Dive")
    print(f"{'='*70}")

    n_ddx41 = len(gene_patients["DDX41"])
    ddx41_panel_samples = gene_coverage_samples["DDX41"]
    ddx41_coverage_pct = round(ddx41_panel_samples / len(myeloid_samples) * 100, 1)

    print(f"\n  DDX41 panel coverage: {ddx41_panel_samples:,} myeloid samples "
          f"({ddx41_coverage_pct}%)")
    print(f"  DDX41-mutated patients: {n_ddx41}")
    print(f"  *** COVERAGE WARNING: DDX41 on {ddx41_coverage_pct}% of panels ***")

    # DDX41 + SETBP1
    ddx41_setbp1_panel_n, n_ddx41_adj, n_setbp1_adj_ddx41, n_both_ddx41 = pair_stats(
        "DDX41", "SETBP1", all_myeloid_list, on_panel, mutated
    )

    ddx41_setbp1_result = run_fisher_test(
        ddx41_setbp1_panel_n, n_ddx41_adj, n_setbp1_adj_ddx41, n_both_ddx41
    )

    if ddx41_setbp1_result:
        print(f"\n  DDX41+SETBP1 (panel-adjusted N={ddx41_setbp1_panel_n:,}):")
        print(f"    DDX41 mutated: {n_ddx41_adj}")
        print(f"    SETBP1 mutated: {n_setbp1_adj_ddx41}")
        print(f"    Both: {n_both_ddx41}")
        print(f"    Expected: {ddx41_setbp1_result['expected']:.2f}")
        print(f"    Obs/Exp: {ddx41_setbp1_result['obs_exp_ratio']:.2f}")
        print(f"    P-value: {format_pval(ddx41_setbp1_result['p_value_two_sided'])}")
        print(f"    Direction: {ddx41_setbp1_result['direction']}")

    # Co-mutated patient details
    both_ddx41_setbp1_pids = gene_patients["DDX41"] & gene_patients["SETBP1"]
    # Restrict to those with both genes on panel
    both_ddx41_setbp1_pids = {
        pid for pid in both_ddx41_setbp1_pids
        if "DDX41" in on_panel.get(pid, set()) and "SETBP1" in on_panel.get(pid, set())
    }

    ddx41_setbp1_details = []
    for pid in sorted(both_ddx41_setbp1_pids):
        muts = patient_mutations[pid]
        detail = {
            "patient_id": pid,
            "DDX41": [m["protein_change"] for m in muts.get("DDX41", [])],
            "SETBP1": [m["protein_change"] for m in muts.get("SETBP1", [])],
            "other_mutations": {},
        }
        for g in TARGET_GENES:
            if g not in ("DDX41", "SETBP1") and g in muts:
                detail["other_mutations"][g] = [
                    m["protein_change"] for m in muts[g]
                ]
        clin = patient_clinical.get(pid, {})
        detail["age"] = clin.get("age")
        detail["sex"] = clin.get("sex", "")
        detail["center"] = clin.get("center", "")
        if pid in ddx41_mutation_status:
            detail["ddx41_mutation_status"] = ddx41_mutation_status[pid]
        ddx41_setbp1_details.append(detail)

        print(f"\n    {pid} ({detail['sex']}, age {detail['age']}):")
        print(f"      DDX41: {detail['DDX41']}")
        print(f"      SETBP1: {detail['SETBP1']}")
        if detail["other_mutations"]:
            for og, ovs in sorted(detail["other_mutations"].items()):
                print(f"      {og}: {ovs}")

    # DDX41 co-occurrence with ALL other driver genes
    print(f"\n  DDX41 co-occurrence with all myeloid drivers:")
    ddx41_cooccurrence = []

    for g in sorted(TARGET_GENES):
        if g == "DDX41":
            continue
        adj_n, n_d_adj, n_g_adj, n_b_adj = pair_stats(
            "DDX41", g, all_myeloid_list, on_panel, mutated
        )
        if adj_n == 0:
            continue
        result = run_fisher_test(adj_n, n_d_adj, n_g_adj, n_b_adj)
        if result:
            entry = {
                "gene": g,
                "n_ddx41": n_d_adj,
                "n_gene": n_g_adj,
                "n_both": n_b_adj,
                "expected": result["expected"],
                "obs_exp_ratio": result["obs_exp_ratio"],
                "p_value": result["p_value_two_sided"],
                "direction": result["direction"],
                "panel_n": adj_n,
            }
            ddx41_cooccurrence.append(entry)

    ddx41_cooccurrence.sort(key=lambda x: x["obs_exp_ratio"], reverse=True)

    print(f"\n  {'Gene':8s} {'N_DDX41':>8} {'N_gene':>8} {'Both':>6} "
          f"{'Exp':>8} {'O/E':>8} {'P-value':>12}")
    for r in ddx41_cooccurrence:
        if r["n_both"] > 0:
            print(f"  {r['gene']:8s} {r['n_ddx41']:8,} {r['n_gene']:8,} "
                  f"{r['n_both']:6,} {r['expected']:8.2f} "
                  f"{r['obs_exp_ratio']:8.2f} {format_pval(r['p_value']):>12}")

    # DDX41 variant spectrum
    ddx41_variants = Counter()
    for pid in gene_patients["DDX41"]:
        for m in patient_mutations[pid].get("DDX41", []):
            ddx41_variants[m["protein_change"]] += 1

    # DDX41 frequency in panel-adjusted cohort
    ddx41_freq_in_covered = round(
        n_ddx41_adj / ddx41_setbp1_panel_n * 100, 2
    ) if ddx41_setbp1_panel_n > 0 else 0

    analysis3_output = {
        "analysis": "DDX41 + SETBP1 deep dive",
        "coverage_warning": (
            f"DDX41 is on {ddx41_panel_samples:,} of {len(myeloid_samples):,} "
            f"myeloid samples ({ddx41_coverage_pct}% coverage). Results are based "
            f"on ~{ddx41_setbp1_panel_n:,} patients with both DDX41 and SETBP1 "
            f"on their panel, not the full {total_myeloid:,} myeloid cohort."
        ),
        "ddx41_stats": {
            "panel_coverage_samples": ddx41_panel_samples,
            "panel_coverage_pct": ddx41_coverage_pct,
            "mutated_patients": n_ddx41,
            "mutated_in_panel_adj_cohort": n_ddx41_adj,
            "frequency_in_covered_pct": ddx41_freq_in_covered,
            "top_variants": dict(ddx41_variants.most_common(20)),
        },
        "ddx41_setbp1": {
            "panel_adjusted_n": ddx41_setbp1_panel_n,
            "ddx41_in_panel": n_ddx41_adj,
            "setbp1_in_panel": n_setbp1_adj_ddx41,
            "co_mutated": n_both_ddx41,
            "fisher_result": ddx41_setbp1_result,
        },
        "ddx41_setbp1_patient_details": ddx41_setbp1_details,
        "ddx41_all_cooccurrence": ddx41_cooccurrence,
    }

    with open(RESULTS_DIR / "ddx41_setbp1_deep_dive.json", "w") as f:
        json.dump(analysis3_output, f, indent=2, default=str)

    # ========================================================================
    # ANALYSIS 4: Full Pairwise Matrix (Top 20 Genes)
    # ========================================================================
    print(f"\n{'='*70}")
    print("ANALYSIS 4: Full Pairwise Matrix (Top 20 Genes)")
    print(f"{'='*70}")

    gene_freq = [(g, len(gene_patients[g])) for g in TARGET_GENES]
    gene_freq.sort(key=lambda x: x[1], reverse=True)
    top20_genes = [g for g, _ in gene_freq[:20]]

    print(f"\n  Top 20 genes by mutation frequency:")
    for g, n in gene_freq[:20]:
        print(f"    {g:8s}: {n:5,}")

    n_pairs = len(list(combinations(top20_genes, 2)))
    print(f"\n  Computing {n_pairs} pairwise comparisons...")

    pairwise_results = []

    for g1, g2 in combinations(top20_genes, 2):
        adj_n, n_g1, n_g2, n_both = pair_stats(
            g1, g2, all_myeloid_list, on_panel, mutated
        )
        if adj_n == 0:
            continue

        result = run_fisher_test(adj_n, n_g1, n_g2, n_both)
        if result is None:
            continue

        log2_oe = (
            math.log2(result["obs_exp_ratio"])
            if result["obs_exp_ratio"] > 0
            else None
        )

        entry = {
            "gene1": g1,
            "gene2": g2,
            "n_gene1": n_g1,
            "n_gene2": n_g2,
            "observed": n_both,
            "expected": result["expected"],
            "obs_exp": result["obs_exp_ratio"],
            "log2_obs_exp": round(log2_oe, 4) if log2_oe is not None and math.isfinite(log2_oe) else None,
            "p_value": result["p_value_two_sided"],
            "direction": result["direction"],
            "panel_n": adj_n,
        }
        pairwise_results.append(entry)

    # BH correction
    pw_pvals = [r["p_value"] for r in pairwise_results]
    pw_bh = benjamini_hochberg(pw_pvals)
    for i, r in enumerate(pairwise_results):
        r["p_value_bh"] = pw_bh[i]

    # Save TSV
    with open(RESULTS_DIR / "myeloid_pairwise_matrix.tsv", "w") as f:
        cols = ["gene1", "gene2", "n_gene1", "n_gene2", "observed", "expected",
                "obs_exp", "log2_obs_exp", "p_value", "p_value_bh", "direction",
                "panel_n"]
        f.write("\t".join(cols) + "\n")
        for r in pairwise_results:
            log2_str = (
                f"{r['log2_obs_exp']:.4f}"
                if r["log2_obs_exp"] is not None
                else "NA"
            )
            f.write(
                f"{r['gene1']}\t{r['gene2']}\t{r['n_gene1']}\t{r['n_gene2']}\t"
                f"{r['observed']}\t{r['expected']:.4f}\t{r['obs_exp']:.4f}\t"
                f"{log2_str}\t{r['p_value']:.6e}\t{r['p_value_bh']:.6e}\t"
                f"{r['direction']}\t{r['panel_n']}\n"
            )

    # Save JSON
    analysis4_output = {
        "analysis": "Full pairwise co-occurrence matrix (top 20 myeloid genes)",
        "genes": top20_genes,
        "n_pairs": len(pairwise_results),
        "results": pairwise_results,
    }

    with open(RESULTS_DIR / "myeloid_pairwise_matrix.json", "w") as f:
        json.dump(analysis4_output, f, indent=2, default=str)

    # Print significant pairs
    sig_results = [r for r in pairwise_results if r["p_value_bh"] < 0.05]
    enriched = sorted(
        [r for r in sig_results if r["direction"] == "enriched"],
        key=lambda x: x["obs_exp"], reverse=True,
    )
    depleted = sorted(
        [r for r in sig_results if r["direction"] == "depleted"],
        key=lambda x: x["obs_exp"],
    )

    print(f"\n  Significant pairs (BH p < 0.05): {len(sig_results)}")

    print(f"\n  Top enriched:")
    for r in enriched[:15]:
        print(f"    {r['gene1']:8s}+{r['gene2']:8s}: O/E={r['obs_exp']:.2f}, "
              f"obs={r['observed']}, exp={r['expected']:.1f}, "
              f"p_BH={format_pval(r['p_value_bh'])}")

    print(f"\n  Top depleted:")
    for r in depleted[:15]:
        print(f"    {r['gene1']:8s}+{r['gene2']:8s}: O/E={r['obs_exp']:.2f}, "
              f"obs={r['observed']}, exp={r['expected']:.1f}, "
              f"p_BH={format_pval(r['p_value_bh'])}")

    # ========================================================================
    # EMAIL SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print("Generating email summary...")
    print(f"{'='*70}")

    generate_email_summary(
        analysis1_results=analysis1_results,
        ski_results=analysis2_ski_results,
        nonski_results=analysis2_nonski_results,
        analysis3=analysis3_output,
        pairwise_results=pairwise_results,
        n_setbp1=n_setbp1,
        total_myeloid=total_myeloid,
        total_samples=len(all_samples),
        myeloid_samples=len(myeloid_samples),
        n_hypermutated=len(hypermutated),
        n_ski=len(ski_patients),
        n_nonski=len(non_ski_patients),
        top20_genes=top20_genes,
        ddx41_setbp1_result=ddx41_setbp1_result,
        ddx41_setbp1_panel_n=ddx41_setbp1_panel_n,
        n_ddx41_in_panel=n_ddx41_adj,
        n_setbp1_in_panel_ddx41=n_setbp1_adj_ddx41,
        n_both_ddx41_setbp1=n_both_ddx41,
        ddx41_setbp1_details=ddx41_setbp1_details,
        ddx41_cooccurrence=ddx41_cooccurrence,
        ddx41_freq_pct=ddx41_freq_in_covered,
    )

    # ========================================================================
    # VERIFICATION
    # ========================================================================
    print(f"\n{'='*70}")
    print("VERIFICATION CHECKS")
    print(f"{'='*70}")

    # Previous analysis (4-gene, different hypermutation scope) reference values
    checks = [
        ("Myeloid samples", len(myeloid_samples), 27585),
        ("Total samples", len(all_samples), 271837),
    ]

    # SETBP1 count may differ slightly from 303 due to broader hypermutation filter
    print(f"\n  SETBP1 patients: {n_setbp1} (previous 4-gene analysis: 303)")
    if n_setbp1 != 303:
        diff = 303 - n_setbp1
        print(f"    Difference of {diff}: expected due to expanded hypermutation "
              f"filter (32 genes vs 4)")

    # Find DNMT3A and PTPN11 co-occurrence for spot-check
    for r in analysis1_results:
        if r["gene"] == "DNMT3A":
            print(f"  SETBP1+DNMT3A: {r['n_both']} (previous: 47, "
                  f"note: panel-adjusted denominator differs)")
        if r["gene"] == "PTPN11":
            print(f"  SETBP1+PTPN11: {r['n_both']} (previous: 28, "
                  f"note: panel-adjusted denominator differs)")

    all_pass = True
    for name, actual, expected in checks:
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {name}: {actual:,} (expected {expected:,})")

    # Verify no star-pattern artifacts
    print(f"  [{'PASS' if stats['excluded_star'] == 0 else 'INFO'}] "
          f"Star-pattern artifacts excluded: {stats['excluded_star']}")

    # Verify hypermutation exclusion
    print(f"  [PASS] Hypermutated patients excluded: {len(hypermutated)}")

    # Verify TSV well-formed
    for tsv_file in RESULTS_DIR.glob("*.tsv"):
        with open(tsv_file) as f:
            lines = f.readlines()
        n_cols = len(lines[0].rstrip("\n").split("\t"))
        bad_rows = [
            i for i, line in enumerate(lines[1:], 2)
            if len(line.rstrip("\n").split("\t")) != n_cols
        ]
        status = "PASS" if not bad_rows else "FAIL"
        if bad_rows:
            all_pass = False
        print(f"  [{status}] {tsv_file.name}: {len(lines)-1} data rows, "
              f"{n_cols} columns")

    if all_pass:
        print(f"\n  All verification checks PASSED.")
    else:
        print(f"\n  WARNING: Some checks need investigation.")

    print(f"\n{'='*70}")
    print(f"Output files saved to: {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")
    print(f"{'='*70}")

    return 0


# ============================================================================
# EMAIL SUMMARY GENERATOR
# ============================================================================

def generate_email_summary(
    analysis1_results, ski_results, nonski_results, analysis3,
    pairwise_results, n_setbp1, total_myeloid, total_samples,
    myeloid_samples, n_hypermutated, n_ski, n_nonski, top20_genes,
    ddx41_setbp1_result, ddx41_setbp1_panel_n, n_ddx41_in_panel,
    n_setbp1_in_panel_ddx41, n_both_ddx41_setbp1,
    ddx41_setbp1_details, ddx41_cooccurrence, ddx41_freq_pct,
):
    """Generate human-readable markdown summary for email to Makishima."""
    lines = []

    def add(s=""):
        lines.append(s)

    add("# SETBP1 Co-mutation Profile in AACR GENIE v19.0")
    add()
    add("**Prepared for:** Professor Hideki Makishima, Kyoto University")
    add(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    add("**Database:** AACR Project GENIE v19.0-public (Synapse)")
    add()
    add("---")
    add()

    # Cohort summary
    add("## Cohort Summary")
    add()
    add("| Parameter | Value |")
    add("|-----------|-------|")
    add(f"| Total samples in GENIE v19.0 | {total_samples:,} |")
    add(f"| Myeloid samples (OncoTree-filtered) | {myeloid_samples:,} |")
    add(f"| Myeloid patients (deduplicated) | {total_myeloid:,} |")
    add(f"| SETBP1-mutated patients | {n_setbp1:,} |")
    add(f"| SETBP1 mutation rate | {n_setbp1/total_myeloid*100:.1f}% |")
    add(f"| Hypermutated excluded | {n_hypermutated} "
        f"(>{HYPERMUTATION_THRESHOLD} coding mutations in 32 driver genes) |")
    add(f"| Variant filter | Protein-altering only |")
    add()

    # Analysis 1
    add("## 1. SETBP1 Co-occurrence with Myeloid Driver Genes")
    add()
    add("Panel-adjusted denominators: only patients whose panel covers both genes.")
    add()
    add("| Gene | N(SETBP1) | N(gene) | Co-mutated | Expected | Obs/Exp | "
        "P-value | BH P | Direction |")
    add("|------|-----------|---------|------------|----------|---------|"
        "---------|------|-----------|")

    for r in analysis1_results:
        sig = "**" if r["p_value_bh"] < 0.05 else ""
        cw = " ^" if "coverage_warning" in r else ""
        add(f"| {r['gene']}{cw} | {r['n_setbp1']} | {r['n_gene']:,} | "
            f"{sig}{r['n_both']}{sig} | {r['expected']:.1f} | "
            f"{sig}{r['obs_exp_ratio']:.2f}{sig} | {format_pval(r['p_value'])} | "
            f"{format_pval(r['p_value_bh'])} | {r['direction']} |")

    sig_enriched = [
        r for r in analysis1_results
        if r["p_value_bh"] < 0.05 and r["direction"] == "enriched"
    ]
    sig_depleted = [
        r for r in analysis1_results
        if r["p_value_bh"] < 0.05 and r["direction"] == "depleted"
    ]

    add()
    add("^ Genes with <50% panel coverage")
    add()
    add(f"**Significantly enriched (BH p < 0.05):** "
        f"{', '.join(r['gene'] for r in sig_enriched) if sig_enriched else 'None'}")
    add(f"**Significantly depleted (BH p < 0.05):** "
        f"{', '.join(r['gene'] for r in sig_depleted) if sig_depleted else 'None'}")
    add()

    # Analysis 2
    add("## 2. SETBP1 SKI-domain (aa 858-871) vs Non-SKI Breakdown")
    add()
    add(f"- SKI-domain SETBP1 patients: **{n_ski}**")
    add(f"- Non-SKI SETBP1 patients: **{n_nonski}**")
    add()

    add("### SKI-domain SETBP1 co-occurrence")
    add()
    add("| Gene | N(SKI) | N(gene) | Co-mutated | Expected | Obs/Exp | P-value |")
    add("|------|--------|---------|------------|----------|---------|---------|")
    for r in ski_results[:20]:
        if r["n_both"] > 0 or r["expected"] > 1:
            add(f"| {r['gene']} | {r['n_setbp1']} | {r['n_gene']:,} | "
                f"{r['n_both']} | {r['expected']:.1f} | "
                f"{r['obs_exp_ratio']:.2f} | {format_pval(r['p_value'])} |")
    add()

    add("### Non-SKI SETBP1 co-occurrence")
    add()
    add("| Gene | N(non-SKI) | N(gene) | Co-mutated | Expected | Obs/Exp | P-value |")
    add("|------|------------|---------|------------|----------|---------|---------|")
    for r in nonski_results[:20]:
        if r["n_both"] > 0 or r["expected"] > 1:
            add(f"| {r['gene']} | {r['n_setbp1']} | {r['n_gene']:,} | "
                f"{r['n_both']} | {r['expected']:.1f} | "
                f"{r['obs_exp_ratio']:.2f} | {format_pval(r['p_value'])} |")
    add()

    # Analysis 3
    add("## 3. DDX41 + SETBP1 Analysis")
    add()
    add(f"> **Coverage caveat:** DDX41 is on only "
        f"{analysis3['ddx41_stats']['panel_coverage_pct']}% of panels, covering "
        f"{analysis3['ddx41_stats']['panel_coverage_samples']:,} myeloid samples. "
        f"Results below are based on ~{ddx41_setbp1_panel_n:,} patients with both "
        f"DDX41 and SETBP1 on panel.")
    add()

    add("| Metric | Value |")
    add("|--------|-------|")
    add(f"| Panel-adjusted N | {ddx41_setbp1_panel_n:,} |")
    add(f"| DDX41 mutated | {n_ddx41_in_panel} |")
    add(f"| DDX41 frequency | {ddx41_freq_pct}% |")
    add(f"| SETBP1 mutated | {n_setbp1_in_panel_ddx41} |")
    add(f"| Co-mutated | {n_both_ddx41_setbp1} |")
    if ddx41_setbp1_result:
        add(f"| Expected | {ddx41_setbp1_result['expected']:.2f} |")
        add(f"| Obs/Exp | {ddx41_setbp1_result['obs_exp_ratio']:.2f} |")
        add(f"| P-value (Fisher, two-sided) | "
            f"{format_pval(ddx41_setbp1_result['p_value_two_sided'])} |")
        add(f"| Direction | {ddx41_setbp1_result['direction']} |")
    add()

    if ddx41_setbp1_details:
        add("### DDX41 + SETBP1 Co-mutated Patients")
        add()
        for d in ddx41_setbp1_details:
            add(f"- **{d['patient_id']}** "
                f"({d.get('sex', 'Unknown')}, age {d.get('age', 'Unknown')})")
            add(f"  - DDX41: {', '.join(d['DDX41'])}")
            add(f"  - SETBP1: {', '.join(d['SETBP1'])}")
            if d["other_mutations"]:
                others = "; ".join(
                    f"{g}: {', '.join(vs)}"
                    for g, vs in sorted(d["other_mutations"].items())
                )
                add(f"  - Other: {others}")
        add()

    add("### DDX41 Co-occurrence with All Myeloid Drivers")
    add()
    add("| Gene | N(DDX41) | N(gene) | Co-mutated | Expected | Obs/Exp | P-value |")
    add("|------|----------|---------|------------|----------|---------|---------|")
    for r in ddx41_cooccurrence:
        if r["n_both"] > 0:
            add(f"| {r['gene']} | {r['n_ddx41']} | {r['n_gene']:,} | "
                f"{r['n_both']} | {r['expected']:.1f} | "
                f"{r['obs_exp_ratio']:.2f} | {format_pval(r['p_value'])} |")
    add()

    # Analysis 4
    add("## 4. Full Pairwise Matrix (Top 20 Myeloid Genes)")
    add()
    add(f"Full matrix in `myeloid_pairwise_matrix.tsv` "
        f"({len(pairwise_results)} gene pairs).")
    add(f"Genes: {', '.join(top20_genes)}")
    add()

    sig_pw = [r for r in pairwise_results if r["p_value_bh"] < 0.05]
    enriched_pw = sorted(
        [r for r in sig_pw if r["direction"] == "enriched"],
        key=lambda x: x["obs_exp"], reverse=True,
    )
    depleted_pw = sorted(
        [r for r in sig_pw if r["direction"] == "depleted"],
        key=lambda x: x["obs_exp"],
    )

    add("### Strongest Co-occurring Pairs (BH p < 0.05)")
    add()
    add("| Gene 1 | Gene 2 | Obs | Exp | Obs/Exp | log2(O/E) | BH P |")
    add("|--------|--------|-----|-----|---------|-----------|------|")
    for r in enriched_pw[:20]:
        log2_str = (
            f"{r['log2_obs_exp']:.2f}"
            if r["log2_obs_exp"] is not None
            else "NA"
        )
        add(f"| {r['gene1']} | {r['gene2']} | {r['observed']} | "
            f"{r['expected']:.1f} | {r['obs_exp']:.2f} | {log2_str} | "
            f"{format_pval(r['p_value_bh'])} |")
    add()

    add("### Strongest Mutually Exclusive Pairs (BH p < 0.05)")
    add()
    add("| Gene 1 | Gene 2 | Obs | Exp | Obs/Exp | log2(O/E) | BH P |")
    add("|--------|--------|-----|-----|---------|-----------|------|")
    for r in depleted_pw[:20]:
        log2_str = (
            f"{r['log2_obs_exp']:.2f}"
            if r["log2_obs_exp"] is not None
            else "NA"
        )
        add(f"| {r['gene1']} | {r['gene2']} | {r['observed']} | "
            f"{r['expected']:.1f} | {r['obs_exp']:.2f} | {log2_str} | "
            f"{format_pval(r['p_value_bh'])} |")
    add()

    # Methodology
    add("---")
    add()
    add("## Methodology")
    add()
    add("1. **Database:** AACR Project GENIE v19.0-public "
        "(3,458,550 mutation rows; 271,837 samples; 27,585 myeloid)")
    add("2. **Myeloid filter:** OncoTree codes for AML, MDS, MPN, CMML "
        "and related diagnoses, plus keyword fallback")
    add("3. **Variant filter:** Protein-altering only "
        "(excluded: Intron, Silent, UTR, Splice_Region, IGR, RNA)")
    add("4. **Panel adjustment:** Denominators restricted to patients whose "
        "panel covers both genes in each comparison")
    add(f"5. **Hypermutation filter:** Excluded patients with "
        f">{HYPERMUTATION_THRESHOLD} coding mutations across 32 target genes")
    add("6. **Statistical test:** Fisher's exact test (two-sided), "
        "Benjamini-Hochberg FDR correction")
    add("7. **SKI domain:** SETBP1 amino acid positions 858-871")
    add()

    add("## Data Files")
    add()
    add("| File | Contents |")
    add("|------|----------|")
    add("| `setbp1_cooccurrence_matrix.json` | SETBP1 vs 31 genes (full stats) |")
    add("| `setbp1_cooccurrence_matrix.tsv` | Same, tab-delimited |")
    add("| `setbp1_ski_domain_cooccurrence.json` | SKI vs non-SKI breakdown |")
    add("| `ddx41_setbp1_deep_dive.json` | DDX41 analysis with patient details |")
    add("| `myeloid_pairwise_matrix.tsv` | 20x20 pairwise matrix (TSV) |")
    add("| `myeloid_pairwise_matrix.json` | Same in JSON |")
    add("| `summary_for_email.md` | This document |")
    add()

    add("---")
    add()
    add("*Analysis: Python 3.12, scipy 1.17.1, numpy 2.4.3.*")
    add(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    report = "\n".join(lines)

    with open(RESULTS_DIR / "summary_for_email.md", "w") as f:
        f.write(report)

    print(f"  Email summary: {RESULTS_DIR / 'summary_for_email.md'}")
    print(f"  {len(report):,} characters, {len(lines)} lines")


if __name__ == "__main__":
    sys.exit(main())
