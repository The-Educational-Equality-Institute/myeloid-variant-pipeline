#!/usr/bin/env python3
"""
four_gene_cooccurrence.py -- Four-gene co-occurrence analysis in GENIE myeloid cohort.

Searches AACR GENIE v19.0 for co-occurrence of DNMT3A R882H + IDH2 R140Q +
SETBP1 G870S + PTPN11 E76Q across myeloid neoplasms.

Methodology:
  1. Load GENIE mutation, clinical, and gene panel data
  2. Filter to myeloid samples (OncoTree codes)
  3. Filter to coding variants (exclude non-coding classes)
  4. Hypermutation filter: exclude samples with >20 coding mutations in target genes
  5. Panel adjustment: only count samples where BOTH genes in a pair are on the panel
  6. Pairwise co-occurrence with Fisher's exact test + Benjamini-Hochberg correction
  7. Triple and quadruple co-occurrence analysis
  8. Progressive funnel: starting from most common mutation, narrow down

Inputs:
    - mutation_profile/data/genie/raw/data_mutations_extended.txt
    - mutation_profile/data/genie/raw/data_clinical_sample.txt
    - mutation_profile/data/genie/raw/genomic_information_*.txt

Outputs:
    - mutation_profile/results/cooccurrence/four_gene_cooccurrence.json
    - mutation_profile/results/cooccurrence/four_gene_cooccurrence.tsv

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/four_gene_cooccurrence.py

Runtime: ~14 seconds
Dependencies: pandas, numpy, scipy, statsmodels
"""

import os
import sys
import json
import glob
import logging
import warnings
from datetime import datetime
from itertools import combinations
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ---------------------------------------------------------------------------
# Paths — relative to this script so it works from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "mutation_profile", "data", "genie", "raw")
RESULTS_DIR = os.path.join(PROJECT_DIR, "mutation_profile", "results", "cooccurrence")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_GENES = sorted([
    "ASXL1", "TET2", "SRSF2", "SF3B1", "RUNX1", "TP53", "FLT3", "NPM1",
    "NRAS", "KRAS", "CBL", "EZH2", "U2AF1", "STAG2", "BCOR", "BCORL1",
    "DDX41", "DNMT3A", "IDH1", "IDH2", "PTPN11", "JAK2", "CALR", "MPL",
    "PHF6", "WT1", "CEBPA", "GATA2", "ZRSR2", "RAD21", "SMC1A", "SMC3",
    "SETBP1", "CSF3R",
])

# The specific mutation profile we are searching for
PATIENT_MUTATIONS = {
    "DNMT3A": "p.R882H",
    "IDH2":   "p.R140Q",
    "SETBP1": "p.G870S",
    "PTPN11": "p.E76Q",
}

# Comprehensive myeloid OncoTree codes in GENIE v19.0
# Derived from CANCER_TYPE fields: Leukemia (myeloid subset), Myelodysplastic
# Syndromes, MDS/MPN, MPN, Mastocytosis, Myeloid Neoplasms w/ Germ Line Pred.
# Also includes the legacy/short-form codes from the original specification.
MYELOID_CODES = {
    # --- AML and myeloid leukemias (from CANCER_TYPE = "Leukemia") ---
    "AML", "AMLNOS", "AMLMRC", "AMLNPM1", "AMLRUNX1RUNX1T1",
    "AMLCBFBMYH11", "AMLMLLT3KMT2A", "AMLGATA2MECOM", "AMLRUNX1",
    "AMLCEBPA", "AMLDEKNUP214", "AMLRGA", "AMLBCRABL1", "AMLMD",
    "AMLRARA", "AMLRBM15MKL1", "APLPMLRARA",
    "AM", "AMKL", "AMML", "AMOL", "APMF", "AWM",
    "TAML", "TMDS", "TMN", "TAM",  # therapy-related
    "MS",  # myeloid sarcoma
    "PERL",  # pure erythroid leukemia
    "MPRDS",  # myeloproliferative/dysplastic syndrome
    "MLADS", "MLNER", "MLNFGFR1", "MLNPCM1JAK2", "MLNPDGFRA", "MLNPDGFRB",
    # Mixed-phenotype acute leukemia (includes myeloid component)
    "MPALBCRABL1", "MPALBNOS", "MPALKMT2A", "MPALTNOS",
    "ALAL", "AUL",  # ambiguous lineage
    # --- MDS ---
    "MDS", "MDSEB", "MDSEB1", "MDSEB2", "MDSID5Q", "MDSMD",
    "MDSRS", "MDSRSMD", "MDSRSSLD", "MDSSLD", "MDSU", "RCYC",
    # --- MDS/MPN overlap ---
    "CMML", "CMML0", "CMML1", "CMML2", "JMML",
    "MDS/MPN", "MDSMPNRST", "MDSMPNU", "ACML",
    # --- MPN ---
    "MPN", "MPNU", "ET", "ETMF", "PMF", "PMFOFS", "PMFPES",
    "PV", "PVMF", "CNL", "CELNOS",
    "CML", "CMLBCRABL1",
    # --- Mastocytosis ---
    "SM", "ASM", "ISM", "CMCD", "MCD", "MCSL", "SMAHN", "SMMCL",
    # --- Other myeloid ---
    "MNGLP",  # myeloid neoplasms with germ line predisposition
    # --- Legacy/short-form codes from original specification ---
    "MPN-U", "RARS", "RCMD", "RAEB", "RAEB-T", "aCML",
    "CMML-1", "CMML-2", "MDS-RS", "MDS-SLD", "MDS-MLD",
    "MDS-EB1", "MDS-EB2", "MDS-U", "AML-MRC", "APL", "AML-NOS", "CEL",
}

NON_CODING_CLASSES = {
    "Intron", "Silent", "3'UTR", "5'UTR", "5'Flank", "3'Flank",
    "IGR", "RNA", "Splice_Region",
}

HYPERMUT_THRESHOLD = 20  # max coding mutations in target genes per sample

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clinical_samples(data_dir: str) -> pd.DataFrame:
    """Load data_clinical_sample.txt, skipping comment rows."""
    path = os.path.join(data_dir, "data_clinical_sample.txt")
    log.info("Loading clinical sample data: %s", path)
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    log.info("  Total samples in GENIE: %d", len(df))
    return df


def load_mutations(data_dir: str) -> pd.DataFrame:
    """Load data_mutations_extended.txt — only columns we need for speed."""
    path = os.path.join(data_dir, "data_mutations_extended.txt")
    log.info("Loading mutation data: %s", path)
    cols = [
        "Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode",
        "HGVSp_Short",
    ]
    df = pd.read_csv(path, sep="\t", usecols=cols, low_memory=False)
    log.info("  Total mutation rows: %d", len(df))
    return df


def load_gene_matrix(data_dir: str) -> pd.DataFrame:
    """Load data_gene_matrix.txt — maps SAMPLE_ID to panel for mutations."""
    path = os.path.join(data_dir, "data_gene_matrix.txt")
    log.info("Loading gene matrix: %s", path)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    log.info("  Gene matrix entries: %d", len(df))
    return df


def load_gene_panels(data_dir: str) -> dict:
    """Parse all data_gene_panel_*.txt files into {panel_id: set(genes)}."""
    panels = {}
    pattern = os.path.join(data_dir, "data_gene_panel_*.txt")
    files = glob.glob(pattern)
    log.info("Loading %d gene panel files", len(files))
    for fp in files:
        panel_id = None
        genes = set()
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("stable_id:"):
                    panel_id = line.split(":", 1)[1].strip()
                elif line.startswith("gene_list:"):
                    gene_part = line.split(":", 1)[1]
                    genes = {g.strip() for g in gene_part.split("\t") if g.strip()}
        if panel_id:
            panels[panel_id] = genes
    log.info("  Panels loaded: %d", len(panels))
    return panels


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_myeloid(clinical: pd.DataFrame) -> set:
    """Return set of SAMPLE_IDs with myeloid OncoTree codes."""
    mask = clinical["ONCOTREE_CODE"].isin(MYELOID_CODES)
    samples = set(clinical.loc[mask, "SAMPLE_ID"])
    log.info("Myeloid samples: %d", len(samples))
    return samples


def filter_coding(mutations: pd.DataFrame) -> pd.DataFrame:
    """Remove non-coding variant classes."""
    before = len(mutations)
    df = mutations[~mutations["Variant_Classification"].isin(NON_CODING_CLASSES)].copy()
    log.info("Coding filter: %d -> %d rows (removed %d non-coding)",
             before, len(df), before - len(df))
    return df


def filter_hypermutated(mutations: pd.DataFrame, target_genes: list) -> tuple:
    """Identify and exclude hypermutated samples (>HYPERMUT_THRESHOLD coding muts
    in target genes). Returns (clean_mutations, set_of_excluded_samples)."""
    target_muts = mutations[mutations["Hugo_Symbol"].isin(target_genes)]
    counts = target_muts.groupby("Tumor_Sample_Barcode").size()
    hypermut = set(counts[counts > HYPERMUT_THRESHOLD].index)
    log.info("Hypermutated samples (>%d coding muts in target genes): %d",
             HYPERMUT_THRESHOLD, len(hypermut))
    clean = mutations[~mutations["Tumor_Sample_Barcode"].isin(hypermut)]
    return clean, hypermut


# ---------------------------------------------------------------------------
# Panel coverage
# ---------------------------------------------------------------------------

def build_sample_panel_map(gene_matrix: pd.DataFrame) -> dict:
    """Map SAMPLE_ID -> panel_id for the mutations assay."""
    mapping = dict(zip(gene_matrix["SAMPLE_ID"], gene_matrix["mutations"]))
    return mapping


def gene_on_panel(gene: str, panel_id: str, panels: dict) -> bool:
    """Check whether a gene is covered by a given sequencing panel."""
    if panel_id not in panels:
        return False
    return gene in panels[panel_id]


def samples_with_both_genes_on_panel(
    gene_a: str, gene_b: str,
    sample_ids: set, sample_panel: dict, panels: dict,
) -> set:
    """Return subset of sample_ids where both genes are on the sequencing panel."""
    result = set()
    for sid in sample_ids:
        pid = sample_panel.get(sid)
        if pid and gene_on_panel(gene_a, pid, panels) and gene_on_panel(gene_b, pid, panels):
            result.add(sid)
    return result


def samples_with_all_genes_on_panel(
    genes: list, sample_ids: set, sample_panel: dict, panels: dict,
) -> set:
    """Return subset where ALL listed genes are on the panel."""
    result = set()
    for sid in sample_ids:
        pid = sample_panel.get(sid)
        if pid and all(gene_on_panel(g, pid, panels) for g in genes):
            result.add(sid)
    return result


# ---------------------------------------------------------------------------
# Co-occurrence statistics
# ---------------------------------------------------------------------------

def build_mutation_sets(mutations: pd.DataFrame, sample_ids: set, genes: list) -> dict:
    """For each gene, return the set of samples carrying a coding mutation."""
    gene_muts = mutations[
        (mutations["Hugo_Symbol"].isin(genes)) &
        (mutations["Tumor_Sample_Barcode"].isin(sample_ids))
    ]
    result = {}
    for g in genes:
        result[g] = set(
            gene_muts.loc[gene_muts["Hugo_Symbol"] == g, "Tumor_Sample_Barcode"]
        )
    return result


def build_specific_mutation_sets(mutations: pd.DataFrame, sample_ids: set,
                                  spec: dict) -> dict:
    """For each gene:variant pair, return set of samples carrying that exact variant."""
    result = {}
    for gene, var in spec.items():
        mask = (
            (mutations["Hugo_Symbol"] == gene) &
            (mutations["HGVSp_Short"] == var) &
            (mutations["Tumor_Sample_Barcode"].isin(sample_ids))
        )
        result[gene] = set(mutations.loc[mask, "Tumor_Sample_Barcode"])
    return result


def fisher_cooccurrence(
    gene_a: str, gene_b: str,
    mut_sets: dict, eligible_samples: set,
) -> dict:
    """2x2 Fisher's exact test for co-occurrence of two genes."""
    a_set = mut_sets[gene_a] & eligible_samples
    b_set = mut_sets[gene_b] & eligible_samples
    n = len(eligible_samples)

    both = len(a_set & b_set)
    a_only = len(a_set - b_set)
    b_only = len(b_set - a_set)
    neither = n - both - a_only - b_only

    table = np.array([[both, a_only], [b_only, neither]])
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

    freq_a = len(a_set) / n if n else 0
    freq_b = len(b_set) / n if n else 0
    expected = freq_a * freq_b * n
    oe_ratio = both / expected if expected > 0 else float("inf")

    return {
        "gene_a": gene_a,
        "gene_b": gene_b,
        "eligible_samples": n,
        "n_a": len(a_set),
        "n_b": len(b_set),
        "freq_a": round(freq_a, 6),
        "freq_b": round(freq_b, 6),
        "observed": both,
        "expected": round(expected, 4),
        "O/E": round(oe_ratio, 4) if expected > 0 else None,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "odds_ratio": round(odds_ratio, 4) if not np.isinf(odds_ratio) else "Inf",
        "p_value": p_value,
    }


def multi_cooccurrence(
    genes: list, mut_sets: dict, eligible_samples: set,
) -> dict:
    """Compute co-occurrence for 3+ genes simultaneously."""
    n = len(eligible_samples)
    sets = [mut_sets[g] & eligible_samples for g in genes]

    co_set = sets[0]
    for s in sets[1:]:
        co_set = co_set & s

    observed = len(co_set)
    freqs = [len(s) / n for s in sets]
    expected = 1.0
    for f in freqs:
        expected *= f
    expected *= n

    return {
        "genes": genes,
        "eligible_samples": n,
        "individual_counts": {g: len(s) for g, s in zip(genes, sets)},
        "individual_freqs": {g: round(len(s) / n, 6) for g, s in zip(genes, sets)},
        "observed": observed,
        "expected": round(expected, 6),
        "O/E": round(observed / expected, 4) if expected > 0 else None,
        "co_occurring_samples": sorted(co_set),
    }


# ---------------------------------------------------------------------------
# Pairwise analysis across all 34 target genes
# ---------------------------------------------------------------------------

def run_pairwise_target_genes(
    mut_sets: dict, myeloid_samples: set,
    sample_panel: dict, panels: dict,
) -> list:
    """Run pairwise Fisher test for all C(34,2) = 561 gene pairs,
    with panel adjustment."""
    pairs = list(combinations(TARGET_GENES, 2))
    results = []
    log.info("Running pairwise co-occurrence for %d gene pairs...", len(pairs))

    for ga, gb in pairs:
        eligible = samples_with_both_genes_on_panel(
            ga, gb, myeloid_samples, sample_panel, panels
        )
        if len(eligible) < 10:
            continue
        res = fisher_cooccurrence(ga, gb, mut_sets, eligible)
        results.append(res)

    # BH correction
    p_values = [r["p_value"] for r in results]
    if p_values:
        _, q_values, _, _ = multipletests(p_values, method="fdr_bh")
        for r, q in zip(results, q_values):
            r["q_value"] = round(q, 8)

    results.sort(key=lambda x: x["p_value"])
    log.info("Pairwise analysis complete: %d pairs tested", len(results))
    return results


# ---------------------------------------------------------------------------
# Funnel analysis
# ---------------------------------------------------------------------------

def progressive_funnel(
    mutations: pd.DataFrame, myeloid_samples: set,
    sample_panel: dict, panels: dict,
    spec: dict,
) -> list:
    """Starting from the most common mutation, progressively narrow by each
    additional mutation. Order: DNMT3A > IDH2 > PTPN11 > SETBP1 (by expected
    prevalence in myeloid)."""
    gene_order = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]
    steps = []
    current_set = myeloid_samples.copy()

    for i, gene in enumerate(gene_order):
        genes_so_far = gene_order[:i + 1]
        # Panel-eligible samples for all genes seen so far
        eligible = samples_with_all_genes_on_panel(
            genes_so_far, current_set, sample_panel, panels
        )

        var = spec[gene]
        mask = (
            (mutations["Hugo_Symbol"] == gene) &
            (mutations["HGVSp_Short"] == var) &
            (mutations["Tumor_Sample_Barcode"].isin(eligible))
        )
        carriers = set(mutations.loc[mask, "Tumor_Sample_Barcode"])

        # Narrow: intersect with previous carriers if not first step
        if i == 0:
            remaining = carriers
        else:
            remaining = remaining & carriers

        step = {
            "step": i + 1,
            "added_mutation": f"{gene} {var}",
            "genes_so_far": [f"{g} {spec[g]}" for g in genes_so_far],
            "panel_eligible_samples": len(eligible),
            "carriers_this_gene": len(carriers),
            "cumulative_match": len(remaining),
            "matching_samples": sorted(remaining),
        }
        steps.append(step)
        log.info("Funnel step %d (%s %s): %d eligible, %d carriers, %d cumulative",
                 i + 1, gene, var, len(eligible), len(carriers), len(remaining))

    return steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = datetime.now()
    log.info("=" * 70)
    log.info("Four-gene co-occurrence analysis — GENIE v19.0")
    log.info("Profile: %s", " + ".join(f"{g} {v}" for g, v in PATIENT_MUTATIONS.items()))
    log.info("=" * 70)

    # --- Load data ---
    clinical = load_clinical_samples(DATA_DIR)
    mutations = load_mutations(DATA_DIR)
    gene_matrix = load_gene_matrix(DATA_DIR)
    panels = load_gene_panels(DATA_DIR)

    total_samples = len(clinical)

    # --- Myeloid filter ---
    myeloid_samples = filter_myeloid(clinical)

    # --- Restrict mutations to myeloid samples ---
    mutations = mutations[mutations["Tumor_Sample_Barcode"].isin(myeloid_samples)]
    log.info("Mutation rows in myeloid samples: %d", len(mutations))

    # --- Coding filter ---
    mutations = filter_coding(mutations)

    # --- Hypermutation filter ---
    mutations, hypermut_excluded = filter_hypermutated(mutations, TARGET_GENES)
    myeloid_clean = myeloid_samples - hypermut_excluded

    # --- Panel map ---
    sample_panel = build_sample_panel_map(gene_matrix)

    # --- Gene-level mutation sets (any coding variant) ---
    gene_mut_sets = build_mutation_sets(mutations, myeloid_clean, TARGET_GENES)

    # --- Specific variant sets for the 4-gene profile ---
    four_genes = list(PATIENT_MUTATIONS.keys())
    spec_mut_sets = build_specific_mutation_sets(
        mutations, myeloid_clean, PATIENT_MUTATIONS
    )

    # --- Pairwise analysis (all 34 target genes, any coding variant) ---
    pairwise_results = run_pairwise_target_genes(
        gene_mut_sets, myeloid_clean, sample_panel, panels
    )

    # --- Highlight key pairs from expected results ---
    highlight_pairs = [("CSF3R", "SETBP1"), ("FLT3", "NPM1")]
    highlighted = {}
    for r in pairwise_results:
        pair = (r["gene_a"], r["gene_b"])
        pair_rev = (r["gene_b"], r["gene_a"])
        for hp in highlight_pairs:
            if pair == hp or pair_rev == hp:
                label = f"{hp[0]}+{hp[1]}"
                highlighted[label] = {
                    "observed": r["observed"],
                    "expected": r["expected"],
                    "O/E": r["O/E"],
                    "p_value": r["p_value"],
                    "q_value": r.get("q_value"),
                    "eligible_samples": r["eligible_samples"],
                }

    # --- Pairwise for the 4 specific variants ---
    four_gene_pairs = list(combinations(four_genes, 2))
    pairwise_specific = []
    for ga, gb in four_gene_pairs:
        eligible = samples_with_both_genes_on_panel(
            ga, gb, myeloid_clean, sample_panel, panels
        )
        if len(eligible) < 10:
            continue
        res = fisher_cooccurrence(ga, gb, spec_mut_sets, eligible)
        pairwise_specific.append(res)

    if pairwise_specific:
        p_vals = [r["p_value"] for r in pairwise_specific]
        _, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")
        for r, q in zip(pairwise_specific, q_vals):
            r["q_value"] = round(q, 8)

    # --- Triple co-occurrence (specific variants) ---
    triple_combos = list(combinations(four_genes, 3))
    triple_results = []
    for combo in triple_combos:
        genes_list = list(combo)
        eligible = samples_with_all_genes_on_panel(
            genes_list, myeloid_clean, sample_panel, panels
        )
        if len(eligible) < 10:
            continue
        res = multi_cooccurrence(genes_list, spec_mut_sets, eligible)
        triple_results.append(res)

    # --- Quadruple co-occurrence (specific variants) ---
    eligible_quad = samples_with_all_genes_on_panel(
        four_genes, myeloid_clean, sample_panel, panels
    )
    quad_result = multi_cooccurrence(four_genes, spec_mut_sets, eligible_quad)

    # --- Progressive funnel ---
    funnel = progressive_funnel(
        mutations, myeloid_clean, sample_panel, panels, PATIENT_MUTATIONS
    )

    # --- Individual variant prevalence in myeloid ---
    variant_prevalence = {}
    for gene, var in PATIENT_MUTATIONS.items():
        eligible = samples_with_all_genes_on_panel(
            [gene], myeloid_clean, sample_panel, panels
        )
        n_elig = len(eligible)
        carriers = spec_mut_sets[gene] & eligible
        n_carriers = len(carriers)
        freq = n_carriers / n_elig if n_elig else 0
        variant_prevalence[f"{gene} {var}"] = {
            "eligible_samples": n_elig,
            "carriers": n_carriers,
            "frequency": round(freq, 6),
            "pct": round(freq * 100, 3),
        }
        log.info("Prevalence %s %s: %d / %d (%.3f%%)",
                 gene, var, n_carriers, n_elig, freq * 100)

    # --- Top 20 co-occurring pairs (all target genes) ---
    top20_cooccurrence = []
    for r in pairwise_results[:20]:
        top20_cooccurrence.append({
            "pair": f"{r['gene_a']}+{r['gene_b']}",
            "observed": r["observed"],
            "expected": r["expected"],
            "O/E": r["O/E"],
            "p_value": r["p_value"],
            "q_value": r.get("q_value"),
        })

    # --- Assemble output ---
    elapsed = (datetime.now() - t0).total_seconds()

    output = {
        "metadata": {
            "analysis": "Four-gene co-occurrence — GENIE v19.0",
            "profile": PATIENT_MUTATIONS,
            "target_genes": TARGET_GENES,
            "myeloid_oncotree_codes": sorted(MYELOID_CODES),
            "non_coding_classes_excluded": sorted(NON_CODING_CLASSES),
            "hypermutation_threshold": HYPERMUT_THRESHOLD,
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": round(elapsed, 1),
        },
        "cohort_summary": {
            "total_genie_samples": total_samples,
            "myeloid_samples": len(myeloid_samples),
            "hypermutated_excluded": len(hypermut_excluded),
            "myeloid_after_hypermut_filter": len(myeloid_clean),
        },
        "variant_prevalence": variant_prevalence,
        "highlighted_pairs": highlighted,
        "pairwise_specific_variants": pairwise_specific,
        "triple_cooccurrence": triple_results,
        "quadruple_cooccurrence": quad_result,
        "progressive_funnel": funnel,
        "top20_gene_level_cooccurrence": top20_cooccurrence,
        "full_pairwise_gene_level_count": len(pairwise_results),
    }

    # --- Save ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "four_gene_cooccurrence.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Results saved to: %s", out_path)

    # --- Console summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total GENIE samples:           {total_samples:,}")
    print(f"Myeloid samples:               {len(myeloid_samples):,}")
    print(f"Hypermutated excluded:          {len(hypermut_excluded):,}")
    print(f"Myeloid (clean):               {len(myeloid_clean):,}")
    print()

    print("--- Individual variant prevalence ---")
    for label, info in variant_prevalence.items():
        print(f"  {label}: {info['carriers']:,} / {info['eligible_samples']:,} "
              f"({info['pct']}%)")
    print()

    print("--- Highlighted gene-level pairs ---")
    for label, info in highlighted.items():
        print(f"  {label}: obs={info['observed']}, exp={info['expected']:.2f}, "
              f"O/E={info['O/E']:.2f}, p={info['p_value']:.2e}")
    print()

    print("--- Pairwise (specific variants) ---")
    for r in pairwise_specific:
        print(f"  {r['gene_a']} + {r['gene_b']}: obs={r['observed']}, "
              f"exp={r['expected']:.4f}, "
              f"O/E={r['O/E'] if r['O/E'] is not None else 'N/A'}, "
              f"p={r['p_value']:.2e}")
    print()

    print("--- Triple co-occurrence (specific variants) ---")
    for r in triple_results:
        genes_str = " + ".join(r["genes"])
        print(f"  {genes_str}: obs={r['observed']}, exp={r['expected']:.6f}")
    print()

    print("--- Quadruple co-occurrence ---")
    genes_str = " + ".join(quad_result["genes"])
    print(f"  {genes_str}")
    print(f"  Eligible samples: {quad_result['eligible_samples']:,}")
    print(f"  Observed: {quad_result['observed']}")
    print(f"  Expected: {quad_result['expected']:.6f}")
    if quad_result["co_occurring_samples"]:
        print(f"  Matching samples: {quad_result['co_occurring_samples']}")
    print()

    print("--- Progressive funnel ---")
    for step in funnel:
        print(f"  Step {step['step']}: {step['added_mutation']} -> "
              f"{step['cumulative_match']} matches "
              f"(from {step['panel_eligible_samples']:,} eligible)")
    print()

    print("--- Top 10 gene-level co-occurring pairs (by p-value) ---")
    for r in top20_cooccurrence[:10]:
        print(f"  {r['pair']}: obs={r['observed']}, exp={r['expected']:.1f}, "
              f"O/E={r['O/E']:.2f}, p={r['p_value']:.2e}, q={r['q_value']:.2e}")
    print()

    print(f"Runtime: {elapsed:.1f} seconds")
    print(f"Full results: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
