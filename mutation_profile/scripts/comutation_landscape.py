#!/usr/bin/env python3
"""
Co-mutation landscape analysis for DNMT3A, IDH2, PTPN11, SETBP1 in GENIE myeloid data.

For each target gene carrier, identifies the top 20 co-mutated genes with Fisher's exact
test odds ratios vs baseline myeloid population. Also profiles the additional mutations
in DNMT3A+IDH2+PTPN11 triple carriers, and checks for EZH2 overlap.

Data: AACR GENIE MAF (data_mutations_extended.txt) + clinical sample file.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAF_FILE = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw" / "data_mutations_extended.txt"
CLINICAL_FILE = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw" / "data_clinical_sample.txt"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "comutation_landscape.json"

# ── Configuration ──────────────────────────────────────────────────────────
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

# Myeloid ONCOTREE_CODE prefixes and exact matches
# Exclude non-myeloid codes that happen to start with these prefixes:
#   MPNST = malignant peripheral nerve sheath tumor
#   MSCC, MSCHW, MSTAD = sarcomas
#   SMZL = splenic marginal zone lymphoma
MYELOID_PREFIXES = ("AML", "MDS", "MPN", "CMML", "JMML", "CML", "TMN", "SM", "CEL", "HES")
MYELOID_EXACT = {"AMML", "APL", "APLPMLRARA", "APMF", "AUL", "MLADS", "MS"}
NON_MYELOID_EXACT = {"MPNST", "MSCC", "MSCHW", "MSTAD", "SMZL"}

# Biologically relevant genes to specifically check
BIO_RELEVANT_GENES = [
    "TET2", "ASXL1", "SRSF2", "SF3B1", "RUNX1", "FLT3", "NPM1",
    "TP53", "NRAS", "KRAS", "CBL", "EZH2",
]


def is_myeloid(oncotree_code: str) -> bool:
    """Check if an ONCOTREE_CODE is myeloid."""
    code = oncotree_code.strip().upper()
    if code in NON_MYELOID_EXACT:
        return False
    if code in MYELOID_EXACT:
        return True
    return any(code.startswith(prefix) for prefix in MYELOID_PREFIXES)


def load_myeloid_samples() -> dict[str, str]:
    """Load myeloid sample IDs from clinical file. Returns {sample_id: patient_id}."""
    print("Loading clinical sample data...")
    sample_to_patient: dict[str, str] = {}
    total = 0
    myeloid = 0

    with open(CLINICAL_FILE) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[0] == "PATIENT_ID":
                continue  # header
            total += 1
            patient_id = fields[0]
            sample_id = fields[1]
            oncotree_code = fields[3] if len(fields) > 3 else ""
            if is_myeloid(oncotree_code):
                sample_to_patient[sample_id] = patient_id
                myeloid += 1

    print(f"  Total samples: {total:,}")
    print(f"  Myeloid samples: {myeloid:,}")
    print(f"  Unique myeloid patients: {len(set(sample_to_patient.values())):,}")
    return sample_to_patient


def load_mutations(myeloid_samples: dict[str, str]) -> dict[str, set[str]]:
    """
    Parse MAF file, filter to myeloid + protein-altering.
    Returns {patient_id: set of mutated gene names}.
    """
    print("\nParsing MAF file (protein-altering variants in myeloid samples)...")
    patient_genes: dict[str, set[str]] = defaultdict(set)
    total_lines = 0
    kept = 0
    skipped_non_myeloid = 0
    skipped_non_altering = 0

    with open(MAF_FILE) as f:
        header = f.readline().strip().split("\t")
        hugo_idx = header.index("Hugo_Symbol")
        vc_idx = header.index("Variant_Classification")
        barcode_idx = header.index("Tumor_Sample_Barcode")

        for line in f:
            total_lines += 1
            if total_lines % 500_000 == 0:
                print(f"  Processed {total_lines:,} lines, kept {kept:,}...")

            fields = line.split("\t")
            sample_id = fields[barcode_idx]

            if sample_id not in myeloid_samples:
                skipped_non_myeloid += 1
                continue

            variant_class = fields[vc_idx]
            if variant_class not in PROTEIN_ALTERING:
                skipped_non_altering += 1
                continue

            gene = fields[hugo_idx]
            patient_id = myeloid_samples[sample_id]
            patient_genes[patient_id].add(gene)
            kept += 1

    print(f"  Total MAF lines: {total_lines:,}")
    print(f"  Skipped (non-myeloid): {skipped_non_myeloid:,}")
    print(f"  Skipped (non-protein-altering): {skipped_non_altering:,}")
    print(f"  Kept mutations: {kept:,}")
    print(f"  Patients with protein-altering mutations: {len(patient_genes):,}")
    return dict(patient_genes)


def compute_fisher_or(
    target_carriers: set[str],
    other_gene_carriers: set[str],
    all_patients: set[str],
) -> tuple[float, float]:
    """
    Compute Fisher's exact test OR and p-value for co-occurrence of
    target gene and other gene.
    """
    a = len(target_carriers & other_gene_carriers)      # both
    b = len(target_carriers - other_gene_carriers)       # target only
    c = len(other_gene_carriers - target_carriers)       # other only
    d = len(all_patients - target_carriers - other_gene_carriers)  # neither
    table = [[a, b], [c, d]]
    try:
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    except Exception:
        odds_ratio, p_value = float("nan"), 1.0
    return odds_ratio, p_value


def analyze_comutation_landscape(patient_genes: dict[str, set[str]]) -> dict:
    """Main analysis: for each target gene, find top 20 co-mutated genes."""
    all_patients = set(patient_genes.keys())
    n_total = len(all_patients)
    print(f"\nTotal myeloid patients with mutations: {n_total:,}")

    # Build gene -> set of patients for all genes
    print("Building gene-patient index...")
    gene_patients: dict[str, set[str]] = defaultdict(set)
    for pid, genes in patient_genes.items():
        for g in genes:
            gene_patients[g].add(pid)

    # Count target gene carriers
    target_carrier_sets: dict[str, set[str]] = {}
    for tg in TARGET_GENES:
        carriers = gene_patients.get(tg, set())
        target_carrier_sets[tg] = carriers
        print(f"  {tg} carriers: {len(carriers):,} ({len(carriers)/n_total*100:.1f}%)")

    # All genes seen (excluding target genes for co-mutation ranking)
    all_other_genes = sorted(set(gene_patients.keys()) - set(TARGET_GENES))
    print(f"  Total distinct genes in myeloid data: {len(gene_patients):,}")
    print(f"  Non-target genes: {len(all_other_genes):,}")

    results = {
        "metadata": {
            "total_myeloid_patients": n_total,
            "target_genes": TARGET_GENES,
            "variant_types_included": sorted(PROTEIN_ALTERING),
            "myeloid_filter": "ONCOTREE_CODE starts with AML/MDS/MPN/CMML/JMML/CML/TMN/SM/CEL/HES or exact AMML/APL/APLPMLRARA/APMF/AUL/MLADS/MS; excluding MPNST/MSCC/MSCHW/MSTAD/SMZL",
        },
        "target_gene_counts": {},
        "top20_comutations": {},
        "biologically_relevant_comutations": {},
        "triple_DNMT3A_IDH2_PTPN11": {},
        "ezh2_in_triples": {},
    }

    # ── Per-target gene: top 20 co-mutated genes ──────────────────────────
    for tg in TARGET_GENES:
        print(f"\n{'='*60}")
        print(f"Analyzing co-mutations for {tg} carriers (n={len(target_carrier_sets[tg]):,})")
        print(f"{'='*60}")

        carriers = target_carrier_sets[tg]
        n_carriers = len(carriers)
        if n_carriers == 0:
            print(f"  No carriers found, skipping.")
            results["target_gene_counts"][tg] = 0
            results["top20_comutations"][tg] = []
            continue

        results["target_gene_counts"][tg] = n_carriers

        # Count co-mutations among carriers
        comut_counts: Counter = Counter()
        for pid in carriers:
            for g in patient_genes[pid]:
                if g != tg and g not in TARGET_GENES:
                    comut_counts[g] += 1

        # Also count co-mutations with the OTHER target genes
        target_comut_counts: Counter = Counter()
        for pid in carriers:
            for g in patient_genes[pid]:
                if g != tg and g in TARGET_GENES:
                    target_comut_counts[g] += 1

        # Top 20 by count (non-target genes)
        top20 = comut_counts.most_common(20)
        top20_list = []
        for gene, count in top20:
            pct = count / n_carriers * 100
            other_carriers = gene_patients.get(gene, set())
            or_val, p_val = compute_fisher_or(carriers, other_carriers, all_patients)
            entry = {
                "gene": gene,
                "count": count,
                "pct_of_target_carriers": round(pct, 1),
                "baseline_count": len(other_carriers),
                "baseline_pct": round(len(other_carriers) / n_total * 100, 1),
                "fisher_OR": round(or_val, 2) if not np.isnan(or_val) else None,
                "fisher_p": f"{p_val:.2e}",
            }
            top20_list.append(entry)
            print(f"  {gene:12s}: {count:5d} ({pct:5.1f}%)  baseline {len(other_carriers)/n_total*100:5.1f}%  OR={or_val:.2f}  p={p_val:.2e}")

        results["top20_comutations"][tg] = top20_list

        # Also record co-occurrence with other target genes
        target_comut_list = []
        for other_tg in TARGET_GENES:
            if other_tg == tg:
                continue
            count = target_comut_counts.get(other_tg, 0)
            pct = count / n_carriers * 100 if n_carriers > 0 else 0
            other_carriers = target_carrier_sets.get(other_tg, set())
            or_val, p_val = compute_fisher_or(carriers, other_carriers, all_patients)
            target_comut_list.append({
                "gene": other_tg,
                "count": count,
                "pct_of_target_carriers": round(pct, 1),
                "fisher_OR": round(or_val, 2) if not np.isnan(or_val) else None,
                "fisher_p": f"{p_val:.2e}",
            })
        results["top20_comutations"][f"{tg}_vs_other_targets"] = target_comut_list

    # ── Biologically relevant co-mutations ─────────────────────────────────
    print(f"\n{'='*60}")
    print("Biologically relevant co-mutations")
    print(f"{'='*60}")

    for tg in TARGET_GENES:
        carriers = target_carrier_sets[tg]
        n_carriers = len(carriers)
        if n_carriers == 0:
            continue

        bio_results = []
        print(f"\n  {tg} (n={n_carriers:,}):")
        for bio_gene in BIO_RELEVANT_GENES:
            bio_carriers = gene_patients.get(bio_gene, set())
            overlap = carriers & bio_carriers
            count = len(overlap)
            pct = count / n_carriers * 100
            or_val, p_val = compute_fisher_or(carriers, bio_carriers, all_patients)
            bio_results.append({
                "gene": bio_gene,
                "count": count,
                "pct_of_target_carriers": round(pct, 1),
                "baseline_count": len(bio_carriers),
                "baseline_pct": round(len(bio_carriers) / n_total * 100, 1),
                "fisher_OR": round(or_val, 2) if not np.isnan(or_val) else None,
                "fisher_p": f"{p_val:.2e}",
            })
            sig = " ***" if p_val < 0.05 else ""
            direction = "enriched" if or_val > 1 else "depleted" if or_val < 1 else "neutral"
            print(f"    {bio_gene:8s}: {count:5d}/{n_carriers} ({pct:5.1f}%)  baseline {len(bio_carriers)/n_total*100:5.1f}%  OR={or_val:.2f}  p={p_val:.2e}  {direction}{sig}")

        results["biologically_relevant_comutations"][tg] = bio_results

    # ── DNMT3A+IDH2+PTPN11 triple carriers: additional mutations ──────────
    print(f"\n{'='*60}")
    print("DNMT3A+IDH2+PTPN11 triple carriers: additional mutations")
    print(f"{'='*60}")

    triple_carriers = (
        target_carrier_sets["DNMT3A"]
        & target_carrier_sets["IDH2"]
        & target_carrier_sets["PTPN11"]
    )
    n_triple = len(triple_carriers)
    print(f"  Triple carriers (DNMT3A+IDH2+PTPN11): {n_triple}")

    if n_triple > 0:
        # Collect ALL additional mutations in triple carriers
        additional_mutations: Counter = Counter()
        triple_patient_genes: dict[str, list[str]] = {}

        for pid in sorted(triple_carriers):
            other_genes = sorted(patient_genes[pid] - {"DNMT3A", "IDH2", "PTPN11"})
            triple_patient_genes[pid] = other_genes
            for g in other_genes:
                additional_mutations[g] += 1

        # Top additional mutations
        print(f"\n  Additional mutations in triple carriers:")
        additional_list = []
        for gene, count in additional_mutations.most_common(30):
            pct = count / n_triple * 100
            baseline_pct = len(gene_patients.get(gene, set())) / n_total * 100
            additional_list.append({
                "gene": gene,
                "count": count,
                "pct_of_triple_carriers": round(pct, 1),
                "baseline_pct": round(baseline_pct, 1),
            })
            print(f"    {gene:12s}: {count:3d}/{n_triple} ({pct:5.1f}%)  baseline {baseline_pct:5.1f}%")

        # Per-patient breakdown
        print(f"\n  Per-patient mutation profiles:")
        per_patient_list = []
        for pid in sorted(triple_carriers):
            all_genes = sorted(patient_genes[pid])
            per_patient_list.append({
                "patient_id": pid,
                "total_mutations": len(all_genes),
                "all_mutated_genes": all_genes,
                "additional_beyond_triple": triple_patient_genes[pid],
            })
            print(f"    {pid}: {', '.join(all_genes)}")

        results["triple_DNMT3A_IDH2_PTPN11"] = {
            "n_carriers": n_triple,
            "patient_ids": sorted(triple_carriers),
            "additional_mutations_ranked": additional_list,
            "per_patient_profiles": per_patient_list,
        }

        # ── Check: do any triple carriers also have EZH2? ────────────────
        print(f"\n  EZH2 in triple carriers:")
        ezh2_patients = gene_patients.get("EZH2", set())
        triple_with_ezh2 = triple_carriers & ezh2_patients
        print(f"    Triple carriers with EZH2: {len(triple_with_ezh2)}/{n_triple}")
        if triple_with_ezh2:
            for pid in sorted(triple_with_ezh2):
                print(f"      {pid}: {sorted(patient_genes[pid])}")

        results["ezh2_in_triples"] = {
            "n_triple_carriers": n_triple,
            "n_with_ezh2": len(triple_with_ezh2),
            "patient_ids_with_ezh2": sorted(triple_with_ezh2),
            "patient_profiles": [
                {
                    "patient_id": pid,
                    "all_mutated_genes": sorted(patient_genes[pid]),
                }
                for pid in sorted(triple_with_ezh2)
            ],
        }

        # Also check SETBP1 in triple carriers (the full quad)
        triple_with_setbp1 = triple_carriers & target_carrier_sets["SETBP1"]
        print(f"\n  SETBP1 in DNMT3A+IDH2+PTPN11 triple carriers: {len(triple_with_setbp1)}/{n_triple}")
        if triple_with_setbp1:
            for pid in sorted(triple_with_setbp1):
                print(f"    {pid}: {sorted(patient_genes[pid])}")
        results["triple_DNMT3A_IDH2_PTPN11"]["n_with_SETBP1"] = len(triple_with_setbp1)
        results["triple_DNMT3A_IDH2_PTPN11"]["patients_with_SETBP1"] = sorted(triple_with_setbp1)

    else:
        results["triple_DNMT3A_IDH2_PTPN11"] = {
            "n_carriers": 0,
            "note": "No DNMT3A+IDH2+PTPN11 triple carriers found in GENIE myeloid data",
        }
        results["ezh2_in_triples"] = {
            "n_triple_carriers": 0,
            "n_with_ezh2": 0,
        }

    # ── Also check all other triple/quad combinations ──────────────────────
    print(f"\n{'='*60}")
    print("All triple and quadruple combinations")
    print(f"{'='*60}")

    from itertools import combinations as combos

    all_combos = {}
    for r in [3, 4]:
        for combo in combos(TARGET_GENES, r):
            carriers = set.intersection(*(target_carrier_sets[g] for g in combo))
            key = "+".join(combo)
            all_combos[key] = len(carriers)
            print(f"  {key}: {len(carriers)}")

    results["all_combinations"] = all_combos

    return results


def main():
    print("=" * 70)
    print("CO-MUTATION LANDSCAPE ANALYSIS")
    print("GENIE myeloid data -- DNMT3A, IDH2, PTPN11, SETBP1")
    print("=" * 70)

    # Step 1: Load myeloid samples
    myeloid_samples = load_myeloid_samples()

    # Step 2: Parse MAF for protein-altering mutations in myeloid patients
    patient_genes = load_mutations(myeloid_samples)

    # Step 3: Analyze co-mutation landscape
    results = analyze_comutation_landscape(patient_genes)

    # Step 4: Write output
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results written to: {OUTPUT_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
