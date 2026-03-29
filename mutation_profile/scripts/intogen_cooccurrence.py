#!/usr/bin/env python3
"""
IntOGen co-occurrence analysis for DNMT3A, IDH2, PTPN11, SETBP1 in AML/MDS.

IntOGen does NOT provide per-sample mutation data. It aggregates driver gene
statistics at the cohort level (e.g., "28 of 257 samples carry DNMT3A mutations
in the Beat AML cohort"). This means:

1. We CANNOT compute actual per-patient co-occurrence counts.
2. We CAN extract per-gene mutation frequencies across AML/MDS cohorts.
3. We CAN compute expected co-occurrence rates under statistical independence
   (for comparison with databases that DO have per-sample data).

This script extracts all available data and documents the limitation.
"""

import csv
import json
import os
from collections import defaultdict
from itertools import combinations

# Paths
DATA_DIR = os.path.expanduser(
    "~/projects/mrna-hematology-research/mutation_profile/data/intogen"
)
COMPENDIUM_FILE = os.path.join(
    DATA_DIR, "drivers_data/2024-06-18_IntOGen-Drivers/Compendium_Cancer_Genes.tsv"
)
UNFILTERED_FILE = os.path.join(
    DATA_DIR, "drivers_data/2024-06-18_IntOGen-Drivers/Unfiltered_drivers.tsv"
)
COHORTS_FILE = os.path.join(
    DATA_DIR, "cohorts_data/2024-06-18_IntOGen-Cohorts/cohorts.tsv"
)
OUTPUT_FILE = os.path.expanduser(
    "~/projects/mrna-hematology-research/mutation_profile/results/intogen_cooccurrence.json"
)

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]
TARGET_CANCER_TYPES = {"AML", "MDS"}


def read_tsv(filepath):
    """Read a TSV file and return list of dicts."""
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def parse_cohorts():
    """Parse cohorts.tsv to get AML/MDS cohort metadata."""
    rows = read_tsv(COHORTS_FILE)
    cohorts = {}
    for row in rows:
        cancer = row.get("CANCER", "")
        if cancer in TARGET_CANCER_TYPES:
            cohort_id = row["COHORT"]
            cohorts[cohort_id] = {
                "cancer_type": cancer,
                "cancer_name": row.get("CANCER_NAME", "").strip(),
                "source": row.get("SOURCE", "").strip(),
                "platform": row.get("PLATFORM", "").strip(),
                "samples": int(row.get("SAMPLES", 0)),
                "mutations": int(row.get("MUTATIONS", 0)),
                "age_group": row.get("AGE", "").strip(),
                "drivers": int(float(row.get("DRIVERS", 0))),
                "cohort_name": row.get("COHORT_NAME", "").strip(),
            }
    return cohorts


def parse_compendium(cohorts):
    """Parse Compendium_Cancer_Genes.tsv for target genes in AML/MDS cohorts."""
    rows = read_tsv(COMPENDIUM_FILE)
    gene_data = defaultdict(list)

    for row in rows:
        gene = row.get("SYMBOL", "")
        cancer_type = row.get("CANCER_TYPE", "")
        cohort = row.get("COHORT", "")

        if gene in TARGET_GENES and cancer_type in TARGET_CANCER_TYPES:
            samples_mutated = float(row.get("SAMPLES", 0))
            total_samples = int(row.get("TOTAL_SAMPLES", 0))
            pct = float(row.get("%_SAMPLES_COHORT", 0)) * 100  # stored as fraction
            mutations = int(row.get("MUTATIONS", 0))

            gene_data[gene].append({
                "cohort": cohort,
                "cancer_type": cancer_type,
                "samples_mutated": int(samples_mutated),
                "total_samples": total_samples,
                "pct_mutated": round(pct, 2),
                "mutations": mutations,
                "methods": row.get("METHODS", "").strip(),
                "role": row.get("ROLE", "").strip(),
                "is_driver": row.get("IS_DRIVER", "").strip(),
            })

    return gene_data


def parse_unfiltered(cohorts):
    """Parse Unfiltered_drivers.tsv for ALL target gene entries in AML/MDS
    (includes genes that didn't pass driver filters, like SETBP1)."""
    rows = read_tsv(UNFILTERED_FILE)
    gene_data = defaultdict(list)

    for row in rows:
        gene = row.get("SYMBOL", "")
        cancer_type = row.get("CANCER_TYPE", "")
        cohort = row.get("COHORT", "")

        if gene in TARGET_GENES and cancer_type in TARGET_CANCER_TYPES:
            samples_cohort = int(row.get("SAMPLES_COHORT", 0))
            mutations = float(row.get("MUTATIONS", 0))

            gene_data[gene].append({
                "cohort": cohort,
                "cancer_type": cancer_type,
                "total_samples": samples_cohort,
                "mutations": int(mutations),
                "all_methods": row.get("ALL_METHODS", "").strip(),
                "sig_methods": row.get("SIG_METHODS", "").strip(),
                "qvalue": row.get("QVALUE_COMBINATION", "").strip(),
                "ranking": row.get("RANKING", "").strip(),
                "role": row.get("ROLE", "").strip(),
                "is_driver": row.get("DRIVER", "").strip(),
                "filter": row.get("FILTER", "").strip(),
            })

    return gene_data


def compute_deduplicated_totals(cohorts):
    """Compute total unique AML/MDS samples, handling potential cohort overlaps.

    Known overlaps:
    - ICGC_WXS_AML_LAML_KR_2019 (50) and ICGC_WXS_AML_LAML_KR_VARSCAN_2019 (67)
      are the SAME Korean LAML cohort processed with different variant callers.
      Use the larger one (67).
    """
    aml_samples = 0
    mds_samples = 0
    excluded_cohorts = set()

    # Known overlapping cohorts - exclude the smaller/duplicate
    overlap_exclusions = {
        "ICGC_WXS_AML_LAML_KR_2019": "Overlaps with ICGC_WXS_AML_LAML_KR_VARSCAN_2019 (same LAML-KR cohort, different caller)",
    }

    for cohort_id, info in cohorts.items():
        if cohort_id in overlap_exclusions:
            excluded_cohorts.add(cohort_id)
            continue

        if info["cancer_type"] == "AML":
            aml_samples += info["samples"]
        elif info["cancer_type"] == "MDS":
            mds_samples += info["samples"]

    return aml_samples, mds_samples, excluded_cohorts


def compute_aggregate_gene_counts(gene_compendium, cohorts):
    """Compute aggregate mutated sample counts per gene across non-overlapping cohorts."""
    # Use the non-overlapping cohort set
    # Exclude ICGC_WXS_AML_LAML_KR_2019 (subsumed by VARSCAN version)
    exclude = {"ICGC_WXS_AML_LAML_KR_2019"}

    gene_totals = {}
    for gene in TARGET_GENES:
        entries = gene_compendium.get(gene, [])
        total = 0
        cohort_details = []
        for entry in entries:
            if entry["cohort"] not in exclude:
                total += entry["samples_mutated"]
                cohort_details.append(entry)
        gene_totals[gene] = {
            "total_mutated_across_cohorts": total,
            "cohort_details": cohort_details,
        }
    return gene_totals


def compute_expected_cooccurrence(gene_totals, total_samples):
    """Compute EXPECTED co-occurrence under independence assumption.

    NOTE: These are statistical estimates only. IntOGen does not provide
    actual per-sample co-occurrence data. These numbers assume:
    1. Mutations are independently distributed across samples
    2. No cohort structure (pooled across all cohorts)

    Both assumptions are known to be violated in reality (DNMT3A+IDH2 co-occur
    more than expected; SETBP1 is enriched in specific subtypes).
    """
    if total_samples == 0:
        return {}, {}, {}

    # Gene frequencies
    freqs = {}
    for gene in TARGET_GENES:
        n = gene_totals.get(gene, {}).get("total_mutated_across_cohorts", 0)
        freqs[gene] = n / total_samples

    # Pairwise expected
    pairwise = {}
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        expected = freqs[g1] * freqs[g2] * total_samples
        pairwise[key] = round(expected, 2)

    # Triple expected
    triple = {}
    for g1, g2, g3 in combinations(TARGET_GENES, 3):
        key = f"{g1}+{g2}+{g3}"
        expected = freqs[g1] * freqs[g2] * freqs[g3] * total_samples
        triple[key] = round(expected, 4)

    # Quadruple expected
    quad_key = "+".join(TARGET_GENES)
    expected_quad = 1.0
    for g in TARGET_GENES:
        expected_quad *= freqs[g]
    expected_quad *= total_samples
    quadruple = {quad_key: round(expected_quad, 6)}

    return pairwise, triple, quadruple


def main():
    print("=" * 70)
    print("IntOGen Co-occurrence Analysis: DNMT3A, IDH2, PTPN11, SETBP1")
    print("=" * 70)

    # 1. Parse cohorts
    cohorts = parse_cohorts()
    print(f"\nFound {len(cohorts)} AML/MDS cohorts:")
    for cid, info in sorted(cohorts.items()):
        print(f"  {cid}: {info['cancer_type']} - {info['samples']} samples - {info['source']}")

    # 2. Parse driver gene data
    gene_compendium = parse_compendium(cohorts)
    gene_unfiltered = parse_unfiltered(cohorts)

    print(f"\n--- Compendium (confirmed drivers) ---")
    for gene in TARGET_GENES:
        entries = gene_compendium.get(gene, [])
        print(f"  {gene}: {len(entries)} cohort entries")
        for e in entries:
            print(f"    {e['cohort']}: {e['samples_mutated']}/{e['total_samples']} ({e['pct_mutated']:.1f}%) - {e['cancer_type']}")

    print(f"\n--- Unfiltered (all tested, including non-drivers) ---")
    for gene in TARGET_GENES:
        entries = gene_unfiltered.get(gene, [])
        print(f"  {gene}: {len(entries)} cohort entries")
        for e in entries:
            print(f"    {e['cohort']}: {e['total_samples']} total samples, {e['mutations']} mutations - driver={e['is_driver']} filter={e['filter']}")

    # 3. Deduplicated totals
    aml_samples, mds_samples, excluded = compute_deduplicated_totals(cohorts)
    total_samples = aml_samples + mds_samples
    print(f"\nDeduplicated sample counts:")
    print(f"  AML: {aml_samples}")
    print(f"  MDS: {mds_samples}")
    print(f"  Total: {total_samples}")
    print(f"  Excluded cohorts: {excluded}")

    # 4. Aggregate gene counts
    gene_totals = compute_aggregate_gene_counts(gene_compendium, cohorts)
    print(f"\nAggregate mutation counts (confirmed drivers only, deduplicated):")
    for gene in TARGET_GENES:
        n = gene_totals[gene]["total_mutated_across_cohorts"]
        pct = (n / total_samples * 100) if total_samples > 0 else 0
        print(f"  {gene}: {n} ({pct:.1f}%)")

    # 5. Expected co-occurrence (statistical only)
    exp_pair, exp_triple, exp_quad = compute_expected_cooccurrence(gene_totals, total_samples)
    print(f"\nExpected co-occurrence under independence (ESTIMATES ONLY):")
    print(f"  Pairwise: {exp_pair}")
    print(f"  Triple: {exp_triple}")
    print(f"  Quadruple: {exp_quad}")

    # 6. Build per-cohort detail for single gene section
    single_gene = {}
    for gene in TARGET_GENES:
        entries = gene_compendium.get(gene, [])
        unfiltered_entries = gene_unfiltered.get(gene, [])

        total_mutated = gene_totals[gene]["total_mutated_across_cohorts"]

        by_cohort = {}
        for e in entries:
            by_cohort[e["cohort"]] = {
                "samples_mutated": e["samples_mutated"],
                "total_samples": e["total_samples"],
                "pct": e["pct_mutated"],
                "cancer_type": e["cancer_type"],
                "methods": e["methods"],
                "role": e["role"],
                "is_confirmed_driver": True,
            }

        # Add unfiltered entries not already in compendium
        for e in unfiltered_entries:
            if e["cohort"] not in by_cohort:
                by_cohort[e["cohort"]] = {
                    "samples_mutated": "unknown (not in compendium)",
                    "total_samples": e["total_samples"],
                    "mutations": e["mutations"],
                    "cancer_type": e["cancer_type"],
                    "methods": e["all_methods"],
                    "role": e["role"],
                    "is_confirmed_driver": e["is_driver"] == "True",
                    "filter_status": e["filter"],
                }

        single_gene[gene] = total_mutated

    # 7. Build overlap warnings
    overlap_warnings = []

    # TCGA overlap with other databases
    tcga_cohorts = [c for c in cohorts if "TCGA" in c]
    if tcga_cohorts:
        overlap_warnings.append(
            f"TCGA_WXS_AML (n=140) samples overlap with TCGA AML data in cBioPortal "
            f"(n=200 in PanCancer Atlas). IntOGen uses a subset."
        )

    # Beat AML overlap
    beat_cohorts = [c for c in cohorts if "BEAT" in c or "OTHER_WXS_AML_PRY" in c]
    if beat_cohorts:
        overlap_warnings.append(
            f"OTHER_WXS_AML_PRY_BEAT_2018 (n=257) is from the Beat AML 2018 cohort "
            f"(Tyner et al. Nature 2018). The Beat AML 2022 cohort in cBioPortal "
            f"(n=903, Bottomly et al.) is a superset. Patient-level overlap is likely."
        )

    # ICGC Korean overlap
    overlap_warnings.append(
        "ICGC_WXS_AML_LAML_KR_2019 (n=50) and ICGC_WXS_AML_LAML_KR_VARSCAN_2019 (n=67) "
        "are the SAME Korean LAML cohort processed with different variant callers. "
        "Only the VARSCAN version (n=67) is used in aggregate counts."
    )

    # PCAWG overlap
    overlap_warnings.append(
        "PCAWG_WGS_MYELOID_AML (n=13) and PCAWG_WGS_MYELOID_MPN (n=23, labeled MDS) "
        "may overlap with TCGA samples, as PCAWG includes some TCGA whole-genome data."
    )

    # 8. Build output JSON
    output = {
        "database": "IntOGen",
        "analysis_date": "2026-03-14",
        "intogen_release": "2024.09.20",
        "total_patients": total_samples,
        "total_aml_patients": aml_samples,
        "total_mds_patients": mds_samples,
        "disease_filter": "AML/MDS cohorts (9 AML cohorts + 1 MDS cohort)",
        "data_files_used": [
            "IntOGen-Drivers-20240920.zip -> Compendium_Cancer_Genes.tsv",
            "IntOGen-Drivers-20240920.zip -> Unfiltered_drivers.tsv",
            "IntOGen-Cohorts-20240920.zip -> cohorts.tsv",
            "analysis_summary.md",
        ],
        "cohorts": {
            cid: {
                "cancer_type": info["cancer_type"],
                "samples": info["samples"],
                "source": info["source"],
                "platform": info["platform"],
                "age_group": info["age_group"],
                "excluded_from_aggregate": cid in excluded,
            }
            for cid, info in sorted(cohorts.items())
        },
        "critical_limitation": (
            "IntOGen does NOT provide per-sample mutation data. It reports only "
            "aggregate cohort-level statistics (e.g., '28 of 257 samples carry DNMT3A mutations'). "
            "Per-patient co-occurrence CANNOT be computed from IntOGen data. "
            "The pairwise/triple/quadruple sections below contain EXPECTED values under "
            "statistical independence, NOT observed counts. For actual co-occurrence data, "
            "use cBioPortal, GDC, or IPSSM datasets which provide per-sample mutation calls."
        ),
        "single_gene": {
            gene: single_gene[gene] for gene in TARGET_GENES
        },
        "single_gene_detail": {},
        "pairwise": {
            "DNMT3A+IDH2": exp_pair.get("DNMT3A+IDH2", 0),
            "DNMT3A+PTPN11": exp_pair.get("DNMT3A+PTPN11", 0),
            "DNMT3A+SETBP1": exp_pair.get("DNMT3A+SETBP1", 0),
            "IDH2+PTPN11": exp_pair.get("IDH2+PTPN11", 0),
            "IDH2+SETBP1": exp_pair.get("IDH2+SETBP1", 0),
            "PTPN11+SETBP1": exp_pair.get("PTPN11+SETBP1", 0),
        },
        "pairwise_note": "VALUES ARE EXPECTED COUNTS UNDER INDEPENDENCE, NOT OBSERVED. IntOGen lacks per-sample data.",
        "triple": {
            "DNMT3A+IDH2+PTPN11": exp_triple.get("DNMT3A+IDH2+PTPN11", 0),
            "DNMT3A+IDH2+SETBP1": exp_triple.get("DNMT3A+IDH2+SETBP1", 0),
            "DNMT3A+PTPN11+SETBP1": exp_triple.get("DNMT3A+PTPN11+SETBP1", 0),
            "IDH2+PTPN11+SETBP1": exp_triple.get("IDH2+PTPN11+SETBP1", 0),
        },
        "triple_note": "VALUES ARE EXPECTED COUNTS UNDER INDEPENDENCE, NOT OBSERVED. IntOGen lacks per-sample data.",
        "quadruple": {
            "DNMT3A+IDH2+PTPN11+SETBP1": exp_quad.get("DNMT3A+IDH2+PTPN11+SETBP1", 0),
        },
        "quadruple_note": "VALUE IS EXPECTED COUNT UNDER INDEPENDENCE, NOT OBSERVED. IntOGen lacks per-sample data.",
        "monosomy7_data_available": False,
        "monosomy7_note": (
            "IntOGen analyzes point mutations and small indels only. Chromosomal-level "
            "alterations (monosomy 7, del(7q), complex karyotype) are not assessed. "
            "The platform cannot provide any monosomy 7 data."
        ),
        "overlap_warnings": " | ".join(overlap_warnings),
        "notes": (
            "IntOGen is a driver gene discovery platform, not a mutation co-occurrence database. "
            "It identifies which genes are statistically significant drivers in each cancer type "
            "using 7 complementary methods, but does not expose which specific samples carry "
            "which mutations. Key findings: (1) DNMT3A is the #1 AML driver (52/661, 7.87%), "
            "(2) IDH2 is #3 (44/661, 6.66%), (3) PTPN11 is #10 (24/661, 3.63%), "
            "(4) SETBP1 is NOT identified as an AML or MDS driver by IntOGen -- its only "
            "driver designation is in Adrenocortical Carcinoma. The SETBP1 absence reflects "
            "dataset limitations (MDS cohort has only 23 samples), not biological reality. "
            "Published literature firmly establishes SETBP1 as a myeloid driver. "
            "For actual per-patient co-occurrence analysis, use cBioPortal (intogen_cooccurrence.json "
            "values are statistical estimates only)."
        ),
    }

    # Build single_gene_detail
    for gene in TARGET_GENES:
        compendium_entries = gene_compendium.get(gene, [])
        unfiltered_entries = gene_unfiltered.get(gene, [])

        detail = {
            "total_mutated_across_aml_mds_cohorts": single_gene[gene],
            "is_intogen_driver_in_aml": any(
                e["cancer_type"] == "AML" for e in compendium_entries
            ),
            "is_intogen_driver_in_mds": any(
                e["cancer_type"] == "MDS" for e in compendium_entries
            ),
            "aml_driver_rank": None,
            "by_cohort": [],
        }

        # Add rank info from summary
        rank_map = {"DNMT3A": 1, "IDH2": 3, "PTPN11": 10, "SETBP1": None}
        detail["aml_driver_rank"] = rank_map.get(gene)
        if detail["aml_driver_rank"]:
            detail["aml_driver_rank_note"] = f"Rank #{rank_map[gene]} of 71 AML driver genes"

        for e in compendium_entries:
            detail["by_cohort"].append({
                "cohort": e["cohort"],
                "cancer_type": e["cancer_type"],
                "samples_mutated": e["samples_mutated"],
                "total_samples": e["total_samples"],
                "pct": e["pct_mutated"],
                "role": e["role"],
                "driver_status": "confirmed_driver",
            })

        output["single_gene_detail"][gene] = detail

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to: {OUTPUT_FILE}")

    return output


if __name__ == "__main__":
    result = main()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total patients: {result['total_patients']}")
    print(f"  AML: {result['total_aml_patients']}")
    print(f"  MDS: {result['total_mds_patients']}")
    print(f"\nSingle gene counts (driver-confirmed, across non-overlapping cohorts):")
    for gene in TARGET_GENES:
        print(f"  {gene}: {result['single_gene'][gene]}")
    print(f"\nExpected pairwise co-occurrence (under independence):")
    for pair, val in result["pairwise"].items():
        print(f"  {pair}: {val}")
    print(f"\nExpected triple co-occurrence (under independence):")
    for trip, val in result["triple"].items():
        print(f"  {trip}: {val}")
    print(f"\nExpected quadruple co-occurrence (under independence):")
    for quad, val in result["quadruple"].items():
        print(f"  {quad}: {val}")
    print(f"\nCRITICAL: These are EXPECTED values, NOT observed co-occurrence counts.")
    print(f"IntOGen does not provide per-sample mutation data.")
