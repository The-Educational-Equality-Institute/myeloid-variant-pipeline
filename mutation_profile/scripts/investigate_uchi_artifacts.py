#!/usr/bin/env python3
"""
Investigate UCHI p.*NNN* artifacts in GENIE data.

The premature analysis found 15 apparent quadruple co-occurrences (DNMT3A+IDH2+PTPN11+SETBP1),
but 13/15 came from UCHI center with suspicious p.*NNN* protein change notation.

This script extracts raw MAF details for these patients to determine:
1. What Variant_Classification do the p.*NNN* entries have?
2. Are they real mutations or panel coverage markers?
3. What does p.*180* mean for SETBP1 in UCHI data?
"""

import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
GENIE_RAW = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
MAF_FILE = GENIE_RAW / "data_mutations_extended.txt"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"

TARGET_GENES = {"DNMT3A", "IDH2", "PTPN11", "SETBP1"}

# All UCHI patients from the premature analysis that had p.*NNN* patterns
UCHI_ARTIFACT_PATIENTS = {
    "GENIE-UCHI-Patient150", "GENIE-UCHI-Patient19", "GENIE-UCHI-Patient195",
    "GENIE-UCHI-Patient309", "GENIE-UCHI-Patient321", "GENIE-UCHI-Patient390",
    "GENIE-UCHI-Patient432", "GENIE-UCHI-Patient454", "GENIE-UCHI-Patient61",
    "GENIE-UCHI-Patient78", "GENIE-UCHI-Patient79", "GENIE-UCHI-Patient98",
    "GENIE-UCHI-Patient103", "GENIE-UCHI-Patient116",
}

# UCHI patients with real-looking variants but p.*180* SETBP1
UCHI_MIXED_PATIENTS = {
    "GENIE-UCHI-OP1119", "GENIE-UCHI-OP1134",
}

# UCSF hypermutated case
UCSF_PATIENT = {"GENIE-UCSF-790748"}

ALL_PATIENTS_OF_INTEREST = UCHI_ARTIFACT_PATIENTS | UCHI_MIXED_PATIENTS | UCSF_PATIENT


def investigate():
    print("=" * 70)
    print("UCHI Artifact Investigation")
    print("=" * 70)

    # Collect all MAF rows for patients of interest (all genes, not just targets)
    patient_rows = defaultdict(list)

    # Also collect stats on p.*NNN* pattern across ALL UCHI patients
    uchi_star_pattern_stats = Counter()  # gene -> count of p.*NNN* entries
    uchi_real_pattern_stats = Counter()  # gene -> count of real protein changes
    uchi_var_class_for_stars = Counter()  # Variant_Classification for p.*NNN* entries

    # And stats on Variant_Classification distribution for UCHI vs non-UCHI
    center_var_class = defaultdict(Counter)  # center -> Variant_Classification -> count

    print(f"\nScanning MAF file: {MAF_FILE}")
    print("This will take several minutes...")

    row_count = 0
    with open(MAF_FILE) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            if line.startswith("Hugo_Symbol") or header is None:
                if "Hugo_Symbol" in line:
                    header = line.strip().split("\t")
                    continue
                header = line.strip().split("\t")
                continue

            row_count += 1
            if row_count % 1_000_000 == 0:
                print(f"  Processed {row_count:,} rows...")

            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))

            row = dict(zip(header, fields))
            gene = row.get("Hugo_Symbol", "")
            sample_id = row.get("Tumor_Sample_Barcode", "")

            # Extract center from sample ID (format: GENIE-CENTER-...)
            parts = sample_id.split("-")
            center = parts[1] if len(parts) >= 3 else "UNKNOWN"

            # Track UCHI-specific patterns for target genes
            if center == "UCHI" and gene in TARGET_GENES:
                protein = row.get("HGVSp_Short", "")
                var_class = row.get("Variant_Classification", "")

                import re
                if re.match(r'^p\.\*\d+\*$', protein):
                    uchi_star_pattern_stats[gene] += 1
                    uchi_var_class_for_stars[var_class] += 1
                else:
                    uchi_real_pattern_stats[gene] += 1

            # Collect full rows for patients of interest (ALL genes)
            # Need to match patient_id which may differ from sample_id
            # Patient ID is embedded: GENIE-UCHI-Patient19 sample might be GENIE-UCHI-Patient19-xxxxx
            patient_match = None
            for pid in ALL_PATIENTS_OF_INTEREST:
                if sample_id.startswith(pid):
                    patient_match = pid
                    break

            if patient_match:
                patient_rows[patient_match].append({
                    "gene": gene,
                    "protein_change": row.get("HGVSp_Short", ""),
                    "variant_classification": row.get("Variant_Classification", ""),
                    "variant_type": row.get("Variant_Type", ""),
                    "chromosome": row.get("Chromosome", ""),
                    "start_pos": row.get("Start_Position", ""),
                    "end_pos": row.get("End_Position", ""),
                    "ref_allele": row.get("Reference_Allele", ""),
                    "alt_allele": row.get("Tumor_Seq_Allele2", ""),
                    "t_depth": row.get("t_depth", ""),
                    "t_ref_count": row.get("t_ref_count", ""),
                    "t_alt_count": row.get("t_alt_count", ""),
                    "n_depth": row.get("n_depth", ""),
                    "mutation_status": row.get("Mutation_Status", ""),
                    "dbSNP_RS": row.get("dbSNP_RS", ""),
                    "sample_id": sample_id,
                    "center": center,
                })

            # Track Variant_Classification by center (for target genes only)
            if gene in TARGET_GENES:
                var_class = row.get("Variant_Classification", "")
                center_var_class[center][var_class] += 1

    print(f"\nTotal rows scanned: {row_count:,}")

    # Report findings
    print(f"\n{'=' * 70}")
    print("UCHI p.*NNN* Pattern Analysis")
    print(f"{'=' * 70}")

    print("\nUCHI target gene mutations with p.*NNN* pattern:")
    for gene in sorted(TARGET_GENES):
        stars = uchi_star_pattern_stats.get(gene, 0)
        real = uchi_real_pattern_stats.get(gene, 0)
        total = stars + real
        pct = stars / total * 100 if total > 0 else 0
        print(f"  {gene}: {stars} p.*NNN* / {total} total ({pct:.1f}% are artifacts)")

    print(f"\nVariant_Classification for p.*NNN* entries:")
    for vc, count in uchi_var_class_for_stars.most_common():
        print(f"  {vc}: {count}")

    print(f"\n{'=' * 70}")
    print("Per-Patient MAF Details")
    print(f"{'=' * 70}")

    for pid in sorted(ALL_PATIENTS_OF_INTEREST):
        rows = patient_rows.get(pid, [])
        if not rows:
            print(f"\n--- {pid}: NO ROWS FOUND ---")
            continue

        # Separate target gene rows from others
        target_rows = [r for r in rows if r["gene"] in TARGET_GENES]
        other_rows = [r for r in rows if r["gene"] not in TARGET_GENES]

        print(f"\n--- {pid} ({len(rows)} total mutations, {len(target_rows)} in target genes) ---")

        for r in sorted(target_rows, key=lambda x: x["gene"]):
            depth_str = ""
            if r["t_depth"]:
                depth_str = f" depth={r['t_depth']}"
                if r["t_alt_count"]:
                    depth_str += f" alt={r['t_alt_count']}"
            print(f"  {r['gene']:8s} {r['protein_change']:25s} {r['variant_classification']:25s} "
                  f"{r['variant_type']:5s} chr{r['chromosome']}:{r['start_pos']}-{r['end_pos']}"
                  f"{depth_str}")

        if other_rows:
            print(f"  + {len(other_rows)} other gene mutations")
            # Show a few
            for r in other_rows[:5]:
                print(f"    {r['gene']:8s} {r['protein_change']:25s} {r['variant_classification']}")

    print(f"\n{'=' * 70}")
    print("Variant_Classification Distribution: UCHI vs Others (target genes)")
    print(f"{'=' * 70}")

    print("\nUCHI:")
    for vc, count in center_var_class.get("UCHI", Counter()).most_common():
        print(f"  {vc}: {count}")

    # Aggregate non-UCHI
    non_uchi = Counter()
    for center, vc_counts in center_var_class.items():
        if center != "UCHI":
            non_uchi += vc_counts

    print("\nAll other centers combined:")
    for vc, count in non_uchi.most_common():
        print(f"  {vc}: {count}")

    # Save detailed results
    results = {
        "investigation_date": "2026-03-19",
        "uchi_star_pattern_counts": dict(uchi_star_pattern_stats),
        "uchi_real_pattern_counts": dict(uchi_real_pattern_stats),
        "uchi_star_variant_classifications": dict(uchi_var_class_for_stars),
        "patient_details": {},
    }

    for pid in sorted(ALL_PATIENTS_OF_INTEREST):
        rows = patient_rows.get(pid, [])
        target_rows = [r for r in rows if r["gene"] in TARGET_GENES]
        results["patient_details"][pid] = {
            "total_mutations": len(rows),
            "target_gene_mutations": target_rows,
            "other_gene_count": len(rows) - len(target_rows),
        }

    output_path = RESULTS_DIR / "uchi_artifact_investigation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    investigate()
