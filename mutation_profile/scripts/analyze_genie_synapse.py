#!/usr/bin/env python3
"""
AACR GENIE Synapse Data Analyzer for DNMT3A, IDH2, PTPN11, SETBP1 Co-occurrence

This script analyzes raw GENIE Synapse data files if/when they are downloaded.
It expects the standard cBioPortal-format files from GENIE releases:
  - data_mutations_extended.txt (MAF format)
  - data_clinical_sample.txt (sample annotations with OncoTree codes)
  - data_clinical_patient.txt (patient demographics)
  - data_gene_panel_*.txt (panel gene lists)

Usage:
  1. Download GENIE v19.0+ files from Synapse (syn7222066) after access approval
  2. Place files in: mutation_profile/data/genie/raw/
  3. Run: python -m mutation_profile.scripts.analyze_genie_synapse
     or:  .venv/bin/python mutation_profile/scripts/analyze_genie_synapse.py

Output:
  mutation_profile/results/genie_v2_cooccurrence.json
"""

import csv
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.expanduser("~/projects/mrna-hematology-research"))
GENIE_RAW_DIR = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]

# OncoTree codes for myeloid neoplasms
MYELOID_ONCOTREE_CODES = {
    # AML
    "AML", "AMLMRC", "AMLRGA", "AMLRR", "AMLNOS", "APL", "AMOL", "APMF",
    "AMLMBC", "AMLCBFB", "AMLRUNX1", "AMLMLLT3", "AMLDEKNUP", "AMLBCR",
    "AMLGATA2MECOM", "AMLNPM1", "AMLCEBPA", "AMLTP53", "AMLDEK",
    # MDS
    "MDS", "MDS5Q", "MDSEB1", "MDSEB2", "MDSMD", "MDSSLD", "MDSRS",
    "MDSU", "MDSRSMD", "MDSSID", "MDSLB", "MDSIB1", "MDSIB2",
    "RCMD", "RARS", "RA", "RAEB", "RAEB1", "RAEB2", "RCUD",
    # MDS/MPN overlap
    "CMML", "CMML0", "CMML1", "CMML2", "JMML", "MDSMPNU",
    "MDSMPNRST", "ACML", "ACML_ATYPICAL",
    # MPN (may carry relevant mutations)
    "MPN", "CML", "ET", "PV", "PMF", "CMLBCRABL1", "SM", "CEL",
    "MPNU", "MPNST",
    # Other myeloid
    "BPDCN", "MPAL", "TMN", "ALAL",
    # Therapy-related
    "TMDS", "TAML", "TMN",
}

# Broader cancer type keywords as fallback
MYELOID_KEYWORDS = [
    "leukemia", "myeloid", "myelodysplast", "mds", "aml",
    "myeloproliferat", "myelomonocytic", "erythroleukemia",
]


def find_genie_files() -> dict:
    """Locate GENIE data files in the raw directory."""
    files = {}

    # Check for standard file names
    candidates = {
        "mutations": [
            "data_mutations_extended.txt",
            "data_mutations.txt",
            "genie_mutations.txt",
        ],
        "clinical_sample": [
            "data_clinical_sample.txt",
            "genie_clinical_sample.txt",
        ],
        "clinical_patient": [
            "data_clinical_patient.txt",
            "genie_clinical_patient.txt",
        ],
    }

    for key, filenames in candidates.items():
        for fn in filenames:
            path = GENIE_RAW_DIR / fn
            if path.exists():
                files[key] = path
                break

    # Also look for any .txt or .tsv files
    if GENIE_RAW_DIR.exists():
        for f in GENIE_RAW_DIR.iterdir():
            if f.suffix in (".txt", ".tsv", ".csv") and f.name.startswith("data_gene_panel"):
                files.setdefault("gene_panels", []).append(f)

    return files


def load_gene_panels(panel_files: list) -> dict:
    """Load gene panel definitions to determine SETBP1 coverage per panel."""
    panels = {}  # panel_name -> set of genes

    for pf in panel_files:
        panel_name = pf.stem.replace("data_gene_panel_", "")
        genes = set()
        with open(pf) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line.startswith("stable_id") or not line:
                    continue
                # Panel files have tab-separated gene lists or one gene per line
                parts = line.split("\t")
                for part in parts:
                    part = part.strip()
                    if part and not part.startswith("gene_list"):
                        genes.add(part)
        panels[panel_name] = genes

    return panels


def load_clinical_samples(clinical_file: Path) -> dict:
    """Load clinical sample data, returning sample_id -> {oncotree_code, patient_id, seq_panel, center}."""
    samples = {}

    with open(clinical_file) as f:
        # Skip comment lines
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
            sample_id = row.get("SAMPLE_ID", "")
            if not sample_id:
                continue

            samples[sample_id] = {
                "patient_id": row.get("PATIENT_ID", ""),
                "oncotree_code": row.get("ONCOTREE_CODE", ""),
                "cancer_type": row.get("CANCER_TYPE", ""),
                "cancer_type_detailed": row.get("CANCER_TYPE_DETAILED", ""),
                "center": row.get("CENTER", ""),
                "seq_panel": row.get("SEQ_ASSAY_ID", ""),
                "sample_type": row.get("SAMPLE_TYPE", ""),
            }

    return samples


def load_clinical_patients(patient_file: Path) -> dict:
    """Load patient-level clinical data."""
    patients = {}

    with open(patient_file) as f:
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
            patient_id = row.get("PATIENT_ID", "")
            if not patient_id:
                continue

            age_str = row.get("AGE_AT_SEQ_REPORT", row.get("AGE_AT_SEQ_REPORT_DAYS", ""))
            try:
                age = float(age_str)
            except (ValueError, TypeError):
                age = None

            patients[patient_id] = {
                "age": age,
                "sex": row.get("SEX", ""),
                "race": row.get("PRIMARY_RACE", ""),
                "ethnicity": row.get("ETHNICITY", ""),
                "vital_status": row.get("DEAD", ""),
                "center": row.get("CENTER", ""),
            }

    return patients


def is_myeloid_sample(sample_info: dict) -> bool:
    """Determine if a sample is a myeloid neoplasm based on OncoTree code and cancer type."""
    # Check OncoTree code first (most reliable)
    code = sample_info.get("oncotree_code", "").upper()
    if code in MYELOID_ONCOTREE_CODES:
        return True

    # Fallback: check cancer type text
    cancer_type = (
        sample_info.get("cancer_type", "") + " " +
        sample_info.get("cancer_type_detailed", "")
    ).lower()

    return any(kw in cancer_type for kw in MYELOID_KEYWORDS)


def load_mutations(mutations_file: Path, myeloid_samples: dict) -> dict:
    """
    Load mutations from MAF file, filtering to myeloid samples and target genes.

    Returns: patient_id -> {gene: [list of mutations]}
    """
    patient_mutations = defaultdict(lambda: defaultdict(list))
    target_genes_set = set(TARGET_GENES)

    total_lines = 0
    matched_lines = 0
    skipped_non_myeloid = 0

    print("  Loading mutations file (this may take several minutes for large files)...")

    with open(mutations_file) as f:
        header = None
        for line in f:
            if line.startswith("#") or line.startswith("Hugo_Symbol"):
                if "Hugo_Symbol" in line:
                    header = line.strip().split("\t")
                continue
            if header is None:
                # Try treating first non-comment line as header
                header = line.strip().split("\t")
                continue

            total_lines += 1
            if total_lines % 1_000_000 == 0:
                print(f"    Processed {total_lines:,} mutation rows...")

            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))

            row = dict(zip(header, fields))

            # Get gene
            gene = row.get("Hugo_Symbol", "")
            if gene not in target_genes_set:
                continue

            # Get sample ID and check if myeloid
            sample_id = row.get("Tumor_Sample_Barcode", row.get("Sample_Id", ""))
            if sample_id not in myeloid_samples:
                skipped_non_myeloid += 1
                continue

            patient_id = myeloid_samples[sample_id].get("patient_id", sample_id)

            # Extract mutation details
            mutation_info = {
                "gene": gene,
                "protein_change": row.get("HGVSp_Short", row.get("Protein_Change", "")),
                "variant_classification": row.get("Variant_Classification", ""),
                "variant_type": row.get("Variant_Type", ""),
                "chromosome": row.get("Chromosome", ""),
                "start_pos": row.get("Start_Position", ""),
                "ref_allele": row.get("Reference_Allele", ""),
                "alt_allele": row.get("Tumor_Seq_Allele2", ""),
                "sample_id": sample_id,
            }

            patient_mutations[patient_id][gene].append(mutation_info)
            matched_lines += 1

    print(f"    Total mutation rows processed: {total_lines:,}")
    print(f"    Target gene mutations in myeloid samples: {matched_lines:,}")
    print(f"    Target gene mutations in non-myeloid samples (skipped): {skipped_non_myeloid:,}")
    print(f"    Unique patients with target gene mutations: {len(patient_mutations):,}")

    return dict(patient_mutations)


def compute_cooccurrence(patient_mutations: dict, patient_clinical: dict = None) -> dict:
    """Compute single, pairwise, triple, and quadruple co-occurrence."""
    genes = TARGET_GENES

    # Determine which patients have each gene mutated
    gene_patients = {g: set() for g in genes}
    for patient_id, gene_muts in patient_mutations.items():
        for gene in genes:
            if gene in gene_muts and len(gene_muts[gene]) > 0:
                gene_patients[gene].add(patient_id)

    # All patients with any target gene mutation
    all_patients = set()
    for pids in gene_patients.values():
        all_patients |= pids

    total = len(all_patients)
    print(f"\n  Total patients with any target gene mutation: {total}")

    # Single gene counts
    single_counts = {}
    for gene in genes:
        count = len(gene_patients[gene])
        single_counts[gene] = count
        print(f"    {gene}: {count}")

    # Pairwise
    pairwise_counts = {}
    pairwise_patients_map = {}
    for g1, g2 in combinations(genes, 2):
        key = f"{g1}+{g2}"
        overlap = gene_patients[g1] & gene_patients[g2]
        pairwise_counts[key] = len(overlap)
        pairwise_patients_map[key] = sorted(list(overlap))
        if overlap:
            print(f"    {key}: {len(overlap)}")

    # Triple
    triple_counts = {}
    triple_patients_map = {}
    for combo in combinations(genes, 3):
        key = "+".join(combo)
        overlap = gene_patients[combo[0]]
        for g in combo[1:]:
            overlap = overlap & gene_patients[g]
        triple_counts[key] = len(overlap)
        triple_patients_map[key] = sorted(list(overlap))
        if overlap:
            print(f"    TRIPLE {key}: {len(overlap)} patients")
            # Print patient details
            for pid in sorted(overlap)[:10]:
                muts = patient_mutations[pid]
                mut_str = ", ".join(
                    f"{g}:{muts[g][0]['protein_change']}"
                    for g in combo if g in muts and muts[g]
                )
                age_str = ""
                if patient_clinical and pid in patient_clinical:
                    age = patient_clinical[pid].get("age")
                    if age is not None:
                        age_str = f" (age {age})"
                print(f"      {pid}: {mut_str}{age_str}")

    # Quadruple
    quad_overlap = gene_patients[genes[0]]
    for g in genes[1:]:
        quad_overlap = quad_overlap & gene_patients[g]
    quad_count = len(quad_overlap)

    if quad_count > 0:
        print(f"\n    *** QUADRUPLE CO-OCCURRENCE FOUND: {quad_count} patients ***")
        for pid in sorted(quad_overlap):
            muts = patient_mutations[pid]
            mut_str = ", ".join(
                f"{g}:{muts[g][0]['protein_change']}"
                for g in genes if g in muts and muts[g]
            )
            age_str = ""
            if patient_clinical and pid in patient_clinical:
                age = patient_clinical[pid].get("age")
                if age is not None:
                    age_str = f" (age {age})"
            print(f"      {pid}: {mut_str}{age_str}")
    else:
        print(f"\n    Quadruple (all four genes): 0 patients")

    return {
        "total_patients_with_target_mutations": total,
        "single_counts": single_counts,
        "pairwise_counts": pairwise_counts,
        "pairwise_patients": pairwise_patients_map,
        "triple_counts": triple_counts,
        "triple_patients": triple_patients_map,
        "quad_count": quad_count,
        "quad_patients": sorted(list(quad_overlap)),
    }


def compute_expected_values(single_counts: dict, total_myeloid: int) -> dict:
    """Compute expected co-occurrence counts under independence."""
    expected = {}
    genes = TARGET_GENES
    freqs = {g: single_counts[g] / total_myeloid for g in genes if total_myeloid > 0}

    for g1, g2 in combinations(genes, 2):
        key = f"{g1}+{g2}"
        expected[key] = freqs.get(g1, 0) * freqs.get(g2, 0) * total_myeloid

    for combo in combinations(genes, 3):
        key = "+".join(combo)
        p = 1.0
        for g in combo:
            p *= freqs.get(g, 0)
        expected[key] = p * total_myeloid

    # Quadruple
    p_all = 1.0
    for g in genes:
        p_all *= freqs.get(g, 0)
    expected["DNMT3A+IDH2+PTPN11+SETBP1"] = p_all * total_myeloid

    return expected


def check_panel_coverage(myeloid_samples: dict, gene_panels: dict) -> dict:
    """Check how many myeloid samples have SETBP1 on their sequencing panel."""
    coverage = {g: 0 for g in TARGET_GENES}
    all_four_coverage = 0
    total = len(myeloid_samples)

    for sample_id, info in myeloid_samples.items():
        panel_id = info.get("seq_panel", "")
        if panel_id in gene_panels:
            panel_genes = gene_panels[panel_id]
            all_covered = True
            for gene in TARGET_GENES:
                if gene in panel_genes:
                    coverage[gene] += 1
                else:
                    all_covered = False
            if all_covered:
                all_four_coverage += 1
        else:
            # If panel not found, assume all genes covered (conservative)
            for gene in TARGET_GENES:
                coverage[gene] += 1
            all_four_coverage += 1

    return {
        "total_myeloid_samples": total,
        "per_gene_coverage": coverage,
        "all_four_genes_covered": all_four_coverage,
    }


def main():
    print("=" * 70)
    print("AACR GENIE Synapse Data Analysis")
    print("Target genes: DNMT3A, IDH2, PTPN11, SETBP1")
    print("=" * 70)

    # Find data files
    files = find_genie_files()
    print(f"\nFiles found in {GENIE_RAW_DIR}:")
    for key, val in files.items():
        if isinstance(val, list):
            for v in val:
                print(f"  {key}: {v.name}")
        else:
            print(f"  {key}: {val.name}")

    if "mutations" not in files:
        print("\n*** ERROR: No mutations file found! ***")
        print(f"Expected one of these files in {GENIE_RAW_DIR}:")
        print("  - data_mutations_extended.txt")
        print("  - data_mutations.txt")
        print("\nTo obtain GENIE data:")
        print("  1. Create account at https://www.synapse.org/")
        print("  2. Navigate to https://www.synapse.org/#!Synapse:syn7222066")
        print("  3. Request and receive data access approval")
        print("  4. Download release files to mutation_profile/data/genie/raw/")
        print("\nSee mutation_profile/data/genie/genie_data_assessment.md for full instructions.")
        sys.exit(1)

    if "clinical_sample" not in files:
        print("\n*** WARNING: No clinical sample file found! ***")
        print("Will attempt to identify myeloid samples from mutation data only.")

    # Load gene panels
    gene_panels = {}
    if "gene_panels" in files:
        print(f"\nLoading {len(files['gene_panels'])} gene panel files...")
        gene_panels = load_gene_panels(files["gene_panels"])
        print(f"  Loaded {len(gene_panels)} panels")

        # Check which panels include SETBP1
        setbp1_panels = [p for p, genes in gene_panels.items() if "SETBP1" in genes]
        print(f"  Panels with SETBP1: {len(setbp1_panels)} of {len(gene_panels)}")
        if setbp1_panels:
            print(f"    Examples: {setbp1_panels[:5]}")

    # Load clinical sample data
    myeloid_samples = {}
    all_samples = {}
    if "clinical_sample" in files:
        print(f"\nLoading clinical sample data...")
        all_samples = load_clinical_samples(files["clinical_sample"])
        print(f"  Total samples: {len(all_samples)}")

        # Filter to myeloid
        myeloid_samples = {
            sid: info for sid, info in all_samples.items()
            if is_myeloid_sample(info)
        }
        print(f"  Myeloid samples: {len(myeloid_samples)}")

        # Count by OncoTree code
        code_counts = defaultdict(int)
        for info in myeloid_samples.values():
            code = info.get("oncotree_code", "UNKNOWN")
            code_counts[code] += 1
        print(f"  OncoTree code distribution (top 20):")
        for code, count in sorted(code_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"    {code}: {count}")

        # Check panel coverage
        if gene_panels:
            print(f"\n  Panel coverage for myeloid samples:")
            coverage = check_panel_coverage(myeloid_samples, gene_panels)
            for gene, count in coverage["per_gene_coverage"].items():
                pct = count / len(myeloid_samples) * 100 if myeloid_samples else 0
                print(f"    {gene}: {count} ({pct:.1f}%)")
            print(f"    All 4 genes: {coverage['all_four_genes_covered']} "
                  f"({coverage['all_four_genes_covered'] / len(myeloid_samples) * 100:.1f}%)")
    else:
        # Without clinical data, use all samples
        print("\n  No clinical sample file -- will process all samples")
        myeloid_samples = None

    # Load patient-level clinical data
    patient_clinical = {}
    if "clinical_patient" in files:
        print(f"\nLoading patient clinical data...")
        patient_clinical = load_clinical_patients(files["clinical_patient"])
        print(f"  Total patients: {len(patient_clinical)}")

    # Load mutations
    print(f"\nLoading mutations...")
    if myeloid_samples:
        patient_mutations = load_mutations(files["mutations"], myeloid_samples)
    else:
        # Without sample filtering, load all -- but this will be slow for 271K samples
        print("  WARNING: Loading ALL mutations (no myeloid filter). This may take a long time.")
        # Create a dummy sample dict that accepts everything
        class AcceptAll(dict):
            def __contains__(self, key):
                return True
            def __getitem__(self, key):
                return {"patient_id": key}
        patient_mutations = load_mutations(files["mutations"], AcceptAll())

    # Compute co-occurrence
    print(f"\nComputing co-occurrence...")
    results = compute_cooccurrence(patient_mutations, patient_clinical)

    # Compute expected values
    total_myeloid = len(myeloid_samples) if myeloid_samples else results["total_patients_with_target_mutations"]
    expected = compute_expected_values(results["single_counts"], total_myeloid)

    # Compute obs/exp ratios
    obs_exp = {}
    for key in expected:
        obs = results["pairwise_counts"].get(key,
              results["triple_counts"].get(key,
              results["quad_count"] if key == "DNMT3A+IDH2+PTPN11+SETBP1" else 0))
        exp = expected[key]
        ratio = obs / exp if exp > 0 else None
        obs_exp[key] = {
            "observed": obs,
            "expected": round(exp, 2),
            "ratio": round(ratio, 2) if ratio is not None else None,
        }

    print(f"\n  Observed/Expected ratios:")
    for key, vals in obs_exp.items():
        print(f"    {key}: obs={vals['observed']}, exp={vals['expected']}, "
              f"ratio={vals['ratio']}")

    # Build output JSON
    output = {
        "database": "AACR GENIE Synapse",
        "version": "v19.0-public (or as downloaded)",
        "analysis_date": "2026-03-14",
        "total_samples_in_dataset": len(all_samples) if all_samples else "unknown",
        "total_myeloid_samples": len(myeloid_samples) if myeloid_samples else "unknown (no clinical file)",
        "total_patients_with_target_mutations": results["total_patients_with_target_mutations"],
        "disease_filter": "AML/MDS/myeloid (OncoTree codes)",
        "oncotree_codes_used": sorted(list(MYELOID_ONCOTREE_CODES)),
        "target_genes": TARGET_GENES,
        "single_gene": results["single_counts"],
        "pairwise": results["pairwise_counts"],
        "triple": results["triple_counts"],
        "quadruple": {
            "DNMT3A+IDH2+PTPN11+SETBP1": results["quad_count"]
        },
        "obs_exp_ratios": obs_exp,
        "triple_patient_details": {},
        "quadruple_patient_details": results["quad_patients"],
    }

    # Add detailed patient info for triples
    for key, patient_ids in results["triple_patients"].items():
        if patient_ids:
            details = []
            for pid in patient_ids[:50]:  # Limit to 50 per triple
                muts = patient_mutations.get(pid, {})
                detail = {"patient_id": pid}
                for gene in TARGET_GENES:
                    if gene in muts and muts[gene]:
                        detail[gene] = muts[gene][0]["protein_change"]
                if patient_clinical and pid in patient_clinical:
                    clin = patient_clinical[pid]
                    if clin.get("age") is not None:
                        detail["age"] = clin["age"]
                    if clin.get("sex"):
                        detail["sex"] = clin["sex"]
                details.append(detail)
            output["triple_patient_details"][key] = details

    # Add quadruple patient details
    if results["quad_patients"]:
        quad_details = []
        for pid in results["quad_patients"]:
            muts = patient_mutations.get(pid, {})
            detail = {"patient_id": pid}
            for gene in TARGET_GENES:
                if gene in muts and muts[gene]:
                    detail[gene] = [m["protein_change"] for m in muts[gene]]
            if patient_clinical and pid in patient_clinical:
                clin = patient_clinical[pid]
                detail["age"] = clin.get("age")
                detail["sex"] = clin.get("sex")
                detail["center"] = clin.get("center")
            quad_details.append(detail)
        output["quadruple_patient_details"] = quad_details

    # Panel coverage info
    if gene_panels and myeloid_samples:
        coverage = check_panel_coverage(myeloid_samples, gene_panels)
        output["panel_coverage"] = coverage

    # Write output
    output_path = RESULTS_DIR / "genie_v2_cooccurrence.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Output written to: {output_path}")
    print(f"{'=' * 70}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total myeloid samples: {output['total_myeloid_samples']}")
    print(f"  Patients with target gene mutations: {results['total_patients_with_target_mutations']}")
    for gene in TARGET_GENES:
        print(f"  {gene}: {results['single_counts'].get(gene, 0)}")
    print(f"  DNMT3A+IDH2+PTPN11+SETBP1 quadruple: {results['quad_count']}")

    if results["quad_count"] == 0:
        print(f"\n  The four-gene combination remains unobserved in GENIE data.")
        print(f"  This confirms findings from cBioPortal proxy analysis.")
    else:
        print(f"\n  *** NOVEL FINDING: {results['quad_count']} patient(s) with all four mutations! ***")
        print(f"  This would be the first documented case(s) in any public database.")

    return output


if __name__ == "__main__":
    result = main()
