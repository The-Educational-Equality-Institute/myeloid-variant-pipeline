#!/usr/bin/env python3
"""
Fixed demographics analysis for GENIE mutation carriers.

Bug in deeper_demographics.py: AGE_AT_SEQ_REPORT was looked up in patient file
(doesn't exist there), and CENTER was looked up from sample file (doesn't exist there).

Fix: AGE_AT_SEQ_REPORT is in data_clinical_sample.txt (column 3).
     CENTER is in data_clinical_patient.txt (column 5).
     SEX is in data_clinical_patient.txt (column 2).
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
GENIE_RAW = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]

# Myeloid OncoTree codes -- comprehensive set
MYELOID_ONCOTREE_PREFIXES = {"AML", "MDS", "MPN", "CMML", "JMML", "CML", "TMN", "SM", "CEL", "HES"}
MYELOID_ONCOTREE_EXACT = {
    "AMML", "APL", "APMF", "AUL", "MLADS", "MS",
    # Extended set from prior analysis
    "AMLMRC", "AMLRGA", "AMLRR", "AMLNOS", "AMOL",
    "AMLMBC", "AMLCBFB", "AMLRUNX1", "AMLMLLT3", "AMLDEKNUP", "AMLBCR",
    "AMLGATA2MECOM", "AMLNPM1", "AMLCEBPA", "AMLTP53", "AMLDEK",
    "MDS5Q", "MDSEB1", "MDSEB2", "MDSMD", "MDSSLD", "MDSRS",
    "MDSU", "MDSRSMD", "MDSSID", "MDSLB", "MDSIB1", "MDSIB2",
    "RCMD", "RARS", "RA", "RAEB", "RAEB1", "RAEB2", "RCUD",
    "CMML0", "CMML1", "CMML2", "MDSMPNU", "MDSMPNRST", "ACML",
    "ET", "PV", "PMF", "CMLBCRABL1", "MPNU", "MPNST",
    "BPDCN", "MPAL", "ALAL", "TMDS", "TAML",
}

DISEASE_CATEGORIES = {
    "AML": {"AML", "AMLMRC", "AMLRGA", "AMLRR", "AMLNOS", "APL", "AMOL", "APMF",
            "AMLMBC", "AMLCBFB", "AMLRUNX1", "AMLMLLT3", "AMLDEKNUP", "AMLBCR",
            "AMLGATA2MECOM", "AMLNPM1", "AMLCEBPA", "AMLTP53", "AMLDEK", "TAML",
            "AMML", "AUL", "MS", "MLADS"},
    "MDS": {"MDS", "MDS5Q", "MDSEB1", "MDSEB2", "MDSMD", "MDSSLD", "MDSRS",
            "MDSU", "MDSRSMD", "MDSSID", "MDSLB", "MDSIB1", "MDSIB2",
            "RCMD", "RARS", "RA", "RAEB", "RAEB1", "RAEB2", "RCUD", "TMDS"},
    "MDS/MPN": {"CMML", "CMML0", "CMML1", "CMML2", "JMML", "MDSMPNU",
                "MDSMPNRST", "ACML"},
    "MPN": {"MPN", "CML", "ET", "PV", "PMF", "CMLBCRABL1", "SM", "CEL",
            "MPNU", "MPNST", "HES"},
    "Other": {"BPDCN", "MPAL", "TMN", "ALAL"},
}

PATHOGENIC_VAR_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    "Translation_Start_Site",
}


def is_myeloid(code: str) -> bool:
    """Check if an OncoTree code is myeloid."""
    code_upper = code.upper().strip()
    if code_upper in MYELOID_ONCOTREE_EXACT:
        return True
    for prefix in MYELOID_ONCOTREE_PREFIXES:
        if code_upper.startswith(prefix):
            return True
    return False


def categorize_disease(code: str) -> str:
    code_upper = code.upper().strip()
    for cat, codes in DISEASE_CATEGORIES.items():
        if code_upper in codes:
            return cat
    return "Other/Unknown"


def parse_age(age_str: str):
    """Parse GENIE age. Can be integer, float, or bracket like '<18' or '>89'."""
    if not age_str or age_str in ("Unknown", "Not Collected", "NA", ""):
        return None
    age_str = age_str.strip()
    if age_str.startswith("<"):
        # e.g. "<18" -- use 17 as representative
        try:
            return float(age_str[1:]) - 1
        except ValueError:
            return None
    if age_str.startswith(">"):
        # e.g. ">89" -- use 90 as representative
        try:
            return float(age_str[1:]) + 1
        except ValueError:
            return None
    try:
        return float(age_str)
    except ValueError:
        return None


def parse_age_display(age_str: str) -> str:
    """Return the raw age string for display (preserving brackets)."""
    if not age_str or age_str in ("Unknown", "Not Collected", "NA", ""):
        return "Unknown"
    return age_str.strip()


def categorize_age(age) -> str:
    if age is None:
        return "Unknown"
    if age < 18:
        return "<18"
    elif age < 40:
        return "18-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    elif age < 80:
        return "70-79"
    else:
        return "80+"


def parse_tsv_with_comments(filepath):
    """Parse a GENIE clinical file: skip # lines, first non-# line is header."""
    rows = []
    header = None
    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            row = {}
            for i, col in enumerate(header):
                row[col] = parts[i] if i < len(parts) else ""
            rows.append(row)
    return header, rows


def main():
    print("=" * 80)
    print("FIXED DEMOGRAPHICS ANALYSIS - GENIE")
    print("=" * 80)

    # ---- Step 1: Parse clinical sample file ----
    print("\n[1] Parsing clinical sample file...")
    sample_header, sample_rows = parse_tsv_with_comments(
        GENIE_RAW / "data_clinical_sample.txt"
    )
    print(f"    Sample file columns: {sample_header}")
    print(f"    Total sample rows: {len(sample_rows):,}")

    # Build sample lookup: SAMPLE_ID -> row
    sample_lookup = {}
    for row in sample_rows:
        sid = row.get("SAMPLE_ID", "")
        if sid:
            sample_lookup[sid] = row

    # ---- Step 2: Parse clinical patient file ----
    print("\n[2] Parsing clinical patient file...")
    patient_header, patient_rows = parse_tsv_with_comments(
        GENIE_RAW / "data_clinical_patient.txt"
    )
    print(f"    Patient file columns: {patient_header}")
    print(f"    Total patient rows: {len(patient_rows):,}")

    # Build patient lookup: PATIENT_ID -> row
    patient_lookup = {}
    for row in patient_rows:
        pid = row.get("PATIENT_ID", "")
        if pid:
            patient_lookup[pid] = row

    # ---- Step 3: Build combined sample info (sample_id -> demographics) ----
    print("\n[3] Building combined sample -> demographics map...")

    # For each sample, merge: age from sample file, sex/center/race from patient file
    sample_info = {}  # sample_id -> dict
    myeloid_sample_ids = set()
    myeloid_patient_ids = set()

    for sid, srow in sample_lookup.items():
        oncotree = srow.get("ONCOTREE_CODE", "").strip()
        if not is_myeloid(oncotree):
            continue

        pid = srow.get("PATIENT_ID", "")
        prow = patient_lookup.get(pid, {})

        # AGE is in SAMPLE file
        age_raw = srow.get("AGE_AT_SEQ_REPORT", "")
        age = parse_age(age_raw)
        age_display = parse_age_display(age_raw)

        # SEX, CENTER, RACE are in PATIENT file
        sex = prow.get("SEX", "Unknown")
        center = prow.get("CENTER", "Unknown")
        race = prow.get("PRIMARY_RACE", "Unknown")

        sample_info[sid] = {
            "patient_id": pid,
            "sample_id": sid,
            "oncotree_code": oncotree,
            "disease_cat": categorize_disease(oncotree),
            "age": age,
            "age_raw": age_display,
            "age_group": categorize_age(age),
            "sex": sex,
            "center": center,
            "race": race,
        }
        myeloid_sample_ids.add(sid)
        myeloid_patient_ids.add(pid)

    print(f"    Myeloid samples: {len(myeloid_sample_ids):,}")
    print(f"    Myeloid patients: {len(myeloid_patient_ids):,}")

    # Quick sanity check: how many have known age/center/sex?
    known_age = sum(1 for s in sample_info.values() if s["age"] is not None)
    known_center = sum(1 for s in sample_info.values() if s["center"] not in ("Unknown", ""))
    known_sex = sum(1 for s in sample_info.values() if s["sex"] not in ("Unknown", ""))
    print(f"    Known age: {known_age:,} ({known_age/len(sample_info)*100:.1f}%)")
    print(f"    Known center: {known_center:,} ({known_center/len(sample_info)*100:.1f}%)")
    print(f"    Known sex: {known_sex:,} ({known_sex/len(sample_info)*100:.1f}%)")

    # Show first 3 samples as sanity check
    print("\n    Sample data check (first 3 myeloid samples):")
    for i, (sid, info) in enumerate(sample_info.items()):
        if i >= 3:
            break
        print(f"      {sid}: age={info['age_raw']}, sex={info['sex']}, "
              f"center={info['center']}, oncotree={info['oncotree_code']}")

    # ---- Step 4: Scan MAF for target gene mutations ----
    print("\n[4] Scanning MAF for target gene mutations (protein-altering only)...")
    TARGET_SET = set(TARGET_GENES)

    # Track: sample_id -> set of mutated genes, and also store variant details
    sample_genes = defaultdict(set)     # sample_id -> {gene, ...}
    sample_variants = defaultdict(list) # sample_id -> [(gene, variant, var_class), ...]

    maf_path = GENIE_RAW / "data_mutations_extended.txt"
    row_count = 0
    matched = 0

    with open(maf_path) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                # Find column indices for speed
                idx_gene = header.index("Hugo_Symbol")
                idx_barcode = header.index("Tumor_Sample_Barcode")
                idx_varclass = header.index("Variant_Classification")
                idx_hgvsp = header.index("HGVSp_Short")
                continue

            row_count += 1
            if row_count % 2_000_000 == 0:
                print(f"    Processed {row_count:,} MAF rows, {matched:,} matched...")

            gene = parts[idx_gene] if idx_gene < len(parts) else ""
            if gene not in TARGET_SET:
                continue

            barcode = parts[idx_barcode] if idx_barcode < len(parts) else ""
            if barcode not in myeloid_sample_ids:
                continue

            var_class = parts[idx_varclass] if idx_varclass < len(parts) else ""
            if var_class not in PATHOGENIC_VAR_CLASSES:
                continue

            hgvsp = parts[idx_hgvsp] if idx_hgvsp < len(parts) else ""
            # Skip synonymous-like (e.g. p.*123*)
            if re.match(r'^p\.\*\d+\*$', hgvsp):
                continue

            sample_genes[barcode].add(gene)
            sample_variants[barcode].append((gene, hgvsp, var_class))
            matched += 1

    print(f"    Total MAF rows scanned: {row_count:,}")
    print(f"    Matched mutations: {matched:,}")
    print(f"    Samples with target mutations: {len(sample_genes):,}")

    # ---- Step 5: Aggregate to patient level (a patient may have multiple samples) ----
    patient_genes = defaultdict(set)    # patient_id -> {gene, ...}
    patient_best_sample = {}            # patient_id -> sample_id (pick first myeloid sample with mutations)

    for sid, genes in sample_genes.items():
        info = sample_info.get(sid)
        if not info:
            continue
        pid = info["patient_id"]
        patient_genes[pid].update(genes)
        if pid not in patient_best_sample:
            patient_best_sample[pid] = sid

    print(f"    Patients with target mutations: {len(patient_genes):,}")

    # ---- Step 6: Build carrier sets per gene ----
    gene_carriers = {g: set() for g in TARGET_GENES}
    for pid, genes in patient_genes.items():
        for g in genes:
            gene_carriers[g].add(pid)

    for g in TARGET_GENES:
        print(f"    {g}: {len(gene_carriers[g]):,} carriers")

    # Build combo carriers
    combo_carriers = {}
    for r in (2, 3):
        for combo in combinations(TARGET_GENES, r):
            key = "+".join(combo)
            overlap = gene_carriers[combo[0]]
            for g in combo[1:]:
                overlap = overlap & gene_carriers[g]
            combo_carriers[key] = overlap

    # Quad
    quad_key = "+".join(TARGET_GENES)
    quad = gene_carriers[TARGET_GENES[0]]
    for g in TARGET_GENES[1:]:
        quad = quad & gene_carriers[g]
    combo_carriers[quad_key] = quad

    print(f"\n    Co-occurrence counts:")
    for key, pids in sorted(combo_carriers.items()):
        print(f"      {key}: {len(pids)}")

    # ---- Helper: get demographics for a set of patient IDs ----
    def get_demographics(patient_ids):
        ages = []
        age_dist = Counter()
        sex_dist = Counter()
        center_dist = Counter()
        disease_dist = Counter()
        race_dist = Counter()

        for pid in patient_ids:
            sid = patient_best_sample.get(pid)
            if not sid:
                # Fallback: find any myeloid sample for this patient
                for s, info in sample_info.items():
                    if info["patient_id"] == pid:
                        sid = s
                        break
            if not sid:
                continue

            info = sample_info[sid]
            age = info["age"]
            if age is not None:
                ages.append(age)
            age_dist[info["age_group"]] += 1
            sex_dist[info["sex"]] += 1
            center_dist[info["center"]] += 1
            disease_dist[info["disease_cat"]] += 1
            race_dist[info["race"]] += 1

        ages.sort()
        n_ages = len(ages)
        result = {
            "n": len(patient_ids),
            "n_with_known_age": n_ages,
            "mean_age": round(statistics.mean(ages), 1) if ages else None,
            "median_age": round(statistics.median(ages), 1) if ages else None,
            "age_range": [round(ages[0], 1), round(ages[-1], 1)] if ages else None,
            "under_40": sum(1 for a in ages if a < 40),
            "under_40_pct": round(sum(1 for a in ages if a < 40) / n_ages * 100, 1) if n_ages else None,
            "age_distribution": {ag: age_dist.get(ag, 0) for ag in
                                 ["<18", "18-39", "40-49", "50-59", "60-69", "70-79", "80+", "Unknown"]},
            "sex_distribution": dict(sex_dist.most_common()),
            "center_top10": dict(center_dist.most_common(10)),
            "disease_distribution": dict(disease_dist.most_common()),
            "race_distribution": dict(race_dist.most_common()),
        }

        # Henrik percentile (age 33)
        if n_ages > 0:
            younger = sum(1 for a in ages if a <= 33)
            result["index_patient_percentile"] = round(younger / n_ages * 100, 1)
        else:
            result["index_patient_percentile"] = None

        return result

    # ====================================================================
    # ANALYSIS OUTPUT
    # ====================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    output = {
        "analysis": "Fixed Demographics - GENIE",
        "total_myeloid_samples": len(myeloid_sample_ids),
        "total_myeloid_patients": len(myeloid_patient_ids),
        "known_age_pct": round(known_age / len(sample_info) * 100, 1) if sample_info else 0,
        "known_center_pct": round(known_center / len(sample_info) * 100, 1) if sample_info else 0,
        "known_sex_pct": round(known_sex / len(sample_info) * 100, 1) if sample_info else 0,
    }

    # ---- 6a: Per-gene demographics ----
    print("\n" + "-" * 80)
    print("PER-GENE DEMOGRAPHICS")
    print("-" * 80)

    output["per_gene"] = {}
    for g in TARGET_GENES:
        carriers = gene_carriers[g]
        demo = get_demographics(carriers)
        output["per_gene"][g] = demo

        print(f"\n  {g} (n={demo['n']}, {demo['n_with_known_age']} with known age)")
        if demo["mean_age"]:
            print(f"    Mean age: {demo['mean_age']}, Median: {demo['median_age']}")
            print(f"    Range: {demo['age_range'][0]} - {demo['age_range'][1]}")
            print(f"    Under 40: {demo['under_40']} ({demo['under_40_pct']}%)")
            print(f"    Index patient (age 33) percentile: {demo['index_patient_percentile']}%")
        print(f"    Age distribution:")
        for ag in ["<18", "18-39", "40-49", "50-59", "60-69", "70-79", "80+", "Unknown"]:
            count = demo["age_distribution"].get(ag, 0)
            pct = count / demo["n"] * 100 if demo["n"] else 0
            bar = "#" * int(pct / 2)
            print(f"      {ag:>8}: {count:>5} ({pct:>5.1f}%) {bar}")
        print(f"    Sex: {demo['sex_distribution']}")
        print(f"    Disease: {demo['disease_distribution']}")
        print(f"    Top centers:")
        for center, count in list(demo["center_top10"].items())[:10]:
            pct = count / demo["n"] * 100 if demo["n"] else 0
            print(f"      {center:>12}: {count:>4} ({pct:>5.1f}%)")

    # ---- 6b: Co-occurring pair demographics ----
    print("\n" + "-" * 80)
    print("CO-OCCURRING PAIR DEMOGRAPHICS")
    print("-" * 80)

    output["pairs"] = {}
    for combo in combinations(TARGET_GENES, 2):
        key = "+".join(combo)
        carriers = combo_carriers[key]
        demo = get_demographics(carriers)
        output["pairs"][key] = demo

        print(f"\n  {key} (n={demo['n']})")
        if demo["n"] == 0:
            print("    No carriers found")
            continue
        if demo["mean_age"]:
            print(f"    Mean age: {demo['mean_age']}, Median: {demo['median_age']}")
            print(f"    Range: {demo['age_range'][0]} - {demo['age_range'][1]}")
            print(f"    Under 40: {demo['under_40']} ({demo['under_40_pct']}%)")
            print(f"    Index patient (age 33) percentile: {demo['index_patient_percentile']}%")
        print(f"    Age distribution:")
        for ag in ["<18", "18-39", "40-49", "50-59", "60-69", "70-79", "80+", "Unknown"]:
            count = demo["age_distribution"].get(ag, 0)
            pct = count / demo["n"] * 100 if demo["n"] else 0
            print(f"      {ag:>8}: {count:>5} ({pct:>5.1f}%)")
        print(f"    Sex: {demo['sex_distribution']}")
        print(f"    Top centers:")
        for center, count in list(demo["center_top10"].items())[:10]:
            pct = count / demo["n"] * 100 if demo["n"] else 0
            print(f"      {center:>12}: {count:>4} ({pct:>5.1f}%)")

    # ---- 6c: Triple demographics ----
    print("\n" + "-" * 80)
    print("TRIPLE COMBINATION DEMOGRAPHICS")
    print("-" * 80)

    output["triples"] = {}
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        carriers = combo_carriers[key]
        demo = get_demographics(carriers)
        output["triples"][key] = demo

        print(f"\n  {key} (n={demo['n']})")
        if demo["n"] == 0:
            print("    No carriers found")
            continue
        if demo["mean_age"]:
            print(f"    Mean age: {demo['mean_age']}, Median: {demo['median_age']}")
            print(f"    Range: {demo['age_range'][0]} - {demo['age_range'][1]}")
            print(f"    Under 40: {demo['under_40']} ({demo['under_40_pct']}%)")
            print(f"    Index patient (age 33) percentile: {demo['index_patient_percentile']}%")
        print(f"    Sex: {demo['sex_distribution']}")
        print(f"    Centers: {demo['center_top10']}")
        print(f"    Disease: {demo['disease_distribution']}")

    # ---- 6d: Quad ----
    print("\n" + "-" * 80)
    print("QUADRUPLE COMBINATION")
    print("-" * 80)

    quad_demo = get_demographics(combo_carriers[quad_key])
    output["quadruple"] = {quad_key: quad_demo}
    print(f"\n  {quad_key} (n={quad_demo['n']})")
    if quad_demo["n"] > 0 and quad_demo["mean_age"]:
        print(f"    Mean age: {quad_demo['mean_age']}, Median: {quad_demo['median_age']}")
        print(f"    Under 40: {quad_demo['under_40']} ({quad_demo['under_40_pct']}%)")
        print(f"    Sex: {quad_demo['sex_distribution']}")
        print(f"    Centers: {quad_demo['center_top10']}")
    elif quad_demo["n"] == 0:
        print("    ZERO carriers -- confirms rarity of Henrik's profile")

    # ---- 6e: Henrik comparison summary ----
    print("\n" + "-" * 80)
    print("HENRIK COMPARISON SUMMARY (age 33 at diagnosis)")
    print("-" * 80)

    index_patient_summary = {}
    for g in TARGET_GENES:
        demo = output["per_gene"][g]
        pct = demo["index_patient_percentile"]
        under40 = demo["under_40"]
        total = demo["n_with_known_age"]
        index_patient_summary[g] = {
            "percentile": pct,
            "under_40_of_total": f"{under40}/{total}",
            "under_40_pct": demo["under_40_pct"],
        }
        print(f"  {g}: Index patient at {pct}th percentile "
              f"({under40}/{total} = {demo['under_40_pct']}% are under 40)")

    for key in combo_carriers:
        if "+" in key:
            demo_key = None
            parts = key.split("+")
            if len(parts) == 2:
                demo_key = output["pairs"].get(key, {})
            elif len(parts) == 3:
                demo_key = output["triples"].get(key, {})
            elif len(parts) == 4:
                demo_key = output["quadruple"].get(key, {})
            if demo_key and demo_key.get("index_patient_percentile") is not None:
                print(f"  {key}: Index patient at {demo_key['index_patient_percentile']}th percentile "
                      f"(under 40: {demo_key.get('under_40', 0)}/{demo_key.get('n_with_known_age', 0)})")

    output["index_patient_comparison"] = index_patient_summary

    # ---- Save ----
    output_path = RESULTS_DIR / "genie_demographics_fixed.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
