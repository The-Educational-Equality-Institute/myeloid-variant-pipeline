#!/usr/bin/env python3
"""
Demographics analysis: age, sex, disease subtype distribution for mutation carriers.
Scans clinical files only (no MAF needed) + uses filtered results.
"""

import json
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
GENIE_RAW = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]

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
    "BPDCN", "MPAL", "TMN", "ALAL", "TMDS", "TAML",
}

# Disease categories
DISEASE_CATEGORIES = {
    "AML": {"AML", "AMLMRC", "AMLRGA", "AMLRR", "AMLNOS", "APL", "AMOL", "APMF",
            "AMLMBC", "AMLCBFB", "AMLRUNX1", "AMLMLLT3", "AMLDEKNUP", "AMLBCR",
            "AMLGATA2MECOM", "AMLNPM1", "AMLCEBPA", "AMLTP53", "AMLDEK", "TAML"},
    "MDS": {"MDS", "MDS5Q", "MDSEB1", "MDSEB2", "MDSMD", "MDSSLD", "MDSRS",
            "MDSU", "MDSRSMD", "MDSSID", "MDSLB", "MDSIB1", "MDSIB2",
            "RCMD", "RARS", "RA", "RAEB", "RAEB1", "RAEB2", "RCUD", "TMDS"},
    "MDS/MPN": {"CMML", "CMML0", "CMML1", "CMML2", "JMML", "MDSMPNU",
                "MDSMPNRST", "ACML", "ACML_ATYPICAL"},
    "MPN": {"MPN", "CML", "ET", "PV", "PMF", "CMLBCRABL1", "SM", "CEL",
            "MPNU", "MPNST"},
    "Other": {"BPDCN", "MPAL", "TMN", "ALAL"},
}

PATHOGENIC_VAR_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    "Translation_Start_Site",
}


def categorize_disease(oncotree_code):
    code = oncotree_code.upper()
    for cat, codes in DISEASE_CATEGORIES.items():
        if code in codes:
            return cat
    return "Other/Unknown"


def categorize_age(age):
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
    elif age < 70:
        return "60-69"
    elif age < 80:
        return "70-79"
    else:
        return "80+"


def main():
    print("=" * 80)
    print("DEMOGRAPHICS ANALYSIS - GENIE v19.0")
    print("=" * 80)

    # Load clinical sample data
    print("\nLoading clinical data...")
    samples = {}
    with open(GENIE_RAW / "data_clinical_sample.txt") as f:
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
            if sid:
                samples[sid] = row

    # Load patient data
    patients = {}
    with open(GENIE_RAW / "data_clinical_patient.txt") as f:
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
            if pid:
                patients[pid] = row

    # Build myeloid sample -> patient map with disease info
    myeloid_patient_info = {}  # patient_id -> {oncotree_code, disease_cat, age, sex, center}
    for sid, srow in samples.items():
        code = srow.get("ONCOTREE_CODE", "").upper()
        if code not in MYELOID_ONCOTREE_CODES:
            continue
        pid = srow.get("PATIENT_ID", "")
        if not pid:
            continue

        prow = patients.get(pid, {})
        age_str = prow.get("AGE_AT_SEQ_REPORT", "")
        try:
            age = float(age_str)
        except (ValueError, TypeError):
            age = None

        if pid not in myeloid_patient_info:
            myeloid_patient_info[pid] = {
                "oncotree_code": code,
                "disease_cat": categorize_disease(code),
                "age": age,
                "age_group": categorize_age(age),
                "sex": prow.get("SEX", "Unknown"),
                "race": prow.get("PRIMARY_RACE", "Unknown"),
                "center": srow.get("CENTER", "Unknown"),
                "sample_id": sid,
            }

    print(f"  Myeloid patients (strict OncoTree): {len(myeloid_patient_info):,}")

    # Scan MAF for mutation status per patient
    print("\nScanning MAF for target gene mutations...")
    import re
    patient_genes = defaultdict(set)  # patient_id -> set of mutated target genes

    myeloid_sids = {info["sample_id"] for info in myeloid_patient_info.values()}

    with open(GENIE_RAW / "data_mutations_extended.txt") as f:
        header = None
        row_count = 0
        for line in f:
            if line.startswith("#"):
                continue
            if "Hugo_Symbol" in line and header is None:
                header = line.strip().split("\t")
                continue
            if header is None:
                header = line.strip().split("\t")
                continue

            row_count += 1
            if row_count % 1_000_000 == 0:
                print(f"  {row_count:,} rows...")

            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))

            gene = row.get("Hugo_Symbol", "")
            if gene not in set(TARGET_GENES):
                continue

            sid = row.get("Tumor_Sample_Barcode", "")
            if sid not in myeloid_sids:
                continue

            var_class = row.get("Variant_Classification", "")
            if var_class not in PATHOGENIC_VAR_CLASSES:
                continue

            protein = row.get("HGVSp_Short", "")
            if re.match(r'^p\.\*\d+\*$', protein):
                continue

            pid = samples[sid].get("PATIENT_ID", "")
            if pid:
                patient_genes[pid].add(gene)

    print(f"  Patients with target gene mutations: {len(patient_genes):,}")

    # ──────────────────────────────────────────────────────────────────
    # ANALYSIS BY GENE
    # ──────────────────────────────────────────────────────────────────
    gene_carriers = {g: set() for g in TARGET_GENES}
    for pid, genes in patient_genes.items():
        for g in genes:
            gene_carriers[g].add(pid)

    # Also build combo carriers
    combo_carriers = {}
    for combo in combinations(TARGET_GENES, 2):
        key = "+".join(combo)
        combo_carriers[key] = gene_carriers[combo[0]] & gene_carriers[combo[1]]
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        overlap = gene_carriers[combo[0]]
        for g in combo[1:]:
            overlap = overlap & gene_carriers[g]
        combo_carriers[key] = overlap

    # ──────────────────────────────────────────────────────────────────
    # 1. AGE DISTRIBUTION
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("1. AGE DISTRIBUTION BY GENE")
    print("─" * 80)

    age_groups = ["<18", "18-39", "40-49", "50-59", "60-69", "70-79", "80+", "Unknown"]

    for g in TARGET_GENES:
        carriers = gene_carriers[g]
        ages = []
        age_dist = Counter()
        for pid in carriers:
            info = myeloid_patient_info.get(pid, {})
            age = info.get("age")
            age_dist[info.get("age_group", "Unknown")] += 1
            if age is not None:
                ages.append(age)

        ages.sort()
        n = len(ages)
        median = ages[n // 2] if n > 0 else None
        mean = sum(ages) / n if n > 0 else None
        young = sum(1 for a in ages if a < 40)

        print(f"\n  {g} (n={len(carriers)}, {n} with known age):")
        if mean:
            print(f"    Mean age: {mean:.1f}, Median: {median:.0f}")
            print(f"    Range: {ages[0]:.0f} - {ages[-1]:.0f}")
            print(f"    Under 40: {young} ({young/n*100:.1f}%)")
        for ag in age_groups:
            count = age_dist.get(ag, 0)
            pct = count / len(carriers) * 100 if carriers else 0
            bar = "#" * int(pct / 2)
            print(f"    {ag:>8}: {count:>5} ({pct:>5.1f}%) {bar}")

    # Henrik comparison
    print(f"\n  HENRIK COMPARISON (age 33 at diagnosis):")
    for g in TARGET_GENES:
        carriers = gene_carriers[g]
        ages = []
        for pid in carriers:
            info = myeloid_patient_info.get(pid, {})
            age = info.get("age")
            if age is not None:
                ages.append(age)
        young = sum(1 for a in ages if a < 40)
        pct = young / len(ages) * 100 if ages else 0
        print(f"    {g}: {young}/{len(ages)} patients under 40 ({pct:.1f}%)")

    # ──────────────────────────────────────────────────────────────────
    # 2. SEX DISTRIBUTION
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("2. SEX DISTRIBUTION BY GENE")
    print("─" * 80)

    for g in TARGET_GENES:
        carriers = gene_carriers[g]
        sex_dist = Counter()
        for pid in carriers:
            info = myeloid_patient_info.get(pid, {})
            sex_dist[info.get("sex", "Unknown")] += 1

        total = len(carriers)
        print(f"\n  {g} (n={total}):")
        for sex in ["Male", "Female", "Unknown"]:
            count = sex_dist.get(sex, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"    {sex:>10}: {count:>5} ({pct:>5.1f}%)")

    # ──────────────────────────────────────────────────────────────────
    # 3. DISEASE SUBTYPE DISTRIBUTION
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("3. DISEASE SUBTYPE DISTRIBUTION BY GENE")
    print("─" * 80)

    categories = ["AML", "MDS", "MDS/MPN", "MPN", "Other/Unknown"]

    # Baseline distribution
    baseline_dist = Counter()
    for pid, info in myeloid_patient_info.items():
        baseline_dist[info["disease_cat"]] += 1

    print(f"\n  Baseline (all myeloid, n={len(myeloid_patient_info):,}):")
    for cat in categories:
        count = baseline_dist.get(cat, 0)
        pct = count / len(myeloid_patient_info) * 100
        print(f"    {cat:>15}: {count:>6} ({pct:>5.1f}%)")

    for g in TARGET_GENES:
        carriers = gene_carriers[g]
        disease_dist = Counter()
        for pid in carriers:
            info = myeloid_patient_info.get(pid, {})
            disease_dist[info.get("disease_cat", "Other/Unknown")] += 1

        total = len(carriers)
        print(f"\n  {g} (n={total}):")
        for cat in categories:
            count = disease_dist.get(cat, 0)
            pct = count / total * 100 if total > 0 else 0
            base_pct = baseline_dist.get(cat, 0) / len(myeloid_patient_info) * 100
            enrichment = pct / base_pct if base_pct > 0 else 0
            enr_str = f"({enrichment:.1f}x)" if enrichment > 1.5 or enrichment < 0.67 else ""
            print(f"    {cat:>15}: {count:>5} ({pct:>5.1f}%) {enr_str}")

    # ──────────────────────────────────────────────────────────────────
    # 4. CO-OCCURRENCE DEMOGRAPHICS
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("4. DEMOGRAPHICS OF CO-OCCURRING PATIENTS")
    print("─" * 80)

    for key, carriers in combo_carriers.items():
        if len(carriers) < 3:
            continue

        ages = []
        sex_dist = Counter()
        disease_dist = Counter()
        for pid in carriers:
            info = myeloid_patient_info.get(pid, {})
            age = info.get("age")
            if age is not None:
                ages.append(age)
            sex_dist[info.get("sex", "Unknown")] += 1
            disease_dist[info.get("disease_cat", "Other/Unknown")] += 1

        n = len(carriers)
        ages.sort()
        median = ages[len(ages) // 2] if ages else None
        mean = sum(ages) / len(ages) if ages else None
        young = sum(1 for a in ages if a < 40)

        print(f"\n  {key} (n={n}):")
        if mean:
            print(f"    Age: mean={mean:.1f}, median={median:.0f}, range={ages[0]:.0f}-{ages[-1]:.0f}")
            print(f"    Under 40: {young} ({young/len(ages)*100:.1f}%)")
        print(f"    Sex: ", end="")
        for sex in ["Male", "Female"]:
            count = sex_dist.get(sex, 0)
            print(f"{sex}={count} ({count/n*100:.0f}%) ", end="")
        print()
        print(f"    Disease: ", end="")
        for cat in categories[:4]:
            count = disease_dist.get(cat, 0)
            if count > 0:
                print(f"{cat}={count} ", end="")
        print()

    # ──────────────────────────────────────────────────────────────────
    # 5. CENTER DISTRIBUTION
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("5. CENTER DISTRIBUTION (top centers by mutation carriers)")
    print("─" * 80)

    for g in TARGET_GENES:
        carriers = gene_carriers[g]
        center_dist = Counter()
        for pid in carriers:
            info = myeloid_patient_info.get(pid, {})
            center_dist[info.get("center", "Unknown")] += 1

        print(f"\n  {g} (n={len(carriers)}):")
        for center, count in center_dist.most_common(10):
            pct = count / len(carriers) * 100
            print(f"    {center:>10}: {count:>5} ({pct:>5.1f}%)")

    # Save results
    output = {
        "analysis": "Demographics - GENIE v19.0",
        "myeloid_patients": len(myeloid_patient_info),
        "gene_carriers": {g: len(gene_carriers[g]) for g in TARGET_GENES},
        "combo_carriers": {k: len(v) for k, v in combo_carriers.items()},
    }

    output_path = RESULTS_DIR / "genie_demographics.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
