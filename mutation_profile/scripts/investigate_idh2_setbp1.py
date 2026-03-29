#!/usr/bin/env python3
"""
Investigate IDH2+SETBP1 co-occurrence discrepancy between GENIE (OR=3.41, enriched)
and IPSS-M (OR=0.22, depleted).

Hypothesis: non-SKI-domain SETBP1 variants (passengers in solid tumors or pan-cancer
panels) inflate the GENIE co-occurrence, while IPSS-M only has MDS with pathogenic
SKI-domain hotspot mutations.

Approach:
  1. Stratify by IDH2 variant class (R140x, R172x, Other)
  2. Stratify by SETBP1 variant class (SKI-domain hotspot vs non-SKI)
  3. Stratify by disease type (AML, MDS, MDS/MPN)
  4. Compute Fisher's exact OR with 95% CI for each stratum
  5. List all co-occurring patients with exact variants
"""

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent.parent  # mrna-hematology-research
MAF_PATH = PROJECT / "mutation_profile/data/genie/raw/data_mutations_extended.txt"
CLINICAL_PATH = PROJECT / "mutation_profile/data/genie/raw/data_clinical_sample.txt"
RESULTS_DIR = PROJECT / "mutation_profile/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = RESULTS_DIR / "idh2_setbp1_investigation.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROTEIN_ALTERING = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Nonstop_Mutation", "Translation_Start_Site",
}

# Myeloid OncoTree codes -- exact matches and prefix matches
# We need to be careful: MS is myeloid sarcoma, but MSCC/MSCHW/MSTAD are not.
# SMN is not myeloid (esophagogastric). SMZL is B-cell.
MYELOID_EXACT = {
    "AML", "AMML", "APL", "APLPMLRARA", "APMF", "AUL", "MLADS", "MS",
    "MDS", "MDS/MPN", "MPN", "CMML", "JMML", "CML", "TMN", "SM",
}

MYELOID_PREFIXES = (
    "AML",    # AML, AMLNOS, AMLNPM1, AMLMRC, etc.
    "MDS",    # MDS, MDSEB, MDSEB1, MDSEB2, etc.
    "MPN",    # MPN, MPNST, MPNU
    "CMML",   # CMML, CMML0, CMML1, CMML2
    "JMML",
    "CML",    # CML, CMLBCRABL1
    "TMN",
    "SM",     # SM, SMAHN, SMMCL -- but NOT SMN, SMZL
    "CEL",    # CELI, CELNOS
    "HES",
)

# Non-myeloid codes that start with a myeloid prefix
NON_MYELOID_EXCEPTIONS = {
    "MSCC", "MSCHW", "MSTAD",  # breast, nerve sheath, esophagogastric
    "SMN", "SMZL",             # esophagogastric, B-cell
}

# SKI-domain hotspot residues (aa 858-871)
SKI_HOTSPOT_RESIDUES = set(range(858, 872))  # 858 through 871 inclusive
SKI_HOTSPOT_VARIANTS = {
    "D868N", "D868Y", "D868G",
    "G870S", "G870D", "G870C", "G870R",
    "I871T", "I871N", "I871S",
    "S869R", "S869N", "S869G",
    "E858K",
    "T864M",
}

# Disease groupings for stratification
AML_PREFIXES = ("AML",)
AML_EXACT_EXTRA = {"AMML", "APL", "APLPMLRARA", "APMF", "AUL", "MLADS", "MS"}
MDS_PREFIXES = ("MDS",)
MDS_EXACT_EXTRA = set()
MDSMPM_CODES = {"MDS/MPN", "CMML", "CMML0", "CMML1", "CMML2", "JMML",
                "MDSMPNRST", "MDSMPNU"}


def is_myeloid(code: str) -> bool:
    """Check if an OncoTree code is a myeloid neoplasm."""
    if not code:
        return False
    code = code.strip()
    if code in NON_MYELOID_EXCEPTIONS:
        return False
    if code in MYELOID_EXACT:
        return True
    for prefix in MYELOID_PREFIXES:
        if code.startswith(prefix):
            return True
    return False


def disease_group(code: str) -> str:
    """Classify myeloid OncoTree code into AML, MDS, MDS/MPN, or Other Myeloid."""
    if not code:
        return "Unknown"
    code = code.strip()
    # AML group
    if code in AML_EXACT_EXTRA:
        return "AML"
    for p in AML_PREFIXES:
        if code.startswith(p):
            return "AML"
    # MDS/MPN group (check before MDS since MDS/MPN starts with MDS)
    if code in MDSMPM_CODES:
        return "MDS/MPN"
    # MDS group
    for p in MDS_PREFIXES:
        if code.startswith(p):
            return "MDS"
    # Everything else
    return "Other Myeloid"


def classify_idh2(hgvsp_short: str) -> str:
    """Classify IDH2 variant: R140x, R172x, or Other."""
    if not hgvsp_short:
        return "Other"
    # Strip p. prefix
    v = hgvsp_short.lstrip("p.")
    if v.startswith("R140"):
        return "R140x"
    if v.startswith("R172"):
        return "R172x"
    return "Other"


def is_r140q(hgvsp_short: str) -> bool:
    """Check if variant is specifically R140Q."""
    if not hgvsp_short:
        return False
    v = hgvsp_short.lstrip("p.")
    return v == "R140Q"


def classify_setbp1(hgvsp_short: str) -> str:
    """Classify SETBP1 variant: SKI-domain or Non-SKI."""
    if not hgvsp_short:
        return "Non-SKI"
    v = hgvsp_short.lstrip("p.")
    # Check against known SKI hotspot variants
    for hotspot in SKI_HOTSPOT_VARIANTS:
        if v.startswith(hotspot[:1]) and v[1:].startswith(hotspot[1:4]):
            # e.g., D868N -- check the residue number
            pass
    # More robust: extract residue number
    m = re.match(r'[A-Z](\d+)', v)
    if m:
        pos = int(m.group(1))
        if pos in SKI_HOTSPOT_RESIDUES:
            return "SKI-domain"
    return "Non-SKI"


def fisher_analysis(a: int, b: int, c: int, d: int) -> dict:
    """
    Run Fisher's exact test on a 2x2 table:
        IDH2+ SETBP1+: a     IDH2+ SETBP1-: b
        IDH2- SETBP1+: c     IDH2- SETBP1-: d

    Returns OR, 95% CI (Woolf's method with 0.5 correction), p-value, direction.
    """
    table = np.array([[a, b], [c, d]])
    odds_ratio, p_value = fisher_exact(table, alternative='two-sided')

    # Woolf's method with 0.5 Haldane correction
    a5, b5, c5, d5 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = math.log(a5 * d5 / (b5 * c5))
    se = math.sqrt(1/a5 + 1/b5 + 1/c5 + 1/d5)
    ci_lower = math.exp(log_or - 1.96 * se)
    ci_upper = math.exp(log_or + 1.96 * se)

    if p_value >= 0.05:
        direction = "neutral (not significant)"
    elif odds_ratio > 1:
        direction = "enriched (co-occurring)"
    else:
        direction = "depleted (mutually exclusive)"

    return {
        "contingency_table": {
            "IDH2+_SETBP1+": a,
            "IDH2+_SETBP1-": b,
            "IDH2-_SETBP1+": c,
            "IDH2-_SETBP1-": d,
        },
        "odds_ratio": round(odds_ratio, 4) if odds_ratio != float('inf') else "Inf",
        "odds_ratio_woolf": round(math.exp(log_or), 4),
        "ci_95_lower": round(ci_lower, 4),
        "ci_95_upper": round(ci_upper, 4),
        "p_value": p_value,
        "p_value_str": f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
        "direction": direction,
        "total_patients": a + b + c + d,
    }


def main():
    # ------------------------------------------------------------------
    # Step 1: Load clinical sample data (sample -> OncoTree code, center)
    # ------------------------------------------------------------------
    print("Loading clinical sample data...")
    sample_to_oncotree = {}
    sample_to_patient = {}
    sample_to_center = {}
    patient_to_samples = defaultdict(set)

    with open(CLINICAL_PATH, 'r') as f:
        # Skip comment lines starting with #
        for line in f:
            if not line.startswith('#'):
                header = line.strip().split('\t')
                break
        reader = csv.DictReader(f, fieldnames=header, delimiter='\t')
        for row in reader:
            sid = row.get('SAMPLE_ID', '')
            pid = row.get('PATIENT_ID', '')
            code = row.get('ONCOTREE_CODE', '')
            sample_to_oncotree[sid] = code
            sample_to_patient[sid] = pid
            patient_to_samples[pid].add(sid)
            # Extract center from sample ID: GENIE-CENTER-...
            parts = sid.split('-')
            center = parts[1] if len(parts) >= 3 else 'Unknown'
            sample_to_center[sid] = center

    print(f"  Loaded {len(sample_to_oncotree)} samples from {len(patient_to_samples)} patients")

    # Identify myeloid samples
    myeloid_samples = {sid for sid, code in sample_to_oncotree.items() if is_myeloid(code)}
    myeloid_patients = {sample_to_patient[sid] for sid in myeloid_samples if sid in sample_to_patient}
    print(f"  Myeloid samples: {len(myeloid_samples)}, Myeloid patients: {len(myeloid_patients)}")

    # Map patient to disease group (use the most specific myeloid sample)
    patient_disease = {}
    for pid in myeloid_patients:
        codes = {sample_to_oncotree.get(sid, '') for sid in patient_to_samples[pid]}
        myeloid_codes = {c for c in codes if is_myeloid(c)}
        groups = {disease_group(c) for c in myeloid_codes}
        # If patient has multiple disease groups, pick the most specific
        if "AML" in groups:
            patient_disease[pid] = "AML"
        elif "MDS/MPN" in groups:
            patient_disease[pid] = "MDS/MPN"
        elif "MDS" in groups:
            patient_disease[pid] = "MDS"
        else:
            patient_disease[pid] = list(groups)[0] if groups else "Unknown"

    # Disease group counts
    disease_counts = defaultdict(int)
    for d in patient_disease.values():
        disease_counts[d] += 1
    print(f"  Disease distribution: {dict(sorted(disease_counts.items(), key=lambda x: -x[1]))}")

    # ------------------------------------------------------------------
    # Step 2: Parse MAF for IDH2 and SETBP1 mutations in myeloid samples
    # ------------------------------------------------------------------
    print("\nParsing MAF for IDH2 and SETBP1 mutations...")

    # Per-patient: store mutation info
    # patient -> {gene -> [list of {variant, class, sample, center}]}
    patient_mutations = defaultdict(lambda: defaultdict(list))

    idh2_count = 0
    setbp1_count = 0

    with open(MAF_PATH, 'r') as f:
        header = f.readline().strip().split('\t')
        col = {name: idx for idx, name in enumerate(header)}

        hugo_idx = col['Hugo_Symbol']
        vc_idx = col['Variant_Classification']
        barcode_idx = col['Tumor_Sample_Barcode']
        hgvsp_idx = col['HGVSp_Short']
        center_idx = col['Center']

        for line in f:
            fields = line.strip().split('\t')
            if len(fields) <= max(hugo_idx, vc_idx, barcode_idx, hgvsp_idx, center_idx):
                continue

            gene = fields[hugo_idx]
            if gene not in ('IDH2', 'SETBP1'):
                continue

            vc = fields[vc_idx]
            if vc not in PROTEIN_ALTERING:
                continue

            sample_id = fields[barcode_idx]
            if sample_id not in myeloid_samples:
                continue

            hgvsp = fields[hgvsp_idx] if hgvsp_idx < len(fields) else ''
            center = fields[center_idx] if center_idx < len(fields) else ''
            patient_id = sample_to_patient.get(sample_id, '')

            if not patient_id:
                continue

            mutation_info = {
                'variant': hgvsp,
                'sample': sample_id,
                'center': center,
                'oncotree': sample_to_oncotree.get(sample_id, ''),
            }

            if gene == 'IDH2':
                mutation_info['idh2_class'] = classify_idh2(hgvsp)
                mutation_info['is_r140q'] = is_r140q(hgvsp)
                patient_mutations[patient_id]['IDH2'].append(mutation_info)
                idh2_count += 1
            elif gene == 'SETBP1':
                mutation_info['setbp1_class'] = classify_setbp1(hgvsp)
                patient_mutations[patient_id]['SETBP1'].append(mutation_info)
                setbp1_count += 1

    print(f"  IDH2 protein-altering mutations in myeloid: {idh2_count}")
    print(f"  SETBP1 protein-altering mutations in myeloid: {setbp1_count}")

    # Build patient-level mutation status
    idh2_patients = {pid for pid in patient_mutations if 'IDH2' in patient_mutations[pid]}
    setbp1_patients = {pid for pid in patient_mutations if 'SETBP1' in patient_mutations[pid]}
    both_patients = idh2_patients & setbp1_patients

    print(f"\n  IDH2-mutated patients: {len(idh2_patients)}")
    print(f"  SETBP1-mutated patients: {len(setbp1_patients)}")
    print(f"  Both IDH2+SETBP1 patients: {len(both_patients)}")

    # Subcategories
    r140x_patients = {pid for pid in idh2_patients
                      if any(m['idh2_class'] == 'R140x' for m in patient_mutations[pid]['IDH2'])}
    r172x_patients = {pid for pid in idh2_patients
                      if any(m['idh2_class'] == 'R172x' for m in patient_mutations[pid]['IDH2'])}
    r140q_patients = {pid for pid in idh2_patients
                      if any(m['is_r140q'] for m in patient_mutations[pid]['IDH2'])}
    idh2_other_patients = idh2_patients - r140x_patients - r172x_patients

    ski_patients = {pid for pid in setbp1_patients
                    if any(m['setbp1_class'] == 'SKI-domain' for m in patient_mutations[pid]['SETBP1'])}
    nonski_patients = {pid for pid in setbp1_patients
                       if any(m['setbp1_class'] == 'Non-SKI' for m in patient_mutations[pid]['SETBP1'])}
    # Patients with ONLY non-SKI
    nonski_only_patients = nonski_patients - ski_patients

    print(f"\n  IDH2 R140x patients: {len(r140x_patients)}")
    print(f"  IDH2 R172x patients: {len(r172x_patients)}")
    print(f"  IDH2 R140Q patients: {len(r140q_patients)}")
    print(f"  IDH2 Other patients: {len(idh2_other_patients)}")
    print(f"  SETBP1 SKI-domain patients: {len(ski_patients)}")
    print(f"  SETBP1 Non-SKI patients: {len(nonski_patients)}")
    print(f"  SETBP1 Non-SKI only patients: {len(nonski_only_patients)}")

    # ------------------------------------------------------------------
    # Step 3: Compute Fisher's exact test for each stratum
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FISHER'S EXACT TEST RESULTS")
    print("=" * 80)

    results = {}

    def compute_stratum(name, idh2_set, setbp1_set, universe):
        """Compute Fisher's test for a given stratum."""
        a = len(idh2_set & setbp1_set & universe)
        b = len((idh2_set - setbp1_set) & universe)
        c = len((setbp1_set - idh2_set) & universe)
        d = len(universe - idh2_set - setbp1_set)
        result = fisher_analysis(a, b, c, d)
        results[name] = result

        print(f"\n--- {name} ---")
        print(f"  Universe: {len(universe)} patients")
        print(f"  2x2 table:")
        print(f"                  SETBP1+    SETBP1-")
        print(f"    IDH2+         {a:<10} {b}")
        print(f"    IDH2-         {c:<10} {d}")
        print(f"  OR (Fisher): {result['odds_ratio']}")
        print(f"  OR (Woolf):  {result['odds_ratio_woolf']}")
        print(f"  95% CI:      [{result['ci_95_lower']}, {result['ci_95_upper']}]")
        print(f"  p-value:     {result['p_value_str']}")
        print(f"  Direction:   {result['direction']}")
        return result

    # a) All coding IDH2 + All coding SETBP1
    compute_stratum(
        "a_all_coding",
        idh2_patients, setbp1_patients, myeloid_patients
    )

    # b) R140x-only IDH2 + SKI-domain-only SETBP1
    compute_stratum(
        "b_R140x_SKI",
        r140x_patients, ski_patients, myeloid_patients
    )

    # c) R140Q-only IDH2 + SKI-domain-only SETBP1
    compute_stratum(
        "c_R140Q_SKI",
        r140q_patients, ski_patients, myeloid_patients
    )

    # d) AML-only (all coding variants)
    aml_patients = {pid for pid, d in patient_disease.items() if d == "AML"}
    compute_stratum(
        "d_AML_only",
        idh2_patients, setbp1_patients, aml_patients
    )

    # e) MDS-only (all coding variants)
    mds_patients = {pid for pid, d in patient_disease.items() if d == "MDS"}
    compute_stratum(
        "e_MDS_only",
        idh2_patients, setbp1_patients, mds_patients
    )

    # f) MDS/MPN-only (all coding variants)
    mdsmpm_patients = {pid for pid, d in patient_disease.items() if d == "MDS/MPN"}
    compute_stratum(
        "f_MDSMPM_only",
        idh2_patients, setbp1_patients, mdsmpm_patients
    )

    # Additional: Non-SKI SETBP1 only vs all IDH2
    compute_stratum(
        "g_all_IDH2_nonSKI_SETBP1",
        idh2_patients, nonski_only_patients, myeloid_patients
    )

    # Additional: SKI-only SETBP1 vs all IDH2
    compute_stratum(
        "h_all_IDH2_SKI_SETBP1",
        idh2_patients, ski_patients, myeloid_patients
    )

    # Additional: R140x IDH2 + Non-SKI SETBP1
    compute_stratum(
        "i_R140x_nonSKI_SETBP1",
        r140x_patients, nonski_only_patients, myeloid_patients
    )

    # ------------------------------------------------------------------
    # Step 4: List co-occurring patients with details
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CO-OCCURRING PATIENTS (IDH2 + SETBP1)")
    print("=" * 80)

    cooccurring_details = []
    for pid in sorted(both_patients):
        idh2_vars = patient_mutations[pid]['IDH2']
        setbp1_vars = patient_mutations[pid]['SETBP1']
        disease = patient_disease.get(pid, 'Unknown')

        idh2_strs = [f"{m['variant']} ({m['idh2_class']})" for m in idh2_vars]
        setbp1_strs = [f"{m['variant']} ({m['setbp1_class']})" for m in setbp1_vars]
        centers = list({m['center'] for m in idh2_vars + setbp1_vars})
        oncotrees = list({m['oncotree'] for m in idh2_vars + setbp1_vars})

        detail = {
            'patient_id': pid,
            'disease_group': disease,
            'oncotree_codes': oncotrees,
            'centers': centers,
            'idh2_variants': [m['variant'] for m in idh2_vars],
            'idh2_classes': [m['idh2_class'] for m in idh2_vars],
            'setbp1_variants': [m['variant'] for m in setbp1_vars],
            'setbp1_classes': [m['setbp1_class'] for m in setbp1_vars],
        }
        cooccurring_details.append(detail)

        print(f"\n  Patient: {pid}")
        print(f"    Disease: {disease} ({', '.join(oncotrees)})")
        print(f"    Center: {', '.join(centers)}")
        print(f"    IDH2: {', '.join(idh2_strs)}")
        print(f"    SETBP1: {', '.join(setbp1_strs)}")

    # ------------------------------------------------------------------
    # Step 5: Check if Non-SKI SETBP1 variants drive enrichment
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS: ARE NON-SKI SETBP1 VARIANTS DRIVING THE ENRICHMENT?")
    print("=" * 80)

    both_ski = both_patients & ski_patients
    both_nonski_only = both_patients & nonski_only_patients
    both_has_nonski = both_patients & nonski_patients

    print(f"\n  Co-occurring patients total: {len(both_patients)}")
    print(f"  Co-occurring with SKI-domain SETBP1: {len(both_ski)}")
    print(f"  Co-occurring with Non-SKI-only SETBP1: {len(both_nonski_only)}")
    print(f"  Co-occurring with any Non-SKI SETBP1: {len(both_has_nonski)}")

    # Breakdown of SETBP1 variants in co-occurring patients
    setbp1_variant_counts = defaultdict(int)
    setbp1_class_in_cooccurrence = defaultdict(int)
    for pid in both_patients:
        for m in patient_mutations[pid]['SETBP1']:
            setbp1_variant_counts[m['variant']] += 1
            setbp1_class_in_cooccurrence[m['setbp1_class']] += 1

    print(f"\n  SETBP1 variant breakdown in co-occurring patients:")
    for v, count in sorted(setbp1_variant_counts.items(), key=lambda x: -x[1]):
        cls = classify_setbp1(v)
        print(f"    {v} ({cls}): {count}")

    print(f"\n  SETBP1 class summary in co-occurring patients:")
    for cls, count in sorted(setbp1_class_in_cooccurrence.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {count}")

    # IDH2 variant breakdown in co-occurring patients
    idh2_variant_counts = defaultdict(int)
    idh2_class_in_cooccurrence = defaultdict(int)
    for pid in both_patients:
        for m in patient_mutations[pid]['IDH2']:
            idh2_variant_counts[m['variant']] += 1
            idh2_class_in_cooccurrence[m['idh2_class']] += 1

    print(f"\n  IDH2 variant breakdown in co-occurring patients:")
    for v, count in sorted(idh2_variant_counts.items(), key=lambda x: -x[1]):
        cls = classify_idh2(v)
        print(f"    {v} ({cls}): {count}")

    # Disease breakdown of co-occurring patients
    disease_breakdown = defaultdict(int)
    for pid in both_patients:
        disease_breakdown[patient_disease.get(pid, 'Unknown')] += 1

    print(f"\n  Disease breakdown of co-occurring patients:")
    for d, count in sorted(disease_breakdown.items(), key=lambda x: -x[1]):
        print(f"    {d}: {count}")

    # ------------------------------------------------------------------
    # Step 6: Summary interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_or = results['a_all_coding']['odds_ratio']
    ski_or = results['h_all_IDH2_SKI_SETBP1']['odds_ratio']
    nonski_or = results['g_all_IDH2_nonSKI_SETBP1']['odds_ratio']
    r140q_ski_or = results['c_R140Q_SKI']['odds_ratio']

    print(f"\n  All coding IDH2 + All coding SETBP1:   OR = {all_or}")
    print(f"  All IDH2 + SKI-domain SETBP1 only:     OR = {ski_or}")
    print(f"  All IDH2 + Non-SKI SETBP1 only:        OR = {nonski_or}")
    print(f"  R140Q IDH2 + SKI-domain SETBP1:        OR = {r140q_ski_or}")

    nonski_fraction = len(both_nonski_only) / len(both_patients) * 100 if both_patients else 0
    print(f"\n  Non-SKI SETBP1 driving enrichment: "
          f"{len(both_nonski_only)}/{len(both_patients)} co-occurring patients "
          f"({nonski_fraction:.0f}%) have ONLY non-SKI SETBP1 variants")

    if isinstance(nonski_or, (int, float)) and isinstance(ski_or, (int, float)):
        if nonski_or > ski_or:
            print("  --> YES: Non-SKI SETBP1 variants show higher OR than SKI-domain variants")
            print("      This likely explains the GENIE vs IPSS-M discrepancy.")
            print("      IPSS-M restricts to pathogenic SKI-domain mutations; GENIE includes all coding.")
        else:
            print("  --> NO: SKI-domain SETBP1 variants show equal or higher OR")
            print("      The enrichment is not driven by passenger non-SKI variants.")

    # ------------------------------------------------------------------
    # Step 7: Save results to JSON
    # ------------------------------------------------------------------
    output = {
        "metadata": {
            "description": "IDH2+SETBP1 co-occurrence investigation in GENIE myeloid cohort",
            "maf_file": str(MAF_PATH),
            "clinical_file": str(CLINICAL_PATH),
            "myeloid_patients_total": len(myeloid_patients),
            "idh2_mutated_patients": len(idh2_patients),
            "setbp1_mutated_patients": len(setbp1_patients),
            "cooccurring_patients": len(both_patients),
        },
        "patient_counts": {
            "idh2_r140x": len(r140x_patients),
            "idh2_r172x": len(r172x_patients),
            "idh2_r140q": len(r140q_patients),
            "idh2_other": len(idh2_other_patients),
            "setbp1_ski_domain": len(ski_patients),
            "setbp1_non_ski": len(nonski_patients),
            "setbp1_non_ski_only": len(nonski_only_patients),
        },
        "disease_distribution": dict(disease_counts),
        "fisher_tests": results,
        "cooccurring_patients": cooccurring_details,
        "cooccurrence_variant_breakdown": {
            "setbp1_variants": dict(setbp1_variant_counts),
            "setbp1_class_summary": dict(setbp1_class_in_cooccurrence),
            "idh2_variants": dict(idh2_variant_counts),
            "idh2_class_summary": dict(idh2_class_in_cooccurrence),
            "disease_breakdown": dict(disease_breakdown),
        },
        "key_finding": {
            "nonski_fraction_of_cooccurrence": round(nonski_fraction, 1),
            "all_coding_OR": all_or,
            "ski_only_OR": ski_or,
            "nonski_only_OR": nonski_or,
            "r140q_ski_OR": r140q_ski_or,
        },
    }

    # Convert numpy/scipy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    output = convert_types(output)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
