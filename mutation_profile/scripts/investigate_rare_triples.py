#!/usr/bin/env python3
"""
Rare Triple Carrier Investigation -- GENIE v19.0

Investigates the rarest triple combinations among the four target genes
(DNMT3A, IDH2, PTPN11, SETBP1) in the GENIE myeloid dataset:

1. IDH2+PTPN11+SETBP1 (n=1) -- the critical bottleneck triple
2. DNMT3A+IDH2+SETBP1 (n=7)
3. DNMT3A+PTPN11+SETBP1 (n=6)

For each triple carrier: complete mutation profile, clinical data, VAF ordering,
hotspot status, and whether they also carry the fourth gene (making them a quad).
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
GENIE_RAW = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"

# ===================================================================
# Configuration
# ===================================================================

TARGET_GENES = {"DNMT3A", "IDH2", "PTPN11", "SETBP1"}

PATHOGENIC_VAR_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    "Translation_Start_Site",
}

# Comprehensive myeloid OncoTree codes -- matches analyze_genie_filtered.py
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
    # Additional from prefix-based matching
    "AMML", "AUL", "MLADS", "MS",
    # Broader MDS/MPN overlap codes
    "MDS/MPN",
}

# Keyword fallback for cancer_type / cancer_type_detailed matching
MYELOID_KEYWORDS = [
    "leukemia", "myeloid", "myelodysplast", "mds", "aml",
    "myeloproliferat", "myelomonocytic", "erythroleukemia",
]

# OncoTree code prefixes for myeloid neoplasms (secondary check)
MYELOID_CODE_PREFIXES = (
    "AML", "MDS", "MPN", "CMML", "JMML", "CML", "TMN", "SM", "CEL", "HES",
)

# Hotspot definitions
IDH2_HOTSPOTS = {"R140Q", "R140W", "R140L", "R172K", "R172W"}
PTPN11_HOTSPOTS = {"E76K", "E76Q", "E76G", "E76A", "D61Y", "D61V", "D61H",
                   "A72V", "A72T", "G503A"}
SETBP1_SKI_HOTSPOTS = {"D868N", "D868Y", "D868G", "G870S", "G870D", "G870C",
                        "G870R", "I871T", "S869R", "S869N", "S869G"}

# The three triples we want to investigate
TRIPLES_OF_INTEREST = [
    ("IDH2", "PTPN11", "SETBP1"),      # n=1, critical bottleneck
    ("DNMT3A", "IDH2", "SETBP1"),       # n=7
    ("DNMT3A", "PTPN11", "SETBP1"),     # n=6
]


# ===================================================================
# Utility functions
# ===================================================================

def is_myeloid_sample(sample_info: dict) -> bool:
    """Check if a sample represents a myeloid neoplasm.
    Uses the same comprehensive filtering as analyze_genie_filtered.py:
    1. Check OncoTree code against explicit set
    2. Check OncoTree code against prefix patterns
    3. Fall back to keyword matching in cancer_type fields
    """
    code = sample_info.get("ONCOTREE_CODE", "").strip().upper()
    if code and code in MYELOID_ONCOTREE_CODES:
        return True
    if code and any(code.startswith(prefix) for prefix in MYELOID_CODE_PREFIXES):
        return True
    # Keyword fallback on cancer type fields
    cancer_text = (
        sample_info.get("CANCER_TYPE", "") + " " +
        sample_info.get("CANCER_TYPE_DETAILED", "")
    ).lower()
    return any(kw in cancer_text for kw in MYELOID_KEYWORDS)


def extract_variant_short(hgvsp_short: str) -> str:
    """Extract short variant name from HGVSp_Short (e.g. 'p.E76Q' -> 'E76Q')."""
    if not hgvsp_short:
        return ""
    return hgvsp_short.replace("p.", "")


def compute_vaf(t_alt_count: str, t_depth: str, t_ref_count: str) -> float | None:
    """Compute VAF from available count fields."""
    try:
        alt = int(t_alt_count)
        if t_depth and t_depth.strip():
            depth = int(t_depth)
        elif t_ref_count and t_ref_count.strip():
            depth = int(t_ref_count) + alt
        else:
            return None
        if depth == 0:
            return None
        return round(alt / depth * 100, 1)
    except (ValueError, TypeError):
        return None


def classify_hotspot(gene: str, variant: str) -> str:
    """Classify a variant as hotspot or non-hotspot for target genes."""
    if gene == "IDH2":
        return "HOTSPOT" if variant in IDH2_HOTSPOTS else "non-hotspot"
    elif gene == "PTPN11":
        return "HOTSPOT" if variant in PTPN11_HOTSPOTS else "non-hotspot"
    elif gene == "SETBP1":
        return "HOTSPOT (SKI domain)" if variant in SETBP1_SKI_HOTSPOTS else "non-hotspot"
    elif gene == "DNMT3A":
        # R882 is the canonical hotspot
        if variant and variant.startswith("R882"):
            return "HOTSPOT (R882)"
        return "non-hotspot"
    return "N/A"


def infer_clonal_hierarchy(mutations: list[dict]) -> list[dict]:
    """Sort mutations by VAF descending to infer clonal hierarchy."""
    # Only include mutations with VAF data
    with_vaf = [m for m in mutations if m.get("VAF_pct") is not None]
    return sorted(with_vaf, key=lambda m: m["VAF_pct"], reverse=True)


# ===================================================================
# Data loading
# ===================================================================

def load_clinical_patients() -> dict:
    """Load patient-level clinical data. Handles comment lines starting with #."""
    patients = {}
    filepath = GENIE_RAW / "data_clinical_patient.txt"
    with open(filepath) as f:
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
    return patients


def load_clinical_samples() -> dict:
    """Load sample-level clinical data. Returns dict keyed by SAMPLE_ID."""
    samples = {}
    filepath = GENIE_RAW / "data_clinical_sample.txt"
    with open(filepath) as f:
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
    return samples


def get_myeloid_sample_ids(samples: dict) -> dict:
    """Return dict of sample_id -> sample_info for myeloid samples."""
    myeloid = {}
    for sid, info in samples.items():
        if is_myeloid_sample(info):
            myeloid[sid] = info
    return myeloid


def build_sample_to_patient(samples: dict) -> dict:
    """Build sample_id -> patient_id mapping from clinical sample data."""
    s2p = {}
    for sid, info in samples.items():
        pid = info.get("PATIENT_ID", "")
        if pid:
            s2p[sid] = pid
    return s2p


def build_patient_to_samples(samples: dict) -> dict:
    """Build patient_id -> set of sample_ids mapping."""
    p2s = defaultdict(set)
    for sid, info in samples.items():
        pid = info.get("PATIENT_ID", "")
        if pid:
            p2s[pid].add(sid)
    return p2s


# ===================================================================
# MAF scanning
# ===================================================================

def scan_maf_for_triple_carriers(myeloid_sids: set, sample_to_patient: dict) -> dict:
    """
    Two-pass scan of the MAF file.

    Pass 1: For each myeloid sample, record which target genes are mutated.
            Identify patients that carry each triple of interest.
    Pass 2: For identified triple carriers, collect ALL mutations.

    Returns dict: triple_key -> list of patient_ids, plus patient_mutations dict.
    """
    maf_path = GENIE_RAW / "data_mutations_extended.txt"

    # -------------------------------------------------------------------
    # Pass 1: Build patient -> set of mutated target genes
    # -------------------------------------------------------------------
    print("  Pass 1: Identifying target gene carriers in myeloid samples...")
    patient_target_genes = defaultdict(set)  # patient_id -> set of target genes mutated
    # Also track which specific mutations each patient has for target genes
    patient_target_mutations = defaultdict(list)  # patient_id -> list of {gene, variant, ...}

    row_count = 0
    with open(maf_path) as f:
        header = f.readline().strip().split("\t")
        col_idx = {name: i for i, name in enumerate(header)}

        hugo_idx = col_idx["Hugo_Symbol"]
        hgvsp_idx = col_idx.get("HGVSp_Short")
        varclass_idx = col_idx["Variant_Classification"]
        sample_idx = col_idx["Tumor_Sample_Barcode"]
        t_alt_idx = col_idx.get("t_alt_count")
        t_ref_idx = col_idx.get("t_ref_count")
        t_depth_idx = col_idx.get("t_depth")
        center_idx = col_idx.get("Center")

        for line in f:
            row_count += 1
            if row_count % 500_000 == 0:
                print(f"    ...{row_count:,} rows")

            fields = line.rstrip("\n").split("\t")
            sample_id = fields[sample_idx]

            if sample_id not in myeloid_sids:
                continue

            var_class = fields[varclass_idx]
            if var_class not in PATHOGENIC_VAR_CLASSES:
                continue

            gene = fields[hugo_idx]
            if gene not in TARGET_GENES:
                continue

            patient_id = sample_to_patient.get(sample_id, "")
            if not patient_id:
                continue

            hgvsp = fields[hgvsp_idx] if hgvsp_idx is not None else ""
            variant = extract_variant_short(hgvsp)

            t_alt = fields[t_alt_idx] if t_alt_idx is not None else ""
            t_ref = fields[t_ref_idx] if t_ref_idx is not None else ""
            t_depth = fields[t_depth_idx] if t_depth_idx is not None else ""
            center = fields[center_idx] if center_idx is not None else ""

            vaf = compute_vaf(t_alt, t_depth, t_ref)

            patient_target_genes[patient_id].add(gene)
            patient_target_mutations[patient_id].append({
                "gene": gene,
                "variant": variant,
                "HGVSp_Short": hgvsp,
                "Variant_Classification": var_class,
                "VAF_pct": vaf,
                "t_alt_count": t_alt,
                "t_ref_count": t_ref,
                "t_depth": t_depth,
                "sample_id": sample_id,
                "center": center,
                "hotspot_status": classify_hotspot(gene, variant),
            })

    print(f"  Pass 1 complete: {row_count:,} rows scanned")

    # Identify triple carriers for each triple of interest
    triple_carriers = {}
    all_triple_patient_ids = set()

    for triple in TRIPLES_OF_INTEREST:
        key = "+".join(triple)
        gene_set = set(triple)
        carriers = []
        for pid, genes in patient_target_genes.items():
            if gene_set.issubset(genes):
                carriers.append(pid)
        triple_carriers[key] = sorted(carriers)
        all_triple_patient_ids.update(carriers)
        print(f"    {key}: {len(carriers)} carrier(s)")

    # Also find quadruple carriers
    quad_carriers = []
    for pid, genes in patient_target_genes.items():
        if TARGET_GENES.issubset(genes):
            quad_carriers.append(pid)
    print(f"    QUADRUPLE (all 4): {len(quad_carriers)} carrier(s)")

    # -------------------------------------------------------------------
    # Pass 2: Collect ALL mutations for triple carriers
    # -------------------------------------------------------------------
    print(f"\n  Pass 2: Collecting all mutations for {len(all_triple_patient_ids)} triple carrier(s)...")

    # Build set of all sample IDs belonging to triple carrier patients
    # (from the myeloid set)
    triple_carrier_sample_ids = set()
    for sid, pid in sample_to_patient.items():
        if pid in all_triple_patient_ids and sid in myeloid_sids:
            triple_carrier_sample_ids.add(sid)

    patient_all_mutations = defaultdict(list)

    row_count2 = 0
    with open(maf_path) as f:
        header = f.readline().strip().split("\t")
        col_idx = {name: i for i, name in enumerate(header)}

        hugo_idx = col_idx["Hugo_Symbol"]
        hgvsp_idx = col_idx.get("HGVSp_Short")
        varclass_idx = col_idx["Variant_Classification"]
        sample_idx = col_idx["Tumor_Sample_Barcode"]
        t_alt_idx = col_idx.get("t_alt_count")
        t_ref_idx = col_idx.get("t_ref_count")
        t_depth_idx = col_idx.get("t_depth")
        center_idx = col_idx.get("Center")
        chrom_idx = col_idx.get("Chromosome")
        start_idx = col_idx.get("Start_Position")
        ref_idx = col_idx.get("Reference_Allele")
        alt_idx = col_idx.get("Tumor_Seq_Allele2")

        for line in f:
            row_count2 += 1
            if row_count2 % 500_000 == 0:
                print(f"    ...{row_count2:,} rows")

            fields = line.rstrip("\n").split("\t")
            sample_id = fields[sample_idx]

            if sample_id not in triple_carrier_sample_ids:
                continue

            var_class = fields[varclass_idx]
            if var_class not in PATHOGENIC_VAR_CLASSES:
                continue

            patient_id = sample_to_patient.get(sample_id, "")
            if not patient_id or patient_id not in all_triple_patient_ids:
                continue

            gene = fields[hugo_idx]
            hgvsp = fields[hgvsp_idx] if hgvsp_idx is not None else ""
            variant = extract_variant_short(hgvsp)
            t_alt = fields[t_alt_idx] if t_alt_idx is not None else ""
            t_ref = fields[t_ref_idx] if t_ref_idx is not None else ""
            t_depth = fields[t_depth_idx] if t_depth_idx is not None else ""
            center = fields[center_idx] if center_idx is not None else ""
            chrom = fields[chrom_idx] if chrom_idx is not None else ""
            start = fields[start_idx] if start_idx is not None else ""
            ref_allele = fields[ref_idx] if ref_idx is not None else ""
            alt_allele = fields[alt_idx] if alt_idx is not None else ""

            vaf = compute_vaf(t_alt, t_depth, t_ref)

            hotspot = classify_hotspot(gene, variant) if gene in TARGET_GENES else "N/A"

            patient_all_mutations[patient_id].append({
                "gene": gene,
                "variant": variant,
                "HGVSp_Short": hgvsp,
                "Variant_Classification": var_class,
                "VAF_pct": vaf,
                "t_alt_count": t_alt,
                "t_ref_count": t_ref,
                "t_depth": t_depth,
                "sample_id": sample_id,
                "center": center,
                "chromosome": chrom,
                "start_position": start,
                "ref_allele": ref_allele,
                "alt_allele": alt_allele,
                "hotspot_status": hotspot,
            })

    print(f"  Pass 2 complete: {row_count2:,} rows scanned")

    return triple_carriers, quad_carriers, patient_target_mutations, patient_all_mutations


# ===================================================================
# Profile building
# ===================================================================

def build_patient_profile(
    patient_id: str,
    mutations: list[dict],
    target_mutations: list[dict],
    samples_clinical: dict,
    patients_clinical: dict,
    sample_to_patient: dict,
    myeloid_samples: dict,
    triple_genes: set,
) -> dict:
    """Build a complete profile for a triple carrier patient."""

    # Find their myeloid sample(s)
    patient_sample_ids = []
    for sid, pid in sample_to_patient.items():
        if pid == patient_id and sid in myeloid_samples:
            patient_sample_ids.append(sid)

    # Use first myeloid sample for clinical info
    primary_sample_id = patient_sample_ids[0] if patient_sample_ids else ""
    sample_info = samples_clinical.get(primary_sample_id, {})
    patient_info = patients_clinical.get(patient_id, {})

    oncotree_code = sample_info.get("ONCOTREE_CODE", "Unknown")
    cancer_type = sample_info.get("CANCER_TYPE_DETAILED", sample_info.get("CANCER_TYPE", "Unknown"))
    age = sample_info.get("AGE_AT_SEQ_REPORT", "Unknown")
    panel = sample_info.get("SEQ_ASSAY_ID", "Unknown")
    center_from_sample = sample_info.get("CENTER", "")
    # Fall back to patient_info or parse from ID
    center = center_from_sample or patient_info.get("CENTER", "")
    if not center and len(patient_id.split("-")) > 1:
        center = patient_id.split("-")[1]
    sample_type = sample_info.get("SAMPLE_TYPE", "Unknown")
    sample_type_detailed = sample_info.get("SAMPLE_TYPE_DETAILED", "Unknown")
    sex = patient_info.get("SEX", "Unknown")
    race = patient_info.get("PRIMARY_RACE", "Unknown")
    ethnicity = patient_info.get("ETHNICITY", "Unknown")
    dead_raw = patient_info.get("DEAD", "Unknown")
    vital_status = "Deceased" if dead_raw == "TRUE" else "Alive" if dead_raw == "FALSE" else dead_raw

    # Sort mutations by VAF descending
    mutations_sorted = sorted(mutations, key=lambda m: (m["VAF_pct"] or 0), reverse=True)

    # Deduplicate mutations: same gene+variant+sample = keep highest VAF entry
    # (patients with multiple samples have repeated entries for the same mutation)
    seen_mutations = {}
    for m in mutations_sorted:
        dedup_key = (m["gene"], m["variant"], m["sample_id"])
        if dedup_key not in seen_mutations:
            seen_mutations[dedup_key] = m
    # For display, also deduplicate across samples (same gene+variant, keep highest VAF)
    unique_mutations = {}
    for m in mutations_sorted:
        gene_var_key = (m["gene"], m["variant"])
        if gene_var_key not in unique_mutations:
            unique_mutations[gene_var_key] = m

    # Check which target genes are present
    patient_genes = {m["gene"] for m in mutations}
    has_genes = {g: g in patient_genes for g in TARGET_GENES}

    # Fourth gene check (the one NOT in this triple)
    fourth_gene = TARGET_GENES - triple_genes
    fourth_gene_name = fourth_gene.pop() if fourth_gene else None
    has_fourth = has_genes.get(fourth_gene_name, False) if fourth_gene_name else False

    # VAF-based clonal hierarchy for target genes (deduplicated)
    target_gene_vafs = []
    seen_target = set()
    for m in mutations_sorted:
        if m["gene"] in TARGET_GENES and m.get("VAF_pct") is not None:
            key = (m["gene"], m["variant"])
            if key not in seen_target:
                seen_target.add(key)
                target_gene_vafs.append({
                    "gene": m["gene"],
                    "variant": m["variant"],
                    "VAF_pct": m["VAF_pct"],
                    "hotspot_status": m.get("hotspot_status", classify_hotspot(m["gene"], m["variant"])),
                })
    target_gene_vafs.sort(key=lambda x: x["VAF_pct"], reverse=True)

    # Full clonal hierarchy (all genes with VAF)
    full_hierarchy = infer_clonal_hierarchy(mutations)

    # Hypermutation check (typical myeloid: 5-15 mutations, >30 = hypermutated)
    total_mutations = len(unique_mutations)  # Use deduplicated count
    total_raw_rows = len(mutations)  # Raw rows including multi-sample duplicates
    hypermutation_status = "HYPERMUTATED" if total_mutations > 30 else (
        "HIGH" if total_mutations > 20 else "TYPICAL" if total_mutations >= 3 else "LOW"
    )

    profile = {
        "patient_id": patient_id,
        "sample_ids": patient_sample_ids,
        "clinical": {
            "sex": sex,
            "race": race,
            "ethnicity": ethnicity,
            "age_at_seq_report": age,
            "oncotree_code": oncotree_code,
            "cancer_type_detailed": cancer_type,
            "sample_type": sample_type,
            "sample_type_detailed": sample_type_detailed,
            "center": center,
            "seq_assay_id": panel,
            "vital_status": vital_status,
        },
        "mutation_summary": {
            "total_protein_altering_mutations": total_mutations,
            "hypermutation_status": hypermutation_status,
            "target_genes_present": has_genes,
            "fourth_gene": fourth_gene_name,
            "has_fourth_gene_making_quadruple": has_fourth,
        },
        "target_gene_details": [],
        "target_gene_vaf_hierarchy": target_gene_vafs,
        "all_mutations_by_vaf": [
            {
                "gene": m["gene"],
                "variant": m["variant"],
                "Variant_Classification": m["Variant_Classification"],
                "VAF_pct": m["VAF_pct"],
                "hotspot_status": m.get("hotspot_status", "N/A"),
                "is_target_gene": m["gene"] in TARGET_GENES,
            }
            for m in sorted(unique_mutations.values(), key=lambda x: (x["VAF_pct"] or 0), reverse=True)
        ],
        "unique_mutation_count": len(unique_mutations),
        "total_raw_mutation_rows": len(mutations),
    }

    # Add target gene detail entries with hotspot status
    for m in target_mutations:
        if m.get("gene") in TARGET_GENES:
            profile["target_gene_details"].append({
                "gene": m["gene"],
                "variant": m["variant"],
                "HGVSp_Short": m.get("HGVSp_Short", ""),
                "Variant_Classification": m.get("Variant_Classification", ""),
                "VAF_pct": m.get("VAF_pct"),
                "hotspot_status": m.get("hotspot_status", classify_hotspot(m["gene"], m["variant"])),
            })

    return profile


# ===================================================================
# Printing functions
# ===================================================================

def print_patient_profile(profile: dict, index: int, total: int, triple_name: str):
    """Print a detailed patient profile to stdout."""
    pid = profile["patient_id"]
    clin = profile["clinical"]
    ms = profile["mutation_summary"]

    print(f"\n{'='*70}")
    print(f"  Patient {index}/{total}: {pid}")
    print(f"  Triple: {triple_name}")
    print(f"{'='*70}")
    print(f"  Sex:             {clin['sex']}")
    print(f"  Race:            {clin['race']}")
    print(f"  Ethnicity:       {clin['ethnicity']}")
    print(f"  Age at seq:      {clin['age_at_seq_report']}")
    print(f"  Diagnosis:       {clin['oncotree_code']} ({clin['cancer_type_detailed']})")
    print(f"  Sample type:     {clin['sample_type']} / {clin['sample_type_detailed']}")
    print(f"  Center:          {clin['center']}")
    print(f"  Panel:           {clin['seq_assay_id']}")
    print(f"  Vital status:    {clin['vital_status']}")
    unique_n = profile.get("unique_mutation_count", ms['total_protein_altering_mutations'])
    raw_n = profile.get("total_raw_mutation_rows", unique_n)
    samples_note = f" ({raw_n} raw rows across {len(profile.get('sample_ids',[]))} sample(s))" if raw_n != unique_n else ""
    print(f"  Unique mutations: {unique_n} ({ms['hypermutation_status']}){samples_note}")

    # Fourth gene check
    fourth = ms["fourth_gene"]
    if ms["has_fourth_gene_making_quadruple"]:
        print(f"  ** ALSO CARRIES {fourth} -- THIS IS A QUADRUPLE CARRIER **")
    else:
        print(f"  Missing {fourth} for quadruple: NOT PRESENT")

    # Target gene details with hotspot status (deduplicated)
    seen = set()
    print(f"\n  Target gene variants:")
    print(f"  {'Gene':<12} {'Variant':<18} {'Classification':<25} {'VAF%':<8} {'Hotspot'}")
    print(f"  {'-'*12} {'-'*18} {'-'*25} {'-'*8} {'-'*20}")
    for td in profile["target_gene_details"]:
        dedup_key = (td['gene'], td['variant'])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        vaf_str = f"{td['VAF_pct']}%" if td['VAF_pct'] is not None else "N/A"
        print(f"  {td['gene']:<12} {td['variant']:<18} {td['Variant_Classification']:<25} {vaf_str:<8} {td['hotspot_status']}")

    # VAF hierarchy for target genes
    if profile["target_gene_vaf_hierarchy"]:
        print(f"\n  Clonal hierarchy (target genes by VAF):")
        for rank, entry in enumerate(profile["target_gene_vaf_hierarchy"], 1):
            role = "FOUNDER" if rank == 1 else f"SUBCLONE-{rank}"
            print(f"    {rank}. {entry['gene']} {entry['variant']}: {entry['VAF_pct']}% [{role}]")

    # Full mutation profile
    print(f"\n  Complete mutation profile ({unique_n} unique mutations):")
    print(f"  {'#':<4} {'Gene':<15} {'Variant':<20} {'Classification':<25} {'VAF%':<8} {'Note'}")
    print(f"  {'-'*4} {'-'*15} {'-'*20} {'-'*25} {'-'*8} {'-'*20}")
    for i, m in enumerate(profile["all_mutations_by_vaf"], 1):
        vaf_str = f"{m['VAF_pct']}%" if m['VAF_pct'] is not None else "N/A"
        note = ""
        if m["is_target_gene"]:
            note = f"<-- TARGET ({m['hotspot_status']})"
        print(f"  {i:<4} {m['gene']:<15} {m['variant']:<20} {m['Variant_Classification']:<25} {vaf_str:<8} {note}")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 80)
    print("RARE TRIPLE CARRIER INVESTIGATION -- GENIE v19.0")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    print("Target triples:")
    for triple in TRIPLES_OF_INTEREST:
        print(f"  - {'+'.join(triple)}")
    print()

    # ------------------------------------------------------------------
    # Load clinical data
    # ------------------------------------------------------------------
    print("Loading clinical data...")
    patients_clinical = load_clinical_patients()
    samples_clinical = load_clinical_samples()
    print(f"  {len(patients_clinical):,} patients, {len(samples_clinical):,} samples")

    myeloid_samples = get_myeloid_sample_ids(samples_clinical)
    myeloid_sids = set(myeloid_samples.keys())
    print(f"  Myeloid samples: {len(myeloid_sids):,}")

    sample_to_patient = build_sample_to_patient(samples_clinical)
    patient_to_samples = build_patient_to_samples(samples_clinical)

    # ------------------------------------------------------------------
    # Scan MAF file
    # ------------------------------------------------------------------
    print("\nScanning MAF file (3.4M rows, two passes)...")
    triple_carriers, quad_carriers, patient_target_muts, patient_all_muts = \
        scan_maf_for_triple_carriers(myeloid_sids, sample_to_patient)

    # ------------------------------------------------------------------
    # Build results
    # ------------------------------------------------------------------
    results = {
        "analysis": "Rare Triple Carrier Investigation",
        "dataset": "GENIE v19.0",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filter": "Myeloid OncoTree codes, protein-altering variants only",
        "myeloid_samples": len(myeloid_sids),
        "quadruple_carriers": {
            "count": len(quad_carriers),
            "patient_ids": sorted(quad_carriers),
        },
        "triples": {},
    }

    # ------------------------------------------------------------------
    # Process each triple
    # ------------------------------------------------------------------
    for triple in TRIPLES_OF_INTEREST:
        key = "+".join(triple)
        triple_genes = set(triple)
        carrier_pids = triple_carriers[key]

        print(f"\n{'#'*80}")
        print(f"# TRIPLE: {key} (n={len(carrier_pids)})")
        print(f"{'#'*80}")

        triple_result = {
            "triple": key,
            "genes": list(triple),
            "carrier_count": len(carrier_pids),
            "patients": [],
        }

        for i, pid in enumerate(carrier_pids, 1):
            mutations = patient_all_muts.get(pid, [])
            target_muts = patient_target_muts.get(pid, [])

            profile = build_patient_profile(
                patient_id=pid,
                mutations=mutations,
                target_mutations=target_muts,
                samples_clinical=samples_clinical,
                patients_clinical=patients_clinical,
                sample_to_patient=sample_to_patient,
                myeloid_samples=myeloid_samples,
                triple_genes=triple_genes,
            )

            print_patient_profile(profile, i, len(carrier_pids), key)

            # Strip full mutation details for JSON (keep summary)
            json_profile = {
                "patient_id": profile["patient_id"],
                "sample_ids": profile["sample_ids"],
                "clinical": profile["clinical"],
                "mutation_summary": profile["mutation_summary"],
                "target_gene_details": profile["target_gene_details"],
                "target_gene_vaf_hierarchy": profile["target_gene_vaf_hierarchy"],
                "all_mutations_by_vaf": profile["all_mutations_by_vaf"],
            }
            triple_result["patients"].append(json_profile)

        # Summary statistics for this triple
        ages = []
        sexes = Counter()
        diagnoses = Counter()
        centers = Counter()
        mutation_counts = []
        has_fourth_count = 0
        gene_freq = Counter()

        for p in triple_result["patients"]:
            age_str = p["clinical"]["age_at_seq_report"]
            try:
                ages.append(int(age_str))
            except (ValueError, TypeError):
                pass
            sexes[p["clinical"]["sex"]] += 1
            diagnoses[p["clinical"]["oncotree_code"]] += 1
            centers[p["clinical"]["center"]] += 1
            mutation_counts.append(p.get("unique_mutation_count", p["mutation_summary"]["total_protein_altering_mutations"]))
            if p["mutation_summary"]["has_fourth_gene_making_quadruple"]:
                has_fourth_count += 1
            for m in p["all_mutations_by_vaf"]:
                gene_freq[m["gene"]] += 1

        triple_result["summary"] = {
            "ages": sorted(ages),
            "age_median": sorted(ages)[len(ages) // 2] if ages else None,
            "age_range": [min(ages), max(ages)] if ages else None,
            "sex_distribution": dict(sexes),
            "diagnosis_distribution": dict(diagnoses.most_common()),
            "center_distribution": dict(centers.most_common()),
            "mutation_counts": mutation_counts,
            "avg_mutations": round(sum(mutation_counts) / len(mutation_counts), 1) if mutation_counts else None,
            "patients_with_fourth_gene": has_fourth_count,
            "fourth_gene": list(TARGET_GENES - triple_genes)[0] if len(TARGET_GENES - triple_genes) == 1 else None,
            "top_co_mutated_genes": dict(gene_freq.most_common(20)),
        }

        results["triples"][key] = triple_result

        # Print summary
        print(f"\n  --- {key} Summary ---")
        print(f"  Carriers: {len(carrier_pids)}")
        print(f"  Ages: {sorted(ages)} (median: {triple_result['summary']['age_median']})")
        print(f"  Sex: {dict(sexes)}")
        print(f"  Diagnoses: {dict(diagnoses.most_common())}")
        print(f"  Centers: {dict(centers.most_common())}")
        print(f"  Avg mutations/patient: {triple_result['summary']['avg_mutations']}")
        fourth = triple_result['summary']['fourth_gene']
        print(f"  Also carry {fourth} (quadruple): {has_fourth_count}/{len(carrier_pids)}")
        print(f"  Top co-mutated genes: {dict(gene_freq.most_common(10))}")

    # ------------------------------------------------------------------
    # Cross-triple analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("CROSS-TRIPLE ANALYSIS")
    print(f"{'='*80}")

    # Check for patients appearing in multiple triples
    all_carrier_sets = {}
    for triple in TRIPLES_OF_INTEREST:
        key = "+".join(triple)
        all_carrier_sets[key] = set(triple_carriers[key])

    for i, t1 in enumerate(TRIPLES_OF_INTEREST):
        for t2 in TRIPLES_OF_INTEREST[i+1:]:
            k1 = "+".join(t1)
            k2 = "+".join(t2)
            overlap = all_carrier_sets[k1] & all_carrier_sets[k2]
            if overlap:
                print(f"\n  Patients in BOTH {k1} AND {k2}: {len(overlap)}")
                for pid in sorted(overlap):
                    genes = patient_target_muts.get(pid, [])
                    gene_list = sorted({m["gene"] for m in genes if m["gene"] in TARGET_GENES})
                    print(f"    {pid}: carries {', '.join(gene_list)}")
                    # These are potentially quadruple carriers
                    if TARGET_GENES.issubset(set(gene_list)):
                        print(f"      ** CONFIRMED QUADRUPLE CARRIER **")

    results["cross_triple_overlap"] = {}
    for i, t1 in enumerate(TRIPLES_OF_INTEREST):
        for t2 in TRIPLES_OF_INTEREST[i+1:]:
            k1 = "+".join(t1)
            k2 = "+".join(t2)
            overlap = all_carrier_sets[k1] & all_carrier_sets[k2]
            if overlap:
                results["cross_triple_overlap"][f"{k1} & {k2}"] = {
                    "count": len(overlap),
                    "patient_ids": sorted(overlap),
                }

    # Quadruple carrier summary
    print(f"\n  QUADRUPLE CARRIERS (DNMT3A+IDH2+PTPN11+SETBP1): {len(quad_carriers)}")
    if quad_carriers:
        for pid in sorted(quad_carriers):
            muts = patient_target_muts.get(pid, [])
            print(f"    {pid}:")
            for m in sorted(muts, key=lambda x: (x.get("VAF_pct") or 0), reverse=True):
                vaf_str = f"{m['VAF_pct']}%" if m.get('VAF_pct') is not None else "N/A"
                hs = m.get("hotspot_status", "")
                print(f"      {m['gene']} {m['variant']}: VAF={vaf_str} [{hs}]")
    else:
        print("    NONE -- confirming 0 quadruple carriers in GENIE v19.0")

    # ------------------------------------------------------------------
    # Henrik comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("COMPARISON WITH HENRIK'S PROFILE")
    print(f"{'='*80}")
    print("  Henrik's mutations: DNMT3A R882H (39%), SETBP1 G870S (34%), PTPN11 E76Q (29%), IDH2 R140Q (2%)")
    print("  All four are canonical hotspots.")
    print()

    # Check if any triple carrier has the same specific variants
    for triple in TRIPLES_OF_INTEREST:
        key = "+".join(triple)
        carrier_pids = triple_carriers[key]
        print(f"  {key} carriers ({len(carrier_pids)}):")
        for pid in carrier_pids:
            target_muts = patient_target_muts.get(pid, [])
            # Deduplicate target gene variants for display
            seen_variants = set()
            unique_target_muts = []
            for m in target_muts:
                vkey = (m["gene"], m["variant"])
                if vkey not in seen_variants:
                    seen_variants.add(vkey)
                    unique_target_muts.append(m)
            matches = set()
            for m in unique_target_muts:
                if m["gene"] == "DNMT3A" and m["variant"] == "R882H":
                    matches.add(f"{m['gene']} {m['variant']}")
                elif m["gene"] == "IDH2" and m["variant"] == "R140Q":
                    matches.add(f"{m['gene']} {m['variant']}")
                elif m["gene"] == "PTPN11" and m["variant"] == "E76Q":
                    matches.add(f"{m['gene']} {m['variant']}")
                elif m["gene"] == "SETBP1" and m["variant"] == "G870S":
                    matches.add(f"{m['gene']} {m['variant']}")
            match_str = f" -- SAME AS HENRIK: {', '.join(sorted(matches))}" if matches else ""
            variant_list = [f"{m['gene']} {m['variant']}" for m in unique_target_muts]
            print(f"    {pid}: {', '.join(variant_list)}{match_str}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "rare_triples_investigation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
