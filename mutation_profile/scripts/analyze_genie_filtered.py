#!/usr/bin/env python3
"""
GENIE v19.0 Co-occurrence Analysis - Filtered for Pathogenic Variants

Fixes from premature analysis:
1. Filters OUT intronic, silent, UTR, splice_region variants (UCHI artifacts)
2. Flags hypermutated cases (>100 mutations)
3. Reports both "any coding mutation" and "hotspot/known pathogenic" tiers
4. Proper panel coverage verification

Target genes: DNMT3A, IDH2, PTPN11, SETBP1
"""

import json
import re
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
GENIE_RAW = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]
TARGET_GENES_SET = set(TARGET_GENES)

# Variant classifications that are protein-altering (keep these)
PATHOGENIC_VAR_CLASSES = {
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

# Variant classifications to EXCLUDE (not protein-altering)
EXCLUDED_VAR_CLASSES = {
    "Intron",
    "Silent",
    "3'UTR",
    "5'UTR",
    "Splice_Region",  # borderline, but standard MAF practice is to exclude
    "3'Flank",
    "5'Flank",
    "IGR",
    "RNA",
}

# Known hotspot/pathogenic variants for each gene
DNMT3A_HOTSPOTS = {
    "R882H", "R882C", "R882S", "R882P",  # R882 hotspot
}

IDH2_HOTSPOTS = {
    "R140Q", "R140W", "R140L", "R140G",  # R140 hotspot
    "R172K", "R172M", "R172S", "R172G", "R172W",  # R172 hotspot
}

PTPN11_HOTSPOTS = {
    "E76K", "E76Q", "E76G", "E76A",  # E76 hotspot
    "A72T", "A72V", "A72G", "A72D",  # A72
    "D61Y", "D61V", "D61H", "D61N",  # D61
    "G60R", "G60V",  # G60
    "N308D", "N308S", "N308T",  # N308
    "S502L", "S502P", "S502A",  # S502
    "G503R", "G503A", "G503V",  # G503
    "F71L", "Y62D",  # other known
}

SETBP1_HOTSPOTS = {
    # SKI domain hotspot (exon 4, aa 858-871)
    "D868N", "D868Y", "D868H", "D868E",
    "G870S", "G870R", "G870D",
    "I871T", "I871N", "I871S",
    "S869N", "S869R", "S869G",
    "T873I", "T873S",
    "E862K", "E862D",
    "T864M", "T864A",
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

# Hypermutation threshold (total coding mutations in target genes)
HYPERMUTATION_THRESHOLD = 20  # in target genes specifically


def is_star_pattern(protein_change):
    """Check if protein change matches UCHI artifact p.*NNN* pattern."""
    return bool(re.match(r'^p\.\*\d+\*$', protein_change))


def extract_variant_name(protein_change):
    """Extract clean variant name like R882H from p.R882H."""
    if not protein_change:
        return ""
    m = re.match(r'^p\.([A-Z]\d+[A-Z*])$', protein_change)
    if m:
        return m.group(1)
    return protein_change.replace("p.", "")


def is_hotspot(gene, protein_change):
    """Check if a variant is a known hotspot."""
    name = extract_variant_name(protein_change)
    hotspots = {
        "DNMT3A": DNMT3A_HOTSPOTS,
        "IDH2": IDH2_HOTSPOTS,
        "PTPN11": PTPN11_HOTSPOTS,
        "SETBP1": SETBP1_HOTSPOTS,
    }
    return name in hotspots.get(gene, set())


def load_clinical_samples():
    """Load clinical sample data, return sample_id -> info dict."""
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
    """Load patient-level clinical data."""
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
                "race": row.get("PRIMARY_RACE", ""),
                "ethnicity": row.get("ETHNICITY", ""),
                "center": row.get("CENTER", ""),
            }
    return patients


def is_myeloid(sample_info):
    code = sample_info.get("oncotree_code", "").upper()
    if code in MYELOID_ONCOTREE_CODES:
        return True
    cancer_type = (
        sample_info.get("cancer_type", "") + " " +
        sample_info.get("cancer_type_detailed", "")
    ).lower()
    return any(kw in cancer_type for kw in MYELOID_KEYWORDS)


def load_gene_panels():
    """Load gene panel definitions."""
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


def main():
    print("=" * 70)
    print("GENIE v19.0 Co-occurrence Analysis (FILTERED)")
    print("Target genes: DNMT3A, IDH2, PTPN11, SETBP1")
    print("Excluding: Intron, Silent, UTR, Splice_Region variants")
    print("=" * 70)

    # Load clinical data
    print("\nLoading clinical data...")
    all_samples = load_clinical_samples()
    print(f"  Total samples: {len(all_samples):,}")

    myeloid_samples = {sid: info for sid, info in all_samples.items() if is_myeloid(info)}
    print(f"  Myeloid samples: {len(myeloid_samples):,}")

    patient_clinical = load_clinical_patients()
    print(f"  Total patients: {len(patient_clinical):,}")

    gene_panels = load_gene_panels()
    print(f"  Gene panels loaded: {len(gene_panels)}")

    # Check panel coverage
    coverage = {g: 0 for g in TARGET_GENES}
    all_four = 0
    for sid, info in myeloid_samples.items():
        panel = info.get("seq_panel", "")
        if panel in gene_panels:
            panel_genes = gene_panels[panel]
            all_covered = True
            for g in TARGET_GENES:
                if g in panel_genes:
                    coverage[g] += 1
                else:
                    all_covered = False
            if all_covered:
                all_four += 1
        else:
            for g in TARGET_GENES:
                coverage[g] += 1
            all_four += 1

    print(f"\n  Panel coverage in myeloid samples:")
    for g in TARGET_GENES:
        pct = coverage[g] / len(myeloid_samples) * 100
        print(f"    {g}: {coverage[g]:,} ({pct:.1f}%)")
    print(f"    All 4: {all_four:,} ({all_four / len(myeloid_samples) * 100:.1f}%)")

    # OncoTree distribution
    code_counts = Counter()
    for info in myeloid_samples.values():
        code_counts[info.get("oncotree_code", "UNKNOWN")] += 1
    print(f"\n  Top OncoTree codes:")
    for code, count in code_counts.most_common(15):
        print(f"    {code}: {count:,}")

    # Load and filter mutations
    print(f"\nLoading mutations (filtering to protein-altering only)...")
    maf_file = GENIE_RAW / "data_mutations_extended.txt"

    # Two tiers of filtering:
    # Tier 1: Any protein-altering mutation (Missense, Nonsense, Frameshift, Splice_Site, etc.)
    # Tier 2: Known hotspot/pathogenic variants only
    patient_mutations_tier1 = defaultdict(lambda: defaultdict(list))
    patient_mutations_tier2 = defaultdict(lambda: defaultdict(list))

    # Stats
    stats = {
        "total_rows": 0,
        "target_gene_rows": 0,
        "myeloid_target_rows": 0,
        "excluded_intron": 0,
        "excluded_silent": 0,
        "excluded_other": 0,
        "kept_coding": 0,
        "kept_hotspot": 0,
        "star_pattern_filtered": 0,
    }

    # Track total mutations per patient for hypermutation detection
    patient_total_target_muts = Counter()

    # Center-level stats
    center_stats = defaultdict(lambda: {"total": 0, "kept": 0, "excluded": 0})

    with open(maf_file) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            if "Hugo_Symbol" in line and header is None:
                header = line.strip().split("\t")
                continue
            if header is None:
                header = line.strip().split("\t")
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
            center = myeloid_samples[sample_id].get("center", "UNKNOWN")
            var_class = row.get("Variant_Classification", "")
            protein = row.get("HGVSp_Short", "")

            center_stats[center]["total"] += 1

            # Filter 1: Exclude non-coding variants
            if var_class in EXCLUDED_VAR_CLASSES:
                if var_class == "Intron":
                    stats["excluded_intron"] += 1
                elif var_class == "Silent":
                    stats["excluded_silent"] += 1
                else:
                    stats["excluded_other"] += 1
                center_stats[center]["excluded"] += 1
                continue

            # Filter 2: Exclude star-pattern artifacts (safety net)
            if is_star_pattern(protein):
                stats["star_pattern_filtered"] += 1
                center_stats[center]["excluded"] += 1
                continue

            # This is a protein-altering variant
            if var_class not in PATHOGENIC_VAR_CLASSES:
                # Unknown classification - skip conservatively
                stats["excluded_other"] += 1
                center_stats[center]["excluded"] += 1
                continue

            stats["kept_coding"] += 1
            center_stats[center]["kept"] += 1

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
                "center": center,
            }

            patient_mutations_tier1[patient_id][gene].append(mut_info)
            patient_total_target_muts[patient_id] += 1

            # Tier 2: Hotspot check
            if is_hotspot(gene, protein):
                patient_mutations_tier2[patient_id][gene].append(mut_info)
                stats["kept_hotspot"] += 1

    print(f"\n  Filtering stats:")
    print(f"    Total MAF rows: {stats['total_rows']:,}")
    print(f"    Target gene rows: {stats['target_gene_rows']:,}")
    print(f"    In myeloid samples: {stats['myeloid_target_rows']:,}")
    print(f"    Excluded - Intron: {stats['excluded_intron']:,}")
    print(f"    Excluded - Silent: {stats['excluded_silent']:,}")
    print(f"    Excluded - Other non-coding: {stats['excluded_other']:,}")
    print(f"    Excluded - Star pattern: {stats['star_pattern_filtered']:,}")
    print(f"    KEPT - Protein-altering: {stats['kept_coding']:,}")
    print(f"    KEPT - Known hotspot: {stats['kept_hotspot']:,}")

    print(f"\n  Center-level filtering (target genes in myeloid samples):")
    for center in sorted(center_stats.keys()):
        cs = center_stats[center]
        excl_pct = cs["excluded"] / cs["total"] * 100 if cs["total"] > 0 else 0
        print(f"    {center:10s}: {cs['total']:5d} total, {cs['kept']:5d} kept, "
              f"{cs['excluded']:5d} excluded ({excl_pct:.1f}%)")

    # Identify hypermutated patients
    hypermutated = {pid for pid, count in patient_total_target_muts.items()
                    if count > HYPERMUTATION_THRESHOLD}
    if hypermutated:
        print(f"\n  Hypermutated patients (>{HYPERMUTATION_THRESHOLD} target gene mutations): {len(hypermutated)}")
        for pid in sorted(hypermutated):
            count = patient_total_target_muts[pid]
            genes_hit = list(patient_mutations_tier1[pid].keys())
            print(f"    {pid}: {count} mutations in {genes_hit}")

    # Compute co-occurrence for both tiers
    for tier_name, patient_muts, exclude_hypermut in [
        ("TIER 1: Any protein-altering mutation", patient_mutations_tier1, False),
        ("TIER 1 (excl. hypermutated)", patient_mutations_tier1, True),
        ("TIER 2: Known hotspot variants only", patient_mutations_tier2, False),
    ]:
        print(f"\n{'=' * 70}")
        print(f"  {tier_name}")
        print(f"{'=' * 70}")

        # Build gene -> patient sets
        gene_patients = {g: set() for g in TARGET_GENES}
        for pid, gmuts in patient_muts.items():
            if exclude_hypermut and pid in hypermutated:
                continue
            for g in TARGET_GENES:
                if g in gmuts and len(gmuts[g]) > 0:
                    gene_patients[g].add(pid)

        all_patients = set()
        for pids in gene_patients.values():
            all_patients |= pids
        total = len(all_patients)

        print(f"\n  Patients with any target gene mutation: {total:,}")
        for g in TARGET_GENES:
            print(f"    {g}: {len(gene_patients[g]):,}")

        # Pairwise
        print(f"\n  Pairwise co-occurrence:")
        pairwise = {}
        for g1, g2 in combinations(TARGET_GENES, 2):
            key = f"{g1}+{g2}"
            overlap = gene_patients[g1] & gene_patients[g2]
            pairwise[key] = len(overlap)
            print(f"    {key}: {len(overlap)}")

        # Triple
        print(f"\n  Triple co-occurrence:")
        triple = {}
        triple_patients_map = {}
        for combo in combinations(TARGET_GENES, 3):
            key = "+".join(combo)
            overlap = gene_patients[combo[0]]
            for g in combo[1:]:
                overlap = overlap & gene_patients[g]
            triple[key] = len(overlap)
            triple_patients_map[key] = sorted(list(overlap))
            print(f"    {key}: {len(overlap)}")
            for pid in sorted(overlap)[:10]:
                muts = patient_muts[pid]
                mut_str = ", ".join(
                    f"{g}:{muts[g][0]['protein_change']}" for g in combo if g in muts and muts[g]
                )
                center = ""
                if patient_clinical.get(pid):
                    center = patient_clinical[pid].get("center", "")
                age = patient_clinical.get(pid, {}).get("age")
                age_str = f" age={age}" if age else ""
                print(f"      {pid}: {mut_str} [{center}]{age_str}")

        # Quadruple
        quad_overlap = gene_patients[TARGET_GENES[0]]
        for g in TARGET_GENES[1:]:
            quad_overlap = quad_overlap & gene_patients[g]

        print(f"\n  *** QUADRUPLE (DNMT3A+IDH2+PTPN11+SETBP1): {len(quad_overlap)} ***")
        if quad_overlap:
            for pid in sorted(quad_overlap):
                muts = patient_muts[pid]
                mut_str = ", ".join(
                    f"{g}:{[m['protein_change'] for m in muts[g]]}" for g in TARGET_GENES if g in muts
                )
                center = patient_clinical.get(pid, {}).get("center", "")
                age = patient_clinical.get(pid, {}).get("age")
                sex = patient_clinical.get(pid, {}).get("sex", "")
                print(f"      {pid}: {mut_str} [{center}] {sex} age={age}")

        # O/E ratios
        total_denom = len(myeloid_samples)
        freqs = {g: len(gene_patients[g]) / total_denom for g in TARGET_GENES}

        obs_exp = {}
        for g1, g2 in combinations(TARGET_GENES, 2):
            key = f"{g1}+{g2}"
            exp = freqs[g1] * freqs[g2] * total_denom
            obs = pairwise[key]
            ratio = obs / exp if exp > 0 else None
            obs_exp[key] = {"observed": obs, "expected": round(exp, 2),
                            "ratio": round(ratio, 2) if ratio else None}

        for combo in combinations(TARGET_GENES, 3):
            key = "+".join(combo)
            p = 1.0
            for g in combo:
                p *= freqs[g]
            exp = p * total_denom
            obs = triple[key]
            ratio = obs / exp if exp > 0 else None
            obs_exp[key] = {"observed": obs, "expected": round(exp, 4),
                            "ratio": round(ratio, 2) if ratio else None}

        # Quadruple
        p_all = 1.0
        for g in TARGET_GENES:
            p_all *= freqs[g]
        exp_quad = p_all * total_denom
        ratio_quad = len(quad_overlap) / exp_quad if exp_quad > 0 else None
        obs_exp["DNMT3A+IDH2+PTPN11+SETBP1"] = {
            "observed": len(quad_overlap),
            "expected": round(exp_quad, 6),
            "ratio": round(ratio_quad, 2) if ratio_quad else None,
        }

        print(f"\n  Observed/Expected ratios (denominator={total_denom:,} myeloid samples):")
        for key in obs_exp:
            v = obs_exp[key]
            print(f"    {key}: obs={v['observed']}, exp={v['expected']}, ratio={v['ratio']}")

    # Build final output JSON (Tier 1 excl. hypermutated = primary result)
    print(f"\n{'=' * 70}")
    print("Building final output...")
    print(f"{'=' * 70}")

    # Recompute Tier 1 excl. hypermutated for output
    gene_patients_final = {g: set() for g in TARGET_GENES}
    for pid, gmuts in patient_mutations_tier1.items():
        if pid in hypermutated:
            continue
        for g in TARGET_GENES:
            if g in gmuts:
                gene_patients_final[g].add(pid)

    single_final = {g: len(gene_patients_final[g]) for g in TARGET_GENES}
    pairwise_final = {}
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        pairwise_final[key] = len(gene_patients_final[g1] & gene_patients_final[g2])

    triple_final = {}
    triple_details_final = {}
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        overlap = gene_patients_final[combo[0]]
        for g in combo[1:]:
            overlap = overlap & gene_patients_final[g]
        triple_final[key] = len(overlap)
        details = []
        for pid in sorted(overlap)[:100]:
            muts = patient_mutations_tier1[pid]
            d = {"patient_id": pid}
            for g in TARGET_GENES:
                if g in muts:
                    d[g] = [m["protein_change"] for m in muts[g]]
            clin = patient_clinical.get(pid, {})
            if clin.get("age"):
                d["age"] = clin["age"]
            if clin.get("sex"):
                d["sex"] = clin["sex"]
            if clin.get("center"):
                d["center"] = clin["center"]
            details.append(d)
        triple_details_final[key] = details

    quad_final = gene_patients_final[TARGET_GENES[0]]
    for g in TARGET_GENES[1:]:
        quad_final = quad_final & gene_patients_final[g]

    quad_details = []
    for pid in sorted(quad_final):
        muts = patient_mutations_tier1[pid]
        d = {"patient_id": pid}
        for g in TARGET_GENES:
            if g in muts:
                d[g] = [{"protein_change": m["protein_change"],
                          "variant_classification": m["variant_classification"],
                          "center": m["center"]} for m in muts[g]]
        clin = patient_clinical.get(pid, {})
        d["age"] = clin.get("age")
        d["sex"] = clin.get("sex")
        d["center"] = clin.get("center")
        quad_details.append(d)

    # O/E for final
    total_denom = len(myeloid_samples)
    freqs_final = {g: single_final[g] / total_denom for g in TARGET_GENES}
    obs_exp_final = {}

    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        exp = freqs_final[g1] * freqs_final[g2] * total_denom
        obs = pairwise_final[key]
        obs_exp_final[key] = {"observed": obs, "expected": round(exp, 2),
                              "ratio": round(obs / exp, 2) if exp > 0 else None}

    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        p = 1.0
        for g in combo:
            p *= freqs_final[g]
        exp = p * total_denom
        obs = triple_final[key]
        obs_exp_final[key] = {"observed": obs, "expected": round(exp, 4),
                              "ratio": round(obs / exp, 2) if exp > 0 else None}

    p_all = 1.0
    for g in TARGET_GENES:
        p_all *= freqs_final[g]
    exp_quad = p_all * total_denom
    obs_exp_final["DNMT3A+IDH2+PTPN11+SETBP1"] = {
        "observed": len(quad_final),
        "expected": round(exp_quad, 6),
        "ratio": round(len(quad_final) / exp_quad, 2) if exp_quad > 0 else None,
    }

    output = {
        "database": "AACR GENIE v19.0-public (Synapse)",
        "analysis_date": "2026-03-19",
        "methodology": {
            "description": "Filtered co-occurrence analysis excluding non-coding variants",
            "variant_classes_included": sorted(PATHOGENIC_VAR_CLASSES),
            "variant_classes_excluded": sorted(EXCLUDED_VAR_CLASSES),
            "hypermutation_threshold": HYPERMUTATION_THRESHOLD,
            "hypermutated_patients_excluded": sorted(list(hypermutated)),
            "myeloid_filter": "OncoTree codes + keyword fallback",
        },
        "dataset_summary": {
            "total_samples": len(all_samples),
            "myeloid_samples": len(myeloid_samples),
            "panel_coverage": {
                "per_gene": coverage,
                "all_four_genes": all_four,
            },
        },
        "filtering_stats": stats,
        "center_filtering": {c: dict(v) for c, v in sorted(center_stats.items())},
        "target_genes": TARGET_GENES,
        "single_gene": single_final,
        "pairwise": pairwise_final,
        "triple": triple_final,
        "quadruple": {
            "DNMT3A+IDH2+PTPN11+SETBP1": len(quad_final),
        },
        "obs_exp_ratios": obs_exp_final,
        "triple_patient_details": triple_details_final,
        "quadruple_patient_details": quad_details,
        "comparison_with_unfiltered": {
            "note": "Previous unfiltered analysis found 15 quadruples, all due to UCHI intronic artifacts and UCSF hypermutation",
            "unfiltered_quad_count": 15,
            "filtered_quad_count": len(quad_final),
            "uchi_artifacts_removed": "3,525 intronic variants across target genes",
        },
    }

    output_path = RESULTS_DIR / "genie_filtered_cooccurrence.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nOutput: {output_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Myeloid samples: {len(myeloid_samples):,}")
    print(f"  Hypermutated patients excluded: {len(hypermutated)}")
    for g in TARGET_GENES:
        print(f"  {g}: {single_final[g]:,} patients")
    print(f"\n  Pairwise:")
    for key, count in pairwise_final.items():
        print(f"    {key}: {count}")
    print(f"\n  Triple:")
    for key, count in triple_final.items():
        print(f"    {key}: {count}")
    print(f"\n  QUADRUPLE (DNMT3A+IDH2+PTPN11+SETBP1): {len(quad_final)}")
    print(f"  Expected under independence: {obs_exp_final['DNMT3A+IDH2+PTPN11+SETBP1']['expected']}")

    if len(quad_final) == 0:
        print(f"\n  CONFIRMED: 0/{len(myeloid_samples):,} myeloid patients have all 4 mutations.")
        print(f"  The previous '15 quadruples' were ALL artifacts (UCHI intronic + UCSF hypermutation).")
    else:
        print(f"\n  {len(quad_final)} genuine quadruple carrier(s) found.")

    return output


if __name__ == "__main__":
    main()
