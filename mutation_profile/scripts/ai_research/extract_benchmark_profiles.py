"""
extract_benchmark_profiles.py -- Extract 20 diverse SETBP1-positive myeloid
patient profiles from GENIE v19.0 for benchmarking.

Inputs:
    - GENIE v19.0 via genie_loader.py (mutations, clinical, panels)

Outputs:
    - mutation_profile/results/ai_research/benchmark/benchmark_profiles.json

Selection criteria:
    1. Myeloid patients (OncoTree: AML, MDS, MPN, CMML, MDS/MPN, JMML, etc.)
    2. Coding variants only (exclude Intron, Silent, UTR, Flank, IGR, RNA, Splice_Region)
    3. SETBP1-positive (at least one SETBP1 coding mutation)
    4. Not hypermutated (<=20 coding mutations in 34 target genes)
    5. At least 3 mutated target genes
    6. 20 profiles selected for diversity

Runtime: ~15 seconds
Dependencies: pandas, numpy, json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for genie_loader import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from genie_loader import (
    GENIEData,
    MYELOID_ONCOTREE_CODES,
    NONCODING_CLASSIFICATIONS,
    TARGET_GENES,
)

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "results"
    / "ai_research"
    / "benchmark"
    / "benchmark_profiles.json"
)

# SETBP1 SKI domain hotspot positions (codons 858-871)
SETBP1_HOTSPOTS = {
    "p.I871T", "p.I871S", "p.D868N", "p.D868Y", "p.D868H",
    "p.G870S", "p.G870R", "p.G870D", "p.S869N", "p.S869R",
    "p.E858K", "p.T873I", "p.T873S",
}


def compute_vaf(row: pd.Series) -> float | None:
    """Compute tumor VAF from t_alt_count / t_depth."""
    alt = row.get("t_alt_count")
    depth = row.get("t_depth")
    if pd.notna(alt) and pd.notna(depth) and depth > 0:
        return round(float(alt) / float(depth), 4)
    return None


def main() -> None:
    print("=" * 70)
    print("BENCHMARK PROFILE EXTRACTION: 20 SETBP1+ Myeloid Patients")
    print("=" * 70)

    # Step 1: Load GENIE data
    loader = GENIEData()

    # Step 2: Get myeloid samples
    myeloid_samples = loader.get_myeloid_samples()
    print(f"\nMyeloid samples: {len(myeloid_samples):,}")

    # Step 3: Get coding mutations in myeloid samples, restricted to target genes
    coding = loader.get_coding_mutations(samples=myeloid_samples)
    target_coding = coding[coding["Hugo_Symbol"].isin(TARGET_GENES)].copy()
    print(f"Coding mutations in target genes (myeloid): {len(target_coding):,}")

    # Step 4: Exclude hypermutated samples
    hyper = loader.get_hypermutated_samples(TARGET_GENES, threshold=20)
    print(f"Hypermutated samples excluded: {len(hyper):,}")
    target_coding = target_coding[
        ~target_coding["Tumor_Sample_Barcode"].isin(hyper)
    ]

    # Step 5: Find SETBP1-positive patients
    setbp1_muts = target_coding[target_coding["Hugo_Symbol"] == "SETBP1"]
    setbp1_samples = set(setbp1_muts["Tumor_Sample_Barcode"])
    print(f"SETBP1+ myeloid samples (post-filter): {len(setbp1_samples):,}")

    # Step 6: For each SETBP1+ patient, collect their full target gene profile
    setbp1_muts_all = target_coding[
        target_coding["Tumor_Sample_Barcode"].isin(setbp1_samples)
    ]

    # Map sample to oncotree code
    clinical = loader.clinical
    sample_to_oncotree = dict(
        zip(clinical["SAMPLE_ID"], clinical["ONCOTREE_CODE"])
    )

    # Build per-patient profiles
    profiles = []
    for sample_id, group in setbp1_muts_all.groupby("Tumor_Sample_Barcode"):
        genes_mutated = set(group["Hugo_Symbol"])
        n_genes = len(genes_mutated)

        # Require at least 3 mutated target genes
        if n_genes < 3:
            continue

        oncotree = sample_to_oncotree.get(sample_id, "Unknown")

        mutations = []
        for _, row in group.iterrows():
            hgvsp = row.get("HGVSp_Short", "")
            if pd.isna(hgvsp):
                hgvsp = ""
            # Strip p. prefix for display
            hgvsp_display = hgvsp.replace("p.", "") if hgvsp else ""

            vaf = compute_vaf(row)

            mutations.append({
                "gene": row["Hugo_Symbol"],
                "hgvsp": hgvsp_display,
                "variant_classification": row["Variant_Classification"],
                "chromosome": str(row.get("Chromosome", "")),
                "start_position": int(row["Start_Position"]) if pd.notna(row.get("Start_Position")) else None,
                "ref_allele": row.get("Reference_Allele", ""),
                "alt_allele": row.get("Tumor_Seq_Allele2", ""),
                "t_vaf": vaf,
            })

        # Check if SETBP1 variant is hotspot or non-hotspot
        setbp1_variants = [m for m in mutations if m["gene"] == "SETBP1"]
        has_hotspot = any(
            f"p.{m['hgvsp']}" in SETBP1_HOTSPOTS for m in setbp1_variants
        )
        has_non_hotspot = any(
            f"p.{m['hgvsp']}" not in SETBP1_HOTSPOTS for m in setbp1_variants
        )

        profiles.append({
            "patient_id": sample_id,
            "oncotree_code": oncotree,
            "n_mutations": len(mutations),
            "n_genes": n_genes,
            "genes": sorted(genes_mutated),
            "has_hotspot_setbp1": has_hotspot,
            "has_non_hotspot_setbp1": has_non_hotspot,
            "mutations": sorted(mutations, key=lambda m: m["gene"]),
        })

    print(f"SETBP1+ patients with >=3 target genes: {len(profiles)}")

    # Step 7: Diverse selection of 20 profiles
    selected = select_diverse_20(profiles)

    # Step 8: Clean up output (remove selection helper fields)
    output_profiles = []
    for p in selected:
        output_profiles.append({
            "patient_id": p["patient_id"],
            "oncotree_code": p["oncotree_code"],
            "n_mutations": p["n_mutations"],
            "mutations": p["mutations"],
        })

    output = {
        "metadata": {
            "description": "20 diverse SETBP1-positive myeloid patient profiles from GENIE v19.0",
            "source": "AACR GENIE v19.0-public",
            "filters": {
                "myeloid_oncotree_codes": sorted(MYELOID_ONCOTREE_CODES),
                "coding_only": True,
                "noncoding_excluded": sorted(NONCODING_CLASSIFICATIONS),
                "hypermutation_threshold": 20,
                "min_mutated_genes": 3,
            },
            "target_genes": TARGET_GENES,
            "n_setbp1_positive_total": len(setbp1_samples),
            "n_with_3plus_genes": len(profiles),
            "n_selected": len(output_profiles),
        },
        "profiles": output_profiles,
    }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved {len(output_profiles)} profiles to {OUTPUT_PATH}")
    print_summary(selected)


def select_diverse_20(profiles: list[dict]) -> list[dict]:
    """Select 20 diverse profiles from candidates.

    Strategy:
    - Include profiles across different oncotree codes
    - Include both hotspot and non-hotspot SETBP1 variants
    - Include profiles with common co-mutations (ASXL1, TET2, DNMT3A)
    - Include profiles with rare co-mutations (DDX41, CALR, MPL, GATA2)
    - Range of mutation counts (3 to many)
    - Prioritize profiles with VAF data available
    """
    if len(profiles) <= 20:
        print(f"  Only {len(profiles)} candidates; returning all.")
        return profiles

    selected = []
    used_ids = set()

    def pick(p: dict) -> None:
        if p["patient_id"] not in used_ids:
            selected.append(p)
            used_ids.add(p["patient_id"])

    # --- Tier 1: Ensure oncotree diversity ---
    oncotree_groups: dict[str, list] = {}
    for p in profiles:
        oncotree_groups.setdefault(p["oncotree_code"], []).append(p)

    print(f"\n  OncoTree distribution: {dict(Counter(p['oncotree_code'] for p in profiles))}")

    # Pick one from each oncotree code (sorted by n_genes desc for richest profile)
    for code in sorted(oncotree_groups.keys()):
        if len(selected) >= 20:
            break
        candidates = sorted(
            oncotree_groups[code],
            key=lambda p: (-p["n_genes"], -p["n_mutations"]),
        )
        for c in candidates:
            if c["patient_id"] not in used_ids:
                pick(c)
                break

    # --- Tier 2: Non-hotspot SETBP1 variants (likely VUS) ---
    non_hotspot = [p for p in profiles if p["has_non_hotspot_setbp1"] and p["patient_id"] not in used_ids]
    non_hotspot.sort(key=lambda p: -p["n_genes"])
    for p in non_hotspot[:4]:
        if len(selected) >= 20:
            break
        pick(p)

    # --- Tier 3: Rare co-mutation partners ---
    rare_genes = {"DDX41", "CALR", "MPL", "GATA2", "PHF6", "SMC1A", "SMC3", "RAD21", "BCORL1"}
    rare_profiles = [
        p for p in profiles
        if p["patient_id"] not in used_ids
        and len(set(p["genes"]) & rare_genes) > 0
    ]
    rare_profiles.sort(key=lambda p: (-len(set(p["genes"]) & rare_genes), -p["n_genes"]))
    for p in rare_profiles[:4]:
        if len(selected) >= 20:
            break
        pick(p)

    # --- Tier 4: High mutation count profiles ---
    remaining = [p for p in profiles if p["patient_id"] not in used_ids]
    remaining.sort(key=lambda p: -p["n_genes"])
    for p in remaining:
        if len(selected) >= 20:
            break
        pick(p)

    # --- Tier 5: Fill with VAF-rich profiles ---
    remaining = [p for p in profiles if p["patient_id"] not in used_ids]
    remaining.sort(key=lambda p: -sum(1 for m in p["mutations"] if m.get("t_vaf") is not None))
    for p in remaining:
        if len(selected) >= 20:
            break
        pick(p)

    return selected[:20]


def print_summary(selected: list[dict]) -> None:
    """Print summary of selected profiles."""
    print(f"\n{'=' * 70}")
    print("SELECTED PROFILES SUMMARY")
    print(f"{'=' * 70}")

    for i, p in enumerate(selected, 1):
        genes = [m["gene"] for m in p["mutations"]]
        gene_str = ", ".join(sorted(set(genes)))
        setbp1_vars = [m["hgvsp"] for m in p["mutations"] if m["gene"] == "SETBP1"]
        vaf_count = sum(1 for m in p["mutations"] if m.get("t_vaf") is not None)

        print(
            f"\n  {i:2d}. {p['patient_id']}"
            f" | {p['oncotree_code']:8s}"
            f" | {p['n_mutations']} muts ({len(set(genes))} genes)"
            f" | SETBP1: {', '.join(setbp1_vars)}"
            f" | VAFs: {vaf_count}/{p['n_mutations']}"
        )
        print(f"      Genes: {gene_str}")

    # Stats
    oncotree_codes = Counter(p["oncotree_code"] for p in selected)
    n_with_vaf = sum(1 for p in selected if any(m.get("t_vaf") is not None for m in p["mutations"]))
    hotspot_count = sum(1 for p in selected if p.get("has_hotspot_setbp1"))
    non_hotspot_count = sum(1 for p in selected if p.get("has_non_hotspot_setbp1"))

    print(f"\n  OncoTree codes: {dict(oncotree_codes)}")
    print(f"  With VAF data: {n_with_vaf}/20")
    print(f"  Hotspot SETBP1: {hotspot_count}, Non-hotspot: {non_hotspot_count}")
    avg_muts = sum(p["n_mutations"] for p in selected) / len(selected)
    print(f"  Avg mutations per profile: {avg_muts:.1f}")


if __name__ == "__main__":
    main()
