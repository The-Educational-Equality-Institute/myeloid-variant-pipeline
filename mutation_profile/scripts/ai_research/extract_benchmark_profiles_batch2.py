"""
extract_benchmark_profiles_batch2.py -- Extract 20 MORE SETBP1-positive myeloid
patient profiles from GENIE v19.0 for extended benchmarking (batch 2).

Excludes all patient IDs already in batch 1.
Prioritizes:
    1. VUS variants (non-hotspot SETBP1 positions)
    2. EZH2 co-mutations (for VUS reclassification testing)
    3. DDX41 co-mutations (rare partner)
    4. OncoTree codes underrepresented or absent from batch 1

Inputs:
    - GENIE v19.0 via genie_loader.py (mutations, clinical, panels)
    - mutation_profile/results/ai_research/benchmark/benchmark_profiles.json (batch 1)

Outputs:
    - mutation_profile/results/ai_research/benchmark/benchmark_profiles_batch2.json

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

BATCH1_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "results"
    / "ai_research"
    / "benchmark"
    / "benchmark_profiles.json"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "results"
    / "ai_research"
    / "benchmark"
    / "benchmark_profiles_batch2.json"
)

# SETBP1 SKI domain hotspot positions (codons 858-871)
SETBP1_HOTSPOTS = {
    "p.I871T", "p.I871S", "p.D868N", "p.D868Y", "p.D868H",
    "p.G870S", "p.G870R", "p.G870D", "p.S869N", "p.S869R",
    "p.E858K", "p.T873I", "p.T873S",
}

# OncoTree codes already well-represented in batch 1
BATCH1_ONCOTREE_CODES = {
    "AML": 11, "CMML": 1, "CNL": 1, "ET": 1, "MDS": 2,
    "MDS/MPN": 1, "MDSMPNU": 1, "MPN": 1, "PMF": 1,
}


def load_batch1_ids() -> set[str]:
    """Load patient IDs from batch 1 to exclude."""
    with open(BATCH1_PATH) as f:
        data = json.load(f)
    ids = {p["patient_id"] for p in data["profiles"]}
    print(f"Batch 1 patient IDs to exclude: {len(ids)}")
    return ids


def compute_vaf(row: pd.Series) -> float | None:
    """Compute tumor VAF from t_alt_count / t_depth."""
    alt = row.get("t_alt_count")
    depth = row.get("t_depth")
    if pd.notna(alt) and pd.notna(depth) and depth > 0:
        return round(float(alt) / float(depth), 4)
    return None


def main() -> None:
    print("=" * 70)
    print("BENCHMARK PROFILE EXTRACTION BATCH 2: 20 NEW SETBP1+ Myeloid Patients")
    print("=" * 70)

    # Step 1: Load batch 1 IDs to exclude
    batch1_ids = load_batch1_ids()

    # Step 2: Load GENIE data
    loader = GENIEData()

    # Step 3: Get myeloid samples
    myeloid_samples = loader.get_myeloid_samples()
    print(f"\nMyeloid samples: {len(myeloid_samples):,}")

    # Step 4: Get coding mutations in myeloid samples, restricted to target genes
    coding = loader.get_coding_mutations(samples=myeloid_samples)
    target_coding = coding[coding["Hugo_Symbol"].isin(TARGET_GENES)].copy()
    print(f"Coding mutations in target genes (myeloid): {len(target_coding):,}")

    # Step 5: Exclude hypermutated samples
    hyper = loader.get_hypermutated_samples(TARGET_GENES, threshold=20)
    print(f"Hypermutated samples excluded: {len(hyper):,}")
    target_coding = target_coding[
        ~target_coding["Tumor_Sample_Barcode"].isin(hyper)
    ]

    # Step 6: Find SETBP1-positive patients
    setbp1_muts = target_coding[target_coding["Hugo_Symbol"] == "SETBP1"]
    setbp1_samples = set(setbp1_muts["Tumor_Sample_Barcode"])
    print(f"SETBP1+ myeloid samples (post-filter): {len(setbp1_samples):,}")

    # Step 7: Exclude batch 1 patients
    setbp1_samples -= batch1_ids
    print(f"SETBP1+ after excluding batch 1: {len(setbp1_samples):,}")

    # Step 8: For each SETBP1+ patient, collect their full target gene profile
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

        # Check for priority genes
        has_ezh2 = "EZH2" in genes_mutated
        has_ddx41 = "DDX41" in genes_mutated

        profiles.append({
            "patient_id": sample_id,
            "oncotree_code": oncotree,
            "n_mutations": len(mutations),
            "n_genes": n_genes,
            "genes": sorted(genes_mutated),
            "has_hotspot_setbp1": has_hotspot,
            "has_non_hotspot_setbp1": has_non_hotspot,
            "has_ezh2": has_ezh2,
            "has_ddx41": has_ddx41,
            "mutations": sorted(mutations, key=lambda m: m["gene"]),
        })

    print(f"SETBP1+ patients with >=3 target genes (excl batch 1): {len(profiles)}")

    # Step 9: Diverse selection of 20 profiles (batch 2 priorities)
    selected = select_diverse_20_batch2(profiles)

    # Step 10: Clean up output
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
            "description": "20 NEW diverse SETBP1-positive myeloid patient profiles from GENIE v19.0 (batch 2)",
            "source": "AACR GENIE v19.0-public",
            "batch": 2,
            "excludes_batch1_ids": sorted(batch1_ids),
            "filters": {
                "myeloid_oncotree_codes": sorted(MYELOID_ONCOTREE_CODES),
                "coding_only": True,
                "noncoding_excluded": sorted(NONCODING_CLASSIFICATIONS),
                "hypermutation_threshold": 20,
                "min_mutated_genes": 3,
            },
            "target_genes": TARGET_GENES,
            "n_setbp1_positive_total": len(setbp1_samples) + len(batch1_ids),
            "n_with_3plus_genes_available": len(profiles),
            "n_selected": len(output_profiles),
            "selection_priorities": [
                "VUS variants (non-hotspot SETBP1 positions)",
                "EZH2 co-mutations (VUS reclassification testing)",
                "DDX41 co-mutations (rare partner)",
                "OncoTree codes absent or underrepresented in batch 1",
            ],
        },
        "profiles": output_profiles,
    }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved {len(output_profiles)} profiles to {OUTPUT_PATH}")
    print_summary(selected)


def select_diverse_20_batch2(profiles: list[dict]) -> list[dict]:
    """Select 20 diverse profiles with batch 2 priorities.

    Priority order:
    1. EZH2 co-mutations (for VUS reclassification testing)
    2. DDX41 co-mutations (rare partner)
    3. Non-hotspot SETBP1 (VUS variants)
    4. OncoTree codes NOT in batch 1 or underrepresented
    5. OncoTree diversity for remaining codes
    6. High gene count profiles
    7. VAF-rich profiles
    """
    if len(profiles) <= 20:
        print(f"  Only {len(profiles)} candidates; returning all.")
        return profiles

    selected = []
    used_ids = set()

    def pick(p: dict) -> bool:
        if p["patient_id"] not in used_ids:
            selected.append(p)
            used_ids.add(p["patient_id"])
            return True
        return False

    oncotree_dist = Counter(p["oncotree_code"] for p in profiles)
    print(f"\n  Available OncoTree distribution: {dict(oncotree_dist)}")

    # --- Priority 1: EZH2 co-mutations ---
    ezh2_profiles = [p for p in profiles if p["has_ezh2"]]
    ezh2_profiles.sort(key=lambda p: (-p["n_genes"], -p["n_mutations"]))
    picked_ezh2 = 0
    for p in ezh2_profiles:
        if len(selected) >= 20:
            break
        if pick(p):
            picked_ezh2 += 1
        if picked_ezh2 >= 4:
            break
    print(f"  Priority 1 (EZH2): picked {picked_ezh2} (available: {len(ezh2_profiles)})")

    # --- Priority 2: DDX41 co-mutations ---
    ddx41_profiles = [p for p in profiles if p["has_ddx41"] and p["patient_id"] not in used_ids]
    ddx41_profiles.sort(key=lambda p: (-p["n_genes"], -p["n_mutations"]))
    picked_ddx41 = 0
    for p in ddx41_profiles:
        if len(selected) >= 20:
            break
        if pick(p):
            picked_ddx41 += 1
        if picked_ddx41 >= 3:
            break
    print(f"  Priority 2 (DDX41): picked {picked_ddx41} (available: {len(ddx41_profiles) + picked_ddx41})")

    # --- Priority 3: Non-hotspot SETBP1 (VUS) ---
    vus_profiles = [
        p for p in profiles
        if p["has_non_hotspot_setbp1"] and not p["has_hotspot_setbp1"]
        and p["patient_id"] not in used_ids
    ]
    vus_profiles.sort(key=lambda p: (-p["n_genes"], -p["n_mutations"]))
    picked_vus = 0
    for p in vus_profiles:
        if len(selected) >= 20:
            break
        if pick(p):
            picked_vus += 1
        if picked_vus >= 5:
            break
    print(f"  Priority 3 (VUS-only SETBP1): picked {picked_vus} (available: {len(vus_profiles) + picked_vus})")

    # --- Priority 4: OncoTree codes NOT in batch 1 ---
    new_oncotree_profiles = [
        p for p in profiles
        if p["oncotree_code"] not in BATCH1_ONCOTREE_CODES
        and p["patient_id"] not in used_ids
    ]
    new_oncotree_profiles.sort(key=lambda p: (-p["n_genes"], -p["n_mutations"]))
    picked_new_onco = 0
    for p in new_oncotree_profiles:
        if len(selected) >= 20:
            break
        if pick(p):
            picked_new_onco += 1
        if picked_new_onco >= 4:
            break
    print(f"  Priority 4 (new OncoTree): picked {picked_new_onco} (available: {len(new_oncotree_profiles) + picked_new_onco})")

    # --- Priority 5: OncoTree diversity for codes underrepresented in batch 1 ---
    # Prefer codes that had only 1 representative in batch 1
    underrep_codes = {code for code, count in BATCH1_ONCOTREE_CODES.items() if count <= 1}
    underrep_profiles = [
        p for p in profiles
        if p["oncotree_code"] in underrep_codes
        and p["patient_id"] not in used_ids
    ]
    underrep_profiles.sort(key=lambda p: (-p["n_genes"], -p["n_mutations"]))
    picked_underrep = 0
    for p in underrep_profiles:
        if len(selected) >= 20:
            break
        if pick(p):
            picked_underrep += 1
        if picked_underrep >= 4:
            break
    print(f"  Priority 5 (underrep OncoTree): picked {picked_underrep}")

    # --- Priority 6: Remaining OncoTree diversity ---
    remaining_by_oncotree: dict[str, list] = {}
    for p in profiles:
        if p["patient_id"] not in used_ids:
            remaining_by_oncotree.setdefault(p["oncotree_code"], []).append(p)

    for code in sorted(remaining_by_oncotree.keys()):
        if len(selected) >= 20:
            break
        candidates = sorted(
            remaining_by_oncotree[code],
            key=lambda p: (-p["n_genes"], -p["n_mutations"]),
        )
        for c in candidates:
            if c["patient_id"] not in used_ids:
                pick(c)
                break

    # --- Priority 7: Fill with high gene count ---
    remaining = [p for p in profiles if p["patient_id"] not in used_ids]
    remaining.sort(key=lambda p: (-p["n_genes"], -p["n_mutations"]))
    for p in remaining:
        if len(selected) >= 20:
            break
        pick(p)

    # --- Priority 8: Fill with VAF-rich profiles ---
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
    print("SELECTED PROFILES SUMMARY (BATCH 2)")
    print(f"{'=' * 70}")

    for i, p in enumerate(selected, 1):
        genes = [m["gene"] for m in p["mutations"]]
        gene_str = ", ".join(sorted(set(genes)))
        setbp1_vars = [m["hgvsp"] for m in p["mutations"] if m["gene"] == "SETBP1"]
        vaf_count = sum(1 for m in p["mutations"] if m.get("t_vaf") is not None)

        flags = []
        if p.get("has_ezh2"):
            flags.append("EZH2")
        if p.get("has_ddx41"):
            flags.append("DDX41")
        if p.get("has_non_hotspot_setbp1") and not p.get("has_hotspot_setbp1"):
            flags.append("VUS")

        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(
            f"\n  {i:2d}. {p['patient_id']}"
            f" | {p['oncotree_code']:8s}"
            f" | {p['n_mutations']} muts ({len(set(genes))} genes)"
            f" | SETBP1: {', '.join(setbp1_vars)}"
            f" | VAFs: {vaf_count}/{p['n_mutations']}"
            f"{flag_str}"
        )
        print(f"      Genes: {gene_str}")

    # Stats
    oncotree_codes = Counter(p["oncotree_code"] for p in selected)
    n_with_vaf = sum(1 for p in selected if any(m.get("t_vaf") is not None for m in p["mutations"]))
    hotspot_count = sum(1 for p in selected if p.get("has_hotspot_setbp1"))
    non_hotspot_count = sum(1 for p in selected if p.get("has_non_hotspot_setbp1"))
    ezh2_count = sum(1 for p in selected if p.get("has_ezh2"))
    ddx41_count = sum(1 for p in selected if p.get("has_ddx41"))

    print(f"\n  OncoTree codes: {dict(oncotree_codes)}")
    print(f"  With VAF data: {n_with_vaf}/{len(selected)}")
    print(f"  Hotspot SETBP1: {hotspot_count}, Non-hotspot: {non_hotspot_count}")
    print(f"  EZH2 co-mutations: {ezh2_count}")
    print(f"  DDX41 co-mutations: {ddx41_count}")
    avg_muts = sum(p["n_mutations"] for p in selected) / len(selected) if selected else 0
    print(f"  Avg mutations per profile: {avg_muts:.1f}")

    # Compare with batch 1
    print(f"\n  Batch 1 OncoTree codes were: {BATCH1_ONCOTREE_CODES}")
    new_codes = set(oncotree_codes.keys()) - set(BATCH1_ONCOTREE_CODES.keys())
    if new_codes:
        print(f"  NEW OncoTree codes in batch 2: {new_codes}")


if __name__ == "__main__":
    main()
