#!/usr/bin/env python3
"""
Clonal Architecture Analysis of Triple Carriers in GENIE v19.0

Analyzes VAF (variant allele frequency) data from the GENIE MAF file to determine
clonal hierarchy in patients carrying 3 of the 4 target genes (DNMT3A, IDH2, PTPN11,
SETBP1). Compares findings to Henrik's clonal pattern.

Data sources:
  - mutation_profile/data/genie/raw/data_mutations_extended.txt (MAF)
  - mutation_profile/data/genie/raw/data_clinical_sample.txt (ONCOTREE_CODE filtering)
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
GENIE_RAW = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]
TARGET_GENES_SET = set(TARGET_GENES)

PATHOGENIC_VAR_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    "Translation_Start_Site",
}

# Myeloid ONCOTREE_CODE prefixes
MYELOID_PREFIXES = ("AML", "MDS", "MPN", "CMML", "JMML", "CML", "TMN", "SM", "CEL", "HES")

# Exact ONCOTREE_CODEs that don't match the prefixes above
MYELOID_EXACT_CODES = {"AMML", "APL", "APMF", "AUL", "MLADS", "MS"}

# Henrik's pattern for comparison
HENRIK_PATTERN = {
    "EZH2": {"variant": "V662A", "vaf": 0.59, "role": "Pathogenic, likely founder"},
    "DNMT3A": {"variant": "R882H", "vaf": 0.39, "role": "Pathogenic hotspot"},
    "SETBP1": {"variant": "G870S", "vaf": 0.34, "role": "Likely pathogenic"},
    "PTPN11": {"variant": "E76Q", "vaf": 0.29, "role": "Pathogenic"},
    "IDH2": {"variant": "R140Q", "vaf": 0.02, "role": "Pathogenic subclone"},
}

MIN_DEPTH = 50  # Minimum sequencing depth for reliable VAF


def is_myeloid(oncotree_code: str) -> bool:
    """Check if an ONCOTREE_CODE represents a myeloid malignancy."""
    code = oncotree_code.upper().strip()
    if code in MYELOID_EXACT_CODES:
        return True
    for prefix in MYELOID_PREFIXES:
        if code.startswith(prefix):
            return True
    return False


def extract_variant_name(protein_change: str) -> str:
    """Extract clean variant name from HGVSp_Short (e.g., 'p.R882H' -> 'R882H')."""
    if not protein_change:
        return ""
    return protein_change.replace("p.", "")


def compute_vaf(t_alt_count: str, t_depth: str) -> tuple[float | None, int | None]:
    """Compute VAF from alt count and depth. Returns (vaf, depth) or (None, None)."""
    try:
        depth = int(t_depth)
        alt = int(t_alt_count)
        if depth >= MIN_DEPTH:
            return alt / depth, depth
    except (ValueError, TypeError):
        pass
    return None, None


def percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[f]
    return data[f] + (k - f) * (data[c] - data[f])


def main():
    print("=" * 80)
    print("CLONAL ARCHITECTURE ANALYSIS - GENIE v19.0 TRIPLE CARRIERS")
    print("Target genes: DNMT3A, IDH2, PTPN11, SETBP1")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Load clinical samples for myeloid filtering
    # ──────────────────────────────────────────────────────────────────────
    print("\n[Step 1] Loading clinical sample data for ONCOTREE_CODE filtering...")
    myeloid_samples = {}  # sample_id -> patient_id
    sample_oncotree = {}  # sample_id -> oncotree_code
    total_samples = 0

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
            total_samples += 1

            sid = row.get("SAMPLE_ID", "")
            pid = row.get("PATIENT_ID", "")
            code = row.get("ONCOTREE_CODE", "")

            if sid and is_myeloid(code):
                myeloid_samples[sid] = pid
                sample_oncotree[sid] = code

    print(f"  Total samples in GENIE: {total_samples:,}")
    print(f"  Myeloid samples: {len(myeloid_samples):,}")

    # Count oncotree code distribution
    code_counts = Counter(sample_oncotree.values())
    print(f"  Top myeloid oncotree codes:")
    for code, count in code_counts.most_common(15):
        print(f"    {code:>10}: {count:>5}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Single-pass MAF scan for target gene mutations
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[Step 2] Scanning MAF file for target gene mutations in myeloid samples...")

    # Per-patient, per-sample mutation data
    # Structure: patient_mutations[pid][gene] = [list of {variant, vaf, depth, sample_id, ...}]
    patient_mutations = defaultdict(lambda: defaultdict(list))

    # Per-sample data (for multi-sample patients)
    sample_mutations = defaultdict(lambda: defaultdict(list))

    # Solo gene carriers (for VAF comparison: solo vs co-occurrence)
    all_gene_vafs = defaultdict(list)  # gene -> [(vaf, depth, pid, variant)]

    row_count = 0
    target_rows = 0

    with open(GENIE_RAW / "data_mutations_extended.txt") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            if header is None:
                header = line.strip().split("\t")
                continue

            row_count += 1
            if row_count % 1_000_000 == 0:
                print(f"  {row_count:,} rows scanned...")

            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))

            gene = fields[header.index("Hugo_Symbol")]
            if gene not in TARGET_GENES_SET:
                continue

            sid = fields[header.index("Tumor_Sample_Barcode")]
            if sid not in myeloid_samples:
                continue

            var_class = fields[header.index("Variant_Classification")]
            if var_class not in PATHOGENIC_VAR_CLASSES:
                continue

            protein = fields[header.index("HGVSp_Short")]
            # Skip synonymous-like annotations
            if re.match(r"^p\.\*\d+\*$", protein):
                continue

            pid = myeloid_samples[sid]
            variant_name = extract_variant_name(protein)
            t_alt = fields[header.index("t_alt_count")]
            t_dep = fields[header.index("t_depth")]
            vaf, depth = compute_vaf(t_alt, t_dep)

            target_rows += 1

            mut_info = {
                "variant": variant_name,
                "protein_change": protein,
                "variant_classification": var_class,
                "vaf": vaf,
                "depth": depth,
                "sample_id": sid,
                "t_alt_count": t_alt,
                "t_depth": t_dep,
            }

            patient_mutations[pid][gene].append(mut_info)
            sample_mutations[sid][gene].append(mut_info)

            if vaf is not None:
                all_gene_vafs[gene].append((vaf, depth, pid, variant_name))

    print(f"  Total rows scanned: {row_count:,}")
    print(f"  Target gene rows in myeloid samples (protein-altering): {target_rows:,}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Identify gene carriers and triple carriers
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[Step 3] Identifying carrier sets and triple carriers...")

    gene_carriers = {g: set() for g in TARGET_GENES}
    for pid, gmuts in patient_mutations.items():
        for g in TARGET_GENES:
            if g in gmuts:
                gene_carriers[g].add(pid)

    for g in TARGET_GENES:
        print(f"  {g}: {len(gene_carriers[g]):,} carriers")

    # Find triple carriers (3 of 4 target genes)
    triple_combos = list(combinations(TARGET_GENES, 3))
    all_triple_carriers = {}  # pid -> {"combo": (g1,g2,g3), "missing": g4}
    combo_counts = {}

    for combo in triple_combos:
        missing = [g for g in TARGET_GENES if g not in combo][0]
        combo_key = "+".join(combo)
        carriers_in_combo = set.intersection(*[gene_carriers[g] for g in combo])

        # Exclude quadruple carriers (if any)
        quad_carriers = set.intersection(*[gene_carriers[g] for g in TARGET_GENES])
        carriers_in_combo -= quad_carriers

        combo_counts[combo_key] = len(carriers_in_combo)
        print(f"  Triple {combo_key} (missing {missing}): {len(carriers_in_combo)} patients")

        for pid in carriers_in_combo:
            if pid not in all_triple_carriers:
                all_triple_carriers[pid] = {"combo": combo, "missing": missing}
            else:
                # Patient might appear in multiple combos if they carry all 4
                # But we excluded quads above, so this shouldn't happen
                pass

    # Also check quadruple
    quad_carriers = set.intersection(*[gene_carriers[g] for g in TARGET_GENES])
    print(f"  Quadruple (all 4): {len(quad_carriers)} patients")

    total_triples = len(all_triple_carriers)
    print(f"\n  Total unique triple carriers: {total_triples}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Extract VAF and determine clonal hierarchy for each triple
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n[Step 4] Extracting VAF data and determining clonal hierarchy...")

    triple_details = []
    founder_gene_counter = Counter()  # Which gene has highest VAF most often
    subclonal_gene_counter = Counter()  # Which gene has lowest VAF most often
    vaf_rank_accumulator = defaultdict(list)  # gene -> list of ranks (1=highest VAF)

    # Track patients with multiple samples
    multi_sample_patients = []

    for pid in sorted(all_triple_carriers.keys()):
        info = all_triple_carriers[pid]
        combo = info["combo"]
        missing = info["missing"]

        # Get all samples for this patient
        patient_sample_ids = set()
        for g in combo:
            for m in patient_mutations[pid][g]:
                patient_sample_ids.add(m["sample_id"])

        # For each sample, get best VAF per gene (highest depth)
        sample_profiles = {}
        for sid in sorted(patient_sample_ids):
            profile = {}
            for g in combo:
                # Find mutations for this gene in this sample
                gene_muts_in_sample = [m for m in patient_mutations[pid][g] if m["sample_id"] == sid]
                if gene_muts_in_sample:
                    # Pick the one with highest depth (most reliable VAF)
                    best = max(gene_muts_in_sample, key=lambda x: x["depth"] or 0)
                    profile[g] = best
            if profile:
                sample_profiles[sid] = profile

        # Determine hierarchy from the sample with most complete VAF data
        best_sample = None
        best_vaf_count = 0
        for sid, profile in sample_profiles.items():
            vaf_count = sum(1 for g in combo if g in profile and profile[g]["vaf"] is not None)
            if vaf_count > best_vaf_count:
                best_vaf_count = vaf_count
                best_sample = sid

        # Build hierarchy
        hierarchy = None
        vaf_ordering = []
        if best_sample and best_vaf_count >= 2:
            profile = sample_profiles[best_sample]
            gene_vafs = []
            for g in combo:
                if g in profile and profile[g]["vaf"] is not None:
                    gene_vafs.append((g, profile[g]["vaf"], profile[g]["variant"]))

            if len(gene_vafs) >= 2:
                gene_vafs.sort(key=lambda x: -x[1])  # Sort by VAF descending
                vaf_ordering = [(g, v, var) for g, v, var in gene_vafs]

                founder = gene_vafs[0][0]
                subclonal = gene_vafs[-1][0]
                founder_gene_counter[founder] += 1
                subclonal_gene_counter[subclonal] += 1

                # Assign ranks
                for rank, (g, v, var) in enumerate(gene_vafs, 1):
                    vaf_rank_accumulator[g].append(rank)

                hierarchy = {
                    "founder_gene": founder,
                    "founder_vaf": gene_vafs[0][1],
                    "subclonal_gene": subclonal,
                    "subclonal_vaf": gene_vafs[-1][1],
                    "ordering": [{"gene": g, "vaf": round(v, 4), "variant": var} for g, v, var in gene_vafs],
                    "sample_id": best_sample,
                }

        # Check for multi-sample patients
        is_multi_sample = len(sample_profiles) > 1
        multi_sample_data = None
        if is_multi_sample:
            multi_sample_data = {}
            for sid, profile in sorted(sample_profiles.items()):
                sample_vafs = {}
                for g in combo:
                    if g in profile and profile[g]["vaf"] is not None:
                        sample_vafs[g] = {
                            "vaf": round(profile[g]["vaf"], 4),
                            "variant": profile[g]["variant"],
                            "depth": profile[g]["depth"],
                        }
                multi_sample_data[sid] = sample_vafs
            multi_sample_patients.append({
                "patient_id": pid,
                "combo": "+".join(combo),
                "samples": multi_sample_data,
            })

        # Build detail record
        detail = {
            "patient_id": pid,
            "triple_combo": "+".join(combo),
            "missing_gene": missing,
            "n_samples": len(sample_profiles),
            "mutations": {},
            "hierarchy": hierarchy,
            "multi_sample_data": multi_sample_data,
        }

        # Add all mutations for this patient across target genes
        for g in combo:
            gene_muts = patient_mutations[pid][g]
            # Deduplicate by (variant, sample_id)
            seen = set()
            unique_muts = []
            for m in gene_muts:
                key = (m["variant"], m["sample_id"])
                if key not in seen:
                    seen.add(key)
                    unique_muts.append({
                        "variant": m["variant"],
                        "vaf": round(m["vaf"], 4) if m["vaf"] is not None else None,
                        "depth": m["depth"],
                        "sample_id": m["sample_id"],
                        "variant_classification": m["variant_classification"],
                    })
            detail["mutations"][g] = unique_muts

        # Check if patient also has the 4th gene subclonally
        if missing in patient_mutations[pid]:
            fourth_muts = patient_mutations[pid][missing]
            detail["has_4th_gene"] = True
            detail["4th_gene_mutations"] = [
                {
                    "variant": m["variant"],
                    "vaf": round(m["vaf"], 4) if m["vaf"] is not None else None,
                    "depth": m["depth"],
                    "sample_id": m["sample_id"],
                }
                for m in fourth_muts
            ]
        else:
            detail["has_4th_gene"] = False

        triple_details.append(detail)

    # Print triple carrier details
    print(f"\n  --- TRIPLE CARRIER PROFILES ---")
    for det in triple_details:
        pid = det["patient_id"]
        combo = det["triple_combo"]
        missing = det["missing_gene"]
        h = det["hierarchy"]
        print(f"\n  Patient {pid}: {combo} (missing {missing})")
        for g, muts in det["mutations"].items():
            for m in muts:
                vaf_str = f"VAF={m['vaf']:.1%}" if m["vaf"] is not None else "VAF=N/A"
                depth_str = f"depth={m['depth']}" if m["depth"] else ""
                print(f"    {g:>8}: {m['variant']:<20} {vaf_str}  {depth_str}")
        if h:
            ordering_str = " > ".join(
                f"{o['gene']}({o['vaf']:.1%})" for o in h["ordering"]
            )
            print(f"    Hierarchy: {ordering_str}")
            print(f"    Founder: {h['founder_gene']}, Subclonal: {h['subclonal_gene']}")
        else:
            print(f"    Hierarchy: insufficient VAF data")
        if det["has_4th_gene"]:
            print(f"    NOTE: 4th gene ({missing}) also detected:")
            for m in det["4th_gene_mutations"]:
                vaf_str = f"VAF={m['vaf']:.1%}" if m["vaf"] is not None else "VAF=N/A"
                print(f"      {missing}: {m['variant']} {vaf_str}")
        if det["multi_sample_data"]:
            print(f"    MULTI-SAMPLE ({det['n_samples']} samples):")
            for sid, sdata in det["multi_sample_data"].items():
                genes_str = ", ".join(
                    f"{g}={d['vaf']:.1%}" for g, d in sorted(sdata.items())
                )
                print(f"      {sid}: {genes_str}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5: Clonal hierarchy patterns
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("[Step 5] CLONAL HIERARCHY PATTERNS")
    print("=" * 80)

    patients_with_hierarchy = sum(1 for d in triple_details if d["hierarchy"] is not None)
    print(f"\n  Triple carriers with VAF-based hierarchy: {patients_with_hierarchy}/{total_triples}")

    print(f"\n  Founder gene (highest VAF) frequency:")
    for gene, count in founder_gene_counter.most_common():
        pct = count / patients_with_hierarchy * 100 if patients_with_hierarchy > 0 else 0
        print(f"    {gene:>8}: {count:>3} ({pct:.1f}%)")

    print(f"\n  Subclonal gene (lowest VAF) frequency:")
    for gene, count in subclonal_gene_counter.most_common():
        pct = count / patients_with_hierarchy * 100 if patients_with_hierarchy > 0 else 0
        print(f"    {gene:>8}: {count:>3} ({pct:.1f}%)")

    print(f"\n  Mean rank per gene (1=highest VAF, lower=more founder-like):")
    for gene in TARGET_GENES:
        ranks = vaf_rank_accumulator.get(gene, [])
        if ranks:
            mean_rank = statistics.mean(ranks)
            print(f"    {gene:>8}: mean rank {mean_rank:.2f} (n={len(ranks)})")
        else:
            print(f"    {gene:>8}: no data")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6: Compare to Henrik's pattern
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("[Step 6] COMPARISON TO HENRIK'S PATTERN")
    print("  Henrik: EZH2(59%) > DNMT3A(39%) > SETBP1(34%) > PTPN11(29%) > IDH2(2%)")
    print("  Henrik's ordering among 4 target genes: DNMT3A > SETBP1 > PTPN11 > IDH2")
    print("=" * 80)

    # Henrik's ranking among the 5 driver genes (EZH2 V662A reclassified Pathogenic):
    # DNMT3A (39%) = rank 1 (founder), IDH2 (2%) = rank 4 (subclone)
    index_patient_ordering = ["DNMT3A", "SETBP1", "PTPN11", "IDH2"]

    # Check how many patients share features with Henrik
    dnmt3a_founder_count = 0
    idh2_subclonal_count = 0
    similar_to_index = []

    for det in triple_details:
        h = det["hierarchy"]
        if h is None:
            continue

        genes_in_ordering = [o["gene"] for o in h["ordering"]]
        is_dnmt3a_founder = h["founder_gene"] == "DNMT3A"
        is_idh2_subclonal = h["subclonal_gene"] == "IDH2"

        if is_dnmt3a_founder:
            dnmt3a_founder_count += 1
        if is_idh2_subclonal:
            idh2_subclonal_count += 1

        # Check if ordering matches Henrik's among shared genes
        shared_genes = [g for g in index_patient_ordering if g in genes_in_ordering]
        if len(shared_genes) >= 2:
            index_sub = [g for g in index_patient_ordering if g in shared_genes]
            patient_sub = [g for g in genes_in_ordering if g in shared_genes]
            if index_sub == patient_sub:
                similar_to_index.append(det)

    print(f"\n  DNMT3A as founder (highest VAF): {dnmt3a_founder_count}/{patients_with_hierarchy}")
    print(f"  IDH2 as subclonal (lowest VAF): {idh2_subclonal_count}/{patients_with_hierarchy}")
    print(f"  Patients matching index patient's gene ordering: {len(similar_to_index)}/{patients_with_hierarchy}")

    if similar_to_index:
        print(f"\n  Patients with Henrik-like ordering:")
        for det in similar_to_index:
            h = det["hierarchy"]
            ordering_str = " > ".join(
                f"{o['gene']}({o['vaf']:.1%})" for o in h["ordering"]
            )
            print(f"    {det['patient_id']}: {ordering_str}")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 7: Multi-sample patients (VAF changes over time)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("[Step 7] MULTI-SAMPLE PATIENTS (VAF changes across samples)")
    print("=" * 80)

    if multi_sample_patients:
        print(f"\n  {len(multi_sample_patients)} patient(s) with multiple samples:")
        for ms in multi_sample_patients:
            print(f"\n  Patient {ms['patient_id']} ({ms['combo']}):")
            for sid, sdata in sorted(ms["samples"].items()):
                genes_str = ", ".join(
                    f"{g}={d['vaf']:.1%} (depth={d['depth']})"
                    for g, d in sorted(sdata.items())
                )
                print(f"    {sid}: {genes_str}")

            # Check for VAF changes
            all_sids = sorted(ms["samples"].keys())
            if len(all_sids) >= 2:
                for g in TARGET_GENES:
                    vafs_across = []
                    for sid in all_sids:
                        if g in ms["samples"].get(sid, {}):
                            vafs_across.append((sid, ms["samples"][sid][g]["vaf"]))
                    if len(vafs_across) >= 2:
                        delta = vafs_across[-1][1] - vafs_across[0][1]
                        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "stable"
                        print(f"    {g}: {vafs_across[0][1]:.1%} -> {vafs_across[-1][1]:.1%} ({direction}, delta={delta:+.1%})")
    else:
        print(f"\n  No multi-sample triple carriers found in GENIE.")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 8: VAF distributions - co-occurrence vs solo
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("[Step 8] VAF DISTRIBUTIONS: CO-OCCURRENCE vs SOLO")
    print("=" * 80)

    # Build sets: triple carriers, pairwise carriers, solo carriers
    triple_carrier_pids = set(all_triple_carriers.keys())

    # Pairwise carriers (2 of 4 genes, but NOT 3 or 4)
    pairwise_carrier_pids = set()
    for combo in combinations(TARGET_GENES, 2):
        pair_carriers = gene_carriers[combo[0]] & gene_carriers[combo[1]]
        pairwise_carrier_pids |= pair_carriers
    pairwise_carrier_pids -= triple_carrier_pids
    pairwise_carrier_pids -= quad_carriers

    # Solo carriers (exactly 1 of 4 target genes)
    solo_carrier_pids = set()
    for g in TARGET_GENES:
        for pid in gene_carriers[g]:
            target_gene_count = sum(1 for tg in TARGET_GENES if pid in gene_carriers[tg])
            if target_gene_count == 1:
                solo_carrier_pids.add(pid)

    print(f"\n  Carrier categories:")
    print(f"    Solo (1 of 4 genes): {len(solo_carrier_pids):,}")
    print(f"    Pairwise (2 of 4):   {len(pairwise_carrier_pids):,}")
    print(f"    Triple (3 of 4):     {len(triple_carrier_pids):,}")
    print(f"    Quadruple (4 of 4):  {len(quad_carriers):,}")

    vaf_comparison = {}
    for gene in TARGET_GENES:
        solo_vafs = []
        pair_vafs = []
        triple_vafs = []

        for vaf, depth, pid, variant in all_gene_vafs[gene]:
            if pid in solo_carrier_pids:
                solo_vafs.append(vaf)
            elif pid in pairwise_carrier_pids:
                pair_vafs.append(vaf)
            elif pid in triple_carrier_pids:
                triple_vafs.append(vaf)

        gene_comparison = {}
        for label, vafs in [("solo", solo_vafs), ("pairwise", pair_vafs), ("triple", triple_vafs)]:
            if vafs:
                vafs_sorted = sorted(vafs)
                gene_comparison[label] = {
                    "n": len(vafs),
                    "mean": round(statistics.mean(vafs), 4),
                    "median": round(statistics.median(vafs), 4),
                    "q25": round(percentile(vafs_sorted, 25), 4),
                    "q75": round(percentile(vafs_sorted, 75), 4),
                    "min": round(min(vafs), 4),
                    "max": round(max(vafs), 4),
                }
            else:
                gene_comparison[label] = {"n": 0}

        vaf_comparison[gene] = gene_comparison

        print(f"\n  {gene}:")
        for label in ["solo", "pairwise", "triple"]:
            d = gene_comparison[label]
            if d["n"] > 0:
                print(f"    {label:>10}: n={d['n']:>5}, mean={d['mean']:.3f}, "
                      f"median={d['median']:.3f}, IQR={d['q25']:.3f}-{d['q75']:.3f}")
            else:
                print(f"    {label:>10}: n=0")

    # ──────────────────────────────────────────────────────────────────────
    # BUILD OUTPUT JSON
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("Writing results...")
    print("=" * 80)

    output = {
        "analysis": "Clonal Architecture of Triple Carriers - GENIE v19.0",
        "analysis_date": "2026-03-19",
        "data_source": "AACR GENIE v19.0 MAF + clinical sample files",
        "methodology": {
            "myeloid_filter": "ONCOTREE_CODE prefix matching (AML, MDS, MPN, CMML, JMML, CML, TMN, SM, CEL, HES) plus exact codes (AMML, APL, APMF, AUL, MLADS, MS)",
            "protein_altering_filter": list(PATHOGENIC_VAR_CLASSES),
            "vaf_calculation": "t_alt_count / t_depth, minimum depth 50x",
            "hierarchy_determination": "Highest VAF = founder/dominant clone, lowest VAF = subclonal",
        },
        "summary": {
            "total_myeloid_samples": len(myeloid_samples),
            "gene_carriers": {g: len(gene_carriers[g]) for g in TARGET_GENES},
            "triple_combo_counts": combo_counts,
            "quadruple_count": len(quad_carriers),
            "total_unique_triple_carriers": total_triples,
            "triple_carriers_with_vaf_hierarchy": patients_with_hierarchy,
        },
        "clonal_hierarchy_patterns": {
            "founder_gene_frequency": dict(founder_gene_counter.most_common()),
            "subclonal_gene_frequency": dict(subclonal_gene_counter.most_common()),
            "mean_rank_per_gene": {
                gene: round(statistics.mean(vaf_rank_accumulator[gene]), 3)
                if vaf_rank_accumulator[gene] else None
                for gene in TARGET_GENES
            },
            "n_ranked_per_gene": {
                gene: len(vaf_rank_accumulator[gene]) for gene in TARGET_GENES
            },
        },
        "index_patient_comparison": {
            "index_patient_ordering": "DNMT3A(39%) > SETBP1(34%) > PTPN11(29%) > IDH2(2%)",
            "index_patient_founder": "DNMT3A",
            "index_patient_subclone": "IDH2",
            "dnmt3a_founder_count": dnmt3a_founder_count,
            "idh2_subclonal_count": idh2_subclonal_count,
            "patients_with_hierarchy": patients_with_hierarchy,
            "patients_matching_index_patient_ordering": len(similar_to_index),
            "matching_patients": [
                {
                    "patient_id": d["patient_id"],
                    "ordering": [
                        {"gene": o["gene"], "vaf": o["vaf"]}
                        for o in d["hierarchy"]["ordering"]
                    ],
                }
                for d in similar_to_index
            ],
        },
        "multi_sample_patients": multi_sample_patients,
        "vaf_distributions_by_context": vaf_comparison,
        "triple_carrier_details": [
            {
                "patient_id": d["patient_id"],
                "triple_combo": d["triple_combo"],
                "missing_gene": d["missing_gene"],
                "n_samples": d["n_samples"],
                "mutations": d["mutations"],
                "hierarchy": d["hierarchy"],
                "has_4th_gene": d["has_4th_gene"],
                "4th_gene_mutations": d.get("4th_gene_mutations"),
                "multi_sample_data": d["multi_sample_data"],
            }
            for d in triple_details
        ],
    }

    # Quadruple carrier details
    if quad_carriers:
        quad_details = []
        for pid in sorted(quad_carriers):
            qd = {"patient_id": pid, "mutations": {}}
            for g in TARGET_GENES:
                gene_muts = patient_mutations[pid][g]
                qd["mutations"][g] = [
                    {
                        "variant": m["variant"],
                        "vaf": round(m["vaf"], 4) if m["vaf"] is not None else None,
                        "depth": m["depth"],
                        "sample_id": m["sample_id"],
                    }
                    for m in gene_muts
                ]
            quad_details.append(qd)
        output["quadruple_carrier_details"] = quad_details

    output_path = RESULTS_DIR / "clonal_architecture.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
