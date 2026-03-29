#!/usr/bin/env python3
"""
PTPN11 E76Q Deep Investigation -- GENIE v19.0

Finds all patients with PTPN11 p.E76Q in myeloid samples, extracts their
complete mutation profiles, cross-references clinical data, and compares
against E76K/E76G/E76A variant carriers.

Key question: Are any of the 8 E76Q myeloid patients also carrying
DNMT3A, IDH2, or SETBP1 mutations?
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
GENIE_RAW = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"

# -------------------------------------------------------------------
# Filter definitions
# -------------------------------------------------------------------

PATHOGENIC_VAR_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    "Translation_Start_Site",
}

# OncoTree codes for myeloid neoplasms -- prefixes that qualify
MYELOID_CODE_PREFIXES = (
    "AML", "MDS", "MPN", "CMML", "JMML", "CML", "TMN", "SM", "CEL",
    "HES", "AMML", "APL", "APMF", "AUL", "MLADS", "MS",
)

# Additional exact codes that don't start with the above prefixes
MYELOID_EXACT_CODES = {
    "BPDCN", "MPAL", "ALAL", "ET", "PV", "PMF", "ACML",
    "RCMD", "RARS", "RA", "RAEB", "RAEB1", "RAEB2", "RCUD",
}

# PTPN11 E76 variants to compare
E76_VARIANTS = {"E76Q", "E76K", "E76G", "E76A"}

# Henrik's co-mutation genes of interest
CO_MUTATION_GENES = {"DNMT3A", "IDH2", "SETBP1"}


def is_myeloid_code(oncotree_code: str) -> bool:
    """Check if an OncoTree code represents a myeloid neoplasm."""
    if not oncotree_code:
        return False
    code = oncotree_code.strip().upper()
    if code in MYELOID_EXACT_CODES:
        return True
    return any(code.startswith(prefix) for prefix in MYELOID_CODE_PREFIXES)


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


def load_clinical_patients() -> dict:
    """Load patient-level clinical data (sex, race, vital status)."""
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


def load_clinical_samples() -> tuple:
    """
    Load sample-level clinical data.
    Returns:
        samples: dict mapping sample_id -> sample info dict
        sid_to_pid: dict mapping sample_id -> patient_id
        pid_to_sids: dict mapping patient_id -> set of sample_ids
    """
    samples = {}
    sid_to_pid = {}
    pid_to_sids = defaultdict(set)
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
            pid = row.get("PATIENT_ID", "")
            if sid:
                samples[sid] = row
                sid_to_pid[sid] = pid
                pid_to_sids[pid].add(sid)
    return samples, sid_to_pid, pid_to_sids


def get_myeloid_sample_ids(samples: dict) -> dict:
    """Return dict mapping sample_id -> sample_info for myeloid samples."""
    myeloid = {}
    for sid, info in samples.items():
        code = info.get("ONCOTREE_CODE", "")
        if is_myeloid_code(code):
            myeloid[sid] = info
    return myeloid


def scan_maf_file(myeloid_sids: set, sid_to_pid: dict, pid_to_sids: dict) -> tuple:
    """
    Two-pass scan of the MAF file.

    Pass 1: Find all PTPN11 E76 mutations in myeloid samples.
    Pass 2: Collect all mutations for those patients' myeloid samples.

    Returns:
        e76_hits: dict mapping variant -> {patient_id -> {sample_id, HGVSp_Short, ...}}
        patient_mutations: dict mapping patient_id -> list of mutation dicts
            (deduplicated per sample -- only the E76-carrying myeloid sample)
    """
    maf_path = GENIE_RAW / "data_mutations_extended.txt"

    # ---- Pass 1: find PTPN11 E76 mutations ----
    print("  Pass 1: Scanning for PTPN11 E76 variants in myeloid samples...")
    e76_hits = defaultdict(dict)  # variant -> {patient_id -> hit_info}
    e76_patient_myeloid_sids = defaultdict(set)  # patient_id -> set of myeloid sample_ids with E76

    row_count = 0
    with open(maf_path) as f:
        header = f.readline().strip().split("\t")
        col = {name: i for i, name in enumerate(header)}

        hugo_i = col["Hugo_Symbol"]
        hgvsp_i = col["HGVSp_Short"]
        varclass_i = col["Variant_Classification"]
        sample_i = col["Tumor_Sample_Barcode"]

        for line in f:
            row_count += 1
            if row_count % 500_000 == 0:
                print(f"    ...{row_count:,} rows scanned")

            fields = line.rstrip("\n").split("\t")
            sample_id = fields[sample_i]

            if sample_id not in myeloid_sids:
                continue

            var_class = fields[varclass_i]
            if var_class not in PATHOGENIC_VAR_CLASSES:
                continue

            gene = fields[hugo_i]
            if gene != "PTPN11":
                continue

            hgvsp = fields[hgvsp_i]
            variant_short = extract_variant_short(hgvsp)
            if variant_short not in E76_VARIANTS:
                continue

            pid = sid_to_pid.get(sample_id, "")
            if not pid:
                continue

            e76_hits[variant_short][pid] = {
                "sample_id": sample_id,
                "HGVSp_Short": hgvsp,
                "Variant_Classification": var_class,
            }
            e76_patient_myeloid_sids[pid].add(sample_id)

    print(f"  Pass 1 complete: {row_count:,} total rows scanned")
    for var in sorted(E76_VARIANTS):
        n = len(e76_hits.get(var, {}))
        print(f"    PTPN11 {var}: {n} myeloid patients")

    # Build the set of all myeloid sample IDs belonging to E76 patients.
    # For each patient, include ALL of their myeloid samples (not just the one
    # where E76 was found), so we capture mutations from serial samples.
    all_e76_pids = set(e76_patient_myeloid_sids.keys())
    e76_target_sids = set()
    for pid in all_e76_pids:
        for sid in pid_to_sids.get(pid, set()):
            if sid in myeloid_sids:
                e76_target_sids.add(sid)

    print(f"\n  E76 patients: {len(all_e76_pids)}")
    print(f"  Their myeloid samples: {len(e76_target_sids)}")

    # ---- Pass 2: collect all mutations for E76 patients' myeloid samples ----
    print(f"\n  Pass 2: Collecting all mutations for E76 patients...")
    # We store per (patient, sample) to allow deduplication
    patient_sample_mutations = defaultdict(lambda: defaultdict(list))

    row_count2 = 0
    with open(maf_path) as f:
        header = f.readline().strip().split("\t")
        col = {name: i for i, name in enumerate(header)}

        hugo_i = col["Hugo_Symbol"]
        hgvsp_i = col["HGVSp_Short"]
        varclass_i = col["Variant_Classification"]
        sample_i = col["Tumor_Sample_Barcode"]
        t_alt_i = col.get("t_alt_count")
        t_ref_i = col.get("t_ref_count")
        t_depth_i = col.get("t_depth")
        center_i = col.get("Center")

        for line in f:
            row_count2 += 1
            if row_count2 % 500_000 == 0:
                print(f"    ...{row_count2:,} rows scanned")

            fields = line.rstrip("\n").split("\t")
            sample_id = fields[sample_i]

            if sample_id not in e76_target_sids:
                continue

            var_class = fields[varclass_i]
            if var_class not in PATHOGENIC_VAR_CLASSES:
                continue

            pid = sid_to_pid.get(sample_id, "")
            gene = fields[hugo_i]
            hgvsp = fields[hgvsp_i] if hgvsp_i is not None else ""
            t_alt = fields[t_alt_i] if t_alt_i is not None else ""
            t_ref = fields[t_ref_i] if t_ref_i is not None else ""
            t_depth = fields[t_depth_i] if t_depth_i is not None else ""
            center = fields[center_i] if center_i is not None else ""

            vaf = compute_vaf(t_alt, t_depth, t_ref)

            patient_sample_mutations[pid][sample_id].append({
                "gene": gene,
                "variant": extract_variant_short(hgvsp),
                "HGVSp_Short": hgvsp,
                "Variant_Classification": var_class,
                "VAF_pct": vaf,
                "t_alt_count": t_alt,
                "t_ref_count": t_ref,
                "t_depth": t_depth,
                "sample_id": sample_id,
                "center": center,
            })

    print(f"  Pass 2 complete: {row_count2:,} rows scanned")

    # For the main profile display, use the myeloid sample that carried the E76 variant.
    # If a patient has multiple myeloid samples, pick the one with the E76 hit.
    # For co-mutation checking, consider all myeloid samples for that patient.
    patient_mutations = {}
    for pid in all_e76_pids:
        # Combine unique gene+variant pairs across all myeloid samples
        seen = set()
        combined = []
        # Priority: start with the E76-carrying sample, then add from others
        e76_sids = e76_patient_myeloid_sids[pid]
        other_sids = set(patient_sample_mutations[pid].keys()) - e76_sids

        for sid in list(e76_sids) + list(other_sids):
            for m in patient_sample_mutations[pid][sid]:
                key = (m["gene"], m["variant"], m["Variant_Classification"])
                if key not in seen:
                    seen.add(key)
                    combined.append(m)

        patient_mutations[pid] = combined

    return e76_hits, patient_mutations


def main():
    print("=" * 80)
    print("PTPN11 E76Q DEEP INVESTIGATION -- GENIE v19.0")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load clinical data
    # ------------------------------------------------------------------
    print("\nLoading clinical data...")
    patients_clinical = load_clinical_patients()
    samples_clinical, sid_to_pid, pid_to_sids = load_clinical_samples()
    print(f"  Loaded {len(patients_clinical):,} patients, {len(samples_clinical):,} samples")

    myeloid_samples = get_myeloid_sample_ids(samples_clinical)
    myeloid_sids = set(myeloid_samples.keys())
    print(f"  Myeloid samples: {len(myeloid_sids):,}")

    # ------------------------------------------------------------------
    # Scan MAF
    # ------------------------------------------------------------------
    print("\nScanning MAF file...")
    e76_hits, patient_mutations = scan_maf_file(myeloid_sids, sid_to_pid, pid_to_sids)

    # ------------------------------------------------------------------
    # Build detailed patient profiles
    # ------------------------------------------------------------------
    results = {
        "analysis": "PTPN11 E76Q Deep Investigation",
        "dataset": "GENIE v19.0",
        "filter": "Myeloid OncoTree codes, protein-altering variants only",
        "e76_variant_counts": {},
        "e76q_patients": [],
        "e76_comparison": {},
        "co_mutation_check": {},
    }

    for var in sorted(E76_VARIANTS):
        results["e76_variant_counts"][var] = len(e76_hits.get(var, {}))

    # ------------------------------------------------------------------
    # E76Q patient deep profiles
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DETAILED E76Q PATIENT PROFILES")
    print("=" * 80)

    e76q_patients = e76_hits.get("E76Q", {})
    co_mutation_found = {gene: [] for gene in CO_MUTATION_GENES}

    for i, (pid, hit_info) in enumerate(sorted(e76q_patients.items()), 1):
        sample_id = hit_info["sample_id"]
        sample_info = samples_clinical.get(sample_id, {})
        patient_info = patients_clinical.get(pid, {})

        # List all myeloid samples for this patient
        all_myeloid_for_patient = [
            sid for sid in pid_to_sids.get(pid, set()) if sid in myeloid_sids
        ]

        oncotree_code = sample_info.get("ONCOTREE_CODE", "Unknown")
        cancer_type = sample_info.get("CANCER_TYPE_DETAILED", sample_info.get("CANCER_TYPE", "Unknown"))
        age = sample_info.get("AGE_AT_SEQ_REPORT", "Unknown")
        panel = sample_info.get("SEQ_ASSAY_ID", "Unknown")
        center = sample_info.get("CENTER", "Unknown")
        sample_type = sample_info.get("SAMPLE_TYPE", "Unknown")
        sex = patient_info.get("SEX", "Unknown")
        race = patient_info.get("PRIMARY_RACE", "Unknown")
        vital_status = patient_info.get("DEAD", "Unknown")

        mutations = patient_mutations.get(pid, [])
        mutations_sorted = sorted(mutations, key=lambda m: (m["VAF_pct"] or 0), reverse=True)

        # Check for co-mutations
        patient_genes = {m["gene"] for m in mutations}
        for gene in CO_MUTATION_GENES:
            if gene in patient_genes:
                gene_muts = [m for m in mutations if m["gene"] == gene]
                co_mutation_found[gene].append({
                    "patient_id": pid,
                    "mutations": gene_muts,
                })

        profile = {
            "patient_id": pid,
            "sample_id": sample_id,
            "all_myeloid_samples": sorted(all_myeloid_for_patient),
            "sex": sex,
            "race": race,
            "age_at_seq": age,
            "oncotree_code": oncotree_code,
            "cancer_type_detailed": cancer_type,
            "sample_type": sample_type,
            "center": center,
            "seq_panel": panel,
            "vital_status": vital_status,
            "total_unique_mutations": len(mutations),
            "mutations": mutations_sorted,
            "co_mutations_with_index_genes": {
                gene: gene in patient_genes for gene in CO_MUTATION_GENES
            },
        }
        results["e76q_patients"].append(profile)

        # Print
        print(f"\n--- Patient {i}/{len(e76q_patients)}: {pid} ---")
        print(f"  Sex:          {sex}")
        print(f"  Race:         {race}")
        print(f"  Age at seq:   {age}")
        print(f"  Diagnosis:    {oncotree_code} ({cancer_type})")
        print(f"  Sample type:  {sample_type}")
        print(f"  Center:       {center}")
        print(f"  Panel:        {panel}")
        print(f"  Vital status: {'Deceased' if vital_status == 'TRUE' else 'Alive' if vital_status == 'FALSE' else vital_status}")
        print(f"  Myeloid samples: {len(all_myeloid_for_patient)}")
        print(f"  Total unique protein-altering mutations: {len(mutations)}")

        # Co-mutations with Henrik's genes
        has_co = []
        for gene in sorted(CO_MUTATION_GENES):
            if gene in patient_genes:
                gene_muts = [m for m in mutations if m["gene"] == gene]
                for gm in gene_muts:
                    vaf_str = f" (VAF {gm['VAF_pct']}%)" if gm['VAF_pct'] else ""
                    has_co.append(f"{gene} {gm['variant']}{vaf_str}")
        if has_co:
            print(f"  ** CO-MUTATIONS WITH HENRIK'S GENES: {', '.join(has_co)} **")
        else:
            print(f"  Co-mutations with DNMT3A/IDH2/SETBP1: NONE")

        print(f"\n  Full mutation profile:")
        print(f"  {'Gene':<15} {'Variant':<25} {'Classification':<25} {'VAF%':<8} {'Sample'}")
        print(f"  {'-'*15} {'-'*25} {'-'*25} {'-'*8} {'-'*20}")
        for m in mutations_sorted:
            vaf_str = f"{m['VAF_pct']}%" if m['VAF_pct'] is not None else "N/A"
            marker = ""
            if m["gene"] == "PTPN11":
                marker = " ** PTPN11"
            elif m["gene"] in CO_MUTATION_GENES:
                marker = " ** Henrik gene"
            # Truncate sample_id for display
            sid_short = m["sample_id"].split("-")[-1] if m["sample_id"] else ""
            print(f"  {m['gene']:<15} {m['variant']:<25} {m['Variant_Classification']:<25} {vaf_str:<8} {sid_short}{marker}")

    # ------------------------------------------------------------------
    # Co-mutation summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CO-MUTATION CHECK: Do any E76Q patients carry DNMT3A, IDH2, or SETBP1?")
    print("=" * 80)

    results["co_mutation_check"] = {}
    for gene in sorted(CO_MUTATION_GENES):
        carriers = co_mutation_found[gene]
        results["co_mutation_check"][gene] = {
            "carriers": len(carriers),
            "patient_ids": [c["patient_id"] for c in carriers],
            "details": [
                {
                    "patient_id": c["patient_id"],
                    "variants": [
                        {"variant": m["variant"], "VAF_pct": m["VAF_pct"]}
                        for m in c["mutations"]
                    ],
                }
                for c in carriers
            ],
        }
        if carriers:
            print(f"\n  {gene}: {len(carriers)} E76Q patient(s) also carry this gene")
            for c in carriers:
                for m in c["mutations"]:
                    vaf_str = f" (VAF {m['VAF_pct']}%)" if m['VAF_pct'] else ""
                    print(f"    - {c['patient_id']}: {gene} {m['variant']}{vaf_str}")
        else:
            print(f"\n  {gene}: 0 E76Q patients carry this gene")

    # Check for any patient with PTPN11 E76Q + DNMT3A + IDH2 + SETBP1
    quad_carriers = []
    for p in results["e76q_patients"]:
        co = p["co_mutations_with_index_genes"]
        if co.get("DNMT3A") and co.get("IDH2") and co.get("SETBP1"):
            quad_carriers.append(p["patient_id"])

    triple_any = []
    for p in results["e76q_patients"]:
        co = p["co_mutations_with_index_genes"]
        count = sum(1 for g in CO_MUTATION_GENES if co.get(g))
        if count >= 2:
            triple_any.append((p["patient_id"], count))

    print(f"\n  Quadruple carriers (E76Q + DNMT3A + IDH2 + SETBP1): {len(quad_carriers)}")
    if quad_carriers:
        for qc in quad_carriers:
            print(f"    - {qc}")
    print(f"  Patients with E76Q + 2+ of {{DNMT3A, IDH2, SETBP1}}: {len(triple_any)}")
    if triple_any:
        for pid, count in triple_any:
            print(f"    - {pid} ({count} of 3 genes)")

    # ------------------------------------------------------------------
    # Comparison: E76Q vs E76K vs E76G vs E76A
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON: E76Q vs E76K vs E76G vs E76A")
    print("=" * 80)

    comparison = {}
    for var in sorted(E76_VARIANTS):
        var_patients = e76_hits.get(var, {})
        if not var_patients:
            comparison[var] = {"n": 0}
            continue

        ages = []
        sexes = Counter()
        centers = Counter()
        oncotree_codes = Counter()
        cancer_types = Counter()
        vital_counts = Counter()
        co_gene_counts = {g: 0 for g in CO_MUTATION_GENES}
        mutation_counts = []
        gene_freq = Counter()

        for pid, hit_info in var_patients.items():
            sid = hit_info["sample_id"]
            sample_info = samples_clinical.get(sid, {})
            patient_info = patients_clinical.get(pid, {})

            age_str = sample_info.get("AGE_AT_SEQ_REPORT", "")
            try:
                age_val = int(age_str)
                ages.append(age_val)
            except (ValueError, TypeError):
                pass

            sex = patient_info.get("SEX", "Unknown")
            sexes[sex] += 1

            center = sample_info.get("CENTER", "Unknown")
            centers[center] += 1

            oncotree = sample_info.get("ONCOTREE_CODE", "Unknown")
            oncotree_codes[oncotree] += 1

            cancer_type = sample_info.get("CANCER_TYPE_DETAILED",
                                          sample_info.get("CANCER_TYPE", "Unknown"))
            cancer_types[cancer_type] += 1

            dead = patient_info.get("DEAD", "Unknown")
            vital_counts[dead] += 1

            muts = patient_mutations.get(pid, [])
            mutation_counts.append(len(muts))
            patient_genes = {m["gene"] for m in muts}
            for g in CO_MUTATION_GENES:
                if g in patient_genes:
                    co_gene_counts[g] += 1
            for m in muts:
                gene_freq[m["gene"]] += 1

        n = len(var_patients)
        avg_age = round(sum(ages) / len(ages), 1) if ages else None
        median_age = sorted(ages)[len(ages) // 2] if ages else None
        age_range = (min(ages), max(ages)) if ages else None
        avg_muts = round(sum(mutation_counts) / len(mutation_counts), 1) if mutation_counts else None

        comp_entry = {
            "n": n,
            "sex_distribution": dict(sexes),
            "vital_status": dict(vital_counts),
            "age_mean": avg_age,
            "age_median": median_age,
            "age_range": list(age_range) if age_range else None,
            "ages": sorted(ages),
            "centers": dict(centers.most_common()),
            "oncotree_codes": dict(oncotree_codes.most_common()),
            "cancer_types": dict(cancer_types.most_common()),
            "avg_unique_mutations_per_patient": avg_muts,
            "mutation_counts": sorted(mutation_counts),
            "co_mutation_rates": {
                g: {"count": co_gene_counts[g], "rate_pct": round(co_gene_counts[g] / n * 100, 1)}
                for g in CO_MUTATION_GENES
            },
            "top_co_mutated_genes": dict(gene_freq.most_common(15)),
        }
        comparison[var] = comp_entry

        print(f"\n  PTPN11 {var} (n={n})")
        print(f"    Sex:       {dict(sexes)}")
        print(f"    Age:       mean={avg_age}, median={median_age}, range={age_range}")
        print(f"    Ages:      {sorted(ages)}")
        print(f"    Vital:     {dict(vital_counts)}")
        print(f"    Centers:   {dict(centers.most_common(5))}")
        print(f"    Diagnoses: {dict(oncotree_codes.most_common(5))}")
        print(f"    Mut counts per patient: {sorted(mutation_counts)}")
        print(f"    Avg unique mutations/patient: {avg_muts}")
        print(f"    DNMT3A co-mut: {co_gene_counts['DNMT3A']}/{n} ({round(co_gene_counts['DNMT3A']/n*100,1)}%)")
        print(f"    IDH2 co-mut:   {co_gene_counts['IDH2']}/{n} ({round(co_gene_counts['IDH2']/n*100,1)}%)")
        print(f"    SETBP1 co-mut: {co_gene_counts['SETBP1']}/{n} ({round(co_gene_counts['SETBP1']/n*100,1)}%)")
        print(f"    Top co-mutated genes: {dict(gene_freq.most_common(10))}")

    results["e76_comparison"] = comparison

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: E76 VARIANT COMPARISON")
    print("=" * 80)

    hdr = f"  {'Variant':<10} {'N':>4} {'M/F':>10} {'Age med':>8} {'Age range':>12} {'DNMT3A':>8} {'IDH2':>6} {'SETBP1':>8} {'Avg muts':>9}"
    print(hdr)
    print(f"  {'-'*10} {'-'*4} {'-'*10} {'-'*8} {'-'*12} {'-'*8} {'-'*6} {'-'*8} {'-'*9}")

    for var in ["E76Q", "E76K", "E76G", "E76A"]:
        c = comparison.get(var, {})
        n = c.get("n", 0)
        if n == 0:
            print(f"  {var:<10} {0:>4}")
            continue

        sex_d = c.get("sex_distribution", {})
        males = sex_d.get("Male", 0)
        females = sex_d.get("Female", 0)
        mf = f"{males}/{females}"

        age_med = c.get("age_median")
        age_med_str = str(age_med) if age_med is not None else "N/A"

        age_rng = c.get("age_range")
        age_rng_str = f"{age_rng[0]}-{age_rng[1]}" if age_rng else "N/A"

        dnmt3a = c.get("co_mutation_rates", {}).get("DNMT3A", {}).get("count", 0)
        idh2 = c.get("co_mutation_rates", {}).get("IDH2", {}).get("count", 0)
        setbp1 = c.get("co_mutation_rates", {}).get("SETBP1", {}).get("count", 0)

        avg_m = c.get("avg_unique_mutations_per_patient")
        avg_m_str = str(avg_m) if avg_m is not None else "N/A"

        print(f"  {var:<10} {n:>4} {mf:>10} {age_med_str:>8} {age_rng_str:>12} {dnmt3a:>8} {idh2:>6} {setbp1:>8} {avg_m_str:>9}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ptpn11_e76q_investigation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
