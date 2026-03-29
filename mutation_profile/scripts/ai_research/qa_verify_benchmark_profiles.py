#!/usr/bin/env python3
"""
QA verification: benchmark profiles vs GENIE v19.0 raw data.

Verifies all 40 benchmark profiles (batch1 + batch2) against the raw
GENIE mutations file. Checks:
  1. Patient ID (Tumor_Sample_Barcode) exists in GENIE
  2. Each mutation's gene, HGVSp, chromosome, start_position match
  3. VAF computation is correct: t_alt_count / (t_ref_count + t_alt_count)
  4. No coding mutations in target genes were missed
  5. No patient appears in both batches
  6. OncoTree code matches clinical data
"""

import json
import csv
import math
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

BASE = Path(__file__).resolve().parents[2]  # mutation_profile/
RAW = BASE / "data" / "genie" / "raw"
RESULTS = BASE / "results" / "ai_research" / "benchmark"

BATCH1_PATH = RESULTS / "benchmark_profiles.json"
BATCH2_PATH = RESULTS / "benchmark_profiles_batch2.json"
MUTATIONS_PATH = RAW / "data_mutations_extended.txt"
CLINICAL_PATH = RAW / "data_clinical_sample.txt"
OUTPUT_PATH = RESULTS / "qa_profile_verification.md"

# Noncoding variant classifications to exclude
NONCODING = frozenset([
    "3'Flank", "3'UTR", "5'Flank", "5'UTR",
    "IGR", "Intron", "RNA", "Silent", "Splice_Region"
])

TARGET_GENES = frozenset([
    "ASXL1", "TET2", "SRSF2", "SF3B1", "RUNX1", "TP53", "FLT3", "NPM1",
    "NRAS", "KRAS", "CBL", "EZH2", "U2AF1", "STAG2", "BCOR", "BCORL1",
    "DDX41", "DNMT3A", "IDH1", "IDH2", "PTPN11", "JAK2", "CALR", "MPL",
    "PHF6", "WT1", "CEBPA", "GATA2", "ZRSR2", "RAD21", "SMC1A", "SMC3",
    "SETBP1", "CSF3R"
])

# Myeloid OncoTree codes (from profile metadata)
MYELOID_CODES = frozenset([
    "AML", "AML-MRC", "AML-NOS", "APL", "CEL", "CMML", "CMML-1", "CMML-2",
    "CNL", "ET", "JMML", "MDS", "MDS-EB1", "MDS-EB2", "MDS-MLD", "MDS-RS",
    "MDS-SLD", "MDS-U", "MDS/MPN", "MDSMPNU", "MPN", "MPN-U", "PMF", "PV",
    "RAEB", "RAEB-T", "RARS", "RCMD", "SM", "aCML"
])

VAF_TOLERANCE = 0.0005  # Allow rounding to 4 decimal places


def load_profiles(path):
    """Load benchmark profiles JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["profiles"], data.get("metadata", {})


def load_genie_mutations(sample_ids):
    """Load GENIE mutations only for specified sample IDs (memory efficient)."""
    sample_set = set(sample_ids)
    mutations = defaultdict(list)

    print(f"  Loading GENIE mutations for {len(sample_set)} samples...")
    with open(MUTATIONS_PATH, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            barcode = row["Tumor_Sample_Barcode"]
            if barcode in sample_set:
                mutations[barcode].append(row)

    print(f"  Found mutations for {len(mutations)} samples")
    return mutations


def load_clinical_data(sample_ids):
    """Load clinical data for specified sample IDs."""
    sample_set = set(sample_ids)
    clinical = {}

    with open(CLINICAL_PATH, "r") as f:
        # Skip comment lines (start with #)
        for line in f:
            if not line.startswith("#"):
                header = line.strip().split("\t")
                break

        reader = csv.DictReader(f, fieldnames=header, delimiter="\t")
        for row in reader:
            sid = row.get("SAMPLE_ID", "")
            if sid in sample_set:
                clinical[sid] = row

    return clinical


def parse_hgvsp_short(hgvsp_raw):
    """Extract short protein change from HGVSp_Short column (e.g., p.R882H -> R882H)."""
    if not hgvsp_raw:
        return ""
    s = hgvsp_raw.strip()
    if s.startswith("p."):
        s = s[2:]
    return s


def compute_vaf(t_ref_str, t_alt_str):
    """Compute VAF from t_ref_count and t_alt_count strings."""
    try:
        t_ref = int(t_ref_str) if t_ref_str else None
        t_alt = int(t_alt_str) if t_alt_str else None
    except (ValueError, TypeError):
        return None, None, None

    if t_ref is None or t_alt is None:
        return None, None, None

    total = t_ref + t_alt
    if total == 0:
        return 0.0, t_ref, t_alt

    return t_alt / total, t_ref, t_alt


def verify_profile(profile, genie_muts, clinical_data):
    """Verify a single profile against GENIE raw data. Returns (errors, warnings, details)."""
    pid = profile["patient_id"]
    errors = []
    warnings = []
    details = []

    # --- Check 1: Patient exists in GENIE ---
    if pid not in genie_muts:
        errors.append(f"SAMPLE NOT FOUND in GENIE raw data: {pid}")
        return errors, warnings, details

    raw_rows = genie_muts[pid]
    details.append(f"Total raw GENIE rows for sample: {len(raw_rows)}")

    # --- Check 2: OncoTree code ---
    if pid in clinical_data:
        genie_onco = clinical_data[pid].get("ONCOTREE_CODE", "")
        profile_onco = profile.get("oncotree_code", "")
        if genie_onco != profile_onco:
            errors.append(
                f"OncoTree MISMATCH: profile={profile_onco}, GENIE={genie_onco}"
            )
        else:
            details.append(f"OncoTree code OK: {genie_onco}")

        if genie_onco not in MYELOID_CODES:
            warnings.append(f"OncoTree code '{genie_onco}' not in standard myeloid set")
    else:
        warnings.append(f"Sample not found in clinical data file")

    # --- Build GENIE lookup keyed by (gene, chromosome, start_position) ---
    # Also index by (gene, hgvsp_short) for additional matching
    genie_by_pos = defaultdict(list)
    genie_by_hgvsp = defaultdict(list)

    for row in raw_rows:
        gene = row["Hugo_Symbol"]
        chrom = row["Chromosome"]
        pos = row["Start_Position"]
        hgvsp_short = parse_hgvsp_short(row.get("HGVSp_Short", ""))
        var_class = row["Variant_Classification"]

        genie_by_pos[(gene, chrom, pos)].append(row)
        if hgvsp_short:
            genie_by_hgvsp[(gene, hgvsp_short)].append(row)

    # --- Check 3: Each profile mutation matches GENIE ---
    matched_genie_keys = set()

    for mut in profile["mutations"]:
        gene = mut["gene"]
        hgvsp = mut["hgvsp"]
        chrom = str(mut["chromosome"])
        pos = str(mut["start_position"])
        expected_vaf = mut["t_vaf"]
        var_class = mut["variant_classification"]
        ref_allele = mut["ref_allele"]
        alt_allele = mut["alt_allele"]

        key_pos = (gene, chrom, pos)
        key_hgvsp = (gene, hgvsp)

        # Try position match first
        matches = genie_by_pos.get(key_pos, [])
        # Fall back to HGVSp match
        if not matches:
            matches = genie_by_hgvsp.get(key_hgvsp, [])

        if not matches:
            errors.append(
                f"MUTATION NOT FOUND: {gene} {hgvsp} at chr{chrom}:{pos}"
            )
            continue

        # Find best match (prefer exact position + hgvsp match)
        best = None
        for m in matches:
            m_hgvsp = parse_hgvsp_short(m.get("HGVSp_Short", ""))
            m_chrom = m["Chromosome"]
            m_pos = m["Start_Position"]
            if m_chrom == chrom and m_pos == pos and m_hgvsp == hgvsp:
                best = m
                break

        if best is None:
            # Try position-only match
            for m in matches:
                m_chrom = m["Chromosome"]
                m_pos = m["Start_Position"]
                if m_chrom == chrom and m_pos == pos:
                    best = m
                    break

        if best is None:
            # Take first HGVSp match
            best = matches[0]
            m_hgvsp = parse_hgvsp_short(best.get("HGVSp_Short", ""))
            warnings.append(
                f"Position mismatch for {gene} {hgvsp}: "
                f"profile=chr{chrom}:{pos}, GENIE=chr{best['Chromosome']}:{best['Start_Position']}"
            )

        # Mark as matched
        matched_genie_keys.add(
            (best["Hugo_Symbol"], best["Chromosome"], best["Start_Position"],
             best.get("HGVSp_Short", ""))
        )

        # --- Verify fields ---
        genie_hgvsp = parse_hgvsp_short(best.get("HGVSp_Short", ""))
        if genie_hgvsp != hgvsp:
            errors.append(
                f"HGVSp MISMATCH for {gene}: profile={hgvsp}, GENIE={genie_hgvsp}"
            )

        genie_chrom = best["Chromosome"]
        if genie_chrom != chrom:
            errors.append(
                f"Chromosome MISMATCH for {gene} {hgvsp}: profile={chrom}, GENIE={genie_chrom}"
            )

        genie_pos = best["Start_Position"]
        if genie_pos != pos:
            errors.append(
                f"Position MISMATCH for {gene} {hgvsp}: profile={pos}, GENIE={genie_pos}"
            )

        genie_var_class = best["Variant_Classification"]
        if genie_var_class != var_class:
            errors.append(
                f"Variant_Classification MISMATCH for {gene} {hgvsp}: "
                f"profile={var_class}, GENIE={genie_var_class}"
            )

        # Check ref/alt alleles
        genie_ref = best.get("Reference_Allele", "")
        genie_alt = best.get("Tumor_Seq_Allele2", "")
        if genie_ref != ref_allele:
            errors.append(
                f"Ref allele MISMATCH for {gene} {hgvsp}: profile={ref_allele}, GENIE={genie_ref}"
            )
        if genie_alt != alt_allele:
            errors.append(
                f"Alt allele MISMATCH for {gene} {hgvsp}: profile={alt_allele}, GENIE={genie_alt}"
            )

        # --- Check 4: VAF computation ---
        computed_vaf, t_ref, t_alt = compute_vaf(
            best.get("t_ref_count", ""), best.get("t_alt_count", "")
        )

        if computed_vaf is not None:
            rounded_computed = round(computed_vaf, 4)
            if expected_vaf is not None:
                if abs(rounded_computed - expected_vaf) > VAF_TOLERANCE:
                    errors.append(
                        f"VAF MISMATCH for {gene} {hgvsp}: "
                        f"profile={expected_vaf}, computed={rounded_computed} "
                        f"(t_ref={t_ref}, t_alt={t_alt})"
                    )
                else:
                    details.append(
                        f"  {gene} {hgvsp}: VAF OK ({expected_vaf}) "
                        f"[t_ref={t_ref}, t_alt={t_alt}]"
                    )
            else:
                # Profile has null VAF but GENIE has counts -- check if counts are empty
                if t_ref == 0 and t_alt == 0:
                    details.append(
                        f"  {gene} {hgvsp}: VAF null in profile, GENIE counts both 0 -- OK"
                    )
                else:
                    warnings.append(
                        f"Profile has null VAF for {gene} {hgvsp} but GENIE has "
                        f"t_ref={t_ref}, t_alt={t_alt} (computed={rounded_computed})"
                    )
        else:
            if expected_vaf is None:
                details.append(
                    f"  {gene} {hgvsp}: VAF null in profile, GENIE counts unavailable -- OK"
                )
            else:
                warnings.append(
                    f"VAF not verifiable for {gene} {hgvsp}: "
                    f"t_ref_count={best.get('t_ref_count', 'MISSING')}, "
                    f"t_alt_count={best.get('t_alt_count', 'MISSING')}"
                )

    # --- Check 5: No coding target-gene mutations missed ---
    missed = []
    for row in raw_rows:
        gene = row["Hugo_Symbol"]
        var_class = row["Variant_Classification"]
        hgvsp_short = parse_hgvsp_short(row.get("HGVSp_Short", ""))
        chrom = row["Chromosome"]
        pos = row["Start_Position"]

        # Only check target genes, coding variants
        if gene not in TARGET_GENES:
            continue
        if var_class in NONCODING:
            continue

        key = (gene, chrom, pos, row.get("HGVSp_Short", ""))
        if key not in matched_genie_keys:
            missed.append(
                f"{gene} {hgvsp_short} ({var_class}) at chr{chrom}:{pos}"
            )

    if missed:
        for m in missed:
            errors.append(f"MISSED MUTATION: {m}")
    else:
        details.append(f"  Completeness OK: no coding target-gene mutations missed")

    # n_mutations check
    expected_n = profile["n_mutations"]
    actual_n = len(profile["mutations"])
    if expected_n != actual_n:
        errors.append(
            f"n_mutations field ({expected_n}) != actual mutation count ({actual_n})"
        )

    return errors, warnings, details


def main():
    print("=" * 70)
    print("BENCHMARK PROFILE QA VERIFICATION")
    print("=" * 70)

    # Load profiles
    print("\nLoading benchmark profiles...")
    b1_profiles, b1_meta = load_profiles(BATCH1_PATH)
    b2_profiles, b2_meta = load_profiles(BATCH2_PATH)

    print(f"  Batch 1: {len(b1_profiles)} profiles, "
          f"{sum(len(p['mutations']) for p in b1_profiles)} mutations")
    print(f"  Batch 2: {len(b2_profiles)} profiles, "
          f"{sum(len(p['mutations']) for p in b2_profiles)} mutations")

    all_profiles = b1_profiles + b2_profiles
    all_sample_ids = [p["patient_id"] for p in all_profiles]

    # Check for duplicates between batches
    b1_ids = set(p["patient_id"] for p in b1_profiles)
    b2_ids = set(p["patient_id"] for p in b2_profiles)
    overlap_ids = b1_ids & b2_ids

    # Load GENIE data for all samples
    print("\nLoading GENIE raw data...")
    genie_muts = load_genie_mutations(all_sample_ids)
    clinical = load_clinical_data(all_sample_ids)

    # Verify each profile
    print("\nVerifying profiles...\n")

    report_lines = []
    total_errors = 0
    total_warnings = 0
    total_mutations_checked = 0
    total_vaf_ok = 0
    total_vaf_unverifiable = 0
    profiles_clean = 0
    profiles_with_errors = 0

    per_batch_results = {1: [], 2: []}

    for batch_num, profiles in [(1, b1_profiles), (2, b2_profiles)]:
        batch_results = per_batch_results[batch_num]

        for profile in profiles:
            pid = profile["patient_id"]
            n_muts = len(profile["mutations"])
            total_mutations_checked += n_muts

            errors, warnings, details = verify_profile(profile, genie_muts, clinical)
            total_errors += len(errors)
            total_warnings += len(warnings)

            # Count VAF results from details
            for d in details:
                if "VAF OK" in d:
                    total_vaf_ok += 1
            for w in warnings:
                if "VAF not verifiable" in w:
                    total_vaf_unverifiable += 1

            status = "PASS" if not errors else "FAIL"
            if not errors:
                profiles_clean += 1
            else:
                profiles_with_errors += 1

            batch_results.append({
                "patient_id": pid,
                "oncotree": profile.get("oncotree_code", "?"),
                "n_muts": n_muts,
                "status": status,
                "errors": errors,
                "warnings": warnings,
                "details": details
            })

            # Console output
            icon = "OK" if status == "PASS" else "FAIL"
            print(f"  [{icon}] {pid} ({n_muts} muts, {profile.get('oncotree_code', '?')})")
            for e in errors:
                print(f"       ERROR: {e}")
            for w in warnings:
                print(f"       WARN:  {w}")

    # Build markdown report
    md = []
    md.append("# Benchmark Profile QA Verification Report")
    md.append("")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Source:** GENIE v19.0 (`data_mutations_extended.txt`, `data_clinical_sample.txt`)")
    md.append("")

    md.append("## Summary")
    md.append("")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Total profiles checked | {len(all_profiles)} |")
    md.append(f"| Batch 1 profiles | {len(b1_profiles)} |")
    md.append(f"| Batch 2 profiles | {len(b2_profiles)} |")
    md.append(f"| Total mutations checked | {total_mutations_checked} |")
    md.append(f"| Profiles PASS | {profiles_clean} |")
    md.append(f"| Profiles FAIL | {profiles_with_errors} |")
    md.append(f"| Total errors | {total_errors} |")
    md.append(f"| Total warnings | {total_warnings} |")
    md.append(f"| VAF computations verified | {total_vaf_ok} |")
    md.append(f"| VAF unverifiable (missing counts) | {total_vaf_unverifiable} |")
    md.append(f"| Batch overlap (patients in both) | {len(overlap_ids)} |")
    md.append("")

    # Batch overlap check
    md.append("## Cross-Batch Overlap Check")
    md.append("")
    if overlap_ids:
        md.append(f"**FAIL:** {len(overlap_ids)} patient(s) appear in BOTH batches:")
        md.append("")
        for oid in sorted(overlap_ids):
            md.append(f"- `{oid}`")
        md.append("")
    else:
        md.append("**PASS:** No patient appears in both batches. All 40 profiles are unique.")
        md.append("")

    # Batch 2 exclusion list verification
    if "excludes_batch1_ids" in b2_meta:
        excluded = set(b2_meta["excludes_batch1_ids"])
        actual_b1 = b1_ids
        if excluded == actual_b1:
            md.append("**Batch 2 exclusion list:** PASS -- exactly matches batch 1 patient IDs.")
        else:
            missing_from_exclusion = actual_b1 - excluded
            extra_in_exclusion = excluded - actual_b1
            if missing_from_exclusion:
                md.append(f"**Batch 2 exclusion list WARNING:** {len(missing_from_exclusion)} batch 1 IDs missing from exclusion list:")
                for m in sorted(missing_from_exclusion):
                    md.append(f"  - `{m}`")
            if extra_in_exclusion:
                md.append(f"**Batch 2 exclusion list WARNING:** {len(extra_in_exclusion)} IDs in exclusion list not in batch 1:")
                for m in sorted(extra_in_exclusion):
                    md.append(f"  - `{m}`")
        md.append("")

    # Detailed results per batch
    for batch_num in [1, 2]:
        results = per_batch_results[batch_num]
        md.append(f"## Batch {batch_num} Results")
        md.append("")

        # Summary table
        md.append("| # | Patient ID | OncoTree | Mutations | Status | Errors | Warnings |")
        md.append("|---|-----------|----------|-----------|--------|--------|----------|")
        for i, r in enumerate(results, 1):
            md.append(
                f"| {i} | `{r['patient_id']}` | {r['oncotree']} | "
                f"{r['n_muts']} | **{r['status']}** | {len(r['errors'])} | {len(r['warnings'])} |"
            )
        md.append("")

        # Error/warning details
        any_issues = any(r["errors"] or r["warnings"] for r in results)
        if any_issues:
            md.append(f"### Batch {batch_num} Issues")
            md.append("")
            for r in results:
                if r["errors"] or r["warnings"]:
                    md.append(f"#### `{r['patient_id']}` ({r['oncotree']})")
                    md.append("")
                    if r["errors"]:
                        md.append("**Errors:**")
                        for e in r["errors"]:
                            md.append(f"- {e}")
                        md.append("")
                    if r["warnings"]:
                        md.append("**Warnings:**")
                        for w in r["warnings"]:
                            md.append(f"- {w}")
                        md.append("")
        else:
            md.append(f"All batch {batch_num} profiles passed with no errors or warnings.")
            md.append("")

    # VAF verification detail
    md.append("## VAF Verification Detail")
    md.append("")
    md.append("VAF formula: `t_alt_count / (t_ref_count + t_alt_count)`")
    md.append(f"Tolerance: +/- {VAF_TOLERANCE}")
    md.append("")

    for batch_num in [1, 2]:
        results = per_batch_results[batch_num]
        md.append(f"### Batch {batch_num} VAF Details")
        md.append("")
        md.append("| Patient | Gene | HGVSp | Profile VAF | Computed VAF | t_ref | t_alt | Match |")
        md.append("|---------|------|-------|-------------|--------------|-------|-------|-------|")

        for r in results:
            pid = r["patient_id"]
            if pid not in genie_muts:
                continue

            raw_rows = genie_muts[pid]
            genie_by_pos = {}
            for row in raw_rows:
                key = (row["Hugo_Symbol"], row["Chromosome"], row["Start_Position"])
                genie_by_pos[key] = row

            # Reconstruct from profiles
            batch_profiles = b1_profiles if batch_num == 1 else b2_profiles
            profile = next(p for p in batch_profiles if p["patient_id"] == pid)

            for mut in profile["mutations"]:
                gene = mut["gene"]
                hgvsp = mut["hgvsp"]
                chrom = str(mut["chromosome"])
                pos = str(mut["start_position"])
                expected_vaf = mut["t_vaf"]

                key = (gene, chrom, pos)
                genie_row = genie_by_pos.get(key)

                vaf_str = f"{expected_vaf:.4f}" if expected_vaf is not None else "null"

                if genie_row:
                    computed, t_ref, t_alt = compute_vaf(
                        genie_row.get("t_ref_count", ""),
                        genie_row.get("t_alt_count", "")
                    )
                    if computed is not None:
                        rounded = round(computed, 4)
                        if expected_vaf is not None:
                            match = "YES" if abs(rounded - expected_vaf) <= VAF_TOLERANCE else "**NO**"
                        else:
                            match = "N/A (profile null)"
                        md.append(
                            f"| `{pid[-20:]}` | {gene} | {hgvsp} | "
                            f"{vaf_str} | {rounded:.4f} | {t_ref} | {t_alt} | {match} |"
                        )
                    else:
                        md.append(
                            f"| `{pid[-20:]}` | {gene} | {hgvsp} | "
                            f"{vaf_str} | N/A | N/A | N/A | N/A |"
                        )
                else:
                    md.append(
                        f"| `{pid[-20:]}` | {gene} | {hgvsp} | "
                        f"{vaf_str} | NOT FOUND | - | - | - |"
                    )
        md.append("")

    # Completeness check summary
    md.append("## Completeness Check")
    md.append("")
    md.append("For each profile, all coding mutations in the 34 target genes from GENIE raw data ")
    md.append("were compared against the profile's mutation list. Noncoding variant classifications ")
    md.append(f"excluded: {', '.join(sorted(NONCODING))}.")
    md.append("")

    missed_any = False
    for batch_num in [1, 2]:
        results = per_batch_results[batch_num]
        for r in results:
            missed = [e for e in r["errors"] if e.startswith("MISSED MUTATION")]
            if missed:
                missed_any = True
                md.append(f"**`{r['patient_id']}`** -- {len(missed)} missed mutation(s):")
                for m in missed:
                    md.append(f"- {m}")
                md.append("")

    if not missed_any:
        md.append("**All profiles are complete.** No coding target-gene mutations were missed.")
        md.append("")

    # Overall verdict
    md.append("## Verdict")
    md.append("")
    if total_errors == 0 and len(overlap_ids) == 0:
        md.append("**ALL CHECKS PASSED.** All 40 benchmark profiles faithfully reproduce GENIE v19.0 raw data.")
    elif total_errors == 0 and len(overlap_ids) > 0:
        md.append("**PARTIAL PASS.** Mutations verified but batch overlap detected.")
    else:
        md.append(f"**ISSUES DETECTED.** {total_errors} error(s) across {profiles_with_errors} profile(s).")
        md.append("Review the per-profile details above.")
    md.append("")

    # Write report
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(md))
    print(f"\nReport written to: {OUTPUT_PATH}")
    print(f"\nFinal: {profiles_clean} PASS, {profiles_with_errors} FAIL, "
          f"{total_errors} errors, {total_warnings} warnings")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
