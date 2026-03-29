#!/usr/bin/env python3
"""
QA Check: Verify NO data leakage between index patient (Henrik) and benchmark profiles.

Checks:
1. No benchmark profile contains all 5 index patient mutations
2. No benchmark profile ID matches IDs from co-occurrence analysis
3. Benchmark does not use any of the index patient's specific scores
4. No benchmark profile is "Patient 2642" from BeatAML
5. Benchmark script does not hardcode any of the 5 patient variants
6. Full independence verification against PATIENT_PROFILE.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/qa_benchmark_leakage.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "benchmark"
SCRIPTS_DIR = PROJECT_ROOT / "mutation_profile" / "scripts" / "ai_research"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"
PATIENT_PROFILE = PROJECT_ROOT / "PATIENT_PROFILE.md"

# Index patient's 5 confirmed driver mutations
INDEX_MUTATIONS = {
    "DNMT3A": "R882H",
    "IDH2": "R140Q",
    "SETBP1": "G870S",
    "PTPN11": "E76Q",
    "EZH2": "V662A",
}

# Index patient's known specific scores (from PATIENT_PROFILE.md and CLAUDE.md)
INDEX_SCORES = {
    "DNMT3A_R882H_VAF": 0.39,
    "IDH2_R140Q_VAF": 0.02,
    "SETBP1_G870S_VAF": 0.34,
    "PTPN11_E76Q_VAF": 0.29,
    "EZH2_V662A_VAF": 0.59,
    "ESM2_DNMT3A_LLR": -8.383,
    "ESM2_SETBP1_LLR": -9.804,
    "ESM2_EZH2_LLR": -2.966,
    "ESM2_SETBP1_L2": 10.75,
    "CADD_EZH2": 33.0,
    "REVEL_EZH2": 0.962,
    "EVE_EZH2": 0.9997,
    "AlphaMissense_EZH2": 0.9952,
}

# Patient 2642 identifiers (BeatAML closest match)
PATIENT_2642_IDS = {"2642", "BA3097D", "patient_2642", "Patient 2642"}

# Known co-occurrence sample IDs from results
COOCCURRENCE_RESULTS_FILES = [
    "vizome_cooccurrence.json",
    "triple_carrier_vaf_analysis.json",
    "deduplication_analysis.json",
]


class QAResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.details: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str):
        self.passed = False
        self.details.append(f"FAIL: {msg}")

    def ok(self, msg: str):
        self.details.append(f"PASS: {msg}")

    def warn(self, msg: str):
        self.warnings.append(f"WARNING: {msg}")


def load_profiles(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("profiles", [])


def check_1_no_quintuple_match(profiles: list[dict]) -> QAResult:
    """Check that NO benchmark profile contains all 5 index patient mutations."""
    result = QAResult("Check 1: No profile contains all 5 index mutations")

    for prof in profiles:
        pid = prof["patient_id"]
        mutations = prof.get("mutations", [])
        gene_variant_map = {}
        for m in mutations:
            gene = m["gene"]
            hgvsp = m.get("hgvsp", "")
            gene_variant_map[gene] = hgvsp

        # Check exact 5-gene match
        matches = {}
        for gene, variant in INDEX_MUTATIONS.items():
            if gene in gene_variant_map:
                if gene_variant_map[gene] == variant:
                    matches[gene] = f"EXACT ({variant})"
                else:
                    matches[gene] = f"DIFFERENT ({gene_variant_map[gene]} vs index {variant})"

        exact_count = sum(1 for v in matches.values() if "EXACT" in v)
        gene_only_count = len(matches)

        if exact_count == 5:
            result.fail(f"Profile {pid} has ALL 5 exact index mutations -- DATA LEAKAGE")
        elif exact_count >= 4:
            result.warn(f"Profile {pid} has {exact_count}/5 exact index mutations: {matches}")
        elif exact_count >= 3:
            result.warn(f"Profile {pid} has {exact_count}/5 exact index mutations: {matches}")

        if gene_only_count == 5:
            result.warn(
                f"Profile {pid} has mutations in all 5 index genes "
                f"(different variants): {matches}"
            )

    if result.passed:
        result.ok(f"No profile out of {len(profiles)} contains all 5 exact index mutations")

    # Also check 4-gene combinations
    four_gene_exact = []
    for prof in profiles:
        pid = prof["patient_id"]
        mutations = {m["gene"]: m.get("hgvsp", "") for m in prof.get("mutations", [])}
        exact = sum(
            1 for g, v in INDEX_MUTATIONS.items()
            if mutations.get(g) == v
        )
        if exact >= 4:
            four_gene_exact.append((pid, exact))

    if four_gene_exact:
        for pid, count in four_gene_exact:
            result.warn(f"Profile {pid} matches {count}/5 exact index variants")
    else:
        result.ok("No profile matches 4 or more exact index variants")

    return result


def check_2_no_cooccurrence_id_overlap(profiles: list[dict]) -> QAResult:
    """Check that no benchmark profile ID matches IDs from co-occurrence analysis."""
    result = QAResult("Check 2: No overlap with co-occurrence analysis IDs")

    # Collect all co-occurrence patient/sample IDs
    cooccurrence_ids = set()
    for filename in COOCCURRENCE_RESULTS_FILES:
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                content = f.read()
            # Extract patient_id values
            for m in re.findall(r'"patient_id"\s*:\s*"?(\w+)"?', content):
                cooccurrence_ids.add(str(m))
            for m in re.findall(r'"sample_id"\s*:\s*"([^"]+)"', content):
                cooccurrence_ids.add(m)

    # Also check the triple/quadruple carrier files
    for json_file in RESULTS_DIR.glob("*.json"):
        try:
            with open(json_file) as f:
                content = f.read()
            for m in re.findall(r'"patient_id"\s*:\s*"?([^",\s}]+)"?', content):
                cooccurrence_ids.add(str(m))
        except Exception:
            pass

    # Check benchmark profile IDs
    benchmark_ids = {p["patient_id"] for p in profiles}
    overlap = benchmark_ids & cooccurrence_ids

    if overlap:
        result.fail(f"Overlapping IDs between benchmark and co-occurrence: {overlap}")
    else:
        result.ok(
            f"No overlap between {len(benchmark_ids)} benchmark IDs "
            f"and {len(cooccurrence_ids)} co-occurrence IDs"
        )

    return result


def check_3_no_index_scores_in_benchmark(profiles: list[dict]) -> QAResult:
    """Check that benchmark profiles do not embed any of the index patient's specific scores."""
    result = QAResult("Check 3: Benchmark does not contain index patient scores")

    # Check each profile's mutations for exact gene+variant+VAF match
    # (same gene AND variant AND identical VAF = possible leakage)
    index_vafs = {
        ("DNMT3A", "R882H"): 0.39,
        ("IDH2", "R140Q"): 0.02,
        ("SETBP1", "G870S"): 0.34,
        ("PTPN11", "E76Q"): 0.29,
        ("EZH2", "V662A"): 0.59,
    }

    for prof in profiles:
        pid = prof["patient_id"]
        for m in prof.get("mutations", []):
            gene = m["gene"]
            hgvsp = m.get("hgvsp", "")
            t_vaf = m.get("t_vaf")
            key = (gene, hgvsp)
            if key in index_vafs and t_vaf is not None:
                index_vaf = index_vafs[key]
                if abs(t_vaf - index_vaf) < 0.001:
                    result.fail(
                        f"Profile {pid} has {gene} {hgvsp} at VAF={t_vaf} "
                        f"(identical to index patient VAF={index_vaf}) -- DATA LEAKAGE"
                    )
                else:
                    result.ok(
                        f"Profile {pid} has {gene} {hgvsp} at VAF={t_vaf:.4f} "
                        f"(index={index_vaf}) -- different VAF, independent"
                    )

    # Check that no profile has the index EZH2 V662A (novel, never seen in GENIE)
    for prof in profiles:
        for m in prof.get("mutations", []):
            if m["gene"] == "EZH2" and m.get("hgvsp") == "V662A":
                result.fail(
                    f"Profile {prof['patient_id']} contains EZH2 V662A "
                    f"(index patient's novel unreported variant -- 0/20,739 in GENIE)"
                )

    if result.passed:
        result.ok("No index patient-specific scores or novel variants found in benchmark data")

    return result


def check_4_no_patient_2642(profiles: list[dict]) -> QAResult:
    """Check if any benchmark profile is Patient 2642 from BeatAML."""
    result = QAResult("Check 4: No benchmark profile is Patient 2642 (BeatAML)")

    # Patient 2642 signature: IDH2 + PTPN11 + SETBP1 + ASXL1 + SRSF2 + STAG2 (no DNMT3A)
    patient_2642_genes = {"IDH2", "PTPN11", "SETBP1", "ASXL1", "SRSF2", "STAG2"}

    for prof in profiles:
        pid = prof["patient_id"]

        # Check ID directly
        for known_id in PATIENT_2642_IDS:
            if known_id.lower() in pid.lower():
                result.fail(f"Profile {pid} matches Patient 2642 identifier '{known_id}'")

        # Check mutation signature match
        prof_genes = {m["gene"] for m in prof.get("mutations", [])}
        if patient_2642_genes.issubset(prof_genes):
            # Check specific variants
            prof_variants = {m["gene"]: m.get("hgvsp", "") for m in prof.get("mutations", [])}
            if (
                prof_variants.get("IDH2") == "R140Q"
                and prof_variants.get("SETBP1") == "G870S"
                and "DNMT3A" not in prof_genes
            ):
                result.warn(
                    f"Profile {pid} has Patient 2642-like signature "
                    f"(IDH2+PTPN11+SETBP1+ASXL1+SRSF2+STAG2, no DNMT3A)"
                )

    # Patient 2642 is from BeatAML, not GENIE -- check that benchmark is GENIE-only
    non_genie = [p["patient_id"] for p in profiles if not p["patient_id"].startswith("GENIE-")]
    if non_genie:
        result.fail(f"Non-GENIE profiles found (could be BeatAML): {non_genie}")
    else:
        result.ok("All benchmark profiles are GENIE-sourced (Patient 2642 is BeatAML-only)")

    if result.passed:
        result.ok(f"No profile matches Patient 2642 identity or mutation signature")

    return result


def check_5_no_hardcoded_variants_in_script() -> QAResult:
    """Check that the benchmark script does not hardcode any of the 5 patient-specific variants."""
    result = QAResult("Check 5: Benchmark script has no hardcoded index patient variants")

    script_path = SCRIPTS_DIR / "benchmark_profiles.py"
    if not script_path.exists():
        result.fail(f"Benchmark script not found: {script_path}")
        return result

    with open(script_path) as f:
        lines = f.readlines()

    # Patterns that would indicate hardcoding of index-patient-specific data
    # (as opposed to generic references in comments/docstrings)
    hardcoded_patterns = {
        "DNMT3A.*R882H": "DNMT3A R882H",
        "IDH2.*R140Q": "IDH2 R140Q",
        "SETBP1.*G870S": "SETBP1 G870S",
        "PTPN11.*E76Q": "PTPN11 E76Q",
        "EZH2.*V662A": "EZH2 V662A",
    }

    # Check for hardcoded variant values in non-comment, non-docstring lines
    in_docstring = False
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track docstrings
        if '"""' in stripped:
            if stripped.count('"""') == 1:
                in_docstring = not in_docstring
            # A line with two """ is a one-liner docstring, skip it
            continue

        if in_docstring:
            continue

        # Skip pure comments
        if stripped.startswith("#"):
            continue

        for pattern, variant_name in hardcoded_patterns.items():
            if re.search(pattern, line):
                # Check if it's in a string literal that could be a hardcoded value
                # vs just a comment at end of line
                code_part = line.split("#")[0] if "#" in line else line
                if re.search(pattern, code_part):
                    if '"' in code_part or "'" in code_part:
                        result.fail(
                            f"Line {lineno}: Possible hardcoded {variant_name} "
                            f"in code: {stripped[:100]}"
                        )

    # Check for hardcoded index patient VAF values
    index_vafs = [0.39, 0.02, 0.34, 0.29, 0.59]
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""'):
            continue
        for vaf in index_vafs:
            # Only flag if it looks like a hardcoded assignment, not a threshold
            if f"= {vaf}" in line or f"={vaf}" in line:
                code_part = line.split("#")[0]
                if f"= {vaf}" in code_part or f"={vaf}" in code_part:
                    result.warn(
                        f"Line {lineno}: Possible hardcoded index VAF {vaf}: {stripped[:100]}"
                    )

    if result.passed:
        result.ok("No hardcoded index patient variants found in benchmark script code")

    # Also check docstring/comment references (informational only)
    comment_refs = []
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") or in_docstring:
            for pattern, variant_name in hardcoded_patterns.items():
                if re.search(pattern, line):
                    comment_refs.append((lineno, variant_name, stripped[:80]))

    if comment_refs:
        result.ok(
            f"Found {len(comment_refs)} comment/docstring references to index variants "
            f"(acceptable as examples, not data leakage)"
        )

    return result


def check_6_patient_profile_independence() -> QAResult:
    """Verify benchmark is fully independent from PATIENT_PROFILE.md data."""
    result = QAResult("Check 6: Benchmark is independent from PATIENT_PROFILE.md")

    if not PATIENT_PROFILE.exists():
        result.fail("PATIENT_PROFILE.md not found")
        return result

    with open(PATIENT_PROFILE) as f:
        profile_content = f.read()

    # Extract all unique identifiers from patient profile
    profile_identifiers = set()

    # Extract genomic coordinates
    for m in re.findall(r'NC_\d+\.\d+:g\.\d+[ACGT]>[ACGT]', profile_content):
        profile_identifiers.add(m)

    # Extract ClinVar VCV IDs
    for m in re.findall(r'VCV\d+\.\d+', profile_content):
        profile_identifiers.add(m)

    # Extract dbSNP IDs
    for m in re.findall(r'rs\d+', profile_content):
        profile_identifiers.add(m)

    # Check benchmark results for these identifiers
    results_json = BENCHMARK_DIR / "benchmark_results.json"
    if results_json.exists():
        with open(results_json) as f:
            results_content = f.read()
        for identifier in profile_identifiers:
            if identifier in results_content:
                # This could be a shared dbSNP for common variants (e.g. R882H)
                # Only flag as warning, not failure, for common variants
                result.warn(
                    f"Identifier {identifier} from PATIENT_PROFILE.md "
                    f"also appears in benchmark results (may be coincidental for common variants)"
                )

    # Verify the benchmark JSON files don't contain patient name or demographics
    for batch_file in [
        BENCHMARK_DIR / "benchmark_profiles.json",
        BENCHMARK_DIR / "benchmark_profiles_batch2.json",
    ]:
        if not batch_file.exists():
            continue
        with open(batch_file) as f:
            content = f.read().lower()
        for personal in ["REDACTED_NAME", "REDACTED_SURNAME", "REDACTED_CITY", "REDACTED_DOB"]:
            if personal in content:
                result.fail(f"Personal identifier '{personal}' found in {batch_file.name}")

    # Check that benchmark source is GENIE only (index patient is NOT in GENIE)
    for batch_file in [
        BENCHMARK_DIR / "benchmark_profiles.json",
        BENCHMARK_DIR / "benchmark_profiles_batch2.json",
    ]:
        if not batch_file.exists():
            continue
        with open(batch_file) as f:
            data = json.load(f)
        source = data.get("metadata", {}).get("source", "")
        if "GENIE" not in source:
            result.warn(f"Benchmark source is not GENIE: {source}")
        else:
            result.ok(f"Benchmark source confirmed as GENIE ({batch_file.name})")

    if result.passed:
        result.ok("Benchmark is fully independent from PATIENT_PROFILE.md")

    return result


def check_specific_variant_overlap(profiles: list[dict]) -> QAResult:
    """Detailed check of which index patient variants appear in benchmark (by gene, not necessarily same variant)."""
    result = QAResult("Check 7: Variant overlap analysis (gene-level)")

    gene_counts = {gene: 0 for gene in INDEX_MUTATIONS}
    exact_match_counts = {gene: 0 for gene in INDEX_MUTATIONS}

    for prof in profiles:
        for m in prof.get("mutations", []):
            gene = m["gene"]
            hgvsp = m.get("hgvsp", "")
            if gene in INDEX_MUTATIONS:
                gene_counts[gene] += 1
                if hgvsp == INDEX_MUTATIONS[gene]:
                    exact_match_counts[gene] += 1

    for gene, variant in INDEX_MUTATIONS.items():
        result.ok(
            f"{gene}: {gene_counts[gene]} profiles have {gene} mutations, "
            f"{exact_match_counts[gene]} with exact {variant}"
        )

    # Special flag for EZH2 V662A
    if exact_match_counts["EZH2"] > 0:
        result.fail(
            f"EZH2 V662A found in {exact_match_counts['EZH2']} benchmark profiles "
            f"-- this is a NOVEL UNREPORTED VARIANT (0/20,739 in GENIE), "
            f"so its presence in benchmark data indicates leakage"
        )
    else:
        result.ok("EZH2 V662A (novel unreported) correctly absent from all benchmark profiles")

    return result


def generate_report(results: list[QAResult], n_profiles: int) -> str:
    """Generate the QA report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# QA Data Leakage Report: Benchmark vs. Index Patient",
        "",
        f"**Generated:** {timestamp}",
        f"**Benchmark profiles checked:** {n_profiles} (20 batch 1 + 20 batch 2)",
        f"**Index patient mutations:** {', '.join(f'{g} {v}' for g, v in INDEX_MUTATIONS.items())}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    all_passed = all(r.passed for r in results)
    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    n_warnings = sum(len(r.warnings) for r in results)

    if all_passed:
        lines.append(
            f"**RESULT: ALL {len(results)} CHECKS PASSED** -- "
            f"No data leakage detected between index patient and benchmark profiles."
        )
    else:
        lines.append(
            f"**RESULT: {n_fail} CHECK(S) FAILED** -- "
            f"Potential data leakage detected. See details below."
        )

    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total checks | {len(results)} |")
    lines.append(f"| Passed | {n_pass} |")
    lines.append(f"| Failed | {n_fail} |")
    lines.append(f"| Warnings | {n_warnings} |")
    lines.append("")

    # Detailed results
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"### {r.name}")
        lines.append(f"")
        lines.append(f"**Status:** {status}")
        lines.append("")
        for d in r.details:
            lines.append(f"- {d}")
        for w in r.warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Independence verification
    lines.extend([
        "---",
        "",
        "## Independence Verification",
        "",
        "The benchmark is considered independent if:",
        "",
        "1. No benchmark profile contains the full quintuple mutation signature "
        "(DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + EZH2 V662A)",
        "2. No benchmark profile ID overlaps with co-occurrence analysis patient IDs",
        "3. The benchmark data does not embed any index patient-specific computed scores",
        "4. No benchmark profile is Patient 2642 from BeatAML (the closest known match)",
        "5. The benchmark pipeline script does not hardcode any of the 5 patient variants",
        "6. No personal identifiers from PATIENT_PROFILE.md appear in benchmark data",
        "7. EZH2 V662A (novel, 0/20,739 in GENIE) is absent from all benchmark profiles",
        "",
        "### Key facts supporting independence:",
        "",
        "- The index patient is NOT in GENIE (diagnosis/treatment in Norway, not a GENIE center)",
        "- Benchmark profiles are sourced exclusively from AACR GENIE v19.0",
        "- Patient 2642 (closest match) is from BeatAML, not GENIE",
        "- EZH2 V662A has never been observed in any public database (0/20,739 GENIE myeloid)",
        "- The benchmark script loads profiles from JSON and scores them generically "
        "via myvariant.info and ESM-2 -- no index patient data is referenced in the pipeline",
        "",
    ])

    if all_passed:
        lines.extend([
            "### Conclusion",
            "",
            "The benchmark dataset is fully independent from the index patient. "
            "There is no data leakage. The benchmark profiles were selected from GENIE v19.0 "
            "using SETBP1-positive myeloid patients with 3+ mutated target genes. "
            "The index patient's data (Norwegian diagnosis, ArcherDx panel) is not in GENIE. "
            "The benchmark pipeline processes variants generically without any reference to "
            "index patient-specific values.",
        ])
    else:
        lines.extend([
            "### Conclusion",
            "",
            "Potential data leakage was detected. Review the FAIL items above and "
            "correct the benchmark data before using it for validation.",
        ])

    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("QA CHECK: Benchmark vs. Index Patient Data Leakage")
    print("=" * 70)

    # Load all profiles
    all_profiles = []
    for batch_file in [
        BENCHMARK_DIR / "benchmark_profiles.json",
        BENCHMARK_DIR / "benchmark_profiles_batch2.json",
    ]:
        if batch_file.exists():
            profiles = load_profiles(batch_file)
            print(f"Loaded {len(profiles)} profiles from {batch_file.name}")
            all_profiles.extend(profiles)
        else:
            print(f"WARNING: {batch_file.name} not found")

    if not all_profiles:
        print("ERROR: No benchmark profiles found")
        sys.exit(1)

    print(f"Total profiles: {len(all_profiles)}")
    print()

    # Run all checks
    results = []

    print("Running Check 1: No quintuple mutation match...")
    r1 = check_1_no_quintuple_match(all_profiles)
    results.append(r1)
    print(f"  {'PASS' if r1.passed else 'FAIL'}: {r1.name}")

    print("Running Check 2: No co-occurrence ID overlap...")
    r2 = check_2_no_cooccurrence_id_overlap(all_profiles)
    results.append(r2)
    print(f"  {'PASS' if r2.passed else 'FAIL'}: {r2.name}")

    print("Running Check 3: No index patient scores in benchmark...")
    r3 = check_3_no_index_scores_in_benchmark(all_profiles)
    results.append(r3)
    print(f"  {'PASS' if r3.passed else 'FAIL'}: {r3.name}")

    print("Running Check 4: No Patient 2642...")
    r4 = check_4_no_patient_2642(all_profiles)
    results.append(r4)
    print(f"  {'PASS' if r4.passed else 'FAIL'}: {r4.name}")

    print("Running Check 5: No hardcoded variants in script...")
    r5 = check_5_no_hardcoded_variants_in_script()
    results.append(r5)
    print(f"  {'PASS' if r5.passed else 'FAIL'}: {r5.name}")

    print("Running Check 6: Independence from PATIENT_PROFILE.md...")
    r6 = check_6_patient_profile_independence()
    results.append(r6)
    print(f"  {'PASS' if r6.passed else 'FAIL'}: {r6.name}")

    print("Running Check 7: Variant overlap analysis...")
    r7 = check_specific_variant_overlap(all_profiles)
    results.append(r7)
    print(f"  {'PASS' if r7.passed else 'FAIL'}: {r7.name}")

    # Generate report
    print()
    report = generate_report(results, len(all_profiles))
    output_path = BENCHMARK_DIR / "qa_data_leakage.md"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {output_path}")

    # Print summary
    print()
    all_passed = all(r.passed for r in results)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED -- No data leakage detected")
    else:
        print("RESULT: CHECKS FAILED -- Review report for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
