#!/usr/bin/env python3
"""
QA check: verify PVS1 (truncating variant) assignments in the benchmark.

Validates that:
1. All variants with pvs1_points > 0 have a truncating variant_classification
2. All variants with pvs1_points > 0 are in LoF-relevant genes (not GoF genes)
3. All truncating variants in LoF genes DID get PVS1 (no missed assignments)
4. No missense variants got PVS1
5. Cross-checks against PATIENT_PROFILE.md: EZH2 V662A is missense, should NOT get PVS1

Saves report to: mutation_profile/results/ai_research/benchmark/qa_pvs1_assignments.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/qa_pvs1_check.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = (
    PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "benchmark"
    / "benchmark_results.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "benchmark"
    / "qa_pvs1_assignments.md"
)

# Copied from benchmark_profiles.py for independent verification
LOF_GENES = {
    "ASXL1", "TET2", "RUNX1", "TP53", "EZH2", "STAG2", "BCOR", "BCORL1",
    "DDX41", "DNMT3A", "PHF6", "WT1", "CEBPA", "GATA2", "ZRSR2", "RAD21",
    "SMC1A", "SMC3", "SETBP1", "CSF3R",
}

# GoF / oncogene genes where PVS1 should NOT apply even if truncating
# (truncating mutations in these genes are typically passenger, not driver)
GOF_GENES = {
    "NRAS", "KRAS", "JAK2", "FLT3", "NPM1", "IDH1", "IDH2", "PTPN11",
    "CALR", "MPL", "CBL", "SF3B1", "SRSF2", "U2AF1",
}

# Truncating variant types that should trigger PVS1
PVS1_FULL_TYPES = {
    "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site",
}
PVS1_MODERATE_TYPES = {"Translation_Start_Site"}
ALL_TRUNCATING = PVS1_FULL_TYPES | PVS1_MODERATE_TYPES

# Missense types that should NEVER get PVS1
MISSENSE_TYPES = {"Missense_Mutation"}

# In-frame indels -- ambiguous, but should NOT get PVS1 full strength
INFRAME_TYPES = {"In_Frame_Del", "In_Frame_Ins"}


def load_benchmark() -> dict:
    """Load benchmark_results.json."""
    if not BENCHMARK_PATH.exists():
        print(f"ERROR: {BENCHMARK_PATH} not found")
        sys.exit(1)
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def run_qa() -> str:
    """Run all PVS1 QA checks and return markdown report."""
    data = load_benchmark()
    variants = data["variants"]
    n_total = len(variants)

    lines: list[str] = []
    bugs: list[str] = []
    warnings: list[str] = []

    lines.append("# QA: PVS1 Assignment Verification")
    lines.append("")
    lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Source:** `benchmark_results.json` ({n_total} variants)")
    lines.append(f"**Method:** Independent cross-check of PVS1 assignments against variant_classification and gene lists")
    lines.append("")

    # =========================================================================
    # 1. All variants with pvs1_points > 0
    # =========================================================================
    pvs1_positive = [v for v in variants if v["pipeline"]["pvs1_points"] > 0]
    pvs1_zero = [v for v in variants if v["pipeline"]["pvs1_points"] == 0]

    lines.append("---")
    lines.append("")
    lines.append("## 1. Variants with PVS1 assigned (pvs1_points > 0)")
    lines.append("")
    lines.append(f"**Count:** {len(pvs1_positive)} / {n_total} variants")
    lines.append("")

    if pvs1_positive:
        lines.append("| Gene | HGVSp | Variant Classification | PVS1 Strength | PVS1 Points | In LoF List | In GoF List | Status |")
        lines.append("|------|-------|----------------------|---------------|-------------|-------------|-------------|--------|")

        for v in pvs1_positive:
            gene = v["gene"]
            hgvsp = v["hgvsp"]
            vc = v["pipeline"]["variant_classification"]
            pvs1_str = v["pipeline"]["pvs1_strength"]
            pvs1_pts = v["pipeline"]["pvs1_points"]
            in_lof = gene in LOF_GENES
            in_gof = gene in GOF_GENES
            is_truncating = vc in ALL_TRUNCATING

            status_parts = []
            if not is_truncating:
                status_parts.append("BUG: not truncating")
                bugs.append(f"{gene} {hgvsp}: PVS1 assigned but variant_classification={vc} is not truncating")
            if not in_lof:
                status_parts.append("BUG: not in LoF list")
                bugs.append(f"{gene} {hgvsp}: PVS1 assigned but gene not in LOF_GENES")
            if in_gof:
                status_parts.append("WARNING: GoF gene")
                warnings.append(f"{gene} {hgvsp}: PVS1 assigned to a GoF/oncogene")
            if not status_parts:
                status_parts.append("PASS")

            status = "; ".join(status_parts)
            lines.append(f"| {gene} | {hgvsp} | {vc} | {pvs1_str} | {pvs1_pts} | {'Yes' if in_lof else 'No'} | {'Yes' if in_gof else 'No'} | {status} |")
    else:
        lines.append("*No variants received PVS1.*")

    # =========================================================================
    # 2. Breakdown by truncating type
    # =========================================================================
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. PVS1 breakdown by truncating type")
    lines.append("")

    type_counter = Counter()
    gene_counter = Counter()
    for v in pvs1_positive:
        vc = v["pipeline"]["variant_classification"]
        type_counter[vc] += 1
        gene_counter[v["gene"]] += 1

    if type_counter:
        lines.append("| Variant Classification | Count |")
        lines.append("|----------------------|-------|")
        for vc, count in type_counter.most_common():
            lines.append(f"| {vc} | {count} |")

        lines.append("")
        lines.append("**By gene:**")
        lines.append("")
        lines.append("| Gene | Count |")
        lines.append("|------|-------|")
        for gene, count in gene_counter.most_common():
            lines.append(f"| {gene} | {count} |")
    else:
        lines.append("*No PVS1 assignments to break down.*")

    # =========================================================================
    # 3. Truncating variants that did NOT get PVS1 (potential missed assignments)
    # =========================================================================
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Truncating variants WITHOUT PVS1 (missed assignments?)")
    lines.append("")

    truncating_no_pvs1 = []
    for v in pvs1_zero:
        vc = v["pipeline"]["variant_classification"]
        if vc in ALL_TRUNCATING:
            truncating_no_pvs1.append(v)

    if truncating_no_pvs1:
        # Split into LoF genes (BUG) vs non-LoF genes (expected)
        missed_lof = [v for v in truncating_no_pvs1 if v["gene"] in LOF_GENES]
        truncating_non_lof = [v for v in truncating_no_pvs1 if v["gene"] not in LOF_GENES]

        if missed_lof:
            lines.append(f"**BUGS FOUND:** {len(missed_lof)} truncating variants in LoF genes did NOT get PVS1:")
            lines.append("")
            lines.append("| Gene | HGVSp | Variant Classification | In LoF List | In GoF List | Status |")
            lines.append("|------|-------|----------------------|-------------|-------------|--------|")
            for v in missed_lof:
                gene = v["gene"]
                hgvsp = v["hgvsp"]
                vc = v["pipeline"]["variant_classification"]
                in_gof = gene in GOF_GENES
                bugs.append(f"{gene} {hgvsp}: truncating ({vc}) in LoF gene but PVS1 NOT assigned")
                lines.append(f"| {gene} | {hgvsp} | {vc} | Yes | {'Yes' if in_gof else 'No'} | BUG: missed PVS1 |")
        else:
            lines.append("**No bugs:** All truncating variants in LoF genes correctly received PVS1.")

        lines.append("")
        if truncating_non_lof:
            lines.append(f"**Expected omissions:** {len(truncating_non_lof)} truncating variants in non-LoF genes (correctly no PVS1):")
            lines.append("")
            lines.append("| Gene | HGVSp | Variant Classification | In LoF List | In GoF List | Reason |")
            lines.append("|------|-------|----------------------|-------------|-------------|--------|")
            for v in truncating_non_lof:
                gene = v["gene"]
                hgvsp = v["hgvsp"]
                vc = v["pipeline"]["variant_classification"]
                in_gof = gene in GOF_GENES
                reason = "GoF/oncogene" if in_gof else "Not in LoF gene list"
                lines.append(f"| {gene} | {hgvsp} | {vc} | No | {'Yes' if in_gof else 'No'} | {reason} |")
    else:
        lines.append("*No truncating variants found without PVS1.* All truncating variants accounted for.")

    # =========================================================================
    # 4. Missense variants - verify NONE got PVS1
    # =========================================================================
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Missense variant PVS1 check (should all be 0)")
    lines.append("")

    missense_variants = [v for v in variants if v["pipeline"]["variant_classification"] in MISSENSE_TYPES]
    missense_with_pvs1 = [v for v in missense_variants if v["pipeline"]["pvs1_points"] > 0]

    lines.append(f"**Total missense variants:** {len(missense_variants)}")
    lines.append(f"**Missense with PVS1 > 0:** {len(missense_with_pvs1)}")
    lines.append("")

    if missense_with_pvs1:
        lines.append("**BUGS FOUND:** Missense variants incorrectly assigned PVS1:")
        lines.append("")
        lines.append("| Gene | HGVSp | PVS1 Points | Status |")
        lines.append("|------|-------|-------------|--------|")
        for v in missense_with_pvs1:
            gene = v["gene"]
            hgvsp = v["hgvsp"]
            pvs1_pts = v["pipeline"]["pvs1_points"]
            bugs.append(f"{gene} {hgvsp}: MISSENSE variant incorrectly assigned PVS1={pvs1_pts}")
            lines.append(f"| {gene} | {hgvsp} | {pvs1_pts} | BUG: missense with PVS1 |")
    else:
        lines.append("**PASS:** No missense variants received PVS1. Correct behavior.")

    # =========================================================================
    # 5. In-frame indels check
    # =========================================================================
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. In-frame indel PVS1 check")
    lines.append("")

    inframe_variants = [v for v in variants if v["pipeline"]["variant_classification"] in INFRAME_TYPES]
    inframe_with_pvs1 = [v for v in inframe_variants if v["pipeline"]["pvs1_points"] > 0]

    lines.append(f"**Total in-frame indels:** {len(inframe_variants)}")
    lines.append(f"**In-frame indels with PVS1 > 0:** {len(inframe_with_pvs1)}")
    lines.append("")

    if inframe_with_pvs1:
        lines.append("**WARNING:** In-frame indels with PVS1 (debatable -- not standard truncating):")
        lines.append("")
        lines.append("| Gene | HGVSp | PVS1 Points |")
        lines.append("|------|-------|-------------|")
        for v in inframe_with_pvs1:
            warnings.append(f"{v['gene']} {v['hgvsp']}: in-frame indel assigned PVS1")
            lines.append(f"| {v['gene']} | {v['hgvsp']} | {v['pipeline']['pvs1_points']} |")
    else:
        lines.append("**PASS:** No in-frame indels received PVS1.")

    # =========================================================================
    # 6. Patient EZH2 V662A cross-check
    # =========================================================================
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Patient variant cross-check: EZH2 V662A")
    lines.append("")
    lines.append("Per PATIENT_PROFILE.md, EZH2 V662A is a **missense** variant (c.1985T>C).")
    lines.append("It should NOT receive PVS1 despite EZH2 being in the LoF gene list.")
    lines.append("")

    # Search for EZH2 V662A in the benchmark (might not be there since it's
    # the index case, not the benchmark set -- but check anyway)
    ezh2_v662a = [
        v for v in variants
        if v["gene"] == "EZH2" and "V662A" in (v.get("hgvsp") or "")
    ]
    # Also check all EZH2 missense in general
    ezh2_missense = [
        v for v in variants
        if v["gene"] == "EZH2" and v["pipeline"]["variant_classification"] in MISSENSE_TYPES
    ]
    ezh2_any = [v for v in variants if v["gene"] == "EZH2"]

    if ezh2_v662a:
        for v in ezh2_v662a:
            pvs1_pts = v["pipeline"]["pvs1_points"]
            if pvs1_pts > 0:
                bugs.append("EZH2 V662A (patient variant): MISSENSE but assigned PVS1")
                lines.append(f"**BUG:** EZH2 V662A found in benchmark with PVS1={pvs1_pts}. This is a missense variant!")
            else:
                lines.append(f"**PASS:** EZH2 V662A found in benchmark with PVS1=0. Correct.")
    else:
        lines.append("EZH2 V662A not present in the benchmark set (expected -- it is the index patient variant, not a GENIE benchmark profile).")

    if ezh2_missense:
        lines.append("")
        lines.append(f"**Other EZH2 missense in benchmark:** {len(ezh2_missense)}")
        all_zero = all(v["pipeline"]["pvs1_points"] == 0 for v in ezh2_missense)
        if all_zero:
            lines.append("All EZH2 missense variants have PVS1=0. Correct.")
        else:
            lines.append("WARNING: Some EZH2 missense variants have PVS1 > 0!")
            for v in ezh2_missense:
                if v["pipeline"]["pvs1_points"] > 0:
                    bugs.append(f"EZH2 {v['hgvsp']}: missense with PVS1={v['pipeline']['pvs1_points']}")

    if ezh2_any:
        lines.append("")
        lines.append(f"**All EZH2 variants in benchmark:** {len(ezh2_any)}")
        lines.append("")
        lines.append("| HGVSp | Classification | PVS1 Points | Status |")
        lines.append("|-------|---------------|-------------|--------|")
        for v in ezh2_any:
            vc = v["pipeline"]["variant_classification"]
            pvs1_pts = v["pipeline"]["pvs1_points"]
            is_ok = (vc in ALL_TRUNCATING and pvs1_pts > 0) or (vc not in ALL_TRUNCATING and pvs1_pts == 0)
            status = "PASS" if is_ok else "BUG"
            lines.append(f"| {v['hgvsp']} | {vc} | {pvs1_pts} | {status} |")

    # =========================================================================
    # 7. GoF genes with truncating variants -- biological concern
    # =========================================================================
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. GoF gene truncating variant review")
    lines.append("")
    lines.append("These genes are primarily GoF/oncogenes. Truncating variants in these genes")
    lines.append("are typically passengers or have different functional consequences than LoF.")
    lines.append("The pipeline correctly excludes them from LOF_GENES, so they should not get PVS1.")
    lines.append("")

    gof_truncating = [
        v for v in variants
        if v["gene"] in GOF_GENES and v["pipeline"]["variant_classification"] in ALL_TRUNCATING
    ]

    if gof_truncating:
        lines.append(f"**GoF gene truncating variants:** {len(gof_truncating)}")
        lines.append("")
        lines.append("| Gene | HGVSp | Variant Classification | PVS1 Points | Status |")
        lines.append("|------|-------|----------------------|-------------|--------|")
        for v in gof_truncating:
            pvs1_pts = v["pipeline"]["pvs1_points"]
            status = "PASS (correctly no PVS1)" if pvs1_pts == 0 else "BUG: GoF gene got PVS1"
            if pvs1_pts > 0:
                bugs.append(f"{v['gene']} {v['hgvsp']}: GoF gene truncating variant got PVS1={pvs1_pts}")
            lines.append(f"| {v['gene']} | {v['hgvsp']} | {v['pipeline']['variant_classification']} | {pvs1_pts} | {status} |")
    else:
        lines.append("*No truncating variants in GoF genes found in the benchmark.*")

    # =========================================================================
    # 8. Genes in LOF_GENES that are also in GOF_GENES (overlap check)
    # =========================================================================
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. LOF_GENES / GoF gene overlap audit")
    lines.append("")

    overlap = LOF_GENES & GOF_GENES
    if overlap:
        lines.append(f"**WARNING:** {len(overlap)} genes appear in BOTH LoF and GoF lists: {', '.join(sorted(overlap))}")
        lines.append("")
        lines.append("These genes may need manual curation. Truncating variants in dual-mechanism")
        lines.append("genes might or might not warrant PVS1 depending on functional context.")
        for gene in sorted(overlap):
            warnings.append(f"{gene} is in both LOF_GENES and GOF_GENES -- dual mechanism gene")
    else:
        lines.append("**PASS:** No overlap between LoF and GoF gene lists. Clean separation.")

    # =========================================================================
    # 9. Complete variant type distribution
    # =========================================================================
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 9. Complete variant type distribution")
    lines.append("")

    vc_counter = Counter()
    for v in variants:
        vc = v["pipeline"]["variant_classification"]
        vc_counter[vc] += 1

    lines.append("| Variant Classification | Count | PVS1-eligible |")
    lines.append("|----------------------|-------|---------------|")
    for vc, count in vc_counter.most_common():
        eligible = "Yes (VeryStrong)" if vc in PVS1_FULL_TYPES else ("Yes (Moderate)" if vc in PVS1_MODERATE_TYPES else "No")
        lines.append(f"| {vc} | {count} | {eligible} |")

    # =========================================================================
    # Summary
    # =========================================================================
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total variants in benchmark:** {n_total}")
    lines.append(f"- **Variants with PVS1 > 0:** {len(pvs1_positive)}")
    lines.append(f"- **Truncating variants (all):** {sum(1 for v in variants if v['pipeline']['variant_classification'] in ALL_TRUNCATING)}")
    lines.append(f"- **Truncating in LoF genes:** {sum(1 for v in variants if v['pipeline']['variant_classification'] in ALL_TRUNCATING and v['gene'] in LOF_GENES)}")
    lines.append(f"- **Missense variants:** {len(missense_variants)}")
    lines.append(f"- **In-frame indels:** {len(inframe_variants)}")
    lines.append("")
    lines.append(f"### Bugs found: {len(bugs)}")
    lines.append("")
    if bugs:
        for b in bugs:
            lines.append(f"- **BUG:** {b}")
    else:
        lines.append("**No bugs found.** All PVS1 assignments are correct.")
    lines.append("")
    lines.append(f"### Warnings: {len(warnings)}")
    lines.append("")
    if warnings:
        for w in warnings:
            lines.append(f"- **WARNING:** {w}")
    else:
        lines.append("No warnings.")
    lines.append("")

    # Verdict
    if bugs:
        verdict = "FAIL"
    elif warnings:
        verdict = "PASS WITH WARNINGS"
    else:
        verdict = "PASS"

    lines.append(f"### Verdict: {verdict}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    report = run_qa()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)
    print(report)
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
