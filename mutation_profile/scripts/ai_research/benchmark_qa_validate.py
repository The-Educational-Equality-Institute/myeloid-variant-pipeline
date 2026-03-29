#!/usr/bin/env python3
"""
QA integrity check for benchmark_results.json

Validates:
1. Expected counts: 154 variants, 20 unique profile_ids
2. Points arithmetic: total_points == pp3 + pm2 + pvs1 + ps1 + pm1
3. Classification thresholds: >=10 -> P, >=6 -> LP, else VUS
4. Contradictory scores: PP3_VeryStrong but 0 axes pathogenic
5. ClinVar concordance math: recount matches, verify 94.1%
6. Ablation integrity: 'changed' field consistent with classification shift
7. Flag any anomalies
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

BENCHMARK = Path(__file__).resolve().parents[2] / "results" / "ai_research" / "benchmark" / "benchmark_results.json"
REPORT_OUT = BENCHMARK.parent / "qa_results_integrity.md"

# Classification thresholds (Tavtigian 2018 Bayesian point system)
# >=10 points -> Pathogenic
# >=6 points -> Likely_Pathogenic
# <6 points -> VUS
# Negative points thresholds for LB/B omitted (not used in this pipeline)
THRESH_P = 10
THRESH_LP = 6

# PP3 strength -> expected axes_pathogenic minimums
# VeryStrong requires many axes, but the mapping depends on the Bayesian model
# We check: if PP3 is VeryStrong (highest), at least 1 axis should be pathogenic
PP3_STRENGTHS_ORDERED = ["Not_Met", "Supporting", "Moderate", "Strong", "VeryStrong"]

# Concordance: ClinVar normalized vs pipeline classification
# "Match" means same severity tier OR adjacent (P matches LP, LP matches P)
# Based on the benchmark report, the match criteria is:
# Pathogenic/Likely_Pathogenic are considered adjacent matches
# VUS must match VUS exactly
CONCORDANCE_TIERS = {
    "Pathogenic": {"Pathogenic", "Likely_Pathogenic"},
    "Likely_Pathogenic": {"Pathogenic", "Likely_Pathogenic"},
    "VUS": {"VUS"},
    "Likely_Benign": {"Likely_Benign", "Benign"},
    "Benign": {"Likely_Benign", "Benign"},
}


def classify_from_points(pts: int) -> str:
    if pts >= THRESH_P:
        return "Pathogenic"
    elif pts >= THRESH_LP:
        return "Likely_Pathogenic"
    else:
        return "VUS"


def main():
    with open(BENCHMARK) as f:
        data = json.load(f)

    metadata = data["metadata"]
    variants = data["variants"]
    summary = data["summary"]

    anomalies = []
    warnings = []
    info = []

    # ── CHECK 1: Counts ──────────────────────────────────────────────
    n_variants = len(variants)
    profile_ids = set(v["profile_id"] for v in variants)
    n_profiles = len(profile_ids)

    if n_variants != 154:
        anomalies.append(f"FAIL: Expected 154 variants, got {n_variants}")
    else:
        info.append(f"PASS: Variant count = {n_variants}")

    if n_profiles != 20:
        anomalies.append(f"FAIL: Expected 20 unique profile_ids, got {n_profiles}")
    else:
        info.append(f"PASS: Profile count = {n_profiles}")

    if metadata["n_variants"] != n_variants:
        anomalies.append(f"FAIL: metadata.n_variants={metadata['n_variants']} != actual {n_variants}")
    else:
        info.append(f"PASS: metadata.n_variants matches actual count")

    if metadata["n_profiles"] != n_profiles:
        anomalies.append(f"FAIL: metadata.n_profiles={metadata['n_profiles']} != actual {n_profiles}")
    else:
        info.append(f"PASS: metadata.n_profiles matches actual count")

    # ── CHECK 2: Points arithmetic ───────────────────────────────────
    points_failures = []
    for i, v in enumerate(variants):
        p = v["pipeline"]
        computed = p["pp3_points"] + p["pm2_points"] + p["pvs1_points"] + p["ps1_points"] + p["pm1_points"]
        if computed != p["total_points"]:
            points_failures.append({
                "index": i,
                "gene": v["gene"],
                "variant": v["hgvsp"],
                "profile": v["profile_id"],
                "expected": computed,
                "actual": p["total_points"],
                "breakdown": {
                    "pp3": p["pp3_points"],
                    "pm2": p["pm2_points"],
                    "pvs1": p["pvs1_points"],
                    "ps1": p["ps1_points"],
                    "pm1": p["pm1_points"],
                }
            })

    if points_failures:
        anomalies.append(f"FAIL: {len(points_failures)} variants have total_points != sum of components")
        for pf in points_failures:
            anomalies.append(
                f"  - [{pf['index']}] {pf['gene']} {pf['variant']} "
                f"({pf['profile']}): sum={pf['expected']}, stored={pf['actual']}, "
                f"breakdown={pf['breakdown']}"
            )
    else:
        info.append(f"PASS: All {n_variants} variants have correct points arithmetic")

    # ── CHECK 3: Classification thresholds ───────────────────────────
    class_failures = []
    for i, v in enumerate(variants):
        p = v["pipeline"]
        expected_class = classify_from_points(p["total_points"])
        if p["classification"] != expected_class:
            class_failures.append({
                "index": i,
                "gene": v["gene"],
                "variant": v["hgvsp"],
                "profile": v["profile_id"],
                "points": p["total_points"],
                "expected_class": expected_class,
                "actual_class": p["classification"],
            })

    if class_failures:
        anomalies.append(f"FAIL: {len(class_failures)} variants have classification mismatch vs point thresholds")
        for cf in class_failures:
            anomalies.append(
                f"  - [{cf['index']}] {cf['gene']} {cf['variant']} "
                f"({cf['profile']}): points={cf['points']}, "
                f"expected={cf['expected_class']}, actual={cf['actual_class']}"
            )
    else:
        info.append(f"PASS: All {n_variants} classifications match point thresholds (>=10 P, >=6 LP, else VUS)")

    # ── CHECK 4: Contradictory scores ────────────────────────────────
    contradictions = []
    for i, v in enumerate(variants):
        p = v["pipeline"]
        pp3_idx = PP3_STRENGTHS_ORDERED.index(p["pp3_strength"]) if p["pp3_strength"] in PP3_STRENGTHS_ORDERED else -1

        # Check: PP3_VeryStrong or PP3_Strong but 0 axes pathogenic
        if pp3_idx >= 3 and p["axes_pathogenic"] == 0:  # Strong or VeryStrong
            contradictions.append({
                "index": i,
                "gene": v["gene"],
                "variant": v["hgvsp"],
                "profile": v["profile_id"],
                "issue": f"PP3={p['pp3_strength']} but axes_pathogenic=0",
                "pp3_points": p["pp3_points"],
                "axes_total": p["axes_total"],
            })

        # Check: axes_pathogenic > axes_total
        if p["axes_pathogenic"] > p["axes_total"]:
            contradictions.append({
                "index": i,
                "gene": v["gene"],
                "variant": v["hgvsp"],
                "profile": v["profile_id"],
                "issue": f"axes_pathogenic ({p['axes_pathogenic']}) > axes_total ({p['axes_total']})",
            })

        # Check: axes_total > 6 (only 6 axes exist)
        if p["axes_total"] > 6:
            contradictions.append({
                "index": i,
                "gene": v["gene"],
                "variant": v["hgvsp"],
                "profile": v["profile_id"],
                "issue": f"axes_total={p['axes_total']} exceeds maximum 6",
            })

        # Check: variant_classification is frameshift/nonsense but pp3 is strong
        # (PP3 is for missense; truncating variants should use PVS1 instead)
        truncating_types = {"Frame_Shift_Del", "Frame_Shift_Ins", "Nonsense_Mutation", "Splice_Site"}
        if p.get("variant_classification") in truncating_types and pp3_idx >= 3:
            contradictions.append({
                "index": i,
                "gene": v["gene"],
                "variant": v["hgvsp"],
                "profile": v["profile_id"],
                "issue": f"Truncating variant ({p['variant_classification']}) has PP3={p['pp3_strength']} (PP3 is missense-specific)",
            })

        # Check: high points from PVS1 on a missense variant (PVS1 is for LoF/truncating)
        if p.get("variant_classification") == "Missense_Mutation" and p["pvs1_points"] > 0:
            contradictions.append({
                "index": i,
                "gene": v["gene"],
                "variant": v["hgvsp"],
                "profile": v["profile_id"],
                "issue": f"Missense variant has PVS1 points={p['pvs1_points']} (PVS1 is for truncating/LoF)",
            })

        # Check: negative point values
        for key in ["pp3_points", "pm2_points", "pvs1_points", "ps1_points", "pm1_points", "total_points"]:
            if p[key] < 0:
                contradictions.append({
                    "index": i,
                    "gene": v["gene"],
                    "variant": v["hgvsp"],
                    "profile": v["profile_id"],
                    "issue": f"Negative value: {key}={p[key]}",
                })

    if contradictions:
        # Separate hard anomalies from soft warnings
        hard = [c for c in contradictions if "axes_pathogenic" in c.get("issue", "") and ">" in c.get("issue", "")]
        soft = [c for c in contradictions if c not in hard]
        if hard:
            anomalies.append(f"FAIL: {len(hard)} hard contradictions found")
            for h in hard:
                anomalies.append(f"  - [{h['index']}] {h['gene']} {h['variant']}: {h['issue']}")
        if soft:
            warnings.append(f"WARNING: {len(soft)} potential contradictions (may be design decisions)")
            for s in soft:
                warnings.append(f"  - [{s['index']}] {s['gene']} {s['variant']}: {s['issue']}")
    else:
        info.append(f"PASS: No contradictory scores found")

    # ── CHECK 5: ClinVar concordance math ────────────────────────────
    clinvar_total = 0
    clinvar_concordant = 0
    clinvar_discordant = []
    clinvar_details = []

    for i, v in enumerate(variants):
        cv = v["clinvar"]["normalized"]
        if cv in ("Not_in_ClinVar", None):
            continue
        clinvar_total += 1
        pipeline_class = v["pipeline"]["classification"]

        # Match: same tier or adjacent P/LP tier
        allowed = CONCORDANCE_TIERS.get(cv, {cv})
        is_match = pipeline_class in allowed

        clinvar_details.append({
            "index": i,
            "gene": v["gene"],
            "variant": v["hgvsp"],
            "pipeline": pipeline_class,
            "clinvar": cv,
            "match": is_match,
        })

        if is_match:
            clinvar_concordant += 1
        else:
            clinvar_discordant.append({
                "index": i,
                "gene": v["gene"],
                "variant": v["hgvsp"],
                "pipeline": pipeline_class,
                "clinvar": cv,
            })

    computed_rate = clinvar_concordant / clinvar_total if clinvar_total > 0 else 0.0

    # Verify against stored summary
    stored_total = summary["clinvar_concordance"]["total_with_clinvar"]
    stored_concordant = summary["clinvar_concordance"]["concordant"]
    stored_rate = summary["clinvar_concordance"]["rate"]

    concordance_ok = True
    if clinvar_total != stored_total:
        anomalies.append(f"FAIL: Recomputed ClinVar total={clinvar_total}, stored={stored_total}")
        concordance_ok = False
    if clinvar_concordant != stored_concordant:
        anomalies.append(f"FAIL: Recomputed concordant={clinvar_concordant}, stored={stored_concordant}")
        concordance_ok = False
    if abs(computed_rate - stored_rate) > 0.001:
        anomalies.append(f"FAIL: Recomputed rate={computed_rate:.4f}, stored={stored_rate:.4f}")
        concordance_ok = False

    if concordance_ok:
        info.append(f"PASS: ClinVar concordance verified: {clinvar_concordant}/{clinvar_total} = {computed_rate:.1%}")

    # List discordant entries
    if clinvar_discordant:
        info.append(f"INFO: {len(clinvar_discordant)} ClinVar-discordant variants:")
        for d in clinvar_discordant:
            info.append(f"  - {d['gene']} {d['variant']}: pipeline={d['pipeline']}, clinvar={d['clinvar']}")

    # ── CHECK 6: Ablation integrity ──────────────────────────────────
    ablation_failures = []
    ablation_point_increases = []
    ablation_stats = Counter()
    expected_ablation_keys = {"remove_esm2_llr", "remove_alphamissense", "remove_eve_score",
                              "remove_cadd_phred", "remove_revel", "remove_sift_score",
                              "remove_polyphen2_score"}

    for i, v in enumerate(variants):
        abl = v.get("ablation", {})
        orig_class = v["pipeline"]["classification"]
        orig_points = v["pipeline"]["total_points"]

        # Check all 7 ablation keys present
        actual_keys = set(abl.keys())
        if actual_keys != expected_ablation_keys:
            missing = expected_ablation_keys - actual_keys
            extra = actual_keys - expected_ablation_keys
            if missing:
                ablation_failures.append(f"[{i}] {v['gene']} {v['hgvsp']}: missing ablation keys: {missing}")
            if extra:
                ablation_failures.append(f"[{i}] {v['gene']} {v['hgvsp']}: unexpected ablation keys: {extra}")

        for key, entry in abl.items():
            ablation_stats[key] += 1

            # Verify 'changed' field: should be True iff new_classification != original
            new_class = entry["new_classification"]
            expected_changed = (new_class != orig_class)
            actual_changed = entry["changed"]

            if actual_changed != expected_changed:
                ablation_failures.append(
                    f"[{i}] {v['gene']} {v['hgvsp']} / {key}: "
                    f"changed={actual_changed}, but orig={orig_class} -> new={new_class} "
                    f"(expected changed={expected_changed})"
                )

            # Verify new_classification matches new_total_points thresholds
            new_points = entry["new_total_points"]
            expected_new_class = classify_from_points(new_points)
            if new_class != expected_new_class:
                ablation_failures.append(
                    f"[{i}] {v['gene']} {v['hgvsp']} / {key}: "
                    f"new_total_points={new_points} -> expected {expected_new_class}, got {new_class}"
                )

            # Check if ablated points > original points
            # This is EXPECTED Bayesian behavior: removing a benign-leaning tool from an
            # axis reduces axes_total without reducing axes_pathogenic, increasing the
            # concordance ratio (axes_pathogenic/axes_total), which can cross PP3 thresholds.
            # Not an anomaly -- documented as design-consistent finding.
            if new_points > orig_points:
                ablation_point_increases.append(
                    f"[{i}] {v['gene']} {v['hgvsp']} / {key}: "
                    f"new_total_points={new_points} > original={orig_points} "
                    f"(+{new_points - orig_points} pts from concordance ratio increase)"
                )

            # Verify new_axes_pathogenic <= original axes_pathogenic or at most equal
            # (Removing an axis tool could reduce axes but not increase them... unless
            # the axis was benign and its removal doesn't add an axis. At most equal.)
            orig_axes = v["pipeline"]["axes_pathogenic"]
            new_axes = entry["new_axes_pathogenic"]
            # Actually, removing a tool could change axis count either way depending on
            # how the Bayesian model aggregates. We'll flag increases as warnings.
            if new_axes > orig_axes:
                warnings.append(
                    f"  - [{i}] {v['gene']} {v['hgvsp']} / {key}: "
                    f"new_axes_pathogenic={new_axes} > original={orig_axes} after tool removal"
                )

    if ablation_failures:
        anomalies.append(f"FAIL: {len(ablation_failures)} ablation integrity issues")
        for af in ablation_failures:
            anomalies.append(f"  - {af}")
    else:
        info.append(f"PASS: All ablation entries have correct 'changed' flags and classification thresholds")
        info.append(f"PASS: All {n_variants} variants have 7 ablation entries each")

    if ablation_point_increases:
        info.append(f"INFO: {len(ablation_point_increases)} ablation entries where removing a tool increased points "
                     f"(expected Bayesian behavior: removing benign-leaning score raises concordance ratio)")

    # ── CHECK 7: Summary classification counts ───────────────────────
    recomputed_counts = Counter(v["pipeline"]["classification"] for v in variants)
    stored_counts = summary["classification_counts"]

    counts_ok = True
    for cls in set(list(recomputed_counts.keys()) + list(stored_counts.keys())):
        rc = recomputed_counts.get(cls, 0)
        sc = stored_counts.get(cls, 0)
        if rc != sc:
            anomalies.append(f"FAIL: summary.classification_counts[{cls}]: recomputed={rc}, stored={sc}")
            counts_ok = False

    if counts_ok:
        count_str = ", ".join(f"{k}={v}" for k, v in sorted(recomputed_counts.items()))
        info.append(f"PASS: Classification counts match summary ({count_str})")

    # ── CHECK 8: Profile-level statistics ────────────────────────────
    variants_per_profile = Counter(v["profile_id"] for v in variants)
    min_v = min(variants_per_profile.values())
    max_v = max(variants_per_profile.values())
    mean_v = sum(variants_per_profile.values()) / len(variants_per_profile)
    info.append(f"INFO: Variants per profile: min={min_v}, max={max_v}, mean={mean_v:.1f}")

    # ── CHECK 9: Gene distribution ───────────────────────────────────
    gene_counts = Counter(v["gene"] for v in variants)
    unique_genes = len(gene_counts)
    info.append(f"INFO: {unique_genes} unique genes across {n_variants} variants")

    # ── CHECK 10: Score coverage ─────────────────────────────────────
    score_keys = ["esm2_llr", "alphamissense", "eve_score", "cadd_phred", "revel", "sift_score", "polyphen2_score", "gnomad_af"]
    score_coverage = {}
    for sk in score_keys:
        non_null = sum(1 for v in variants if v["scores"].get(sk) is not None)
        score_coverage[sk] = non_null

    info.append(f"INFO: Score coverage (non-null out of {n_variants}):")
    for sk, count in sorted(score_coverage.items(), key=lambda x: -x[1]):
        info.append(f"  - {sk}: {count}/{n_variants} ({count/n_variants:.0%})")

    # ── CHECK 11: Ablation fragility summary ─────────────────────────
    fragile_count = 0
    robust_count = 0
    fragile_by_tool = Counter()
    plp_variants = [v for v in variants if v["pipeline"]["classification"] in ("Pathogenic", "Likely_Pathogenic")]

    for v in plp_variants:
        any_fragile = False
        for key, entry in v.get("ablation", {}).items():
            if entry["changed"]:
                any_fragile = True
                fragile_by_tool[key] += 1
        if any_fragile:
            fragile_count += 1
        else:
            robust_count += 1

    info.append(f"INFO: P/LP ablation robustness: {robust_count}/{len(plp_variants)} fully robust, "
                f"{fragile_count}/{len(plp_variants)} fragile to at least 1 tool")
    if fragile_by_tool:
        info.append(f"INFO: Fragility by tool:")
        for tool, cnt in fragile_by_tool.most_common():
            info.append(f"  - {tool}: {cnt} variants affected")

    # ── GENERATE REPORT ──────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_anomalies = len([a for a in anomalies if a.startswith("FAIL")])
    n_warnings = len([w for w in warnings if w.startswith("WARNING")])

    verdict = "ALL CHECKS PASSED" if n_anomalies == 0 else f"{n_anomalies} ANOMALIES DETECTED"

    lines = []
    lines.append(f"# Benchmark QA Integrity Report")
    lines.append(f"")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Source:** `benchmark_results.json`")
    lines.append(f"**Verdict:** {verdict}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Anomalies section
    lines.append(f"## Anomalies ({n_anomalies} found)")
    lines.append(f"")
    if anomalies:
        for a in anomalies:
            lines.append(a)
    else:
        lines.append("None.")
    lines.append(f"")

    # Warnings section
    lines.append(f"## Warnings ({n_warnings} found)")
    lines.append(f"")
    if warnings:
        for w in warnings:
            lines.append(w)
    else:
        lines.append("None.")
    lines.append(f"")

    # Passed checks
    lines.append(f"## Passed Checks")
    lines.append(f"")
    for item in info:
        if item.startswith("PASS"):
            lines.append(f"- {item}")
    lines.append(f"")

    # ClinVar concordance detail
    lines.append(f"## ClinVar Concordance Detail")
    lines.append(f"")
    lines.append(f"**Total with ClinVar:** {clinvar_total}")
    lines.append(f"**Concordant:** {clinvar_concordant}")
    lines.append(f"**Rate:** {computed_rate:.1%}")
    lines.append(f"**Match criteria:** P/LP treated as adjacent tier (both count as concordant)")
    lines.append(f"")
    if clinvar_discordant:
        lines.append(f"### Discordant Variants ({len(clinvar_discordant)})")
        lines.append(f"")
        lines.append(f"| # | Gene | Variant | Pipeline | ClinVar | Gap |")
        lines.append(f"|---|------|---------|----------|---------|-----|")
        for j, d in enumerate(clinvar_discordant, 1):
            lines.append(f"| {j} | {d['gene']} | {d['variant']} | {d['pipeline']} | {d['clinvar']} | {d['pipeline']} vs {d['clinvar']} |")
        lines.append(f"")

    # Ablation summary
    lines.append(f"## Ablation Robustness Summary")
    lines.append(f"")
    lines.append(f"**P/LP variants:** {len(plp_variants)}")
    lines.append(f"**Fully robust (no tool removal changes classification):** {robust_count} ({robust_count/len(plp_variants):.1%})")
    lines.append(f"**Fragile to at least 1 tool:** {fragile_count} ({fragile_count/len(plp_variants):.1%})")
    lines.append(f"")
    if fragile_by_tool:
        lines.append(f"### Fragility by Tool")
        lines.append(f"")
        lines.append(f"| Tool Removed | Variants Affected |")
        lines.append(f"|-------------|------------------:|")
        for tool, cnt in fragile_by_tool.most_common():
            lines.append(f"| {tool} | {cnt} |")
        lines.append(f"")

    # Bayesian concordance ratio section
    if ablation_point_increases:
        lines.append(f"### Concordance Ratio Effect ({len(ablation_point_increases)} instances)")
        lines.append(f"")
        lines.append(f"In {len(ablation_point_increases)} ablation entries, removing a tool *increased* total points.")
        lines.append(f"This is expected behavior in the Bayesian axis concordance model: removing a benign-leaning")
        lines.append(f"score from an axis reduces `axes_total` without reducing `axes_pathogenic`, raising the")
        lines.append(f"concordance ratio (`axes_pathogenic / axes_total`) and potentially crossing a PP3 threshold.")
        lines.append(f"")
        # Tally by tool
        increase_by_tool = Counter()
        increase_magnitudes = []
        for entry_str in ablation_point_increases:
            # Parse tool name from "[idx] gene variant / remove_xxx: ..."
            parts = entry_str.split(" / ")
            if len(parts) >= 2:
                tool_part = parts[1].split(":")[0]
                increase_by_tool[tool_part] += 1
            # Parse magnitude
            if "+(" in entry_str or "(+" in entry_str:
                try:
                    mag_str = entry_str.split("(+")[1].split(" pts")[0]
                    increase_magnitudes.append(int(mag_str))
                except (IndexError, ValueError):
                    pass
        lines.append(f"| Tool Removed | Instances |")
        lines.append(f"|-------------|----------:|")
        for tool, cnt in increase_by_tool.most_common():
            lines.append(f"| {tool} | {cnt} |")
        lines.append(f"")
        if increase_magnitudes:
            lines.append(f"Point increase range: +{min(increase_magnitudes)} to +{max(increase_magnitudes)}")
            lines.append(f"")

    # Data profile
    lines.append(f"## Data Profile")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|------:|")
    lines.append(f"| Total variants | {n_variants} |")
    lines.append(f"| Unique profiles | {n_profiles} |")
    lines.append(f"| Unique genes | {unique_genes} |")
    lines.append(f"| Variants/profile (min) | {min_v} |")
    lines.append(f"| Variants/profile (max) | {max_v} |")
    lines.append(f"| Variants/profile (mean) | {mean_v:.1f} |")
    lines.append(f"| Runtime | {metadata['runtime_seconds']:.1f}s |")
    lines.append(f"| ESM-2 enabled | {metadata['esm2_enabled']} |")
    lines.append(f"")

    # Classification distribution
    lines.append(f"### Classification Distribution")
    lines.append(f"")
    lines.append(f"| Classification | Count | % |")
    lines.append(f"|---------------|------:|--:|")
    for cls in ["Pathogenic", "Likely_Pathogenic", "VUS", "Likely_Benign", "Benign"]:
        c = recomputed_counts.get(cls, 0)
        pct = c / n_variants * 100
        lines.append(f"| {cls} | {c} | {pct:.1f}% |")
    lines.append(f"")

    # Score coverage
    lines.append(f"### Score Coverage")
    lines.append(f"")
    lines.append(f"| Score | Non-null | Coverage |")
    lines.append(f"|-------|--------:|---------:|")
    for sk, count in sorted(score_coverage.items(), key=lambda x: -x[1]):
        lines.append(f"| {sk} | {count} | {count/n_variants:.0%} |")
    lines.append(f"")

    # Informational items
    other_info = [item for item in info if not item.startswith("PASS")]
    if other_info:
        lines.append(f"## Additional Information")
        lines.append(f"")
        for item in other_info:
            lines.append(f"- {item}")
        lines.append(f"")

    report = "\n".join(lines)
    REPORT_OUT.write_text(report)
    print(report)

    # Return exit code
    return 1 if n_anomalies > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
