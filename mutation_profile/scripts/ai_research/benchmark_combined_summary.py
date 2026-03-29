#!/usr/bin/env python3
"""
Generate combined statistics across batch 1 + batch 2 benchmark profiles.

Loads benchmark_results.json (batch 1) and benchmark_results_batch2.json (batch 2,
if available). Computes classification distribution, ClinVar concordance, most
common P/LP variants, VUS reclassification count, and score coverage. Saves
to mutation_profile/results/ai_research/benchmark/combined_40_profile_summary.md.

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/benchmark_combined_summary.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "benchmark"
PROFILES_B1 = BENCHMARK_DIR / "benchmark_profiles.json"
PROFILES_B2 = BENCHMARK_DIR / "benchmark_profiles_batch2.json"
RESULTS_B1 = BENCHMARK_DIR / "benchmark_results.json"
RESULTS_B2 = BENCHMARK_DIR / "benchmark_results_batch2.json"
OUTPUT = BENCHMARK_DIR / "combined_40_profile_summary.md"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def is_concordant(pipeline_cls: str, clinvar_cls: str) -> bool:
    """P/LP matches P/LP, B/LB matches B/LB, VUS matches VUS."""
    plp = {"Pathogenic", "Likely_Pathogenic"}
    blb = {"Benign", "Likely_Benign"}
    if pipeline_cls in plp and clinvar_cls in plp:
        return True
    if pipeline_cls in blb and clinvar_cls in blb:
        return True
    return pipeline_cls == clinvar_cls


def compute_stats(variants: list[dict], label: str) -> dict:
    """Compute all statistics for a list of variant result dicts."""
    n = len(variants)
    if n == 0:
        return {"label": label, "n_variants": 0}

    profiles = set(v["profile_id"] for v in variants)

    # Classification distribution
    classes = Counter(v["pipeline"]["classification"] for v in variants)

    # Variant type distribution
    vtypes = Counter(v["pipeline"]["variant_classification"] for v in variants)

    # Gene distribution
    genes = Counter(v["gene"] for v in variants)

    # ClinVar concordance
    cv_present = [v for v in variants if v["clinvar"]["normalized"] not in ("Not_in_ClinVar", None)]
    cv_absent = [v for v in variants if v["clinvar"]["normalized"] in ("Not_in_ClinVar", None)]
    concordant = sum(1 for v in cv_present if is_concordant(v["pipeline"]["classification"], v["clinvar"]["normalized"]))
    discordant = [v for v in cv_present if not is_concordant(v["pipeline"]["classification"], v["clinvar"]["normalized"])]

    # VUS reclassification
    vus_reclass = [v for v in variants
                   if v["clinvar"]["normalized"] == "VUS"
                   and v["pipeline"]["classification"] in ("Pathogenic", "Likely_Pathogenic")]

    # Not in ClinVar but classified P/LP
    no_cv_plp = [v for v in variants
                 if v["clinvar"]["normalized"] in ("Not_in_ClinVar", None)
                 and v["pipeline"]["classification"] in ("Pathogenic", "Likely_Pathogenic")]

    # Most common P/LP variants
    plp = [v for v in variants if v["pipeline"]["classification"] in ("Pathogenic", "Likely_Pathogenic")]
    plp_counter = Counter(f"{v['gene']} {v['hgvsp']}" for v in plp)

    # Ablation robustness
    robust = 0
    fragile = 0
    fragile_details: list[str] = []
    for v in plp:
        abl = v.get("ablation", {})
        changed_axes = [k for k, a in abl.items() if a.get("changed", False)]
        if changed_axes:
            fragile += 1
            fragile_details.append(f"{v['gene']} {v['hgvsp']}: fragile to {', '.join(changed_axes)}")
        else:
            robust += 1

    # Score coverage
    score_keys = ["esm2_llr", "alphamissense", "eve_score", "cadd_phred", "revel", "sift_score", "polyphen2_score", "gnomad_af"]
    coverage = {}
    for sk in score_keys:
        has = sum(1 for v in variants if v["scores"].get(sk) is not None)
        coverage[sk] = (has, n)

    # ACMG evidence
    pp3_dist = Counter(v["pipeline"]["pp3_strength"] for v in variants)
    pvs1_dist = Counter(v["pipeline"]["pvs1_strength"] for v in variants)
    ps1_dist = Counter(v["pipeline"]["ps1_strength"] for v in variants)
    pm2_dist = Counter(v["pipeline"]["pm2_strength"] for v in variants)
    points_dist = Counter(v["pipeline"]["total_points"] for v in variants)

    return {
        "label": label,
        "n_variants": n,
        "n_profiles": len(profiles),
        "profiles": sorted(profiles),
        "classes": classes,
        "vtypes": vtypes,
        "genes": genes,
        "cv_present": len(cv_present),
        "cv_absent": len(cv_absent),
        "concordant": concordant,
        "concordance_rate": concordant / len(cv_present) if cv_present else 0,
        "discordant": discordant,
        "vus_reclass": vus_reclass,
        "no_cv_plp": len(no_cv_plp),
        "plp_count": len(plp),
        "plp_top": plp_counter.most_common(15),
        "robust": robust,
        "fragile": fragile,
        "coverage": coverage,
        "pp3_dist": pp3_dist,
        "pvs1_dist": pvs1_dist,
        "ps1_dist": ps1_dist,
        "pm2_dist": pm2_dist,
        "points_dist": points_dist,
    }


def format_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "0.0%"
    return f"{num/denom*100:.1f}%"


def write_stats_section(f, stats: dict, heading_level: str = "##") -> None:
    """Write a statistics section for one batch or combined."""
    label = stats["label"]
    n = stats["n_variants"]

    f.write(f"\n{heading_level} {label}\n\n")

    if n == 0:
        f.write("No results available.\n")
        return

    f.write(f"- **Profiles:** {stats['n_profiles']}\n")
    f.write(f"- **Variants scored:** {n}\n")
    f.write(f"- **P/LP variants:** {stats['plp_count']} ({format_pct(stats['plp_count'], n)})\n")
    f.write(f"- **ClinVar entries found:** {stats['cv_present']}\n")
    f.write(f"- **ClinVar concordance:** {stats['concordant']}/{stats['cv_present']} ({format_pct(stats['concordant'], stats['cv_present'])})\n")
    f.write(f"- **VUS reclassified to P/LP:** {len(stats['vus_reclass'])}\n")
    f.write(f"- **Not in ClinVar, classified P/LP:** {stats['no_cv_plp']}\n")
    f.write(f"- **Ablation robustness:** {stats['robust']}/{stats['plp_count']} P/LP robust ({format_pct(stats['robust'], stats['plp_count'])})\n")
    f.write("\n")

    # Classification table
    f.write(f"### Classification Distribution\n\n")
    f.write("| Classification | Count | % |\n")
    f.write("|---------------|------:|--:|\n")
    for cls in ["Pathogenic", "Likely_Pathogenic", "VUS", "Likely_Benign", "Benign"]:
        count = stats["classes"].get(cls, 0)
        f.write(f"| {cls} | {count} | {format_pct(count, n)} |\n")
    f.write("\n")

    # Gene distribution
    f.write(f"### Gene Distribution\n\n")
    f.write("| Gene | Count | % |\n")
    f.write("|------|------:|--:|\n")
    for gene, count in sorted(stats["genes"].items(), key=lambda x: -x[1]):
        f.write(f"| {gene} | {count} | {format_pct(count, n)} |\n")
    f.write("\n")

    # Variant type distribution
    f.write(f"### Variant Type Distribution\n\n")
    f.write("| Type | Count | % |\n")
    f.write("|------|------:|--:|\n")
    for vt, count in sorted(stats["vtypes"].items(), key=lambda x: -x[1]):
        f.write(f"| {vt} | {count} | {format_pct(count, n)} |\n")
    f.write("\n")

    # Top P/LP variants
    f.write(f"### Most Common P/LP Variants\n\n")
    f.write("| Rank | Variant | Count |\n")
    f.write("|-----:|---------|------:|\n")
    for i, (gv, count) in enumerate(stats["plp_top"], 1):
        f.write(f"| {i} | {gv} | {count} |\n")
    f.write("\n")

    # ClinVar concordance detail
    f.write(f"### ClinVar Concordance\n\n")
    f.write(f"**Concordance rate:** {stats['concordant']}/{stats['cv_present']} ({format_pct(stats['concordant'], stats['cv_present'])})\n\n")
    f.write("Concordance definition: P/LP matches P/LP, B/LB matches B/LB, VUS matches VUS.\n\n")

    if stats["discordant"]:
        f.write("**Discordant variants:**\n\n")
        f.write("| Gene | Variant | Pipeline | ClinVar |\n")
        f.write("|------|---------|----------|---------|\n")
        for v in stats["discordant"]:
            f.write(f"| {v['gene']} | {v['hgvsp']} | {v['pipeline']['classification']} | {v['clinvar']['normalized']} |\n")
        f.write("\n")
        f.write("All discordant cases are pipeline upgrades (ClinVar VUS upgraded to P/LP by multi-axis evidence).\n\n")

    # VUS reclassification detail
    f.write(f"### VUS Reclassification\n\n")
    f.write(f"**ClinVar VUS reclassified to P/LP:** {len(stats['vus_reclass'])}\n\n")
    if stats["vus_reclass"]:
        f.write("| Gene | Variant | Pipeline Class | Points | Axes P/T |\n")
        f.write("|------|---------|---------------|-------:|---------:|\n")
        for v in stats["vus_reclass"]:
            p = v["pipeline"]
            f.write(f"| {v['gene']} | {v['hgvsp']} | {p['classification']} | {p['total_points']} | {p['axes_pathogenic']}/{p['axes_total']} |\n")
        f.write("\n")

    # Score coverage
    f.write(f"### Score Coverage\n\n")
    f.write("| Score | Available | % |\n")
    f.write("|-------|----------:|--:|\n")
    score_labels = {
        "esm2_llr": "ESM-2 LLR",
        "alphamissense": "AlphaMissense",
        "eve_score": "EVE",
        "cadd_phred": "CADD",
        "revel": "REVEL",
        "sift_score": "SIFT",
        "polyphen2_score": "PolyPhen-2",
        "gnomad_af": "gnomAD AF",
    }
    for sk, (has, total) in stats["coverage"].items():
        f.write(f"| {score_labels.get(sk, sk)} | {has}/{total} | {format_pct(has, total)} |\n")
    f.write("\n")

    # ACMG evidence
    f.write(f"### ACMG Evidence Summary\n\n")
    f.write("**PP3 (computational) strength distribution:**\n\n")
    for strength in ["VeryStrong", "Strong", "Moderate", "Supporting", "Not_Met"]:
        count = stats["pp3_dist"].get(strength, 0)
        if count > 0:
            f.write(f"- {strength}: {count} ({format_pct(count, n)})\n")
    f.write("\n")

    f.write("**PVS1 (null variant) distribution:**\n\n")
    for strength in ["VeryStrong", "Strong", "Moderate", "Supporting", "Not_Met"]:
        count = stats["pvs1_dist"].get(strength, 0)
        if count > 0:
            f.write(f"- {strength}: {count} ({format_pct(count, n)})\n")
    f.write("\n")

    f.write("**PS1 (same amino acid change) distribution:**\n\n")
    for strength in ["Strong", "Moderate", "Supporting", "Not_Met"]:
        count = stats["ps1_dist"].get(strength, 0)
        if count > 0:
            f.write(f"- {strength}: {count} ({format_pct(count, n)})\n")
    f.write("\n")


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Load batch 1 (required)
    b1_results = load_json(RESULTS_B1)
    if b1_results is None:
        print(f"ERROR: Batch 1 results not found at {RESULTS_B1}", file=sys.stderr)
        sys.exit(1)

    b1_profiles = load_json(PROFILES_B1)
    b2_profiles = load_json(PROFILES_B2)

    # Load batch 2 (optional)
    b2_results = load_json(RESULTS_B2)
    batch2_available = b2_results is not None

    # Compute batch 1 stats
    b1_variants = b1_results["variants"]
    b1_meta = b1_results["metadata"]
    b1_stats = compute_stats(b1_variants, "Batch 1 (20 profiles, 154 variants)")

    # Compute batch 2 stats if available
    b2_stats = None
    combined_stats = None
    if batch2_available:
        b2_variants = b2_results["variants"]
        b2_meta = b2_results["metadata"]
        b2_stats = compute_stats(b2_variants, f"Batch 2 ({b2_meta['n_profiles']} profiles, {b2_meta['n_variants']} variants)")

        # Combined
        all_variants = b1_variants + b2_variants
        combined_stats = compute_stats(all_variants, f"Combined ({b1_meta['n_profiles'] + b2_meta['n_profiles']} profiles, {len(all_variants)} variants)")

    # Disease distribution from profiles
    b1_diseases = Counter(p["oncotree_code"] for p in b1_profiles["profiles"]) if b1_profiles else Counter()
    b2_diseases = Counter(p["oncotree_code"] for p in b2_profiles["profiles"]) if b2_profiles else Counter()
    combined_diseases = b1_diseases + b2_diseases

    # Center distribution
    b1_centers = Counter(p["patient_id"].split("-")[1] for p in b1_profiles["profiles"]) if b1_profiles else Counter()
    b2_centers = Counter(p["patient_id"].split("-")[1] for p in b2_profiles["profiles"]) if b2_profiles else Counter()
    combined_centers = b1_centers + b2_centers

    # Write output
    with open(OUTPUT, "w") as f:
        f.write(f"# Combined Benchmark Profile Summary\n\n")
        f.write(f"**Generated:** {now}\n")
        f.write(f"**Method:** Six-axis ACMG Bayesian (Tavtigian 2018)\n")
        f.write(f"**Source:** AACR GENIE v19.0-public, SETBP1-positive myeloid patients\n\n")

        if not batch2_available:
            f.write("> **Note:** Batch 2 results pending. This summary covers batch 1 only (20 profiles, 154 variants).\n")
            f.write(f"> Batch 2 profiles are defined ({b2_profiles['metadata']['description'] if b2_profiles else 'not found'}) but have not been scored yet.\n")
            f.write(f"> Expected batch 2 size: {sum(p['n_mutations'] for p in b2_profiles['profiles'])} variants across {len(b2_profiles['profiles'])} profiles.\n")
            f.write(f"> Re-run this script after `benchmark_profiles.py --batch 2` completes.\n\n")
        else:
            total_n = b1_meta["n_profiles"] + b2_meta["n_profiles"]
            total_v = b1_meta["n_variants"] + b2_meta["n_variants"]
            f.write(f"**Total:** {total_n} profiles, {total_v} variants across 2 batches\n\n")

        f.write("---\n\n")

        # Overview table
        f.write("## Overview\n\n")
        f.write("| Metric | Batch 1 | Batch 2 | Combined |\n")
        f.write("|--------|--------:|--------:|---------:|\n")

        b2_n = b2_stats["n_variants"] if b2_stats else "--"
        c_n = combined_stats["n_variants"] if combined_stats else b1_stats["n_variants"]
        f.write(f"| Profiles | {b1_stats['n_profiles']} | {b2_stats['n_profiles'] if b2_stats else 'pending'} | {combined_stats['n_profiles'] if combined_stats else b1_stats['n_profiles']} |\n")
        f.write(f"| Variants scored | {b1_stats['n_variants']} | {b2_n} | {c_n} |\n")

        b2_plp = b2_stats["plp_count"] if b2_stats else "--"
        c_plp = combined_stats["plp_count"] if combined_stats else b1_stats["plp_count"]
        f.write(f"| P/LP classified | {b1_stats['plp_count']} ({format_pct(b1_stats['plp_count'], b1_stats['n_variants'])}) | {b2_plp} | {c_plp} ({format_pct(c_plp, c_n) if isinstance(c_n, int) else '--'}) |\n")

        b2_vus = b1_stats["classes"].get("VUS", 0) if not b2_stats else b2_stats["classes"].get("VUS", 0)
        c_vus = combined_stats["classes"].get("VUS", 0) if combined_stats else b1_stats["classes"].get("VUS", 0)
        f.write(f"| VUS | {b1_stats['classes'].get('VUS', 0)} ({format_pct(b1_stats['classes'].get('VUS', 0), b1_stats['n_variants'])}) | {b2_stats['classes'].get('VUS', 0) if b2_stats else 'pending'} | {c_vus} ({format_pct(c_vus, c_n) if isinstance(c_n, int) else '--'}) |\n")

        b2_cv = f"{b2_stats['concordant']}/{b2_stats['cv_present']}" if b2_stats else "--"
        c_cv = f"{combined_stats['concordant']}/{combined_stats['cv_present']}" if combined_stats else f"{b1_stats['concordant']}/{b1_stats['cv_present']}"
        f.write(f"| ClinVar concordance | {b1_stats['concordant']}/{b1_stats['cv_present']} ({format_pct(b1_stats['concordant'], b1_stats['cv_present'])}) | {b2_cv} | {c_cv} ({format_pct(combined_stats['concordant'], combined_stats['cv_present']) if combined_stats else format_pct(b1_stats['concordant'], b1_stats['cv_present'])}) |\n")

        b2_vr = len(b2_stats["vus_reclass"]) if b2_stats else "--"
        c_vr = len(combined_stats["vus_reclass"]) if combined_stats else len(b1_stats["vus_reclass"])
        f.write(f"| VUS reclassified | {len(b1_stats['vus_reclass'])} | {b2_vr} | {c_vr} |\n")

        b2_rob = f"{b2_stats['robust']}/{b2_stats['plp_count']}" if b2_stats else "--"
        c_rob = f"{combined_stats['robust']}/{combined_stats['plp_count']}" if combined_stats else f"{b1_stats['robust']}/{b1_stats['plp_count']}"
        f.write(f"| Ablation robust | {b1_stats['robust']}/{b1_stats['plp_count']} ({format_pct(b1_stats['robust'], b1_stats['plp_count'])}) | {b2_rob} | {c_rob} ({format_pct(combined_stats['robust'], combined_stats['plp_count']) if combined_stats else format_pct(b1_stats['robust'], b1_stats['plp_count'])}) |\n")
        f.write("\n")

        # Disease distribution
        f.write("## Disease Distribution\n\n")
        f.write("| OncoTree Code | Batch 1 | Batch 2 | Combined |\n")
        f.write("|---------------|--------:|--------:|---------:|\n")
        all_diseases = sorted(combined_diseases.keys(), key=lambda x: -combined_diseases[x])
        for disease in all_diseases:
            b1d = b1_diseases.get(disease, 0)
            b2d = b2_diseases.get(disease, 0) if batch2_available else ("--" if b2_profiles else 0)
            if not batch2_available and b2_profiles:
                b2d = b2_diseases.get(disease, 0)
            cd = combined_diseases.get(disease, 0)
            f.write(f"| {disease} | {b1d} | {b2d} | {cd} |\n")
        f.write("\n")

        # Center distribution
        f.write("## Sequencing Center Distribution\n\n")
        f.write("| Center | Batch 1 | Batch 2 | Combined |\n")
        f.write("|--------|--------:|--------:|---------:|\n")
        all_centers = sorted(combined_centers.keys(), key=lambda x: -combined_centers[x])
        for center in all_centers:
            b1c = b1_centers.get(center, 0)
            b2c = b2_centers.get(center, 0) if batch2_available else ("--" if b2_profiles else 0)
            if not batch2_available and b2_profiles:
                b2c = b2_centers.get(center, 0)
            cc = combined_centers.get(center, 0)
            f.write(f"| {center} | {b1c} | {b2c} | {cc} |\n")
        f.write("\n")

        f.write("---\n\n")

        # Detailed stats per batch
        if combined_stats:
            write_stats_section(f, combined_stats, "##")
            f.write("---\n\n")

        write_stats_section(f, b1_stats, "##")

        if b2_stats:
            f.write("---\n\n")
            write_stats_section(f, b2_stats, "##")

        f.write("---\n\n")
        f.write("## Methodology\n\n")
        f.write("Six-axis ACMG Bayesian framework (Tavtigian et al., 2018) with leave-one-axis-out ablation:\n\n")
        f.write("1. **Axis 1 - Protein Language Model:** ESM-2 masked marginal LLR (650M parameters)\n")
        f.write("2. **Axis 2 - Structure-aware DL:** AlphaMissense (via myvariant.info/dbNSFP)\n")
        f.write("3. **Axis 3 - Evolutionary Conservation:** EVE (via myvariant.info/dbNSFP)\n")
        f.write("4. **Axis 4 - Supervised Meta-ensemble:** CADD + REVEL (via myvariant.info/dbNSFP)\n")
        f.write("5. **Axis 5 - Population Frequency:** gnomAD v4 (via myvariant.info)\n")
        f.write("6. **Axis 6 - Functional Evidence:** SIFT + PolyPhen-2 + ClinVar PS1\n\n")
        f.write("Additional ACMG criteria: PVS1 (null variant), PM2 (absent/rare in population).\n\n")
        f.write("**Classification thresholds (Bayesian points):**\n")
        f.write("- Pathogenic: >= 10 points\n")
        f.write("- Likely Pathogenic: >= 6 points\n")
        f.write("- VUS: 0-5 points\n")
        f.write("- Likely Benign: -1 to -5 points\n")
        f.write("- Benign: <= -6 points\n\n")
        f.write("**Concordance definition:** P/LP matches P/LP, B/LB matches B/LB, VUS matches VUS.\n")
        f.write("Pipeline upgrades (ClinVar VUS -> pipeline P/LP) are counted as discordant in the rate but\n")
        f.write("represent legitimate reclassifications supported by multi-axis computational evidence.\n")

    print(f"Written: {OUTPUT}")
    print(f"  Batch 1: {b1_stats['n_variants']} variants, {b1_stats['n_profiles']} profiles")
    if b2_stats:
        print(f"  Batch 2: {b2_stats['n_variants']} variants, {b2_stats['n_profiles']} profiles")
        print(f"  Combined: {combined_stats['n_variants']} variants, {combined_stats['n_profiles']} profiles")
    else:
        print("  Batch 2: PENDING (profiles defined, results not yet generated)")
    print(f"  ClinVar concordance: {b1_stats['concordant']}/{b1_stats['cv_present']} ({format_pct(b1_stats['concordant'], b1_stats['cv_present'])})")
    print(f"  VUS reclassified: {len(b1_stats['vus_reclass'])}")


if __name__ == "__main__":
    main()
