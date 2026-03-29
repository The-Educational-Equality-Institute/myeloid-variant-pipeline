#!/usr/bin/env python3
"""
Five-axis vs six-axis pipeline comparison for ISMB 2026 claim validation.

The v4 benchmark results (benchmark_results_v4.json, benchmark_results_batch2_v4.json)
contain 284 variants across 40 SETBP1-positive myeloid profiles. Although metadata
says esm2_enabled=true, all variants have esm2_llr=null -- ESM-2 scores were computed
separately and stored in esm2_benchmark_scores.json and esm2_benchmark_scores_batch2.json.

This script:
  1. Uses the stored v4 pipeline results as the 5-axis baseline (since ESM-2 was
     effectively absent from all classifications)
  2. Merges ESM-2 LLR scores from the separate score files
  3. Recomputes axes/PP3/total_points/classification with ESM-2 included (6-axis)
  4. Keeps PVS1, PS1, PM1, PM2 identical to stored values (they don't depend on ESM-2)
  5. Runs leave-one-axis-out ablation for both modes
  6. Reports exact deltas for every metric

Outputs:
  - five_vs_six_axis_comparison.json
  - five_vs_six_axis_comparison.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/five_vs_six_axis_comparison.py
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = (
    PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "benchmark"
)

# PP3 thresholds (ClinGen SVI: Pejaver et al. 2022) -- matches benchmark_profiles.py
THRESHOLDS = {
    "alphamissense": {"pathogenic": 0.564, "strong": 0.927},
    "cadd_phred": {"pathogenic": 17.3},
    "revel": {"pathogenic": 0.5, "strong": 0.773},
    "eve": {"pathogenic": 0.5},
    "sift": {"pathogenic": 0.05},
    "polyphen2": {"pathogenic": 0.453},
    "esm2_llr": {"pathogenic": -2.0},
}


def is_pathogenic(tool: str, score: float | None) -> bool | None:
    if score is None:
        return None
    thresh = THRESHOLDS.get(tool, {})
    cutoff = thresh.get("pathogenic")
    if cutoff is None:
        return None
    if tool in ("sift", "esm2_llr"):
        return score <= cutoff
    return score >= cutoff


def compute_axes(scores: dict, include_esm2: bool = True) -> tuple[int, int, dict]:
    """Count how many axes classify pathogenic. Returns (pathogenic, total, axis_results)."""
    esm2_val = scores.get("esm2_llr") if include_esm2 else None
    tools = {
        "esm2": is_pathogenic("esm2_llr", esm2_val),
        "alphamissense": is_pathogenic("alphamissense", scores.get("alphamissense")),
        "eve": is_pathogenic("eve", scores.get("eve_score")),
        "sift": is_pathogenic("sift", scores.get("sift_score")),
        "cadd": is_pathogenic("cadd_phred", scores.get("cadd_phred")),
        "revel": is_pathogenic("revel", scores.get("revel")),
        "gnomad": (
            scores["gnomad_af"] < 0.0001
        ) if scores.get("gnomad_af") is not None else None,
        "polyphen2": is_pathogenic("polyphen2", scores.get("polyphen2_score")),
    }

    def _any_on_axis(keys: list[str]) -> bool | None:
        vals = [tools[k] for k in keys if tools[k] is not None]
        return any(vals) if vals else None

    axis_results = {
        "Axis1_ProteinLM": tools["esm2"],
        "Axis2_StructureDL": tools["alphamissense"],
        "Axis3_Conservation": _any_on_axis(["eve", "sift"]),
        "Axis4_MetaEnsemble": _any_on_axis(["cadd", "revel"]),
        "Axis5_Population": tools["gnomad"],
        "Axis6_Functional": tools["polyphen2"],
    }

    scored = [v for v in axis_results.values() if v is not None]
    pathogenic = sum(1 for v in scored if v)
    return pathogenic, len(scored), axis_results


def compute_pp3(axes_pathogenic: int, axes_total: int, scores: dict) -> tuple[str, int]:
    """Determine PP3 strength from axis concordance (matches benchmark_profiles.py v4 logic)."""
    if axes_total == 0:
        return "Not_Met", 0
    concordance = axes_pathogenic / axes_total
    am_strong = scores.get("alphamissense") is not None and scores["alphamissense"] >= 0.927
    revel_strong = scores.get("revel") is not None and scores["revel"] >= 0.773
    if concordance >= 0.85 and am_strong:
        return "VeryStrong", 8
    if concordance >= 0.70 and (am_strong or revel_strong):
        return "Strong", 4
    if concordance >= 0.60:
        return "Moderate", 2
    if concordance >= 0.50:
        return "Supporting", 1
    return "Not_Met", 0


def classify_variant(total_points: int) -> str:
    if total_points >= 10:
        return "Pathogenic"
    if total_points >= 6:
        return "Likely_Pathogenic"
    if total_points >= 0:
        return "VUS"
    if total_points >= -6:
        return "Likely_Benign"
    return "Benign"


def normalize_clinvar(raw: str | None) -> str:
    if not raw:
        return "Not_in_ClinVar"
    r = raw.lower().strip()
    if "pathogenic" in r and "likely" in r:
        return "Likely_Pathogenic"
    if "pathogenic" in r:
        return "Pathogenic"
    if "benign" in r and "likely" in r:
        return "Likely_Benign"
    if "benign" in r:
        return "Benign"
    if "uncertain" in r or "vus" in r:
        return "VUS"
    if "conflicting" in r:
        return "Conflicting"
    return "Other"


def reclassify_with_esm2(variant: dict) -> dict:
    """Recompute classification with ESM-2 included, keeping stored PVS1/PS1/PM1/PM2.

    The stored v4 results have the correct PVS1/PS1/PM1/PM2 values. Only PP3 depends
    on the 6-axis concordance, so we recompute axes + PP3 + total + classification.
    """
    scores = variant["scores"]
    stored = variant["pipeline"]

    axes_p, axes_t, _ = compute_axes(scores, include_esm2=True)
    pp3_str, pp3_pts = compute_pp3(axes_p, axes_t, scores)

    # Keep stored non-PP3 criteria
    pm2_pts = stored["pm2_points"]
    pvs1_pts = stored["pvs1_points"]
    ps1_pts = stored["ps1_points"]
    pm1_pts = stored["pm1_points"]

    total = pp3_pts + pm2_pts + pvs1_pts + ps1_pts + pm1_pts
    classification = classify_variant(total)

    return {
        "axes_pathogenic": axes_p,
        "axes_total": axes_t,
        "pp3_strength": pp3_str,
        "pp3_points": pp3_pts,
        "pm2_points": pm2_pts,
        "pvs1_points": pvs1_pts,
        "ps1_points": ps1_pts,
        "pm1_points": pm1_pts,
        "total_points": total,
        "classification": classification,
    }


def run_ablation(variant: dict, include_esm2: bool = True) -> dict:
    """Leave-one-axis-out ablation.

    For 6-axis mode: ablates all 6 axes.
    For 5-axis mode: ablates only axes 2-6 (Axis 1 absent).
    Uses stored PVS1/PS1/PM1/PM2 from v4 results.
    """
    scores = variant["scores"]
    stored = variant["pipeline"]

    # Get base classification for this mode
    if include_esm2:
        base = reclassify_with_esm2(variant)
    else:
        # 5-axis = stored results (which had no ESM-2)
        base = {
            "classification": stored["classification"],
            "total_points": stored["total_points"],
        }

    axis_tools = {}
    if include_esm2:
        axis_tools["Axis1_ProteinLM"] = ["esm2_llr"]
    axis_tools["Axis2_StructureDL"] = ["alphamissense"]
    axis_tools["Axis3_Conservation"] = ["eve_score", "sift_score"]
    axis_tools["Axis4_MetaEnsemble"] = ["cadd_phred", "revel"]
    axis_tools["Axis5_Population"] = ["gnomad_af"]
    axis_tools["Axis6_Functional"] = ["polyphen2_score"]

    results = {}
    for axis_name, tools in axis_tools.items():
        mod_scores = dict(scores)
        for t in tools:
            mod_scores[t] = None

        axes_p, axes_t, _ = compute_axes(mod_scores, include_esm2=include_esm2)
        pp3_str, pp3_pts = compute_pp3(axes_p, axes_t, mod_scores)

        # PM2 is zeroed when we remove Axis 5 (population)
        pm2_pts = 0 if axis_name == "Axis5_Population" else stored["pm2_points"]
        pvs1_pts = stored["pvs1_points"]
        ps1_pts = stored["ps1_points"]
        pm1_pts = stored["pm1_points"]

        total = pp3_pts + pm2_pts + pvs1_pts + ps1_pts + pm1_pts
        new_class = classify_variant(total)

        results[f"remove_{axis_name}"] = {
            "tools_removed": tools,
            "new_axes_pathogenic": axes_p,
            "new_total_points": total,
            "new_classification": new_class,
            "changed": new_class != base["classification"],
        }

    return results


def merge_esm2_scores(variants: list[dict], esm2_file: Path) -> int:
    """Merge ESM-2 LLR scores from a separate score file into variant dicts.

    Returns the number of variants matched.
    """
    if not esm2_file.exists():
        log.warning("ESM-2 score file not found: %s", esm2_file)
        return 0

    with open(esm2_file) as f:
        esm2_data = json.load(f)

    # Build lookup: (gene, hgvsp) -> esm2_llr
    score_map: dict[tuple[str, str], float] = {}
    for sv in esm2_data["variants"]:
        key = (sv["gene"], sv["hgvsp"])
        score_map[key] = sv["esm2_llr"]

    matched = 0
    for v in variants:
        key = (v["gene"], v["hgvsp"])
        if key in score_map:
            v["scores"]["esm2_llr"] = score_map[key]
            matched += 1

    return matched


def main():
    batch1_path = BENCHMARK_DIR / "benchmark_results_v4.json"
    batch2_path = BENCHMARK_DIR / "benchmark_results_batch2_v4.json"
    esm2_batch1_path = BENCHMARK_DIR / "esm2_benchmark_scores.json"
    esm2_batch2_path = BENCHMARK_DIR / "esm2_benchmark_scores_batch2.json"

    for p in [batch1_path, batch2_path]:
        if not p.exists():
            log.error("Missing %s", p)
            return

    with open(batch1_path) as f:
        batch1 = json.load(f)
    with open(batch2_path) as f:
        batch2 = json.load(f)

    # Deep copy so we can merge ESM-2 scores without modifying originals
    b1_variants = deepcopy(batch1["variants"])
    b2_variants = deepcopy(batch2["variants"])

    # Merge ESM-2 scores from separate files
    esm2_matched_b1 = merge_esm2_scores(b1_variants, esm2_batch1_path)
    esm2_matched_b2 = merge_esm2_scores(b2_variants, esm2_batch2_path)
    log.info("Merged ESM-2 scores: batch1=%d/%d, batch2=%d/%d",
             esm2_matched_b1, len(b1_variants), esm2_matched_b2, len(b2_variants))

    all_variants = b1_variants + b2_variants
    n_total = len(all_variants)

    # Count ESM-2 coverage
    has_esm2 = [v for v in all_variants if v["scores"].get("esm2_llr") is not None]
    no_esm2 = [v for v in all_variants if v["scores"].get("esm2_llr") is None]
    log.info("ESM-2 coverage: %d/%d variants (%.1f%%)",
             len(has_esm2), n_total, 100 * len(has_esm2) / n_total)

    # =========================================================================
    # 5-AXIS BASELINE: stored v4 pipeline results (all esm2_llr were null)
    # =========================================================================
    five_axis_results = []
    for v in all_variants:
        stored = v["pipeline"]
        five_axis_results.append({
            "axes_pathogenic": stored["axes_pathogenic"],
            "axes_total": stored["axes_total"],
            "pp3_strength": stored["pp3_strength"],
            "pp3_points": stored["pp3_points"],
            "total_points": stored["total_points"],
            "classification": stored["classification"],
        })

    # =========================================================================
    # 6-AXIS: recompute with ESM-2 merged in
    # =========================================================================
    six_axis_results = []
    for v in all_variants:
        result = reclassify_with_esm2(v)
        six_axis_results.append(result)

    # Validate: for variants WITHOUT ESM-2, 6-axis should match stored
    validation_ok = 0
    validation_fail = 0
    for v, r6 in zip(all_variants, six_axis_results):
        if v["scores"].get("esm2_llr") is None:
            stored = v["pipeline"]
            if r6["classification"] == stored["classification"]:
                validation_ok += 1
            else:
                validation_fail += 1
                log.warning("Validation fail (no ESM-2): %s %s stored=%s computed=%s",
                            v["gene"], v["hgvsp"], stored["classification"], r6["classification"])
    log.info("Validation (no-ESM-2 variants): %d ok, %d fail", validation_ok, validation_fail)

    # =========================================================================
    # COMPARISON METRICS
    # =========================================================================

    # Classification counts
    def count_classes(results):
        counts = {"Pathogenic": 0, "Likely_Pathogenic": 0, "VUS": 0,
                  "Likely_Benign": 0, "Benign": 0}
        for r in results:
            cls = r["classification"]
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    six_counts = count_classes(six_axis_results)
    five_counts = count_classes(five_axis_results)

    six_plp = six_counts["Pathogenic"] + six_counts["Likely_Pathogenic"]
    five_plp = five_counts["Pathogenic"] + five_counts["Likely_Pathogenic"]

    # Reclassified variants
    classification_changes = []
    for i, (v, r6, r5) in enumerate(zip(all_variants, six_axis_results, five_axis_results)):
        if r6["classification"] != r5["classification"]:
            classification_changes.append({
                "gene": v["gene"],
                "hgvsp": v["hgvsp"],
                "profile_id": v.get("profile_id", ""),
                "esm2_llr": v["scores"].get("esm2_llr"),
                "five_axis_class": r5["classification"],
                "six_axis_class": r6["classification"],
                "five_axis_points": r5["total_points"],
                "six_axis_points": r6["total_points"],
                "five_axis_pp3": r5["pp3_strength"],
                "six_axis_pp3": r6["pp3_strength"],
                "five_axis_axes": f"{r5['axes_pathogenic']}/{r5['axes_total']}",
                "six_axis_axes": f"{r6['axes_pathogenic']}/{r6['axes_total']}",
            })

    # ClinVar concordance
    def compute_concordance(variants, results):
        concordant = 0
        total = 0
        for v, r in zip(variants, results):
            cv_norm = normalize_clinvar(v["clinvar"]["classification"])
            if cv_norm in ("Not_in_ClinVar", "Other", "Conflicting"):
                continue
            total += 1
            pipe_class = r["classification"]
            if (
                (cv_norm in ("Pathogenic", "Likely_Pathogenic")
                 and pipe_class in ("Pathogenic", "Likely_Pathogenic"))
                or (cv_norm == "VUS" and pipe_class == "VUS")
                or (cv_norm in ("Benign", "Likely_Benign")
                    and pipe_class in ("Benign", "Likely_Benign", "VUS"))
            ):
                concordant += 1
        return concordant, total

    conc_6, total_cv = compute_concordance(all_variants, six_axis_results)
    conc_5, total_cv_5 = compute_concordance(all_variants, five_axis_results)
    assert total_cv == total_cv_5, "ClinVar totals must match"

    # ClinVar concordance details (which variants differ)
    concordance_changes = []
    for v, r6, r5 in zip(all_variants, six_axis_results, five_axis_results):
        cv_norm = normalize_clinvar(v["clinvar"]["classification"])
        if cv_norm in ("Not_in_ClinVar", "Other", "Conflicting"):
            continue
        def _is_concordant(pipe_class, cv):
            return (
                (cv in ("Pathogenic", "Likely_Pathogenic")
                 and pipe_class in ("Pathogenic", "Likely_Pathogenic"))
                or (cv == "VUS" and pipe_class == "VUS")
                or (cv in ("Benign", "Likely_Benign")
                    and pipe_class in ("Benign", "Likely_Benign", "VUS"))
            )
        c6 = _is_concordant(r6["classification"], cv_norm)
        c5 = _is_concordant(r5["classification"], cv_norm)
        if c6 != c5:
            concordance_changes.append({
                "gene": v["gene"],
                "hgvsp": v["hgvsp"],
                "clinvar": cv_norm,
                "five_axis_class": r5["classification"],
                "six_axis_class": r6["classification"],
                "five_axis_concordant": c5,
                "six_axis_concordant": c6,
            })

    # Ablation robustness
    def compute_robustness(variants, results, include_esm2):
        robust = 0
        plp_total = 0
        axis_names = ["Axis1_ProteinLM", "Axis2_StructureDL", "Axis3_Conservation",
                       "Axis4_MetaEnsemble", "Axis5_Population", "Axis6_Functional"]
        if not include_esm2:
            axis_names = axis_names[1:]
        fragile_by_axis = {a: 0 for a in axis_names}
        margins = []

        for v, r in zip(variants, results):
            if r["classification"] not in ("Pathogenic", "Likely_Pathogenic"):
                continue
            plp_total += 1
            margin = r["total_points"] - 6  # distance from LP boundary
            margins.append(margin)
            ablation = run_ablation(v, include_esm2=include_esm2)
            any_changed = any(a["changed"] for a in ablation.values())
            if not any_changed:
                robust += 1
            for axis_key, a in ablation.items():
                if a["changed"]:
                    axis_name = axis_key.replace("remove_", "")
                    if axis_name in fragile_by_axis:
                        fragile_by_axis[axis_name] += 1

        return robust, plp_total, fragile_by_axis, margins

    rob_6, plp_6, frag_6, margins_6 = compute_robustness(
        all_variants, six_axis_results, include_esm2=True
    )
    rob_5, plp_5, frag_5, margins_5 = compute_robustness(
        all_variants, five_axis_results, include_esm2=False
    )

    # PP3 strength distribution
    six_pp3_dist = {}
    five_pp3_dist = {}
    for r6, r5 in zip(six_axis_results, five_axis_results):
        six_pp3_dist[r6["pp3_strength"]] = six_pp3_dist.get(r6["pp3_strength"], 0) + 1
        five_pp3_dist[r5["pp3_strength"]] = five_pp3_dist.get(r5["pp3_strength"], 0) + 1

    # Axis concordance rate
    six_conc_rates = []
    five_conc_rates = []
    for r6, r5 in zip(six_axis_results, five_axis_results):
        if r6["axes_total"] > 0:
            six_conc_rates.append(r6["axes_pathogenic"] / r6["axes_total"])
        if r5["axes_total"] > 0:
            five_conc_rates.append(r5["axes_pathogenic"] / r5["axes_total"])
    mean_conc_6 = sum(six_conc_rates) / len(six_conc_rates) if six_conc_rates else 0
    mean_conc_5 = sum(five_conc_rates) / len(five_conc_rates) if five_conc_rates else 0

    # Points distribution
    six_points = [r["total_points"] for r in six_axis_results]
    five_points = [r["total_points"] for r in five_axis_results]
    mean_pts_6 = sum(six_points) / len(six_points) if six_points else 0
    mean_pts_5 = sum(five_points) / len(five_points) if five_points else 0

    # Mean margins
    mean_margin_6 = sum(margins_6) / len(margins_6) if margins_6 else 0
    mean_margin_5 = sum(margins_5) / len(margins_5) if margins_5 else 0

    # =========================================================================
    # ADDITIONAL: per-variant ESM-2 impact analysis
    # =========================================================================
    esm2_impact = []
    for v, r6, r5 in zip(all_variants, six_axis_results, five_axis_results):
        llr = v["scores"].get("esm2_llr")
        if llr is not None:
            esm2_pathogenic = llr <= -2.0
            esm2_impact.append({
                "gene": v["gene"],
                "hgvsp": v["hgvsp"],
                "esm2_llr": llr,
                "esm2_pathogenic": esm2_pathogenic,
                "five_axes": f"{r5['axes_pathogenic']}/{r5['axes_total']}",
                "six_axes": f"{r6['axes_pathogenic']}/{r6['axes_total']}",
                "five_pp3": r5["pp3_strength"],
                "six_pp3": r6["pp3_strength"],
                "five_class": r5["classification"],
                "six_class": r6["classification"],
                "pp3_changed": r5["pp3_strength"] != r6["pp3_strength"],
                "class_changed": r5["classification"] != r6["classification"],
            })

    pp3_changes = [e for e in esm2_impact if e["pp3_changed"]]
    class_changes_esm2 = [e for e in esm2_impact if e["class_changed"]]

    # =========================================================================
    # BUILD JSON OUTPUT
    # =========================================================================
    rate_6 = conc_6 / total_cv if total_cv > 0 else 0
    rate_5 = conc_5 / total_cv if total_cv > 0 else 0
    rob_rate_6 = 100 * rob_6 / plp_6 if plp_6 > 0 else 0
    rob_rate_5 = 100 * rob_5 / plp_5 if plp_5 > 0 else 0

    output = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "description": "Five-axis vs six-axis pipeline comparison for ISMB 2026",
            "method": "Stored v4 results as 5-axis baseline; ESM-2 scores merged for 6-axis",
            "total_variants": n_total,
            "variants_with_esm2": len(has_esm2),
            "variants_without_esm2": len(no_esm2),
            "esm2_coverage_pct": round(100 * len(has_esm2) / n_total, 1),
            "batch1_variants": len(batch1["variants"]),
            "batch2_variants": len(batch2["variants"]),
            "profiles": 40,
            "validation_no_esm2_ok": validation_ok,
            "validation_no_esm2_fail": validation_fail,
        },
        "classification_comparison": {
            "six_axis": {
                "Pathogenic": six_counts["Pathogenic"],
                "Likely_Pathogenic": six_counts["Likely_Pathogenic"],
                "P_LP_total": six_plp,
                "VUS": six_counts["VUS"],
            },
            "five_axis": {
                "Pathogenic": five_counts["Pathogenic"],
                "Likely_Pathogenic": five_counts["Likely_Pathogenic"],
                "P_LP_total": five_plp,
                "VUS": five_counts["VUS"],
            },
            "delta": {
                "Pathogenic": six_counts["Pathogenic"] - five_counts["Pathogenic"],
                "Likely_Pathogenic": six_counts["Likely_Pathogenic"] - five_counts["Likely_Pathogenic"],
                "P_LP_total": six_plp - five_plp,
                "VUS": six_counts["VUS"] - five_counts["VUS"],
            },
            "variants_reclassified": len(classification_changes),
            "reclassified_details": classification_changes,
        },
        "clinvar_concordance": {
            "six_axis": {"concordant": conc_6, "total": total_cv, "rate": round(rate_6, 4)},
            "five_axis": {"concordant": conc_5, "total": total_cv, "rate": round(rate_5, 4)},
            "delta_concordant": conc_6 - conc_5,
            "delta_rate": round(rate_6 - rate_5, 4),
            "concordance_changes": concordance_changes,
        },
        "ablation_robustness": {
            "six_axis": {
                "robust_count": rob_6,
                "plp_total": plp_6,
                "robustness_rate": round(rob_rate_6, 2),
                "fragile_by_axis": frag_6,
            },
            "five_axis": {
                "robust_count": rob_5,
                "plp_total": plp_5,
                "robustness_rate": round(rob_rate_5, 2),
                "fragile_by_axis": frag_5,
            },
            "delta_robust": rob_6 - rob_5,
            "delta_robustness_rate_pp": round(rob_rate_6 - rob_rate_5, 2),
        },
        "pp3_strength_distribution": {
            "six_axis": six_pp3_dist,
            "five_axis": five_pp3_dist,
        },
        "axis_concordance_rate": {
            "six_axis_mean": round(mean_conc_6, 4),
            "five_axis_mean": round(mean_conc_5, 4),
            "delta": round(mean_conc_6 - mean_conc_5, 4),
        },
        "points_distribution": {
            "six_axis_mean": round(mean_pts_6, 2),
            "five_axis_mean": round(mean_pts_5, 2),
            "delta_mean": round(mean_pts_6 - mean_pts_5, 2),
        },
        "margin_from_boundary": {
            "six_axis_mean": round(mean_margin_6, 2),
            "five_axis_mean": round(mean_margin_5, 2),
            "delta_mean": round(mean_margin_6 - mean_margin_5, 2),
        },
        "esm2_impact_analysis": {
            "total_with_esm2": len(esm2_impact),
            "esm2_pathogenic_count": sum(1 for e in esm2_impact if e["esm2_pathogenic"]),
            "esm2_benign_count": sum(1 for e in esm2_impact if not e["esm2_pathogenic"]),
            "pp3_strength_changed": len(pp3_changes),
            "classification_changed": len(class_changes_esm2),
            "details": esm2_impact,
        },
    }

    json_path = BENCHMARK_DIR / "five_vs_six_axis_comparison.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Wrote %s", json_path)

    # =========================================================================
    # BUILD MARKDOWN REPORT
    # =========================================================================
    md = []
    md.append("# Five-Axis vs Six-Axis Pipeline Comparison")
    md.append("")
    md.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    md.append("**Purpose:** Quantify exact improvement from adding Axis 1 (ESM-2 protein LM)")
    md.append(f"**Variants:** {n_total} across 40 SETBP1-positive myeloid profiles (GENIE v19.0)")
    md.append(f"**Method:** Stored v4 results as 5-axis baseline; ESM-2 scores merged for 6-axis recomputation")
    md.append("")
    md.append("---")
    md.append("")

    # ESM-2 coverage
    md.append("## 1. ESM-2 Score Coverage")
    md.append("")
    md.append(f"- Variants with ESM-2 LLR scores: **{len(has_esm2)}/{n_total} ({output['metadata']['esm2_coverage_pct']}%)**")
    md.append(f"- Variants without ESM-2: **{len(no_esm2)}/{n_total}** (non-missense, scoring failure, or no UniProt match)")
    md.append(f"- ESM-2 pathogenic (LLR <= -2.0): **{output['esm2_impact_analysis']['esm2_pathogenic_count']}/{len(has_esm2)}**")
    md.append(f"- ESM-2 benign/neutral (LLR > -2.0): **{output['esm2_impact_analysis']['esm2_benign_count']}/{len(has_esm2)}**")
    md.append("")
    md.append("---")
    md.append("")

    # Classification impact
    md.append("## 2. Classification Impact")
    md.append("")
    md.append("| Metric | 6-Axis | 5-Axis | Delta |")
    md.append("|--------|-------:|-------:|------:|")
    md.append(f"| Pathogenic | {six_counts['Pathogenic']} | {five_counts['Pathogenic']} | {six_counts['Pathogenic'] - five_counts['Pathogenic']:+d} |")
    md.append(f"| Likely Pathogenic | {six_counts['Likely_Pathogenic']} | {five_counts['Likely_Pathogenic']} | {six_counts['Likely_Pathogenic'] - five_counts['Likely_Pathogenic']:+d} |")
    md.append(f"| **P/LP total** | **{six_plp}** | **{five_plp}** | **{six_plp - five_plp:+d}** |")
    md.append(f"| VUS | {six_counts['VUS']} | {five_counts['VUS']} | {six_counts['VUS'] - five_counts['VUS']:+d} |")
    md.append("")

    if classification_changes:
        md.append(f"### Reclassified Variants ({len(classification_changes)})")
        md.append("")
        md.append("| Gene | Variant | ESM-2 LLR | 5-Axis | 6-Axis | 5-Ax Pts | 6-Ax Pts | 5-Ax PP3 | 6-Ax PP3 | 5-Ax Axes | 6-Ax Axes |")
        md.append("|------|---------|----------:|--------|--------|--------:|--------:|----------|----------|----------|----------|")
        for c in classification_changes:
            esm2 = f"{c['esm2_llr']:.2f}" if c["esm2_llr"] is not None else "null"
            md.append(
                f"| {c['gene']} | {c['hgvsp']} | {esm2} "
                f"| {c['five_axis_class']} | {c['six_axis_class']} "
                f"| {c['five_axis_points']} | {c['six_axis_points']} "
                f"| {c['five_axis_pp3']} | {c['six_axis_pp3']} "
                f"| {c['five_axis_axes']} | {c['six_axis_axes']} |"
            )
        md.append("")
    else:
        md.append("No variants were reclassified by adding ESM-2 (Axis 1).")
        md.append("")

    # ClinVar concordance
    md.append("---")
    md.append("")
    md.append("## 3. ClinVar Concordance")
    md.append("")
    md.append("| Pipeline | Concordant | Total | Rate |")
    md.append("|----------|----------:|------:|-----:|")
    md.append(f"| 6-axis | {conc_6} | {total_cv} | {100*rate_6:.1f}% |")
    md.append(f"| 5-axis | {conc_5} | {total_cv} | {100*rate_5:.1f}% |")
    md.append(f"| **Delta** | **{conc_6 - conc_5:+d}** | | **{100*(rate_6 - rate_5):+.1f} pp** |")
    md.append("")

    if concordance_changes:
        md.append("### Concordance Changes")
        md.append("")
        md.append("| Gene | Variant | ClinVar | 5-Axis | 6-Axis | 5-Ax Conc | 6-Ax Conc |")
        md.append("|------|---------|---------|--------|--------|-----------|-----------|")
        for c in concordance_changes:
            md.append(
                f"| {c['gene']} | {c['hgvsp']} | {c['clinvar']} "
                f"| {c['five_axis_class']} | {c['six_axis_class']} "
                f"| {'Y' if c['five_axis_concordant'] else 'N'} "
                f"| {'Y' if c['six_axis_concordant'] else 'N'} |"
            )
        md.append("")

    # Ablation robustness
    md.append("---")
    md.append("")
    md.append("## 4. Ablation Robustness")
    md.append("")
    md.append("Robustness = fraction of P/LP variants whose classification survives removal of any single axis.")
    md.append("")
    md.append("| Pipeline | Robust | P/LP Total | Robustness Rate |")
    md.append("|----------|-------:|-----------:|----------------:|")
    md.append(f"| 6-axis | {rob_6} | {plp_6} | {rob_rate_6:.1f}% |")
    md.append(f"| 5-axis | {rob_5} | {plp_5} | {rob_rate_5:.1f}% |")
    md.append(f"| **Delta** | **{rob_6 - rob_5:+d}** | | **{rob_rate_6 - rob_rate_5:+.1f} pp** |")
    md.append("")

    md.append("### Per-Axis Fragility")
    md.append("")
    md.append("| Axis | 6-Axis Fragile | 5-Axis Fragile | Delta |")
    md.append("|------|---------------:|---------------:|------:|")
    all_axes = ["Axis1_ProteinLM", "Axis2_StructureDL", "Axis3_Conservation",
                "Axis4_MetaEnsemble", "Axis5_Population", "Axis6_Functional"]
    for axis in all_axes:
        f6 = frag_6.get(axis, 0)
        f5 = frag_5.get(axis, 0)
        note = " (6-axis only)" if axis == "Axis1_ProteinLM" else ""
        md.append(f"| {axis}{note} | {f6} | {f5 if axis != 'Axis1_ProteinLM' else 'N/A'} | {f6 - f5 if axis != 'Axis1_ProteinLM' else 'N/A':>4} |")
    md.append("")

    # PP3 distribution
    md.append("---")
    md.append("")
    md.append("## 5. PP3 Strength Distribution")
    md.append("")
    md.append("| PP3 Strength | 6-Axis | 5-Axis | Delta |")
    md.append("|-------------|-------:|-------:|------:|")
    pp3_order = ["VeryStrong", "Strong", "Moderate", "Supporting", "Not_Met"]
    for s in pp3_order:
        n6 = six_pp3_dist.get(s, 0)
        n5 = five_pp3_dist.get(s, 0)
        md.append(f"| {s} | {n6} | {n5} | {n6 - n5:+d} |")
    md.append("")

    # ESM-2 impact detail
    md.append("---")
    md.append("")
    md.append("## 6. ESM-2 Per-Variant Impact")
    md.append("")
    md.append(f"Of {len(has_esm2)} variants with ESM-2 scores:")
    md.append(f"- **{len(pp3_changes)}** had PP3 strength changed")
    md.append(f"- **{len(class_changes_esm2)}** had final classification changed")
    md.append("")

    if pp3_changes:
        md.append("### PP3 Strength Changes")
        md.append("")
        md.append("| Gene | Variant | ESM-2 LLR | Path? | 5-Ax PP3 | 6-Ax PP3 | 5-Ax Axes | 6-Ax Axes | Class Changed |")
        md.append("|------|---------|----------:|------:|----------|----------|----------|----------|:-------------:|")
        for e in pp3_changes:
            md.append(
                f"| {e['gene']} | {e['hgvsp']} | {e['esm2_llr']:.2f} "
                f"| {'Y' if e['esm2_pathogenic'] else 'N'} "
                f"| {e['five_pp3']} | {e['six_pp3']} "
                f"| {e['five_axes']} | {e['six_axes']} "
                f"| {'Y' if e['class_changed'] else 'N'} |"
            )
        md.append("")

    # Points and margins
    md.append("---")
    md.append("")
    md.append("## 7. Points and Margin Distribution")
    md.append("")
    md.append("| Metric | 6-Axis | 5-Axis | Delta |")
    md.append("|--------|-------:|-------:|------:|")
    md.append(f"| Mean total points | {mean_pts_6:.2f} | {mean_pts_5:.2f} | {mean_pts_6 - mean_pts_5:+.2f} |")
    md.append(f"| Mean margin above LP boundary (P/LP only) | {mean_margin_6:.2f} | {mean_margin_5:.2f} | {mean_margin_6 - mean_margin_5:+.2f} |")
    md.append(f"| Mean axis concordance rate | {mean_conc_6:.4f} | {mean_conc_5:.4f} | {mean_conc_6 - mean_conc_5:+.4f} |")
    md.append("")

    # Summary verdict
    md.append("---")
    md.append("")
    md.append("## 8. Verdict: Does 6-axis outperform 5-axis?")
    md.append("")

    improvements = []
    unchanged = []
    degradations = []

    # P/LP count
    if six_plp > five_plp:
        improvements.append(f"P/LP count: {six_plp} vs {five_plp} (+{six_plp - five_plp})")
    elif six_plp == five_plp:
        unchanged.append(f"P/LP count: {six_plp} (identical)")
    else:
        degradations.append(f"P/LP count: {six_plp} vs {five_plp} ({six_plp - five_plp})")

    # ClinVar concordance
    if conc_6 > conc_5:
        improvements.append(f"ClinVar concordance: {100*rate_6:.1f}% vs {100*rate_5:.1f}% (+{100*(rate_6-rate_5):.1f} pp)")
    elif conc_6 == conc_5:
        unchanged.append(f"ClinVar concordance: {100*rate_6:.1f}% (identical)")
    else:
        degradations.append(f"ClinVar concordance: {100*rate_6:.1f}% vs {100*rate_5:.1f}% ({100*(rate_6-rate_5):.1f} pp)")

    # Ablation robustness
    if rob_rate_6 > rob_rate_5:
        improvements.append(f"Ablation robustness: {rob_rate_6:.1f}% vs {rob_rate_5:.1f}% (+{rob_rate_6 - rob_rate_5:.1f} pp)")
    elif rob_rate_6 == rob_rate_5:
        unchanged.append(f"Ablation robustness: {rob_rate_6:.1f}% (identical)")
    else:
        degradations.append(f"Ablation robustness: {rob_rate_6:.1f}% vs {rob_rate_5:.1f}% ({rob_rate_6 - rob_rate_5:.1f} pp)")

    # Mean points
    if mean_pts_6 > mean_pts_5:
        improvements.append(f"Mean points: {mean_pts_6:.2f} vs {mean_pts_5:.2f} (+{mean_pts_6 - mean_pts_5:.2f})")
    elif mean_pts_6 == mean_pts_5:
        unchanged.append(f"Mean points: {mean_pts_6:.2f} (identical)")
    else:
        degradations.append(f"Mean points: {mean_pts_6:.2f} vs {mean_pts_5:.2f} ({mean_pts_6 - mean_pts_5:.2f})")

    # Mean margin
    if mean_margin_6 > mean_margin_5:
        improvements.append(f"Mean margin: {mean_margin_6:.2f} vs {mean_margin_5:.2f} (+{mean_margin_6 - mean_margin_5:.2f})")
    elif mean_margin_6 == mean_margin_5:
        unchanged.append(f"Mean margin: {mean_margin_6:.2f} (identical)")
    else:
        degradations.append(f"Mean margin: {mean_margin_6:.2f} vs {mean_margin_5:.2f} ({mean_margin_6 - mean_margin_5:.2f})")

    # PP3 changes
    if pp3_changes:
        upgrades = sum(1 for c in pp3_changes
                       if _pp3_rank(c["six_pp3"]) > _pp3_rank(c["five_pp3"]))
        downgrades = sum(1 for c in pp3_changes
                         if _pp3_rank(c["six_pp3"]) < _pp3_rank(c["five_pp3"]))
        if upgrades > downgrades:
            improvements.append(f"PP3 strength: {upgrades} upgrades, {downgrades} downgrades")
        elif downgrades > upgrades:
            degradations.append(f"PP3 strength: {downgrades} downgrades, {upgrades} upgrades")
        else:
            unchanged.append(f"PP3 strength changes: {upgrades} up, {downgrades} down (balanced)")

    if improvements:
        md.append("**Improvements (6-axis better):**")
        md.append("")
        for item in improvements:
            md.append(f"- {item}")
        md.append("")
    if unchanged:
        md.append("**Unchanged:**")
        md.append("")
        for item in unchanged:
            md.append(f"- {item}")
        md.append("")
    if degradations:
        md.append("**Degradations (6-axis worse):**")
        md.append("")
        for item in degradations:
            md.append(f"- {item}")
        md.append("")

    # Final verdict
    md.append("### Conclusion")
    md.append("")
    if improvements and not degradations:
        md.append("The ISMB claim \"full pipeline outperforms 5-axis ablation\" is **SUPPORTED**.")
        md.append("The 6-axis pipeline strictly outperforms the 5-axis version on all changed metrics.")
    elif not improvements and not degradations:
        md.append("The ISMB claim \"full pipeline outperforms 5-axis ablation\" is **NOT SUPPORTED by the current data**.")
        md.append("The 6-axis and 5-axis pipelines produce identical results across all metrics.")
        md.append("")
        md.append("**Root cause:** ESM-2 scores, while available for missense variants, do not change any")
        md.append("variant's final ACMG classification because the axis concordance ratios remain above")
        md.append("or below PP3 thresholds regardless of ESM-2's contribution.")
    elif improvements and degradations:
        md.append("The ISMB claim has **MIXED** support. The 6-axis pipeline improves some metrics but degrades others.")
    else:
        md.append("The ISMB claim is **NOT SUPPORTED**. The 6-axis pipeline performs worse than 5-axis on all changed metrics.")
    md.append("")

    md_path = BENCHMARK_DIR / "five_vs_six_axis_comparison.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    log.info("Wrote %s", md_path)

    # Print summary to console
    print("\n" + "=" * 70)
    print("FIVE-AXIS vs SIX-AXIS COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nVariants: {n_total} total, {len(has_esm2)} with ESM-2 ({output['metadata']['esm2_coverage_pct']}%)")
    print(f"\nClassification:")
    print(f"  6-axis P/LP: {six_plp}  (P={six_counts['Pathogenic']}, LP={six_counts['Likely_Pathogenic']}, VUS={six_counts['VUS']})")
    print(f"  5-axis P/LP: {five_plp}  (P={five_counts['Pathogenic']}, LP={five_counts['Likely_Pathogenic']}, VUS={five_counts['VUS']})")
    print(f"  Delta P/LP:  {six_plp - five_plp:+d}")
    print(f"  Reclassified: {len(classification_changes)} variants")
    print(f"\nClinVar concordance:")
    print(f"  6-axis: {conc_6}/{total_cv} ({100*rate_6:.1f}%)")
    print(f"  5-axis: {conc_5}/{total_cv} ({100*rate_5:.1f}%)")
    print(f"  Delta: {conc_6 - conc_5:+d} ({100*(rate_6-rate_5):+.1f} pp)")
    print(f"\nAblation robustness:")
    print(f"  6-axis: {rob_6}/{plp_6} ({rob_rate_6:.1f}%)")
    print(f"  5-axis: {rob_5}/{plp_5} ({rob_rate_5:.1f}%)")
    print(f"  Delta: {rob_rate_6 - rob_rate_5:+.1f} pp")
    print(f"\nPP3 impact (variants with ESM-2):")
    print(f"  PP3 changed: {len(pp3_changes)}/{len(has_esm2)}")
    print(f"  Classification changed: {len(class_changes_esm2)}/{len(has_esm2)}")
    print(f"\nMean points: 6-axis={mean_pts_6:.2f}, 5-axis={mean_pts_5:.2f}, delta={mean_pts_6-mean_pts_5:+.2f}")
    print(f"Mean margin: 6-axis={mean_margin_6:.2f}, 5-axis={mean_margin_5:.2f}, delta={mean_margin_6-mean_margin_5:+.2f}")
    print("=" * 70)


def _pp3_rank(strength: str) -> int:
    """Rank PP3 strength for comparison."""
    ranks = {"Not_Met": 0, "Supporting": 1, "Moderate": 2, "Strong": 3, "VeryStrong": 4}
    return ranks.get(strength, -1)


if __name__ == "__main__":
    main()
