#!/usr/bin/env python3
"""
Combined ablation summary across all 40 benchmark profiles.

Loads v4 results (6-axis, axis-level ablation) from both batches,
computes cross-profile ablation robustness statistics, and generates
a 2-panel ISMB figure.

Usage:
    python mutation_profile/scripts/ai_research/benchmark_ablation_summary.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "results" / "ai_research" / "benchmark"

BATCH1 = BENCHMARK_DIR / "benchmark_results_v5.json"
BATCH2 = BENCHMARK_DIR / "benchmark_results_batch2_v5.json"


def load_variants(path: Path) -> list[dict]:
    if not path.exists():
        print(f"WARN: {path.name} not found, skipping")
        return []
    with open(path) as f:
        return json.load(f)["variants"]


def main() -> None:
    v1 = load_variants(BATCH1)
    v2 = load_variants(BATCH2)
    all_v = v1 + v2

    if not all_v:
        print("No results found. Run benchmark_profiles.py --output-suffix _v4 first.")
        sys.exit(1)

    n_total = len(all_v)
    pl_variants = [v for v in all_v if v["pipeline"]["classification"] in ("Pathogenic", "Likely_Pathogenic")]
    n_pl = len(pl_variants)

    print(f"Total variants: {n_total} ({len(v1)} batch1, {len(v2)} batch2)")
    print(f"P/LP variants: {n_pl}")

    # Axis-level ablation robustness
    axes = ["remove_Axis1_ProteinLM", "remove_Axis2_StructureDL", "remove_Axis3_Conservation",
            "remove_Axis4_MetaEnsemble", "remove_Axis5_Population", "remove_Axis6_Functional"]

    robust_count = 0
    fragile_by_axis = {a: 0 for a in axes}
    margins = []

    for v in pl_variants:
        abl = v.get("ablation", {})
        any_changed = False
        for axis in axes:
            entry = abl.get(axis, {})
            if entry.get("changed", False):
                any_changed = True
                fragile_by_axis[axis] += 1

        if not any_changed:
            robust_count += 1

        # Margin above threshold
        pts = v["pipeline"]["total_points"]
        threshold = 10 if v["pipeline"]["classification"] == "Pathogenic" else 6
        margins.append(pts - threshold)

    pct_robust = 100 * robust_count / n_pl if n_pl else 0

    print(f"\nAxis-level ablation robustness:")
    print(f"  Robust to any single axis removal: {robust_count}/{n_pl} ({pct_robust:.1f}%)")
    print(f"\nFragile by axis (how many P/LP variants reclassify when axis removed):")
    for axis in axes:
        short = axis.replace("remove_", "")
        n = fragile_by_axis[axis]
        print(f"  {short:25s} {n:4d}/{n_pl} ({100*n/n_pl:.1f}%)")

    print(f"\nMargin distribution (points above threshold):")
    print(f"  Mean: {np.mean(margins):.1f}")
    print(f"  Median: {np.median(margins):.1f}")
    print(f"  Min: {np.min(margins)}")
    print(f"  Max: {np.max(margins)}")

    # ClinVar concordance for 6-axis run
    def norm_cv(raw):
        if not raw: return None
        r = raw.lower()
        if "pathogenic" in r and "likely" in r: return "LP"
        if "pathogenic" in r: return "P"
        if "uncertain" in r or "vus" in r: return "VUS"
        if "benign" in r: return "B"
        return None

    cv_variants = [(v, norm_cv(v["clinvar"]["classification"])) for v in all_v if v["clinvar"]["classification"]]
    concordant = sum(
        1 for v, cv in cv_variants
        if (cv in ("P", "LP") and v["pipeline"]["classification"] in ("Pathogenic", "Likely_Pathogenic"))
        or (cv == "VUS" and v["pipeline"]["classification"] == "VUS")
        or (cv == "B" and v["pipeline"]["classification"] in ("VUS", "Likely_Benign", "Benign"))
    )

    print(f"\n6-axis ClinVar concordance: {concordant}/{len(cv_variants)} ({100*concordant/len(cv_variants):.1f}%)")

    # ── Generate 2-panel figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Ablation fragility by axis
    axis_labels = ["Protein LM\n(ESM-2)", "Structure DL\n(AlphaMissense)", "Conservation\n(EVE+SIFT)",
                   "Meta-ensemble\n(CADD+REVEL)", "Population\n(gnomAD)", "Functional\n(PolyPhen-2)"]
    frag_pcts = [100 * fragile_by_axis[a] / n_pl for a in axes]
    colors = ["#e74c3c" if p > 30 else "#f39c12" if p > 10 else "#27ae60" for p in frag_pcts]

    bars = ax1.barh(axis_labels, frag_pcts, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("% of P/LP variants reclassified on axis removal", fontsize=11)
    ax1.set_title(f"Axis-Level Ablation Fragility\n(N={n_pl} P/LP variants, 40 profiles)", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 100)
    for bar, pct in zip(bars, frag_pcts):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{pct:.0f}%", va="center", fontsize=10)
    ax1.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    ax1.invert_yaxis()

    # Panel B: Robustness margin distribution
    ax2.hist(margins, bins=range(min(margins), max(margins)+2), color="#3498db",
             edgecolor="black", linewidth=0.5, alpha=0.85)
    ax2.axvline(x=0, color="red", linewidth=2, label="Classification threshold")
    ax2.set_xlabel("Points above classification threshold", fontsize=11)
    ax2.set_ylabel("Number of P/LP variants", fontsize=11)
    ax2.set_title(f"Robustness Margin Distribution\n({pct_robust:.0f}% robust to single-axis removal)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    fig_path = BENCHMARK_DIR / "ablation_6axis_figure.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # Save summary JSON
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_variants": n_total,
        "pl_variants": n_pl,
        "robust_to_single_axis_removal": robust_count,
        "robustness_rate": round(pct_robust, 2),
        "fragile_by_axis": {a.replace("remove_", ""): fragile_by_axis[a] for a in axes},
        "margin_stats": {
            "mean": round(float(np.mean(margins)), 1),
            "median": round(float(np.median(margins)), 1),
            "min": int(np.min(margins)),
            "max": int(np.max(margins)),
        },
        "clinvar_concordance": {
            "concordant": concordant,
            "total": len(cv_variants),
            "rate": round(concordant / len(cv_variants), 4) if cv_variants else 0,
        },
    }
    json_path = BENCHMARK_DIR / "ablation_6axis_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {json_path}")

    # Save markdown report
    md = [
        "# Six-Axis Ablation Robustness: 40 SETBP1+ Myeloid Profiles",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Variants:** {n_total} total, {n_pl} classified P/LP",
        f"**Profiles:** 40 (20 batch 1, 20 batch 2)",
        "",
        "## Key Finding",
        "",
        f"**{robust_count}/{n_pl} ({pct_robust:.1f}%) of P/LP variants are robust to removal of any single axis.**",
        "",
        "## Per-Axis Fragility",
        "",
        "| Axis | Tools | Reclassified | % |",
        "|------|-------|-------------|---|",
    ]
    for axis, label in zip(axes, axis_labels):
        n = fragile_by_axis[axis]
        md.append(f"| {label.replace(chr(10), ' ')} | {', '.join([a.replace('remove_','') for a in [axis]])} | {n}/{n_pl} | {100*n/n_pl:.1f}% |")

    md.extend([
        "",
        f"## ClinVar Concordance (6-axis): {concordant}/{len(cv_variants)} ({100*concordant/len(cv_variants):.1f}%)",
        "",
        f"## Margin Distribution: mean={np.mean(margins):.1f}, median={np.median(margins):.1f}, range=[{np.min(margins)}, {np.max(margins)}]",
    ])

    md_path = BENCHMARK_DIR / "ablation_6axis_report.md"
    md_path.write_text("\n".join(md))
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
