#!/usr/bin/env python3
"""
Ablation robustness analysis for all 154 benchmark variants.

For each variant classified as P or LP:
1. Count how many tools can be removed before classification drops
2. Compute the robustness margin (total_points - threshold)
3. Identify the most fragile and most robust variants

Outputs:
- benchmark/ablation_histogram.png -- histogram of robustness margins
- benchmark/ablation_analysis.md  -- fragile variants table + summary
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "ai_research" / "benchmark"
RESULTS_JSON = BENCHMARK_DIR / "benchmark_results.json"

# Tavtigian 2018 ACMG Bayesian thresholds
THRESHOLDS = {
    "Pathogenic": 10,
    "Likely_Pathogenic": 6,
}


def load_data() -> list[dict]:
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    return data["variants"]


def classify(total_points: int) -> str:
    if total_points >= 10:
        return "Pathogenic"
    if total_points >= 6:
        return "Likely_Pathogenic"
    if total_points >= 0:
        return "VUS"
    if total_points >= -6:
        return "Likely_Benign"
    return "Benign"


def analyze_variant(v: dict) -> dict | None:
    """Analyze ablation robustness for a single P/LP variant."""
    pipeline = v["pipeline"]
    cls = pipeline["classification"]
    if cls not in THRESHOLDS:
        return None

    total_points = pipeline["total_points"]
    threshold = THRESHOLDS[cls]
    margin = total_points - threshold

    ablation = v.get("ablation", {})
    tools_that_change = []
    tools_that_survive = []
    max_point_drop = 0

    for tool_name, result in ablation.items():
        new_pts = result["new_total_points"]
        point_drop = total_points - new_pts
        if point_drop > max_point_drop:
            max_point_drop = point_drop

        if result["changed"]:
            tools_that_change.append({
                "tool": tool_name.replace("remove_", ""),
                "new_points": new_pts,
                "new_class": result["new_classification"],
                "point_drop": point_drop,
            })
        else:
            tools_that_survive.append(tool_name.replace("remove_", ""))

    # Count how many single-tool removals the classification survives
    n_survivable = len(tools_that_survive)
    n_fatal = len(tools_that_change)
    n_total_tools = n_survivable + n_fatal

    return {
        "gene": v["gene"],
        "hgvsp": v["hgvsp"],
        "profile_id": v["profile_id"],
        "classification": cls,
        "total_points": total_points,
        "threshold": threshold,
        "margin": margin,
        "pp3_points": pipeline["pp3_points"],
        "pm2_points": pipeline["pm2_points"],
        "pp3_strength": pipeline["pp3_strength"],
        "axes_pathogenic": pipeline["axes_pathogenic"],
        "axes_total": pipeline["axes_total"],
        "n_tools_tested": n_total_tools,
        "n_survivable_removals": n_survivable,
        "n_fatal_removals": n_fatal,
        "tools_that_change": tools_that_change,
        "max_point_drop": max_point_drop,
        "robust_to_any_single": n_fatal == 0,
    }


def generate_histogram(results: list[dict], output_path: Path) -> None:
    """Generate histogram of robustness margins (points above threshold)."""
    margins = [r["margin"] for r in results]
    unique_margins = sorted(set(margins))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by margin level
    colors = []
    for m in unique_margins:
        if m == 0:
            colors.append("#e74c3c")  # red -- at threshold
        elif m <= 2:
            colors.append("#f39c12")  # orange -- fragile
        else:
            colors.append("#27ae60")  # green -- robust

    counts = [margins.count(m) for m in unique_margins]
    bars = ax.bar(unique_margins, counts, color=colors, edgecolor="black",
                  linewidth=0.5, width=0.8)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Robustness Margin (points above classification threshold)", fontsize=12)
    ax.set_ylabel("Number of Variants", fontsize=12)
    ax.set_title("Ablation Robustness of P/LP Benchmark Variants\n"
                 "(Tavtigian 2018 Bayesian point system)", fontsize=13)

    # Add threshold annotations
    ax.axvline(x=0, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(0.15, ax.get_ylim()[1] * 0.92, "threshold", color="#e74c3c",
            fontsize=9, fontstyle="italic")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", edgecolor="black", label="Margin = 0 (at threshold)"),
        Patch(facecolor="#f39c12", edgecolor="black", label="Margin 1-2 (fragile)"),
        Patch(facecolor="#27ae60", edgecolor="black", label="Margin 3+ (robust)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_xticks(unique_margins)
    ax.set_xlim(min(unique_margins) - 1, max(unique_margins) + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram: {output_path}")


def generate_report(results: list[dict], all_variants: list[dict], output_path: Path) -> None:
    """Generate ablation analysis markdown report."""
    n_total = len(all_variants)
    n_plp = len(results)
    n_vus = n_total - n_plp

    margins = [r["margin"] for r in results]
    mean_margin = np.mean(margins)
    median_margin = np.median(margins)
    min_margin = min(margins)
    max_margin = max(margins)

    n_robust = sum(1 for r in results if r["robust_to_any_single"])
    n_fragile_single = sum(1 for r in results if not r["robust_to_any_single"])
    pct_robust = 100.0 * n_robust / n_plp

    # Classify fragility tiers
    at_threshold = [r for r in results if r["margin"] == 0]
    fragile = [r for r in results if 0 < r["margin"] <= 2]
    robust = [r for r in results if r["margin"] >= 3]

    # Count which tools cause the most reclassifications
    tool_change_count: Counter = Counter()
    for r in results:
        for t in r["tools_that_change"]:
            tool_change_count[t["tool"]] += 1

    lines = [
        "# Ablation Robustness Analysis",
        "",
        f"**Date:** 2026-03-28",
        f"**Method:** Leave-one-tool-out ablation on Tavtigian 2018 Bayesian point system",
        f"**Source:** `benchmark_results.json` (154 variants, 20 patient profiles)",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total benchmark variants | {n_total} |",
        f"| Classified P or LP | {n_plp} ({100*n_plp/n_total:.1f}%) |",
        f"| Classified VUS | {n_vus} ({100*n_vus/n_total:.1f}%) |",
        f"| **Mean robustness margin** | **{mean_margin:.1f} points** |",
        f"| **Median robustness margin** | **{median_margin:.1f} points** |",
        f"| Min margin | {min_margin} points |",
        f"| Max margin | {max_margin} points |",
        f"| **Robust to any single removal** | **{n_robust}/{n_plp} ({pct_robust:.1f}%)** |",
        f"| Fragile (reclassified on single removal) | {n_fragile_single}/{n_plp} ({100*n_fragile_single/n_plp:.1f}%) |",
        "",
        "### Margin Distribution",
        "",
        f"| Margin | Count | % of P/LP |",
        f"|--------|-------|-----------|",
    ]

    margin_counts = Counter(margins)
    for m in sorted(margin_counts.keys()):
        c = margin_counts[m]
        lines.append(f"| {m} | {c} | {100*c/n_plp:.1f}% |")

    lines += [
        "",
        "### Classification Breakdown",
        "",
        f"| Classification | Threshold | Count |",
        f"|----------------|-----------|-------|",
        f"| Pathogenic | >= 10 pts | {sum(1 for r in results if r['classification'] == 'Pathogenic')} |",
        f"| Likely Pathogenic | >= 6 pts | {sum(1 for r in results if r['classification'] == 'Likely_Pathogenic')} |",
        "",
        "---",
        "",
        "## Tool Sensitivity",
        "",
        "Which tools, when removed, cause the most reclassifications:",
        "",
        f"| Tool | Reclassifications | % of P/LP |",
        f"|------|-------------------|-----------|",
    ]

    for tool, count in tool_change_count.most_common():
        lines.append(f"| {tool} | {count} | {100*count/n_plp:.1f}% |")

    if not tool_change_count:
        lines.append("| (none) | 0 | 0% |")

    lines += [
        "",
        "**Interpretation:** A tool that causes many reclassifications when removed is a critical dependency.",
        "If all P/LP variants survive every single-tool removal, the pipeline is robust against",
        "individual tool failures.",
        "",
        "---",
        "",
        "## Fragile Variants (margin <= 2)",
        "",
        "These variants are closest to the classification boundary and most vulnerable to tool removal.",
        "",
    ]

    fragile_all = sorted(at_threshold + fragile, key=lambda r: (r["margin"], r["gene"], r["hgvsp"]))

    if fragile_all:
        lines += [
            f"| Gene | Variant | Class | Points | Threshold | Margin | PP3 Strength | Fatal Tools |",
            f"|------|---------|-------|--------|-----------|--------|--------------|-------------|",
        ]
        for r in fragile_all:
            fatal_tools = ", ".join(t["tool"] for t in r["tools_that_change"]) or "none"
            lines.append(
                f"| {r['gene']} | {r['hgvsp']} | {r['classification']} | {r['total_points']} | "
                f"{r['threshold']} | {r['margin']} | {r['pp3_strength']} | {fatal_tools} |"
            )
    else:
        lines.append("*No fragile variants found.*")

    lines += [
        "",
        "---",
        "",
        "## High-Margin Variants (margin >= 3)",
        "",
        "These variants have the largest point buffers above their classification threshold,",
        "but may still depend on a single tool (see Fatal column in detailed table).",
        "",
    ]

    robust_sorted = sorted(robust, key=lambda r: (-r["margin"], r["gene"], r["hgvsp"]))

    if robust_sorted:
        lines += [
            f"| Gene | Variant | Class | Points | Threshold | Margin | PP3 Strength | Axes P/T |",
            f"|------|---------|-------|--------|-----------|--------|--------------|----------|",
        ]
        for r in robust_sorted:
            lines.append(
                f"| {r['gene']} | {r['hgvsp']} | {r['classification']} | {r['total_points']} | "
                f"{r['threshold']} | {r['margin']} | {r['pp3_strength']} | {r['axes_pathogenic']}/{r['axes_total']} |"
            )
    else:
        lines.append("*No robust variants found.*")

    lines += [
        "",
        "---",
        "",
        "## Detailed Ablation: All P/LP Variants",
        "",
        f"| # | Gene | Variant | Class | Points | Margin | Survivable | Fatal | Max Drop | Robust |",
        f"|---|------|---------|-------|--------|--------|------------|-------|----------|--------|",
    ]

    sorted_results = sorted(results, key=lambda r: (r["margin"], r["gene"], r["hgvsp"]))
    for i, r in enumerate(sorted_results, 1):
        robust_flag = "Yes" if r["robust_to_any_single"] else "**No**"
        lines.append(
            f"| {i} | {r['gene']} | {r['hgvsp']} | {r['classification']} | "
            f"{r['total_points']} | {r['margin']} | {r['n_survivable_removals']}/{r['n_tools_tested']} | "
            f"{r['n_fatal_removals']} | {r['max_point_drop']} | {robust_flag} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Methodology",
        "",
        "1. **Data source:** `benchmark_results.json` containing 154 variants from 20 GENIE patient profiles",
        "2. **Classification:** Tavtigian 2018 Bayesian point system (P >= 10, LP >= 6, VUS >= 0)",
        "3. **Ablation:** For each of 7 tools (ESM-2 LLR, AlphaMissense, EVE, CADD, REVEL, SIFT, PolyPhen-2),",
        "   the tool score is set to null and the variant is reclassified",
        "4. **Robustness margin:** `total_points - classification_threshold` (LP threshold = 6, P threshold = 10)",
        "5. **Robust to single removal:** Classification unchanged after removing any one tool",
        "6. **Fatal tool:** A tool whose removal causes reclassification (typically LP -> VUS)",
        "",
        "### Key Finding",
        "",
    ]

    if pct_robust < 50:
        lines.append(
            f"Only {pct_robust:.1f}% of P/LP variants are robust to any single-tool removal. "
            f"This indicates the pipeline relies heavily on individual tools for classification."
        )
    else:
        lines.append(
            f"{pct_robust:.1f}% of P/LP variants are robust to any single-tool removal. "
        )
        if pct_robust == 0:
            lines.append(
                "Every P/LP variant has at least one tool whose removal changes the classification. "
                "This is expected when classifications are near threshold boundaries."
            )
        elif pct_robust == 100:
            lines.append(
                "All P/LP variants maintain their classification even after any single tool is removed, "
                "indicating strong multi-tool concordance and pipeline robustness."
            )

    # Add AlphaMissense-specific note if it dominates
    if tool_change_count and tool_change_count.most_common(1)[0][0] == "alphamissense":
        am_count = tool_change_count["alphamissense"]
        am_pct = 100.0 * am_count / n_plp
        lines += [
            "",
            f"**AlphaMissense dependency:** {am_count}/{n_plp} ({am_pct:.1f}%) of P/LP variants are "
            f"reclassified when AlphaMissense is removed. AlphaMissense (DeepMind, 2023) is the "
            f"single most critical tool in the pipeline. Without it, these variants lose enough "
            f"axis concordance to drop below the PP3 VeryStrong threshold, falling from 8 to 4 "
            f"PP3 points (Strong), which reduces total points below the LP cutoff of 6.",
        ]

    lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"Saved report: {output_path}")


def main() -> None:
    variants = load_data()
    print(f"Loaded {len(variants)} benchmark variants")

    # Analyze all P/LP variants
    results = []
    for v in variants:
        r = analyze_variant(v)
        if r is not None:
            results.append(r)

    print(f"Found {len(results)} P/LP variants for ablation analysis")

    # Generate outputs
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    generate_histogram(results, BENCHMARK_DIR / "ablation_histogram.png")
    generate_report(results, variants, BENCHMARK_DIR / "ablation_analysis.md")

    # Print summary to stdout
    margins = [r["margin"] for r in results]
    n_robust = sum(1 for r in results if r["robust_to_any_single"])
    print(f"\n--- Ablation Robustness Summary ---")
    print(f"P/LP variants: {len(results)}")
    print(f"Mean margin: {np.mean(margins):.1f}")
    print(f"Median margin: {np.median(margins):.1f}")
    print(f"Range: {min(margins)} to {max(margins)}")
    print(f"Robust to any single removal: {n_robust}/{len(results)} ({100*n_robust/len(results):.1f}%)")

    # Count fragile
    tool_counts: Counter = Counter()
    for r in results:
        for t in r["tools_that_change"]:
            tool_counts[t["tool"]] += 1
    if tool_counts:
        print(f"\nMost critical tools (removal causes reclassification):")
        for tool, count in tool_counts.most_common():
            print(f"  {tool}: {count} variants affected")


if __name__ == "__main__":
    main()
