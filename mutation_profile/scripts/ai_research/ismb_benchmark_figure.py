#!/usr/bin/env python3
"""
Generate definitive 2-panel ISMB benchmark figure.

Panel A: Per-profile concordance across 40 myeloid profiles
Panel B: Leave-one-out axis importance (ablation analysis)

Uses data from:
  - benchmark_results_v4.json (batch 1, 20 profiles)
  - benchmark_results_batch2_v4.json (batch 2, 20 profiles)
  - benchmark_profiles.json (batch 1 OncoTree codes)
  - benchmark_profiles_batch2.json (batch 2 OncoTree codes)
  - ablation_6axis_summary.json (axis fragility)

Output:
  - ismb_benchmark_figure_final.png (300 DPI)
  - ismb_benchmark_figure_final_preview.png (150 DPI)
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

# Paths
BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "ai_research" / "benchmark"
RESULTS_V4_B1 = BENCH_DIR / "benchmark_results_v4.json"
RESULTS_V4_B2 = BENCH_DIR / "benchmark_results_batch2_v4.json"
PROFILES_B1 = BENCH_DIR / "benchmark_profiles.json"
PROFILES_B2 = BENCH_DIR / "benchmark_profiles_batch2.json"
ABLATION_SUMMARY = BENCH_DIR / "ablation_6axis_summary.json"
OUT_FINAL = BENCH_DIR / "ismb_benchmark_figure_final.png"
OUT_PREVIEW = BENCH_DIR / "ismb_benchmark_figure_final_preview.png"


# ---------------------------------------------------------------------------
# Colorblind-safe palette for OncoTree disease codes
# Uses Wong (2011) palette — verified distinguishable under deuteranopia,
# protanopia, and tritanopia.
# ---------------------------------------------------------------------------
DISEASE_COLORS = {
    "AML": "#D55E00",      # vermillion (red-orange)
    "MDS": "#0072B2",      # blue
    "CMML": "#009E73",     # bluish-green
    "MPN": "#CC79A7",      # reddish-purple
    "CNL": "#E69F00",      # orange
    "MDS/MPN": "#56B4E9",  # sky blue
    "MDSMPNU": "#56B4E9",  # sky blue (same family as MDS/MPN)
    "ET": "#F0E442",       # yellow
    "PMF": "#CC79A7",      # reddish-purple (same family as MPN)
}

# Axis display names (short)
AXIS_NAMES = {
    "Axis1_ProteinLM": "Protein LM\n(ESM-2)",
    "Axis2_StructureDL": "Structure DL\n(AlphaMissense)",
    "Axis3_Conservation": "Conservation\n(EVE, SIFT)",
    "Axis4_MetaEnsemble": "Meta-ensemble\n(CADD, REVEL)",
    "Axis5_Population": "Population\n(gnomAD)",
    "Axis6_Functional": "Functional\n(PolyPhen-2)",
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_profile_oncotree_map(profiles_json: dict) -> dict[str, str]:
    """Map patient_id -> OncoTree code from profiles JSON."""
    mapping = {}
    for p in profiles_json["profiles"]:
        mapping[p["patient_id"]] = p["oncotree_code"]
    return mapping


def normalize_oncotree(code: str) -> str:
    """Normalize OncoTree subtypes to broad categories for coloring."""
    code_upper = code.upper()
    if code_upper in ("AML", "AML-MRC", "AML-NOS", "APL", "RAEB-T"):
        return "AML"
    if code_upper in ("MDS", "MDS-EB1", "MDS-EB2", "MDS-MLD", "MDS-RS",
                       "MDS-SLD", "MDS-U", "RAEB", "RARS", "RCMD"):
        return "MDS"
    if code_upper in ("CMML", "CMML-1", "CMML-2", "JMML", "ACML"):
        return "CMML"
    if code_upper in ("MPN", "MPN-U", "PV"):
        return "MPN"
    if code_upper in ("MDSMPNU", "MDS/MPN"):
        return "MDS/MPN"
    if code_upper == "ET":
        return "ET"
    if code_upper == "PMF":
        return "PMF"
    if code_upper == "CNL":
        return "CNL"
    return code


def compute_per_profile_stats(
    results_json: dict,
    oncotree_map: dict[str, str],
) -> list[dict]:
    """Compute concordance rate and variant count per profile."""
    profiles: dict[str, dict] = defaultdict(lambda: {
        "total_variants": 0,
        "clinvar_total": 0,
        "clinvar_concordant": 0,
        "oncotree_raw": "Unknown",
    })

    for v in results_json["variants"]:
        pid = v["profile_id"]
        profiles[pid]["total_variants"] += 1
        profiles[pid]["oncotree_raw"] = oncotree_map.get(pid, "Unknown")

        clinvar_norm = v["clinvar"]["normalized"]
        pipeline_class = v["pipeline"]["classification"]

        if clinvar_norm == "Not_in_ClinVar":
            continue

        profiles[pid]["clinvar_total"] += 1

        # Concordance: P/LP matches P/LP, VUS matches VUS, B/LB matches B/LB
        cv_group = _classify_group(clinvar_norm)
        pl_group = _classify_group(pipeline_class)
        if cv_group == pl_group:
            profiles[pid]["clinvar_concordant"] += 1

    result = []
    for pid, stats in profiles.items():
        if stats["clinvar_total"] > 0:
            concordance = stats["clinvar_concordant"] / stats["clinvar_total"] * 100
        else:
            concordance = None  # no ClinVar entries to compare
        result.append({
            "profile_id": pid,
            "n_variants": stats["total_variants"],
            "clinvar_total": stats["clinvar_total"],
            "clinvar_concordant": stats["clinvar_concordant"],
            "concordance_pct": concordance,
            "oncotree_raw": stats["oncotree_raw"],
            "oncotree_broad": normalize_oncotree(stats["oncotree_raw"]),
        })

    return result


def _classify_group(classification: str) -> str:
    if classification in ("Pathogenic", "Likely_Pathogenic"):
        return "P/LP"
    if classification in ("Benign", "Likely_Benign"):
        return "B/LB"
    return "VUS"


def make_figure(profiles: list[dict], ablation: dict) -> tuple[plt.Figure, plt.Axes]:
    """Create the 2-panel figure."""

    # Filter profiles with ClinVar data and sort by concordance (descending)
    profiles_with_cv = [p for p in profiles if p["concordance_pct"] is not None]
    profiles_with_cv.sort(key=lambda p: p["concordance_pct"], reverse=True)

    # Also include profiles without ClinVar (show as 0-height bars with annotation)
    profiles_no_cv = [p for p in profiles if p["concordance_pct"] is None]

    all_profiles = profiles_with_cv + profiles_no_cv
    n_profiles = len(all_profiles)

    mean_concordance = np.mean([p["concordance_pct"] for p in profiles_with_cv])

    # --- Figure setup ---
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(18, 7.5),
        gridspec_kw={"width_ratios": [2.8, 1], "wspace": 0.32},
    )

    # =====================================================================
    # Panel A: Per-profile concordance
    # =====================================================================
    x = np.arange(n_profiles)
    bar_colors = [DISEASE_COLORS.get(p["oncotree_broad"], "#999999") for p in all_profiles]

    concordance_vals = [
        p["concordance_pct"] if p["concordance_pct"] is not None else 0
        for p in all_profiles
    ]

    bars = ax_a.bar(
        x, concordance_vals, color=bar_colors, edgecolor="white", linewidth=0.5,
        width=0.78, zorder=3,
    )

    # Mean concordance line
    ax_a.axhline(
        mean_concordance, color="#333333", linestyle="--", linewidth=1.2,
        zorder=4, label=f"Mean = {mean_concordance:.1f}%",
    )
    ax_a.text(
        n_profiles - 0.5, mean_concordance + 1.5,
        f"Mean {mean_concordance:.1f}%",
        ha="right", va="bottom", fontsize=8.5, color="#333333",
        fontweight="bold",
    )

    # N variants annotation on each bar
    for i, p in enumerate(all_profiles):
        n = p["n_variants"]
        conc = concordance_vals[i]
        # Place inside bar for tall bars, above for short bars
        if conc >= 70:
            ax_a.text(
                i, conc * 0.5, f"n={n}",
                ha="center", va="center", fontsize=5.8, color="white",
                fontweight="bold", rotation=90,
            )
        elif conc >= 20:
            ax_a.text(
                i, conc + 2, f"n={n}",
                ha="center", va="bottom", fontsize=5.5, color="#555555",
                rotation=90,
            )
        else:
            ax_a.text(
                i, max(conc, 1) + 2, f"n={n}",
                ha="center", va="bottom", fontsize=5.5, color="#555555",
                rotation=90,
            )

    # Anonymized profile labels
    profile_labels = [f"P{i+1:02d}" for i in range(n_profiles)]
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(profile_labels, fontsize=6, rotation=45, ha="right")
    ax_a.set_xlim(-0.6, n_profiles - 0.4)
    ax_a.set_ylim(0, 108)
    ax_a.set_ylabel("ClinVar Concordance (%)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Patient Profiles (sorted by concordance)", fontsize=10)
    ax_a.set_title(
        "A   Pipeline Performance Across 40 Myeloid Profiles",
        fontsize=13, fontweight="bold", loc="left", pad=12,
    )

    # Y-axis formatting
    ax_a.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax_a.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax_a.tick_params(axis="y", labelsize=9)

    # Grid
    ax_a.grid(axis="y", alpha=0.3, linestyle="-", zorder=0)
    ax_a.set_axisbelow(True)

    # Disease legend
    unique_diseases = sorted(set(p["oncotree_broad"] for p in all_profiles))
    legend_patches = [
        Patch(facecolor=DISEASE_COLORS.get(d, "#999999"), edgecolor="white", label=d)
        for d in unique_diseases
    ]
    # Add count per disease
    disease_counts = defaultdict(int)
    for p in all_profiles:
        disease_counts[p["oncotree_broad"]] += 1
    legend_patches_labeled = [
        Patch(
            facecolor=DISEASE_COLORS.get(d, "#999999"),
            edgecolor="white",
            label=f"{d} (n={disease_counts[d]})",
        )
        for d in unique_diseases
    ]
    legend = ax_a.legend(
        handles=legend_patches_labeled,
        loc="upper right",
        fontsize=7.5,
        title="OncoTree",
        title_fontsize=8.5,
        framealpha=0.9,
        edgecolor="#cccccc",
        ncol=2,
    )
    legend.get_frame().set_linewidth(0.5)

    # Summary stats annotation
    n_with_cv = len(profiles_with_cv)
    total_cv_variants = sum(p["clinvar_total"] for p in profiles_with_cv)
    total_concordant = sum(p["clinvar_concordant"] for p in profiles_with_cv)
    overall_concordance = total_concordant / total_cv_variants * 100 if total_cv_variants else 0

    n_perfect = sum(1 for p in profiles_with_cv if p["concordance_pct"] == 100)
    total_variants_all = sum(p["n_variants"] for p in all_profiles)

    summary_text = (
        f"N={n_profiles} profiles, {total_variants_all} variants\n"
        f"Overall: {total_concordant}/{total_cv_variants} ({overall_concordance:.1f}%)\n"
        f"Perfect concordance: {n_perfect}/{n_with_cv} profiles"
    )
    ax_a.text(
        0.02, 0.97, summary_text,
        transform=ax_a.transAxes,
        fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.95),
    )

    # Spines
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # =====================================================================
    # Panel B: Leave-one-out axis importance
    # =====================================================================
    fragile = ablation["fragile_by_axis"]
    total_pl = ablation["pl_variants"]
    robustness_rate = ablation["robustness_rate"]

    axis_keys = [
        "Axis1_ProteinLM",
        "Axis2_StructureDL",
        "Axis3_Conservation",
        "Axis4_MetaEnsemble",
        "Axis5_Population",
        "Axis6_Functional",
    ]

    axis_labels = [AXIS_NAMES[k] for k in axis_keys]
    fragile_counts = [fragile[k] for k in axis_keys]
    fragile_pcts = [c / total_pl * 100 for c in fragile_counts]

    # Sort by fragility (largest impact at top)
    sorted_indices = np.argsort(fragile_pcts)[::-1]
    axis_labels_sorted = [axis_labels[i] for i in sorted_indices]
    fragile_pcts_sorted = [fragile_pcts[i] for i in sorted_indices]
    fragile_counts_sorted = [fragile_counts[i] for i in sorted_indices]

    # Color gradient: green (low impact) -> yellow -> red (high impact)
    max_pct = max(fragile_pcts_sorted) if fragile_pcts_sorted else 1
    cmap = plt.cm.RdYlGn_r  # reversed so green=low, red=high
    norm_vals = [v / max_pct if max_pct > 0 else 0 for v in fragile_pcts_sorted]
    bar_colors_b = [cmap(0.15 + 0.7 * nv) for nv in norm_vals]

    y_pos = np.arange(len(axis_keys))

    hbars = ax_b.barh(
        y_pos, fragile_pcts_sorted, color=bar_colors_b,
        edgecolor="white", linewidth=0.5, height=0.65, zorder=3,
    )

    # Percentage labels on bars
    for i, (pct, count) in enumerate(zip(fragile_pcts_sorted, fragile_counts_sorted)):
        if pct > 0:
            ax_b.text(
                pct + 0.8, i,
                f"{pct:.1f}%  ({count}/{total_pl})",
                va="center", ha="left", fontsize=8.5, fontweight="bold",
                color="#333333",
            )
        else:
            ax_b.text(
                0.8, i,
                f"0.0%  (0/{total_pl})",
                va="center", ha="left", fontsize=8.5, fontweight="bold",
                color="#888888",
            )

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(axis_labels_sorted, fontsize=9)
    ax_b.invert_yaxis()  # highest impact at top
    ax_b.set_xlabel("Classifications Changed (%)", fontsize=10, fontweight="bold")
    ax_b.set_title(
        f"B   Axis Importance: Leave-One-Out\n"
        f"     Robustness: {robustness_rate:.1f}% of P/LP survive any single-axis removal",
        fontsize=12, fontweight="bold", loc="left", pad=12,
    )

    # X-axis
    ax_b.set_xlim(0, max(fragile_pcts_sorted) * 1.45 if fragile_pcts_sorted else 50)
    ax_b.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax_b.tick_params(axis="x", labelsize=9)

    # Grid
    ax_b.grid(axis="x", alpha=0.3, linestyle="-", zorder=0)
    ax_b.set_axisbelow(True)

    # Summary annotation
    robust_n = ablation["robust_to_single_axis_removal"]
    summary_b = (
        f"N={total_pl} P/LP variants tested\n"
        f"Robust: {robust_n} ({robustness_rate:.1f}%)\n"
        f"Fragile: {total_pl - robust_n} ({100 - robustness_rate:.1f}%)"
    )
    ax_b.text(
        0.98, 0.05, summary_b,
        transform=ax_b.transAxes,
        fontsize=7.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.95),
    )

    # Spines
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Overall figure
    fig.suptitle(
        "Six-Axis ACMG Bayesian Classification Benchmark  |  "
        "GENIE v19.0 SETBP1-Positive Myeloid Cohort",
        fontsize=12, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.01,
        "Source: AACR GENIE v19.0-public  |  Method: Tavtigian et al. (2018) Bayesian framework  |  "
        "Six orthogonal computational axes",
        ha="center", fontsize=7.5, color="#777777",
    )

    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.12, top=0.87, wspace=0.35)

    return fig, (ax_a, ax_b)


def main():
    print("Loading data...")

    # Load profile metadata (OncoTree codes)
    profiles_b1_json = load_json(PROFILES_B1)
    profiles_b2_json = load_json(PROFILES_B2)
    oncotree_map = {}
    oncotree_map.update(build_profile_oncotree_map(profiles_b1_json))
    oncotree_map.update(build_profile_oncotree_map(profiles_b2_json))
    print(f"  OncoTree mapping: {len(oncotree_map)} profiles")

    # Load scored results
    results_b1 = load_json(RESULTS_V4_B1)
    results_b2 = load_json(RESULTS_V4_B2)
    print(f"  Batch 1: {results_b1['metadata']['n_profiles']} profiles, "
          f"{results_b1['metadata']['n_variants']} variants")
    print(f"  Batch 2: {results_b2['metadata']['n_profiles']} profiles, "
          f"{results_b2['metadata']['n_variants']} variants")

    # Compute per-profile concordance
    stats_b1 = compute_per_profile_stats(results_b1, oncotree_map)
    stats_b2 = compute_per_profile_stats(results_b2, oncotree_map)
    all_stats = stats_b1 + stats_b2
    print(f"  Combined: {len(all_stats)} profiles")

    # Load ablation summary
    ablation = load_json(ABLATION_SUMMARY)
    print(f"  Ablation: {ablation['pl_variants']} P/LP, "
          f"robustness {ablation['robustness_rate']:.1f}%")

    # Generate figure
    print("Generating figure...")
    fig, axes = make_figure(all_stats, ablation)

    # Save 300 DPI (publication)
    fig.savefig(OUT_FINAL, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {OUT_FINAL}")

    # Save 150 DPI (preview)
    fig.savefig(OUT_PREVIEW, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {OUT_PREVIEW}")

    plt.close(fig)

    # Print summary statistics
    profiles_with_cv = [p for p in all_stats if p["concordance_pct"] is not None]
    profiles_no_cv = [p for p in all_stats if p["concordance_pct"] is None]
    print(f"\n--- Summary ---")
    print(f"Total profiles: {len(all_stats)}")
    print(f"Profiles with ClinVar data: {len(profiles_with_cv)}")
    print(f"Profiles without ClinVar data: {len(profiles_no_cv)}")
    if profiles_with_cv:
        conc_vals = [p["concordance_pct"] for p in profiles_with_cv]
        print(f"Mean concordance: {np.mean(conc_vals):.1f}%")
        print(f"Median concordance: {np.median(conc_vals):.1f}%")
        print(f"Range: {min(conc_vals):.1f}% - {max(conc_vals):.1f}%")
        n_perfect = sum(1 for v in conc_vals if v == 100)
        print(f"Perfect (100%): {n_perfect}/{len(profiles_with_cv)}")

    # Disease distribution
    from collections import Counter
    disease_dist = Counter(p["oncotree_broad"] for p in all_stats)
    print(f"\nDisease distribution:")
    for disease, count in disease_dist.most_common():
        print(f"  {disease}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
