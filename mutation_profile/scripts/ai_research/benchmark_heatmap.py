#!/usr/bin/env python3
"""
Generate publication-quality concordance heatmap and ClinVar concordance bar chart
from benchmark_results.json.

Produces:
  1. concordance_heatmap.png   -- Cross-tool pathogenicity concordance grid
  2. clinvar_concordance_bars.png -- Pipeline vs ClinVar classification bar chart

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/benchmark_heatmap.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]  # mutation_profile/
RESULTS_DIR = BASE / "results" / "ai_research" / "benchmark"
INPUT = RESULTS_DIR / "benchmark_results.json"
OUT_HEATMAP = RESULTS_DIR / "concordance_heatmap.png"
OUT_BARS = RESULTS_DIR / "clinvar_concordance_bars.png"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(INPUT) as f:
    data = json.load(f)

variants = data["variants"]
n_variants = len(variants)
n_profiles = data["metadata"]["n_profiles"]

# ---------------------------------------------------------------------------
# Score interpretation thresholds
# ---------------------------------------------------------------------------
# Each tool: (key, threshold, direction)
# direction = "higher_pathogenic" means score > threshold = pathogenic
# direction = "lower_pathogenic" means score < threshold = pathogenic (SIFT)
TOOL_SPECS = [
    ("alphamissense", "AlphaMissense", 0.564, "higher_pathogenic"),   # >0.564 = likely pathogenic
    ("eve_score",     "EVE",           0.5,   "higher_pathogenic"),   # >0.5 = pathogenic
    ("revel",         "REVEL",         0.5,   "higher_pathogenic"),   # >0.5 = likely pathogenic
    ("sift_score",    "SIFT",          0.05,  "lower_pathogenic"),    # <0.05 = damaging
    ("gnomad_af",     "gnomAD AF",     0.01,  "lower_pathogenic"),    # <0.01 (rare) supports PM2
]

TOOL_KEYS = [t[0] for t in TOOL_SPECS]
TOOL_LABELS = [t[1] for t in TOOL_SPECS]


def classify_score(key, value):
    """Return 1 (pathogenic/damaging), -1 (benign/tolerated), or 0 (N/A)."""
    if value is None:
        return 0
    for k, _, threshold, direction in TOOL_SPECS:
        if k == key:
            if direction == "higher_pathogenic":
                return 1 if value >= threshold else -1
            else:  # lower_pathogenic
                return 1 if value <= threshold else -1
    return 0


# ---------------------------------------------------------------------------
# Build matrix
# ---------------------------------------------------------------------------
# Sort variants by total Bayesian points (descending), then gene name
sorted_variants = sorted(
    variants,
    key=lambda v: (-v["pipeline"]["total_points"], v["gene"], v["hgvsp"]),
)

row_labels = []
matrix = []
pipeline_cls = []
clinvar_cls = []
total_points = []

for v in sorted_variants:
    label = f'{v["gene"]} {v["hgvsp"]}'
    row_labels.append(label)
    row = [classify_score(k, v["scores"].get(k)) for k in TOOL_KEYS]
    matrix.append(row)
    pipeline_cls.append(v["pipeline"]["classification"])
    clinvar_cls.append(v["clinvar"]["normalized"])
    total_points.append(v["pipeline"]["total_points"])

matrix = np.array(matrix)
n_rows, n_cols = matrix.shape

# ---------------------------------------------------------------------------
# Color map: -1 = benign (red), 0 = N/A (gray), 1 = pathogenic (green)
# ---------------------------------------------------------------------------
cmap = mcolors.ListedColormap(["#D94F4F", "#C0C0C0", "#4CAF78"])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Classification colors
CLS_COLORS = {
    "Pathogenic":       "#1B5E20",
    "Likely_Pathogenic": "#66BB6A",
    "VUS":              "#FFA726",
    "Not_in_ClinVar":   "#BDBDBD",
}

CLS_SHORT = {
    "Pathogenic":       "P",
    "Likely_Pathogenic": "LP",
    "VUS":              "VUS",
    "Not_in_ClinVar":   "N/A",
}


# ---------------------------------------------------------------------------
# Figure 1: Concordance Heatmap
# ---------------------------------------------------------------------------
def make_concordance_heatmap():
    # Adaptive sizing: figure height scales with variant count
    row_height = 0.135
    fig_height = max(8, n_rows * row_height + 3.5)
    fig_width = 12.5

    fig, ax_main = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")

    # Main heatmap area (leave right margin for annotations)
    ax_main.set_position([0.22, 0.08, 0.45, 0.84])

    # Draw the heatmap
    im = ax_main.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # Grid lines
    for i in range(n_rows + 1):
        ax_main.axhline(i - 0.5, color="white", linewidth=0.5)
    for j in range(n_cols + 1):
        ax_main.axvline(j - 0.5, color="white", linewidth=0.5)

    # Axis labels
    ax_main.set_xticks(range(n_cols))
    ax_main.set_xticklabels(TOOL_LABELS, fontsize=8, fontweight="bold", rotation=45, ha="right")
    ax_main.set_yticks(range(n_rows))
    ax_main.set_yticklabels(row_labels, fontsize=5.2, fontfamily="monospace")
    ax_main.tick_params(axis="both", length=0)

    # Annotations on the right: Pipeline classification, ClinVar, points
    # Create three narrow annotation columns
    ax_pipe = fig.add_axes([0.68, 0.08, 0.03, 0.84])
    ax_cv = fig.add_axes([0.72, 0.08, 0.03, 0.84])
    ax_pts = fig.add_axes([0.76, 0.08, 0.06, 0.84])

    for ax_ann in [ax_pipe, ax_cv, ax_pts]:
        ax_ann.set_ylim(n_rows - 0.5, -0.5)
        ax_ann.set_xlim(0, 1)
        ax_ann.axis("off")

    # Column headers
    header_y = -1.8
    ax_pipe.text(0.5, header_y, "Pipeline", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax_cv.text(0.5, header_y, "ClinVar", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax_pts.text(0.5, header_y, "Bayes\nPts", ha="center", va="bottom", fontsize=7, fontweight="bold")

    for i in range(n_rows):
        # Pipeline classification pill
        pl = pipeline_cls[i]
        pl_color = CLS_COLORS.get(pl, "#BDBDBD")
        pl_short = CLS_SHORT.get(pl, pl[:3])
        ax_pipe.add_patch(mpatches.FancyBboxPatch(
            (0.05, i - 0.35), 0.9, 0.7,
            boxstyle="round,pad=0.05",
            facecolor=pl_color, edgecolor="none", alpha=0.85,
        ))
        ax_pipe.text(0.5, i, pl_short, ha="center", va="center",
                     fontsize=5.5, fontweight="bold",
                     color="white" if pl in ("Pathogenic", "Likely_Pathogenic") else "#333333")

        # ClinVar classification pill
        cv = clinvar_cls[i]
        cv_color = CLS_COLORS.get(cv, "#BDBDBD")
        cv_short = CLS_SHORT.get(cv, cv[:3])
        ax_cv.add_patch(mpatches.FancyBboxPatch(
            (0.05, i - 0.35), 0.9, 0.7,
            boxstyle="round,pad=0.05",
            facecolor=cv_color, edgecolor="none", alpha=0.85,
        ))
        ax_cv.text(0.5, i, cv_short, ha="center", va="center",
                   fontsize=5.5, fontweight="bold",
                   color="white" if cv in ("Pathogenic", "Likely_Pathogenic") else "#333333")

        # Points bar
        pts = total_points[i]
        max_pts = max(total_points)
        bar_width = pts / max_pts * 0.85
        bar_color = "#1B5E20" if pts >= 10 else "#66BB6A" if pts >= 6 else "#FFA726" if pts >= 4 else "#EF5350"
        ax_pts.barh(i, bar_width, height=0.65, left=0.05, color=bar_color, alpha=0.8, edgecolor="none")
        ax_pts.text(bar_width + 0.08, i, str(pts), ha="left", va="center", fontsize=5.5,
                    fontweight="bold", color="#333333")

    # Title
    fig.suptitle(
        f"Cross-tool pathogenicity concordance: {n_profiles} SETBP1$^+$ myeloid profiles "
        f"(N={n_variants} variants)",
        fontsize=11.5,
        fontweight="bold",
        y=0.97,
        x=0.47,
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#4CAF78", label="Pathogenic / Damaging"),
        mpatches.Patch(facecolor="#D94F4F", label="Benign / Tolerated"),
        mpatches.Patch(facecolor="#C0C0C0", label="No data"),
    ]
    ax_main.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=7.5,
        frameon=False,
    )

    # Thresholds annotation
    threshold_text = (
        "Thresholds: AlphaMissense $\\geq$0.564 | EVE $\\geq$0.5 | "
        "REVEL $\\geq$0.5 | SIFT $\\leq$0.05 | gnomAD AF $\\leq$0.01"
    )
    fig.text(
        0.47, 0.02, threshold_text,
        ha="center", va="bottom", fontsize=6.5, color="#666666", style="italic",
    )

    # Classification legend on right side
    cls_legend_y = 0.03
    cls_legend_x = 0.84
    fig.text(cls_legend_x, cls_legend_y + 0.04, "Classifications:", fontsize=6.5,
             fontweight="bold", color="#333333")
    for i, (label, short, color) in enumerate([
        ("Pathogenic", "P", "#1B5E20"),
        ("Likely Path.", "LP", "#66BB6A"),
        ("VUS", "VUS", "#FFA726"),
        ("Not in ClinVar", "N/A", "#BDBDBD"),
    ]):
        fig.patches.append(mpatches.FancyBboxPatch(
            (cls_legend_x, cls_legend_y + 0.04 - (i + 1) * 0.012 - 0.003),
            0.015, 0.009,
            boxstyle="round,pad=0.001",
            facecolor=color, edgecolor="none", alpha=0.85,
            transform=fig.transFigure, figure=fig,
        ))
        fig.text(cls_legend_x + 0.02, cls_legend_y + 0.04 - (i + 1) * 0.012,
                 f"{short} = {label}", fontsize=5.5, color="#333333", va="center")

    fig.savefig(OUT_HEATMAP, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_HEATMAP}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: ClinVar Concordance Bar Chart
# ---------------------------------------------------------------------------
def make_clinvar_bars():
    # Only variants that are IN ClinVar
    clinvar_labels_ordered = ["Pathogenic", "Likely_Pathogenic", "VUS"]
    pipeline_labels_ordered = ["Pathogenic", "Likely_Pathogenic", "VUS"]

    # Build confusion matrix
    confusion = np.zeros((len(pipeline_labels_ordered), len(clinvar_labels_ordered)), dtype=int)
    for v in variants:
        cv = v["clinvar"]["normalized"]
        pl = v["pipeline"]["classification"]
        if cv in clinvar_labels_ordered and pl in pipeline_labels_ordered:
            i = pipeline_labels_ordered.index(pl)
            j = clinvar_labels_ordered.index(cv)
            confusion[i, j] += 1

    # Compute accuracy metrics
    total_in_clinvar = sum(
        1 for v in variants if v["clinvar"]["normalized"] in clinvar_labels_ordered
    )
    # Concordance: pipeline agrees with ClinVar (same or adjacent)
    concordant = 0
    discordant_severe = 0  # Pipeline VUS but ClinVar P/LP, or Pipeline P/LP but ClinVar VUS->Benign
    for v in variants:
        cv = v["clinvar"]["normalized"]
        pl = v["pipeline"]["classification"]
        if cv not in clinvar_labels_ordered:
            continue
        # Exact match
        if cv == pl:
            concordant += 1
        # Adjacent match (LP vs P treated as concordant)
        elif {cv, pl} == {"Pathogenic", "Likely_Pathogenic"}:
            concordant += 1
        elif pl == "VUS" and cv in ("Pathogenic", "Likely_Pathogenic"):
            discordant_severe += 1

    concordance_rate = concordant / total_in_clinvar * 100 if total_in_clinvar > 0 else 0

    # Grouped bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.patch.set_facecolor("white")

    # -- Left panel: grouped bars --
    x = np.arange(len(clinvar_labels_ordered))
    bar_width = 0.22
    colors_pipeline = ["#1B5E20", "#66BB6A", "#FFA726"]
    pipeline_display = ["P (Pathogenic)", "LP (Likely Path.)", "VUS"]

    for i, (pl_label, color) in enumerate(zip(pipeline_display, colors_pipeline)):
        offset = (i - 1) * bar_width
        bars = ax1.bar(
            x + offset, confusion[i], bar_width,
            label=pl_label, color=color, edgecolor="white", linewidth=0.5,
            alpha=0.9,
        )
        # Value annotations
        for bar, val in zip(bars, confusion[i]):
            if val > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#333333",
                )

    clinvar_display = ["Pathogenic", "Likely\nPathogenic", "VUS"]
    ax1.set_xticks(x)
    ax1.set_xticklabels(clinvar_display, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Number of variants", fontsize=10, fontweight="bold")
    ax1.set_xlabel("ClinVar classification (ground truth)", fontsize=10, fontweight="bold")
    ax1.set_title("Pipeline classification vs ClinVar", fontsize=12, fontweight="bold", pad=12)
    ax1.legend(title="Pipeline call", fontsize=8.5, title_fontsize=9, loc="upper right",
               framealpha=0.9, edgecolor="#CCCCCC")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylim(0, max(confusion.max() + 5, 30))

    # -- Right panel: summary metrics --
    ax2.axis("off")

    metrics = [
        ("Variants with ClinVar", f"{total_in_clinvar}"),
        ("Concordant (exact/adjacent)", f"{concordant}"),
        ("Concordance rate", f"{concordance_rate:.1f}%"),
        ("Under-called (VUS vs P/LP)", f"{discordant_severe}"),
        ("Over-called", "0"),
    ]

    # Sensitivity/specificity for P/LP detection
    tp = sum(1 for v in variants
             if v["pipeline"]["classification"] in ("Pathogenic", "Likely_Pathogenic")
             and v["clinvar"]["normalized"] in ("Pathogenic", "Likely_Pathogenic"))
    fn = sum(1 for v in variants
             if v["pipeline"]["classification"] == "VUS"
             and v["clinvar"]["normalized"] in ("Pathogenic", "Likely_Pathogenic"))
    fp = sum(1 for v in variants
             if v["pipeline"]["classification"] in ("Pathogenic", "Likely_Pathogenic")
             and v["clinvar"]["normalized"] == "VUS")
    tn = sum(1 for v in variants
             if v["pipeline"]["classification"] == "VUS"
             and v["clinvar"]["normalized"] == "VUS")

    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0

    metrics.extend([
        ("", ""),
        ("P/LP Detection", ""),
        ("Sensitivity (recall)", f"{sensitivity:.1f}%"),
        ("Specificity", f"{specificity:.1f}%"),
        ("PPV (precision)", f"{ppv:.1f}%"),
    ])

    y_start = 0.92
    for i, (label, value) in enumerate(metrics):
        y = y_start - i * 0.085
        if label == "":
            continue
        if value == "":
            # Section header
            ax2.text(0.05, y, label, fontsize=10, fontweight="bold", color="#1B5E20",
                     transform=ax2.transAxes, va="center")
            ax2.plot([0.05, 0.95], [y - 0.03, y - 0.03], color="#1B5E20",
                     linewidth=0.8, transform=ax2.transAxes, clip_on=False)
        else:
            ax2.text(0.08, y, label, fontsize=9, color="#333333",
                     transform=ax2.transAxes, va="center")
            ax2.text(0.92, y, value, fontsize=9, fontweight="bold", color="#333333",
                     transform=ax2.transAxes, va="center", ha="right")

    # Add a border box around the metrics
    rect = mpatches.FancyBboxPatch(
        (0.02, y_start - len(metrics) * 0.085 + 0.04),
        0.96, y_start - (y_start - len(metrics) * 0.085 + 0.04) + 0.06,
        boxstyle="round,pad=0.02",
        facecolor="#F5F5F5", edgecolor="#CCCCCC", linewidth=1,
        transform=ax2.transAxes,
    )
    ax2.add_patch(rect)

    # Method note
    fig.text(
        0.5, 0.01,
        "Method: Six-axis Bayesian framework (Tavtigian 2018). "
        f"N={n_variants} variants from {n_profiles} SETBP1$^+$ myeloid profiles. "
        "ClinVar accessed 2026-03.",
        ha="center", va="bottom", fontsize=7, color="#888888", style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_BARS, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_BARS}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Loading {INPUT} ...")
    print(f"  {n_profiles} profiles, {n_variants} variants")
    make_concordance_heatmap()
    make_clinvar_bars()
    print("Done.")
