#!/usr/bin/env python3
"""
Clonal evolution fish plot and visualization suite (v2).

Creates publication-quality figures from Henrik's MDS-AML clonal architecture:
1. Fish plot (Muller plot) showing nested clonal dynamics across clinical timeline
2. Clonal tree diagram with CCF labels and therapeutic annotations
3. Bubble chart showing clone sizes at diagnosis
4. Lollipop plot showing VAF mapped to chromosomal positions

Data source: PyClone-VI manual clustering (clonal_tree_report.md)
Patient: Henrik, MDS-IB2/MDS-AML, diagnosed Sept 2023, allo-HSCT Nov 2023
Mutations verified against PATIENT_PROFILE.md (ArcherDx VariantPlex, 18.09.2023)

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/fish_plot.py
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "clonal_evolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# Hematology-inspired color palette -- carefully chosen for distinguishability
COLORS = {
    "EZH2":    "#1b4f72",   # Dark steel blue - founder
    "DNMT3A":  "#5dade2",   # Light blue - trunk (distinct from EZH2)
    "SETBP1":  "#e67e22",   # Orange - branch A
    "PTPN11":  "#c0392b",   # Red - branch A nested
    "IDH2":    "#27ae60",   # Green - branch B (competing)
    "normal":  "#ecf0f1",   # Very light gray - normal cells
    "bg":      "#f8f9fa",   # Background
}

# Mutation data from verified pathology report (ArcherDx VariantPlex, 18.09.2023)
MUTATIONS = {
    "EZH2": {
        "variant": "V662A", "vaf": 0.59, "ccf": 0.92, "cn": 1,
        "pathway": "PRC2 chromatin", "chromosome": "chr7", "position": 148_811_000,
        "classification": "Pathogenic (novel)",
    },
    "DNMT3A": {
        "variant": "R882H", "vaf": 0.39, "ccf": 1.00, "cn": 2,
        "pathway": "DNA methylation", "chromosome": "chr2", "position": 25_457_242,
        "classification": "Pathogenic",
    },
    "SETBP1": {
        "variant": "G870S", "vaf": 0.34, "ccf": 0.87, "cn": 2,
        "pathway": "PP2A inhibition", "chromosome": "chr18", "position": 42_531_912,
        "classification": "Likely pathogenic",
    },
    "PTPN11": {
        "variant": "E76Q", "vaf": 0.29, "ccf": 0.74, "cn": 2,
        "pathway": "RAS-MAPK signaling", "chromosome": "chr12", "position": 112_888_210,
        "classification": "Pathogenic",
    },
    "IDH2": {
        "variant": "R140Q", "vaf": 0.02, "ccf": 0.05, "cn": 2,
        "pathway": "2-HG metabolite", "chromosome": "chr15", "position": 90_631_934,
        "classification": "Pathogenic",
    },
}

# Clonal tree structure (from PyClone-VI)
CLONAL_TREE = {
    "clone_0": {"genes": ["EZH2"], "ccf": 0.92, "parent": None, "label": "Founder + monosomy 7"},
    "clone_1": {"genes": ["DNMT3A"], "ccf": 1.00, "parent": "clone_0", "label": "Epigenetic trunk"},
    "clone_2": {"genes": ["SETBP1"], "ccf": 0.87, "parent": "clone_1", "label": "MDS/MPN axis"},
    "clone_3": {"genes": ["PTPN11"], "ccf": 0.74, "parent": "clone_2", "label": "RAS-MAPK signaling"},
    "clone_4": {"genes": ["IDH2"], "ccf": 0.05, "parent": "clone_1", "label": "Minor subclone"},
}

# Clinical timeline labels
TIMEPOINT_LABELS = [
    "HSC",
    "CHIP",
    "MDS",
    "Diagnosis\n(Sept 2023)",
    "Post-induction\n(Oct 2023)",
    "Pre-HSCT\n(Nov 2023)",
    "Post-HSCT\n(2024-2026)",
]

# Chromosome sizes for ideogram (GRCh38, in Mb)
CHR_SIZES = {
    "chr2": 242, "chr7": 159, "chr12": 133, "chr15": 102, "chr18": 80,
}


def _smooth_curve(x: np.ndarray, y: np.ndarray, n_points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Create a smooth interpolated curve through sparse points."""
    try:
        from scipy.interpolate import PchipInterpolator
        interp = PchipInterpolator(x, y)
        x_smooth = np.linspace(x[0], x[-1], n_points)
        y_smooth = np.clip(interp(x_smooth), 0, 1)
        return x_smooth, y_smooth
    except (ImportError, ValueError):
        return x, y


def create_fish_plot() -> Path:
    """
    Create a nested fish plot (Muller plot) showing clonal dynamics.

    Uses a conceptual timeline from HSC through CHIP, MDS, diagnosis,
    treatment, and post-HSCT eradication. The nested structure shows
    each subclone contained within its parent clone.
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    # 7 timepoints: HSC, CHIP, MDS, Diagnosis, Post-induction, Pre-HSCT, Post-HSCT
    timepoints = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)

    # Simulated cellular fractions at each timepoint (nested, not additive)
    # Based on clonal hierarchy: each child is contained within its parent
    # Tumor purity at diagnosis: 78%
    clone_data = {
        "EZH2":   np.array([0.000, 0.005, 0.15, 0.78, 0.08, 0.02, 0.000]),
        "DNMT3A": np.array([0.000, 0.004, 0.14, 0.78, 0.08, 0.02, 0.000]),
        "SETBP1": np.array([0.000, 0.000, 0.08, 0.68, 0.06, 0.01, 0.000]),
        "PTPN11": np.array([0.000, 0.000, 0.03, 0.58, 0.04, 0.01, 0.000]),
        "IDH2":   np.array([0.000, 0.000, 0.00, 0.04, 0.005, 0.001, 0.000]),
    }

    # Smooth all curves
    n_smooth = 500
    t_smooth = np.linspace(0, 6, n_smooth)
    smoothed: dict[str, np.ndarray] = {}
    for gene, fracs in clone_data.items():
        _, y = _smooth_curve(timepoints, fracs, n_smooth)
        smoothed[gene] = y

    # Draw nested stacked areas (fish plot = nested, outermost first)
    draw_order = ["EZH2", "DNMT3A", "SETBP1", "PTPN11", "IDH2"]

    for gene in draw_order:
        y = smoothed[gene]
        ax.fill_between(
            t_smooth, -y / 2, y / 2,
            color=COLORS[gene], alpha=0.85,
            label=f"{gene} {MUTATIONS[gene]['variant']} (CCF {MUTATIONS[gene]['ccf']:.0%})",
            linewidth=0.8, edgecolor="white",
        )

    # Normal cell background
    ax.set_facecolor(COLORS["normal"])

    # Timepoint markers
    for i, label in enumerate(TIMEPOINT_LABELS):
        ax.axvline(x=i, color="#95a5a6", linestyle=":", linewidth=0.6, alpha=0.4)
        ax.text(i, 0.46, label, ha="center", va="top", fontsize=8,
                fontweight="bold" if i == 3 else "normal", color="#2c3e50")

    # Phase brackets at top
    phases = [
        (0, 2, "Pre-malignant evolution"),
        (2, 3, "Disease"),
        (3, 4, "Treatment"),
        (4, 6, "Remission"),
    ]
    for x0, x1, phase_label in phases:
        mid = (x0 + x1) / 2
        ax.annotate("", xy=(x0 + 0.1, 0.44), xytext=(x1 - 0.1, 0.44),
                     arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=0.8))
        ax.text(mid, 0.435, phase_label, ha="center", va="top", fontsize=7,
                color="#7f8c8d", style="italic")

    # Diagnosis annotation
    ax.annotate(
        "BM blasts 12-15%\n5 driver mutations\nVAF measured here",
        xy=(3, 0.39), xytext=(3.8, 0.38),
        fontsize=7, color="#2c3e50",
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
    )

    # HSCT annotation
    ax.annotate(
        "Allo-HSCT\n23.11.2023\nMUD 10/10",
        xy=(5.3, 0.0), xytext=(5.5, 0.18),
        fontsize=8, color="#c0392b", fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
    )

    # MRD negative annotation
    ax.text(6.0, -0.02, "MRD neg\n>99% donor", ha="center", va="top",
            fontsize=7, color="#27ae60", fontweight="bold")

    ax.set_xlim(-0.2, 6.5)
    ax.set_ylim(-0.50, 0.52)
    ax.set_ylabel("Cellular fraction", fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_title(
        "Clonal Evolution Fish Plot - MDS-IB2/MDS-AML",
        fontsize=14, fontweight="bold", pad=18,
    )

    legend = ax.legend(
        loc="upper left", fontsize=9, framealpha=0.95, edgecolor="#bdc3c7",
        title="Clone (CCF at diagnosis)", title_fontsize=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Source citation
    ax.text(
        0.99, -0.08,
        "Data: ArcherDx VariantPlex Myeloid panel, BM 18.09.2023 | CCF: PyClone-VI",
        transform=ax.transAxes, fontsize=6, color="#95a5a6", ha="right", va="top",
    )

    fig.tight_layout()

    paths = {}
    for ext in ("png", "svg"):
        path = OUTPUT_DIR / f"fishplot_clonal_evolution.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        paths[ext] = str(path)
    plt.close(fig)

    print(f"  Fish plot saved to {OUTPUT_DIR}/fishplot_clonal_evolution.png/.svg")
    return paths


def create_clonal_tree() -> dict[str, str]:
    """
    Create a publication-quality clonal tree diagram.

    Tree structure (preferred linear model from Bayesian analysis):
      Normal HSC
        -> Clone 0: EZH2 V662A + monosomy 7 (founder, CCF 92%)
          -> Clone 1: + DNMT3A R882H (trunk, CCF 100%)
            |-> Branch A: + SETBP1 G870S (CCF 87%) -> + PTPN11 E76Q (CCF 74%)
            |-> Branch B: + IDH2 R140Q (minor, CCF 5%)
    """
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.5, 10.5)
    ax.axis("off")

    def draw_node(x: float, y: float, gene: str, variant: str, ccf: float,
                  extra: str = "", width: float = 3.0, height: float = 1.1) -> None:
        color = COLORS.get(gene, "#95a5a6")
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor="white",
            linewidth=2.5, alpha=0.92,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.18, f"{gene} {variant}", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(x, y - 0.18, f"CCF {ccf:.0%}", ha="center", va="center",
                fontsize=9, color="white", alpha=0.9)
        if extra:
            ax.text(x, y - 0.45, extra, ha="center", va="center",
                    fontsize=7.5, color="white", alpha=0.8, style="italic")

    def draw_arrow(x1: float, y1: float, x2: float, y2: float,
                   color: str = "#2c3e50", lw: float = 2.0) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    connectionstyle="arc3,rad=0.0",
                                    mutation_scale=15))

    # Normal HSC
    ax.text(6.5, 9.8, "Normal HSC", ha="center", va="center", fontsize=12,
            fontweight="bold", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="#bdc3c7"))

    # EZH2 - founder
    draw_arrow(6.5, 9.35, 6.5, 8.55, color=COLORS["EZH2"])
    draw_node(6.5, 7.9, "EZH2", "V662A", 0.92, extra="Founder + monosomy 7 (biallelic LoF)")

    # DNMT3A - trunk
    draw_arrow(6.5, 7.3, 6.5, 6.5, color=COLORS["DNMT3A"])
    draw_node(6.5, 5.9, "DNMT3A", "R882H", 1.00, extra="Epigenetic trunk")

    # Branch point
    ax.text(6.5, 5.05, "Branch point", ha="center", va="center", fontsize=9,
            color="#7f8c8d", style="italic")

    # Branch A: SETBP1
    draw_arrow(6.5, 5.3, 4.0, 4.55, color=COLORS["SETBP1"])
    draw_node(4.0, 3.9, "SETBP1", "G870S", 0.87, extra="MDS/MPN proliferative axis")

    # Branch A nested: PTPN11
    draw_arrow(4.0, 3.3, 4.0, 2.55, color=COLORS["PTPN11"])
    draw_node(4.0, 1.9, "PTPN11", "E76Q", 0.74, extra="RAS-MAPK gain-of-function")

    # Branch B: IDH2
    draw_arrow(6.5, 5.3, 9.5, 4.55, color=COLORS["IDH2"])
    draw_node(9.5, 3.9, "IDH2", "R140Q", 0.05, extra="Minor metabolic subclone")

    # Axis labels
    ax.text(0.8, 7.0, "EPIGENETIC\nAXIS", ha="center", va="center", fontsize=9,
            fontweight="bold", color="#1b4f72", rotation=90, alpha=0.8)
    ax.plot([1.5, 1.5], [5.2, 8.6], color="#1b4f72", linewidth=2.5, alpha=0.2)

    ax.text(12.2, 3.0, "PROLIFERATIVE\nAXIS", ha="center", va="center", fontsize=9,
            fontweight="bold", color="#c0392b", rotation=90, alpha=0.8)
    ax.plot([11.5, 11.5], [1.2, 4.6], color="#c0392b", linewidth=2.5, alpha=0.2)

    # Branch labels
    ax.text(2.0, 4.7, "Branch A\n(major, 87%)", ha="center", va="center", fontsize=9,
            color=COLORS["SETBP1"], fontweight="bold")
    ax.text(11.0, 4.7, "Branch B\n(minor, 5%)", ha="center", va="center", fontsize=9,
            color=COLORS["IDH2"], fontweight="bold")

    # Druggable target annotation
    ax.annotate(
        "Druggable target:\nEnasidenib (FDA 2017)\nBut only 5% of tumor",
        xy=(9.5, 3.3), xytext=(11.2, 2.0),
        fontsize=7.5, color="#27ae60", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#27ae60", alpha=0.9),
    )

    # Venetoclax resistance
    ax.annotate(
        "Venetoclax resistance\n(PTPN11 RAS pathway activation)",
        xy=(4.0, 1.3), xytext=(1.5, 0.3),
        fontsize=7.5, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdedec", edgecolor="#c0392b", alpha=0.9),
    )

    # Tazemetostat contraindication
    ax.annotate(
        "Tazemetostat CONTRAINDICATED\n(EZH2 is LoF, not GoF)",
        xy=(6.5, 7.25), xytext=(9.5, 8.5),
        fontsize=7, color="#8e44ad",
        arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5eef8", edgecolor="#8e44ad", alpha=0.9),
    )

    ax.set_title("Clonal Tree - MDS-IB2/MDS-AML (Linear Model, preferred)",
                 fontsize=14, fontweight="bold", pad=15)

    # Source citation
    ax.text(0.99, 0.01,
            "Data: ArcherDx VariantPlex, BM 18.09.2023 | Tree: PyClone-VI (linear model LL=-0.51)",
            transform=ax.transAxes, fontsize=6, color="#95a5a6", ha="right")

    fig.tight_layout()

    paths = {}
    for ext in ("png", "svg"):
        path = OUTPUT_DIR / f"clonal_tree.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        paths[ext] = str(path)
    plt.close(fig)

    print(f"  Clonal tree saved to {OUTPUT_DIR}/clonal_tree.png/.svg")
    return paths


def create_bubble_chart() -> dict[str, str]:
    """
    Create a bubble chart showing clone sizes at diagnosis.

    Each bubble represents a clone with:
    - Size proportional to CCF
    - Color matching the gene
    - Position showing hierarchy
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    genes = list(MUTATIONS.keys())
    ccfs = [MUTATIONS[g]["ccf"] for g in genes]
    vafs = [MUTATIONS[g]["vaf"] for g in genes]
    variants = [MUTATIONS[g]["variant"] for g in genes]
    pathways = [MUTATIONS[g]["pathway"] for g in genes]
    colors = [COLORS[g] for g in genes]

    # Position: x = hierarchy level, y = CCF
    x_positions = [1, 2, 3, 4, 3.5]  # IDH2 offset from main branch
    y_positions = ccfs

    # Bubble sizes proportional to CCF (min size for visibility)
    max_size = 3000
    min_size = 300
    sizes = [max(ccf * max_size, min_size) for ccf in ccfs]

    # Draw connecting lines (tree edges)
    edges = [(0, 1), (1, 2), (2, 3), (1, 4)]  # parent -> child indices
    for parent_idx, child_idx in edges:
        ax.plot(
            [x_positions[parent_idx], x_positions[child_idx]],
            [y_positions[parent_idx], y_positions[child_idx]],
            color="#bdc3c7", linewidth=2, zorder=1, linestyle="--", alpha=0.6,
        )

    # Draw bubbles
    for i in range(len(genes)):
        ax.scatter(
            x_positions[i], y_positions[i], s=sizes[i],
            c=colors[i], alpha=0.85, edgecolors="white", linewidth=2.5, zorder=5,
        )
        # Gene label inside bubble
        ax.text(x_positions[i], y_positions[i] + 0.02, f"{genes[i]}\n{variants[i]}",
                ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=6)
        # CCF label below bubble
        offset = -0.065 if ccfs[i] > 0.1 else -0.035
        ax.text(x_positions[i], y_positions[i] + offset, f"CCF {ccfs[i]:.0%}",
                ha="center", va="top", fontsize=8, color="white", zorder=6)

    # VAF annotations outside bubbles
    for i in range(len(genes)):
        x_off = 0.35 if i != 4 else -0.35
        ax.annotate(
            f"VAF {vafs[i]:.0%}\n{pathways[i]}",
            xy=(x_positions[i], y_positions[i]),
            xytext=(x_positions[i] + x_off, y_positions[i] + 0.08),
            fontsize=7, color="#7f8c8d", ha="center",
            arrowprops=dict(arrowstyle="-", color="#bdc3c7", lw=0.5),
        )

    # Clone hierarchy labels at bottom
    hierarchy_labels = ["Founder\n(Clone 0)", "Trunk\n(Clone 1)", "Branch A\n(Clone 2)",
                        "Branch A nested\n(Clone 3)", "Branch B\n(Clone 4)"]
    for i, label in enumerate(hierarchy_labels):
        ax.text(x_positions[i], -0.08, label, ha="center", va="top",
                fontsize=7, color="#7f8c8d")

    ax.set_xlim(0.2, 5.0)
    ax.set_ylim(-0.15, 1.12)
    ax.set_xlabel("Clonal hierarchy (acquisition order)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cancer Cell Fraction (CCF)", fontsize=11, fontweight="bold")
    ax.set_title("Clone Size Bubble Chart at Diagnosis (Sept 2023)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks([])

    # Reference lines
    ax.axhline(y=1.0, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.text(4.8, 1.01, "Clonal", fontsize=7, color="#e74c3c", alpha=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.text(0.99, 0.01,
            "Data: ArcherDx VariantPlex, BM 18.09.2023 | CCF: PyClone-VI (copy-number aware)",
            transform=ax.transAxes, fontsize=6, color="#95a5a6", ha="right")

    fig.tight_layout()

    paths = {}
    for ext in ("png", "svg"):
        path = OUTPUT_DIR / f"clone_bubble_chart.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        paths[ext] = str(path)
    plt.close(fig)

    print(f"  Bubble chart saved to {OUTPUT_DIR}/clone_bubble_chart.png/.svg")
    return paths


def create_vaf_lollipop() -> dict[str, str]:
    """
    Create a VAF lollipop plot with chromosomal positions.

    Shows each mutation positioned on its chromosome with VAF as the lollipop height.
    Includes copy number context and CCF annotation.
    """
    fig, (ax_main, ax_chr) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.35},
    )

    # --- Top panel: VAF lollipop ---
    genes = list(MUTATIONS.keys())
    vafs = [MUTATIONS[g]["vaf"] for g in genes]
    ccfs = [MUTATIONS[g]["ccf"] for g in genes]
    variants = [MUTATIONS[g]["variant"] for g in genes]
    pathways = [MUTATIONS[g]["pathway"] for g in genes]
    chroms = [MUTATIONS[g]["chromosome"] for g in genes]
    colors = [COLORS[g] for g in genes]

    y_pos = np.arange(len(genes))

    # Stems
    for i, (yp, vaf) in enumerate(zip(y_pos, vafs)):
        ax_main.plot([0, vaf], [yp, yp], color=colors[i], linewidth=3, alpha=0.7)

    # Dots
    ax_main.scatter(vafs, y_pos, c=colors, s=250, zorder=5,
                    edgecolors="white", linewidth=2.5)

    # VAF labels
    for i, (vaf, yp) in enumerate(zip(vafs, y_pos)):
        ax_main.text(vaf + 0.018, yp, f"{vaf:.0%}", va="center", fontsize=11,
                     fontweight="bold", color=colors[i])

    # CCF and pathway annotations
    for i, (ccf, pathway, chrom, yp) in enumerate(zip(ccfs, pathways, chroms, y_pos)):
        ax_main.text(0.75, yp + 0.05, f"CCF {ccf:.0%}  |  {chrom}", va="center", ha="left",
                     fontsize=8, color="#555555")
        ax_main.text(0.75, yp - 0.22, pathway, va="center", ha="left",
                     fontsize=7.5, color="#95a5a6", style="italic")

    # Y-axis labels
    gene_labels = [f"{g} {variants[i]}" for i, g in enumerate(genes)]
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(gene_labels, fontsize=11, fontweight="bold")

    # Purity/2 reference line
    ax_main.axvline(x=0.39, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5)
    ax_main.text(0.395, len(genes) - 0.5, "Purity/2\n(39%)", fontsize=7, color="#e74c3c",
                 alpha=0.7)

    # Monosomy 7 annotation for EZH2
    ax_main.annotate(
        "Hemizygous\n(monosomy 7, CN=1)",
        xy=(0.59, 0), xytext=(0.62, -0.7),
        fontsize=7.5, color="#1b4f72", ha="center",
        arrowprops=dict(arrowstyle="->", color="#1b4f72", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#d6eaf8", edgecolor="#1b4f72", alpha=0.8),
    )

    ax_main.set_xlim(-0.02, 0.95)
    ax_main.set_ylim(-1.2, len(genes) - 0.3)
    ax_main.set_xlabel("Variant Allele Frequency (VAF)", fontsize=11, fontweight="bold")
    ax_main.set_title("Somatic Driver Mutations - VAF at Diagnosis (18.09.2023)",
                      fontsize=14, fontweight="bold", pad=15)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # --- Bottom panel: Chromosomal ideogram ---
    # Sort chromosomes by number for display
    chrom_order = ["chr2", "chr7", "chr12", "chr15", "chr18"]
    chrom_x_start = {}
    cumulative = 0
    gap = 10
    for chrom in chrom_order:
        chrom_x_start[chrom] = cumulative
        cumulative += CHR_SIZES[chrom] + gap

    total_width = cumulative - gap

    # Draw chromosome bars
    bar_height = 0.3
    for chrom in chrom_order:
        x_start = chrom_x_start[chrom]
        size = CHR_SIZES[chrom]
        ax_chr.barh(0, size, left=x_start, height=bar_height,
                    color="#d5d8dc", edgecolor="#95a5a6", linewidth=0.8)
        ax_chr.text(x_start + size / 2, -0.3, chrom.replace("chr", ""),
                    ha="center", va="top", fontsize=9, fontweight="bold", color="#2c3e50")

    # Plot mutations as lollipops on chromosomes
    for gene in genes:
        m = MUTATIONS[gene]
        chrom = m["chromosome"]
        pos_mb = m["position"] / 1_000_000
        x = chrom_x_start[chrom] + pos_mb
        color = COLORS[gene]

        # Stem
        ax_chr.plot([x, x], [bar_height / 2, 0.6 + m["vaf"] * 0.8], color=color,
                    linewidth=2, alpha=0.8)
        # Dot
        ax_chr.scatter(x, 0.6 + m["vaf"] * 0.8, c=color, s=120, zorder=5,
                       edgecolors="white", linewidth=1.5)
        # Label
        ax_chr.text(x, 0.65 + m["vaf"] * 0.8, f"{gene}\n{m['variant']}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold", color=color)

    # Monosomy 7 shading
    chr7_start = chrom_x_start["chr7"]
    chr7_size = CHR_SIZES["chr7"]
    ax_chr.axvspan(chr7_start - 2, chr7_start + chr7_size + 2,
                   color="#e74c3c", alpha=0.08)
    ax_chr.text(chr7_start + chr7_size / 2, -0.55, "Monosomy 7",
                ha="center", va="top", fontsize=7, color="#e74c3c", fontweight="bold")

    ax_chr.set_xlim(-5, total_width + 5)
    ax_chr.set_ylim(-0.7, 1.8)
    ax_chr.set_xlabel("Chromosomal position", fontsize=10, fontweight="bold")
    ax_chr.set_yticks([])
    ax_chr.spines["top"].set_visible(False)
    ax_chr.spines["right"].set_visible(False)
    ax_chr.spines["left"].set_visible(False)
    ax_chr.set_title("Chromosomal Distribution of Driver Mutations",
                     fontsize=11, fontweight="bold")

    # Source citation
    ax_chr.text(0.99, -0.15,
                "Data: ArcherDx VariantPlex Myeloid panel, BM 18.09.2023 | Coordinates: GRCh38",
                transform=ax_chr.transAxes, fontsize=6, color="#95a5a6", ha="right")

    fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.08, hspace=0.4)

    paths = {}
    for ext in ("png", "svg"):
        path = OUTPUT_DIR / f"vaf_lollipop.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        paths[ext] = str(path)
    plt.close(fig)

    print(f"  VAF lollipop saved to {OUTPUT_DIR}/vaf_lollipop.png/.svg")
    return paths


def save_metadata(figure_paths: dict[str, dict[str, str]]) -> Path:
    """Save metadata JSON for all generated figures."""
    metadata = {
        "generated": datetime.now().isoformat(),
        "script": "mutation_profile/scripts/ai_research/fish_plot.py",
        "patient": {
            "diagnosis": "MDS-IB2/MDS-AML",
            "diagnosis_date": "2023-09-18",
            "sample": "Bone marrow, ArcherDx VariantPlex Myeloid panel",
            "tumor_purity": 0.78,
            "hsct_date": "2023-11-23",
            "current_status": "MRD negative, >99% donor chimerism",
        },
        "mutations": {
            gene: {
                "variant": m["variant"],
                "vaf": m["vaf"],
                "ccf": m["ccf"],
                "copy_number": m["cn"],
                "chromosome": m["chromosome"],
                "position_grch38": m["position"],
                "pathway": m["pathway"],
                "classification": m["classification"],
            }
            for gene, m in MUTATIONS.items()
        },
        "clonal_tree": {
            clone_id: {
                "genes": clone["genes"],
                "ccf": clone["ccf"],
                "parent": clone["parent"],
                "label": clone["label"],
            }
            for clone_id, clone in CLONAL_TREE.items()
        },
        "tree_model": {
            "method": "PyClone-VI (Dirichlet Process Mixture Model)",
            "model": "Linear (preferred)",
            "log_likelihood": -0.51,
            "alternative_model_ll": -14.11,
            "tumor_purity": 0.78,
            "ccf_formula": "copy-number-aware (Roth et al. 2014)",
        },
        "figures": {
            name: {
                "files": paths,
                "dpi": DPI,
            }
            for name, paths in figure_paths.items()
        },
        "key_findings": [
            "5 confirmed driver mutations spanning 5 oncogenic axes",
            "Zero matches across ~10,000 myeloid patients in 10 databases",
            "Branch A (SETBP1+PTPN11) dominates at CCF 87%/74% -- predicts venetoclax resistance",
            "Branch B (IDH2 R140Q) is only druggable target but represents 5% of tumor",
            "EZH2 V662A is a novel unreported variant (biallelic LoF with monosomy 7)",
            "Tazemetostat (EZH2 inhibitor) is contraindicated -- variant is LoF, not GoF",
        ],
    }

    path = OUTPUT_DIR / "clonal_plots_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"  Metadata saved to {path}")
    return path


def main() -> None:
    """Generate all clonal evolution visualizations and metadata."""
    print("=" * 70)
    print("Clonal Evolution Fish Plot & Visualization Suite")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    figure_paths: dict[str, dict[str, str]] = {}

    print("[1/4] Fish plot (Muller plot)...")
    figure_paths["fishplot_clonal_evolution"] = create_fish_plot()

    print("[2/4] Clonal tree diagram...")
    figure_paths["clonal_tree"] = create_clonal_tree()

    print("[3/4] Clone size bubble chart...")
    figure_paths["clone_bubble_chart"] = create_bubble_chart()

    print("[4/4] VAF lollipop + chromosomal positions...")
    figure_paths["vaf_lollipop"] = create_vaf_lollipop()

    print()
    print("Saving metadata...")
    save_metadata(figure_paths)

    print()
    print("=" * 70)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("Formats: PNG ({} dpi) + SVG".format(DPI))
    print("=" * 70)


if __name__ == "__main__":
    main()
