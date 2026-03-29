#!/usr/bin/env python3
"""
Clonal evolution fish plot and visualization suite.

Creates publication-quality figures from Henrik's MDS-AML clonal architecture:
1. Fish plot showing clonal proportions over simulated timepoints
2. Clonal tree diagram (matplotlib, no graphviz dependency)
3. VAF lollipop plot for all 5 driver mutations
4. Bubble plot of clonal competition (Branch A vs Branch B)

Data source: PyClone-VI manual clustering (clonal_tree_report.md)
Patient: Henrik, MDS-IB2/MDS-AML, diagnosed Sept 2023, allo-HSCT Nov 2023

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/clonal_fishplot.py
"""

from pathlib import Path
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import PatchCollection
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "clonal_evolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# Hematology-inspired color palette (publication standard)
COLORS = {
    "EZH2":    "#1b4f72",   # Dark steel blue - founder
    "DNMT3A":  "#2e86c1",   # Medium blue - trunk
    "SETBP1":  "#e67e22",   # Orange - branch A
    "PTPN11":  "#d35400",   # Dark orange - branch A nested
    "IDH2":    "#27ae60",   # Green - branch B (competing)
    "normal":  "#d5d8dc",   # Light gray - normal cells
}

# Mutation data from verified pathology report (ArcherDx VariantPlex, 18.09.2023)
MUTATIONS = {
    "EZH2":    {"variant": "V662A",  "vaf": 0.59, "ccf": 0.92, "cn": 1, "pathway": "PRC2 chromatin"},
    "DNMT3A":  {"variant": "R882H",  "vaf": 0.39, "ccf": 1.00, "cn": 2, "pathway": "DNA methylation"},
    "SETBP1":  {"variant": "G870S",  "vaf": 0.34, "ccf": 0.87, "cn": 2, "pathway": "PP2A inhibition"},
    "PTPN11":  {"variant": "E76Q",   "vaf": 0.29, "ccf": 0.74, "cn": 2, "pathway": "RAS-MAPK"},
    "IDH2":    {"variant": "R140Q",  "vaf": 0.02, "ccf": 0.05, "cn": 2, "pathway": "2-HG metabolite"},
}

# Simulated timepoints for fish plot
# T0: Normal hematopoiesis (pre-malignant)
# T1: Diagnosis (Sept 2023) - actual VAF measurements
# T2: Post-induction (Oct 2023) - remission achieved
# T3: Pre-HSCT (Nov 2023) - deep remission
# T4: Post-HSCT (2024-2026) - MRD negative, >99% donor chimerism
TIMEPOINT_LABELS = [
    "Pre-malignant",
    "Diagnosis\n(Sept 2023)",
    "Post-induction\n(Oct 2023)",
    "Pre-HSCT\n(Nov 2023)",
    "Post-HSCT\n(2024-2026)",
]


def _smooth_curve(x: np.ndarray, y: np.ndarray, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Create a smooth interpolated curve through sparse points."""
    from scipy.interpolate import PchipInterpolator
    try:
        interp = PchipInterpolator(x, y)
        x_smooth = np.linspace(x[0], x[-1], n_points)
        y_smooth = np.clip(interp(x_smooth), 0, 1)
        return x_smooth, y_smooth
    except (ImportError, ValueError):
        return x, y


def create_fish_plot() -> Path:
    """
    Create a stacked-area fish plot showing clonal dynamics over time.

    Simulates 5 timepoints based on clinical course:
    - Clonal expansion from founding clone to diagnosis
    - Rapid reduction with induction chemotherapy
    - Deep remission pre-HSCT
    - Eradication post-HSCT (MRD negative)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    timepoints = np.array([0, 1, 2, 3, 4], dtype=float)

    # Simulated cellular fractions at each timepoint (stacked, bottom-up)
    # Based on clonal hierarchy: EZH2 (founder) > DNMT3A > SETBP1 > PTPN11 > IDH2
    # All fractions are proportions of total bone marrow cells (tumor purity 78% at diagnosis)
    clone_data = {
        # Clone fractions represent proportion of ALL marrow cells
        "EZH2":   np.array([0.01, 0.78, 0.08, 0.02, 0.00]),   # Founder - matches purity
        "DNMT3A": np.array([0.01, 0.78, 0.08, 0.02, 0.00]),   # Trunk (CCF 100% of tumor)
        "SETBP1": np.array([0.00, 0.68, 0.06, 0.01, 0.00]),   # Branch A (CCF 87%)
        "PTPN11": np.array([0.00, 0.58, 0.04, 0.01, 0.00]),   # Branch A nested (CCF 74%)
        "IDH2":   np.array([0.00, 0.04, 0.005, 0.001, 0.00]), # Branch B minor (CCF 5%)
    }

    # Smooth the curves
    t_smooth = np.linspace(0, 4, 300)
    smoothed: dict[str, np.ndarray] = {}
    for gene, fracs in clone_data.items():
        _, y = _smooth_curve(timepoints, fracs, 300)
        smoothed[gene] = y

    # Draw nested stacked areas (fish plot = nested clones, not additive)
    # Order from outermost (founder) to innermost (latest subclone)
    draw_order = ["EZH2", "DNMT3A", "SETBP1", "PTPN11", "IDH2"]

    for gene in draw_order:
        y = smoothed[gene]
        ax.fill_between(t_smooth, -y / 2, y / 2, color=COLORS[gene], alpha=0.85,
                        label=f"{gene} {MUTATIONS[gene]['variant']} (CCF {MUTATIONS[gene]['ccf']:.0%})",
                        linewidth=0.5, edgecolor="white")

    # Add normal cell background
    ax.set_facecolor(COLORS["normal"])

    # Timepoint markers
    for i, label in enumerate(TIMEPOINT_LABELS):
        ax.axvline(x=i, color="#7f8c8d", linestyle=":", linewidth=0.8, alpha=0.5)
        y_pos = 0.48 if i != 1 else 0.48
        ax.text(i, y_pos, label, ha="center", va="top", fontsize=8,
                fontweight="bold" if i == 1 else "normal", color="#2c3e50")

    # Diagnosis annotation
    ax.annotate("BM blasts 12-15%\nVAF measured here",
                xy=(1, 0.39), xytext=(1.5, 0.42),
                fontsize=7, color="#2c3e50",
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.9))

    # HSCT annotation
    ax.annotate("Allo-HSCT\n23.11.2023",
                xy=(3.5, 0.0), xytext=(3.5, 0.2),
                fontsize=8, color="#c0392b", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    ax.set_xlim(-0.1, 4.3)
    ax.set_ylim(-0.50, 0.52)
    ax.set_ylabel("Cellular fraction", fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_title("Clonal Evolution - MDS-IB2/MDS-AML (Henrik)", fontsize=13, fontweight="bold", pad=15)

    legend = ax.legend(loc="upper left", fontsize=8, framealpha=0.95, edgecolor="#bdc3c7",
                       title="Clone (CCF at diagnosis)", title_fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()

    for ext in ("png", "svg"):
        path = OUTPUT_DIR / f"fishplot_clonal_evolution.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Fish plot saved to {OUTPUT_DIR}/fishplot_clonal_evolution.png/.svg")
    return OUTPUT_DIR / "fishplot_clonal_evolution.png"


def create_clonal_tree() -> Path:
    """
    Create a publication-quality clonal tree diagram using matplotlib.

    Tree structure (preferred linear model from Bayesian analysis):
      Normal HSC
        -> Clone 0: EZH2 V662A + monosomy 7 (founder, CCF ~92%)
          -> Clone 1: + DNMT3A R882H (trunk, CCF ~100%)
            |-> Branch A: + SETBP1 G870S (CCF ~87%) -> + PTPN11 E76Q (CCF ~74%)
            |-> Branch B: + IDH2 R140Q (minor, CCF ~5%)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_node(x: float, y: float, gene: str, variant: str, ccf: float,
                  extra: str = "", width: float = 2.8, height: float = 1.0) -> None:
        color = COLORS.get(gene, "#95a5a6")
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.9
        )
        ax.add_patch(box)
        # Gene name
        ax.text(x, y + 0.15, f"{gene} {variant}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        # CCF
        ax.text(x, y - 0.22, f"CCF {ccf:.0%}", ha="center", va="center",
                fontsize=8, color="white", alpha=0.9)
        if extra:
            ax.text(x, y - 0.45, extra, ha="center", va="center",
                    fontsize=7, color="white", alpha=0.8, style="italic")

    def draw_arrow(x1: float, y1: float, x2: float, y2: float,
                   style: str = "->", color: str = "#2c3e50", lw: float = 2.0) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    connectionstyle="arc3,rad=0.0"))

    # Normal HSC at top
    ax.text(6, 9.3, "Normal HSC", ha="center", va="center", fontsize=11,
            fontweight="bold", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="#bdc3c7"))

    # EZH2 - founder
    draw_arrow(6, 8.9, 6, 8.15)
    draw_node(6, 7.6, "EZH2", "V662A", 0.92, extra="Founder + monosomy 7")

    # DNMT3A - trunk
    draw_arrow(6, 7.05, 6, 6.25)
    draw_node(6, 5.7, "DNMT3A", "R882H", 1.00, extra="Epigenetic trunk")

    # Branch point label
    ax.text(6, 4.85, "Branch point", ha="center", va="center", fontsize=8,
            color="#7f8c8d", style="italic")

    # Branch A: SETBP1
    draw_arrow(6, 5.15, 4, 4.35, color=COLORS["SETBP1"])
    draw_node(4, 3.8, "SETBP1", "G870S", 0.87, extra="MDS/MPN axis")

    # Branch A nested: PTPN11
    draw_arrow(4, 3.25, 4, 2.45, color=COLORS["PTPN11"])
    draw_node(4, 1.9, "PTPN11", "E76Q", 0.74, extra="RAS-MAPK signaling")

    # Branch B: IDH2
    draw_arrow(6, 5.15, 8.5, 4.35, color=COLORS["IDH2"])
    draw_node(8.5, 3.8, "IDH2", "R140Q", 0.05, extra="Minor subclone")

    # Axis labels
    # Left: epigenetic
    ax.text(0.5, 6.7, "EPIGENETIC\nAXIS", ha="center", va="center", fontsize=9,
            fontweight="bold", color="#1b4f72", rotation=90)
    ax.plot([1.3, 1.3], [5.0, 8.3], color="#1b4f72", linewidth=2, alpha=0.3)

    # Right: proliferative
    ax.text(11.5, 3.0, "PROLIFERATIVE\nAXIS", ha="center", va="center", fontsize=9,
            fontweight="bold", color="#d35400", rotation=90)
    ax.plot([10.7, 10.7], [1.2, 4.5], color="#d35400", linewidth=2, alpha=0.3)

    # Branch labels
    ax.text(2.0, 4.5, "Branch A\n(major)", ha="center", va="center", fontsize=8,
            color=COLORS["SETBP1"], fontweight="bold")
    ax.text(10.5, 4.5, "Branch B\n(minor)", ha="center", va="center", fontsize=8,
            color=COLORS["IDH2"], fontweight="bold")

    # Druggable target annotation
    ax.annotate("Only druggable target:\nEnasidenib (FDA 2017)",
                xy=(8.5, 3.25), xytext=(10.2, 2.2),
                fontsize=7, color="#27ae60", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor="#27ae60", alpha=0.9))

    # Venetoclax resistance annotation
    ax.annotate("Venetoclax resistance\n(RAS pathway)",
                xy=(4, 1.35), xytext=(1.5, 0.7),
                fontsize=7, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdedec", edgecolor="#c0392b", alpha=0.9))

    ax.set_title("Clonal Tree - MDS-IB2/MDS-AML (Linear Model, preferred)",
                 fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()

    for ext in ("png", "svg"):
        path = OUTPUT_DIR / f"clonal_tree.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Clonal tree saved to {OUTPUT_DIR}/clonal_tree.png/.svg")
    return OUTPUT_DIR / "clonal_tree.png"


def create_vaf_lollipop() -> Path:
    """
    Create a VAF lollipop plot for all 5 driver mutations.

    Horizontal lollipop plot with VAF values, colored by pathway.
    Includes CCF annotation and copy number context.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    genes = list(MUTATIONS.keys())
    vafs = [MUTATIONS[g]["vaf"] for g in genes]
    ccfs = [MUTATIONS[g]["ccf"] for g in genes]
    pathways = [MUTATIONS[g]["pathway"] for g in genes]
    variants = [MUTATIONS[g]["variant"] for g in genes]
    colors = [COLORS[g] for g in genes]

    y_pos = np.arange(len(genes))

    # Draw stems
    for i, (yp, vaf) in enumerate(zip(y_pos, vafs)):
        ax.plot([0, vaf], [yp, yp], color=colors[i], linewidth=2.5, alpha=0.7)

    # Draw dots
    scatter = ax.scatter(vafs, y_pos, c=colors, s=200, zorder=5, edgecolors="white", linewidth=2)

    # Labels on dots
    for i, (vaf, yp) in enumerate(zip(vafs, y_pos)):
        ax.text(vaf + 0.015, yp, f"{vaf:.0%}", va="center", fontsize=10,
                fontweight="bold", color=colors[i])

    # CCF annotations
    for i, (ccf, yp) in enumerate(zip(ccfs, y_pos)):
        ax.text(0.72, yp, f"CCF {ccf:.0%}", va="center", ha="left", fontsize=8,
                color="#7f8c8d")

    # Pathway annotations
    for i, (pathway, yp) in enumerate(zip(pathways, y_pos)):
        ax.text(0.72, yp - 0.25, pathway, va="center", ha="left", fontsize=7,
                color="#95a5a6", style="italic")

    # Y-axis gene labels
    gene_labels = [f"{g} {variants[i]}" for i, g in enumerate(genes)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gene_labels, fontsize=10, fontweight="bold")

    # Formatting
    ax.set_xlim(-0.02, 0.95)
    ax.set_xlabel("Variant Allele Frequency (VAF)", fontsize=11, fontweight="bold")
    ax.set_title("Somatic Driver Mutations - VAF at Diagnosis (18.09.2023)",
                 fontsize=13, fontweight="bold", pad=15)

    # Reference lines
    ax.axvline(x=0.40, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(0.405, len(genes) - 0.3, "Purity/2\n(~39%)", fontsize=7, color="#e74c3c", alpha=0.7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Monosomy 7 note for EZH2
    ax.annotate("Hemizygous (monosomy 7)",
                xy=(0.59, -0.15), xytext=(0.40, -0.8),
                fontsize=7, color="#1b4f72", ha="center",
                arrowprops=dict(arrowstyle="->", color="#1b4f72", lw=0.8))

    fig.tight_layout()

    for ext in ("png", "svg"):
        path = OUTPUT_DIR / f"vaf_lollipop.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  VAF lollipop saved to {OUTPUT_DIR}/vaf_lollipop.png/.svg")
    return OUTPUT_DIR / "vaf_lollipop.png"


def create_competition_bubble() -> Path:
    """
    Create a bubble plot visualizing clonal competition between Branch A and Branch B.

    X-axis: time (simulated)
    Y-axis: clone fraction
    Bubble size: number of mutations in that branch
    Shows Branch A (SETBP1+PTPN11) dominant vs Branch B (IDH2) minor.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})

    # --- Left panel: Competition dynamics ---
    timepoints = np.array([0, 0.3, 0.6, 0.8, 1.0])
    time_labels = ["Early\nclone", "Expansion", "Diagnosis", "Treatment", "Post-HSCT"]

    # Branch A (SETBP1 + PTPN11): dominant
    branch_a_setbp1 = np.array([0.0, 0.30, 0.68, 0.06, 0.00])
    branch_a_ptpn11 = np.array([0.0, 0.15, 0.58, 0.04, 0.00])

    # Branch B (IDH2): minor competing
    branch_b_idh2 = np.array([0.0, 0.005, 0.04, 0.005, 0.00])

    # Smooth
    t_smooth = np.linspace(0, 1, 200)
    try:
        from scipy.interpolate import PchipInterpolator
        sa = PchipInterpolator(timepoints, branch_a_setbp1)(t_smooth)
        pa = PchipInterpolator(timepoints, branch_a_ptpn11)(t_smooth)
        ib = PchipInterpolator(timepoints, branch_b_idh2)(t_smooth)
        sa = np.clip(sa, 0, 1)
        pa = np.clip(pa, 0, 1)
        ib = np.clip(ib, 0, 1)
    except ImportError:
        t_smooth = timepoints
        sa, pa, ib = branch_a_setbp1, branch_a_ptpn11, branch_b_idh2

    ax1.fill_between(t_smooth, 0, sa, color=COLORS["SETBP1"], alpha=0.4, label="SETBP1 G870S (CCF 87%)")
    ax1.fill_between(t_smooth, 0, pa, color=COLORS["PTPN11"], alpha=0.5, label="PTPN11 E76Q (CCF 74%)")
    ax1.fill_between(t_smooth, 0, ib, color=COLORS["IDH2"], alpha=0.6, label="IDH2 R140Q (CCF 5%)")

    ax1.plot(t_smooth, sa, color=COLORS["SETBP1"], linewidth=2)
    ax1.plot(t_smooth, pa, color=COLORS["PTPN11"], linewidth=2)
    ax1.plot(t_smooth, ib, color=COLORS["IDH2"], linewidth=2)

    # Diagnosis line
    ax1.axvline(x=0.6, color="#7f8c8d", linestyle=":", linewidth=1, alpha=0.5)
    ax1.text(0.61, 0.72, "Diagnosis", fontsize=8, color="#7f8c8d", rotation=90)

    ax1.set_xlabel("Disease progression", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Clone fraction (of total marrow)", fontsize=10, fontweight="bold")
    ax1.set_title("Branch A vs Branch B - Clonal Competition", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax1.set_xticks(timepoints)
    ax1.set_xticklabels(time_labels, fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylim(0, 0.85)

    # --- Right panel: Bubble comparison at diagnosis ---
    branches = ["Branch A\nSETBP1+PTPN11\n(MDS/MPN axis)", "Branch B\nIDH2\n(Metabolic axis)"]
    ccf_values = [0.87, 0.05]  # Use SETBP1 CCF for Branch A
    n_mutations = [2, 1]
    branch_colors = [COLORS["SETBP1"], COLORS["IDH2"]]

    bubble_sizes = [n * 800 for n in n_mutations]
    x_pos = [0.3, 0.7]

    for i in range(2):
        ax2.scatter(x_pos[i], ccf_values[i], s=bubble_sizes[i], c=branch_colors[i],
                    alpha=0.7, edgecolors="white", linewidth=2, zorder=5)
        ax2.text(x_pos[i], ccf_values[i], f"{ccf_values[i]:.0%}",
                 ha="center", va="center", fontsize=12, fontweight="bold", color="white")

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(branches, fontsize=8, ha="center")
    ax2.set_ylabel("CCF at diagnosis", fontsize=10, fontweight="bold")
    ax2.set_title("Branch Size at Diagnosis", fontsize=12, fontweight="bold")

    # Ratio annotation
    ax2.text(0.5, 0.55, f"17.4x", ha="center", va="center", fontsize=14,
             fontweight="bold", color="#2c3e50",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#bdc3c7"))
    ax2.annotate("", xy=(0.3, 0.72), xytext=(0.5, 0.55),
                 arrowprops=dict(arrowstyle="<-", color="#7f8c8d", lw=1))
    ax2.annotate("", xy=(0.7, 0.15), xytext=(0.5, 0.55),
                 arrowprops=dict(arrowstyle="<-", color="#7f8c8d", lw=1))

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()

    for ext in ("png", "svg"):
        path = OUTPUT_DIR / f"clonal_competition_bubble.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"  Competition bubble saved to {OUTPUT_DIR}/clonal_competition_bubble.png/.svg")
    return OUTPUT_DIR / "clonal_competition_bubble.png"


def generate_report() -> Path:
    """Generate a brief markdown report summarizing all figures."""
    report = f"""# Clonal Evolution Visualization Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Patient:** Henrik (MDS-IB2/MDS-AML, diagnosed Sept 2023)
**Data source:** ArcherDx VariantPlex Myeloid panel, bone marrow 18.09.2023

---

## Figures

### 1. Fish Plot (fishplot_clonal_evolution.png/svg)

Stacked nested-area plot showing clonal proportions across 5 simulated timepoints:
pre-malignant, diagnosis, post-induction, pre-HSCT, and post-HSCT. The fish plot
demonstrates the nested clonal architecture where all mutations are contained within
the founder EZH2/monosomy 7 clone. At diagnosis, tumor purity is 78% (22% normal cells).
Post-HSCT, all clones are eradicated (MRD negative, >99% donor chimerism).

### 2. Clonal Tree (clonal_tree.png/svg)

Publication-quality clonal tree following the preferred linear model (log-likelihood
-0.51 vs -14.11 for branching). Shows the two-axis architecture:

- **Epigenetic axis:** EZH2 V662A (founder) -> DNMT3A R882H (trunk)
- **Proliferative axis:** SETBP1 G870S -> PTPN11 E76Q (Branch A)
- **Metabolic subclone:** IDH2 R140Q (Branch B, minor competing clone)

Key clinical annotations: venetoclax resistance (PTPN11 RAS pathway) and the only
druggable target (enasidenib for IDH2 R140Q, though targeting only 5% of tumor cells).

### 3. VAF Lollipop Plot (vaf_lollipop.png/svg)

Horizontal lollipop plot displaying all 5 driver mutations ordered by VAF:
- EZH2 V662A: 59% (hemizygous on monosomy 7, CCF 92%)
- DNMT3A R882H: 39% (diploid, CCF 100%)
- SETBP1 G870S: 34% (CCF 87%)
- PTPN11 E76Q: 29% (CCF 74%)
- IDH2 R140Q: 2% (CCF 5%)

The purity/2 reference line at 39% marks the expected VAF for a heterozygous clonal
mutation at 78% tumor purity.

### 4. Clonal Competition Bubble Plot (clonal_competition_bubble.png/svg)

Two-panel figure. Left panel shows temporal dynamics of Branch A (SETBP1+PTPN11,
MDS/MPN axis) versus Branch B (IDH2, metabolic axis). Right panel shows bubble
comparison at diagnosis: Branch A dominates at CCF 87% vs Branch B at CCF 5%
(17.4x ratio).

---

## Clonal Hierarchy Summary

| Clone | Mutations | CCF | Role |
|-------|-----------|-----|------|
| Founder | EZH2 V662A + monosomy 7 | 92% | Biallelic PRC2 inactivation |
| Trunk | + DNMT3A R882H | 100% | Epigenetic deregulation |
| Branch A | + SETBP1 G870S + PTPN11 E76Q | 87%/74% | MDS/MPN proliferative axis |
| Branch B | + IDH2 R140Q | 5% | Metabolic subclone |

## Key Observations

1. **Two-axis bridging.** This tumor simultaneously activates epigenetic (DNMT3A/EZH2/IDH2)
   and proliferative (SETBP1/PTPN11) programs that are normally mutually exclusive.
   IDH2-SETBP1 mutual exclusivity OR=0.22 in IPSS-M.

2. **Branch A dominance.** The MDS/MPN proliferative branch (SETBP1+PTPN11) comprises
   the vast majority of tumor cells. This predicts venetoclax resistance at relapse.

3. **Therapeutic vulnerability.** The only druggable target (IDH2 R140Q, enasidenib)
   resides in the minor Branch B subclone (5% of tumor). Treatment would target a
   small fraction of the disease.

4. **No precedent.** Zero patients in ~10,000 myeloid genomes carry all 4-5 of these
   driver mutations. Standard treatment algorithms do not address this architecture.

## Methods

- CCF estimation: copy-number-aware formula (Roth et al. 2014)
- Clustering: Dirichlet Process Mixture Model (CRP, alpha=1.0, sigma=0.05)
- Tree model selection: linear vs branching (log-likelihood comparison)
- Tumor purity: 78% (from DNMT3A R882H VAF, cross-validated with EZH2 on monosomy 7)
- Visualization: matplotlib {matplotlib.__version__}, numpy {np.__version__}
- Timepoints: diagnosis is measured; pre-malignant, post-induction, pre-HSCT, and
  post-HSCT are simulated based on clinical course

---
*Script: mutation_profile/scripts/ai_research/clonal_fishplot.py*
"""
    path = OUTPUT_DIR / "fishplot_report.md"
    path.write_text(report)
    print(f"  Report saved to {path}")
    return path


def main() -> None:
    """Generate all clonal evolution visualizations."""
    print("Generating clonal evolution visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    create_fish_plot()
    create_clonal_tree()
    create_vaf_lollipop()
    create_competition_bubble()
    generate_report()

    print()
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("Formats: PNG (300 dpi) + SVG")


if __name__ == "__main__":
    main()
