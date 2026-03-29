#!/usr/bin/env python3
"""
visualize_pipeline_diagram.py -- Data flow diagram showing 6 independent evidence
axes converging into Bayesian aggregation.

Layout: INPUT (left) -> 6 axis nodes (center, 3x2 grid) -> ACMG aggregation -> OUTPUT (right)
The visual argument: 6 parallel arrows converging into a single node = independence.

Clinical validation enters from below as a separate validation stream.

Output: SVG (primary), PDF (vector), PNG (300 DPI raster).
Figure size: 7.5 x 3.3 inches (wide landscape, tight).

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/visualize_pipeline_diagram.py

Runtime: <2 seconds
Dependencies: matplotlib (already in venv)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Publication-grade matplotlib settings
# ---------------------------------------------------------------------------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica",
]
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["patch.linewidth"] = 0.8
plt.rcParams["text.usetex"] = False
plt.rcParams["text.antialiased"] = True

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# ---------------------------------------------------------------------------
# Figure dimensions
# ---------------------------------------------------------------------------
FIG_W = 7.5
FIG_H = 3.3

# ---------------------------------------------------------------------------
# Node positions (in figure coordinates)
# ---------------------------------------------------------------------------

# INPUT box (left side)
INPUT_X = 0.15
INPUT_Y = FIG_H / 2 + 0.08
INPUT_W = 0.82
INPUT_H = 0.72

# 6 axis nodes: 3 rows x 2 columns in the center
AXIS_W = 1.42
AXIS_H = 0.50
AXIS_COL_GAP = 0.14
AXIS_ROW_GAP = 0.10

# Grid origin (top-left corner of the 3x2 grid)
GRID_LEFT = INPUT_X + INPUT_W + 0.42
GRID_TOP = FIG_H - 0.52

# ACMG aggregation box (right of grid)
AGG_X = GRID_LEFT + 2 * AXIS_W + AXIS_COL_GAP + 0.42
AGG_Y = FIG_H / 2 + 0.08
AGG_W = 0.82
AGG_H = 0.76

# OUTPUT box (far right)
OUT_X = AGG_X + AGG_W + 0.32
OUT_Y = FIG_H / 2 + 0.08
OUT_W = 0.78
OUT_H = 0.76

# Clinical validation box (below grid, entering aggregation from below)
CLIN_Y_OFFSET = 0.30

# ---------------------------------------------------------------------------
# Colors -- distinct per axis, colorblind-safe muted palette
# ---------------------------------------------------------------------------
AXIS_COLORS = [
    {"bg": "#DAEAF6", "border": "#2171B5", "accent": "#084594"},  # Axis 1: PLM (blue)
    {"bg": "#D4EDEE", "border": "#1B7C7D", "accent": "#0E5C5D"},  # Axis 2: Structure DL (teal)
    {"bg": "#D9F0D3", "border": "#41A841", "accent": "#1E6E1E"},  # Axis 3: Conservation (green)
    {"bg": "#FDE8CE", "border": "#D48A20", "accent": "#8B5A00"},  # Axis 4: Structure/Dock (orange)
    {"bg": "#E5DAF0", "border": "#7B52A0", "accent": "#4A2070"},  # Axis 5: Population (purple)
    {"bg": "#F5DDD8", "border": "#B85040", "accent": "#7A2A20"},  # Axis 6: Functional (brown)
]

INPUT_COLOR = {"bg": "#E8E8E8", "border": "#555555"}
AGG_COLOR = {"bg": "#FFF3CD", "border": "#B8860B"}
OUT_COLOR = {"bg": "#F2D0D0", "border": "#8B1A1A"}
CLIN_COLOR = {"bg": "#EDE0D4", "border": "#8B6A4F", "accent": "#5A3E28"}
DASHED_BOX_COLOR = "#888888"

# Arrow styling
ARROW_COLOR = "#444444"
ARROW_LW = 1.1
ARROW_MUTATION_SCALE = 10

# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------
AXES = [
    {"name": "Axis 1: Protein LM", "tools": "ESM-2 (650M)"},
    {"name": "Axis 2: Structure DL", "tools": "AlphaMissense, PrimateAI-3D"},
    {"name": "Axis 3: Conservation", "tools": "EVE, CADD, REVEL, SIFT"},
    {"name": "Axis 4: Structure/Dock", "tools": "Chai-1, Boltz-1, Vina"},
    {"name": "Axis 5: Population", "tools": "gnomAD v4 (AC = 0)"},
    {"name": "Axis 6: Functional", "tools": "SHP2 DMS, Chase 2020"},
]


def _axis_center(row: int, col: int) -> tuple[float, float]:
    """Return (cx, cy) for axis node at grid position (row, col)."""
    cx = GRID_LEFT + col * (AXIS_W + AXIS_COL_GAP) + AXIS_W / 2
    cy = GRID_TOP - row * (AXIS_H + AXIS_ROW_GAP) - AXIS_H / 2
    return cx, cy


def _draw_rounded_box(
    ax,
    cx: float,
    cy: float,
    w: float,
    h: float,
    bg: str,
    border: str,
    lw: float = 1.2,
    zorder: int = 3,
    pad: float = 0.04,
) -> None:
    """Draw a rounded rectangle centered at (cx, cy)."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle=f"round,pad={pad}",
        fc=bg,
        ec=border,
        lw=lw,
        zorder=zorder,
    )
    ax.add_patch(box)


def _draw_arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = ARROW_COLOR,
    lw: float = ARROW_LW,
    rad: float = 0.0,
    style: str = "-|>",
    zorder: int = 1,
) -> None:
    """Draw a curved arrow from (x0,y0) to (x1,y1)."""
    conn = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=style,
        mutation_scale=ARROW_MUTATION_SCALE,
        lw=lw,
        color=color,
        connectionstyle=conn,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def generate_pipeline_diagram() -> None:
    """Generate the complete data flow diagram."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # -----------------------------------------------------------------------
    # 1. INPUT box
    # -----------------------------------------------------------------------
    _draw_rounded_box(ax, INPUT_X + INPUT_W / 2, INPUT_Y, INPUT_W, INPUT_H,
                      INPUT_COLOR["bg"], INPUT_COLOR["border"])
    ax.text(INPUT_X + INPUT_W / 2, INPUT_Y + 0.16, "INPUT",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            fontfamily="DejaVu Sans", color="#1A1A1A", zorder=5)
    ax.text(INPUT_X + INPUT_W / 2, INPUT_Y - 0.06, "5 somatic\nvariants\n+ monosomy 7",
            ha="center", va="center", fontsize=6.5, fontweight="normal",
            fontfamily="DejaVu Sans", color="#333333", zorder=5,
            linespacing=1.15)

    # -----------------------------------------------------------------------
    # 2. Arrows from INPUT to each axis node (fan-out)
    # -----------------------------------------------------------------------
    input_right_x = INPUT_X + INPUT_W + 0.02

    # Compute all axis centers
    axis_centers = []
    for idx in range(6):
        row, col = divmod(idx, 2)
        axis_centers.append(_axis_center(row, col))

    # Fan-out arrows from input to each axis
    for idx, (acx, acy) in enumerate(axis_centers):
        target_left = acx - AXIS_W / 2 - 0.02
        # Curvature: positive for nodes above center, negative below
        dy = acy - INPUT_Y
        rad = -0.15 * (dy / (FIG_H / 2)) if abs(dy) > 0.05 else 0.0
        _draw_arrow(ax, input_right_x, INPUT_Y, target_left, acy,
                    color="#666666", lw=0.9, rad=rad)

    # -----------------------------------------------------------------------
    # 3. Dashed box around the 6 axis nodes ("Independent Evidence Axes")
    # -----------------------------------------------------------------------
    dbox_pad = 0.14
    dbox_left = GRID_LEFT - dbox_pad
    dbox_right = GRID_LEFT + 2 * AXIS_W + AXIS_COL_GAP + dbox_pad
    dbox_top = GRID_TOP + dbox_pad
    dbox_bottom = GRID_TOP - 3 * AXIS_H - 2 * AXIS_ROW_GAP - dbox_pad + AXIS_H

    # Recalculate bottom properly
    _, bottom_cy = _axis_center(2, 0)
    dbox_bottom = bottom_cy - AXIS_H / 2 - dbox_pad

    dashed_rect = plt.Rectangle(
        (dbox_left, dbox_bottom),
        dbox_right - dbox_left,
        dbox_top - dbox_bottom,
        fill=False,
        ec=DASHED_BOX_COLOR,
        lw=1.0,
        ls=(0, (4, 3)),
        zorder=1,
    )
    ax.add_patch(dashed_rect)
    ax.text(
        (dbox_left + dbox_right) / 2, dbox_top + 0.03,
        "Independent Evidence Axes",
        ha="center", va="bottom", fontsize=6.5, fontweight="bold",
        fontfamily="DejaVu Sans", color="#666666", zorder=5,
        fontstyle="italic",
    )

    # -----------------------------------------------------------------------
    # 4. Draw the 6 axis nodes (3 rows x 2 columns)
    # -----------------------------------------------------------------------
    for idx, axis_def in enumerate(AXES):
        row, col = divmod(idx, 2)
        cx, cy = axis_centers[idx]
        colors = AXIS_COLORS[idx]

        _draw_rounded_box(ax, cx, cy, AXIS_W, AXIS_H,
                          colors["bg"], colors["border"], lw=1.4, zorder=3)

        # Axis name (bold, centered, upper half of box)
        ax.text(cx, cy + 0.08, axis_def["name"],
                ha="center", va="center", fontsize=7, fontweight="bold",
                fontfamily="DejaVu Sans", color=colors["accent"], zorder=5)

        # Tool names (smaller, centered, lower half of box)
        ax.text(cx, cy - 0.09, axis_def["tools"],
                ha="center", va="center", fontsize=5.5, fontweight="normal",
                fontfamily="DejaVu Sans", color="#444444", zorder=5)

    # -----------------------------------------------------------------------
    # 5. Arrows from each axis to ACMG aggregation (fan-in / convergence)
    # -----------------------------------------------------------------------
    agg_cx = AGG_X + AGG_W / 2
    agg_left = AGG_X - 0.02

    # Curvature values to create the fan-in visual
    # Top row fans down, middle row goes straight, bottom row fans up
    fan_rads = [0.12, 0.10, 0.04, 0.02, -0.08, -0.12]

    for idx, (acx, acy) in enumerate(axis_centers):
        source_right = acx + AXIS_W / 2 + 0.02
        rad = fan_rads[idx]
        _draw_arrow(ax, source_right, acy, agg_left, AGG_Y,
                    color=AXIS_COLORS[idx]["border"], lw=1.2, rad=rad)

    # -----------------------------------------------------------------------
    # 6. ACMG Bayesian aggregation box
    # -----------------------------------------------------------------------
    _draw_rounded_box(ax, agg_cx, AGG_Y, AGG_W, AGG_H,
                      AGG_COLOR["bg"], AGG_COLOR["border"], lw=1.5)
    ax.text(agg_cx, AGG_Y + 0.18, "ACMG",
            ha="center", va="center", fontsize=8, fontweight="bold",
            fontfamily="DejaVu Sans", color="#6B4C00", zorder=5)
    ax.text(agg_cx, AGG_Y + 0.02, "Bayesian",
            ha="center", va="center", fontsize=7, fontweight="bold",
            fontfamily="DejaVu Sans", color="#6B4C00", zorder=5)
    ax.text(agg_cx, AGG_Y - 0.15, "14-25 pts\nper variant",
            ha="center", va="center", fontsize=5.5, fontweight="normal",
            fontfamily="DejaVu Sans", color="#555555", zorder=5,
            linespacing=1.1)

    # -----------------------------------------------------------------------
    # 7. Arrow from aggregation to OUTPUT
    # -----------------------------------------------------------------------
    out_cx = OUT_X + OUT_W / 2
    _draw_arrow(ax, agg_cx + AGG_W / 2 + 0.02, AGG_Y,
                OUT_X - 0.02, OUT_Y,
                color=OUT_COLOR["border"], lw=1.3)

    # -----------------------------------------------------------------------
    # 8. OUTPUT box
    # -----------------------------------------------------------------------
    _draw_rounded_box(ax, out_cx, OUT_Y, OUT_W, OUT_H,
                      OUT_COLOR["bg"], OUT_COLOR["border"], lw=1.5)
    ax.text(out_cx, OUT_Y + 0.18, "OUTPUT",
            ha="center", va="center", fontsize=8, fontweight="bold",
            fontfamily="DejaVu Sans", color="#5A0A0A", zorder=5)
    ax.text(out_cx, OUT_Y + 0.0, "5/5",
            ha="center", va="center", fontsize=9, fontweight="bold",
            fontfamily="DejaVu Sans", color="#8B1A1A", zorder=5)
    ax.text(out_cx, OUT_Y - 0.14, "Pathogenic",
            ha="center", va="center", fontsize=6.5, fontweight="bold",
            fontfamily="DejaVu Sans", color="#8B1A1A", zorder=5)
    ax.text(out_cx, OUT_Y - 0.28, "0 / 26,642",
            ha="center", va="center", fontsize=5.5, fontweight="normal",
            fontfamily="DejaVu Sans", color="#555555", zorder=5)

    # -----------------------------------------------------------------------
    # 9. Clinical validation box (entering from below the aggregation)
    # -----------------------------------------------------------------------
    clin_cx = agg_cx
    clin_cy = dbox_bottom - CLIN_Y_OFFSET
    clin_w = 1.80
    clin_h = 0.35

    _draw_rounded_box(ax, clin_cx, clin_cy, clin_w, clin_h,
                      CLIN_COLOR["bg"], CLIN_COLOR["border"], lw=1.0)
    ax.text(clin_cx, clin_cy + 0.05, "Clinical Validation",
            ha="center", va="center", fontsize=6.5, fontweight="bold",
            fontfamily="DejaVu Sans", color=CLIN_COLOR["accent"], zorder=5)
    ax.text(clin_cx, clin_cy - 0.08, "OncoKB  |  CIViC  |  ClinGen  |  VICC",
            ha="center", va="center", fontsize=5.2, fontweight="normal",
            fontfamily="DejaVu Sans", color="#555555", zorder=5)

    # Arrow from clinical validation up into aggregation
    _draw_arrow(ax, clin_cx, clin_cy + clin_h / 2 + 0.02,
                agg_cx, AGG_Y - AGG_H / 2 - 0.02,
                color=CLIN_COLOR["border"], lw=0.9, rad=0.0,
                style="-|>")

    # -----------------------------------------------------------------------
    # 10. Title
    # -----------------------------------------------------------------------
    ax.text(
        FIG_W / 2, FIG_H - 0.04,
        "Six-Axis Computational Pipeline for Somatic Variant Characterization",
        ha="center", va="top",
        fontsize=9, fontweight="bold",
        fontfamily="DejaVu Sans", color="#1A1A1A", zorder=10,
    )

    # -----------------------------------------------------------------------
    # 11. Bottom annotation
    # -----------------------------------------------------------------------
    annotation = (
        r"N = 26,642 myeloid patients (GENIE + HARMONY)   |   "
        r"0 quintuple matches   |   "
        r"Expected $\sim$7.7 $\times$ 10$^{-13}$"
    )
    ann_y = clin_cy - clin_h / 2 - 0.18
    ax.text(
        FIG_W / 2, ann_y,
        annotation,
        ha="center", va="center",
        fontsize=6.5, color="#222222",
        fontfamily="DejaVu Sans",
        bbox=dict(
            boxstyle="round,pad=0.10",
            fc="#F5F5F5", ec="#999999", lw=0.6,
        ),
        zorder=10,
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01)

    svg_path = OUTPUT_DIR / "pipeline_architecture.svg"
    pdf_path = OUTPUT_DIR / "pipeline_architecture.pdf"
    png_path = OUTPUT_DIR / "pipeline_architecture.png"

    fig.savefig(str(svg_path), format="svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(str(png_path), format="png", bbox_inches="tight", pad_inches=0.08, dpi=DPI)

    plt.close(fig)

    print("Pipeline diagram saved:")
    print(f"  SVG: {svg_path}")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")
    print(f"  Size: {FIG_W} x {FIG_H} inches")
    print(f"  DPI: {DPI}")


if __name__ == "__main__":
    generate_pipeline_diagram()
