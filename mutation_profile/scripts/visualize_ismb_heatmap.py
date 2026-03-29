"""
Publication-quality pathogenicity concordance heatmap for ISMB 2026 long abstract.

Data source: mutation_profile/results/ai_research/UNIFIED_PATHOGENICITY_TABLE.md
Output: deadlines/ismb-2026/figures/pathogenicity_concordance_heatmap.{png,svg}

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/visualize_ismb_heatmap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "deadlines" / "ismb-2026" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data from UNIFIED_PATHOGENICITY_TABLE.md (Section 1A + 1B)
# ---------------------------------------------------------------------------

VARIANTS = [
    "DNMT3A R882H",
    "IDH2 R140Q",
    "SETBP1 G870S",
    "PTPN11 E76Q",
    "EZH2 V662A",
]

TOOLS = [
    "AlphaMissense",
    "EVE",
    "CADD",
    "REVEL",
    "ESM-2",
    "SIFT",
    "PolyPhen-2",
    "CancerVar",
    "OncoKB",
]

# Classification: P = Pathogenic/Damaging, LP = Likely Pathogenic, U = VUS/Uncertain, B = Benign, NA = not available
# Scores are display strings; classifications drive cell color.

# Row order matches VARIANTS; column order matches TOOLS.
# Each entry: (display_score, classification)
DATA = {
    # ---- DNMT3A R882H ----
    ("DNMT3A R882H", "AlphaMissense"): ("0.9953", "P"),
    ("DNMT3A R882H", "EVE"):           ("0.620",  "U"),
    ("DNMT3A R882H", "CADD"):          ("33.0",   "P"),
    ("DNMT3A R882H", "REVEL"):         ("0.742",  "P"),
    ("DNMT3A R882H", "ESM-2"):         ("-8.38",  "P"),
    ("DNMT3A R882H", "SIFT"):          ("0.010",  "P"),
    ("DNMT3A R882H", "PolyPhen-2"):    ("0.147",  "B"),
    ("DNMT3A R882H", "CancerVar"):     ("Tier II", "LP"),
    ("DNMT3A R882H", "OncoKB"):        ("Onco",   "P"),
    # ---- IDH2 R140Q ----
    ("IDH2 R140Q", "AlphaMissense"):   ("0.9872", "P"),
    ("IDH2 R140Q", "EVE"):             ("0.886",  "P"),
    ("IDH2 R140Q", "CADD"):            ("28.1",   "P"),
    ("IDH2 R140Q", "REVEL"):           ("0.891",  "P"),
    ("IDH2 R140Q", "ESM-2"):           ("-1.48",  "U"),
    ("IDH2 R140Q", "SIFT"):            ("0.010",  "P"),
    ("IDH2 R140Q", "PolyPhen-2"):      ("0.990",  "P"),
    ("IDH2 R140Q", "CancerVar"):       ("Tier I", "P"),
    ("IDH2 R140Q", "OncoKB"):          ("Onco",   "P"),
    # ---- SETBP1 G870S ----
    ("SETBP1 G870S", "AlphaMissense"): ("0.9962", "P"),
    ("SETBP1 G870S", "EVE"):           ("0.746",  "U"),
    ("SETBP1 G870S", "CADD"):          ("27.9",   "P"),
    ("SETBP1 G870S", "REVEL"):         ("0.716",  "P"),
    ("SETBP1 G870S", "ESM-2"):         ("-9.80",  "P"),
    ("SETBP1 G870S", "SIFT"):          ("0.000",  "P"),
    ("SETBP1 G870S", "PolyPhen-2"):    ("0.999",  "P"),
    ("SETBP1 G870S", "CancerVar"):     ("Tier II", "LP"),
    ("SETBP1 G870S", "OncoKB"):        ("L. Onco", "LP"),
    # ---- PTPN11 E76Q ----
    ("PTPN11 E76Q", "AlphaMissense"):  ("0.9972", "P"),
    ("PTPN11 E76Q", "EVE"):            ("0.307",  "U"),
    ("PTPN11 E76Q", "CADD"):           ("27.3",   "P"),
    ("PTPN11 E76Q", "REVEL"):          ("0.852",  "P"),
    ("PTPN11 E76Q", "ESM-2"):          ("-1.87",  "P"),
    ("PTPN11 E76Q", "SIFT"):           ("0.010",  "P"),
    ("PTPN11 E76Q", "PolyPhen-2"):     ("0.969",  "P"),
    ("PTPN11 E76Q", "CancerVar"):      ("Tier II", "LP"),
    ("PTPN11 E76Q", "OncoKB"):         ("L. Onco", "LP"),
    # ---- EZH2 V662A ----
    ("EZH2 V662A", "AlphaMissense"):   ("0.9984", "P"),
    ("EZH2 V662A", "EVE"):             ("0.9997", "P"),
    ("EZH2 V662A", "CADD"):            ("33.0",   "P"),
    ("EZH2 V662A", "REVEL"):           ("0.962",  "P"),
    ("EZH2 V662A", "ESM-2"):           ("-2.97",  "P"),
    ("EZH2 V662A", "SIFT"):            ("0.000",  "P"),
    ("EZH2 V662A", "PolyPhen-2"):      ("0.992",  "P"),
    ("EZH2 V662A", "CancerVar"):       ("Tier II", "LP"),
    ("EZH2 V662A", "OncoKB"):          ("L. Onco", "LP"),
}

# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "P":  "#C62828",   # deep red -- Pathogenic / Damaging / Oncogenic
    "LP": "#E65100",   # deep orange -- Likely Pathogenic / Likely Oncogenic
    "U":  "#F9A825",   # amber -- VUS / Uncertain
    "B":  "#2E7D32",   # green -- Benign
    "NA": "#9E9E9E",   # gray -- not available
}

CLASS_TEXT_COLORS = {
    "P":  "#FFFFFF",
    "LP": "#FFFFFF",
    "U":  "#212121",
    "B":  "#FFFFFF",
    "NA": "#FFFFFF",
}

# ---------------------------------------------------------------------------
# Build matrices
# ---------------------------------------------------------------------------
n_variants = len(VARIANTS)
n_tools = len(TOOLS)

score_matrix = [[None] * n_tools for _ in range(n_variants)]
class_matrix = [[None] * n_tools for _ in range(n_variants)]

for i, variant in enumerate(VARIANTS):
    for j, tool in enumerate(TOOLS):
        key = (variant, tool)
        if key in DATA:
            score_matrix[i][j] = DATA[key][0]
            class_matrix[i][j] = DATA[key][1]
        else:
            score_matrix[i][j] = "--"
            class_matrix[i][j] = "NA"

# Numeric matrix for the colormap (mapped to ordinal for imshow)
CLASS_TO_NUM = {"P": 4, "LP": 3, "U": 2, "B": 1, "NA": 0}
num_matrix = np.array(
    [[CLASS_TO_NUM[class_matrix[i][j]] for j in range(n_tools)] for i in range(n_variants)],
    dtype=float,
)

# ---------------------------------------------------------------------------
# Build discrete colormap
# ---------------------------------------------------------------------------
cmap_colors = [
    CLASS_COLORS["NA"],
    CLASS_COLORS["B"],
    CLASS_COLORS["U"],
    CLASS_COLORS["LP"],
    CLASS_COLORS["P"],
]
cmap = mcolors.ListedColormap(cmap_colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# ---------------------------------------------------------------------------
# Compute concordance summary (P or LP counts as concordant pathogenic)
# ---------------------------------------------------------------------------
concordance = []
for i in range(n_variants):
    count = sum(1 for j in range(n_tools) if class_matrix[i][j] in ("P", "LP"))
    concordance.append(f"{count}/{n_tools}")

# ---------------------------------------------------------------------------
# Plot — 170 mm target width at 300 DPI ≈ 10.0 inches with summary column
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Font sizes tuned for 170 mm print width (roughly A4 column)
# ---------------------------------------------------------------------------
FONT_CELL = 10          # score text inside cells
FONT_TOOL_LABEL = 11    # column labels (tool names)
FONT_VARIANT_LABEL = 12 # row labels (variant names)
FONT_CONCORDANCE = 12   # concordance fraction text
FONT_TITLE = 13
FONT_LEGEND = 9.5
CELL_BORDER_LW = 2.0    # white border linewidth

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

fig, (ax, ax_sum) = plt.subplots(
    1, 2,
    figsize=(10.5, 5.8),
    gridspec_kw={"width_ratios": [n_tools, 1.5], "wspace": 0.10},
)

im = ax.imshow(num_matrix, cmap=cmap, norm=norm, aspect="auto")

# Remove default ticks/spines for clean look
ax.set_xticks(range(n_tools))
ax.set_yticks(range(n_variants))
ax.set_xticklabels(
    TOOLS,
    fontsize=FONT_TOOL_LABEL,
    fontweight="medium",
    rotation=40,
    ha="right",
    rotation_mode="anchor",
)
ax.set_yticklabels(
    VARIANTS,
    fontsize=FONT_VARIANT_LABEL,
    fontstyle="italic",
)

# White cell borders
for i in range(n_variants + 1):
    ax.axhline(i - 0.5, color="white", linewidth=CELL_BORDER_LW)
for j in range(n_tools + 1):
    ax.axvline(j - 0.5, color="white", linewidth=CELL_BORDER_LW)

ax.tick_params(axis="both", which="both", length=0)

# Annotate each cell with score text
for i in range(n_variants):
    for j in range(n_tools):
        cls = class_matrix[i][j]
        txt = score_matrix[i][j]
        color = CLASS_TEXT_COLORS.get(cls, "#000000")
        ax.text(
            j, i, txt,
            ha="center", va="center",
            fontsize=FONT_CELL, fontweight="bold",
            color=color,
        )

# Title spanning both axes
fig.suptitle(
    "Cross-tool pathogenicity concordance for five driver variants",
    fontsize=FONT_TITLE, fontweight="bold", y=0.97,
)

# --- Summary column (right panel) ---
summary_colors = []
for i in range(n_variants):
    p_lp = sum(1 for j in range(n_tools) if class_matrix[i][j] in ("P", "LP"))
    if p_lp == n_tools:
        summary_colors.append(CLASS_COLORS["P"])
    elif p_lp >= n_tools - 1:
        summary_colors.append("#D32F2F")  # slightly lighter red
    else:
        summary_colors.append("#E65100")

# Build a single-column numeric matrix for the summary
summary_nums = np.array(
    [[sum(1 for j in range(n_tools) if class_matrix[i][j] in ("P", "LP"))]
     for i in range(n_variants)],
    dtype=float,
)
min_conc = int(summary_nums.min())
sum_cmap = mcolors.LinearSegmentedColormap.from_list(
    "sum_green_red", ["#E65100", "#D32F2F", "#C62828"], N=256
)
ax_sum.imshow(
    summary_nums,
    cmap=sum_cmap,
    vmin=min_conc - 1, vmax=n_tools,
    aspect="auto",
)

ax_sum.set_xticks([0])
ax_sum.set_xticklabels(
    ["Concordance"],
    fontsize=FONT_TOOL_LABEL,
    fontweight="medium",
    rotation=40,
    ha="right",
    rotation_mode="anchor",
)
ax_sum.set_yticks(range(n_variants))
ax_sum.set_yticklabels([])  # labels already on left panel
ax_sum.tick_params(axis="both", which="both", length=0)

# White cell borders on summary column
for i in range(n_variants + 1):
    ax_sum.axhline(i - 0.5, color="white", linewidth=CELL_BORDER_LW)
ax_sum.axvline(-0.5, color="white", linewidth=CELL_BORDER_LW)
ax_sum.axvline(0.5, color="white", linewidth=CELL_BORDER_LW)

# Annotate summary cells -- bold white on colored background
for i in range(n_variants):
    ax_sum.text(
        0, i, concordance[i],
        ha="center", va="center",
        fontsize=FONT_CONCORDANCE, fontweight="extra bold",
        color="#FFFFFF",
    )

for spine in ax_sum.spines.values():
    spine.set_visible(False)

# Legend -- positioned with generous clearance below x-tick labels
legend_entries = [
    ("Pathogenic / Damaging", CLASS_COLORS["P"]),
    ("Likely Pathogenic", CLASS_COLORS["LP"]),
    ("VUS / Uncertain", CLASS_COLORS["U"]),
    ("Benign", CLASS_COLORS["B"]),
    ("N/A", CLASS_COLORS["NA"]),
]
patches = [
    mpatches.Patch(facecolor=c, edgecolor="white", linewidth=0.5, label=lab)
    for lab, c in legend_entries
]
ax.legend(
    handles=patches,
    loc="upper center",
    bbox_to_anchor=(0.55, -0.25),
    ncol=5,
    frameon=False,
    fontsize=FONT_LEGEND,
    handlelength=1.4,
    handletextpad=0.5,
    columnspacing=1.2,
)

# Remove outer frame
for spine in ax.spines.values():
    spine.set_visible(False)

fig.subplots_adjust(left=0.13, right=0.96, top=0.90, bottom=0.30)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
png_path = OUTPUT_DIR / "pathogenicity_concordance_heatmap.png"
svg_path = OUTPUT_DIR / "pathogenicity_concordance_heatmap.svg"

fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved PNG: {png_path}")
print(f"Saved SVG: {svg_path}")
print(f"Resolution: 300 DPI")
print(f"Grid: {n_variants} variants x {n_tools} tools = {n_variants * n_tools} cells")
