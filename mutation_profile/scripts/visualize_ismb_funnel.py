#!/usr/bin/env python3
"""
Compact progressive funnel chart for ISMB 2026 abstract PDF.

Optimized for small rendering: ~170mm wide x ~60mm tall in the final PDF.
Uses 7x3 inch figure with DejaVu Sans to match the pathogenicity heatmap style.

Data source: AACR GENIE v19.0 exact-variant co-occurrence analysis.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# -- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COOCCURRENCE_JSON = (
    PROJECT_ROOT / "mutation_profile" / "results" / "cooccurrence"
    / "four_gene_cooccurrence.json"
)
OUTPUT_DIR = PROJECT_ROOT / "deadlines" / "ismb-2026" / "figures"


def load_funnel_data() -> list[dict]:
    """
    Load exact-variant progressive funnel data from result files.

    Returns a list of dicts with keys: label, count, eligible.
    """
    with open(COOCCURRENCE_JSON) as f:
        data = json.load(f)

    cohort = data["cohort_summary"]
    variants = data["variant_prevalence"]
    pairwise = {
        (p["gene_a"], p["gene_b"]): p
        for p in data["pairwise_specific_variants"]
    }
    triples = data["triple_cooccurrence"]

    dnmt3a_idh2 = pairwise[("DNMT3A", "IDH2")]

    triple_dns = next(
        t for t in triples
        if set(t["genes"]) == {"DNMT3A", "IDH2", "SETBP1"}
    )

    quad = data["quadruple_cooccurrence"]

    funnel = [
        {
            "label": "Myeloid cohort",
            "count": cohort["myeloid_after_hypermut_filter"],
            "eligible": cohort["myeloid_after_hypermut_filter"],
        },
        {
            "label": "DNMT3A R882H",
            "count": variants["DNMT3A p.R882H"]["carriers"],
            "eligible": variants["DNMT3A p.R882H"]["eligible_samples"],
        },
        {
            "label": "+ IDH2 R140Q",
            "count": dnmt3a_idh2["observed"],
            "eligible": dnmt3a_idh2["eligible_samples"],
        },
        {
            "label": "+ SETBP1 G870S",
            "count": triple_dns["observed"],
            "eligible": triple_dns["eligible_samples"],
        },
        {
            "label": "+ PTPN11 E76Q",
            "count": quad["observed"],
            "eligible": quad["eligible_samples"],
        },
        {
            "label": "+ EZH2 V662A",
            "count": 0,
            "eligible": quad["eligible_samples"],
        },
    ]

    return funnel


def build_funnel_chart(funnel: list[dict], output_dir: Path) -> None:
    """Render a compact horizontal bar funnel optimized for small PDF rendering."""

    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(funnel)
    counts = [step["count"] for step in funnel]
    max_count = max(counts)

    # -- Font setup (bold sans-serif for readability at small sizes) --------
    font_family = "DejaVu Sans"

    # -- Layout parameters -------------------------------------------------
    fig_width = 7.0
    fig_height = 3.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    # Margins in figure coordinates
    label_right_x = 1.55       # right edge of label column
    bar_left_x = 1.70          # left edge of bars
    bar_max_width = 3.90       # maximum bar width
    # Vertical layout: slightly more breathing room with taller figure
    bar_height = 0.34
    bar_gap = 0.08
    total_bar_height = n * bar_height + (n - 1) * bar_gap
    top_margin = (fig_height - total_bar_height) / 2 + total_bar_height

    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    # -- Colour palette: gradient from blue through warm to deep red -------
    step_colors = [
        "#1B4F8A",  # deep blue (cohort)
        "#2E7AB8",  # medium blue (DNMT3A)
        "#D4912A",  # amber (IDH2)
        "#C44E3E",  # brick red (SETBP1, zero)
        "#A82828",  # crimson (PTPN11, zero)
        "#7A1414",  # dark maroon (EZH2, zero)
    ]

    # -- Compute bar widths (log-proportional for non-zero) ----------------
    bar_widths = []
    for i, count in enumerate(counts):
        if count > 0:
            frac = np.log10(count + 1) / np.log10(max_count + 1)
            bar_widths.append(max(frac * bar_max_width, 0.40))
        else:
            zero_idx = sum(1 for c in counts[:i] if c == 0)
            bar_widths.append(max(0.32 - zero_idx * 0.06, 0.16))

    # -- Draw bars ---------------------------------------------------------
    for i, step in enumerate(funnel):
        count = step["count"]
        label = step["label"]
        color = step_colors[i]

        y_top = top_margin - i * (bar_height + bar_gap)
        y_bottom = y_top - bar_height
        y_mid = (y_top + y_bottom) / 2

        bar_w = bar_widths[i]

        # Bar rectangle (left-aligned, thin gray border like heatmap cells)
        rect = mpatches.FancyBboxPatch(
            (bar_left_x, y_bottom),
            bar_w, bar_height,
            boxstyle=mpatches.BoxStyle.Round(pad=0.02),
            facecolor=color,
            edgecolor="#CCCCCC",
            linewidth=0.5,
        )
        ax.add_patch(rect)

        # -- Count inside bar (or right of bar for zero) -------------------
        if count > 0:
            count_text = f"n = {count:,}"
            # Place inside bar if it fits, otherwise right of bar
            text_fits = bar_w > 1.0
            if text_fits:
                ax.text(
                    bar_left_x + bar_w / 2, y_mid,
                    count_text,
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    fontfamily=font_family, color="white",
                )
            else:
                ax.text(
                    bar_left_x + bar_w + 0.08, y_mid,
                    count_text,
                    ha="left", va="center",
                    fontsize=14, fontweight="bold",
                    fontfamily=font_family, color=color,
                )
        else:
            ax.text(
                bar_left_x + bar_w + 0.08, y_mid,
                "n = 0",
                ha="left", va="center",
                fontsize=14, fontweight="bold",
                fontfamily=font_family, color="#A82828",
            )

        # -- Label (left column, right-aligned) ----------------------------
        ax.text(
            label_right_x, y_mid,
            label,
            ha="right", va="center",
            fontsize=12,
            fontfamily=font_family,
            fontweight="bold" if i == 0 else "normal",
            color="#222222",
        )

        # -- Connecting trapezoid between bars -----------------------------
        if i < n - 1:
            next_w = bar_widths[i + 1]
            trap_y_top = y_bottom
            trap_y_bot = y_bottom - bar_gap

            verts = [
                (bar_left_x, trap_y_top),
                (bar_left_x + bar_w, trap_y_top),
                (bar_left_x + next_w, trap_y_bot),
                (bar_left_x, trap_y_bot),
            ]
            trap = mpatches.Polygon(
                verts,
                closed=True,
                facecolor=color,
                edgecolor="none",
                alpha=0.15,
            )
            ax.add_patch(trap)

    # -- Save --------------------------------------------------------------
    png_path = output_dir / "progressive_funnel.png"
    svg_path = output_dir / "progressive_funnel.svg"

    fig.savefig(
        png_path, dpi=300, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    fig.savefig(
        svg_path, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    print(f"\nFunnel data:")
    for step in funnel:
        print(f"  {step['label']}: n={step['count']:,}")


if __name__ == "__main__":
    funnel = load_funnel_data()
    build_funnel_chart(funnel, OUTPUT_DIR)
