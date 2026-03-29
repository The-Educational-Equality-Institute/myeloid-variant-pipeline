#!/usr/bin/env python3
"""
visualize_funnel.py -- Progressive funnel chart for the 4-gene search.

Shows how patient count drops at each successive mutation filter:
  All myeloid -> DNMT3A R882H -> +IDH2 R140Q -> +PTPN11 E76Q -> +SETBP1 G870S = 0

Inputs:
    - mutation_profile/results/cooccurrence/four_gene_cooccurrence.json

Outputs:
    - mutation_profile/results/cooccurrence/four_gene_funnel.html

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/visualize_funnel.py

Runtime: <1 second
Dependencies: plotly
"""

import json
import sys
from pathlib import Path

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "cooccurrence"

JSON_INPUT = RESULTS_DIR / "four_gene_cooccurrence.json"
HTML_OUTPUT = RESULTS_DIR / "four_gene_funnel.html"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_funnel_data(path: Path) -> dict:
    """Load the four-gene progressive filtering results.

    Expected JSON structure (flexible — several layouts accepted):

    Option A (preferred):
    {
        "funnel_steps": [
            {"label": "All myeloid patients", "count": 27585},
            {"label": "DNMT3A mutated", "count": 4210},
            {"label": "+ IDH2 mutated", "count": 412},
            {"label": "+ PTPN11 mutated", "count": 12},
            {"label": "+ SETBP1 mutated", "count": 0}
        ],
        ...
    }

    Option B (flat keys):
    {
        "total_myeloid": 27585,
        "dnmt3a": 4210,
        "dnmt3a_idh2": 412,
        "dnmt3a_idh2_ptpn11": 12,
        "dnmt3a_idh2_ptpn11_setbp1": 0,
        ...
    }

    Option C (nested progressive_filter / steps):
    {
        "progressive_filter": { ... same as B ... }
    }
    """
    with open(path) as fh:
        raw = json.load(fh)

    # Option A: explicit funnel_steps list
    for key in ("funnel_steps", "steps", "funnel"):
        if key in raw and isinstance(raw[key], list):
            steps = raw[key]
            labels = [s.get("label", s.get("name", f"Step {i}"))
                      for i, s in enumerate(steps)]
            counts = [s.get("count", s.get("n", s.get("patients", 0)))
                      for s in steps]
            return {"labels": labels, "counts": counts}

    # Option B / C: flat or nested dict with known keys
    source = raw
    for key in ("progressive_filter", "progressive", "filter_steps"):
        if key in raw and isinstance(raw[key], dict):
            source = raw[key]
            break

    # Map of possible key names to canonical labels
    KEY_MAP = [
        (
            ["total_myeloid", "all_myeloid", "myeloid_total", "n_myeloid"],
            "All myeloid patients",
        ),
        (
            ["dnmt3a", "DNMT3A", "dnmt3a_count", "step1"],
            "DNMT3A mutated",
        ),
        (
            ["dnmt3a_idh2", "DNMT3A_IDH2", "dnmt3a+idh2", "step2"],
            "+ IDH2 mutated",
        ),
        (
            ["dnmt3a_idh2_ptpn11", "DNMT3A_IDH2_PTPN11", "step3"],
            "+ PTPN11 mutated",
        ),
        (
            ["dnmt3a_idh2_ptpn11_setbp1", "DNMT3A_IDH2_PTPN11_SETBP1",
             "quadruple", "step4"],
            "+ SETBP1 mutated",
        ),
    ]

    labels = []
    counts = []
    for candidates, label in KEY_MAP:
        for k in candidates:
            if k in source:
                labels.append(label)
                val = source[k]
                counts.append(int(val) if not isinstance(val, dict) else
                              int(val.get("count", val.get("n", 0))))
                break

    if len(labels) < 2:
        raise ValueError(
            f"Could not extract funnel steps from JSON. "
            f"Keys found: {list(source.keys())}"
        )

    return {"labels": labels, "counts": counts}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def create_funnel(labels: list, counts: list) -> go.Figure:
    """Build a publication-quality funnel chart."""

    # Color gradient from teal (broad) to deep red (narrow)
    n = len(labels)
    colors = [
        f"rgba({int(33 + (178 - 33) * i / (n - 1))}, "
        f"{int(150 - 126 * i / (n - 1))}, "
        f"{int(172 - 129 * i / (n - 1))}, 0.85)"
        for i in range(n)
    ]

    # Percentage annotations relative to first step
    total = counts[0] if counts[0] > 0 else 1
    pct_texts = [
        f"{c:,} patients ({100 * c / total:.2f}%)" if c > 0
        else f"0 patients (0%)"
        for c in counts
    ]

    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=counts,
            textinfo="text",
            text=pct_texts,
            textposition="inside",
            textfont=dict(size=13, color="white"),
            marker=dict(color=colors, line=dict(width=1, color="white")),
            connector=dict(line=dict(color="rgb(180, 180, 180)", width=1)),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Patients: %{x:,}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add an annotation for the final zero step (since funnel hides zero bars)
    if counts[-1] == 0:
        fig.add_annotation(
            x=0.5,
            y=-0.08,
            xref="paper",
            yref="paper",
            text=(
                "<b>0 patients carry all four mutations</b><br>"
                f"Expected: ~0.007 | Observed: 0"
            ),
            showarrow=False,
            font=dict(size=13, color="rgb(178, 24, 43)"),
            align="center",
            bgcolor="rgba(255, 240, 240, 0.9)",
            bordercolor="rgb(178, 24, 43)",
            borderwidth=1,
            borderpad=8,
        )

    fig.update_layout(
        title=dict(
            text=(
                "Progressive Mutation Filtering: 4-Gene Combination<br>"
                "<sub>DNMT3A + IDH2 + PTPN11 + SETBP1 | "
                "AACR GENIE v19.0 myeloid cohort</sub>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=17),
        ),
        width=750,
        height=550,
        margin=dict(l=30, r=30, t=100, b=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif"),
        funnelmode="stack",
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not JSON_INPUT.exists():
        print(
            f"ERROR: Input file not found: {JSON_INPUT}\n"
            f"Run the four-gene co-occurrence analysis first.",
            file=sys.stderr,
        )
        sys.exit(1)

    data = load_funnel_data(JSON_INPUT)
    labels = data["labels"]
    counts = data["counts"]

    print("Funnel steps:")
    for label, count in zip(labels, counts):
        print(f"  {label}: {count:,}")

    fig = create_funnel(labels, counts)

    fig.write_html(
        str(HTML_OUTPUT),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"\nSaved funnel chart: {HTML_OUTPUT}")


if __name__ == "__main__":
    main()
