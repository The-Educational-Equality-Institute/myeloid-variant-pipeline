#!/usr/bin/env python3
"""
visualize_heatmap.py -- Co-occurrence heatmap for myeloid gene pairs.

Generates a publication-quality 20x20 (or NxN) heatmap from pairwise
co-occurrence results. Uses log2(O/E) for color scale:
  - Blue = mutual exclusivity (log2(O/E) < 0)
  - Red  = co-occurrence (log2(O/E) > 0)
Significant pairs (BH-adjusted p < 0.05) are marked with asterisks.

Inputs:
    - mutation_profile/results/cooccurrence/myeloid_pairwise_results.json
      (fallback: myeloid_pairwise_results.tsv)

Outputs:
    - mutation_profile/results/cooccurrence/myeloid_cooccurrence_heatmap.html
    - mutation_profile/results/cooccurrence/myeloid_cooccurrence_heatmap.png

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/visualize_heatmap.py

Runtime: ~2 seconds
Dependencies: plotly, pandas, numpy
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "cooccurrence"

JSON_INPUT = RESULTS_DIR / "myeloid_pairwise_results.json"
TSV_INPUT = RESULTS_DIR / "myeloid_pairwise_results.tsv"

HTML_OUTPUT = RESULTS_DIR / "myeloid_cooccurrence_heatmap.html"
PNG_OUTPUT = RESULTS_DIR / "myeloid_cooccurrence_heatmap.png"

# Significance threshold after BH correction
ALPHA = 0.05

# Cap extreme log2(O/E) values to keep the color scale readable
LOG2_CAP = 5.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_json(path: Path) -> pd.DataFrame:
    """Load pairwise results from the JSON export.

    Expected structure (list of records):
    [
        {
            "gene_a": "DNMT3A",
            "gene_b": "IDH2",
            "observed": 95,
            "expected": 74.3,
            "oe_ratio": 1.28,
            "log2_oe": 0.36,
            "p_value": 0.012,
            "p_bh": 0.034,
            ...
        },
        ...
    ]

    Also accepts a top-level dict with a "pairs" or "results" key containing
    the list.
    """
    with open(path) as fh:
        raw = json.load(fh)

    # Unwrap if nested under a key
    if isinstance(raw, dict):
        for key in ("pairs", "results", "pairwise_results", "data"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            raise ValueError(
                f"JSON is a dict but no recognized list key found. "
                f"Keys present: {list(raw.keys())}"
            )

    return pd.DataFrame(raw)


def load_tsv(path: Path) -> pd.DataFrame:
    """Load pairwise results from a TSV file.

    Expected columns (at minimum): gene_a, gene_b, log2_oe, p_bh.
    Column names are normalized to lowercase with underscores.
    """
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_results() -> pd.DataFrame:
    """Try JSON first, then TSV."""
    if JSON_INPUT.exists():
        print(f"Loading from {JSON_INPUT}")
        return load_json(JSON_INPUT)
    elif TSV_INPUT.exists():
        print(f"Loading from {TSV_INPUT}")
        return load_tsv(TSV_INPUT)
    else:
        print(
            f"ERROR: No input file found.\n"
            f"  Expected: {JSON_INPUT}\n"
            f"       or:  {TSV_INPUT}",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------
def build_matrices(df: pd.DataFrame):
    """Build symmetric NxN matrices for log2(O/E) and significance.

    Returns:
        genes   — sorted list of gene names
        log2_mx — NxN numpy array of log2(O/E) values (diagonal = 0)
        sig_mx  — NxN numpy bool array (True = BH p < ALPHA)
    """
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Identify the log2(O/E) column (various naming conventions)
    log2_col = None
    for candidate in ("log2_oe", "log2_o_e", "log2oe", "log2_oe_ratio"):
        if candidate in df.columns:
            log2_col = candidate
            break
    if log2_col is None and "oe_ratio" in df.columns:
        df["log2_oe"] = df["oe_ratio"].apply(
            lambda x: math.log2(x) if x > 0 else -LOG2_CAP
        )
        log2_col = "log2_oe"
    if log2_col is None:
        raise ValueError(
            f"Cannot find log2(O/E) column. Available: {list(df.columns)}"
        )

    # Identify the BH-adjusted p-value column
    p_col = None
    for candidate in ("p_bh", "bh_p", "p_adj", "fdr", "q_value", "p_value_bh"):
        if candidate in df.columns:
            p_col = candidate
            break
    if p_col is None:
        print("WARNING: No BH p-value column found. Falling back to raw p_value.")
        if "p_value" in df.columns:
            p_col = "p_value"
        else:
            # Create a dummy column so the script still produces a heatmap
            df["p_bh"] = 1.0
            p_col = "p_bh"

    # Build gene list from both columns
    genes = sorted(set(df["gene_a"]).union(set(df["gene_b"])))
    n = len(genes)
    gene_idx = {g: i for i, g in enumerate(genes)}

    log2_mx = np.zeros((n, n), dtype=float)
    sig_mx = np.zeros((n, n), dtype=bool)

    for _, row in df.iterrows():
        i = gene_idx[row["gene_a"]]
        j = gene_idx[row["gene_b"]]
        val = float(row[log2_col]) if pd.notna(row[log2_col]) else 0.0
        val = max(-LOG2_CAP, min(LOG2_CAP, val))  # clamp
        sig = float(row[p_col]) < ALPHA if pd.notna(row[p_col]) else False

        log2_mx[i, j] = val
        log2_mx[j, i] = val
        sig_mx[i, j] = sig
        sig_mx[j, i] = sig

    return genes, log2_mx, sig_mx


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def create_heatmap(genes, log2_mx, sig_mx) -> go.Figure:
    """Build the Plotly heatmap figure."""
    n = len(genes)

    # Build annotation text: asterisks for significant pairs
    annotations = []
    for i in range(n):
        for j in range(n):
            if i == j:
                text = ""
            elif sig_mx[i, j]:
                text = "*"
            else:
                text = ""
            annotations.append(
                dict(
                    x=genes[j],
                    y=genes[i],
                    text=text,
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    xref="x",
                    yref="y",
                )
            )

    # Hover text with full detail
    hover_text = []
    for i in range(n):
        row_text = []
        for j in range(n):
            if i == j:
                row_text.append(f"{genes[i]} (self)")
            else:
                direction = "co-occurrence" if log2_mx[i, j] > 0 else "exclusivity"
                sig_label = "significant" if sig_mx[i, j] else "n.s."
                row_text.append(
                    f"{genes[i]} + {genes[j]}<br>"
                    f"log2(O/E) = {log2_mx[i, j]:.2f}<br>"
                    f"Direction: {direction}<br>"
                    f"BH p < 0.05: {sig_label}"
                )
            hover_text.append(row_text)

    # Diverging color scale: blue (exclusivity) -- white -- red (co-occurrence)
    colorscale = [
        [0.0, "rgb(33, 102, 172)"],    # strong blue
        [0.25, "rgb(103, 169, 207)"],   # light blue
        [0.5, "rgb(255, 255, 255)"],    # white (O/E = 1, log2 = 0)
        [0.75, "rgb(239, 138, 98)"],    # light red
        [1.0, "rgb(178, 24, 43)"],      # strong red
    ]

    # Symmetric color range
    abs_max = max(abs(log2_mx.min()), abs(log2_mx.max()), 1.0)

    fig = go.Figure(
        data=go.Heatmap(
            z=log2_mx,
            x=genes,
            y=genes,
            colorscale=colorscale,
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            text=hover_text,
            hoverinfo="text",
            colorbar=dict(
                title=dict(text="log<sub>2</sub>(O/E)", side="right"),
                tickvals=[-abs_max, -abs_max / 2, 0, abs_max / 2, abs_max],
                ticktext=[
                    f"{-abs_max:.1f}",
                    f"{-abs_max / 2:.1f}",
                    "0",
                    f"{abs_max / 2:.1f}",
                    f"{abs_max:.1f}",
                ],
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "Myeloid Gene Co-occurrence Heatmap<br>"
                "<sub>AACR GENIE v19.0 | * = BH-adjusted p &lt; 0.05</sub>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="",
            tickangle=-45,
            tickfont=dict(size=10),
            side="bottom",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=10),
            autorange="reversed",
        ),
        annotations=annotations,
        width=900,
        height=850,
        margin=dict(l=100, r=80, t=100, b=100),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif"),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_results()
    print(f"Loaded {len(df)} gene pairs")

    genes, log2_mx, sig_mx = build_matrices(df)
    print(f"Matrix size: {len(genes)} x {len(genes)} genes")
    print(f"Significant pairs (BH p < {ALPHA}): {sig_mx.sum() // 2}")

    fig = create_heatmap(genes, log2_mx, sig_mx)

    # Save interactive HTML
    fig.write_html(
        str(HTML_OUTPUT),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"Saved interactive heatmap: {HTML_OUTPUT}")

    # Try saving static PNG (requires kaleido)
    try:
        fig.write_image(str(PNG_OUTPUT), scale=2, width=900, height=850)
        print(f"Saved static PNG: {PNG_OUTPUT}")
    except (ImportError, ValueError) as exc:
        print(
            f"NOTE: Could not save PNG ({exc}). "
            f"Install kaleido: pip install kaleido"
        )


if __name__ == "__main__":
    main()
