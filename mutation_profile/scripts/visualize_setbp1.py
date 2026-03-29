#!/usr/bin/env python3
"""
visualize_setbp1.py -- SETBP1-specific co-occurrence visualizations.

Two panels:
  1. Horizontal bar chart of O/E ratios for all 33 SETBP1 partner genes,
     colored by significance (BH p < 0.05) and direction.
  2. SKI domain vs non-SKI domain mutation comparison.

Inputs:
    - mutation_profile/results/setbp1_makishima/setbp1_cooccurrence_results.json
      (fallback: .tsv)
    - mutation_profile/results/setbp1_makishima/setbp1_ski_domain_analysis.json
      (optional, for panel 2)

Outputs:
    - mutation_profile/results/setbp1_makishima/setbp1_cooccurrence_chart.html

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/visualize_setbp1.py

Runtime: <1 second
Dependencies: plotly, pandas, numpy
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "setbp1_makishima"

COOC_JSON = RESULTS_DIR / "setbp1_cooccurrence_results.json"
COOC_TSV = RESULTS_DIR / "setbp1_cooccurrence_results.tsv"
SKI_JSON = RESULTS_DIR / "setbp1_ski_domain_analysis.json"

HTML_OUTPUT = RESULTS_DIR / "setbp1_cooccurrence_chart.html"

ALPHA = 0.05

# SETBP1 SKI domain hotspot range (Makishima et al., Nat Genet 2013)
# Codons 858-871 in the SKI homology domain
SKI_DOMAIN_START = 858
SKI_DOMAIN_END = 871

# All 33 partner genes (SETBP1 excluded from pairing with itself)
PARTNER_GENES = [
    "ASXL1", "TET2", "SRSF2", "SF3B1", "RUNX1", "TP53", "FLT3", "NPM1",
    "NRAS", "KRAS", "CBL", "EZH2", "U2AF1", "STAG2", "BCOR", "BCORL1",
    "DDX41", "DNMT3A", "IDH1", "IDH2", "PTPN11", "JAK2", "CALR", "MPL",
    "PHF6", "WT1", "CEBPA", "GATA2", "ZRSR2", "RAD21", "SMC1A", "SMC3",
    "CSF3R",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cooccurrence() -> pd.DataFrame:
    """Load SETBP1 co-occurrence results from JSON or TSV.

    Expected JSON structure (list of records):
    [
        {
            "gene_a": "SETBP1",
            "gene_b": "ASXL1",
            "observed": 120,
            "expected": 45.3,
            "oe_ratio": 2.65,
            "log2_oe": 1.41,
            "p_value": 1.2e-15,
            "p_bh": 3.4e-14,
            ...
        },
        ...
    ]

    Also accepts a dict wrapper with "pairs" / "results" key.
    """
    path = None
    if COOC_JSON.exists():
        path = COOC_JSON
        with open(path) as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            for key in ("pairs", "results", "pairwise_results", "data",
                        "setbp1_pairs"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
            else:
                raise ValueError(
                    f"JSON dict has no recognized list key. "
                    f"Keys: {list(raw.keys())}"
                )
        df = pd.DataFrame(raw)
    elif COOC_TSV.exists():
        path = COOC_TSV
        df = pd.read_csv(path, sep="\t")
    else:
        print(
            f"ERROR: No SETBP1 co-occurrence file found.\n"
            f"  Expected: {COOC_JSON}\n"
            f"       or:  {COOC_TSV}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded SETBP1 co-occurrence from {path}: {len(df)} pairs")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure we have an oe_ratio column
    if "oe_ratio" not in df.columns:
        for candidate in ("o_e", "o/e", "oe", "observed_expected_ratio"):
            if candidate in df.columns:
                df["oe_ratio"] = df[candidate]
                break
        else:
            if "observed" in df.columns and "expected" in df.columns:
                df["oe_ratio"] = df.apply(
                    lambda r: r["observed"] / r["expected"]
                    if r["expected"] > 0 else 0.0,
                    axis=1,
                )
            else:
                raise ValueError(
                    "Cannot derive O/E ratio. "
                    f"Available columns: {list(df.columns)}"
                )

    # Ensure log2_oe
    if "log2_oe" not in df.columns:
        for candidate in ("log2_o_e", "log2oe", "log2_oe_ratio"):
            if candidate in df.columns:
                df["log2_oe"] = df[candidate]
                break
        else:
            df["log2_oe"] = df["oe_ratio"].apply(
                lambda x: math.log2(x) if x > 0 else -5.0
            )

    # Ensure BH p-value column
    p_col = None
    for candidate in ("p_bh", "bh_p", "p_adj", "fdr", "q_value"):
        if candidate in df.columns:
            p_col = candidate
            break
    if p_col is None:
        if "p_value" in df.columns:
            p_col = "p_value"
            print("WARNING: Using raw p_value (no BH column found)")
        else:
            df["p_bh"] = 1.0
            p_col = "p_bh"
    df["p_bh_final"] = df[p_col].astype(float)

    # Extract partner gene name (SETBP1 could be gene_a or gene_b)
    if "gene_a" in df.columns and "gene_b" in df.columns:
        df["partner"] = df.apply(
            lambda r: r["gene_b"] if str(r["gene_a"]).upper() == "SETBP1"
            else r["gene_a"],
            axis=1,
        )
    elif "partner" in df.columns or "gene" in df.columns:
        if "gene" in df.columns and "partner" not in df.columns:
            df["partner"] = df["gene"]
    else:
        raise ValueError(
            "Cannot identify partner gene column. "
            f"Available: {list(df.columns)}"
        )

    return df


def load_ski_data() -> dict | None:
    """Load optional SKI domain analysis results.

    Expected structure:
    {
        "ski_domain": {
            "n_patients": 180,
            "codon_range": "858-871",
            "partner_enrichment": { "ASXL1": {"oe": 3.1, "p": 0.001}, ... }
        },
        "non_ski": {
            "n_patients": 91,
            "partner_enrichment": { "ASXL1": {"oe": 1.2, "p": 0.45}, ... }
        }
    }
    """
    if not SKI_JSON.exists():
        print(f"NOTE: SKI domain file not found ({SKI_JSON}), skipping panel 2")
        return None

    with open(SKI_JSON) as fh:
        data = json.load(fh)

    print(f"Loaded SKI domain analysis from {SKI_JSON}")
    return data


# ---------------------------------------------------------------------------
# Panel 1: O/E bar chart
# ---------------------------------------------------------------------------
def create_oe_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of O/E ratios for all SETBP1 partner genes."""

    # Sort by O/E ratio descending
    df_sorted = df.sort_values("oe_ratio", ascending=True).copy()

    # Color by significance and direction
    colors = []
    for _, row in df_sorted.iterrows():
        sig = row["p_bh_final"] < ALPHA
        oe = row["oe_ratio"]
        if sig and oe > 1:
            colors.append("rgb(178, 24, 43)")      # significant co-occurrence
        elif sig and oe <= 1:
            colors.append("rgb(33, 102, 172)")      # significant exclusivity
        elif oe > 1:
            colors.append("rgb(239, 180, 170)")     # non-sig co-occurrence
        else:
            colors.append("rgb(170, 200, 225)")     # non-sig exclusivity

    # Hover text
    hover_texts = []
    for _, row in df_sorted.iterrows():
        obs = row.get("observed", "?")
        exp = row.get("expected", "?")
        sig_label = "Yes" if row["p_bh_final"] < ALPHA else "No"
        hover_texts.append(
            f"<b>SETBP1 + {row['partner']}</b><br>"
            f"O/E = {row['oe_ratio']:.2f}<br>"
            f"log2(O/E) = {row['log2_oe']:.2f}<br>"
            f"Observed: {obs}<br>"
            f"Expected: {exp}<br>"
            f"BH p = {row['p_bh_final']:.2e}<br>"
            f"Significant: {sig_label}"
        )

    fig = go.Figure(
        go.Bar(
            y=df_sorted["partner"],
            x=df_sorted["oe_ratio"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgb(50, 50, 50)", width=0.5),
            ),
            text=[f"{v:.2f}" for v in df_sorted["oe_ratio"]],
            textposition="outside",
            textfont=dict(size=9),
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )

    # Reference line at O/E = 1 (independence)
    fig.add_vline(
        x=1.0,
        line_dash="dash",
        line_color="black",
        line_width=1.5,
        annotation_text="O/E = 1",
        annotation_position="top",
        annotation_font_size=10,
    )

    n_sig = (df["p_bh_final"] < ALPHA).sum()
    n_cooc = ((df["p_bh_final"] < ALPHA) & (df["oe_ratio"] > 1)).sum()
    n_excl = ((df["p_bh_final"] < ALPHA) & (df["oe_ratio"] <= 1)).sum()

    fig.update_layout(
        title=dict(
            text=(
                f"SETBP1 Co-occurrence with 33 Myeloid Genes<br>"
                f"<sub>{n_sig} significant (BH p &lt; 0.05): "
                f"{n_cooc} co-occurring, {n_excl} mutually exclusive | "
                f"AACR GENIE v19.0</sub>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Observed / Expected ratio",
            range=[0, max(df["oe_ratio"].max() * 1.15, 2)],
            gridcolor="rgb(230, 230, 230)",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=10),
        ),
        height=max(500, len(df_sorted) * 22 + 150),
        width=800,
        margin=dict(l=90, r=60, t=100, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif"),
    )

    return fig


# ---------------------------------------------------------------------------
# Panel 2: SKI domain vs non-SKI comparison
# ---------------------------------------------------------------------------
def create_ski_comparison(ski_data: dict) -> go.Figure | None:
    """Grouped bar chart comparing SKI domain vs non-SKI mutations."""

    ski = ski_data.get("ski_domain", ski_data.get("ski", {}))
    non_ski = ski_data.get("non_ski", ski_data.get("non_ski_domain", {}))

    ski_enrich = ski.get("partner_enrichment", ski.get("enrichment", {}))
    non_ski_enrich = non_ski.get("partner_enrichment",
                                 non_ski.get("enrichment", {}))

    if not ski_enrich or not non_ski_enrich:
        print("WARNING: SKI enrichment data incomplete, skipping comparison")
        return None

    # Build parallel arrays for the genes present in both
    all_genes = sorted(set(ski_enrich.keys()) | set(non_ski_enrich.keys()))

    genes = []
    ski_oe = []
    non_ski_oe = []
    for g in all_genes:
        if g in ski_enrich and g in non_ski_enrich:
            genes.append(g)
            s = ski_enrich[g]
            n = non_ski_enrich[g]
            ski_oe.append(
                s.get("oe", s.get("oe_ratio", s.get("o_e", 1.0)))
            )
            non_ski_oe.append(
                n.get("oe", n.get("oe_ratio", n.get("o_e", 1.0)))
            )

    if len(genes) == 0:
        print("WARNING: No overlapping genes between SKI and non-SKI data")
        return None

    # Sort by difference (SKI O/E - non-SKI O/E)
    diff = [s - n for s, n in zip(ski_oe, non_ski_oe)]
    sort_idx = sorted(range(len(genes)), key=lambda i: diff[i], reverse=True)
    genes = [genes[i] for i in sort_idx]
    ski_oe = [ski_oe[i] for i in sort_idx]
    non_ski_oe = [non_ski_oe[i] for i in sort_idx]

    ski_n = ski.get("n_patients", ski.get("n", "?"))
    non_ski_n = non_ski.get("n_patients", non_ski.get("n", "?"))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=f"SKI domain (n={ski_n})",
        x=genes,
        y=ski_oe,
        marker_color="rgb(178, 24, 43)",
        marker_line=dict(color="rgb(120, 16, 28)", width=0.5),
    ))

    fig.add_trace(go.Bar(
        name=f"Non-SKI (n={non_ski_n})",
        x=genes,
        y=non_ski_oe,
        marker_color="rgb(33, 102, 172)",
        marker_line=dict(color="rgb(22, 68, 115)", width=0.5),
    ))

    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="black",
        line_width=1,
        annotation_text="O/E = 1",
        annotation_position="bottom right",
        annotation_font_size=10,
    )

    fig.update_layout(
        title=dict(
            text=(
                "SETBP1 Co-occurrence: SKI Domain vs Non-SKI Mutations<br>"
                f"<sub>SKI domain: codons {SKI_DOMAIN_START}-{SKI_DOMAIN_END} "
                f"(Makishima et al., 2013)</sub>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        barmode="group",
        xaxis=dict(
            title="Partner gene",
            tickangle=-45,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Observed / Expected ratio",
            gridcolor="rgb(230, 230, 230)",
        ),
        height=500,
        width=900,
        margin=dict(l=70, r=40, t=100, b=100),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif"),
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgb(200, 200, 200)",
            borderwidth=1,
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Combine into single HTML
# ---------------------------------------------------------------------------
def create_combined_figure(oe_fig: go.Figure,
                           ski_fig: go.Figure | None) -> go.Figure:
    """If SKI data is available, combine both panels into a single figure
    using subplots. Otherwise, return the O/E bar chart alone."""

    if ski_fig is None:
        return oe_fig

    # Use make_subplots for a 2-row layout
    combined = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "SETBP1 Partner Gene Co-occurrence (O/E Ratio)",
            "SKI Domain vs Non-SKI Domain Comparison",
        ),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
    )

    # Transfer traces from oe_fig (panel 1)
    for trace in oe_fig.data:
        combined.add_trace(trace, row=1, col=1)

    # Transfer traces from ski_fig (panel 2)
    for trace in ski_fig.data:
        combined.add_trace(trace, row=2, col=1)

    combined.update_layout(
        title=dict(
            text=(
                "SETBP1 Mutation Co-occurrence Analysis<br>"
                "<sub>AACR GENIE v19.0 myeloid cohort | "
                "Makishima et al. replication</sub>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=17),
        ),
        height=1200,
        width=900,
        margin=dict(l=100, r=60, t=100, b=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif"),
        showlegend=True,
    )

    # Add O/E = 1 reference lines to both subplots
    combined.add_vline(
        x=1.0, row=1, col=1,
        line_dash="dash", line_color="black", line_width=1,
    )
    combined.add_hline(
        y=1.0, row=2, col=1,
        line_dash="dash", line_color="black", line_width=1,
    )

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_cooccurrence()
    print(f"  Partners: {len(df)}")
    print(f"  Significant (BH p < {ALPHA}): "
          f"{(df['p_bh_final'] < ALPHA).sum()}")

    # Panel 1: O/E bar chart
    oe_fig = create_oe_bar_chart(df)

    # Panel 2: SKI domain comparison (optional)
    ski_data = load_ski_data()
    ski_fig = None
    if ski_data is not None:
        ski_fig = create_ski_comparison(ski_data)

    # If SKI data exists, save the combined figure; otherwise just the bar chart
    if ski_fig is not None:
        final_fig = create_combined_figure(oe_fig, ski_fig)
    else:
        final_fig = oe_fig

    final_fig.write_html(
        str(HTML_OUTPUT),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"\nSaved SETBP1 visualization: {HTML_OUTPUT}")


if __name__ == "__main__":
    main()
