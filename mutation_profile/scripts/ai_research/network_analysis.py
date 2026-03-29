#!/usr/bin/env python3
"""
network_analysis.py -- STRING database network biology analysis for 4+1 gene mutation profile.

Patient mutation profile:
  DNMT3A R882H (VAF 39%) - DNA methylation / epigenetic regulation
  IDH2 R140Q   (VAF 2%)  - TCA cycle / 2-HG / epigenetic hypermethylation
  SETBP1 G870S (VAF 34%) - SET/PP2A / chromatin remodeling / self-renewal
  PTPN11 E76Q  (VAF 29%) - RAS/MAPK signaling (SHP2 upstream activator)
  EZH2 V662A Pathogenic (VAF 59%) - PRC2 / H3K27 methylation

Analyses:
  1. STRING v12 protein-protein interaction network (4+1 genes + first-shell interactors)
  2. STRING functional enrichment (KEGG, Reactome, GO Biological Process)
  3. Pairwise pathway overlap and Jaccard similarity
  4. Pairwise STRING functional scores
  5. Evidence for/against pathway redundancy as mutual exclusivity mechanism

Inputs:
    STRING v12 REST API (no authentication required)

Outputs:
    - mutation_profile/results/ai_research/string_network.json
    - mutation_profile/results/ai_research/pathway_enrichment.json
    - mutation_profile/results/ai_research/network_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/network_analysis.py

Runtime: ~30 seconds (API calls)
Dependencies: requests
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "mutation_profile", "results", "ai_research")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATIENT_GENES = ["DNMT3A", "IDH2", "SETBP1", "PTPN11", "EZH2"]
SPECIES = 9606  # Homo sapiens
STRING_BASE = "https://string-db.org/api/json"
REQUEST_DELAY = 1.0  # Seconds between API calls to be polite

# Gene pairs of special interest (from co-occurrence analysis)
SPECIAL_PAIRS = [
    ("SETBP1", "IDH2"),   # Mutually exclusive (O/E=0.13, p=0.013)
    ("DNMT3A", "IDH2"),   # Co-occurring in AML
    ("DNMT3A", "PTPN11"), # Both in founding clone
    ("SETBP1", "PTPN11"), # Both high-VAF
    ("IDH2", "PTPN11"),   # IDH2 minor clone
    ("SETBP1", "EZH2"),   # Both chromatin-related
]


def query_string_network(genes: list[str], add_nodes: int = 20,
                         required_score: int = 700) -> list[dict]:
    """Query STRING network API for gene set with first-shell interactors."""
    identifiers = "%0d".join(genes)
    url = (
        f"{STRING_BASE}/network"
        f"?identifiers={identifiers}"
        f"&species={SPECIES}"
        f"&required_score={required_score}"
        f"&add_nodes={add_nodes}"
    )
    print(f"  Querying STRING network: {len(genes)} genes, add_nodes={add_nodes}, "
          f"required_score={required_score}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print(f"  -> {len(data)} interactions returned")
    return data


def query_string_enrichment(genes: list[str]) -> list[dict]:
    """Query STRING functional enrichment API."""
    identifiers = "%0d".join(genes)
    url = (
        f"{STRING_BASE}/enrichment"
        f"?identifiers={identifiers}"
        f"&species={SPECIES}"
    )
    print(f"  Querying STRING enrichment: {len(genes)} genes")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print(f"  -> {len(data)} enrichment terms returned")
    return data


def query_pairwise_score(gene_a: str, gene_b: str) -> list[dict]:
    """Query STRING for direct interaction score between two genes."""
    identifiers = f"{gene_a}%0d{gene_b}"
    url = (
        f"{STRING_BASE}/network"
        f"?identifiers={identifiers}"
        f"&species={SPECIES}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def build_gene_pathway_map(enrichment: list[dict]) -> dict[str, set[str]]:
    """Build gene -> set of pathway IDs mapping from enrichment data."""
    gene_pathways: dict[str, set[str]] = defaultdict(set)
    for term in enrichment:
        term_id = term.get("term", "")
        category = term.get("category", "")
        # Focus on pathway databases (KEGG, Reactome) and GO BP
        if category not in (
            "KEGG", "Reactome", "Process", "GOCC", "GOMF",
            "KEGG Pathways", "Reactome Pathways",
            "NetworkNeighborAL", "RCTM",
        ):
            # Accept all categories STRING returns
            pass
        genes_in_term = term.get("preferredNames", "")
        if isinstance(genes_in_term, str):
            gene_list = [g.strip() for g in genes_in_term.split(",") if g.strip()]
        else:
            gene_list = genes_in_term
        for gene in gene_list:
            if gene in PATIENT_GENES:
                gene_pathways[gene].add(f"{category}:{term_id}")
    return {g: s for g, s in gene_pathways.items()}


def compute_pairwise_overlap(gene_pathways: dict[str, set[str]]) -> list[dict]:
    """Compute Jaccard similarity and shared pathway counts for all gene pairs."""
    results = []
    for g1, g2 in combinations(PATIENT_GENES, 2):
        s1 = gene_pathways.get(g1, set())
        s2 = gene_pathways.get(g2, set())
        shared = s1 & s2
        union = s1 | s2
        jaccard = len(shared) / len(union) if union else 0.0
        results.append({
            "gene_a": g1,
            "gene_b": g2,
            "pathways_a": len(s1),
            "pathways_b": len(s2),
            "shared_count": len(shared),
            "union_count": len(union),
            "jaccard_similarity": round(jaccard, 4),
            "shared_pathways": sorted(shared),
        })
    results.sort(key=lambda x: x["jaccard_similarity"], reverse=True)
    return results


def categorize_enrichment(enrichment: list[dict]) -> dict[str, list[dict]]:
    """Categorize enrichment terms by database/category."""
    categorized: dict[str, list[dict]] = defaultdict(list)
    for term in enrichment:
        cat = term.get("category", "Unknown")
        p_value = term.get("p_value", 1.0)
        fdr = term.get("fdr", 1.0)
        if fdr > 0.05:
            continue
        genes_in_term = term.get("preferredNames", "")
        if isinstance(genes_in_term, str):
            gene_list = [g.strip() for g in genes_in_term.split(",") if g.strip()]
        else:
            gene_list = genes_in_term
        patient_genes_in_term = [g for g in gene_list if g in PATIENT_GENES]
        categorized[cat].append({
            "term": term.get("term", ""),
            "description": term.get("description", ""),
            "p_value": p_value,
            "fdr": fdr,
            "gene_count": term.get("number_of_genes", 0),
            "background_count": term.get("number_of_genes_in_background", 0),
            "patient_genes": patient_genes_in_term,
            "all_genes": gene_list,
        })
    for cat in categorized:
        categorized[cat].sort(key=lambda x: x["fdr"])
    return dict(categorized)


def generate_report(
    network_data: list[dict],
    enrichment_data: list[dict],
    categorized: dict[str, list[dict]],
    pairwise_overlap: list[dict],
    pairwise_scores: dict[str, dict],
    gene_pathways: dict[str, set[str]],
) -> str:
    """Generate comprehensive markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    lines.append("# STRING Network Biology Analysis")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Genes:** {', '.join(PATIENT_GENES)}")
    lines.append(f"**Species:** Homo sapiens (taxid {SPECIES})")
    lines.append(f"**STRING version:** v12")
    lines.append(f"**Required score:** 700 (high confidence)")
    lines.append("")

    # --- Section 1: Network overview ---
    lines.append("## 1. Protein-Protein Interaction Network")
    lines.append("")
    all_nodes = set()
    for edge in network_data:
        all_nodes.add(edge.get("preferredName_A", ""))
        all_nodes.add(edge.get("preferredName_B", ""))
    lines.append(f"- **Total interactions:** {len(network_data)}")
    lines.append(f"- **Nodes in network:** {len(all_nodes)}")
    lines.append(f"- **Patient genes found:** "
                 f"{', '.join(g for g in PATIENT_GENES if g in all_nodes)}")
    missing = [g for g in PATIENT_GENES if g not in all_nodes]
    if missing:
        lines.append(f"- **Patient genes NOT in network:** {', '.join(missing)}")
    lines.append("")

    # Direct interactions between patient genes
    lines.append("### Direct interactions between patient genes")
    lines.append("")
    direct = []
    for edge in network_data:
        a = edge.get("preferredName_A", "")
        b = edge.get("preferredName_B", "")
        if a in PATIENT_GENES and b in PATIENT_GENES:
            direct.append(edge)
    if direct:
        lines.append("| Gene A | Gene B | Combined Score | Experimental | Database | Textmining | Co-expression |")
        lines.append("|--------|--------|---------------|-------------|----------|------------|---------------|")
        for edge in direct:
            lines.append(
                f"| {edge.get('preferredName_A', '')} "
                f"| {edge.get('preferredName_B', '')} "
                f"| {edge.get('score', 0):.3f} "
                f"| {edge.get('escore', 0):.3f} "
                f"| {edge.get('dscore', 0):.3f} "
                f"| {edge.get('tscore', 0):.3f} "
                f"| {edge.get('ascore', 0):.3f} |"
            )
    else:
        lines.append("No direct STRING interactions found between patient genes at "
                      "the required confidence threshold.")
    lines.append("")

    # Top interactors shared by multiple patient genes
    lines.append("### Shared first-shell interactors")
    lines.append("")
    gene_neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in network_data:
        a = edge.get("preferredName_A", "")
        b = edge.get("preferredName_B", "")
        if a in PATIENT_GENES and b not in PATIENT_GENES:
            gene_neighbors[a].add(b)
        if b in PATIENT_GENES and a not in PATIENT_GENES:
            gene_neighbors[b].add(a)

    # Find interactors connected to 2+ patient genes
    interactor_connections: dict[str, list[str]] = defaultdict(list)
    for gene, neighbors in gene_neighbors.items():
        for n in neighbors:
            interactor_connections[n].append(gene)
    shared_interactors = {k: sorted(v) for k, v in interactor_connections.items()
                          if len(v) >= 2}
    if shared_interactors:
        lines.append("| Interactor | Connected Patient Genes | # Connections |")
        lines.append("|------------|----------------------|---------------|")
        for interactor, genes in sorted(shared_interactors.items(),
                                         key=lambda x: len(x[1]), reverse=True):
            lines.append(
                f"| {interactor} | {', '.join(genes)} | {len(genes)} |"
            )
    else:
        lines.append("No shared first-shell interactors found.")
    lines.append("")

    # --- Section 2: Pairwise STRING scores ---
    lines.append("## 2. Pairwise STRING Functional Scores")
    lines.append("")
    lines.append("Direct STRING association scores between each pair of patient genes.")
    lines.append("")
    lines.append("| Gene A | Gene B | Combined | Experimental | Database | Textmining | Co-expression | Cooccurrence |")
    lines.append("|--------|--------|----------|-------------|----------|------------|---------------|-------------|")
    for pair_key, score_data in sorted(pairwise_scores.items()):
        if score_data.get("interactions"):
            edge = score_data["interactions"][0]
            lines.append(
                f"| {edge.get('preferredName_A', '')} "
                f"| {edge.get('preferredName_B', '')} "
                f"| {edge.get('score', 0):.3f} "
                f"| {edge.get('escore', 0):.3f} "
                f"| {edge.get('dscore', 0):.3f} "
                f"| {edge.get('tscore', 0):.3f} "
                f"| {edge.get('ascore', 0):.3f} "
                f"| {edge.get('nscore', 0):.3f} |"
            )
        else:
            g1, g2 = pair_key.split("-")
            lines.append(f"| {g1} | {g2} | -- | -- | -- | -- | -- | -- |")
    lines.append("")

    # Highlight SETBP1-IDH2 vs DNMT3A-IDH2
    lines.append("### Key comparison: SETBP1-IDH2 vs DNMT3A-IDH2")
    lines.append("")
    si = pairwise_scores.get("IDH2-SETBP1", pairwise_scores.get("SETBP1-IDH2", {}))
    di = pairwise_scores.get("DNMT3A-IDH2", pairwise_scores.get("IDH2-DNMT3A", {}))
    si_score = si["interactions"][0]["score"] if si.get("interactions") else 0
    di_score = di["interactions"][0]["score"] if di.get("interactions") else 0
    lines.append(f"- SETBP1-IDH2 combined score: **{si_score:.3f}**")
    lines.append(f"- DNMT3A-IDH2 combined score: **{di_score:.3f}**")
    if si_score > 0 and di_score > 0:
        if di_score > si_score:
            lines.append(f"- DNMT3A-IDH2 is {di_score/si_score:.1f}x more strongly "
                          "connected than SETBP1-IDH2 in STRING")
        elif si_score > di_score:
            lines.append(f"- SETBP1-IDH2 is {si_score/di_score:.1f}x more strongly "
                          "connected than DNMT3A-IDH2 in STRING")
    lines.append("")
    lines.append(
        "The SETBP1-IDH2 pair is mutually exclusive in GENIE (O/E=0.13, p=0.013). "
        "If STRING shows low functional connectivity between them relative to "
        "DNMT3A-IDH2 (which co-occur), this supports the hypothesis that mutations "
        "in functionally redundant pathways are selected against."
    )
    lines.append("")

    # --- Section 3: Pathway enrichment ---
    lines.append("## 3. Functional Enrichment")
    lines.append("")
    total_sig = sum(len(terms) for terms in categorized.values())
    lines.append(f"Total significant enrichment terms (FDR < 0.05): **{total_sig}**")
    lines.append("")
    for cat, terms in sorted(categorized.items()):
        lines.append(f"### {cat} ({len(terms)} terms)")
        lines.append("")
        lines.append("| Term | Description | FDR | Patient Genes |")
        lines.append("|------|-------------|-----|---------------|")
        for t in terms[:15]:
            pg = ", ".join(t["patient_genes"])
            lines.append(
                f"| {t['term']} | {t['description'][:80]} "
                f"| {t['fdr']:.2e} | {pg} |"
            )
        if len(terms) > 15:
            lines.append(f"| ... | *{len(terms) - 15} more terms* | | |")
        lines.append("")

    # --- Section 4: Pathway overlap matrix ---
    lines.append("## 4. Pairwise Pathway Overlap")
    lines.append("")
    lines.append("Jaccard similarity of pathway memberships between gene pairs. "
                 "Higher values indicate greater pathway redundancy.")
    lines.append("")

    # Summary table
    lines.append("| Gene A | Gene B | Shared | Union | Jaccard | Pathways A | Pathways B |")
    lines.append("|--------|--------|--------|-------|---------|------------|------------|")
    for ov in pairwise_overlap:
        lines.append(
            f"| {ov['gene_a']} | {ov['gene_b']} "
            f"| {ov['shared_count']} | {ov['union_count']} "
            f"| {ov['jaccard_similarity']:.4f} "
            f"| {ov['pathways_a']} | {ov['pathways_b']} |"
        )
    lines.append("")

    # Per-gene pathway counts
    lines.append("### Per-gene pathway membership")
    lines.append("")
    for gene in PATIENT_GENES:
        paths = gene_pathways.get(gene, set())
        lines.append(f"- **{gene}:** {len(paths)} pathways")
    lines.append("")

    # Top shared pathways for highest-overlap pairs
    lines.append("### Top shared pathways (highest Jaccard pairs)")
    lines.append("")
    for ov in pairwise_overlap[:3]:
        if ov["shared_count"] > 0:
            lines.append(f"**{ov['gene_a']} - {ov['gene_b']}** "
                          f"(Jaccard={ov['jaccard_similarity']:.4f}, "
                          f"{ov['shared_count']} shared):")
            lines.append("")
            for p in ov["shared_pathways"][:10]:
                lines.append(f"  - {p}")
            if len(ov["shared_pathways"]) > 10:
                lines.append(f"  - *...{len(ov['shared_pathways']) - 10} more*")
            lines.append("")

    # --- Section 5: Mutual exclusivity interpretation ---
    lines.append("## 5. Pathway Redundancy and Mutual Exclusivity")
    lines.append("")
    lines.append(
        "The pathway redundancy hypothesis states that mutations in genes with "
        "overlapping functional roles are mutually exclusive because one mutation "
        "is sufficient to disrupt the shared pathway. If two genes act on the same "
        "pathway, acquiring mutations in both provides no additional selective "
        "advantage (functional redundancy)."
    )
    lines.append("")

    # Find the SETBP1-IDH2 overlap entry
    si_ov = next((o for o in pairwise_overlap
                  if set([o["gene_a"], o["gene_b"]]) == {"SETBP1", "IDH2"}), None)
    di_ov = next((o for o in pairwise_overlap
                  if set([o["gene_a"], o["gene_b"]]) == {"DNMT3A", "IDH2"}), None)

    lines.append("### Evidence assessment")
    lines.append("")

    # SETBP1-IDH2 (mutually exclusive)
    lines.append("**SETBP1-IDH2 (mutually exclusive in GENIE, O/E=0.13, p=0.013):**")
    lines.append("")
    if si_ov:
        lines.append(f"- Pathway overlap: Jaccard={si_ov['jaccard_similarity']:.4f} "
                      f"({si_ov['shared_count']} shared pathways)")
    lines.append(f"- STRING functional score: {si_score:.3f}")
    # Compare with DNMT3A-IDH2 Jaccard for context
    di_jaccard = di_ov["jaccard_similarity"] if di_ov else 0
    si_jaccard = si_ov["jaccard_similarity"] if si_ov else 0
    if si_ov and abs(si_jaccard - di_jaccard) < 0.05:
        lines.append("- **Mixed evidence** for pathway redundancy: annotation overlap "
                      "is comparable to co-occurring DNMT3A-IDH2, so broad pathway "
                      "overlap alone does not explain the mutual exclusivity")
        lines.append("- The mutual exclusivity may instead be driven by synthetic "
                      "lethality, clonal competition, or mechanistic redundancy at a "
                      "finer level than captured by pathway annotations")
    elif si_ov and si_jaccard > di_jaccard + 0.05:
        lines.append("- **Supports** pathway redundancy hypothesis: substantially "
                      "higher pathway overlap than the co-occurring DNMT3A-IDH2 pair")
    elif si_ov and si_jaccard > 0:
        lines.append("- **Partial support** for pathway redundancy: some shared pathways "
                      "exist but overlap is modest")
    else:
        lines.append("- **Does not strongly support** pathway redundancy based on "
                      "annotated pathways alone")
    lines.append("")

    # DNMT3A-IDH2 (co-occurring)
    lines.append("**DNMT3A-IDH2 (co-occurring in AML):**")
    lines.append("")
    if di_ov:
        lines.append(f"- Pathway overlap: Jaccard={di_ov['jaccard_similarity']:.4f} "
                      f"({di_ov['shared_count']} shared pathways)")
    lines.append(f"- STRING functional score: {di_score:.3f}")
    lines.append("- DNMT3A and IDH2 are known to synergize in AML: DNMT3A causes focal "
                 "hypomethylation while IDH2 R140Q produces 2-HG causing "
                 "hypermethylation at different loci, creating a combined epigenetic "
                 "disruption greater than either alone")
    lines.append("")

    # Overall interpretation
    lines.append("### Summary")
    lines.append("")
    lines.append("| Pair | Co-occurrence | STRING Score | Jaccard | Interpretation |")
    lines.append("|------|-------------|-------------|---------|----------------|")
    for pair_key, score_data in sorted(pairwise_scores.items()):
        g1, g2 = pair_key.split("-")
        score_val = score_data["interactions"][0]["score"] if score_data.get("interactions") else 0
        ov_entry = next((o for o in pairwise_overlap
                        if set([o["gene_a"], o["gene_b"]]) == {g1, g2}), None)
        jaccard_val = ov_entry["jaccard_similarity"] if ov_entry else 0
        # Determine co-occurrence status
        if {g1, g2} == {"SETBP1", "IDH2"}:
            cooc = "Mutually exclusive"
        elif {g1, g2} == {"DNMT3A", "IDH2"}:
            cooc = "Co-occurring"
        else:
            cooc = "Variable"
        lines.append(f"| {g1}-{g2} | {cooc} | {score_val:.3f} | {jaccard_val:.4f} | |")
    lines.append("")

    lines.append("## 6. Methods")
    lines.append("")
    lines.append("- STRING v12 REST API (https://string-db.org/api/json/)")
    lines.append("- Network query: 5 patient genes + 20 first-shell interactors, "
                 "required score >= 700 (high confidence)")
    lines.append("- Enrichment: all STRING-supported databases, FDR < 0.05 filter")
    lines.append("- Pairwise overlap: Jaccard similarity = |A intersect B| / |A union B| "
                 "over pathway memberships")
    lines.append("- Pairwise scores: direct STRING queries for each of 10 gene pairs")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("STRING Network Biology Analysis")
    print("=" * 70)
    print()

    # Step 1: Query full network with first-shell interactors
    print("[1/5] Querying STRING network (5 genes + 20 interactors)...")
    network_data = query_string_network(PATIENT_GENES, add_nodes=20, required_score=700)
    time.sleep(REQUEST_DELAY)

    # Step 2: Query enrichment
    print("[2/5] Querying STRING functional enrichment...")
    enrichment_data = query_string_enrichment(PATIENT_GENES)
    time.sleep(REQUEST_DELAY)

    # Step 3: Compute pathway overlap
    print("[3/5] Computing pairwise pathway overlap...")
    gene_pathways = build_gene_pathway_map(enrichment_data)
    for gene in PATIENT_GENES:
        count = len(gene_pathways.get(gene, set()))
        print(f"  {gene}: {count} pathways")
    pairwise_overlap = compute_pairwise_overlap(gene_pathways)
    print(f"  Computed {len(pairwise_overlap)} gene pairs")

    # Step 4: Query pairwise scores for all 10 pairs
    print("[4/5] Querying pairwise STRING scores...")
    pairwise_scores: dict[str, dict] = {}
    for g1, g2 in combinations(PATIENT_GENES, 2):
        pair_key = f"{g1}-{g2}"
        print(f"  Querying {pair_key}...")
        interactions = query_pairwise_score(g1, g2)
        pairwise_scores[pair_key] = {"interactions": interactions}
        if interactions:
            score = interactions[0].get("score", 0)
            print(f"    -> score={score:.3f}")
        else:
            print("    -> no interaction found")
        time.sleep(REQUEST_DELAY)

    # Step 5: Categorize enrichment and generate report
    print("[5/5] Generating report...")
    categorized = categorize_enrichment(enrichment_data)
    for cat, terms in sorted(categorized.items()):
        print(f"  {cat}: {len(terms)} significant terms")

    # Save network data
    network_path = os.path.join(RESULTS_DIR, "string_network.json")
    with open(network_path, "w") as f:
        json.dump({
            "query_genes": PATIENT_GENES,
            "species": SPECIES,
            "required_score": 700,
            "add_nodes": 20,
            "timestamp": datetime.now().isoformat(),
            "interactions": network_data,
            "pairwise_scores": pairwise_scores,
        }, f, indent=2)
    print(f"  Saved network data: {network_path}")

    # Save enrichment data
    enrichment_path = os.path.join(RESULTS_DIR, "pathway_enrichment.json")
    # Convert sets to lists for JSON serialization
    serializable_pathways = {g: sorted(s) for g, s in gene_pathways.items()}
    serializable_overlap = []
    for ov in pairwise_overlap:
        entry = dict(ov)
        entry["shared_pathways"] = sorted(entry["shared_pathways"])
        serializable_overlap.append(entry)
    with open(enrichment_path, "w") as f:
        json.dump({
            "query_genes": PATIENT_GENES,
            "timestamp": datetime.now().isoformat(),
            "enrichment_raw": enrichment_data,
            "categorized_significant": {
                cat: terms for cat, terms in categorized.items()
            },
            "gene_pathway_map": serializable_pathways,
            "pairwise_overlap": serializable_overlap,
        }, f, indent=2)
    print(f"  Saved enrichment data: {enrichment_path}")

    # Generate and save report
    report = generate_report(
        network_data, enrichment_data, categorized,
        pairwise_overlap, pairwise_scores, gene_pathways,
    )
    report_path = os.path.join(RESULTS_DIR, "network_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved report: {report_path}")

    print()
    print("=" * 70)
    print("Analysis complete.")
    print(f"  Network:    {network_path}")
    print(f"  Enrichment: {enrichment_path}")
    print(f"  Report:     {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
