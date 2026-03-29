#!/usr/bin/env python3
"""
primekg_analysis.py -- PrimeKG (Harvard Precision Medicine Knowledge Graph) analysis
for the patient's 5 driver genes.

Patient mutation profile (5 confirmed drivers):
  DNMT3A R882H (VAF 39%) - DNA methylation / epigenetic regulation
  IDH2 R140Q   (VAF 2%)  - TCA cycle / 2-HG oncometabolite
  SETBP1 G870S (VAF 34%) - PP2A inhibition / MDS-MPN overlap
  PTPN11 E76Q  (VAF 29%) - RAS-MAPK signaling (gain-of-function)
  EZH2 V662A   (VAF 59%) - PRC2 chromatin remodeling (biallelic loss with -7)

Analyses:
  1. Download PrimeKG data from Harvard Dataverse (cached locally)
  2. For each gene: find all directly connected nodes (diseases, drugs, pathways, proteins)
  3. Shared pathway neighbors between all 10 gene pairs
  4. Pathway redundancy scores between gene pairs
  5. Convergence nodes (connected to 3+ patient genes)
  6. SETBP1-IDH2 mutual exclusivity hypothesis test via pathway redundancy

Outputs:
    - mutation_profile/results/ai_research/primekg_results.json
    - mutation_profile/results/ai_research/primekg_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/primekg_analysis.py

Runtime: ~2-5 min (first run downloads ~300MB, subsequent runs use cache)
Dependencies: requests, pandas, networkx
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent
DATA_DIR = PROJECT_DIR / "mutation_profile" / "data" / "primekg"
RESULTS_DIR = PROJECT_DIR / "mutation_profile" / "results" / "ai_research"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KG_CSV_PATH = DATA_DIR / "kg.csv"

# PrimeKG download URL (Harvard Dataverse)
PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"

# ---------------------------------------------------------------------------
# Patient genes
# ---------------------------------------------------------------------------
PATIENT_GENES = ["DNMT3A", "IDH2", "SETBP1", "PTPN11", "EZH2"]

GENE_DETAILS = {
    "DNMT3A": {
        "mutation": "R882H",
        "vaf": 0.39,
        "pathway": "DNA methylation / epigenetic regulation",
        "function": "DNA methyltransferase 3 alpha",
    },
    "IDH2": {
        "mutation": "R140Q",
        "vaf": 0.02,
        "pathway": "TCA cycle / 2-HG oncometabolite",
        "function": "Isocitrate dehydrogenase 2",
    },
    "SETBP1": {
        "mutation": "G870S",
        "vaf": 0.34,
        "pathway": "PP2A inhibition / self-renewal",
        "function": "SET binding protein 1",
    },
    "PTPN11": {
        "mutation": "E76Q",
        "vaf": 0.29,
        "pathway": "RAS-MAPK signaling",
        "function": "SHP2 protein tyrosine phosphatase",
    },
    "EZH2": {
        "mutation": "V662A",
        "vaf": 0.59,
        "pathway": "PRC2 chromatin remodeling",
        "function": "Enhancer of zeste homolog 2 (H3K27 methyltransferase)",
    },
}


# ---------------------------------------------------------------------------
# Step 1: Download PrimeKG
# ---------------------------------------------------------------------------
def download_primekg() -> Path:
    """Download PrimeKG kg.csv from Harvard Dataverse if not cached."""
    if KG_CSV_PATH.exists():
        size_mb = KG_CSV_PATH.stat().st_size / (1024 * 1024)
        print(f"[OK] PrimeKG already cached: {KG_CSV_PATH} ({size_mb:.1f} MB)")
        return KG_CSV_PATH

    print(f"[DOWNLOAD] Downloading PrimeKG from Harvard Dataverse...")
    print(f"  URL: {PRIMEKG_URL}")
    print(f"  Target: {KG_CSV_PATH}")

    start = time.time()
    response = requests.get(PRIMEKG_URL, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(KG_CSV_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192 * 16):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                mb = downloaded / (1024 * 1024)
                print(f"\r  Progress: {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    elapsed = time.time() - start
    size_mb = KG_CSV_PATH.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Downloaded {size_mb:.1f} MB in {elapsed:.0f}s")
    return KG_CSV_PATH


# ---------------------------------------------------------------------------
# Step 2: Load and index PrimeKG
# ---------------------------------------------------------------------------
def load_primekg(path: Path) -> pd.DataFrame:
    """Load PrimeKG CSV into a DataFrame."""
    print("[LOAD] Reading PrimeKG CSV...")
    start = time.time()
    df = pd.read_csv(path, low_memory=False)
    elapsed = time.time() - start
    print(f"[OK] Loaded {len(df):,} edges in {elapsed:.1f}s")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Relation types: {df['relation'].nunique() if 'relation' in df.columns else 'N/A'}")
    return df


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a NetworkX graph from PrimeKG edges (vectorized for speed)."""
    print("[BUILD] Constructing NetworkX graph...")
    G = nx.MultiDiGraph()

    # Add nodes from x side
    x_nodes = df[["x_name", "x_type", "x_id"]].drop_duplicates("x_name")
    for _, row in x_nodes.iterrows():
        name = str(row["x_name"])
        if name:
            G.add_node(name, node_type=str(row["x_type"]), node_id=str(row["x_id"]))

    # Add nodes from y side
    y_nodes = df[["y_name", "y_type", "y_id"]].drop_duplicates("y_name")
    for _, row in y_nodes.iterrows():
        name = str(row["y_name"])
        if name and name not in G:
            G.add_node(name, node_type=str(row["y_type"]), node_id=str(row["y_id"]))

    # Add edges in bulk
    display_col = "display_relation" if "display_relation" in df.columns else "relation"
    edges = list(zip(
        df["x_name"].astype(str),
        df["y_name"].astype(str),
        df["relation"].astype(str),
        df[display_col].astype(str),
    ))
    for x, y, rel, disp in edges:
        if x and y:
            G.add_edge(x, y, relation=rel, display_relation=disp)

    print(f"[OK] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def build_simple_graph(df: pd.DataFrame) -> nx.Graph:
    """Build an undirected simple graph for shortest path computation."""
    print("[BUILD] Constructing simple undirected graph for path analysis...")
    G = nx.Graph()

    edges = list(zip(df["x_name"].astype(str), df["y_name"].astype(str)))
    G.add_edges_from(edges)

    print(f"[OK] Simple graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ---------------------------------------------------------------------------
# Step 3: Gene neighborhood analysis
# ---------------------------------------------------------------------------
def find_gene_neighbors(G: nx.MultiDiGraph, gene: str) -> dict:
    """Find all directly connected nodes for a gene, organized by type."""
    # Try exact match first, then case-insensitive
    node = None
    if gene in G:
        node = gene
    else:
        for n in G.nodes():
            if n.upper() == gene.upper():
                node = n
                break

    if node is None:
        # Try partial match (gene name might be embedded in protein name)
        candidates = [n for n in G.nodes() if gene.upper() in n.upper()]
        if candidates:
            # Prefer exact gene/protein node
            for c in candidates:
                ntype = G.nodes[c].get("node_type", "").lower()
                if ntype in ("gene/protein", "gene", "protein"):
                    node = c
                    break
            if node is None:
                node = candidates[0]

    if node is None:
        return {
            "gene": gene,
            "found_in_kg": False,
            "node_name": None,
            "neighbors_by_type": {},
            "neighbor_count": 0,
            "relations": [],
        }

    neighbors_by_type = defaultdict(list)
    relations = []

    # Outgoing edges
    for _, target, data in G.out_edges(node, data=True):
        target_type = G.nodes[target].get("node_type", "unknown")
        rel = data.get("display_relation", data.get("relation", "unknown"))
        neighbors_by_type[target_type].append(target)
        relations.append({
            "source": node,
            "target": target,
            "target_type": target_type,
            "relation": rel,
            "direction": "outgoing",
        })

    # Incoming edges
    for source, _, data in G.in_edges(node, data=True):
        source_type = G.nodes[source].get("node_type", "unknown")
        rel = data.get("display_relation", data.get("relation", "unknown"))
        neighbors_by_type[source_type].append(source)
        relations.append({
            "source": source,
            "target": node,
            "source_type": source_type,
            "relation": rel,
            "direction": "incoming",
        })

    # Deduplicate neighbor lists
    neighbors_by_type = {k: sorted(set(v)) for k, v in neighbors_by_type.items()}
    total = sum(len(v) for v in neighbors_by_type.values())

    return {
        "gene": gene,
        "found_in_kg": True,
        "node_name": node,
        "neighbors_by_type": dict(neighbors_by_type),
        "neighbor_count": total,
        "relations": relations,
    }


# ---------------------------------------------------------------------------
# Step 4: Shared neighbors and pathway redundancy
# ---------------------------------------------------------------------------
def compute_pairwise_overlap(gene_neighborhoods: dict) -> dict:
    """Compute shared neighbors between all gene pairs (10 pairs from 5 genes)."""
    pairs = {}

    for g1, g2 in combinations(PATIENT_GENES, 2):
        n1 = gene_neighborhoods.get(g1, {})
        n2 = gene_neighborhoods.get(g2, {})

        if not n1.get("found_in_kg") or not n2.get("found_in_kg"):
            pairs[f"{g1}-{g2}"] = {
                "gene_1": g1,
                "gene_2": g2,
                "computable": False,
                "reason": f"{'g1' if not n1.get('found_in_kg') else g2} not found in PrimeKG",
            }
            continue

        # Flatten all neighbors for each gene
        all_n1 = set()
        for nodes in n1.get("neighbors_by_type", {}).values():
            all_n1.update(nodes)

        all_n2 = set()
        for nodes in n2.get("neighbors_by_type", {}).values():
            all_n2.update(nodes)

        shared = all_n1 & all_n2
        union = all_n1 | all_n2

        # Jaccard similarity
        jaccard = len(shared) / len(union) if union else 0.0

        # Overlap coefficient (min-based)
        overlap_coeff = len(shared) / min(len(all_n1), len(all_n2)) if min(len(all_n1), len(all_n2)) > 0 else 0.0

        # Categorize shared nodes by type
        shared_by_type = defaultdict(list)
        for node in shared:
            ntype = "unknown"
            if node in (G.nodes if 'G' in dir() else {}):
                pass
            # We need the graph to get types; store for later
            shared_by_type["all"].append(node)

        pairs[f"{g1}-{g2}"] = {
            "gene_1": g1,
            "gene_2": g2,
            "computable": True,
            "neighbors_gene_1": len(all_n1),
            "neighbors_gene_2": len(all_n2),
            "shared_neighbors": len(shared),
            "shared_neighbor_names": sorted(shared),
            "union_size": len(union),
            "jaccard_similarity": round(jaccard, 4),
            "overlap_coefficient": round(overlap_coeff, 4),
        }

    return pairs


def compute_pairwise_overlap_with_graph(
    gene_neighborhoods: dict, G: nx.MultiDiGraph
) -> dict:
    """Compute shared neighbors between all gene pairs with type information."""
    pairs = {}

    for g1, g2 in combinations(PATIENT_GENES, 2):
        n1 = gene_neighborhoods.get(g1, {})
        n2 = gene_neighborhoods.get(g2, {})

        if not n1.get("found_in_kg") or not n2.get("found_in_kg"):
            pairs[f"{g1}-{g2}"] = {
                "gene_1": g1,
                "gene_2": g2,
                "computable": False,
                "reason": f"{g1 if not n1.get('found_in_kg') else g2} not found in PrimeKG",
            }
            continue

        # Flatten all neighbors
        all_n1 = set()
        for nodes in n1.get("neighbors_by_type", {}).values():
            all_n1.update(nodes)

        all_n2 = set()
        for nodes in n2.get("neighbors_by_type", {}).values():
            all_n2.update(nodes)

        shared = all_n1 & all_n2
        union = all_n1 | all_n2

        jaccard = len(shared) / len(union) if union else 0.0
        overlap_coeff = (
            len(shared) / min(len(all_n1), len(all_n2))
            if min(len(all_n1), len(all_n2)) > 0
            else 0.0
        )

        # Categorize shared nodes by type
        shared_by_type = defaultdict(list)
        for node in shared:
            ntype = G.nodes[node].get("node_type", "unknown") if node in G else "unknown"
            shared_by_type[ntype].append(node)
        shared_by_type = {k: sorted(v) for k, v in shared_by_type.items()}

        pairs[f"{g1}-{g2}"] = {
            "gene_1": g1,
            "gene_2": g2,
            "computable": True,
            "neighbors_gene_1": len(all_n1),
            "neighbors_gene_2": len(all_n2),
            "shared_neighbors": len(shared),
            "shared_by_type": dict(shared_by_type),
            "union_size": len(union),
            "jaccard_similarity": round(jaccard, 4),
            "overlap_coefficient": round(overlap_coeff, 4),
        }

    return pairs


# ---------------------------------------------------------------------------
# Step 5: Convergence nodes
# ---------------------------------------------------------------------------
def find_convergence_nodes(gene_neighborhoods: dict, min_genes: int = 3) -> dict:
    """Find nodes connected to min_genes or more patient genes."""
    node_gene_map = defaultdict(set)

    for gene, info in gene_neighborhoods.items():
        if not info.get("found_in_kg"):
            continue
        for nodes in info.get("neighbors_by_type", {}).values():
            for node in nodes:
                node_gene_map[node].add(gene)

    convergence = {}
    for node, genes in node_gene_map.items():
        if len(genes) >= min_genes:
            convergence[node] = {
                "connected_genes": sorted(genes),
                "gene_count": len(genes),
            }

    # Sort by gene count descending
    convergence = dict(
        sorted(convergence.items(), key=lambda x: x[1]["gene_count"], reverse=True)
    )
    return convergence


def annotate_convergence_nodes(
    convergence: dict, G: nx.MultiDiGraph
) -> dict:
    """Add node type information to convergence nodes."""
    annotated = {}
    for node, info in convergence.items():
        ntype = G.nodes[node].get("node_type", "unknown") if node in G else "unknown"
        annotated[node] = {
            **info,
            "node_type": ntype,
        }
    return annotated


# ---------------------------------------------------------------------------
# Step 6: Drug-gene-disease multi-hop paths
# ---------------------------------------------------------------------------
def find_drug_gene_disease_paths(
    G: nx.MultiDiGraph, gene_neighborhoods: dict
) -> list:
    """Find drug -> gene -> disease paths through patient genes."""
    paths = []

    for gene in PATIENT_GENES:
        info = gene_neighborhoods.get(gene, {})
        if not info.get("found_in_kg"):
            continue

        node_name = info["node_name"]

        # Find drug neighbors
        drugs = set()
        diseases = set()

        for rel in info.get("relations", []):
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            src_type = rel.get("source_type", "")
            tgt_type = rel.get("target_type", "")

            if "drug" in src_type.lower():
                drugs.add(src)
            if "drug" in tgt_type.lower():
                drugs.add(tgt)
            if "disease" in src_type.lower():
                diseases.add(src)
            if "disease" in tgt_type.lower():
                diseases.add(tgt)

        for drug in drugs:
            for disease in diseases:
                paths.append({
                    "drug": drug,
                    "gene": gene,
                    "disease": disease,
                    "path": f"{drug} -> {node_name} -> {disease}",
                })

    return paths


# ---------------------------------------------------------------------------
# Step 7: SETBP1-IDH2 mutual exclusivity hypothesis
# ---------------------------------------------------------------------------
def analyze_setbp1_idh2_redundancy(
    gene_neighborhoods: dict, pairwise: dict
) -> dict:
    """
    Test hypothesis: SETBP1-IDH2 mutual exclusivity (OR=0.22) is due to
    pathway redundancy - i.e. they affect overlapping downstream targets,
    making co-occurrence provide diminishing returns for the tumor.
    """
    pair_key = "SETBP1-IDH2"
    # Could also be IDH2-SETBP1 depending on ordering
    if pair_key not in pairwise:
        pair_key = "IDH2-SETBP1"

    pair_data = pairwise.get(pair_key, {})

    setbp1_info = gene_neighborhoods.get("SETBP1", {})
    idh2_info = gene_neighborhoods.get("IDH2", {})

    # Collect all neighbor sets
    setbp1_neighbors = set()
    for nodes in setbp1_info.get("neighbors_by_type", {}).values():
        setbp1_neighbors.update(nodes)

    idh2_neighbors = set()
    for nodes in idh2_info.get("neighbors_by_type", {}).values():
        idh2_neighbors.update(nodes)

    shared = setbp1_neighbors & idh2_neighbors

    # Compare with other pairs to see if SETBP1-IDH2 has unusually high overlap
    all_jaccard = {}
    for key, data in pairwise.items():
        if data.get("computable"):
            all_jaccard[key] = data["jaccard_similarity"]

    setbp1_idh2_jaccard = pair_data.get("jaccard_similarity", 0.0) if pair_data.get("computable") else None
    avg_jaccard = sum(all_jaccard.values()) / len(all_jaccard) if all_jaccard else 0.0

    # Rank this pair among all pairs
    sorted_pairs = sorted(all_jaccard.items(), key=lambda x: x[1], reverse=True)
    rank = None
    for i, (k, _) in enumerate(sorted_pairs):
        if k == pair_key:
            rank = i + 1
            break

    hypothesis_result = {
        "pair": pair_key,
        "observed_mutual_exclusivity_or": 0.22,
        "setbp1_neighbor_count": len(setbp1_neighbors),
        "idh2_neighbor_count": len(idh2_neighbors),
        "shared_neighbor_count": len(shared),
        "shared_neighbors_sample": sorted(shared)[:20],
        "jaccard_similarity": setbp1_idh2_jaccard,
        "mean_jaccard_all_pairs": round(avg_jaccard, 4),
        "rank_among_10_pairs": rank,
        "all_pair_jaccard_scores": dict(sorted_pairs),
        "hypothesis_supported": None,
        "interpretation": "",
    }

    if setbp1_idh2_jaccard is not None:
        if setbp1_idh2_jaccard > avg_jaccard * 1.5:
            hypothesis_result["hypothesis_supported"] = True
            hypothesis_result["interpretation"] = (
                f"SETBP1-IDH2 Jaccard ({setbp1_idh2_jaccard:.4f}) is substantially "
                f"above the mean ({avg_jaccard:.4f}), supporting the hypothesis that "
                f"pathway redundancy underlies their mutual exclusivity (OR=0.22). "
                f"Shared targets: {len(shared)} nodes."
            )
        elif setbp1_idh2_jaccard > avg_jaccard:
            hypothesis_result["hypothesis_supported"] = "partial"
            hypothesis_result["interpretation"] = (
                f"SETBP1-IDH2 Jaccard ({setbp1_idh2_jaccard:.4f}) is above "
                f"the mean ({avg_jaccard:.4f}) but not dramatically so. "
                f"Partial support for pathway redundancy hypothesis. "
                f"Other mechanisms (e.g., synthetic lethality, clonal competition) "
                f"may also contribute to the observed mutual exclusivity."
            )
        else:
            hypothesis_result["hypothesis_supported"] = False
            hypothesis_result["interpretation"] = (
                f"SETBP1-IDH2 Jaccard ({setbp1_idh2_jaccard:.4f}) is at or below "
                f"the mean ({avg_jaccard:.4f}). Pathway redundancy alone does NOT "
                f"explain the mutual exclusivity (OR=0.22). Alternative hypotheses: "
                f"synthetic lethality, clonal competition, or distinct cell-of-origin "
                f"requirements."
            )

    return hypothesis_result


# ---------------------------------------------------------------------------
# Step 8: Shortest paths between gene pairs
# ---------------------------------------------------------------------------
def compute_shortest_paths(G_simple: nx.Graph, gene_neighborhoods: dict) -> dict:
    """Compute shortest paths between all 10 pairs of patient genes."""
    print("[ANALYZE] Computing shortest paths between gene pairs...")
    paths = {}

    for g1, g2 in combinations(PATIENT_GENES, 2):
        n1 = gene_neighborhoods.get(g1, {})
        n2 = gene_neighborhoods.get(g2, {})

        if not n1.get("found_in_kg") or not n2.get("found_in_kg"):
            paths[f"{g1}-{g2}"] = {
                "gene_1": g1,
                "gene_2": g2,
                "distance": None,
                "path": None,
                "reason": "Gene not found in PrimeKG",
            }
            continue

        node1 = n1["node_name"]
        node2 = n2["node_name"]

        try:
            path = nx.shortest_path(G_simple, node1, node2)
            distance = len(path) - 1
            # Annotate path with node types from the directed graph
            paths[f"{g1}-{g2}"] = {
                "gene_1": g1,
                "gene_2": g2,
                "distance": distance,
                "path": path,
            }
            print(f"  {g1} <-> {g2}: distance={distance}, path={' -> '.join(path)}")
        except nx.NetworkXNoPath:
            paths[f"{g1}-{g2}"] = {
                "gene_1": g1,
                "gene_2": g2,
                "distance": None,
                "path": None,
                "reason": "No path exists",
            }
            print(f"  {g1} <-> {g2}: NO PATH")

    return paths


# ---------------------------------------------------------------------------
# Step 9: Pathway convergence score for mutually exclusive pairs
# ---------------------------------------------------------------------------
def compute_pathway_convergence(
    gene_neighborhoods: dict, G: nx.MultiDiGraph
) -> dict:
    """Compute pathway convergence score: shared downstream targets for each pair.

    Focuses on biologically meaningful node types (biological_process, pathway,
    disease, molecular_function) rather than anatomy or generic nodes.
    """
    print("[ANALYZE] Computing pathway convergence scores...")

    MEANINGFUL_TYPES = {
        "biological_process",
        "pathway",
        "disease",
        "molecular_function",
        "cellular_component",
        "gene/protein",
        "drug",
        "effect/phenotype",
    }

    convergence = {}
    for g1, g2 in combinations(PATIENT_GENES, 2):
        n1 = gene_neighborhoods.get(g1, {})
        n2 = gene_neighborhoods.get(g2, {})

        if not n1.get("found_in_kg") or not n2.get("found_in_kg"):
            convergence[f"{g1}-{g2}"] = {
                "score": 0.0,
                "meaningful_shared": 0,
                "total_shared": 0,
            }
            continue

        # Collect typed neighbors
        typed_n1 = defaultdict(set)
        for ntype, nodes in n1.get("neighbors_by_type", {}).items():
            typed_n1[ntype].update(nodes)

        typed_n2 = defaultdict(set)
        for ntype, nodes in n2.get("neighbors_by_type", {}).items():
            typed_n2[ntype].update(nodes)

        # All neighbors
        all_n1 = set()
        for nodes in typed_n1.values():
            all_n1.update(nodes)
        all_n2 = set()
        for nodes in typed_n2.values():
            all_n2.update(nodes)

        total_shared = all_n1 & all_n2

        # Meaningful shared (non-anatomy)
        meaningful_n1 = set()
        meaningful_n2 = set()
        for ntype in MEANINGFUL_TYPES:
            meaningful_n1.update(typed_n1.get(ntype, set()))
            meaningful_n2.update(typed_n2.get(ntype, set()))

        meaningful_shared = meaningful_n1 & meaningful_n2
        meaningful_union = meaningful_n1 | meaningful_n2

        meaningful_jaccard = (
            len(meaningful_shared) / len(meaningful_union) if meaningful_union else 0.0
        )

        # Shared by type
        shared_by_type = {}
        for ntype in MEANINGFUL_TYPES:
            s1 = typed_n1.get(ntype, set())
            s2 = typed_n2.get(ntype, set())
            common = s1 & s2
            if common:
                shared_by_type[ntype] = sorted(common)

        convergence[f"{g1}-{g2}"] = {
            "meaningful_jaccard": round(meaningful_jaccard, 4),
            "meaningful_shared": len(meaningful_shared),
            "meaningful_union": len(meaningful_union),
            "total_shared": len(total_shared),
            "shared_by_type": shared_by_type,
        }
        print(
            f"  {g1}-{g2}: meaningful_jaccard={meaningful_jaccard:.4f}, "
            f"meaningful_shared={len(meaningful_shared)}, "
            f"total_shared={len(total_shared)}"
        )

    return convergence


# ---------------------------------------------------------------------------
# Step 10: Pathway redundancy scores
# ---------------------------------------------------------------------------
def compute_redundancy_matrix(pairwise: dict) -> dict:
    """Compute a pathway redundancy matrix for all gene pairs."""
    matrix = {}
    for gene in PATIENT_GENES:
        matrix[gene] = {}
        for gene2 in PATIENT_GENES:
            if gene == gene2:
                matrix[gene][gene2] = 1.0
            else:
                key = f"{gene}-{gene2}"
                alt_key = f"{gene2}-{gene}"
                data = pairwise.get(key, pairwise.get(alt_key, {}))
                if data.get("computable"):
                    matrix[gene][gene2] = data["jaccard_similarity"]
                else:
                    matrix[gene][gene2] = None

    return matrix


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    gene_neighborhoods: dict,
    pairwise: dict,
    convergence: dict,
    drug_paths: list,
    hypothesis: dict,
    redundancy_matrix: dict,
    graph_stats: dict,
    shortest_paths: dict = None,
    pathway_convergence: dict = None,
) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# PrimeKG Analysis - 5-Gene Driver Mutation Profile")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Data source:** PrimeKG (Harvard Precision Medicine Knowledge Graph)")
    lines.append(f"**Graph size:** {graph_stats['nodes']:,} nodes, {graph_stats['edges']:,} edges")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Gene overview
    lines.append("## 1. Patient Gene Summary")
    lines.append("")
    lines.append("| Gene | Mutation | VAF | Pathway | Found in PrimeKG | Neighbors |")
    lines.append("|------|----------|-----|---------|-----------------|-----------|")
    for gene in PATIENT_GENES:
        d = GENE_DETAILS[gene]
        info = gene_neighborhoods.get(gene, {})
        found = "Yes" if info.get("found_in_kg") else "No"
        count = info.get("neighbor_count", 0)
        lines.append(
            f"| {gene} | {d['mutation']} | {d['vaf']:.0%} | {d['pathway']} | {found} | {count} |"
        )
    lines.append("")

    # Neighbor details per gene
    lines.append("## 2. Direct Connections per Gene")
    lines.append("")
    for gene in PATIENT_GENES:
        info = gene_neighborhoods.get(gene, {})
        if not info.get("found_in_kg"):
            lines.append(f"### {gene} - NOT FOUND in PrimeKG")
            lines.append("")
            continue

        lines.append(f"### {gene} ({info.get('node_name', gene)})")
        lines.append(f"Total neighbors: {info['neighbor_count']}")
        lines.append("")
        lines.append("| Node Type | Count | Examples |")
        lines.append("|-----------|-------|----------|")
        for ntype, nodes in sorted(info.get("neighbors_by_type", {}).items()):
            examples = ", ".join(nodes[:5])
            if len(nodes) > 5:
                examples += f" (+{len(nodes)-5} more)"
            lines.append(f"| {ntype} | {len(nodes)} | {examples} |")
        lines.append("")

    # Shortest paths
    if shortest_paths:
        lines.append("## 3. Shortest Paths Between Gene Pairs")
        lines.append("")
        lines.append("| Gene Pair | Distance (hops) | Path |")
        lines.append("|-----------|----------------|------|")
        for key, data in sorted(shortest_paths.items(), key=lambda x: x[1].get("distance") or 999):
            dist = data.get("distance")
            path = data.get("path")
            if dist is not None and path:
                path_str = " -> ".join(path)
                lines.append(f"| {key} | {dist} | {path_str} |")
            else:
                reason = data.get("reason", "No path")
                lines.append(f"| {key} | N/A | {reason} |")
        lines.append("")

        distances = [d["distance"] for d in shortest_paths.values() if d.get("distance") is not None]
        if distances:
            avg = sum(distances) / len(distances)
            lines.append(
                f"Average path length: **{avg:.1f} hops** "
                f"(range: {min(distances)}-{max(distances)}). "
            )
            direct = [k for k, v in shortest_paths.items() if v.get("distance") == 1]
            if direct:
                lines.append(f"Directly connected pairs: {', '.join(direct)}.")
            lines.append("")

    # Pairwise overlap
    lines.append("## 4. Pairwise Pathway Overlap (10 gene pairs)")
    lines.append("")
    lines.append("| Gene Pair | Shared Neighbors | Jaccard | Overlap Coeff | Interpretation |")
    lines.append("|-----------|-----------------|---------|---------------|----------------|")
    sorted_pairs = sorted(
        pairwise.items(),
        key=lambda x: x[1].get("jaccard_similarity", 0) if x[1].get("computable") else -1,
        reverse=True,
    )
    for key, data in sorted_pairs:
        if not data.get("computable"):
            lines.append(f"| {data['gene_1']}-{data['gene_2']} | N/A | N/A | N/A | {data.get('reason', 'N/A')} |")
            continue
        interp = ""
        j = data["jaccard_similarity"]
        if j > 0.3:
            interp = "High redundancy"
        elif j > 0.1:
            interp = "Moderate overlap"
        elif j > 0.01:
            interp = "Low overlap"
        else:
            interp = "Minimal overlap"
        lines.append(
            f"| {data['gene_1']}-{data['gene_2']} | {data['shared_neighbors']} "
            f"| {j:.4f} | {data['overlap_coefficient']:.4f} | {interp} |"
        )
    lines.append("")

    # Redundancy matrix
    lines.append("## 5. Pathway Redundancy Matrix (Jaccard similarity)")
    lines.append("")
    header = "| | " + " | ".join(PATIENT_GENES) + " |"
    sep = "|---|" + "|".join(["---"] * len(PATIENT_GENES)) + "|"
    lines.append(header)
    lines.append(sep)
    for g1 in PATIENT_GENES:
        row = f"| **{g1}** |"
        for g2 in PATIENT_GENES:
            val = redundancy_matrix.get(g1, {}).get(g2)
            if val is None:
                row += " N/A |"
            elif g1 == g2:
                row += " 1.000 |"
            else:
                row += f" {val:.4f} |"
        lines.append(row)
    lines.append("")

    # Convergence nodes
    lines.append("## 6. Convergence Nodes (connected to 3+ patient genes)")
    lines.append("")
    if convergence:
        lines.append(f"Found **{len(convergence)}** convergence nodes.")
        lines.append("")
        lines.append("| Node | Type | Connected Genes | Gene Count |")
        lines.append("|------|------|----------------|------------|")
        for node, info in list(convergence.items())[:50]:
            genes_str = ", ".join(info["connected_genes"])
            ntype = info.get("node_type", "unknown")
            lines.append(f"| {node} | {ntype} | {genes_str} | {info['gene_count']} |")

        if len(convergence) > 50:
            lines.append(f"| ... | ... | ... | ({len(convergence)-50} more) |")
    else:
        lines.append("No convergence nodes found (no nodes connected to 3+ patient genes).")
    lines.append("")

    # Count convergence by type
    if convergence:
        type_counts = Counter(v.get("node_type", "unknown") for v in convergence.values())
        lines.append("### Convergence by node type")
        lines.append("")
        lines.append("| Node Type | Count |")
        lines.append("|-----------|-------|")
        for ntype, count in type_counts.most_common():
            lines.append(f"| {ntype} | {count} |")
        lines.append("")

    # Drug-gene-disease paths
    lines.append("## 7. Drug-Gene-Disease Paths")
    lines.append("")
    if drug_paths:
        lines.append(f"Found **{len(drug_paths)}** drug-gene-disease paths through patient genes.")
        lines.append("")

        # Group by gene
        by_gene = defaultdict(list)
        for p in drug_paths:
            by_gene[p["gene"]].append(p)

        for gene in PATIENT_GENES:
            paths = by_gene.get(gene, [])
            if not paths:
                continue
            lines.append(f"### {gene} ({len(paths)} paths)")
            lines.append("")
            # Show unique drugs
            unique_drugs = sorted(set(p["drug"] for p in paths))
            unique_diseases = sorted(set(p["disease"] for p in paths))
            lines.append(f"- **Drugs ({len(unique_drugs)}):** {', '.join(unique_drugs[:10])}")
            if len(unique_drugs) > 10:
                lines.append(f"  (+{len(unique_drugs)-10} more)")
            lines.append(f"- **Diseases ({len(unique_diseases)}):** {', '.join(unique_diseases[:10])}")
            if len(unique_diseases) > 10:
                lines.append(f"  (+{len(unique_diseases)-10} more)")
            lines.append("")
    else:
        lines.append("No drug-gene-disease paths found.")
    lines.append("")

    # SETBP1-IDH2 hypothesis
    lines.append("## 8. SETBP1-IDH2 Mutual Exclusivity Hypothesis")
    lines.append("")
    lines.append("**Background:** IDH2-SETBP1 show significant mutual exclusivity ")
    lines.append("(OR=0.22, p<0.05 in IPSS-M dataset). The combination has never been ")
    lines.append("observed with DNMT3A in any database (~10,000 patients).")
    lines.append("")
    lines.append("**Hypothesis:** Pathway redundancy makes co-occurrence provide ")
    lines.append("diminishing fitness returns for the tumor.")
    lines.append("")

    h = hypothesis
    if h.get("jaccard_similarity") is not None:
        lines.append(f"- SETBP1 neighbors: {h['setbp1_neighbor_count']}")
        lines.append(f"- IDH2 neighbors: {h['idh2_neighbor_count']}")
        lines.append(f"- Shared neighbors: {h['shared_neighbor_count']}")
        lines.append(f"- Jaccard similarity: {h['jaccard_similarity']:.4f}")
        lines.append(f"- Mean Jaccard (all pairs): {h['mean_jaccard_all_pairs']:.4f}")
        lines.append(f"- Rank among 10 pairs: #{h['rank_among_10_pairs']}")
        lines.append("")
        lines.append(f"**Supported:** {h['hypothesis_supported']}")
        lines.append("")
        lines.append(f"**Interpretation:** {h['interpretation']}")
    else:
        lines.append("Could not compute - one or both genes not found in PrimeKG.")
    lines.append("")

    # All pairs ranked
    if h.get("all_pair_jaccard_scores"):
        lines.append("### All pairs ranked by Jaccard similarity")
        lines.append("")
        lines.append("| Rank | Pair | Jaccard |")
        lines.append("|------|------|---------|")
        for i, (pair, score) in enumerate(h["all_pair_jaccard_scores"].items(), 1):
            marker = " **<--**" if pair in ("SETBP1-IDH2", "IDH2-SETBP1") else ""
            lines.append(f"| {i} | {pair} | {score:.4f}{marker} |")
        lines.append("")

    # Pathway convergence
    if pathway_convergence:
        lines.append("## 9. Pathway Convergence Scores (excluding anatomy)")
        lines.append("")
        lines.append(
            "Jaccard index computed on biologically meaningful neighbor types only "
            "(biological_process, pathway, disease, molecular_function, cellular_component, "
            "gene/protein, drug, effect/phenotype). Anatomy nodes excluded to avoid "
            "co-expression bias."
        )
        lines.append("")
        lines.append("| Gene Pair | Meaningful Jaccard | Meaningful Shared | Total Shared |")
        lines.append("|-----------|-------------------|------------------|-------------|")
        ranked = sorted(
            pathway_convergence.items(),
            key=lambda x: x[1].get("meaningful_jaccard", 0),
            reverse=True,
        )
        for pair, data in ranked:
            mj = data.get("meaningful_jaccard", 0)
            ms = data.get("meaningful_shared", 0)
            ts = data.get("total_shared", 0)
            lines.append(f"| {pair} | {mj:.4f} | {ms} | {ts} |")
        lines.append("")

        # Detail for top pair
        if ranked:
            top_pair, top_data = ranked[0]
            lines.append(f"**Most convergent pair (functional):** {top_pair} "
                         f"(meaningful Jaccard={top_data['meaningful_jaccard']:.4f})")
            lines.append("")
            shared_by_type = top_data.get("shared_by_type", {})
            if shared_by_type:
                lines.append(f"Shared connections for {top_pair} by type:")
                lines.append("")
                for ntype, nodes in sorted(shared_by_type.items()):
                    sample = ", ".join(nodes[:10])
                    extra = f" (+{len(nodes)-10} more)" if len(nodes) > 10 else ""
                    lines.append(f"- **{ntype}** ({len(nodes)}): {sample}{extra}")
                lines.append("")

    # Clinical significance
    lines.append("## 10. Clinical Significance")
    lines.append("")
    lines.append("### Key findings from PrimeKG analysis:")
    lines.append("")

    # Count found genes
    found = sum(1 for g in PATIENT_GENES if gene_neighborhoods.get(g, {}).get("found_in_kg"))
    lines.append(f"1. **{found}/5 patient genes** represented in PrimeKG")

    if convergence:
        top_5 = list(convergence.items())[:5]
        conv_genes_max = max(v["gene_count"] for v in convergence.values())
        lines.append(
            f"2. **{len(convergence)} convergence nodes** found (max {conv_genes_max} genes)"
        )
        lines.append(f"   Top convergence nodes: {', '.join(n for n, _ in top_5)}")

    if drug_paths:
        unique_drugs = sorted(set(p["drug"] for p in drug_paths))
        lines.append(f"3. **{len(unique_drugs)} unique drugs** connected to patient genes")

    lines.append("")
    lines.append("### Limitations")
    lines.append("")
    lines.append("- PrimeKG is a general biomedical knowledge graph, not MDS/AML-specific")
    lines.append("- Edge semantics vary (protein-protein, drug-target, gene-disease associations)")
    lines.append("- Jaccard similarity on heterogeneous graphs has known biases toward high-degree nodes")
    lines.append("- Mutual exclusivity mechanisms are multifactorial; pathway redundancy is one component")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PrimeKG Analysis - 5-Gene Driver Mutation Profile")
    print("=" * 70)
    print()

    # Step 1: Download
    kg_path = download_primekg()

    # Step 2: Load
    df = load_primekg(kg_path)

    # Build directed graph (for neighborhood analysis)
    G = build_graph(df)
    graph_stats = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

    # Build simple undirected graph (for shortest path computation)
    G_simple = build_simple_graph(df)

    # Step 3: Gene neighborhoods
    print("\n[ANALYZE] Finding gene neighborhoods...")
    gene_neighborhoods = {}
    for gene in PATIENT_GENES:
        info = find_gene_neighbors(G, gene)
        gene_neighborhoods[gene] = info
        status = "FOUND" if info["found_in_kg"] else "NOT FOUND"
        print(f"  {gene}: {status} ({info['neighbor_count']} neighbors)")

    # Step 4: Pairwise overlap
    print("\n[ANALYZE] Computing pairwise pathway overlap...")
    pairwise = compute_pairwise_overlap_with_graph(gene_neighborhoods, G)
    for key, data in pairwise.items():
        if data.get("computable"):
            print(f"  {key}: {data['shared_neighbors']} shared, Jaccard={data['jaccard_similarity']:.4f}")
        else:
            print(f"  {key}: {data.get('reason', 'N/A')}")

    # Step 5: Convergence nodes
    print("\n[ANALYZE] Finding convergence nodes (3+ genes)...")
    convergence = find_convergence_nodes(gene_neighborhoods, min_genes=3)
    convergence = annotate_convergence_nodes(convergence, G)
    print(f"  Found {len(convergence)} convergence nodes")
    for node, info in list(convergence.items())[:10]:
        print(f"    {node} ({info['node_type']}): {info['gene_count']} genes - {info['connected_genes']}")

    # Step 6: Drug-gene-disease paths
    print("\n[ANALYZE] Finding drug-gene-disease paths...")
    drug_paths = find_drug_gene_disease_paths(G, gene_neighborhoods)
    print(f"  Found {len(drug_paths)} paths")

    # Step 7: SETBP1-IDH2 hypothesis
    print("\n[ANALYZE] Testing SETBP1-IDH2 mutual exclusivity hypothesis...")
    hypothesis = analyze_setbp1_idh2_redundancy(gene_neighborhoods, pairwise)
    print(f"  Supported: {hypothesis['hypothesis_supported']}")
    if hypothesis.get("interpretation"):
        print(f"  {hypothesis['interpretation']}")

    # Step 8: Shortest paths
    print("\n")
    shortest_paths = compute_shortest_paths(G_simple, gene_neighborhoods)

    # Step 9: Pathway convergence (meaningful types only, excluding anatomy)
    print("\n")
    pathway_convergence = compute_pathway_convergence(gene_neighborhoods, G)

    # Step 10: Redundancy matrix
    print("\n[ANALYZE] Computing redundancy matrix...")
    redundancy_matrix = compute_redundancy_matrix(pairwise)

    # Save JSON results
    print("\n[SAVE] Writing results...")

    # Prepare serializable results (trim large neighbor lists for JSON)
    results = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "data_source": "PrimeKG (Harvard Dataverse)",
            "primekg_url": PRIMEKG_URL,
            "reference": "Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable precision medicine. Scientific Data 10, 67 (2023)",
            "graph_nodes": graph_stats["nodes"],
            "graph_edges": graph_stats["edges"],
            "patient_genes": PATIENT_GENES,
        },
        "gene_neighborhoods": {
            gene: {
                "gene": info["gene"],
                "found_in_kg": info["found_in_kg"],
                "node_name": info.get("node_name"),
                "neighbor_count": info["neighbor_count"],
                "neighbors_by_type": {
                    k: {"count": len(v), "examples": v[:10]}
                    for k, v in info.get("neighbors_by_type", {}).items()
                },
            }
            for gene, info in gene_neighborhoods.items()
        },
        "shortest_paths": shortest_paths,
        "pairwise_overlap": {
            k: {
                key: val
                for key, val in v.items()
                if key != "shared_by_type"  # can be large
            }
            for k, v in pairwise.items()
        },
        "pairwise_shared_by_type": {
            k: v.get("shared_by_type", {})
            for k, v in pairwise.items()
            if v.get("computable")
        },
        "pathway_convergence": {
            k: {
                key: val
                for key, val in v.items()
                if key != "shared_by_type"  # store separately
            }
            for k, v in pathway_convergence.items()
        },
        "pathway_convergence_details": {
            k: v.get("shared_by_type", {})
            for k, v in pathway_convergence.items()
        },
        "convergence_nodes": {
            "count": len(convergence),
            "nodes": {
                k: v for k, v in list(convergence.items())[:100]
            },
        },
        "drug_gene_disease_paths": {
            "count": len(drug_paths),
            "unique_drugs": sorted(set(p["drug"] for p in drug_paths)),
            "paths_by_gene": {
                gene: [p for p in drug_paths if p["gene"] == gene][:20]
                for gene in PATIENT_GENES
            },
        },
        "setbp1_idh2_hypothesis": hypothesis,
        "redundancy_matrix": redundancy_matrix,
    }

    results_path = RESULTS_DIR / "primekg_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {results_path}")

    # Generate report
    report = generate_report(
        gene_neighborhoods, pairwise, convergence, drug_paths,
        hypothesis, redundancy_matrix, graph_stats,
        shortest_paths=shortest_paths,
        pathway_convergence=pathway_convergence,
    )
    report_path = RESULTS_DIR / "primekg_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    print("\n[DONE] PrimeKG analysis complete.")
    print(f"  Results: {results_path}")
    print(f"  Report:  {report_path}")


if __name__ == "__main__":
    main()
