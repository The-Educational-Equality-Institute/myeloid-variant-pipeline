#!/usr/bin/env python3
"""
genept_similarity.py -- Functional similarity analysis using text-derived gene embeddings.

Inspired by GenePT (Chen & Zou, Stanford 2024), which demonstrated that GPT-3.5
embeddings of gene function summaries capture biological relationships comparable
to single-cell expression profiles. This script uses the same principle with
sentence-transformers (all-MiniLM-L6-v2, already cached in the project) to compute
functional similarity between the patient's 5 mutated genes and all 34 target genes.

The key hypothesis under test:
    If mutually exclusive gene pairs (e.g., IDH2-SETBP1, O/E=0.905) have
    HIGHER functional similarity than co-occurring pairs, this supports
    pathway redundancy as the mechanism for mutual exclusivity.

Patient mutations:
    DNMT3A R882H (VAF 39%) - epigenetic regulator
    IDH2 R140Q (VAF 2%) - metabolic/epigenetic via 2-HG
    SETBP1 G870S (VAF 34%) - PP2A inhibitor / cell proliferation
    PTPN11 E76Q (VAF 29%) - RAS/MAPK signaling
    EZH2 V662A (VAF 59%) - PRC2 histone methylation

Inputs:
    - Gene functional descriptions (curated from UniProt/NCBI Gene/literature)
    - sentence-transformers all-MiniLM-L6-v2 model (384-dim, cached)

Outputs:
    - mutation_profile/results/ai_research/genept_similarity.json
    - mutation_profile/results/ai_research/genept_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/genept_similarity.py

Runtime: ~5 seconds (CPU)
Dependencies: sentence-transformers, numpy, scipy
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Gene functional descriptions
# ---------------------------------------------------------------------------
# Patient genes: detailed descriptions incorporating mutation-specific context
# Target genes: brief functional summaries from UniProt/NCBI Gene

PATIENT_GENES = {
    "DNMT3A": (
        "DNA methyltransferase 3A catalyzes de novo DNA methylation at CpG "
        "dinucleotides, essential for epigenetic gene regulation and "
        "hematopoietic stem cell differentiation. Methylates cytosine residues "
        "in genomic DNA to establish tissue-specific methylation patterns. "
        "Loss-of-function mutations cause DNA hypomethylation and aberrant "
        "gene activation in myeloid malignancies."
    ),
    "IDH2": (
        "Isocitrate dehydrogenase 2 catalyzes oxidative decarboxylation of "
        "isocitrate to alpha-ketoglutarate in the TCA cycle within "
        "mitochondria. R140Q neomorphic mutation produces the oncometabolite "
        "2-hydroxyglutarate which competitively inhibits TET2 dioxygenase and "
        "histone demethylases, causing epigenetic dysregulation through DNA "
        "and histone hypermethylation."
    ),
    "SETBP1": (
        "SET binding protein 1 stabilizes the SET/I2PP2A oncoprotein which "
        "inhibits the PP2A tumor suppressor phosphatase. G870S gain-of-function "
        "mutation in the SKI homology domain prevents ubiquitin-mediated "
        "proteasomal degradation by disrupting the beta-TrCP degron motif, "
        "leading to SETBP1 protein accumulation, PP2A inactivation, and "
        "enhanced cell proliferation signaling."
    ),
    "PTPN11": (
        "SHP2 protein tyrosine phosphatase encoded by PTPN11 activates "
        "RAS/MAPK and PI3K/AKT signaling downstream of receptor tyrosine "
        "kinases. E76Q gain-of-function mutation disrupts the autoinhibitory "
        "N-SH2/PTP domain interface, constitutively activating RAS/MAPK "
        "proliferation and survival signaling in hematopoietic cells."
    ),
    "EZH2": (
        "Enhancer of zeste homolog 2 is the catalytic subunit of the "
        "Polycomb Repressive Complex 2, responsible for trimethylation of "
        "histone H3 at lysine 27 for transcriptional gene silencing. "
        "V662A loss-of-function mutation in the SET domain reduces "
        "methyltransferase activity, causing derepression of PRC2 target "
        "genes involved in differentiation and tumor suppression."
    ),
}

TARGET_GENES = {
    "ASXL1": (
        "ASXL1 is a Polycomb group protein and chromatin regulator that "
        "interacts with the PRC2 complex. Truncating mutations in myeloid "
        "malignancies cause loss of H3K27 trimethylation and epigenetic "
        "dysregulation of hematopoietic gene expression."
    ),
    "TET2": (
        "TET2 is a dioxygenase that converts 5-methylcytosine to "
        "5-hydroxymethylcytosine in the DNA demethylation pathway. "
        "Loss-of-function mutations cause DNA hypermethylation and impaired "
        "hematopoietic stem cell differentiation in myeloid malignancies."
    ),
    "SRSF2": (
        "SRSF2 is an essential RNA splicing factor that recognizes exonic "
        "splicing enhancers. Hotspot mutations alter RNA binding specificity, "
        "causing widespread mis-splicing of myeloid differentiation genes."
    ),
    "SF3B1": (
        "SF3B1 is a core component of the U2 snRNP spliceosome complex. "
        "Hotspot mutations cause aberrant 3-prime splice site recognition "
        "and cryptic exon usage, characteristic of MDS with ring sideroblasts."
    ),
    "RUNX1": (
        "RUNX1 is a master transcription factor for definitive hematopoiesis "
        "that regulates myeloid differentiation gene expression. Mutations "
        "cause impaired differentiation and expansion of immature progenitors."
    ),
    "TP53": (
        "TP53 is a tumor suppressor transcription factor that responds to "
        "DNA damage by activating cell cycle arrest, apoptosis, and DNA "
        "repair pathways. Mutations cause genomic instability and therapy "
        "resistance in myeloid malignancies."
    ),
    "FLT3": (
        "FLT3 is a receptor tyrosine kinase expressed on hematopoietic "
        "progenitors. Internal tandem duplication and tyrosine kinase domain "
        "mutations constitutively activate STAT5 and RAS/MAPK signaling, "
        "driving proliferation in AML."
    ),
    "NPM1": (
        "NPM1 is a nucleolar phosphoprotein involved in ribosome biogenesis "
        "and centrosome duplication. Frameshift mutations cause cytoplasmic "
        "mislocalization, a defining feature of a distinct AML subtype with "
        "favorable prognosis."
    ),
    "NRAS": (
        "NRAS is a small GTPase that activates RAF/MEK/ERK and PI3K/AKT "
        "signaling cascades. Activating mutations at codons 12, 13, and 61 "
        "lock NRAS in the GTP-bound active state, driving proliferation "
        "in myeloid malignancies."
    ),
    "KRAS": (
        "KRAS is a small GTPase in the RAS/MAPK signaling pathway. "
        "Activating mutations constitutively signal through RAF/MEK/ERK "
        "and PI3K/AKT, promoting cell proliferation and survival."
    ),
    "CBL": (
        "CBL is an E3 ubiquitin ligase that negatively regulates receptor "
        "tyrosine kinase signaling by promoting ubiquitination and "
        "degradation. Loss-of-function mutations sustain cytokine receptor "
        "signaling in myeloproliferative neoplasms."
    ),
    "U2AF1": (
        "U2AF1 is an RNA splicing factor that recognizes the 3-prime splice "
        "site AG dinucleotide. Hotspot mutations alter splice site recognition "
        "specificity, causing mis-splicing of hematopoietic genes in MDS."
    ),
    "STAG2": (
        "STAG2 is a component of the cohesin complex required for sister "
        "chromatid cohesion and DNA repair. Loss-of-function mutations "
        "impair chromosomal segregation and transcriptional regulation "
        "in myeloid malignancies."
    ),
    "BCOR": (
        "BCOR is a transcriptional corepressor in the PRC1 Polycomb complex "
        "that represses target gene expression. Truncating mutations cause "
        "derepression of developmental genes in myeloid malignancies."
    ),
    "BCORL1": (
        "BCORL1 is a transcriptional corepressor related to BCOR that "
        "interacts with class II histone deacetylases. Loss-of-function "
        "mutations deregulate gene expression in myeloid neoplasms."
    ),
    "DDX41": (
        "DDX41 is a DEAD-box RNA helicase involved in RNA splicing and "
        "innate immune sensing of cytosolic nucleic acids. Germline and "
        "somatic mutations predispose to familial MDS/AML."
    ),
    "IDH1": (
        "IDH1 catalyzes oxidative decarboxylation of isocitrate to "
        "alpha-ketoglutarate in the cytoplasm. R132H neomorphic mutation "
        "produces 2-hydroxyglutarate oncometabolite, inhibiting TET2 and "
        "histone demethylases to cause epigenetic dysregulation."
    ),
    "JAK2": (
        "JAK2 is a non-receptor tyrosine kinase in the JAK-STAT signaling "
        "pathway downstream of cytokine receptors. V617F gain-of-function "
        "mutation constitutively activates STAT3/5 signaling, characteristic "
        "of myeloproliferative neoplasms."
    ),
    "CALR": (
        "Calreticulin is an endoplasmic reticulum chaperone. Frameshift "
        "mutations in exon 9 create a novel C-terminus that activates "
        "MPL/JAK-STAT signaling, driving megakaryocyte proliferation in "
        "essential thrombocythemia and primary myelofibrosis."
    ),
    "MPL": (
        "MPL is the thrombopoietin receptor that activates JAK2-STAT "
        "signaling for megakaryocyte development. Gain-of-function mutations "
        "at W515 constitutively activate signaling in myeloproliferative "
        "neoplasms."
    ),
    "PHF6": (
        "PHF6 is a chromatin reader protein with two PHD zinc finger domains "
        "involved in transcriptional regulation. Loss-of-function mutations "
        "are enriched in male patients with myeloid malignancies."
    ),
    "WT1": (
        "WT1 is a zinc finger transcription factor that regulates cell "
        "growth and differentiation in hematopoiesis. Mutations impair "
        "myeloid differentiation and are associated with adverse prognosis "
        "in AML."
    ),
    "CEBPA": (
        "CEBPA is a leucine zipper transcription factor essential for "
        "granulocytic differentiation. Biallelic mutations define a "
        "favorable-risk AML subtype. Single mutations impair myeloid "
        "differentiation."
    ),
    "GATA2": (
        "GATA2 is a zinc finger transcription factor essential for "
        "hematopoietic stem cell maintenance and lymphoid development. "
        "Germline mutations predispose to familial MDS/AML."
    ),
    "ZRSR2": (
        "ZRSR2 is a splicing factor involved in recognition of the minor "
        "U12-type intron splice sites. Loss-of-function mutations cause "
        "retention of U12 introns in myeloid malignancies."
    ),
    "RAD21": (
        "RAD21 is a core subunit of the cohesin ring complex. Mutations "
        "impair sister chromatid cohesion and disrupt the cohesin-mediated "
        "transcriptional regulation important for hematopoietic differentiation."
    ),
    "SMC1A": (
        "SMC1A is a structural maintenance of chromosomes protein and core "
        "cohesin subunit. Mutations disrupt cohesin complex function affecting "
        "chromosomal segregation and gene regulation in myeloid malignancies."
    ),
    "SMC3": (
        "SMC3 is a core structural component of the cohesin ring complex "
        "required for sister chromatid cohesion and DNA damage repair. "
        "Mutations impair cohesin function in hematopoietic cells."
    ),
    "CSF3R": (
        "CSF3R is the granulocyte colony-stimulating factor receptor that "
        "activates JAK-STAT and SRC kinase signaling for granulocyte "
        "development. Activating mutations drive chronic neutrophilic "
        "leukemia and atypical CML."
    ),
}

# Co-occurrence statistics from GENIE v19.0 pairwise analysis
COOCCURRENCE_DATA = {
    ("DNMT3A", "IDH2"): {"oe_ratio": 2.741, "p_value": 2.46e-87, "direction": "co-occurrence"},
    ("DNMT3A", "PTPN11"): {"oe_ratio": 1.953, "p_value": 1.60e-15, "direction": "co-occurrence"},
    ("DNMT3A", "SETBP1"): {"oe_ratio": 1.060, "p_value": 0.618, "direction": "neutral"},
    ("IDH2", "PTPN11"): {"oe_ratio": 1.439, "p_value": 0.020, "direction": "co-occurrence"},
    ("IDH2", "SETBP1"): {"oe_ratio": 0.905, "p_value": 0.740, "direction": "mutual_exclusivity"},
    ("PTPN11", "SETBP1"): {"oe_ratio": 3.623, "p_value": 1.14e-12, "direction": "co-occurrence"},
}

# Functional pathway annotations for grouping analysis
PATHWAY_ANNOTATIONS = {
    "DNMT3A": ["epigenetic", "DNA_methylation"],
    "IDH2": ["metabolism", "epigenetic", "TCA_cycle"],
    "SETBP1": ["proliferation", "PP2A_inhibition"],
    "PTPN11": ["signaling", "RAS_MAPK"],
    "EZH2": ["epigenetic", "histone_methylation", "PRC2"],
    "ASXL1": ["epigenetic", "PRC2"],
    "TET2": ["epigenetic", "DNA_demethylation"],
    "IDH1": ["metabolism", "epigenetic", "TCA_cycle"],
    "SRSF2": ["RNA_splicing"],
    "SF3B1": ["RNA_splicing"],
    "U2AF1": ["RNA_splicing"],
    "ZRSR2": ["RNA_splicing"],
    "RUNX1": ["transcription_factor", "differentiation"],
    "TP53": ["tumor_suppressor", "DNA_damage"],
    "FLT3": ["signaling", "RAS_MAPK", "receptor_tyrosine_kinase"],
    "NPM1": ["nucleolar", "ribosome"],
    "NRAS": ["signaling", "RAS_MAPK"],
    "KRAS": ["signaling", "RAS_MAPK"],
    "CBL": ["signaling", "ubiquitin"],
    "JAK2": ["signaling", "JAK_STAT"],
    "CALR": ["signaling", "JAK_STAT"],
    "MPL": ["signaling", "JAK_STAT"],
    "STAG2": ["cohesin", "chromosomal"],
    "RAD21": ["cohesin", "chromosomal"],
    "SMC1A": ["cohesin", "chromosomal"],
    "SMC3": ["cohesin", "chromosomal"],
    "BCOR": ["epigenetic", "PRC1"],
    "BCORL1": ["epigenetic", "transcriptional_repression"],
    "DDX41": ["RNA_helicase", "innate_immunity"],
    "PHF6": ["epigenetic", "chromatin_reader"],
    "WT1": ["transcription_factor", "differentiation"],
    "CEBPA": ["transcription_factor", "differentiation"],
    "GATA2": ["transcription_factor", "stem_cell"],
    "CSF3R": ["signaling", "JAK_STAT"],
}


def load_model() -> SentenceTransformer:
    """Load sentence-transformers model (cached)."""
    log.info("Loading %s model...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    log.info("Model loaded: %d-dim embeddings", model.get_sentence_embedding_dimension())
    return model


def compute_embeddings(
    model: SentenceTransformer,
    gene_descriptions: dict[str, str],
) -> dict[str, np.ndarray]:
    """Compute normalized embeddings for all gene descriptions."""
    genes = list(gene_descriptions.keys())
    texts = [gene_descriptions[g] for g in genes]

    log.info("Encoding %d gene descriptions...", len(genes))
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    return {gene: emb for gene, emb in zip(genes, embeddings)}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    return float(1.0 - cosine_distance(a, b))


def compute_pairwise_matrix(
    embeddings: dict[str, np.ndarray],
    genes: list[str],
) -> dict[str, dict[str, float]]:
    """Compute pairwise cosine similarity matrix for given genes."""
    matrix: dict[str, dict[str, float]] = {}
    for g1 in genes:
        matrix[g1] = {}
        for g2 in genes:
            matrix[g1][g2] = cosine_similarity(embeddings[g1], embeddings[g2])
    return matrix


def classify_pair_relationship(
    sim: float,
    all_sims: list[float],
) -> str:
    """Classify similarity as high/medium/low relative to distribution."""
    p75 = float(np.percentile(all_sims, 75))
    p25 = float(np.percentile(all_sims, 25))
    if sim >= p75:
        return "HIGH"
    elif sim >= p25:
        return "MEDIUM"
    else:
        return "LOW"


def compute_pathway_overlap(gene_a: str, gene_b: str) -> dict:
    """Compute pathway overlap between two genes."""
    paths_a = set(PATHWAY_ANNOTATIONS.get(gene_a, []))
    paths_b = set(PATHWAY_ANNOTATIONS.get(gene_b, []))
    shared = paths_a & paths_b
    union = paths_a | paths_b
    jaccard = len(shared) / len(union) if union else 0.0
    return {
        "shared_pathways": sorted(shared),
        "jaccard_index": round(jaccard, 4),
        "gene_a_pathways": sorted(paths_a),
        "gene_b_pathways": sorted(paths_b),
    }


def find_nearest_neighbors(
    embeddings: dict[str, np.ndarray],
    query_gene: str,
    top_k: int = 10,
) -> list[dict]:
    """Find top-k most similar genes to query gene."""
    query_emb = embeddings[query_gene]
    sims = []
    for gene, emb in embeddings.items():
        if gene == query_gene:
            continue
        sim = cosine_similarity(query_emb, emb)
        sims.append({"gene": gene, "similarity": round(sim, 4)})
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    return sims[:top_k]


def analyze_hypothesis(
    patient_matrix: dict[str, dict[str, float]],
    all_pair_sims: list[float],
) -> dict:
    """
    Test: do mutually exclusive pairs have higher functional similarity
    than co-occurring pairs (supporting pathway redundancy)?
    """
    patient_genes = list(PATIENT_GENES.keys())
    pair_analysis = []

    for g1, g2 in combinations(patient_genes, 2):
        sim = patient_matrix[g1][g2]
        key = tuple(sorted([g1, g2]))
        cooc = COOCCURRENCE_DATA.get(key, COOCCURRENCE_DATA.get((key[1], key[0]), {}))
        pathway = compute_pathway_overlap(g1, g2)
        classification = classify_pair_relationship(sim, all_pair_sims)

        pair_analysis.append({
            "gene_a": g1,
            "gene_b": g2,
            "cosine_similarity": round(sim, 4),
            "similarity_class": classification,
            "percentile": round(
                float(np.mean([s <= sim for s in all_pair_sims]) * 100), 1
            ),
            "cooccurrence_oe": cooc.get("oe_ratio"),
            "cooccurrence_p": cooc.get("p_value"),
            "cooccurrence_direction": cooc.get("direction", "unknown"),
            "pathway_overlap": pathway,
        })

    pair_analysis.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    # Separate by co-occurrence direction
    cooccurring = [p for p in pair_analysis if p["cooccurrence_direction"] == "co-occurrence"]
    exclusive = [p for p in pair_analysis if p["cooccurrence_direction"] == "mutual_exclusivity"]
    neutral = [p for p in pair_analysis if p["cooccurrence_direction"] == "neutral"]

    mean_cooc = np.mean([p["cosine_similarity"] for p in cooccurring]) if cooccurring else None
    mean_excl = np.mean([p["cosine_similarity"] for p in exclusive]) if exclusive else None
    mean_neutral = np.mean([p["cosine_similarity"] for p in neutral]) if neutral else None

    return {
        "pair_analysis": pair_analysis,
        "summary": {
            "mean_similarity_cooccurring": round(float(mean_cooc), 4) if mean_cooc is not None else None,
            "mean_similarity_exclusive": round(float(mean_excl), 4) if mean_excl is not None else None,
            "mean_similarity_neutral": round(float(mean_neutral), 4) if mean_neutral is not None else None,
            "n_cooccurring": len(cooccurring),
            "n_exclusive": len(exclusive),
            "n_neutral": len(neutral),
            "hypothesis_supported": (
                mean_excl is not None
                and mean_cooc is not None
                and float(mean_excl) > float(mean_cooc)
            ),
        },
    }


def compute_functional_clusters(
    embeddings: dict[str, np.ndarray],
    patient_genes: list[str],
) -> dict:
    """Identify functional clusters among patient genes based on embeddings."""
    epigenetic_genes = [g for g, p in PATHWAY_ANNOTATIONS.items()
                        if "epigenetic" in p and g in embeddings]
    signaling_genes = [g for g, p in PATHWAY_ANNOTATIONS.items()
                       if "signaling" in p and g in embeddings]
    splicing_genes = [g for g, p in PATHWAY_ANNOTATIONS.items()
                      if "RNA_splicing" in p and g in embeddings]

    clusters = {}
    for gene in patient_genes:
        emb = embeddings[gene]
        epi_sims = [cosine_similarity(emb, embeddings[g]) for g in epigenetic_genes if g != gene]
        sig_sims = [cosine_similarity(emb, embeddings[g]) for g in signaling_genes if g != gene]
        spl_sims = [cosine_similarity(emb, embeddings[g]) for g in splicing_genes if g != gene]

        clusters[gene] = {
            "mean_epigenetic_similarity": round(float(np.mean(epi_sims)), 4) if epi_sims else None,
            "mean_signaling_similarity": round(float(np.mean(sig_sims)), 4) if sig_sims else None,
            "mean_splicing_similarity": round(float(np.mean(spl_sims)), 4) if spl_sims else None,
            "primary_cluster": max(
                [
                    ("epigenetic", float(np.mean(epi_sims)) if epi_sims else 0),
                    ("signaling", float(np.mean(sig_sims)) if sig_sims else 0),
                    ("splicing", float(np.mean(spl_sims)) if spl_sims else 0),
                ],
                key=lambda x: x[1],
            )[0],
        }

    return clusters


def generate_report(results: dict) -> str:
    """Generate markdown report from analysis results."""
    lines = [
        "# Functional Similarity Analysis: Text-Derived Gene Embeddings",
        "",
        f"**Generated:** {results['metadata']['timestamp']}",
        f"**Model:** {results['metadata']['model']}",
        f"**Embedding dim:** {results['metadata']['embedding_dim']}",
        f"**Genes analyzed:** {results['metadata']['n_patient_genes']} patient + "
        f"{results['metadata']['n_target_genes']} target = {results['metadata']['n_total_genes']} total",
        "",
        "## Method",
        "",
        "Inspired by GenePT (Chen & Zou, Stanford 2024), this analysis uses sentence-transformer",
        "embeddings of curated gene functional descriptions to compute semantic similarity between",
        "genes. The principle: genes with similar biological functions produce similar text embeddings.",
        "This captures pathway relationships without requiring expression data.",
        "",
        "Each gene was described using a 2-4 sentence functional summary incorporating:",
        "- Protein function and molecular activity",
        "- Pathway membership and interactions",
        "- Mutation-specific consequences in myeloid malignancies",
        "",
        "---",
        "",
        "## Patient Gene Similarity Matrix (5x5)",
        "",
    ]

    # Matrix table
    patient_genes = list(PATIENT_GENES.keys())
    matrix = results["patient_matrix"]

    header = "| | " + " | ".join(patient_genes) + " |"
    sep = "|---|" + "|".join(["---"] * len(patient_genes)) + "|"
    lines.append(header)
    lines.append(sep)
    for g1 in patient_genes:
        row_vals = []
        for g2 in patient_genes:
            val = matrix[g1][g2]
            if g1 == g2:
                row_vals.append("**1.000**")
            elif val >= 0.6:
                row_vals.append(f"**{val:.3f}**")
            else:
                row_vals.append(f"{val:.3f}")
        lines.append(f"| **{g1}** | " + " | ".join(row_vals) + " |")

    lines.extend(["", "---", "", "## Hypothesis Test: Pathway Redundancy and Mutual Exclusivity", ""])

    hyp = results["hypothesis_test"]
    summary = hyp["summary"]

    lines.append("**Hypothesis:** Mutually exclusive gene pairs have higher functional similarity")
    lines.append("than co-occurring pairs, supporting pathway redundancy as the mechanism.")
    lines.append("")

    # Pair details table
    lines.append("| Pair | Cosine Sim | Percentile | O/E Ratio | Direction | Shared Pathways |")
    lines.append("|------|-----------|------------|-----------|-----------|-----------------|")
    for p in hyp["pair_analysis"]:
        oe = f"{p['cooccurrence_oe']:.3f}" if p["cooccurrence_oe"] else "N/A"
        pval_str = ""
        if p["cooccurrence_p"] is not None:
            if p["cooccurrence_p"] < 0.001:
                pval_str = f" (p={p['cooccurrence_p']:.1e})"
            elif p["cooccurrence_p"] < 0.05:
                pval_str = f" (p={p['cooccurrence_p']:.3f})"
            else:
                pval_str = f" (p={p['cooccurrence_p']:.2f})"
        shared = ", ".join(p["pathway_overlap"]["shared_pathways"]) or "none"
        lines.append(
            f"| {p['gene_a']}-{p['gene_b']} | {p['cosine_similarity']:.4f} | "
            f"{p['percentile']}th | {oe}{pval_str} | {p['cooccurrence_direction']} | "
            f"{shared} |"
        )

    lines.extend(["", "### Summary Statistics", ""])
    if summary["mean_similarity_cooccurring"] is not None:
        lines.append(f"- **Co-occurring pairs** (n={summary['n_cooccurring']}): "
                      f"mean similarity = {summary['mean_similarity_cooccurring']:.4f}")
    if summary["mean_similarity_exclusive"] is not None:
        lines.append(f"- **Mutually exclusive pairs** (n={summary['n_exclusive']}): "
                      f"mean similarity = {summary['mean_similarity_exclusive']:.4f}")
    if summary["mean_similarity_neutral"] is not None:
        lines.append(f"- **Neutral pairs** (n={summary['n_neutral']}): "
                      f"mean similarity = {summary['mean_similarity_neutral']:.4f}")

    lines.append("")
    if summary["hypothesis_supported"]:
        diff = summary["mean_similarity_exclusive"] - summary["mean_similarity_cooccurring"]
        lines.append(
            f"**Result: SUPPORTED.** Mutually exclusive pairs have {diff:.4f} higher "
            f"mean similarity than co-occurring pairs, consistent with pathway redundancy."
        )
    else:
        lines.append(
            "**Result: NOT SUPPORTED.** Co-occurring pairs have equal or higher "
            "functional similarity than mutually exclusive pairs. Mutual exclusivity "
            "may be driven by mechanisms other than simple pathway redundancy "
            "(e.g., synthetic lethality, clonal competition)."
        )

    # IDH2-SETBP1 specific analysis
    lines.extend(["", "### IDH2-SETBP1 Mutual Exclusivity", ""])
    idh2_setbp1 = next(
        (p for p in hyp["pair_analysis"]
         if {p["gene_a"], p["gene_b"]} == {"IDH2", "SETBP1"}),
        None,
    )
    if idh2_setbp1:
        lines.append(
            f"IDH2-SETBP1 cosine similarity: **{idh2_setbp1['cosine_similarity']:.4f}** "
            f"({idh2_setbp1['percentile']}th percentile)"
        )
        lines.append(f"- O/E ratio: {idh2_setbp1['cooccurrence_oe']:.3f} (below 1.0 = trend toward exclusivity)")
        lines.append(f"- Shared pathways: {', '.join(idh2_setbp1['pathway_overlap']['shared_pathways']) or 'none'}")
        lines.append(f"- Pathway Jaccard index: {idh2_setbp1['pathway_overlap']['jaccard_index']:.3f}")
        lines.append("")
        if idh2_setbp1["pathway_overlap"]["jaccard_index"] == 0:
            lines.append(
                "IDH2 (TCA cycle/epigenetic) and SETBP1 (PP2A/proliferation) operate through "
                "distinct molecular pathways. Their mutual exclusivity is unlikely driven by "
                "pathway redundancy and more likely reflects clonal competition or synthetic "
                "lethality when co-mutated."
            )
        else:
            lines.append(
                "IDH2 and SETBP1 share pathway annotations, suggesting partial functional "
                "overlap that could contribute to their mutual exclusivity through redundancy."
            )

    # Functional clusters
    lines.extend(["", "---", "", "## Functional Cluster Assignment", ""])
    lines.append("Each patient gene's average similarity to known pathway groups:")
    lines.append("")
    lines.append("| Gene | Epigenetic | Signaling | Splicing | Primary Cluster |")
    lines.append("|------|-----------|-----------|---------|-----------------|")
    clusters = results["functional_clusters"]
    for gene in PATIENT_GENES:
        c = clusters[gene]
        epi = f"{c['mean_epigenetic_similarity']:.3f}" if c["mean_epigenetic_similarity"] else "N/A"
        sig = f"{c['mean_signaling_similarity']:.3f}" if c["mean_signaling_similarity"] else "N/A"
        spl = f"{c['mean_splicing_similarity']:.3f}" if c["mean_splicing_similarity"] else "N/A"
        lines.append(f"| **{gene}** | {epi} | {sig} | {spl} | {c['primary_cluster']} |")

    # Nearest neighbors
    lines.extend(["", "---", "", "## Nearest Neighbors (Top 5 per Patient Gene)", ""])
    for gene in PATIENT_GENES:
        nn = results["nearest_neighbors"][gene][:5]
        lines.append(f"### {gene}")
        lines.append("| Rank | Gene | Similarity |")
        lines.append("|------|------|-----------|")
        for i, n in enumerate(nn, 1):
            lines.append(f"| {i} | {n['gene']} | {n['similarity']:.4f} |")
        lines.append("")

    # Key findings
    lines.extend(["---", "", "## Key Findings", ""])

    pairs_sorted = sorted(hyp["pair_analysis"], key=lambda x: x["cosine_similarity"], reverse=True)
    highest = pairs_sorted[0]
    lowest = pairs_sorted[-1]

    lines.append(f"1. **Highest patient-gene similarity:** {highest['gene_a']}-{highest['gene_b']} "
                 f"({highest['cosine_similarity']:.4f})")
    lines.append(f"2. **Lowest patient-gene similarity:** {lowest['gene_a']}-{lowest['gene_b']} "
                 f"({lowest['cosine_similarity']:.4f})")

    # Check if epigenetic genes cluster
    epi_genes = [g for g in PATIENT_GENES if "epigenetic" in PATHWAY_ANNOTATIONS.get(g, [])]
    if len(epi_genes) >= 2:
        epi_sims = []
        for g1, g2 in combinations(epi_genes, 2):
            epi_sims.append(matrix[g1][g2])
        non_epi_sims = []
        for g1, g2 in combinations(PATIENT_GENES, 2):
            if g1 not in epi_genes or g2 not in epi_genes:
                non_epi_sims.append(matrix[g1][g2])
        mean_epi = np.mean(epi_sims) if epi_sims else 0
        mean_non = np.mean(non_epi_sims) if non_epi_sims else 0
        lines.append(f"3. **Epigenetic cluster:** {', '.join(epi_genes)} mean pairwise similarity = "
                     f"{mean_epi:.4f} vs non-epigenetic pairs = {mean_non:.4f}")

    lines.append(f"4. **Hypothesis (pathway redundancy drives exclusivity):** "
                 f"{'SUPPORTED' if summary['hypothesis_supported'] else 'NOT SUPPORTED'}")

    lines.extend([
        "",
        "---",
        "",
        "## Limitations",
        "",
        "1. Text embeddings capture semantic similarity of descriptions, not direct molecular "
        "interaction data. Description authoring influences results.",
        "2. all-MiniLM-L6-v2 (384-dim) is a general-purpose model, not trained on biomedical "
        "text. Domain-specific models (BioSentVec, PubMedBERT) may yield different rankings.",
        "3. GenePT used GPT-3.5 embeddings of NCBI Gene summaries; this implementation uses "
        "curated descriptions with mutation-specific context, which may bias toward the "
        "patient's specific variants.",
        "4. Functional similarity is one axis; mutual exclusivity can also arise from synthetic "
        "lethality, clonal dynamics, or sampling bias.",
        "5. n=1 mutually exclusive pair (IDH2-SETBP1) limits statistical power for the "
        "hypothesis test.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the full functional similarity analysis."""
    start_time = time.time()
    log.info("Starting functional similarity analysis")

    # Merge all gene descriptions
    all_descriptions = {**PATIENT_GENES, **TARGET_GENES}
    patient_genes = list(PATIENT_GENES.keys())

    # Load model and compute embeddings
    model = load_model()
    embeddings = compute_embeddings(model, all_descriptions)
    embedding_dim = model.get_sentence_embedding_dimension()

    # Compute patient gene matrix
    log.info("Computing 5x5 patient gene similarity matrix...")
    patient_matrix = compute_pairwise_matrix(embeddings, patient_genes)

    # Compute all pairwise similarities for percentile context
    all_genes = list(all_descriptions.keys())
    all_pair_sims = []
    for g1, g2 in combinations(all_genes, 2):
        all_pair_sims.append(cosine_similarity(embeddings[g1], embeddings[g2]))

    # Hypothesis test
    log.info("Testing pathway redundancy hypothesis...")
    hypothesis_results = analyze_hypothesis(patient_matrix, all_pair_sims)

    # Functional clusters
    log.info("Computing functional cluster assignments...")
    clusters = compute_functional_clusters(embeddings, patient_genes)

    # Nearest neighbors
    log.info("Finding nearest neighbors for patient genes...")
    nearest = {}
    for gene in patient_genes:
        nearest[gene] = find_nearest_neighbors(embeddings, gene, top_k=10)

    # Full matrix stats
    full_matrix = compute_pairwise_matrix(embeddings, all_genes)
    all_pair_values = [
        full_matrix[g1][g2]
        for g1, g2 in combinations(all_genes, 2)
    ]

    elapsed = time.time() - start_time

    # Assemble results
    results = {
        "metadata": {
            "analysis": "Functional similarity via text-derived gene embeddings",
            "method": "GenePT-inspired: sentence-transformer embeddings of gene functional descriptions",
            "model": MODEL_NAME,
            "embedding_dim": embedding_dim,
            "n_patient_genes": len(patient_genes),
            "n_target_genes": len(TARGET_GENES),
            "n_total_genes": len(all_descriptions),
            "n_total_pairs": len(all_pair_sims),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": round(elapsed, 2),
            "reference": "Chen H, Zou J. GenePT: A Simple But Effective Foundation Model for Genes and Cells Built on ChatGPT. bioRxiv 2024.",
        },
        "patient_matrix": {
            g1: {g2: round(v, 4) for g2, v in row.items()}
            for g1, row in patient_matrix.items()
        },
        "hypothesis_test": hypothesis_results,
        "functional_clusters": clusters,
        "nearest_neighbors": nearest,
        "distribution_stats": {
            "all_pairs_mean": round(float(np.mean(all_pair_values)), 4),
            "all_pairs_std": round(float(np.std(all_pair_values)), 4),
            "all_pairs_median": round(float(np.median(all_pair_values)), 4),
            "all_pairs_min": round(float(np.min(all_pair_values)), 4),
            "all_pairs_max": round(float(np.max(all_pair_values)), 4),
            "all_pairs_p25": round(float(np.percentile(all_pair_values, 25)), 4),
            "all_pairs_p75": round(float(np.percentile(all_pair_values, 75)), 4),
        },
    }

    # Save JSON
    json_path = RESULTS_DIR / "genept_similarity.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved JSON results: %s", json_path)

    # Generate and save report
    report = generate_report(results)
    report_path = RESULTS_DIR / "genept_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved report: %s", report_path)

    # Print summary
    print("\n" + "=" * 70)
    print("FUNCTIONAL SIMILARITY ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Runtime: {elapsed:.1f}s")
    print(f"Model: {MODEL_NAME} ({embedding_dim}-dim)")
    print(f"Genes: {len(patient_genes)} patient + {len(TARGET_GENES)} target")
    print(f"\nPatient Gene Similarity Matrix:")
    print(f"{'':>10s}", end="")
    for g in patient_genes:
        print(f"{g:>10s}", end="")
    print()
    for g1 in patient_genes:
        print(f"{g1:>10s}", end="")
        for g2 in patient_genes:
            print(f"{patient_matrix[g1][g2]:10.4f}", end="")
        print()

    print(f"\nHypothesis (pathway redundancy): "
          f"{'SUPPORTED' if results['hypothesis_test']['summary']['hypothesis_supported'] else 'NOT SUPPORTED'}")
    s = results["hypothesis_test"]["summary"]
    if s["mean_similarity_cooccurring"] is not None:
        print(f"  Co-occurring mean:       {s['mean_similarity_cooccurring']:.4f}")
    if s["mean_similarity_exclusive"] is not None:
        print(f"  Mutually exclusive mean: {s['mean_similarity_exclusive']:.4f}")
    if s["mean_similarity_neutral"] is not None:
        print(f"  Neutral mean:            {s['mean_similarity_neutral']:.4f}")

    print(f"\nOutputs:")
    print(f"  {json_path}")
    print(f"  {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
