#!/usr/bin/env python3
"""
pathway_analysis.py -- Pathway and signaling network analysis for 4-gene mutation profile.

Patient mutation profile:
  DNMT3A R882H (VAF 39%) - DNA methylation / epigenetic regulation
  IDH2 R140Q   (VAF 2%)  - TCA cycle / 2-HG / epigenetic hypermethylation
  SETBP1 G870S (VAF 34%) - SET/PP2A / chromatin remodeling / self-renewal
  PTPN11 E76Q  (VAF 29%) - RAS/MAPK signaling (SHP2 upstream activator)
  Monosomy 7              - Haploinsufficiency of chr7 tumor suppressors

Analyses:
  1. Pathway mapping per mutation
  2. Convergence analysis (epigenetic, signaling, chr7 interactions)
  3. Network graph (JSON + Plotly visualization)
  4. Therapeutic vulnerability ranking
  5. Clonal evolution model from VAF data

Inputs:
    None (uses hardcoded pathway knowledge).

Outputs:
    - mutation_profile/results/ai_research/pathway_analysis/pathway_map.json
    - mutation_profile/results/ai_research/pathway_analysis/convergence_analysis.json
    - mutation_profile/results/ai_research/pathway_analysis/network_graph.html
    - mutation_profile/results/ai_research/pathway_analysis/therapeutic_vulnerabilities.json
    - mutation_profile/results/ai_research/pathway_analysis/clonal_evolution.json
    - mutation_profile/results/ai_research/pathway_analysis/pathway_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/pathway_analysis.py

Runtime: ~5 seconds
Dependencies: networkx, plotly
"""

import os
import json
import math
from datetime import datetime

import plotly.graph_objects as go
import networkx as nx

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
RESULTS_DIR = os.path.join(
    PROJECT_DIR, "mutation_profile", "results", "ai_research", "pathway_analysis"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Patient mutation profile
# ---------------------------------------------------------------------------
MUTATIONS = {
    "DNMT3A_R882H": {
        "gene": "DNMT3A",
        "variant": "R882H",
        "vaf": 0.39,
        "type": "loss-of-function (dominant-negative)",
        "mechanism": (
            "DNMT3A R882H is a dominant-negative mutation in the methyltransferase "
            "domain that reduces de novo DNA methylation activity by ~80%. The mutant "
            "subunit poisons the DNMT3A tetramer, leading to focal hypomethylation at "
            "CpG islands of stem cell and self-renewal genes (MEIS1, HOXA cluster, "
            "MYC enhancers). This unlocks a self-renewal transcriptional program "
            "normally silenced during differentiation."
        ),
    },
    "IDH2_R140Q": {
        "gene": "IDH2",
        "variant": "R140Q",
        "vaf": 0.02,
        "type": "gain-of-function (neomorphic)",
        "mechanism": (
            "IDH2 R140Q is a neomorphic mutation that converts alpha-ketoglutarate "
            "(aKG) to the oncometabolite 2-hydroxyglutarate (2-HG). 2-HG competitively "
            "inhibits TET2 and Jumonji-domain histone demethylases (KDM4A/B), causing "
            "DNA and histone hypermethylation. This blocks myeloid differentiation and "
            "expands the progenitor compartment. Paradoxically, while DNMT3A R882H "
            "causes focal hypomethylation, IDH2 R140Q causes global hypermethylation "
            "-- the two create a complex epigenetic dysregulation."
        ),
    },
    "SETBP1_G870S": {
        "gene": "SETBP1",
        "variant": "G870S",
        "vaf": 0.34,
        "type": "gain-of-function (stabilizing)",
        "mechanism": (
            "SETBP1 G870S is a hotspot mutation in the SKI-homology domain / "
            "degron (residues 858-871) that disrupts beta-TrCP-mediated ubiquitination "
            "and proteasomal degradation. The stabilized SETBP1 protein accumulates and "
            "binds SET (I2PP2A), forming the SETBP1-SET-PP2A complex. SET inhibits "
            "PP2A, a major tumor suppressor phosphatase. Loss of PP2A activity "
            "de-represses multiple oncogenic pathways: AKT, MYC, ERK, and WNT/beta-catenin. "
            "SETBP1 also functions as a transcription factor activating HOXA9/HOXA10 "
            "and repressing RUNX1, directly promoting self-renewal."
        ),
    },
    "PTPN11_E76Q": {
        "gene": "PTPN11",
        "variant": "E76Q",
        "vaf": 0.29,
        "type": "gain-of-function (constitutive activation)",
        "mechanism": (
            "PTPN11 E76Q is a somatic gain-of-function mutation in SHP2. E76 is in "
            "the N-SH2 domain at the autoinhibitory interface; the E76Q substitution "
            "destabilizes the closed/autoinhibited conformation, resulting in constitutive "
            "phosphatase activity. Active SHP2 dephosphorylates GAB1/GAB2 and RAS-GAP "
            "binding sites, releasing RAS from negative regulation. This leads to "
            "constitutive RAS-MAPK (RAF-MEK-ERK) and PI3K-AKT activation. SHP2 also "
            "signals through JAK-STAT and SRC family kinases."
        ),
    },
    "Monosomy7": {
        "gene": "Chromosome 7",
        "variant": "monosomy (-7)",
        "vaf": None,
        "type": "haploinsufficiency / loss of heterozygosity",
        "mechanism": (
            "Monosomy 7 results in haploinsufficiency of multiple tumor suppressor "
            "genes on chromosome 7. Key genes affected: EZH2 (7q36.1, PRC2 "
            "histone methyltransferase -- loss compounds epigenetic dysregulation), "
            "CUX1 (7q22.1, transcription factor controlling cell cycle and DNA repair), "
            "MLL3/KMT2C (7q36.1, histone methyltransferase), SAMD9/SAMD9L (7q21.2, "
            "innate immune regulators), LUC7L2 (7q34, splicing factor). Loss of one "
            "copy reduces expression by ~50%, insufficient for normal tumor suppression. "
            "Monosomy 7 in MDS/AML carries adverse prognosis and often co-occurs with "
            "RAS pathway mutations (PTPN11, NRAS, KRAS)."
        ),
    },
}

# ---------------------------------------------------------------------------
# 1. Pathway mapping — structured data
# ---------------------------------------------------------------------------

def build_pathway_mapping():
    """Build detailed pathway mapping for each mutation."""
    pathways = {
        "DNMT3A_R882H": {
            "primary_pathway": "DNA methylation / Epigenetic regulation",
            "affected_processes": [
                {
                    "process": "De novo DNA methylation",
                    "effect": "~80% reduction in methyltransferase activity",
                    "consequence": "Focal CpG island hypomethylation",
                },
                {
                    "process": "Hematopoietic stem cell (HSC) self-renewal",
                    "effect": "De-repression of MEIS1, HOXA9, HOXA10, MYC",
                    "consequence": "Expanded HSC pool with impaired differentiation",
                },
                {
                    "process": "Clonal hematopoiesis",
                    "effect": "HSC competitive advantage",
                    "consequence": "DNMT3A R882H is the most common CHIP mutation; "
                                   "confers ~10-fold expansion advantage",
                },
                {
                    "process": "Chromatin accessibility",
                    "effect": "Hypomethylated enhancers become accessible",
                    "consequence": "Altered transcription factor binding landscapes",
                },
            ],
            "downstream_targets": [
                "MEIS1", "HOXA9", "HOXA10", "MYC", "MYCN",
                "PBX3", "FLT3", "CDK6",
            ],
            "interacting_pathways": [
                "TET2-mediated demethylation",
                "PRC2 (EZH2/SUZ12) histone methylation",
                "WNT/beta-catenin (via methylation of WNT antagonists)",
            ],
        },
        "IDH2_R140Q": {
            "primary_pathway": "TCA cycle / 2-HG oncometabolite production",
            "affected_processes": [
                {
                    "process": "TCA cycle (isocitrate -> alpha-ketoglutarate)",
                    "effect": "Neomorphic: aKG converted to 2-HG instead",
                    "consequence": "2-HG accumulates to millimolar concentrations",
                },
                {
                    "process": "TET2-mediated DNA demethylation",
                    "effect": "2-HG competitively inhibits TET2 (Ki ~5 uM vs aKG Km ~55 uM)",
                    "consequence": "Global DNA hypermethylation, blocked differentiation",
                },
                {
                    "process": "Histone demethylation (KDM4A/B, KDM2A)",
                    "effect": "2-HG inhibits Jumonji-domain histone demethylases",
                    "consequence": "H3K9me3 and H3K27me3 accumulation, gene silencing",
                },
                {
                    "process": "Collagen maturation (prolyl hydroxylases)",
                    "effect": "2-HG inhibits PHDs",
                    "consequence": "Altered bone marrow niche / HIF pathway modulation",
                },
            ],
            "downstream_targets": [
                "TET2", "KDM4A", "KDM4B", "KDM2A", "PHD1", "PHD2",
                "AlkBH2", "FTO",
            ],
            "interacting_pathways": [
                "DNMT3A-mediated methylation (antagonistic: IDH2 adds methyl, DNMT3A loss removes it)",
                "PRC2 histone methylation (synergistic: both silence genes)",
                "HIF-1alpha signaling (PHD inhibition stabilizes HIF)",
            ],
        },
        "SETBP1_G870S": {
            "primary_pathway": "SET/PP2A inhibition / Chromatin remodeling / Self-renewal",
            "affected_processes": [
                {
                    "process": "PP2A tumor suppressor phosphatase",
                    "effect": "SETBP1 stabilization -> increased SET binding -> PP2A inhibition",
                    "consequence": "Loss of PP2A de-represses AKT, ERK, MYC, WNT",
                },
                {
                    "process": "HOXA9/HOXA10 transcriptional activation",
                    "effect": "SETBP1 acts as transcription factor at HOXA locus",
                    "consequence": "Direct self-renewal gene activation (convergent with DNMT3A)",
                },
                {
                    "process": "RUNX1 repression",
                    "effect": "SETBP1 represses RUNX1 transcription",
                    "consequence": "Blocked myeloid differentiation (RUNX1 is master regulator)",
                },
                {
                    "process": "Protein stability / ubiquitin-proteasome",
                    "effect": "G870S in degron escapes beta-TrCP ubiquitination",
                    "consequence": "SETBP1 protein half-life increases ~5-10x",
                },
            ],
            "downstream_targets": [
                "SET (I2PP2A)", "PP2A", "AKT", "ERK1/2", "MYC",
                "HOXA9", "HOXA10", "RUNX1", "beta-catenin",
            ],
            "interacting_pathways": [
                "RAS-MAPK (via PP2A de-repression of ERK)",
                "PI3K-AKT (via PP2A de-repression of AKT)",
                "WNT/beta-catenin (via PP2A de-repression)",
                "MYC transcriptional program",
            ],
        },
        "PTPN11_E76Q": {
            "primary_pathway": "RAS-MAPK signaling (SHP2 upstream activator)",
            "affected_processes": [
                {
                    "process": "RAS-MAPK (RAF-MEK-ERK) cascade",
                    "effect": "Constitutive SHP2 activity removes RAS-GAP binding, activates RAS",
                    "consequence": "Sustained ERK phosphorylation, proliferation, survival",
                },
                {
                    "process": "PI3K-AKT pathway",
                    "effect": "SHP2 activates PI3K via GAB1/GAB2 scaffolding",
                    "consequence": "mTOR activation, anti-apoptosis",
                },
                {
                    "process": "JAK-STAT signaling",
                    "effect": "SHP2 modulates JAK2/STAT3/STAT5 signaling",
                    "consequence": "Cytokine hypersensitivity, altered differentiation",
                },
                {
                    "process": "SRC family kinase signaling",
                    "effect": "SHP2 dephosphorylates inhibitory pY of SRC kinases",
                    "consequence": "Enhanced SRC-mediated survival and migration",
                },
            ],
            "downstream_targets": [
                "RAS (NRAS/KRAS/HRAS)", "RAF", "MEK1/2", "ERK1/2",
                "PI3K", "AKT", "mTOR", "JAK2", "STAT3", "STAT5",
                "SRC", "GAB1", "GAB2",
            ],
            "interacting_pathways": [
                "SETBP1/PP2A (converge on ERK and AKT)",
                "Growth factor receptors (FLT3, KIT, CSF1R)",
                "Cytokine receptors (IL-3R, GM-CSFR, EPO-R)",
            ],
        },
        "Monosomy7": {
            "primary_pathway": "Haploinsufficiency of multiple tumor suppressors",
            "affected_processes": [
                {
                    "process": "PRC2-mediated gene silencing (EZH2 at 7q36.1)",
                    "effect": "~50% reduction in EZH2 dosage",
                    "consequence": "Impaired H3K27me3, de-repression of oncogenic programs. "
                                   "Compounds the epigenetic chaos from DNMT3A + IDH2.",
                },
                {
                    "process": "Cell cycle / DNA repair (CUX1 at 7q22.1)",
                    "effect": "Haploinsufficiency of CUX1",
                    "consequence": "Impaired DNA damage response, genomic instability",
                },
                {
                    "process": "Innate immunity (SAMD9/SAMD9L at 7q21.2)",
                    "effect": "Reduced SAMD9/SAMD9L dosage",
                    "consequence": "Impaired innate immune tumor surveillance",
                },
                {
                    "process": "RNA splicing (LUC7L2 at 7q34)",
                    "effect": "Haploinsufficiency of splicing factor",
                    "consequence": "Aberrant mRNA processing, compounding with any spliceosome mutations",
                },
                {
                    "process": "Epigenetic regulation (MLL3/KMT2C at 7q36.1)",
                    "effect": "Reduced H3K4 methylation at enhancers",
                    "consequence": "Enhancer dysfunction, altered gene expression",
                },
            ],
            "downstream_targets": [
                "EZH2", "CUX1", "KMT2C/MLL3", "SAMD9", "SAMD9L",
                "LUC7L2", "DOCK4", "MLL5/KMT2E",
            ],
            "interacting_pathways": [
                "PRC2 / H3K27me3 silencing (EZH2 loss)",
                "DNA damage repair (CUX1 loss)",
                "RAS pathway (monosomy 7 frequently co-occurs with RAS mutations)",
            ],
        },
    }
    return pathways


# ---------------------------------------------------------------------------
# 2. Convergence analysis
# ---------------------------------------------------------------------------

def build_convergence_analysis():
    """Analyze how the mutations converge on shared pathways and synergize."""
    convergences = [
        {
            "convergence_id": "EPIGENETIC_AXIS",
            "title": "Epigenetic dysregulation: DNMT3A + IDH2 + Monosomy 7 (EZH2)",
            "mutations_involved": ["DNMT3A R882H", "IDH2 R140Q", "Monosomy 7 (EZH2 loss)"],
            "mechanism": (
                "Three independent hits on the epigenetic machinery create a uniquely "
                "chaotic chromatin landscape:\n"
                "  (1) DNMT3A R882H: focal hypomethylation at self-renewal genes\n"
                "  (2) IDH2 R140Q: global hypermethylation via 2-HG inhibition of TET2 "
                "and histone demethylases\n"
                "  (3) EZH2 haploinsufficiency (chr7 loss): impaired H3K27me3 gene silencing\n\n"
                "Net effect: some loci are hypomethylated (DNMT3A targets), others are "
                "hypermethylated (IDH2/2-HG targets), and PRC2-mediated silencing is "
                "weakened (EZH2 loss). This triple epigenetic hit is exceptionally rare "
                "and creates a differentiation block with paradoxical gene expression patterns."
            ),
            "clinical_significance": (
                "DNMT3A + IDH2 co-mutation is well-described in AML and associated with "
                "response to IDH2 inhibitors (enasidenib). However, the addition of EZH2 "
                "haploinsufficiency from monosomy 7 may alter the epigenetic equilibrium "
                "in ways that are not well-characterized."
            ),
            "therapeutic_relevance": [
                "Enasidenib (IDH2 inhibitor) -- directly targets 2-HG production",
                "Hypomethylating agents (azacitidine/decitabine) -- may partially restore "
                "methylation at DNMT3A-hypomethylated loci, but paradoxically help at "
                "IDH2-hypermethylated loci too",
                "EZH2 status may affect HMA response",
            ],
        },
        {
            "convergence_id": "RAS_PP2A_AXIS",
            "title": "RAS/MAPK + PP2A convergence: PTPN11 + SETBP1",
            "mutations_involved": ["PTPN11 E76Q", "SETBP1 G870S"],
            "mechanism": (
                "PTPN11 E76Q directly activates RAS-MAPK through constitutive SHP2 "
                "phosphatase activity. Simultaneously, SETBP1 G870S inhibits PP2A, "
                "which is a major negative regulator of ERK and AKT. The result is "
                "dual activation of the MAPK cascade:\n"
                "  (1) SHP2 activates RAS at the top of the cascade\n"
                "  (2) PP2A inhibition prevents dephosphorylation of ERK at the bottom\n\n"
                "This creates a feed-forward loop where ERK is activated AND cannot be "
                "inactivated. Both mutations also converge on AKT: SHP2 activates PI3K, "
                "and PP2A loss de-represses AKT directly.\n\n"
                "This dual RAS activation may explain the aggressive clinical behavior "
                "and is consistent with the known co-occurrence of SETBP1 with RAS "
                "pathway mutations in atypical CML and MDS/MPN overlap."
            ),
            "clinical_significance": (
                "MEK inhibitors (trametinib) may be partially effective but the PP2A "
                "inhibition by SETBP1 provides a bypass mechanism. Targeting both arms "
                "simultaneously may be needed."
            ),
            "therapeutic_relevance": [
                "SHP2 inhibitors (TNO155, RMC-4630) -- directly target PTPN11",
                "MEK inhibitors (trametinib) -- target downstream of both mutations",
                "PP2A reactivators (LB-100 controversy, DT-061/SMAPs) -- could counteract SETBP1",
                "Combined SHP2i + MEKi -- addresses both input and output of the cascade",
            ],
        },
        {
            "convergence_id": "SELF_RENEWAL_AXIS",
            "title": "HOXA self-renewal convergence: DNMT3A + SETBP1",
            "mutations_involved": ["DNMT3A R882H", "SETBP1 G870S"],
            "mechanism": (
                "Both DNMT3A R882H and SETBP1 G870S independently activate the HOXA "
                "self-renewal program, but through different mechanisms:\n"
                "  (1) DNMT3A R882H: hypomethylation of HOXA cluster regulatory regions "
                "removes epigenetic silencing\n"
                "  (2) SETBP1 G870S: direct transcriptional activation of HOXA9/HOXA10 "
                "as a DNA-binding transcription factor\n\n"
                "The dual activation may create a particularly strong self-renewal signal "
                "that is resistant to single-agent targeting. Both also converge on MYC: "
                "DNMT3A via enhancer hypomethylation, SETBP1 via PP2A-mediated MYC "
                "stabilization."
            ),
            "clinical_significance": (
                "The HOXA convergence may explain the high VAFs of both mutations "
                "(39% and 34%), suggesting they co-exist in the dominant clone and "
                "provide a strong selective advantage together."
            ),
            "therapeutic_relevance": [
                "Menin inhibitors (revumenib/ziftomenib) -- disrupt HOXA transcription",
                "DOT1L inhibitors -- HOXA expression depends on H3K79 methylation",
                "Indirect: target downstream (MYC inhibitors, CDK inhibitors)",
            ],
        },
        {
            "convergence_id": "CHR7_SYNERGY",
            "title": "Monosomy 7 interactions with all four mutations",
            "mutations_involved": [
                "Monosomy 7",
                "DNMT3A R882H",
                "IDH2 R140Q",
                "SETBP1 G870S",
                "PTPN11 E76Q",
            ],
            "mechanism": (
                "Monosomy 7 is not a passive bystander; it synergizes with each mutation:\n\n"
                "  With DNMT3A + IDH2: EZH2 haploinsufficiency (7q36.1) adds a third "
                "epigenetic hit. KMT2C/MLL3 loss further disrupts enhancer methylation.\n\n"
                "  With PTPN11: Monosomy 7 and RAS pathway mutations (PTPN11, NRAS, KRAS) "
                "are statistically co-associated in MDS/AML. Proposed mechanism: "
                "haploinsufficiency of negative RAS regulators on chr7 (e.g., DOCK4 at "
                "7q31.1 is a RAC-GEF that modulates RAS-MAPK feedback).\n\n"
                "  With SETBP1: SETBP1 mutations are enriched in patients with monosomy 7 / "
                "del(7q) in MDS/MPN overlap syndromes. The combination may select for "
                "clones with both self-renewal advantage (SETBP1) and impaired differentiation "
                "(chr7 tumor suppressors).\n\n"
                "  CUX1 loss (7q22.1): impairs DNA damage response, potentially allowing "
                "accumulation of additional mutations and chromosomal instability."
            ),
            "clinical_significance": (
                "Monosomy 7 is independently adverse in both ELN 2022 AML risk and "
                "IPSS-M MDS risk stratification. The combination with RAS pathway "
                "mutations (PTPN11) and SETBP1 places this in the highest-risk category."
            ),
            "therapeutic_relevance": [
                "No direct therapy for monosomy 7",
                "Allogeneic HSCT -- only curative option for adverse cytogenetics",
                "EZH2 loss may predict resistance to EZH2 inhibitors (already haploinsufficient)",
                "Intensive monitoring for clonal evolution / relapse",
            ],
        },
        {
            "convergence_id": "MYC_HUB",
            "title": "MYC as a central convergence node",
            "mutations_involved": [
                "DNMT3A R882H",
                "SETBP1 G870S",
                "PTPN11 E76Q",
                "IDH2 R140Q",
            ],
            "mechanism": (
                "MYC is activated through four independent mechanisms:\n"
                "  (1) DNMT3A R882H: hypomethylation of MYC super-enhancer\n"
                "  (2) SETBP1 G870S: PP2A inhibition prevents MYC dephosphorylation "
                "(pS62 stabilization, pT58 dephosphorylation blocked)\n"
                "  (3) PTPN11 E76Q: ERK phosphorylates MYC at S62, stabilizing it; "
                "PI3K-AKT-GSK3beta axis modulates T58 phosphorylation\n"
                "  (4) IDH2 R140Q: 2-HG may indirectly affect MYC targets through "
                "epigenetic silencing of MYC antagonists\n\n"
                "This quadruple hit on MYC stabilization/activation represents a critical "
                "vulnerability. MYC protein is notoriously 'undruggable' directly, but "
                "CDK7/CDK9 inhibitors, BET bromodomain inhibitors, and Aurora kinase "
                "inhibitors can target MYC-dependent transcription."
            ),
            "clinical_significance": (
                "MYC overexpression/stabilization is associated with AML transformation "
                "from MDS and poor prognosis."
            ),
            "therapeutic_relevance": [
                "BET inhibitors (birabresib, mivebresib) -- suppress MYC transcription",
                "CDK7 inhibitors (samuraciclib) -- disrupt MYC-dependent transcription",
                "Aurora kinase inhibitors (alisertib) -- synthetic lethal with MYC",
                "Indirect: PP2A reactivators + MEK inhibitors to reduce MYC stability",
            ],
        },
    ]
    return convergences


# ---------------------------------------------------------------------------
# 3. Network graph — nodes and edges
# ---------------------------------------------------------------------------

def build_network():
    """Build a node-edge graph of the signaling cascades with mutation annotations."""

    # Define node categories for coloring
    # mutation: directly mutated gene/element
    # pathway_node: signaling molecule affected downstream
    # process: biological process
    # target: transcriptional target / effector

    nodes = [
        # Mutated genes (tier 1)
        {"id": "DNMT3A", "label": "DNMT3A R882H", "category": "mutation",
         "mutation": "DNMT3A_R882H", "vaf": 0.39,
         "description": "Dominant-negative loss of DNA methyltransferase"},
        {"id": "IDH2", "label": "IDH2 R140Q", "category": "mutation",
         "mutation": "IDH2_R140Q", "vaf": 0.02,
         "description": "Neomorphic: produces 2-HG oncometabolite"},
        {"id": "SETBP1", "label": "SETBP1 G870S", "category": "mutation",
         "mutation": "SETBP1_G870S", "vaf": 0.34,
         "description": "Stabilized: escapes proteasomal degradation"},
        {"id": "PTPN11", "label": "PTPN11 E76Q\n(SHP2)", "category": "mutation",
         "mutation": "PTPN11_E76Q", "vaf": 0.29,
         "description": "Constitutively active phosphatase"},
        {"id": "CHR7", "label": "Monosomy 7\n(-7)", "category": "mutation",
         "mutation": "Monosomy7", "vaf": None,
         "description": "Loss of entire chromosome 7"},

        # Signaling pathway nodes (tier 2)
        {"id": "2HG", "label": "2-HG", "category": "oncometabolite",
         "description": "Oncometabolite produced by mutant IDH2"},
        {"id": "TET2", "label": "TET2", "category": "pathway_node",
         "description": "DNA demethylase inhibited by 2-HG"},
        {"id": "SET", "label": "SET\n(I2PP2A)", "category": "pathway_node",
         "description": "PP2A inhibitor, bound by SETBP1"},
        {"id": "PP2A", "label": "PP2A", "category": "pathway_node",
         "description": "Tumor suppressor phosphatase, inhibited by SET-SETBP1"},
        {"id": "RAS", "label": "RAS", "category": "pathway_node",
         "description": "Activated by SHP2, central oncogenic switch"},
        {"id": "RAF", "label": "RAF", "category": "pathway_node",
         "description": "MAPK cascade kinase"},
        {"id": "MEK", "label": "MEK1/2", "category": "pathway_node",
         "description": "MAPK cascade kinase"},
        {"id": "ERK", "label": "ERK1/2", "category": "convergence",
         "description": "CONVERGENCE: activated by RAS-MAPK AND de-repressed by PP2A loss"},
        {"id": "PI3K", "label": "PI3K", "category": "pathway_node",
         "description": "Lipid kinase, activated by SHP2"},
        {"id": "AKT", "label": "AKT", "category": "convergence",
         "description": "CONVERGENCE: activated by PI3K AND de-repressed by PP2A loss"},
        {"id": "mTOR", "label": "mTOR", "category": "pathway_node",
         "description": "Master growth/metabolism regulator"},
        {"id": "GSK3B", "label": "GSK3-beta", "category": "pathway_node",
         "description": "Kinase regulated by AKT, phosphorylates MYC T58"},
        {"id": "BCATENIN", "label": "beta-catenin", "category": "pathway_node",
         "description": "WNT effector, de-repressed by PP2A loss"},

        # Epigenetic nodes
        {"id": "DNA_METH", "label": "DNA\nmethylation", "category": "process",
         "description": "Reduced by DNMT3A R882H"},
        {"id": "H3K27ME3", "label": "H3K27me3", "category": "process",
         "description": "Reduced by EZH2 haploinsufficiency"},
        {"id": "HISTONE_METH", "label": "Histone\nhypermethylation", "category": "process",
         "description": "Increased by 2-HG inhibition of KDMs"},
        {"id": "EZH2", "label": "EZH2\n(chr7 loss)", "category": "chr7_gene",
         "description": "PRC2 methyltransferase, haploinsufficient"},
        {"id": "CUX1", "label": "CUX1\n(chr7 loss)", "category": "chr7_gene",
         "description": "Cell cycle/DNA repair TF, haploinsufficient"},
        {"id": "KMT2C", "label": "KMT2C/MLL3\n(chr7 loss)", "category": "chr7_gene",
         "description": "Enhancer methyltransferase, haploinsufficient"},

        # Transcriptional targets / effectors (tier 3)
        {"id": "MYC", "label": "MYC", "category": "convergence",
         "description": "CONVERGENCE: stabilized by ERK (S62), de-repressed by PP2A loss, "
                        "enhancer hypomethylated by DNMT3A"},
        {"id": "HOXA", "label": "HOXA9/10", "category": "convergence",
         "description": "CONVERGENCE: activated by both DNMT3A hypomethylation and "
                        "SETBP1 transcription factor activity"},
        {"id": "RUNX1", "label": "RUNX1", "category": "target",
         "description": "Master myeloid TF, repressed by SETBP1"},

        # Cellular outcomes
        {"id": "SELF_RENEWAL", "label": "Self-renewal", "category": "outcome",
         "description": "Enhanced by HOXA, MYC, DNMT3A, SETBP1"},
        {"id": "DIFF_BLOCK", "label": "Differentiation\nblock", "category": "outcome",
         "description": "Caused by IDH2/2-HG hypermethylation + RUNX1 repression"},
        {"id": "PROLIFERATION", "label": "Proliferation", "category": "outcome",
         "description": "Driven by RAS-MAPK-ERK + PI3K-AKT-mTOR"},
        {"id": "SURVIVAL", "label": "Anti-apoptosis", "category": "outcome",
         "description": "AKT-mediated + MYC-driven"},
    ]

    edges = [
        # DNMT3A pathway
        {"source": "DNMT3A", "target": "DNA_METH", "type": "inhibits",
         "label": "dominant-negative\n(~80% reduction)"},
        {"source": "DNA_METH", "target": "HOXA", "type": "inhibits",
         "label": "hypomethylation\nde-represses"},
        {"source": "DNA_METH", "target": "MYC", "type": "inhibits",
         "label": "enhancer\nhypomethylation"},

        # IDH2 pathway
        {"source": "IDH2", "target": "2HG", "type": "produces",
         "label": "neomorphic\nenzyme"},
        {"source": "2HG", "target": "TET2", "type": "inhibits",
         "label": "competitive\ninhibition"},
        {"source": "TET2", "target": "DNA_METH", "type": "inhibits",
         "label": "demethylation\nblocked"},
        {"source": "2HG", "target": "HISTONE_METH", "type": "activates",
         "label": "KDM inhibition\n-> hypermethylation"},
        {"source": "HISTONE_METH", "target": "DIFF_BLOCK", "type": "activates",
         "label": "gene silencing"},

        # SETBP1 pathway
        {"source": "SETBP1", "target": "SET", "type": "activates",
         "label": "stabilized binding"},
        {"source": "SET", "target": "PP2A", "type": "inhibits",
         "label": "PP2A inhibition"},
        {"source": "PP2A", "target": "ERK", "type": "inhibits",
         "label": "dephosphorylation\nblocked"},
        {"source": "PP2A", "target": "AKT", "type": "inhibits",
         "label": "dephosphorylation\nblocked"},
        {"source": "PP2A", "target": "MYC", "type": "inhibits",
         "label": "MYC pS62\nstabilized"},
        {"source": "PP2A", "target": "BCATENIN", "type": "inhibits",
         "label": "WNT de-repressed"},
        {"source": "SETBP1", "target": "HOXA", "type": "activates",
         "label": "direct TF\nactivation"},
        {"source": "SETBP1", "target": "RUNX1", "type": "inhibits",
         "label": "transcriptional\nrepression"},
        {"source": "RUNX1", "target": "DIFF_BLOCK", "type": "activates",
         "label": "loss of myeloid\ndifferentiation"},

        # PTPN11/SHP2 pathway
        {"source": "PTPN11", "target": "RAS", "type": "activates",
         "label": "RAS-GAP release"},
        {"source": "RAS", "target": "RAF", "type": "activates", "label": ""},
        {"source": "RAF", "target": "MEK", "type": "activates", "label": ""},
        {"source": "MEK", "target": "ERK", "type": "activates", "label": "phosphorylation"},
        {"source": "PTPN11", "target": "PI3K", "type": "activates",
         "label": "via GAB1/2"},
        {"source": "PI3K", "target": "AKT", "type": "activates",
         "label": "PIP3 generation"},
        {"source": "AKT", "target": "mTOR", "type": "activates", "label": ""},
        {"source": "AKT", "target": "GSK3B", "type": "inhibits",
         "label": "phosphorylation"},
        {"source": "ERK", "target": "MYC", "type": "activates",
         "label": "S62 phosphorylation\n(stabilization)"},

        # Monosomy 7 effects
        {"source": "CHR7", "target": "EZH2", "type": "inhibits",
         "label": "haploinsufficiency"},
        {"source": "CHR7", "target": "CUX1", "type": "inhibits",
         "label": "haploinsufficiency"},
        {"source": "CHR7", "target": "KMT2C", "type": "inhibits",
         "label": "haploinsufficiency"},
        {"source": "EZH2", "target": "H3K27ME3", "type": "activates",
         "label": "reduced PRC2\nactivity"},
        {"source": "H3K27ME3", "target": "SELF_RENEWAL", "type": "inhibits",
         "label": "de-repression of\nself-renewal genes"},

        # Convergence outputs
        {"source": "HOXA", "target": "SELF_RENEWAL", "type": "activates",
         "label": "stem cell\nprogram"},
        {"source": "MYC", "target": "SELF_RENEWAL", "type": "activates",
         "label": "proliferative\nself-renewal"},
        {"source": "MYC", "target": "PROLIFERATION", "type": "activates",
         "label": "cell cycle\nentry"},
        {"source": "ERK", "target": "PROLIFERATION", "type": "activates",
         "label": "mitogenic\nsignaling"},
        {"source": "AKT", "target": "SURVIVAL", "type": "activates",
         "label": "BAD inactivation"},
        {"source": "mTOR", "target": "PROLIFERATION", "type": "activates",
         "label": "protein synthesis"},
        {"source": "BCATENIN", "target": "SELF_RENEWAL", "type": "activates",
         "label": "WNT target\ngenes"},
    ]

    network = {"nodes": nodes, "edges": edges}
    return network


def create_network_visualization(network, output_path):
    """Create an interactive Plotly network visualization."""

    G = nx.DiGraph()

    # Add nodes with attributes
    for node in network["nodes"]:
        G.add_node(node["id"], **node)

    # Add edges
    for edge in network["edges"]:
        G.add_edge(edge["source"], edge["target"],
                    type=edge["type"], label=edge.get("label", ""))

    # Layout — use kamada_kawai for better separation
    pos = nx.kamada_kawai_layout(G, scale=2.0)

    # Manual position adjustments for biological clarity
    # Place mutations at top, outcomes at bottom, signaling in middle
    layer_y = {
        "mutation": 1.8,
        "oncometabolite": 1.0,
        "pathway_node": 0.3,
        "convergence": -0.3,
        "process": 0.6,
        "chr7_gene": 1.0,
        "target": -0.6,
        "outcome": -1.5,
    }

    # Seed positions by category, then refine
    for node_id, data in G.nodes(data=True):
        cat = data.get("category", "pathway_node")
        if cat in layer_y:
            # Keep x from layout, override y by category
            pos[node_id] = (pos[node_id][0], layer_y[cat] + pos[node_id][1] * 0.3)

    # Specific position overrides for clarity
    manual_pos = {
        "DNMT3A":       (-2.0, 2.0),
        "IDH2":         (-0.5, 2.0),
        "SETBP1":       (1.0, 2.0),
        "PTPN11":       (2.5, 2.0),
        "CHR7":         (-3.2, 2.0),

        "DNA_METH":     (-2.0, 1.2),
        "2HG":          (-0.5, 1.2),
        "TET2":         (-1.2, 0.6),
        "HISTONE_METH": (0.0, 0.6),
        "SET":          (1.0, 1.2),
        "PP2A":         (1.0, 0.3),
        "RAS":          (2.5, 1.2),
        "RAF":          (2.5, 0.7),
        "MEK":          (2.5, 0.2),
        "PI3K":         (3.2, 0.8),

        "EZH2":         (-3.2, 1.0),
        "CUX1":         (-3.8, 0.5),
        "KMT2C":        (-3.5, 0.2),
        "H3K27ME3":     (-3.2, 0.0),

        "ERK":          (2.0, -0.3),
        "AKT":          (1.5, -0.3),
        "mTOR":         (2.0, -0.8),
        "GSK3B":        (1.0, -0.6),
        "BCATENIN":     (0.2, -0.3),

        "MYC":          (0.0, -0.8),
        "HOXA":         (-1.5, -0.5),
        "RUNX1":        (0.5, 0.0),

        "SELF_RENEWAL": (-1.5, -1.5),
        "DIFF_BLOCK":   (0.0, -1.5),
        "PROLIFERATION":(1.5, -1.5),
        "SURVIVAL":     (2.5, -1.5),
    }
    for k, v in manual_pos.items():
        if k in pos:
            pos[k] = v

    # Color mapping by category
    color_map = {
        "mutation":       "#E74C3C",   # red
        "oncometabolite": "#E67E22",   # orange
        "pathway_node":   "#3498DB",   # blue
        "convergence":    "#9B59B6",   # purple (convergence points)
        "process":        "#1ABC9C",   # teal
        "chr7_gene":      "#E74C3C",   # red (also mutated/lost)
        "target":         "#2ECC71",   # green
        "outcome":        "#34495E",   # dark gray
    }

    # Size by importance
    size_map = {
        "mutation": 40,
        "oncometabolite": 25,
        "pathway_node": 22,
        "convergence": 35,
        "process": 22,
        "chr7_gene": 25,
        "target": 22,
        "outcome": 30,
    }

    # Build edge traces
    edge_traces = []
    annotation_list = []

    for edge in network["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]

        # Shorten edges slightly so arrows don't overlap nodes
        dx, dy = x1 - x0, y1 - y0
        length = math.sqrt(dx**2 + dy**2) if (dx**2 + dy**2) > 0 else 1
        shrink = 0.08
        x0a = x0 + dx * shrink / length
        y0a = y0 + dy * shrink / length
        x1a = x1 - dx * shrink / length
        y1a = y1 - dy * shrink / length

        edge_color = "#C0392B" if edge["type"] == "inhibits" else "#27AE60"
        dash = "dot" if edge["type"] == "inhibits" else "solid"

        edge_traces.append(
            go.Scatter(
                x=[x0a, x1a, None],
                y=[y0a, y1a, None],
                mode="lines",
                line=dict(width=1.5, color=edge_color, dash=dash),
                hoverinfo="none",
                showlegend=False,
            )
        )

        # Arrowhead annotation
        annotation_list.append(dict(
            ax=x0a, ay=y0a,
            x=x1a, y=y1a,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor=edge_color,
            opacity=0.6,
        ))

    # Build node traces (one per category for legend)
    node_traces = {}
    for node in network["nodes"]:
        cat = node.get("category", "pathway_node")
        if cat not in node_traces:
            node_traces[cat] = {
                "x": [], "y": [], "text": [], "hover": [],
                "size": [], "ids": [],
            }
        x, y = pos[node["id"]]
        node_traces[cat]["x"].append(x)
        node_traces[cat]["y"].append(y)
        node_traces[cat]["text"].append(node["label"])
        vaf_str = f"<br>VAF: {node['vaf']*100:.0f}%" if node.get("vaf") else ""
        node_traces[cat]["hover"].append(
            f"<b>{node['label']}</b>{vaf_str}<br>{node.get('description', '')}"
        )
        node_traces[cat]["size"].append(size_map.get(cat, 22))
        node_traces[cat]["ids"].append(node["id"])

    category_labels = {
        "mutation": "Mutated genes",
        "oncometabolite": "Oncometabolite",
        "pathway_node": "Signaling nodes",
        "convergence": "Convergence points",
        "process": "Epigenetic processes",
        "chr7_gene": "Chr7 genes (lost)",
        "target": "Transcription factors",
        "outcome": "Cellular outcomes",
    }

    scatter_traces = []
    for cat, data in node_traces.items():
        scatter_traces.append(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="markers+text",
                marker=dict(
                    size=data["size"],
                    color=color_map.get(cat, "#95A5A6"),
                    line=dict(width=2, color="white"),
                    opacity=0.9,
                ),
                text=data["text"],
                textposition="middle center",
                textfont=dict(size=7, color="white", family="Arial Black"),
                hovertext=data["hover"],
                hoverinfo="text",
                name=category_labels.get(cat, cat),
                legendgroup=cat,
            )
        )

    fig = go.Figure(data=edge_traces + scatter_traces)

    # Add arrowheads
    for ann in annotation_list:
        fig.add_annotation(**ann)

    fig.update_layout(
        title=dict(
            text=(
                "Pathway Interaction Network: DNMT3A R882H + IDH2 R140Q + "
                "SETBP1 G870S + PTPN11 E76Q + Monosomy 7<br>"
                "<sub>Red/dashed = inhibition | Green/solid = activation | "
                "Purple nodes = convergence points</sub>"
            ),
            font=dict(size=14),
        ),
        showlegend=True,
        legend=dict(
            x=1.02, y=1, bordercolor="black", borderwidth=1,
            font=dict(size=10),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4.5, 4.0]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.0, 2.5]),
        plot_bgcolor="white",
        width=1200,
        height=900,
        margin=dict(l=20, r=200, t=80, b=20),
        hovermode="closest",
    )

    fig.write_html(output_path, include_plotlyjs=True)
    return fig


# ---------------------------------------------------------------------------
# 4. Therapeutic vulnerability analysis
# ---------------------------------------------------------------------------

def build_therapeutic_targets():
    """Rank therapeutic targets by pathway convergence, druggability, and evidence."""

    targets = [
        {
            "rank": 1,
            "target": "IDH2 (enasidenib / AG-221)",
            "drug_class": "IDH2 inhibitor",
            "drugs": ["Enasidenib (FDA-approved)", "Vorasidenib (brain-penetrant)"],
            "pathways_affected": ["2-HG production", "TET2 inhibition", "Histone hypermethylation",
                                  "Differentiation block"],
            "n_pathways": 4,
            "druggability": "Approved (FDA 2017 for relapsed/refractory AML)",
            "clinical_evidence": "Strong",
            "evidence_detail": (
                "Enasidenib: 40% ORR, 20% CR in R/R IDH2-mutant AML (AG221-C-001 trial). "
                "However, IDH2 VAF is only 2%, meaning this is a small subclone. "
                "Enasidenib would eliminate the IDH2 subclone but not the dominant clone. "
                "CRITICAL: the 2% VAF means IDH2 is NOT the driver of the dominant disease. "
                "Still therapeutically relevant because: (a) it may be the clone driving "
                "AML transformation, (b) even small subclones can expand rapidly under "
                "selective pressure."
            ),
            "patient_specific_notes": (
                "IDH2 R140Q at 2% VAF is a late subclone (see clonal architecture). "
                "DNMT3A + IDH2 co-mutation is a known high-risk combination. "
                "Enasidenib may be most effective if IDH2 clone expands."
            ),
        },
        {
            "rank": 2,
            "target": "SHP2 / PTPN11 (SHP2 inhibitors)",
            "drug_class": "SHP2 allosteric inhibitor",
            "drugs": ["TNO155 (Novartis, Phase I/II)", "RMC-4630 (Revolution, Phase I/II)",
                      "ERAS-601 (Erasca, Phase I)", "JAB-3068 (Jacobio, Phase I/II)"],
            "pathways_affected": ["RAS-MAPK", "PI3K-AKT", "JAK-STAT", "MYC (indirect)"],
            "n_pathways": 4,
            "druggability": "Clinical trials (multiple Phase I/II)",
            "clinical_evidence": "Emerging",
            "evidence_detail": (
                "SHP2 inhibitors show promise in RAS-driven solid tumors and are being "
                "tested in myeloid neoplasms. TNO155 + trametinib combination shows "
                "synergy in preclinical models. DiffDock analysis of this patient's "
                "PTPN11 E76Q showed strong binding: RMC-4550 confidence -9.26, "
                "SHP099 -7.83, TNO155 -7.49."
            ),
            "patient_specific_notes": (
                "PTPN11 E76Q at 29% VAF is in the dominant clone. SHP2 inhibitors "
                "would directly target the main disease-driving clone. Combination with "
                "MEK inhibitor addresses both the RAS input (SHP2) and output (MEK)."
            ),
        },
        {
            "rank": 3,
            "target": "MEK1/2 (trametinib / selumetinib)",
            "drug_class": "MEK inhibitor",
            "drugs": ["Trametinib (FDA-approved for melanoma; off-label)",
                      "Selumetinib", "Binimetinib", "Cobimetinib"],
            "pathways_affected": ["RAS-MAPK (downstream of both PTPN11 and SETBP1/PP2A)",
                                  "ERK-MYC stabilization", "Proliferation"],
            "n_pathways": 3,
            "druggability": "Approved (different indication); clinical trials in myeloid",
            "clinical_evidence": "Moderate",
            "evidence_detail": (
                "Trametinib showed responses in RAS-mutant MDS/CMML in small studies. "
                "In this patient, MEK inhibition would block ERK activation from BOTH "
                "the SHP2->RAS arm (PTPN11) and the PP2A loss arm (SETBP1). "
                "However, the PP2A-mediated AKT activation would not be blocked by MEK "
                "inhibitors alone."
            ),
            "patient_specific_notes": (
                "MEK is the convergence point of both PTPN11 and SETBP1 pathways. "
                "Combination SHP2i + MEKi may provide the deepest MAPK suppression."
            ),
        },
        {
            "rank": 4,
            "target": "Menin-KMT2A interaction (menin inhibitors)",
            "drug_class": "Menin inhibitor",
            "drugs": ["Revumenib (FDA-approved 2024)", "Ziftomenib (Phase II/III)"],
            "pathways_affected": ["HOXA9/HOXA10 transcription (target of both DNMT3A and SETBP1)",
                                  "Self-renewal program", "MYC (indirect via HOXA)"],
            "n_pathways": 3,
            "druggability": "Approved (revumenib for KMT2A-r AML, 2024)",
            "clinical_evidence": "Moderate (expanding beyond KMT2A-r)",
            "evidence_detail": (
                "Revumenib/ziftomenib disrupt the menin-KMT2A complex required for HOXA "
                "gene expression. Originally developed for KMT2A-rearranged AML, they "
                "also show activity in NPM1-mutant AML that depends on HOXA. In this "
                "patient, HOXA is activated by TWO independent mechanisms (DNMT3A "
                "hypomethylation + SETBP1 TF activity), making menin inhibition a "
                "high-value convergence target."
            ),
            "patient_specific_notes": (
                "DNMT3A R882H (39% VAF) and SETBP1 G870S (34% VAF) both converge on "
                "HOXA. Menin inhibitors could disrupt this convergence node."
            ),
        },
        {
            "rank": 5,
            "target": "Hypomethylating agents (azacitidine / decitabine)",
            "drug_class": "DNA methyltransferase inhibitor",
            "drugs": ["Azacitidine (Vidaza)", "Decitabine (Dacogen)",
                      "Oral decitabine/cedazuridine (Inqovi)"],
            "pathways_affected": ["DNA methylation (complex: helps both hypo and hyper states)",
                                  "Epigenetic reprogramming", "Differentiation"],
            "n_pathways": 3,
            "druggability": "Approved (standard of care for MDS)",
            "clinical_evidence": "Strong",
            "evidence_detail": (
                "HMAs are standard of care for higher-risk MDS and unfit AML. "
                "They may have paradoxical dual benefits here: (a) in IDH2-hypermethylated "
                "regions, HMAs provide further demethylation that may restore gene "
                "expression, and (b) azacitidine + enasidenib combination showed improved "
                "responses in IDH2-mutant AML (AGILE trial). However, DNMT3A R882H "
                "already causes hypomethylation, so the interaction is complex."
            ),
            "patient_specific_notes": (
                "Likely part of backbone therapy regardless of targeted agents. "
                "Azacitidine + venetoclax is the current SOC for unfit AML."
            ),
        },
        {
            "rank": 6,
            "target": "BCL-2 (venetoclax)",
            "drug_class": "BCL-2 inhibitor (BH3 mimetic)",
            "drugs": ["Venetoclax (Venclexta, FDA-approved in combination)"],
            "pathways_affected": ["Apoptosis evasion", "IDH2-mutant cells are BCL-2 dependent",
                                  "Mitochondrial priming"],
            "n_pathways": 2,
            "druggability": "Approved (AML combination, CLL monotherapy)",
            "clinical_evidence": "Strong",
            "evidence_detail": (
                "Venetoclax + azacitidine is SOC for unfit AML (VIALE-A trial). IDH-mutant "
                "AML is particularly sensitive to venetoclax because 2-HG inhibits "
                "cytochrome c oxidase, creating BCL-2 dependency. The IDH2 subclone (2% VAF) "
                "would be highly vulnerable. Even the non-IDH2 clone may benefit from "
                "venetoclax given the AKT-mediated survival signaling."
            ),
            "patient_specific_notes": (
                "Ven+Aza is the likely backbone. PTPN11 mutations may confer some "
                "resistance to venetoclax through MAPK-mediated MCL-1 upregulation. "
                "This is an area of active investigation."
            ),
        },
        {
            "rank": 7,
            "target": "PP2A reactivation (SMAPs / DT-061)",
            "drug_class": "PP2A activator (small molecule activator of PP2A)",
            "drugs": ["DT-061 (preclinical)", "DBK-1154 (preclinical)",
                      "FTY720/fingolimod (repurposed, Phase I/II in AML)"],
            "pathways_affected": ["PP2A reactivation (direct counter to SETBP1 mechanism)",
                                  "ERK de-phosphorylation", "AKT de-phosphorylation",
                                  "MYC destabilization"],
            "n_pathways": 4,
            "druggability": "Preclinical / early clinical",
            "clinical_evidence": "Emerging",
            "evidence_detail": (
                "SMAPs (Small Molecule Activators of PP2A) directly counteract the "
                "SETBP1-SET-PP2A inhibition axis. DT-061 shows preclinical activity in "
                "AML models. FTY720 (fingolimod, approved for MS) has PP2A-activating "
                "properties and is in early AML trials. This is the most direct approach "
                "to counteract SETBP1 G870S."
            ),
            "patient_specific_notes": (
                "SETBP1 G870S at 34% VAF is a dominant clone mutation. PP2A reactivation "
                "would directly counteract the oncogenic mechanism. However, clinical "
                "tools are still early-stage."
            ),
        },
        {
            "rank": 8,
            "target": "Allogeneic HSCT",
            "drug_class": "Cellular immunotherapy / transplant",
            "drugs": ["Allogeneic hematopoietic stem cell transplant"],
            "pathways_affected": ["All pathways (graft-vs-leukemia effect replaces malignant clone)",
                                  "Monosomy 7 (no targeted therapy exists)"],
            "n_pathways": 5,
            "druggability": "Established procedure",
            "clinical_evidence": "Strong",
            "evidence_detail": (
                "Allo-HSCT is the only curative option for adverse-risk MDS/AML with "
                "monosomy 7. The complex multi-hit mutation profile (4 gene mutations + "
                "monosomy 7) makes complete pharmacologic eradication unlikely. HSCT "
                "provides graft-vs-leukemia immune surveillance against residual disease."
            ),
            "patient_specific_notes": (
                "Given the adverse risk profile (monosomy 7, complex mutations), allo-HSCT "
                "should be considered the primary curative strategy if the patient is fit. "
                "Note: patient is post-HSCT -- monitoring for relapse and considering "
                "donor lymphocyte infusion (DLI) or second transplant if needed."
            ),
        },
    ]

    return targets


def write_therapeutic_targets_md(targets, output_path):
    """Write therapeutic targets as formatted markdown."""
    lines = [
        "# Therapeutic Vulnerability Analysis",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Patient mutation profile: DNMT3A R882H (VAF 39%) + IDH2 R140Q (VAF 2%) + "
        "SETBP1 G870S (VAF 34%) + PTPN11 E76Q (VAF 29%) + Monosomy 7",
        "",
        "---",
        "",
        "## Ranking criteria",
        "",
        "| Criterion | Weight | Description |",
        "|-----------|--------|-------------|",
        "| Pathways affected | High | Number of dysregulated pathways targeted |",
        "| Druggability | High | Approved > clinical trial > preclinical |",
        "| Clinical evidence | High | Evidence in myeloid neoplasms specifically |",
        "| Patient specificity | Medium | Relevance to this specific mutation profile |",
        "",
        "---",
        "",
    ]

    for t in targets:
        lines.append(f"## {t['rank']}. {t['target']}")
        lines.append("")
        lines.append(f"**Drug class:** {t['drug_class']}")
        lines.append("")
        lines.append(f"**Drugs:** {', '.join(t['drugs'])}")
        lines.append("")
        lines.append(f"**Pathways affected ({t['n_pathways']}):**")
        for p in t["pathways_affected"]:
            lines.append(f"- {p}")
        lines.append("")
        lines.append(f"**Druggability:** {t['druggability']}")
        lines.append("")
        lines.append(f"**Clinical evidence:** {t['clinical_evidence']}")
        lines.append("")
        lines.append(f"**Evidence detail:** {t['evidence_detail']}")
        lines.append("")
        lines.append(f"**Patient-specific notes:** {t['patient_specific_notes']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend([
        "## Combination strategies (ranked by expected synergy)",
        "",
        "### 1. Venetoclax + Azacitidine + Enasidenib (backbone + IDH2 targeting)",
        "- Standard AML backbone + IDH2-specific agent",
        "- Addresses: epigenetic dysregulation, IDH2 subclone, BCL-2 dependency",
        "- Evidence: VIALE-A (ven+aza) + AGILE (aza+enasidenib)",
        "- Limitation: does not address PTPN11 or SETBP1 pathways",
        "",
        "### 2. SHP2 inhibitor + MEK inhibitor (dual RAS pathway blockade)",
        "- Targets: PTPN11 E76Q directly + downstream MAPK",
        "- Addresses convergence of PTPN11 and SETBP1 on ERK",
        "- Evidence: TNO155 + trametinib Phase I/II ongoing",
        "- Limitation: does not address epigenetic axis",
        "",
        "### 3. Venetoclax + Azacitidine + SHP2 inhibitor (backbone + signaling)",
        "- Combines epigenetic therapy with RAS pathway inhibition",
        "- Addresses: proliferative signaling (PTPN11), apoptosis evasion, methylation",
        "- Evidence: preclinical rationale strong, no clinical data yet",
        "",
        "### 4. Menin inhibitor as add-on (HOXA convergence targeting)",
        "- Add revumenib/ziftomenib to any of the above",
        "- Directly blocks the HOXA self-renewal node where DNMT3A + SETBP1 converge",
        "- Evidence: expanding from KMT2A-r AML to HOXA-dependent contexts",
        "",
        "### 5. Post-HSCT maintenance / relapse strategy",
        "- If relapse post-transplant: enasidenib (if IDH2 clone expanded) or",
        "  SHP2i + venetoclax (if dominant clone returns)",
        "- DLI (donor lymphocyte infusion) for graft-vs-leukemia boost",
        "- Monitoring: track DNMT3A (39% VAF founder), SETBP1, PTPN11 by NGS",
        "",
        "---",
        "",
        "*Note: This analysis is for research purposes. All therapeutic decisions should "
        "be made by the treating hematology/oncology team based on the complete clinical "
        "picture, performance status, and available clinical trials.*",
        "",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# 5. Clonal evolution model
# ---------------------------------------------------------------------------

def build_clonal_model():
    """Model clonal architecture from VAF data.

    VAFs:
      DNMT3A R882H  = 39% -> ~78% cancer cell fraction (if heterozygous in diploid)
      SETBP1 G870S  = 34% -> ~68% CCF
      PTPN11 E76Q   = 29% -> ~58% CCF
      IDH2 R140Q    = 2%  -> ~4% CCF

    Monosomy 7 affects copy number calculations but for the non-chr7 mutations
    (all four are on other chromosomes), standard diploid assumption applies.

    Clonal hierarchy inference:
      - DNMT3A at highest VAF (39%) = likely founder/earliest event
      - SETBP1 slightly lower (34%) = likely in founder clone or early subclone
      - PTPN11 lower (29%) = subclone within SETBP1+ cells, or branching
      - IDH2 at 2% = late, small subclone

    Most parsimonious model (linear):
      DNMT3A -> SETBP1 -> PTPN11 -> IDH2

    But branching is also possible:
      DNMT3A -> SETBP1 -> PTPN11 (29%)
                        -> IDH2 (2%)  [branch]
    """

    model = {
        "title": "Clonal evolution model based on VAF data",
        "assumptions": [
            "All mutations are heterozygous in diploid regions (CCF = 2 * VAF)",
            "DNMT3A (chr2), SETBP1 (chr18), PTPN11 (chr12), IDH2 (chr15) -- none on chr7",
            "Monosomy 7 does not affect VAF calculation for these four genes",
            "Tumor purity estimated at ~80% based on DNMT3A VAF (39% / ~0.5 = ~78% CCF)",
        ],
        "vafs": {
            "DNMT3A R882H": {"vaf": 0.39, "ccf": 0.78, "chromosome": "2p23.3"},
            "SETBP1 G870S": {"vaf": 0.34, "ccf": 0.68, "chromosome": "18q12.3"},
            "PTPN11 E76Q":  {"vaf": 0.29, "ccf": 0.58, "chromosome": "12q24.13"},
            "IDH2 R140Q":   {"vaf": 0.02, "ccf": 0.04, "chromosome": "15q26.1"},
            "Monosomy 7":   {"vaf": None, "ccf": None, "chromosome": "7 (entire)"},
        },
        "clonal_tree": {
            "description": "Most parsimonious linear clonal hierarchy",
            "clones": [
                {
                    "id": "clone_0",
                    "name": "Normal HSC",
                    "mutations": [],
                    "fraction": 0.22,
                    "color": "#2ECC71",
                },
                {
                    "id": "clone_1",
                    "name": "Founder clone (CHIP)",
                    "mutations": ["DNMT3A R882H"],
                    "additional_events": ["Monosomy 7 (timing uncertain, likely early)"],
                    "fraction": 0.10,
                    "color": "#3498DB",
                    "note": "DNMT3A R882H is the most common CHIP mutation. "
                            "This clone has self-renewal advantage but is pre-malignant.",
                },
                {
                    "id": "clone_2",
                    "name": "Expanded clone (self-renewal)",
                    "mutations": ["DNMT3A R882H", "SETBP1 G870S"],
                    "fraction": 0.10,
                    "color": "#E67E22",
                    "note": "SETBP1 G870S adds PP2A inhibition and HOXA activation. "
                            "This clone has strong self-renewal. May represent MDS phase.",
                },
                {
                    "id": "clone_3",
                    "name": "Dominant clone (proliferative)",
                    "mutations": ["DNMT3A R882H", "SETBP1 G870S", "PTPN11 E76Q"],
                    "fraction": 0.54,
                    "color": "#E74C3C",
                    "note": "PTPN11 E76Q adds RAS-MAPK proliferative signaling. "
                            "This is the dominant clone (58% CCF). Drives disease.",
                },
                {
                    "id": "clone_4",
                    "name": "Late subclone (IDH2+)",
                    "mutations": ["DNMT3A R882H", "SETBP1 G870S", "PTPN11 E76Q", "IDH2 R140Q"],
                    "fraction": 0.04,
                    "color": "#9B59B6",
                    "note": "IDH2 R140Q at 2% VAF is a late acquisition. "
                            "Small subclone, but may be the clone undergoing AML transformation. "
                            "IDH mutations often associated with blast expansion.",
                },
            ],
            "edges": [
                {"from": "clone_0", "to": "clone_1", "event": "DNMT3A R882H (+mono 7?)"},
                {"from": "clone_1", "to": "clone_2", "event": "SETBP1 G870S"},
                {"from": "clone_2", "to": "clone_3", "event": "PTPN11 E76Q"},
                {"from": "clone_3", "to": "clone_4", "event": "IDH2 R140Q"},
            ],
        },
        "clinical_implications": [
            "DNMT3A R882H as founder: likely present in residual CHIP even after treatment. "
            "Post-HSCT, DNMT3A+ CHIP of donor origin is common; monitoring requires "
            "distinguishing patient vs donor DNMT3A mutations.",
            "IDH2 at 2%: too small to be the primary therapeutic target, but could expand "
            "under selective pressure (e.g., if dominant clone is suppressed by therapy). "
            "Serial monitoring of IDH2 VAF is important.",
            "Monosomy 7 timing: could be concurrent with DNMT3A (founding event) or later. "
            "Cytogenetic data alone cannot determine timing. FISH on sorted populations "
            "could clarify.",
            "Treatment response prediction: the dominant clone (DNMT3A+SETBP1+PTPN11) at "
            "~58% CCF will determine initial treatment response. The IDH2 subclone may "
            "emerge at relapse.",
        ],
    }
    return model


def create_clonal_tree_visualization(model, output_path):
    """Create a clonal tree visualization using Plotly (fish plot style + tree)."""

    clones = model["clonal_tree"]["clones"]

    fig = go.Figure()

    # =========================================================================
    # Part 1: Clonal tree (top half)
    # =========================================================================
    tree_y_base = 1.5

    # Node positions for tree
    tree_positions = {
        "clone_0": (0.5, tree_y_base + 1.2),
        "clone_1": (0.5, tree_y_base + 0.8),
        "clone_2": (0.5, tree_y_base + 0.4),
        "clone_3": (0.5, tree_y_base + 0.0),
        "clone_4": (0.5, tree_y_base - 0.4),
    }

    # Draw edges (vertical lines with event labels)
    for edge in model["clonal_tree"]["edges"]:
        x0, y0 = tree_positions[edge["from"]]
        x1, y1 = tree_positions[edge["to"]]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color="#7F8C8D", width=3),
            showlegend=False,
            hoverinfo="none",
        ))
        # Event label
        mid_y = (y0 + y1) / 2
        fig.add_annotation(
            x=x0 + 0.35, y=mid_y,
            text=f"<b>{edge['event']}</b>",
            showarrow=False,
            font=dict(size=10, color="#2C3E50"),
            xanchor="left",
        )

    # Draw nodes
    for clone in clones:
        cid = clone["id"]
        x, y = tree_positions[cid]
        color = clone["color"]
        frac_pct = clone["fraction"] * 100

        # Node circle
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=35 + frac_pct * 0.5, color=color,
                        line=dict(width=2, color="white")),
            text=f"{frac_pct:.0f}%",
            textposition="middle center",
            textfont=dict(size=10, color="white", family="Arial Black"),
            hovertext=(
                f"<b>{clone['name']}</b><br>"
                f"Fraction: {frac_pct:.0f}%<br>"
                f"Mutations: {', '.join(clone['mutations']) if clone['mutations'] else 'None'}<br>"
                f"{clone.get('note', '')}"
            ),
            hoverinfo="text",
            name=clone["name"],
            showlegend=True,
        ))

        # Clone label
        fig.add_annotation(
            x=x - 0.35, y=y,
            text=clone["name"],
            showarrow=False,
            font=dict(size=9, color=color),
            xanchor="right",
        )

    # =========================================================================
    # Part 2: Stacked bar / onion plot (bottom half) showing nested clones
    # =========================================================================
    bar_y = -0.2

    # Draw nested rectangles (largest first)
    clone_order = ["clone_1", "clone_2", "clone_3", "clone_4"]
    # Accumulate from inside out: clone_4 is innermost
    # Actually show them as nested: clone_1 is outermost (78% CCF), clone_4 innermost (4%)

    ccf_values = {
        "clone_1": 0.78,  # DNMT3A
        "clone_2": 0.68,  # + SETBP1
        "clone_3": 0.58,  # + PTPN11
        "clone_4": 0.04,  # + IDH2
    }

    clone_labels = {
        "clone_1": "DNMT3A",
        "clone_2": "+ SETBP1",
        "clone_3": "+ PTPN11",
        "clone_4": "+ IDH2",
    }

    bar_height = 0.35
    max_width = 3.0  # width of 100%

    for cid in clone_order:
        clone_data = next(c for c in clones if c["id"] == cid)
        ccf = ccf_values[cid]
        w = ccf * max_width
        color = clone_data["color"]

        fig.add_shape(
            type="rect",
            x0=0.5 - w/2, y0=bar_y - bar_height/2,
            x1=0.5 + w/2, y1=bar_y + bar_height/2,
            fillcolor=color,
            opacity=0.5,
            line=dict(color=color, width=2),
            layer="below",
        )

        # Label at right edge
        fig.add_annotation(
            x=0.5 + w/2 + 0.05, y=bar_y,
            text=f"{clone_labels[cid]} ({ccf*100:.0f}% CCF)",
            showarrow=False,
            font=dict(size=9, color=color),
            xanchor="left",
        )

    # Normal cells bar
    fig.add_shape(
        type="rect",
        x0=0.5 - max_width/2, y0=bar_y - bar_height/2,
        x1=0.5 + max_width/2, y1=bar_y + bar_height/2,
        fillcolor="#2ECC71",
        opacity=0.2,
        line=dict(color="#2ECC71", width=2),
        layer="below",
    )
    fig.add_annotation(
        x=0.5 + max_width/2 + 0.05, y=bar_y + bar_height/3,
        text="Normal (22%)",
        showarrow=False,
        font=dict(size=9, color="#2ECC71"),
        xanchor="left",
    )

    fig.add_annotation(
        x=0.5, y=bar_y - bar_height/2 - 0.15,
        text="<b>Nested clonal architecture (CCF estimates)</b>",
        showarrow=False,
        font=dict(size=11),
    )

    # =========================================================================
    # Part 3: VAF bar chart (far bottom)
    # =========================================================================
    vaf_y_base = -1.2
    vaf_data = [
        ("DNMT3A\nR882H", 0.39, "#3498DB"),
        ("SETBP1\nG870S", 0.34, "#E67E22"),
        ("PTPN11\nE76Q", 0.29, "#E74C3C"),
        ("IDH2\nR140Q", 0.02, "#9B59B6"),
    ]

    bar_width = 0.3
    bar_spacing = 0.55
    start_x = 0.5 - (len(vaf_data) - 1) * bar_spacing / 2

    for i, (label, vaf, color) in enumerate(vaf_data):
        x_center = start_x + i * bar_spacing
        bar_h = vaf * 1.5  # scale factor

        fig.add_shape(
            type="rect",
            x0=x_center - bar_width/2, y0=vaf_y_base,
            x1=x_center + bar_width/2, y1=vaf_y_base + bar_h,
            fillcolor=color,
            opacity=0.8,
            line=dict(color=color, width=1),
        )

        # VAF label on top of bar
        fig.add_annotation(
            x=x_center, y=vaf_y_base + bar_h + 0.03,
            text=f"<b>{vaf*100:.0f}%</b>",
            showarrow=False,
            font=dict(size=11, color=color),
        )

        # Gene label below bar
        fig.add_annotation(
            x=x_center, y=vaf_y_base - 0.08,
            text=label,
            showarrow=False,
            font=dict(size=9, color="#2C3E50"),
            yanchor="top",
        )

    fig.add_annotation(
        x=0.5, y=vaf_y_base - 0.28,
        text="<b>Variant Allele Frequencies (VAF)</b>",
        showarrow=False,
        font=dict(size=11),
    )

    # =========================================================================
    # Layout
    # =========================================================================
    fig.update_layout(
        title=dict(
            text=(
                "Clonal Evolution Model: DNMT3A R882H -> SETBP1 G870S -> "
                "PTPN11 E76Q -> IDH2 R140Q<br>"
                "<sub>Based on variant allele frequencies | "
                "CCF = Cancer Cell Fraction (2 x VAF for heterozygous diploid)</sub>"
            ),
            font=dict(size=13),
        ),
        showlegend=True,
        legend=dict(x=1.02, y=1, bordercolor="black", borderwidth=1,
                    font=dict(size=9)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.5, 3.0]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.6, 3.2]),
        plot_bgcolor="white",
        width=1000,
        height=1100,
        margin=dict(l=20, r=200, t=80, b=40),
    )

    fig.write_html(output_path, include_plotlyjs=True)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = datetime.now()
    print("=" * 70)
    print("PATHWAY AND SIGNALING NETWORK ANALYSIS")
    print("Patient: DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + Monosomy 7")
    print("=" * 70)
    print()

    # 1. Pathway mapping
    print("[1/5] Building pathway mapping...")
    pathway_mapping = build_pathway_mapping()
    print(f"  Mapped {len(pathway_mapping)} mutations to pathways")

    # 2. Convergence analysis
    print("[2/5] Running convergence analysis...")
    convergences = build_convergence_analysis()
    print(f"  Identified {len(convergences)} convergence axes")
    for c in convergences:
        print(f"    - {c['title']}")

    # 3. Network graph
    print("[3/5] Building signaling network...")
    network = build_network()
    print(f"  Network: {len(network['nodes'])} nodes, {len(network['edges'])} edges")

    # Identify convergence nodes
    convergence_nodes = [n for n in network["nodes"] if n["category"] == "convergence"]
    print(f"  Convergence nodes: {', '.join(n['id'] for n in convergence_nodes)}")

    # Save network JSON
    network_json_path = os.path.join(RESULTS_DIR, "pathway_network.json")
    # Renamed: the comprehensive mapping goes to pathway_mapping.json
    # The network topology goes into pathway_network.json as well (combined)

    # 4. Therapeutic targets
    print("[4/5] Analyzing therapeutic vulnerabilities...")
    targets = build_therapeutic_targets()
    print(f"  Ranked {len(targets)} therapeutic targets")
    for t in targets:
        print(f"    {t['rank']}. {t['target']} "
              f"({t['n_pathways']} pathways, {t['clinical_evidence']} evidence)")

    # 5. Clonal evolution
    print("[5/5] Building clonal evolution model...")
    clonal_model = build_clonal_model()
    clones = clonal_model["clonal_tree"]["clones"]
    print(f"  Clonal tree: {len(clones)} clones")
    for c in clones:
        pct = c['fraction'] * 100
        muts = ', '.join(c['mutations']) if c['mutations'] else 'None'
        print(f"    {c['name']}: {pct:.0f}% ({muts})")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    print()
    print("Saving results...")

    # pathway_mapping.json — comprehensive output
    full_output = {
        "metadata": {
            "analysis": "Pathway and signaling network analysis",
            "patient_profile": {
                k: {"gene": v["gene"], "variant": v["variant"],
                     "vaf": v["vaf"], "type": v["type"]}
                for k, v in MUTATIONS.items()
            },
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": round((datetime.now() - t0).total_seconds(), 1),
        },
        "pathway_mapping": pathway_mapping,
        "convergence_analysis": convergences,
        "network": network,
        "clonal_model": clonal_model,
    }

    mapping_path = os.path.join(RESULTS_DIR, "pathway_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"  -> {mapping_path}")

    # pathway_network.html — interactive Plotly network
    network_html_path = os.path.join(RESULTS_DIR, "pathway_network.html")
    create_network_visualization(network, network_html_path)
    print(f"  -> {network_html_path}")

    # clonal_tree.html — interactive clonal tree
    clonal_html_path = os.path.join(RESULTS_DIR, "clonal_tree.html")
    create_clonal_tree_visualization(clonal_model, clonal_html_path)
    print(f"  -> {clonal_html_path}")

    # therapeutic_targets.md
    targets_md_path = os.path.join(RESULTS_DIR, "therapeutic_targets.md")
    write_therapeutic_targets_md(targets, targets_md_path)
    print(f"  -> {targets_md_path}")

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    elapsed = (datetime.now() - t0).total_seconds()
    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()

    print("KEY FINDINGS:")
    print()
    print("1. EPIGENETIC TRIPLE HIT (DNMT3A + IDH2 + EZH2/chr7):")
    print("   - DNMT3A R882H causes focal hypomethylation (self-renewal genes)")
    print("   - IDH2 R140Q causes global hypermethylation (differentiation block)")
    print("   - EZH2 haploinsufficiency (monosomy 7) impairs H3K27me3 silencing")
    print("   - Net: paradoxical epigenetic chaos, unique chromatin landscape")
    print()

    print("2. RAS/PP2A DUAL ACTIVATION (PTPN11 + SETBP1):")
    print("   - PTPN11 E76Q activates RAS at the top of the MAPK cascade")
    print("   - SETBP1 G870S inhibits PP2A, preventing ERK dephosphorylation")
    print("   - Result: feed-forward loop — ERK activated AND cannot be deactivated")
    print("   - Both also converge on AKT (SHP2 via PI3K, PP2A loss directly)")
    print()

    print("3. HOXA SELF-RENEWAL CONVERGENCE (DNMT3A + SETBP1):")
    print("   - DNMT3A R882H: epigenetic de-repression of HOXA cluster")
    print("   - SETBP1 G870S: direct transcriptional activation of HOXA9/10")
    print("   - Dual activation -> strong self-renewal signal")
    print()

    print("4. MYC AS CENTRAL HUB:")
    print("   - Activated/stabilized by 4 independent mechanisms")
    print("   - DNMT3A (enhancer hypomethylation), SETBP1 (PP2A-mediated stabilization)")
    print("   - PTPN11 (ERK-S62 phosphorylation), IDH2 (indirect)")
    print()

    print("5. CLONAL ARCHITECTURE:")
    print("   - Founder: DNMT3A R882H (39% VAF, ~78% CCF)")
    print("   - Early expansion: + SETBP1 G870S (34% VAF, ~68% CCF)")
    print("   - Dominant driver: + PTPN11 E76Q (29% VAF, ~58% CCF)")
    print("   - Late subclone: + IDH2 R140Q (2% VAF, ~4% CCF)")
    print("   - Monosomy 7: timing uncertain, likely early (co-founding or clone_1)")
    print()

    print("6. TOP THERAPEUTIC TARGETS:")
    for t in targets[:5]:
        print(f"   {t['rank']}. {t['target']} ({t['clinical_evidence']} evidence)")
    print()

    print(f"Runtime: {elapsed:.1f}s")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
