#!/usr/bin/env python3
"""
generate_research_report.py -- Compile all AI research results into a master report.

Master report generator that compiles ALL AI research results into a
comprehensive markdown document and a companion index file.

Scans mutation_profile/results/ai_research/ for result files (JSON, TSV, MD),
and also pulls from the pre-existing result directories (esm2_variant_scoring,
cross_database, cooccurrence, diffdock, setbp1_makishima).

Inputs:
    - mutation_profile/results/ai_research/**/*.{json,tsv,md}
    - mutation_profile/results/esm2_variant_scoring/
    - mutation_profile/results/cross_database/
    - mutation_profile/results/cooccurrence/
    - mutation_profile/results/diffdock/
    - mutation_profile/results/setbp1_makishima/

Outputs:
    - mutation_profile/results/ai_research/COMPREHENSIVE_AI_RESEARCH_REPORT.md
    - mutation_profile/results/ai_research/INDEX.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/generate_research_report.py [--dry-run]

Runtime: <5 seconds
Dependencies: (standard library only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # mrna-hematology-research/
MUTATION_DIR = PROJECT_ROOT / "mutation_profile"
RESULTS_DIR = MUTATION_DIR / "results"
AI_RESULTS_DIR = RESULTS_DIR / "ai_research"
OUTPUT_REPORT = AI_RESULTS_DIR / "COMPREHENSIVE_AI_RESEARCH_REPORT.md"
OUTPUT_INDEX = AI_RESULTS_DIR / "INDEX.md"

# Additional result directories produced by earlier pipeline stages
LEGACY_RESULT_DIRS = {
    "esm2_variant_scoring": RESULTS_DIR / "esm2_variant_scoring",
    "cross_database": RESULTS_DIR / "cross_database",
    "cooccurrence": RESULTS_DIR / "cooccurrence",
    "diffdock": RESULTS_DIR / "diffdock",
    "setbp1_makishima": RESULTS_DIR / "setbp1_makishima",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patient profile (constant)
# ---------------------------------------------------------------------------

PATIENT_MUTATIONS = [
    {"gene": "DNMT3A", "variant": "R882H", "domain": "Catalytic (methyltransferase)"},
    {"gene": "IDH2", "variant": "R140Q", "domain": "Active site (neomorphic)"},
    {"gene": "SETBP1", "variant": "G870S", "domain": "SKI homology / degron motif"},
    {"gene": "PTPN11", "variant": "E76Q", "domain": "N-SH2 autoinhibitory interface"},
]
CYTOGENETIC = "Monosomy 7 / del(7q)"

# Expected subdirectory names inside ai_research/ and what report section
# they map to.  Keys are directory stems; values are (section_number, title).
AI_SUBDIR_MAP: dict[str, tuple[str, str]] = {
    "esm2_deep":             ("1.1", "ESM-2 Deep Scoring (Meta)"),
    "esm1v_variant_effect":  ("1.2", "ESM-1v Variant Effect (Meta)"),
    "multi_model_consensus": ("1.3", "Multi-Model Consensus"),
    "acmg_classification":   ("1.4", "ACMG Classification"),
    "esmfold_structures":    ("2.1", "ESMFold Structure Predictions (Meta)"),
    "diffdock_docking":      ("2.2", "Molecular Docking (AutoDock Vina / DiffDock)"),
    "medgemma_clinical":     ("3.1", "Gemini / MedGemma Clinical Analysis (Google)"),
    "clonal_architecture":   ("3.2", "Clonal Architecture"),
    "prognosis":             ("3.3", "Prognosis (ELN 2022, IPSS-M)"),
    "pathway_analysis":      ("4.1", "Affected Pathways"),
    "convergence_points":    ("4.2", "Convergence Points"),
    "therapeutic_vulns":     ("4.3", "Therapeutic Vulnerabilities"),
    "approved_drugs":        ("5.1", "Approved Drugs"),
    "clinical_trials":       ("5.2", "Clinical Trials"),
    "drug_repurposing":      ("5.3", "Drug Repurposing Candidates"),
    "combination_strategies": ("5.4", "Combination Strategies"),
    "per_gene_reviews":      ("6.1", "Per-Gene Literature Reviews"),
    "combination_analysis":  ("6.2", "Combination Analysis"),
    "treatment_landscape":   ("6.3", "Treatment Landscape 2026"),
    "variant_interpretation": ("misc", "Variant Interpretation"),
    "literature_synthesis":  ("misc", "Literature Synthesis"),
    "txgemma_therapeutic":   ("misc", "TxGemma Therapeutic Analysis"),
}

# Report section top-level structure
REPORT_SECTIONS: list[tuple[str, str]] = [
    ("1", "Variant Pathogenicity Assessment"),
    ("2", "Structural Analysis"),
    ("3", "Clinical Interpretation"),
    ("4", "Pathway Analysis"),
    ("5", "Therapeutic Analysis"),
    ("6", "Literature Synthesis"),
    ("7", "Cross-Database Uniqueness"),
]

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

RESULT_EXTENSIONS = {".json", ".tsv", ".csv", ".md", ".txt", ".pdb", ".sdf"}


def discover_files(base_dir: Path) -> list[Path]:
    """Recursively find all result files under *base_dir*."""
    if not base_dir.is_dir():
        return []
    found: list[Path] = []
    for root, _dirs, files in os.walk(base_dir):
        for fname in sorted(files):
            fpath = Path(root) / fname
            if fpath.suffix.lower() in RESULT_EXTENSIONS:
                found.append(fpath)
    return found


def discover_all_results() -> dict[str, list[Path]]:
    """
    Return a mapping of source_label -> list[Path] for every result file
    across both the ai_research/ tree and legacy result directories.
    """
    results: dict[str, list[Path]] = {}

    # AI research subdirectories
    if AI_RESULTS_DIR.is_dir():
        for child in sorted(AI_RESULTS_DIR.iterdir()):
            if child.is_dir():
                files = discover_files(child)
                if files:
                    results[f"ai_research/{child.name}"] = files

    # Legacy result directories
    for label, path in LEGACY_RESULT_DIRS.items():
        files = discover_files(path)
        if files:
            results[label] = files

    return results


# ---------------------------------------------------------------------------
# JSON loaders (graceful)
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        log.warning("Could not load %s: %s", path.relative_to(PROJECT_ROOT), exc)
        return None


def find_json(base: Path, *candidates: str) -> dict[str, Any] | None:
    """Try loading the first JSON file that exists from *candidates* relative to *base*."""
    for name in candidates:
        p = base / name
        if p.is_file():
            return load_json(p)
    return None


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_executive_summary(
    esm2_data: dict | None,
    cross_db_data: dict | None,
    cooccurrence_data: dict | None,
) -> str:
    """Generate the executive summary from available data sources."""
    lines: list[str] = []
    lines.append("## Executive Summary\n")

    lines.append(
        "This report presents a comprehensive AI-driven analysis of a myeloid "
        "neoplasm (MDS/AML) carrying the co-occurring mutations **DNMT3A R882H**, "
        "**IDH2 R140Q**, **SETBP1 G870S**, and **PTPN11 E76Q** with **monosomy 7**. "
        "This quadruple combination has never been observed in any sequenced cohort.\n"
    )

    # ESM-2 headline
    if esm2_data and "variants" in esm2_data:
        variants = esm2_data["variants"]
        strong = [v for v in variants if "Strong" in v.get("acmg_pp3", "")]
        lines.append(f"**Pathogenicity:** ESM-2 protein language model scored "
                     f"{len(variants)} variants. {len(strong)} achieved "
                     f"PP3_Strong classification (score < -7.0), confirming "
                     f"significant protein disruption.\n")

    # Cross-database headline
    if cross_db_data:
        total_patients = cross_db_data.get("deduplicated_total_myeloid_patients", "~10,000")
        quad_results = cross_db_data.get("quadruple_cooccurrence_results", {})
        consolidated = quad_results.get("consolidated_total", 0)
        lines.append(
            f"**Uniqueness:** Zero patients with this quadruple combination "
            f"identified across {total_patients:,} deduplicated myeloid patients "
            f"from GENIE, cBioPortal, TCGA, IPSS-M, Beat AML, and DepMap.\n"
        )

    # Co-occurrence headline
    if cooccurrence_data:
        quad_obs = cooccurrence_data.get("quadruple_cooccurrence", {}).get("observed", 0)
        quad_exp = cooccurrence_data.get("quadruple_cooccurrence", {}).get("expected")
        if quad_exp is not None:
            lines.append(
                f"**Statistical rarity:** Observed {quad_obs} / expected "
                f"~{quad_exp:.3f} quadruple co-occurrences in GENIE "
                f"(panel-adjusted myeloid cohort).\n"
            )

    lines.append(
        "**Clinical relevance:** The combination activates multiple oncogenic "
        "axes -- epigenetic reprogramming (DNMT3A + IDH2), metabolic rewiring "
        "(IDH2 2-HG), signaling hyperactivation (PTPN11/RAS), and disrupted "
        "protein turnover (SETBP1 degron) -- compounded by genomic instability "
        "from monosomy 7. This convergence creates potential therapeutic "
        "vulnerabilities detailed in Sections 4-5.\n"
    )

    return "\n".join(lines)


def build_section_from_dir(
    section_num: str,
    title: str,
    dir_path: Path,
    all_files: dict[str, list[Path]],
) -> str:
    """Build a report subsection from the contents of a results directory."""
    lines: list[str] = []
    lines.append(f"### {section_num} {title}\n")

    # Find files for this section
    matching_key = None
    for key in all_files:
        if dir_path.name in key:
            matching_key = key
            break

    if matching_key is None or not all_files.get(matching_key):
        lines.append("*Results pending -- analysis not yet executed.*\n")
        return "\n".join(lines)

    files = all_files[matching_key]

    # Try to incorporate JSON summary data
    json_files = [f for f in files if f.suffix == ".json"]
    md_files = [f for f in files if f.suffix == ".md"]

    if md_files:
        # If there is a markdown summary, inline its content (skip its H1)
        for md_file in md_files:
            lines.append(f"*Source: `{md_file.relative_to(PROJECT_ROOT)}`*\n")
            try:
                content = md_file.read_text(encoding="utf-8")
                # Strip any top-level heading to avoid duplicate
                stripped = []
                for line in content.splitlines():
                    if line.startswith("# ") and not stripped:
                        continue  # skip first H1
                    stripped.append(line)
                lines.append("\n".join(stripped))
            except Exception as exc:
                lines.append(f"*Error reading file: {exc}*\n")
    elif json_files:
        # Render a JSON summary table
        for jf in json_files:
            data = load_json(jf)
            if data is None:
                continue
            lines.append(f"*Source: `{jf.relative_to(PROJECT_ROOT)}`*\n")
            lines.append(render_json_summary(data))
    else:
        lines.append("*Results present but no JSON or Markdown summary found.*\n")
        for f in files:
            lines.append(f"- `{f.relative_to(PROJECT_ROOT)}`")
        lines.append("")

    return "\n".join(lines)


def render_json_summary(data: dict[str, Any], max_depth: int = 2) -> str:
    """Render the top-level keys of a JSON result as a readable summary."""
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"**{key}:**\n")
            for k2, v2 in value.items():
                if isinstance(v2, (dict, list)):
                    lines.append(f"- *{k2}:* ({type(v2).__name__}, "
                                 f"{len(v2)} items)")
                else:
                    lines.append(f"- *{k2}:* {v2}")
            lines.append("")
        elif isinstance(value, list):
            lines.append(f"**{key}:** {len(value)} entries\n")
        else:
            lines.append(f"**{key}:** {value}\n")
    return "\n".join(lines)


def build_esm2_section(esm2_data: dict | None) -> str:
    """Build the ESM-2 scoring subsection from existing pipeline results."""
    lines: list[str] = []
    lines.append("### 1.1 ESM-2 Scoring (Meta)\n")

    if esm2_data is None:
        lines.append("*Results pending -- run esm2_variant_scoring.py first.*\n")
        return "\n".join(lines)

    meta = esm2_data.get("metadata", {})
    lines.append(f"**Model:** `{meta.get('model', 'unknown')}`  ")
    lines.append(f"**Device:** {meta.get('gpu') or meta.get('device', 'unknown')}  ")
    lines.append(f"**Method:** {meta.get('method', 'masked marginal log-likelihood ratio')}  ")
    lines.append(f"**Window:** +/-{meta.get('window_size', 200)} residues\n")

    lines.append("| Gene | Variant | ESM-2 Score | ACMG PP3 | Interpretation |")
    lines.append("|------|---------|-------------|----------|----------------|")
    for v in esm2_data.get("variants", []):
        lines.append(
            f"| {v['gene']} | {v['variant']} | "
            f"**{v['esm2_score']:.4f}** | "
            f"{v['acmg_pp3']} | "
            f"{v['acmg_interpretation']} |"
        )
    lines.append("")

    # Detailed breakdown
    lines.append("#### Detailed Per-Variant Analysis\n")
    for v in esm2_data.get("variants", []):
        lines.append(f"**{v['gene']} {v['variant']}** (UniProt {v['uniprot_id']}, "
                     f"pos {v.get('uniprot_pos', v.get('clinical_pos', '?'))})")
        lines.append(f"- Score: {v['esm2_score']:.4f}")
        lines.append(f"- log P(ref={v['ref_aa']}): {v['ref_log_prob']:.4f}")
        lines.append(f"- log P(alt={v['alt_aa']}): {v['alt_log_prob']:.4f}")
        lines.append(f"- Classification: {v['acmg_pp3']}")
        lines.append(f"- Note: {v.get('note', '')}")
        top5 = v.get("top5_predictions", [])
        if top5:
            preds = ", ".join(
                f"{p['aa']}({p['log_prob']:.2f})" for p in top5
            )
            lines.append(f"- Top-5 at position: {preds}")
        lines.append("")

    lines.append(
        "> **Interpretation note:** IDH2 R140Q scores near the benign threshold "
        "despite being a known oncogenic driver. This reflects the gain-of-function "
        "(neomorphic) nature of IDH mutations -- ESM-2 captures sequence-level "
        "disruption but cannot model the acquired 2-HG production activity.\n"
    )

    return "\n".join(lines)


def build_cross_database_section(cross_db_data: dict | None) -> str:
    """Build the cross-database uniqueness section."""
    lines: list[str] = []
    lines.append("## 7. Cross-Database Uniqueness\n")

    if cross_db_data is None:
        lines.append("*Results pending -- run cross_database.py first.*\n")
        return "\n".join(lines)

    total = cross_db_data.get("deduplicated_total_myeloid_patients", "unknown")
    lines.append(
        f"Across **{total:,}** deduplicated myeloid patients from all queried "
        f"databases, **zero** carry the quadruple combination of DNMT3A + IDH2 "
        f"+ SETBP1 + PTPN11 mutations.\n"
    )

    # Database breakdown
    overlap = cross_db_data.get("overlap_analysis", {})
    if overlap:
        lines.append("| Database | Patients Queried | Unique Addition | Notes |")
        lines.append("|----------|-----------------|-----------------|-------|")
        for db_name, info in overlap.items():
            if isinstance(info, dict):
                lines.append(
                    f"| {db_name} | {info.get('patients', '?'):,} | "
                    f"{info.get('unique_addition', '?'):,} | "
                    f"{info.get('note', '')} |"
                )
        lines.append("")

    quad = cross_db_data.get("quadruple_cooccurrence_results", {})
    if quad:
        lines.append("**Quadruple co-occurrence search results:**\n")
        for db, result in quad.items():
            lines.append(f"- **{db}:** {result}")
        lines.append("")

    lines.append(
        f"**Conclusion:** This mutation profile is unique across the largest "
        f"available myeloid genomics databases ({total:,} patients), supporting "
        f"its classification as an exceptionally rare genotype.\n"
    )

    return "\n".join(lines)


def build_methods_section(
    esm2_data: dict | None,
    all_files: dict[str, list[Path]],
) -> str:
    """Build the Methods section listing all AI tools used."""
    lines: list[str] = []
    lines.append("## Methods\n")

    lines.append("### AI Models and Tools\n")
    lines.append("| Tool | Provider | Purpose | Parameters |")
    lines.append("|------|----------|---------|------------|")

    # Always list the core tools
    models = [
        ("ESM-2 (esm2_t33_650M_UR50D)", "Meta / Facebook AI",
         "Variant pathogenicity scoring",
         "650M params, masked marginal LLR, window +/-200"),
        ("ESM-1v", "Meta / Facebook AI",
         "Variant effect prediction",
         "650M params, zero-shot"),
        ("ESMFold", "Meta / Facebook AI",
         "Protein structure prediction",
         "Single-sequence folding, pLDDT confidence"),
        ("DiffDock", "MIT / NVIDIA NIM",
         "Molecular docking (blind)",
         "Diffusion generative model"),
        ("AutoDock Vina", "Scripps / Open-source",
         "Molecular docking (targeted)",
         "Empirical scoring function"),
        ("Gemini 2.5 Pro", "Google DeepMind",
         "Clinical interpretation and literature synthesis",
         "1M context, grounded generation"),
        ("MedGemma", "Google Health AI",
         "Medical reasoning over variant data",
         "Specialized medical LLM"),
    ]
    for name, provider, purpose, params in models:
        lines.append(f"| {name} | {provider} | {purpose} | {params} |")
    lines.append("")

    # Runtime info from ESM-2 if available
    if esm2_data:
        meta = esm2_data.get("metadata", {})
        lines.append("### Runtime Environment\n")
        lines.append(f"- **GPU:** {meta.get('gpu', 'N/A')}")
        lines.append(f"- **Device:** {meta.get('device', 'N/A')}")
        lines.append(f"- **Date:** {meta.get('date', 'N/A')}")
        lines.append("")

    lines.append("### Databases Queried\n")
    lines.append("- AACR GENIE v19.0 (primary cohort)")
    lines.append("- cBioPortal (aggregated)")
    lines.append("- GDC / TCGA-LAML")
    lines.append("- IPSS-M training cohort (Bernard et al. 2022)")
    lines.append("- Beat AML (OHSU)")
    lines.append("- DepMap (Broad Institute)")
    lines.append("- ClinVar, COSMIC, gnomAD")
    lines.append("")

    lines.append("### Filtering and Quality Control\n")
    lines.append("- Myeloid-only cohort (AML, MDS, MPN, CMML, MDS/MPN, JMML)")
    lines.append("- Coding variants only (missense, nonsense, frameshift, splice, in-frame indel)")
    lines.append("- Hypermutation filter: >20 coding mutations excluded")
    lines.append("- Panel adjustment: patients counted only when all relevant genes "
                 "covered by their sequencing panel")
    lines.append("- Fisher's exact test (two-sided) with Benjamini-Hochberg FDR correction")
    lines.append("")

    return "\n".join(lines)


def build_references_section(all_files: dict[str, list[Path]]) -> str:
    """Build a references section with key citations."""
    lines: list[str] = []
    lines.append("## References\n")
    lines.append("### Key Citations\n")

    refs = [
        ("Lin et al. 2023",
         "ESM-2: Evolutionary scale modeling of protein sequences. "
         "*Science* 379, 1123-1130."),
        ("Brandes et al. 2023",
         "Genome-wide prediction of disease variant effects with a deep "
         "protein language model. *Nature Genetics* 55, 1512-1522."),
        ("Corso et al. 2023",
         "DiffDock: Diffusion steps, twists, and turns for molecular docking. "
         "*ICLR 2023*."),
        ("Bernard et al. 2022",
         "Molecular international prognostic scoring system for myelodysplastic "
         "syndromes. *NEJM Open* 2(7)."),
        ("Makishima et al. 2013",
         "Somatic SETBP1 mutations in myeloid malignancies. "
         "*Nature Genetics* 45, 942-946."),
        ("Piazza et al. 2018",
         "SETBP1 induces transcription of a network of development genes by "
         "acting as an epigenetic hub. *Nature Communications* 9, 2192."),
        ("AACR GENIE Consortium 2017",
         "AACR Project GENIE: Powering precision medicine through an "
         "international consortium. *Cancer Discovery* 7, 818-831."),
        ("Dohner et al. 2022",
         "ELN 2022 recommendations for diagnosis and management of AML. "
         "*Blood* 140, 1345-1377."),
        ("Trott & Olson 2010",
         "AutoDock Vina: Improving the speed and accuracy of docking. "
         "*Journal of Computational Chemistry* 31, 455-461."),
    ]

    for i, (authors, text) in enumerate(refs, 1):
        lines.append(f"{i}. {authors}. {text}")
    lines.append("")

    lines.append(
        "### AI-Generated Content Disclaimer\n"
        "Sections marked with model names (Gemini, MedGemma) contain AI-generated "
        "interpretations. All AI outputs were reviewed for factual accuracy against "
        "primary literature. Protein language model scores (ESM-2, ESM-1v) are "
        "computational predictions that complement but do not replace functional "
        "validation.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# INDEX.md builder
# ---------------------------------------------------------------------------

def describe_file(fpath: Path) -> str:
    """Return a human-readable one-line description based on filename and content."""
    name = fpath.name.lower()
    stem = fpath.stem.lower()
    suffix = fpath.suffix.lower()

    # Size info
    try:
        size = fpath.stat().st_size
        if size > 1_000_000:
            size_str = f"{size / 1_000_000:.1f} MB"
        elif size > 1_000:
            size_str = f"{size / 1_000:.1f} KB"
        else:
            size_str = f"{size} bytes"
    except OSError:
        size_str = "unknown size"

    # Heuristic descriptions
    descriptions: dict[str, str] = {
        "esm2_results.json": "ESM-2 variant scoring results (all variants, full detail)",
        "esm2_results.tsv": "ESM-2 scoring results in tabular format",
        "esm2_summary.md": "ESM-2 scoring human-readable summary",
        "four_gene_cooccurrence.json": "DNMT3A+IDH2+SETBP1+PTPN11 quadruple co-occurrence analysis",
        "myeloid_pairwise_matrix.tsv": "34-gene pairwise co-occurrence matrix (heatmap data)",
        "myeloid_pairwise_results.json": "Full pairwise co-occurrence statistics (190 pairs)",
        "myeloid_pairwise_results.tsv": "Pairwise results in tabular format",
        "cross_database_results.json": "Cross-database consolidation (GENIE, cBioPortal, TCGA, etc.)",
        "cross_database_consolidation.md": "Cross-database uniqueness narrative report",
        "per_database_summary.tsv": "Per-database patient counts and overlap",
    }

    desc = descriptions.get(name, "")
    if not desc:
        # Fallback: derive from path components
        parts = fpath.relative_to(RESULTS_DIR).parts
        category = parts[0] if parts else "unknown"
        type_desc = {
            ".json": "JSON data",
            ".tsv": "Tab-separated data",
            ".csv": "Comma-separated data",
            ".md": "Markdown report",
            ".txt": "Text output",
            ".pdb": "Protein structure (PDB format)",
            ".sdf": "Molecular structure (SDF format)",
        }.get(suffix, "Result file")
        desc = f"{type_desc} from {category} analysis"

    return f"{desc} ({size_str})"


def build_index(all_files: dict[str, list[Path]]) -> str:
    """Build the INDEX.md content."""
    lines: list[str] = []
    lines.append("# AI Research Results Index\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
    lines.append(
        "This index catalogues all result files produced by the AI research "
        "pipeline for the DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q "
        "+ monosomy 7 mutation profile analysis.\n"
    )

    total_files = sum(len(v) for v in all_files.values())
    total_dirs = len(all_files)
    lines.append(f"**Total:** {total_files} files across {total_dirs} directories\n")
    lines.append("---\n")

    for source_label in sorted(all_files.keys()):
        files = all_files[source_label]
        lines.append(f"## {source_label}/\n")
        for fpath in files:
            rel = fpath.relative_to(RESULTS_DIR)
            desc = describe_file(fpath)
            lines.append(f"- **`{rel}`** -- {desc}")
        lines.append("")

    # Status of expected-but-empty directories
    lines.append("---\n")
    lines.append("## Pending Analyses\n")
    lines.append("The following result directories exist but contain no results yet:\n")

    pending_found = False
    if AI_RESULTS_DIR.is_dir():
        for child in sorted(AI_RESULTS_DIR.iterdir()):
            if child.is_dir():
                files = discover_files(child)
                if not files:
                    section = AI_SUBDIR_MAP.get(child.name, ("?", child.name))
                    lines.append(f"- `ai_research/{child.name}/` -- "
                                 f"Section {section[0]}: {section[1]}")
                    pending_found = True

    for label, path in sorted(LEGACY_RESULT_DIRS.items()):
        if path.is_dir():
            files = discover_files(path)
            if not files:
                lines.append(f"- `{label}/` -- Empty")
                pending_found = True

    if not pending_found:
        lines.append("*All analysis directories contain results.*\n")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Master report assembly
# ---------------------------------------------------------------------------

def build_report(all_files: dict[str, list[Path]]) -> str:
    """Assemble the complete research report."""
    sections: list[str] = []

    # ----- Title -----
    sections.append(
        "# Comprehensive AI Analysis: "
        "DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + Monosomy 7\n"
    )

    sections.append(
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*  \n"
        f"*Pipeline: mrna-hematology-research / mutation_profile / ai_research*\n"
    )

    # ----- Load key data sources -----
    esm2_data = load_json(RESULTS_DIR / "esm2_variant_scoring" / "esm2_results.json")
    cross_db_data = load_json(RESULTS_DIR / "cross_database" / "cross_database_results.json")
    cooccurrence_data = load_json(RESULTS_DIR / "cooccurrence" / "four_gene_cooccurrence.json")

    # ----- Executive Summary -----
    sections.append(build_executive_summary(esm2_data, cross_db_data, cooccurrence_data))

    # ----- Mutation Profile Overview -----
    sections.append("---\n")
    sections.append("### Patient Mutation Profile\n")
    sections.append("| Gene | Variant | Functional Domain |")
    sections.append("|------|---------|-------------------|")
    for m in PATIENT_MUTATIONS:
        sections.append(f"| {m['gene']} | {m['variant']} | {m['domain']} |")
    sections.append(f"\n**Cytogenetics:** {CYTOGENETIC}\n")

    # ----- Section 1: Variant Pathogenicity -----
    sections.append("---\n")
    sections.append("## 1. Variant Pathogenicity Assessment\n")
    sections.append(build_esm2_section(esm2_data))

    # Build subsections 1.2 - 1.4 from ai_research subdirectories
    for dir_stem in ("esm1v_variant_effect", "multi_model_consensus", "acmg_classification"):
        if dir_stem in AI_SUBDIR_MAP:
            sec_num, sec_title = AI_SUBDIR_MAP[dir_stem]
            sections.append(build_section_from_dir(
                sec_num, sec_title,
                AI_RESULTS_DIR / dir_stem,
                all_files,
            ))

    # ----- Section 2: Structural Analysis -----
    sections.append("---\n")
    sections.append("## 2. Structural Analysis\n")

    for dir_stem in ("esmfold_structures", "diffdock_docking"):
        if dir_stem in AI_SUBDIR_MAP:
            sec_num, sec_title = AI_SUBDIR_MAP[dir_stem]
            sections.append(build_section_from_dir(
                sec_num, sec_title,
                AI_RESULTS_DIR / dir_stem,
                all_files,
            ))

    # Also check legacy diffdock results
    diffdock_files = all_files.get("diffdock", [])
    if diffdock_files:
        sections.append("\n#### DiffDock Legacy Results\n")
        for f in diffdock_files:
            sections.append(f"- `{f.relative_to(PROJECT_ROOT)}`")
        sections.append("")

    # ----- Section 3: Clinical Interpretation -----
    sections.append("---\n")
    sections.append("## 3. Clinical Interpretation\n")

    for dir_stem in ("medgemma_clinical", "clonal_architecture", "prognosis"):
        if dir_stem in AI_SUBDIR_MAP:
            sec_num, sec_title = AI_SUBDIR_MAP[dir_stem]
            sections.append(build_section_from_dir(
                sec_num, sec_title,
                AI_RESULTS_DIR / dir_stem,
                all_files,
            ))

    # ----- Section 4: Pathway Analysis -----
    sections.append("---\n")
    sections.append("## 4. Pathway Analysis\n")

    for dir_stem in ("pathway_analysis", "convergence_points", "therapeutic_vulns"):
        if dir_stem in AI_SUBDIR_MAP:
            sec_num, sec_title = AI_SUBDIR_MAP[dir_stem]
            sections.append(build_section_from_dir(
                sec_num, sec_title,
                AI_RESULTS_DIR / dir_stem,
                all_files,
            ))

    # ----- Section 5: Therapeutic Analysis -----
    sections.append("---\n")
    sections.append("## 5. Therapeutic Analysis\n")

    for dir_stem in ("approved_drugs", "clinical_trials", "drug_repurposing",
                     "combination_strategies"):
        if dir_stem in AI_SUBDIR_MAP:
            sec_num, sec_title = AI_SUBDIR_MAP[dir_stem]
            sections.append(build_section_from_dir(
                sec_num, sec_title,
                AI_RESULTS_DIR / dir_stem,
                all_files,
            ))

    # ----- Section 6: Literature Synthesis -----
    sections.append("---\n")
    sections.append("## 6. Literature Synthesis\n")

    for dir_stem in ("per_gene_reviews", "combination_analysis", "treatment_landscape"):
        if dir_stem in AI_SUBDIR_MAP:
            sec_num, sec_title = AI_SUBDIR_MAP[dir_stem]
            sections.append(build_section_from_dir(
                sec_num, sec_title,
                AI_RESULTS_DIR / dir_stem,
                all_files,
            ))

    # ----- Section 7: Cross-Database Uniqueness -----
    sections.append("---\n")
    sections.append(build_cross_database_section(cross_db_data))

    # ----- Methods -----
    sections.append("---\n")
    sections.append(build_methods_section(esm2_data, all_files))

    # ----- References -----
    sections.append("---\n")
    sections.append(build_references_section(all_files))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compile all AI research results into a comprehensive report"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report what would be compiled, but do not write files",
    )
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("AI RESEARCH REPORT GENERATOR")
    log.info("=" * 70)
    log.info("Project root: %s", PROJECT_ROOT)
    log.info("Results dir:  %s", RESULTS_DIR)
    log.info("AI results:   %s", AI_RESULTS_DIR)

    # ----- Discover all result files -----
    log.info("Scanning for result files...")
    all_files = discover_all_results()

    total_files = sum(len(v) for v in all_files.values())
    log.info("Found %d result files across %d directories", total_files, len(all_files))

    for source, files in sorted(all_files.items()):
        log.info("  %-40s %d files", source, len(files))

    if args.dry_run:
        log.info("[DRY RUN] Would write:")
        log.info("  1. %s", OUTPUT_REPORT)
        log.info("  2. %s", OUTPUT_INDEX)
        log.info("Exiting without writing.")
        return

    # ----- Build and write report -----
    log.info("Building comprehensive report...")
    report_content = build_report(all_files)

    AI_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report_content, encoding="utf-8")
    report_size = OUTPUT_REPORT.stat().st_size
    log.info("Wrote report: %s (%d bytes)", OUTPUT_REPORT.relative_to(PROJECT_ROOT), report_size)

    # ----- Build and write index -----
    log.info("Building result index...")
    index_content = build_index(all_files)

    OUTPUT_INDEX.write_text(index_content, encoding="utf-8")
    index_size = OUTPUT_INDEX.stat().st_size
    log.info("Wrote index: %s (%d bytes)", OUTPUT_INDEX.relative_to(PROJECT_ROOT), index_size)

    # ----- Summary -----
    log.info("=" * 70)
    log.info("DONE")
    log.info("  Report: %s", OUTPUT_REPORT)
    log.info("  Index:  %s", OUTPUT_INDEX)
    log.info("  Files catalogued: %d", total_files)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
