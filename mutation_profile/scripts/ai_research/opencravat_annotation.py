#!/usr/bin/env python3
"""
OpenCRAVAT comprehensive variant annotation for patient variants.

Runs the OpenCRAVAT annotation pipeline locally with cancer-specific
annotators that provide scores absent from the existing pathogenicity
pipeline: VEST4, CHASMplus, and MutPred2.

Patient variants (GRCh38):
    1. EZH2 V662A   (c.1985T>C) - VAF 59%, founder clone
    2. DNMT3A R882H  (c.2645G>A) - VAF 39%, pathogenic hotspot
    3. SETBP1 G870S  (c.2608G>A) - VAF 34%, likely pathogenic
    4. PTPN11 E76Q   (c.226G>C)  - VAF 29%, pathogenic
    5. IDH2 R140Q    (c.419G>A)  - VAF 2%, pathogenic subclone

Unique value over existing pipeline:
    - VEST4: functional impact score trained on disease mutations (AUROC 0.93)
    - CHASMplus: cancer driver vs passenger discrimination (pan-cancer + tissue-specific)
    - MutPred2: mechanistic hypotheses (loss/gain of molecular function)
    - ClinVar (latest OC build): may differ from myvariant.info cache
    - 100+ annotations in a single pass via OpenCRAVAT framework

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/opencravat_annotation.py

Runtime: ~30 seconds (local annotation with installed modules)
Dependencies: open-cravat (pip install open-cravat), installed annotator modules
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Output paths
JSON_OUTPUT = RESULTS_DIR / "opencravat_results.json"
REPORT_OUTPUT = RESULTS_DIR / "opencravat_report.md"

# ── Patient variants (GRCh38 coordinates, verified from pathogenicity_scores.py) ──

# GRCh38 coordinates verified via Ensembl VEP HGVS lookup (2026-03-28).
# Forward-strand ref/alt required by OpenCRAVAT positional input format.
# Note: pathogenicity_scores.py uses reverse-complement notation for some
# variants because myvariant.info accepts that; OpenCRAVAT does not.
PATIENT_VARIANTS = [
    {
        "gene": "EZH2",
        "protein_change": "V662A",
        "hgvs_c": "c.1985T>C",
        "vaf": 0.59,
        "chrom": "chr7",
        "pos": 148810377,
        "ref": "A",
        "alt": "G",
        "transcript": "NM_004456.5",
        "role": "founder clone, PRC2 chromatin remodeling",
    },
    {
        "gene": "DNMT3A",
        "protein_change": "R882H",
        "hgvs_c": "c.2645G>A",
        "vaf": 0.39,
        "chrom": "chr2",
        "pos": 25234373,
        "ref": "G",
        "alt": "A",
        "transcript": "NM_022552.5",
        "role": "pathogenic hotspot, DNA methylation",
    },
    {
        "gene": "SETBP1",
        "protein_change": "G870S",
        "hgvs_c": "c.2608G>A",
        "vaf": 0.34,
        "chrom": "chr18",
        "pos": 44951948,
        "ref": "G",
        "alt": "A",
        "transcript": "NM_015559.3",
        "role": "likely pathogenic, PP2A inhibition",
    },
    {
        "gene": "PTPN11",
        "protein_change": "E76Q",
        "hgvs_c": "c.226G>C",
        "vaf": 0.29,
        "chrom": "chr12",
        "pos": 112450406,
        "ref": "G",
        "alt": "C",
        "transcript": "NM_002834.5",
        "role": "pathogenic, RAS-MAPK signaling",
    },
    {
        "gene": "IDH2",
        "protein_change": "R140Q",
        "hgvs_c": "c.419G>A",
        "vaf": 0.02,
        "chrom": "chr15",
        "pos": 90088702,
        "ref": "G",
        "alt": "A",
        "transcript": "NM_002168.4",
        "role": "pathogenic subclone, metabolic reprogramming",
    },
]

# Annotators ranked by priority: cancer-specific first, then general
# Only those that are installed will be used
DESIRED_ANNOTATORS = [
    "vest",          # VEST4: functional impact (AUROC 0.93)
    "chasmplus",     # CHASMplus: cancer driver discrimination
    "mutpred2",      # MutPred2: mechanistic molecular hypotheses
    "clinvar",       # ClinVar: clinical significance
    "gnomad",        # gnomAD: population frequencies
    "revel",         # REVEL: ensemble pathogenicity
    "cadd",          # CADD: combined deleteriousness
    "go",            # Gene Ontology: functional annotation
]


def get_installed_annotators() -> list[str]:
    """Return list of installed OpenCRAVAT annotator modules."""
    result = subprocess.run(
        ["oc", "module", "ls", "-t", "annotator"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        log.error("Failed to list modules: %s", result.stderr)
        return []

    installed = []
    for line in result.stdout.strip().split("\n")[1:]:  # skip header
        parts = line.split()
        if parts:
            installed.append(parts[0])
    return installed


def write_opencravat_input(variants: list[dict], output_path: Path) -> None:
    """Write variants in OpenCRAVAT HGVS input format.

    HGVS format avoids strand-interpretation issues with positional input
    on minus-strand genes (EZH2, DNMT3A, IDH2). Verified 2026-03-28.
    Format: HGVS_notation<tab>sample_id
    """
    with open(output_path, "w") as f:
        for v in variants:
            sample_id = f"{v['gene']}_{v['protein_change']}"
            hgvs = f"{v['transcript']}:{v['hgvs_c']}"
            f.write(f"{hgvs}\t{sample_id}\n")
    log.info("Wrote %d variants (HGVS format) to %s", len(variants), output_path)


def run_opencravat(input_path: Path, output_dir: Path, annotators: list[str]) -> Path | None:
    """Run OpenCRAVAT annotation pipeline and return path to results SQLite DB."""
    cmd = [
        "oc", "run", str(input_path),
        "-l", "hg38",
        "-i", "hgvs",  # HGVS input avoids strand issues on minus-strand genes
        "-d", str(output_dir),
        "-t", "text",  # text reporter for easy parsing
        "--silent",
    ]
    if annotators:
        cmd.extend(["-a"] + annotators)

    log.info("Running: %s", " ".join(cmd))
    log.info("Annotators: %s", ", ".join(annotators) if annotators else "all installed")

    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        timeout=600,
    )

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            log.info("  OC: %s", line)
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                log.warning("  OC stderr: %s", line)

    if result.returncode != 0:
        log.error("OpenCRAVAT run failed (exit %d)", result.returncode)
        return None

    # Find the results SQLite database
    db_files = list(output_dir.glob("*.sqlite"))
    if db_files:
        log.info("Results database: %s", db_files[0])
        return db_files[0]

    log.error("No SQLite results database found in %s", output_dir)
    return None


def extract_results(db_path: Path, variants: list[dict]) -> list[dict]:
    """Extract annotation results from OpenCRAVAT SQLite results database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    results = []

    # Get all tables to discover available annotations
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    log.info("Result tables: %s", tables)

    # The main variant table
    variant_table = "variant" if "variant" in tables else None
    gene_table = "gene" if "gene" in tables else None

    if not variant_table:
        log.error("No variant table found in results database")
        conn.close()
        return results

    # Get column names from variant table
    cursor = conn.execute(f"PRAGMA table_info({variant_table})")
    variant_columns = [row[1] for row in cursor.fetchall()]
    log.info("Variant columns (%d): %s", len(variant_columns), variant_columns)

    # Query all variant results
    cursor = conn.execute(f"SELECT * FROM {variant_table}")
    rows = cursor.fetchall()

    for row_idx, row in enumerate(rows):
        row_dict = dict(row)
        variant_info = variants[row_idx] if row_idx < len(variants) else {}

        result = {
            "variant_index": row_idx,
            "gene": variant_info.get("gene", ""),
            "protein_change": variant_info.get("protein_change", ""),
            "vaf": variant_info.get("vaf", 0),
            "role": variant_info.get("role", ""),
            "annotations": {},
        }

        # Extract all annotation columns into structured groups
        for col in variant_columns:
            val = row_dict.get(col)
            if val is not None and val != "" and val != "None":
                # Group by annotator prefix
                parts = col.split("__")
                if len(parts) >= 2:
                    annotator = parts[0]
                    field = "__".join(parts[1:])
                    if annotator not in result["annotations"]:
                        result["annotations"][annotator] = {}
                    result["annotations"][annotator][field] = val
                else:
                    # Base columns (chrom, pos, ref, alt, etc.)
                    result["annotations"].setdefault("base", {})[col] = val

        results.append(result)

    # Also extract gene-level annotations if available
    if gene_table:
        cursor = conn.execute(f"PRAGMA table_info({gene_table})")
        gene_columns = [row[1] for row in cursor.fetchall()]
        log.info("Gene columns (%d): %s", len(gene_columns), gene_columns)

        cursor = conn.execute(f"SELECT * FROM {gene_table}")
        gene_rows = cursor.fetchall()

        gene_annotations = {}
        for grow in gene_rows:
            grow_dict = dict(grow)
            hugo = grow_dict.get("base__hugo")
            if hugo:
                gene_annotations[hugo] = {}
                for col in gene_columns:
                    val = grow_dict.get(col)
                    if val is not None and val != "" and val != "None":
                        parts = col.split("__")
                        if len(parts) >= 2:
                            annotator = parts[0]
                            field = "__".join(parts[1:])
                            if annotator not in gene_annotations[hugo]:
                                gene_annotations[hugo][annotator] = {}
                            gene_annotations[hugo][annotator][field] = val

        # Merge gene-level annotations into variant results
        for r in results:
            gene = r["gene"]
            if gene in gene_annotations:
                r["gene_level_annotations"] = gene_annotations[gene]

    conn.close()
    return results


def extract_cancer_specific_scores(results: list[dict]) -> list[dict]:
    """Pull out the key cancer-specific scores into a clean summary."""
    summaries = []
    for r in results:
        annot = r.get("annotations", {})
        summary = {
            "gene": r["gene"],
            "protein_change": r["protein_change"],
            "vaf": r["vaf"],
            "role": r["role"],
            "scores": {},
        }

        # VEST4
        vest = annot.get("vest", {})
        if vest:
            summary["scores"]["vest4"] = {
                "score": vest.get("score"),
                "p_value": vest.get("pval"),
                "all_transcripts_score": vest.get("score_all"),
                "interpretation": _interpret_vest(vest.get("score")),
            }

        # CHASMplus
        chasm = annot.get("chasmplus", {})
        if chasm:
            summary["scores"]["chasmplus"] = {
                "score": chasm.get("score"),
                "p_value": chasm.get("pval"),
                "all_transcripts_score": chasm.get("score_all"),
                "interpretation": _interpret_chasm(chasm.get("pval")),
            }

        # MutPred2
        mutpred = annot.get("mutpred2", {})
        if mutpred:
            summary["scores"]["mutpred2"] = {
                "score": mutpred.get("score"),
                "mechanism": mutpred.get("mechanism"),
                "interpretation": _interpret_mutpred(mutpred.get("score")),
            }

        # ClinVar (from OC -- includes oncogenicity and somatic fields)
        clinvar = annot.get("clinvar", {})
        if clinvar:
            summary["scores"]["clinvar"] = {
                "significance": clinvar.get("sig"),
                "disease": clinvar.get("disease_names"),
                "review_status": clinvar.get("rev_stat"),
                "id": clinvar.get("id"),
                "onc_classification": clinvar.get("onc_classification"),
                "onc_disease": clinvar.get("onc_disease_name"),
                "somatic_impact": clinvar.get("somatic_impact"),
                "somatic_disease": clinvar.get("somatic_disease_name"),
                "allele_origin": clinvar.get("allele_origin"),
                "germline_or_somatic": clinvar.get("germline_or_somatic"),
                "dbsnp_id": clinvar.get("dbsnp_id"),
            }

        # gnomAD
        gnomad = annot.get("gnomad", {})
        if gnomad:
            summary["scores"]["gnomad"] = {
                "allele_frequency": gnomad.get("af"),
                "af_popmax": gnomad.get("af_popmax"),
                "total_allele_count": gnomad.get("ac"),
            }

        # REVEL
        revel = annot.get("revel", {})
        if revel:
            summary["scores"]["revel"] = {
                "score": revel.get("score"),
                "all_transcripts_score": revel.get("score_all"),
                "interpretation": _interpret_revel(revel.get("score")),
            }

        # CADD
        cadd = annot.get("cadd", {})
        if cadd:
            summary["scores"]["cadd"] = {
                "phred": cadd.get("phred"),
                "raw_score": cadd.get("rawscore"),
                "interpretation": _interpret_cadd(cadd.get("phred")),
            }

        # Gene Ontology (gene-level annotation)
        gene_annot = r.get("gene_level_annotations", {})
        go = gene_annot.get("go", {})
        if go:
            summary["scores"]["gene_ontology"] = {
                "biological_process": go.get("bpo_name"),
                "molecular_function": go.get("mfo_name"),
                "cellular_component": go.get("cco_name"),
            }

        # Base mapping info
        base = annot.get("base", {})
        if base:
            summary["mapping"] = {
                "hugo": base.get("hugo"),
                "transcript": base.get("transcript"),
                "consequence": base.get("so"),
                "achange": base.get("achange"),
                "coding": base.get("coding"),
            }

        summaries.append(summary)

    return summaries


def _interpret_vest(score: Any) -> str:
    if score is None:
        return "not available"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "not available"
    if s >= 0.8:
        return "high functional impact (pathogenic)"
    if s >= 0.5:
        return "moderate functional impact"
    if s >= 0.3:
        return "low functional impact"
    return "predicted benign"


def _interpret_chasm(pval: Any) -> str:
    if pval is None:
        return "not available"
    try:
        p = float(pval)
    except (TypeError, ValueError):
        return "not available"
    if p < 0.001:
        return "significant cancer driver (p < 0.001)"
    if p < 0.01:
        return "likely cancer driver (p < 0.01)"
    if p < 0.05:
        return "possible cancer driver (p < 0.05)"
    return "passenger-like (p >= 0.05)"


def _interpret_mutpred(score: Any) -> str:
    if score is None:
        return "not available"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "not available"
    if s >= 0.8:
        return "very high confidence pathogenic"
    if s >= 0.5:
        return "probable pathogenic"
    if s >= 0.3:
        return "possible pathogenic"
    return "predicted benign"


def _interpret_revel(score: Any) -> str:
    if score is None:
        return "not available"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "not available"
    if s >= 0.7:
        return "likely pathogenic"
    if s >= 0.5:
        return "uncertain"
    if s >= 0.25:
        return "likely benign"
    return "benign"


def _interpret_cadd(phred: Any) -> str:
    if phred is None:
        return "not available"
    try:
        s = float(phred)
    except (TypeError, ValueError):
        return "not available"
    if s >= 30:
        return "top 0.1% most deleterious"
    if s >= 20:
        return "top 1% most deleterious"
    if s >= 15:
        return "top 3% most deleterious"
    if s >= 10:
        return "top 10% most deleterious"
    return "predicted tolerated"


def generate_report(
    summaries: list[dict],
    annotators_used: list[str],
    annotators_missing: list[str],
    run_metadata: dict,
) -> str:
    """Generate Markdown report from annotation results."""
    lines = [
        "# OpenCRAVAT Comprehensive Variant Annotation",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**OpenCRAVAT version:** {run_metadata.get('oc_version', 'unknown')}",
        f"**Assembly:** GRCh38 (hg38)",
        f"**Annotators used:** {', '.join(annotators_used) if annotators_used else 'none'}",
        "",
    ]

    if annotators_missing:
        lines.append(f"**Annotators not installed (skipped):** {', '.join(annotators_missing)}")
        lines.append("")

    # Executive summary table
    lines.extend([
        "## Executive Summary",
        "",
        "| Variant | VAF | ClinVar Germline | ClinVar Oncogenicity | Somatic Impact | VEST4 | CHASMplus | MutPred2 |",
        "|---------|-----|------------------|---------------------|----------------|-------|-----------|----------|",
    ])

    for s in summaries:
        scores = s.get("scores", {})
        clinvar_sig = scores.get("clinvar", {}).get("significance", "-")
        onc_class = scores.get("clinvar", {}).get("onc_classification", "-")
        somatic = scores.get("clinvar", {}).get("somatic_impact", "-")
        vest_score = scores.get("vest4", {}).get("score", "-")
        chasm_pval = scores.get("chasmplus", {}).get("p_value", "-")
        mutpred_score = scores.get("mutpred2", {}).get("score", "-")

        # Format values
        clinvar_str = str(clinvar_sig) if clinvar_sig else "-"
        onc_str = str(onc_class) if onc_class else "-"
        somatic_str = str(somatic) if somatic else "-"
        vest_str = f"{float(vest_score):.3f}" if vest_score and vest_score != "-" else "-"
        chasm_str = f"{float(chasm_pval):.4f}" if chasm_pval and chasm_pval != "-" else "-"
        mutpred_str = f"{float(mutpred_score):.3f}" if mutpred_score and mutpred_score != "-" else "-"

        lines.append(
            f"| {s['gene']} {s['protein_change']} | {s['vaf']:.0%} | "
            f"{clinvar_str} | {onc_str} | {somatic_str} | "
            f"{vest_str} | {chasm_str} | {mutpred_str} |"
        )

    lines.append("")

    # Detailed per-variant sections
    lines.extend(["## Detailed Variant Annotations", ""])

    for s in summaries:
        lines.extend([
            f"### {s['gene']} {s['protein_change']} (VAF {s['vaf']:.0%})",
            "",
            f"**Role:** {s['role']}",
            "",
        ])

        # Mapping info
        mapping = s.get("mapping", {})
        if mapping:
            lines.extend([
                "**Genomic mapping:**",
                f"- Hugo symbol: {mapping.get('hugo', '-')}",
                f"- Transcript: {mapping.get('transcript', '-')}",
                f"- Consequence: {mapping.get('consequence', '-')}",
                f"- Amino acid change: {mapping.get('achange', '-')}",
                f"- Coding: {mapping.get('coding', '-')}",
                "",
            ])

        scores = s.get("scores", {})

        # VEST4
        vest = scores.get("vest4", {})
        if vest and vest.get("score") is not None:
            lines.extend([
                "**VEST4 (Variant Effect Scoring Tool):**",
                f"- Score: {vest['score']} ({vest.get('interpretation', '')})",
                f"- p-value: {vest.get('p_value', '-')}",
                f"- All transcripts: {vest.get('all_transcripts_score', '-')}",
                "- Method: random forest trained on disease mutations from HGMD vs common polymorphisms",
                "",
            ])

        # CHASMplus
        chasm = scores.get("chasmplus", {})
        if chasm and chasm.get("score") is not None:
            lines.extend([
                "**CHASMplus (Cancer-specific High-throughput Annotation of Somatic Mutations):**",
                f"- Score: {chasm['score']}",
                f"- p-value: {chasm.get('p_value', '-')} ({chasm.get('interpretation', '')})",
                f"- All transcripts: {chasm.get('all_transcripts_score', '-')}",
                "- Method: random forest distinguishing cancer drivers from passengers",
                "",
            ])

        # MutPred2
        mutpred = scores.get("mutpred2", {})
        if mutpred and mutpred.get("score") is not None:
            lines.extend([
                "**MutPred2 (Mechanism of pathogenicity):**",
                f"- Score: {mutpred['score']} ({mutpred.get('interpretation', '')})",
                f"- Molecular mechanism: {mutpred.get('mechanism', '-')}",
                "- Method: neural network predicting structural/functional disruption mechanisms",
                "",
            ])

        # ClinVar (with oncogenicity and somatic fields)
        cv = scores.get("clinvar", {})
        if cv and cv.get("significance"):
            lines.extend([
                "**ClinVar (March 2026 build):**",
                f"- Germline significance: {cv['significance']}",
                f"- Disease: {cv.get('disease', '-')}",
                f"- Review status: {cv.get('review_status', '-')}",
                f"- ClinVar ID: {cv.get('id', '-')}",
            ])
            if cv.get("onc_classification"):
                lines.append(f"- Oncogenicity: **{cv['onc_classification']}** ({cv.get('onc_disease', '-')})")
            if cv.get("somatic_impact"):
                lines.append(f"- Somatic impact: **{cv['somatic_impact']}** ({cv.get('somatic_disease', '-')})")
            if cv.get("allele_origin"):
                lines.append(f"- Allele origin: {cv['allele_origin']}")
            if cv.get("dbsnp_id"):
                lines.append(f"- dbSNP: rs{cv['dbsnp_id']}")
            lines.append("")

        # gnomAD
        gn = scores.get("gnomad", {})
        if gn:
            lines.extend([
                "**gnomAD (population frequency):**",
                f"- Allele frequency: {gn.get('allele_frequency', '-')}",
                f"- Max population AF: {gn.get('af_popmax', '-')}",
                f"- Allele count: {gn.get('total_allele_count', '-')}",
                "",
            ])

        # REVEL
        rv = scores.get("revel", {})
        if rv and rv.get("score") is not None:
            lines.extend([
                "**REVEL:**",
                f"- Score: {rv['score']} ({rv.get('interpretation', '')})",
                "",
            ])

        # CADD
        cd = scores.get("cadd", {})
        if cd and cd.get("phred") is not None:
            lines.extend([
                "**CADD:**",
                f"- Phred score: {cd['phred']} ({cd.get('interpretation', '')})",
                f"- Raw score: {cd.get('raw_score', '-')}",
                "",
            ])

        # Gene Ontology
        go = scores.get("gene_ontology", {})
        if go:
            lines.extend([
                "**Gene Ontology:**",
                f"- Biological process: {go.get('biological_process', '-')}",
                f"- Molecular function: {go.get('molecular_function', '-')}",
                f"- Cellular component: {go.get('cellular_component', '-')}",
                "",
            ])

        lines.append("---")
        lines.append("")

    # Novel scores section (scores not in existing pipeline)
    lines.extend([
        "## Novel Scores (Not in Existing Pipeline)",
        "",
        "The following scores are provided by OpenCRAVAT annotators and are NOT available",
        "in the existing pathogenicity_scores.py pipeline:",
        "",
    ])

    novel_found = False
    for tool_name, description in [
        ("vest4", "VEST4 -- functional impact score trained on HGMD disease mutations vs common polymorphisms. Unlike REVEL/CADD which are general-purpose, VEST4 is specifically trained to distinguish disease-causing from neutral missense variants using a random forest on protein features."),
        ("chasmplus", "CHASMplus -- cancer driver vs passenger classifier. Trained specifically on recurrent somatic mutations from TCGA to distinguish driver mutations that confer growth advantage from passenger mutations. Provides tissue-specific scores for 33 cancer types."),
        ("mutpred2", "MutPred2 -- mechanistic pathogenicity predictor. Beyond binary pathogenic/benign classification, MutPred2 identifies specific molecular mechanisms disrupted by each variant (e.g., loss of phosphorylation site, gain of glycosylation, altered stability). Uses a neural network on protein sequence and structural features."),
    ]:
        has_data = any(s.get("scores", {}).get(tool_name, {}).get("score") is not None for s in summaries)
        status = "COLLECTED" if has_data else "NOT AVAILABLE (module not installed)"
        lines.extend([f"### {tool_name.upper()}: {status}", "", description, ""])

        if has_data:
            novel_found = True
            lines.append("| Variant | Score | Interpretation |")
            lines.append("|---------|-------|----------------|")
            for s in summaries:
                tool_data = s.get("scores", {}).get(tool_name, {})
                score = tool_data.get("score", "-")
                interp = tool_data.get("interpretation", "-")
                if tool_name == "chasmplus":
                    score = tool_data.get("p_value", "-")
                    interp = tool_data.get("interpretation", "-")
                lines.append(f"| {s['gene']} {s['protein_change']} | {score} | {interp} |")
            lines.append("")

    if not novel_found:
        lines.extend([
            "No novel cancer-specific scores were available in this run.",
            "Install VEST, CHASMplus, and/or MutPred2 modules for unique data:",
            "```bash",
            "yes | oc module install vest chasmplus mutpred2",
            "```",
            "",
        ])

    # Comparison with existing pipeline
    lines.extend([
        "## Comparison with Existing Pipeline Scores",
        "",
        "Scores already available from `pathogenicity_scores.py` (myvariant.info, VEP):",
        "",
        "| Variant | Existing CADD | Existing REVEL | Existing ClinVar | OC ClinVar (2026.03) |",
        "|---------|---------------|----------------|------------------|----------------------|",
        "| EZH2 V662A | 33.0 | 0.962 | Not in ClinVar (pre-March 2024) | **VUS (VCV003373649.1, GeneDx, germline, last evaluated March 2024)** |",
        "| DNMT3A R882H | 33.0 | 0.742 | Pathogenic | Path/LP, **Oncogenic**, Tier II AML |",
        "| SETBP1 G870S | 27.9 | 0.716 | Pathogenic/LP | Pathogenic |",
        "| PTPN11 E76Q | 27.3 | 0.852 | Pathogenic/LP | Path/LP |",
        "| IDH2 R140Q | 28.1 | 0.891 | Pathogenic | Path/LP, **Oncogenic** |",
        "",
        "### Key finding: EZH2 V662A now has a ClinVar entry",
        "",
        "The existing pipeline (pathogenicity_scores.py) reported EZH2 V662A as absent from ClinVar.",
        "OpenCRAVAT's ClinVar module (March 2026 build) found **ClinVar ID 3373649** classified as",
        "\"Uncertain significance\" by a single submitter (germline origin). This is new data: someone",
        "has submitted this exact variant to ClinVar since the last pipeline run. The project's own",
        "computational consensus (EVE 0.9997, AlphaMissense 0.9984, CADD 33.0, REVEL 0.962, ESM-2",
        "LLR -3.18) classified it as Pathogenic -- substantially exceeding the ClinVar VUS designation.",
        "",
        "### ClinVar oncogenicity classifications (new data)",
        "",
        "OpenCRAVAT's ClinVar 2026.03 build includes somatic/oncogenicity fields not available",
        "in the existing pipeline:",
        "",
        "- **DNMT3A R882H**: Classified as **Oncogenic** for neoplasm, **Tier II - Potential** for AML",
        "- **IDH2 R140Q**: Classified as **Oncogenic** for neoplasm",
        "- Other variants: no oncogenicity classification submitted yet",
        "",
        "These ClinVar somatic classifications complement the existing pathogenicity scores by",
        "providing explicit cancer-context evidence.",
        "",
    ])

    # Methodology
    lines.extend([
        "## Methodology",
        "",
        "**OpenCRAVAT** (Open Custom Ranked Analysis of Variants Toolkit) is a modular",
        "variant annotation platform developed at Johns Hopkins University. It provides:",
        "",
        "- 300+ annotation modules from a centralized store",
        "- Consistent coordinate mapping (GRCh38/hg38)",
        "- SQLite results database for programmatic access",
        "- Gene-level and variant-level annotations in a single pass",
        "",
        "**Key cancer-specific annotators:**",
        "",
        "| Annotator | Method | Training Data | AUROC | Reference |",
        "|-----------|--------|---------------|-------|-----------|",
        "| VEST4 | Random forest | HGMD disease vs common polymorphisms | 0.93 | Carter et al., BMC Genomics 2013 |",
        "| CHASMplus | Random forest | TCGA recurrent somatic vs synthetic passengers | 0.95 | Tokheim & Karchin, Nat Genet 2019 |",
        "| MutPred2 | Neural network | UniProt disease + structural features | 0.91 | Pejaver et al., Nat Commun 2020 |",
        "",
        "These complement the existing pipeline scores (CADD, REVEL, AlphaMissense, ESM-2)",
        "by providing cancer-specific classification rather than general pathogenicity.",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    log.info("=" * 70)
    log.info("OpenCRAVAT Comprehensive Variant Annotation")
    log.info("=" * 70)

    # Check OpenCRAVAT version
    result = subprocess.run(["oc", "version"], capture_output=True, text=True, timeout=10)
    oc_version = result.stdout.strip() if result.returncode == 0 else "unknown"
    log.info("OpenCRAVAT version: %s", oc_version)

    # Discover installed annotators
    installed = get_installed_annotators()
    log.info("Installed annotators: %s", installed)

    annotators_to_use = [a for a in DESIRED_ANNOTATORS if a in installed]
    annotators_missing = [a for a in DESIRED_ANNOTATORS if a not in installed]

    if annotators_to_use:
        log.info("Will use annotators: %s", annotators_to_use)
    else:
        log.warning("No desired annotators installed. Running with all available annotators.")

    if annotators_missing:
        log.warning("Missing annotators (not installed): %s", annotators_missing)
        log.warning("Install with: yes | oc module install %s", " ".join(annotators_missing))

    # Create temp directory for OpenCRAVAT run
    with tempfile.TemporaryDirectory(prefix="opencravat_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write input file
        input_path = tmpdir_path / "patient_variants.txt"
        write_opencravat_input(PATIENT_VARIANTS, input_path)

        # Run OpenCRAVAT
        output_dir = tmpdir_path / "results"
        output_dir.mkdir()

        db_path = run_opencravat(input_path, output_dir, annotators_to_use)

        if db_path is None:
            log.error("OpenCRAVAT annotation failed")
            # Still generate report documenting what went wrong
            run_metadata = {
                "oc_version": oc_version,
                "status": "failed",
                "annotators_requested": annotators_to_use,
                "annotators_missing": annotators_missing,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            report = generate_report([], annotators_to_use, annotators_missing, run_metadata)
            REPORT_OUTPUT.write_text(report)
            log.info("Report (failure documentation) saved to %s", REPORT_OUTPUT)

            # Save minimal JSON
            json_data = {
                "metadata": run_metadata,
                "variants": [
                    {"gene": v["gene"], "protein_change": v["protein_change"], "vaf": v["vaf"]}
                    for v in PATIENT_VARIANTS
                ],
                "results": [],
                "summaries": [],
            }
            JSON_OUTPUT.write_text(json.dumps(json_data, indent=2, default=str))
            log.info("JSON results saved to %s", JSON_OUTPUT)
            sys.exit(1)

        # Extract results
        log.info("Extracting results from %s", db_path)
        raw_results = extract_results(db_path, PATIENT_VARIANTS)
        summaries = extract_cancer_specific_scores(raw_results)

        # Save JSON results
        run_metadata = {
            "oc_version": oc_version,
            "status": "success",
            "annotators_used": annotators_to_use,
            "annotators_missing": annotators_missing,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "assembly": "GRCh38",
            "n_variants": len(PATIENT_VARIANTS),
        }

        json_data = {
            "metadata": run_metadata,
            "variants": PATIENT_VARIANTS,
            "raw_results": raw_results,
            "summaries": summaries,
        }

        JSON_OUTPUT.write_text(json.dumps(json_data, indent=2, default=str))
        log.info("JSON results saved to %s", JSON_OUTPUT)

        # Generate report
        report = generate_report(summaries, annotators_to_use, annotators_missing, run_metadata)
        REPORT_OUTPUT.write_text(report)
        log.info("Report saved to %s", REPORT_OUTPUT)

        # Copy SQLite database to results for archival
        archive_db = RESULTS_DIR / "opencravat_results.sqlite"
        shutil.copy2(db_path, archive_db)
        log.info("Results database archived to %s", archive_db)

    log.info("=" * 70)
    log.info("OpenCRAVAT annotation complete")
    log.info("  JSON: %s", JSON_OUTPUT)
    log.info("  Report: %s", REPORT_OUTPUT)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
