#!/usr/bin/env python3
"""
Cancer Genome Interpreter (CGI) classification for patient variants.

Queries CGI (cancergenomeinterpreter.org) resources to obtain:
    - Driver/passenger classification from the validated oncogenic mutations catalog
    - BoostDM model availability and driver gene roles from IntOGen
    - Drug response biomarkers from the CGI biomarkers database
    - Resistance detection via biomarker association fields

CGI REST API requires registration and an auth token. This script:
    1. Attempts the CGI REST API (POST /api/v1) with token from env
    2. Downloads and parses the CGI catalog of validated oncogenic mutations
    3. Downloads and parses the CGI biomarkers database (drug response)
    4. Downloads IntOGen driver gene compendium (BoostDM gene-level roles)
    5. Compiles all results per variant

Patient variants:
    1. EZH2 V662A (c.1985T>C) - VAF 59%, founder clone
    2. DNMT3A R882H (c.2645G>A) - VAF 39%, pathogenic hotspot
    3. SETBP1 G870S (c.2608G>A) - VAF 34%, likely pathogenic
    4. PTPN11 E76Q (c.226G>C) - VAF 29%, pathogenic
    5. IDH2 R140Q (c.419G>A) - VAF 2%, pathogenic subclone

Data sources:
    - CGI catalog of validated oncogenic mutations (DoCM, ClinVar, OncoKB)
    - CGI biomarkers database (1,170 drug response entries)
    - IntOGen driver gene compendium (2024-09-20 release)
    - BoostDM gene availability (IDH2, PTPN11, EZH2 have models; DNMT3A, SETBP1 do not)

Outputs:
    - mutation_profile/results/ai_research/cgi_results.json
    - mutation_profile/results/ai_research/cgi_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/cgi_classification.py

Runtime: ~30 seconds (three HTTP downloads)
Dependencies: requests
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "ai_research"

# ---------------------------------------------------------------------------
# Patient variants (GRCh37 coordinates matching CGI catalog)
# ---------------------------------------------------------------------------

# Coordinates verified via VariantValidator API 2026-03-28. See PATIENT_PROFILE.md section 3.8.
PATIENT_VARIANTS = [
    {
        "gene": "EZH2",
        "protein_change": "V662A",
        "hgvs_p": "p.V662A",
        "hgvs_c": "c.1985T>C",
        "vaf": 0.59,
        "grch37_gdna": "chr7:g.148507469A>G",
        "grch38_gdna": "chr7:g.148810377A>G",
        "transcript": "NM_004456.5",
        "ensembl_transcript": "ENST00000320356",
        "classification": "Pathogenic",
        "pathway": "PRC2 chromatin remodeling (loss-of-function)",
    },
    {
        "gene": "DNMT3A",
        "protein_change": "R882H",
        "hgvs_p": "p.R882H",
        "hgvs_c": "c.2645G>A",
        "vaf": 0.39,
        "grch37_gdna": "chr2:g.25457242C>T",
        "grch38_gdna": "chr2:g.25234373C>T",
        "transcript": "NM_022552.5",
        "ensembl_transcript": "ENST00000264709",
        "classification": "Pathogenic (known hotspot)",
        "pathway": "DNA methyltransferase (epigenetic)",
    },
    {
        "gene": "SETBP1",
        "protein_change": "G870S",
        "hgvs_p": "p.G870S",
        "hgvs_c": "c.2608G>A",
        "vaf": 0.34,
        "grch37_gdna": "chr18:g.42531913G>A",
        "grch38_gdna": "chr18:g.44951948G>A",
        "transcript": "NM_015559.3",
        "ensembl_transcript": "ENST00000649279",
        "classification": "Likely pathogenic",
        "pathway": "PP2A tumor suppressor inhibition",
    },
    {
        "gene": "PTPN11",
        "protein_change": "E76Q",
        "hgvs_p": "p.E76Q",
        "hgvs_c": "c.226G>C",
        "vaf": 0.29,
        "grch37_gdna": "chr12:g.112888210G>C",
        "grch38_gdna": "chr12:g.112450406G>C",
        "transcript": "NM_002834.5",
        "ensembl_transcript": "ENST00000351677",
        "classification": "Pathogenic",
        "pathway": "RAS-MAPK signaling (gain-of-function)",
    },
    {
        "gene": "IDH2",
        "protein_change": "R140Q",
        "hgvs_p": "p.R140Q",
        "hgvs_c": "c.419G>A",
        "vaf": 0.02,
        "grch37_gdna": "chr15:g.90631934C>T",
        "grch38_gdna": "chr15:g.90088702C>T",
        "transcript": "NM_002168.4",
        "ensembl_transcript": "ENST00000330062",
        "classification": "Pathogenic (subclone)",
        "pathway": "Metabolic (2-HG oncometabolite)",
    },
]

# CGI cancer types for our disease
CGI_CANCER_TYPES = ["AML", "MDS", "HEMATO", "MML", "CANCER"]

# ---------------------------------------------------------------------------
# CGI REST API
# ---------------------------------------------------------------------------

CGI_API_BASE = "https://www.cancergenomeinterpreter.org/api/v1"


def attempt_cgi_api(variants: list[dict]) -> dict | None:
    """Attempt CGI REST API submission. Requires CGI_EMAIL and CGI_TOKEN env vars."""
    email = os.environ.get("CGI_EMAIL", "")
    token = os.environ.get("CGI_TOKEN", "")

    if not email or not token:
        log.info(
            "CGI_EMAIL/CGI_TOKEN not set -- skipping REST API. "
            "Register at cancergenomeinterpreter.org to obtain a token."
        )
        return None

    log.info("Attempting CGI REST API submission...")

    # Build mutations file in protein change format (tab-separated)
    lines = ["protein"]
    for v in variants:
        # CGI accepts gene:protein_change format
        lines.append(f"{v['gene']}:{v['hgvs_p']}")
    mutations_content = "\n".join(lines)

    headers = {"Authorization": f"{email} {token}"}
    payload = {
        "cancer_type": "AML",
        "title": "Patient_5_variants_MDS_AML",
        "reference": "hg38",
    }

    try:
        resp = requests.post(
            CGI_API_BASE,
            headers=headers,
            files={"mutations": ("mutations.tsv", mutations_content, "text/plain")},
            data=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            job_id = resp.text.strip().strip('"')
            log.info("CGI job submitted: %s", job_id)

            # Poll for completion (max 5 minutes)
            for attempt in range(30):
                time.sleep(10)
                status_resp = requests.get(
                    f"{CGI_API_BASE}/{job_id}",
                    headers=headers,
                    timeout=30,
                )
                if status_resp.status_code == 200:
                    status = status_resp.json()
                    if status.get("status") == "Done":
                        log.info("CGI job complete, downloading results...")
                        dl_resp = requests.get(
                            f"{CGI_API_BASE}/{job_id}",
                            headers=headers,
                            params={"action": "download"},
                            timeout=60,
                        )
                        if dl_resp.status_code == 200:
                            return _parse_cgi_zip(dl_resp.content)
                    elif status.get("status") == "Error":
                        log.error("CGI job failed: %s", status)
                        return None
                    log.info("  Poll %d: status=%s", attempt + 1, status.get("status"))

            log.warning("CGI job timed out after 5 minutes")
            return None

        log.warning("CGI API returned %d: %s", resp.status_code, resp.text[:300])
        return None

    except requests.RequestException as exc:
        log.warning("CGI API request failed: %s", exc)
        return None


def _parse_cgi_zip(content: bytes) -> dict:
    """Parse CGI results ZIP file."""
    z = zipfile.ZipFile(io.BytesIO(content))
    result = {"files": z.namelist(), "alterations": [], "biomarkers": []}
    for name in z.namelist():
        if "alterations" in name.lower():
            data = z.read(name).decode("utf-8", errors="replace")
            result["alterations_raw"] = data
        elif "biomarker" in name.lower():
            data = z.read(name).decode("utf-8", errors="replace")
            result["biomarkers_raw"] = data
    return result


# ---------------------------------------------------------------------------
# CGI Catalog of Validated Oncogenic Mutations
# ---------------------------------------------------------------------------

CGI_CATALOG_URL = (
    "https://www.cancergenomeinterpreter.org/data/mutations/"
    "catalog_of_validated_oncogenic_mutations_latest.zip?ts=20180216"
)


def download_cgi_catalog() -> list[dict]:
    """Download and parse CGI catalog of validated oncogenic mutations."""
    log.info("Downloading CGI catalog of validated oncogenic mutations...")
    resp = requests.get(CGI_CATALOG_URL, timeout=60)
    resp.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    data = z.read("catalog_of_validated_oncogenic_mutations.tsv").decode("utf-8")
    lines = data.strip().split("\n")
    header = lines[0].split("\t")

    entries = []
    for line in lines[1:]:
        fields = line.split("\t")
        row = dict(zip(header, fields))
        entries.append(row)

    log.info("CGI catalog: %d validated oncogenic mutations loaded", len(entries))
    return entries


def match_catalog(entries: list[dict], variant: dict) -> dict:
    """Match a patient variant against the CGI catalog."""
    gene = variant["gene"]
    protein = variant["hgvs_p"]
    gdna = variant["grch37_gdna"]

    # Exact protein match
    exact_matches = [
        e for e in entries
        if e.get("gene") == gene and e.get("protein") == protein
    ]

    # Same-codon matches (e.g., E76K when we have E76Q)
    residue = protein.split(".")[1][:-1]  # e.g., "E76" from "p.E76Q"
    codon_matches = [
        e for e in entries
        if e.get("gene") == gene
        and e.get("protein", "").startswith(f"p.{residue}")
        and e not in exact_matches
    ]

    # All gene entries
    gene_entries = [e for e in entries if e.get("gene") == gene]

    result = {
        "exact_match": bool(exact_matches),
        "exact_entries": exact_matches,
        "codon_match": bool(codon_matches),
        "codon_entries": codon_matches,
        "gene_total_catalog_entries": len(gene_entries),
    }

    if exact_matches:
        entry = exact_matches[0]
        result["cgi_classification"] = "Validated oncogenic (driver)"
        result["cancer_type"] = entry.get("cancer_acronym", "")
        result["source"] = entry.get("source", "")
        result["context"] = entry.get("context", "")
        result["references"] = entry.get("reference", "")
    elif codon_matches:
        result["cgi_classification"] = "Codon-level validated (same residue, different substitution)"
        result["cancer_types_at_codon"] = list({
            e.get("cancer_acronym", "") for e in codon_matches
        })
        result["substitutions_at_codon"] = [
            e.get("protein", "") for e in codon_matches
        ]
        result["sources_at_codon"] = list({
            e.get("source", "") for e in codon_matches
        })
    else:
        result["cgi_classification"] = "Not in CGI validated catalog"

    return result


# ---------------------------------------------------------------------------
# CGI Biomarkers Database
# ---------------------------------------------------------------------------

CGI_BIOMARKERS_URL = "https://www.cancergenomeinterpreter.org/biomarkers"


def download_cgi_biomarkers() -> list[dict]:
    """Download CGI biomarkers from the web page JavaScript dataset."""
    log.info("Downloading CGI biomarkers database...")
    import re

    resp = requests.get(CGI_BIOMARKERS_URL, timeout=30)
    resp.raise_for_status()

    match = re.search(r"var\s+DATASET\s*=\s*(\[.*?\])\s*;", resp.text, re.DOTALL)
    if not match:
        log.warning("Could not parse CGI biomarkers DATASET from page")
        return []

    biomarkers = json.loads(match.group(1))
    log.info("CGI biomarkers: %d entries loaded", len(biomarkers))
    return biomarkers


def match_biomarkers(biomarkers: list[dict], variant: dict) -> dict:
    """Match a variant against CGI drug response biomarkers."""
    gene = variant["gene"]
    protein_change = variant["protein_change"]

    # Match by gene name in Biomarker field
    gene_matches = [
        b for b in biomarkers
        if gene in b.get("Biomarker", "")
    ]

    # Exact variant match (e.g., "R140Q" appears in biomarker string)
    exact_drug_matches = [
        b for b in gene_matches
        if protein_change in b.get("Biomarker", "")
        or f"{protein_change[0]}{protein_change[1:-1]}" in b.get("Biomarker", "")
    ]

    # "oncogenic mutation" class matches (apply to any oncogenic mutation in gene)
    oncogenic_matches = [
        b for b in gene_matches
        if "oncogenic mutation" in b.get("Biomarker", "").lower()
    ]

    # Residue-level matches (same position, different substitution)
    residue = protein_change[:-1]  # e.g., "R140" from "R140Q"
    residue_matches = [
        b for b in gene_matches
        if residue in b.get("Biomarker", "")
        and b not in exact_drug_matches
    ]

    all_relevant = exact_drug_matches + oncogenic_matches + residue_matches
    # Deduplicate by id
    seen_ids: set[int] = set()
    unique_matches: list[dict] = []
    for m in all_relevant:
        mid = m.get("id", id(m))
        if mid not in seen_ids:
            seen_ids.add(mid)
            unique_matches.append(m)

    result = {
        "drug_response_matches": len(unique_matches),
        "biomarker_entries": [],
    }

    for m in unique_matches:
        entry = {
            "biomarker": m.get("Biomarker", ""),
            "drug": m.get("Drug full name", ""),
            "association": m.get("Association", ""),
            "evidence_level": m.get("Evidence level", ""),
            "tumor_type": m.get("Primary Tumor type", ""),
            "source": m.get("Source", ""),
        }
        # Flag resistance biomarkers
        assoc = m.get("Association", "").lower()
        if "resist" in assoc:
            entry["resistance_flag"] = True
        result["biomarker_entries"].append(entry)

    # Summarize evidence levels
    levels = [m.get("Evidence level", "") for m in unique_matches]
    result["highest_evidence_level"] = _highest_evidence(levels)
    result["has_fda_approved"] = "FDA guidelines" in levels
    result["has_clinical_trials"] = any("trial" in l.lower() for l in levels)
    result["has_preclinical"] = "Pre-clinical" in levels

    return result


def _highest_evidence(levels: list[str]) -> str:
    """Return the highest CGI evidence level from a list."""
    hierarchy = [
        "FDA guidelines",
        "NCCN guidelines",
        "European LeukemiaNet guidelines",
        "NCCN/CAP guidelines",
        "Late trials",
        "Early trials",
        "Case report",
        "Pre-clinical",
    ]
    for level in hierarchy:
        if level in levels:
            return level
    return levels[0] if levels else "None"


# ---------------------------------------------------------------------------
# IntOGen Driver Gene Compendium (includes BoostDM gene-level data)
# ---------------------------------------------------------------------------

INTOGEN_DRIVERS_URL = (
    "https://www.intogen.org/download?file=IntOGen-Drivers-20240920.zip"
)


def download_intogen_drivers() -> tuple[list[dict], list[dict]]:
    """Download IntOGen driver gene compendium and unfiltered drivers."""
    log.info("Downloading IntOGen driver gene compendium (2024-09-20)...")
    resp = requests.get(INTOGEN_DRIVERS_URL, timeout=60)
    resp.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(resp.content))

    compendium = []
    unfiltered = []

    for name in z.namelist():
        if "Compendium_Cancer_Genes" in name:
            data = z.read(name).decode("utf-8")
            lines = data.strip().split("\n")
            header = lines[0].split("\t")
            for line in lines[1:]:
                fields = line.split("\t")
                compendium.append(dict(zip(header, fields)))
        elif "Unfiltered_drivers" in name:
            data = z.read(name).decode("utf-8")
            lines = data.strip().split("\n")
            header = lines[0].split("\t")
            for line in lines[1:]:
                fields = line.split("\t")
                unfiltered.append(dict(zip(header, fields)))

    log.info(
        "IntOGen: %d compendium entries, %d unfiltered driver entries",
        len(compendium), len(unfiltered),
    )
    return compendium, unfiltered


def match_intogen(
    compendium: list[dict],
    unfiltered: list[dict],
    variant: dict,
) -> dict:
    """Match a variant's gene against IntOGen driver gene data."""
    gene = variant["gene"]

    # Compendium entries for this gene
    comp_entries = [e for e in compendium if e.get("SYMBOL") == gene]
    unfilt_entries = [e for e in unfiltered if e.get("SYMBOL") == gene]

    # AML/myeloid-specific entries
    myeloid_types = {"AML", "MDS", "CMML", "MPN", "MDS/MPN"}
    comp_aml = [
        e for e in comp_entries
        if e.get("CANCER_TYPE", "") in myeloid_types
    ]
    unfilt_aml = [
        e for e in unfilt_entries
        if e.get("CANCER_TYPE", "") in myeloid_types
    ]

    # BoostDM model availability
    boostdm_available = _check_boostdm_availability(gene)

    # Gene role (Act = activating/oncogene, LoF = loss-of-function/TSG)
    roles = list({e.get("ROLE", "") for e in comp_entries if e.get("ROLE")})
    aml_roles = list({e.get("ROLE", "") for e in comp_aml if e.get("ROLE")})

    # CGC (Cancer Gene Census) status
    cgc_gene = any(e.get("CGC_GENE") == "True" for e in comp_entries)
    cgc_cancer_gene = any(e.get("CGC_CANCER_GENE") == "True" for e in comp_entries)

    # Detection methods
    methods = set()
    for e in comp_entries:
        for m in e.get("METHODS", "").split(","):
            m = m.strip()
            if m:
                methods.add(m)

    # Q-values (significance)
    qvalues = []
    for e in comp_aml:
        try:
            qvalues.append(float(e.get("QVALUE_COMBINATION", "1")))
        except (ValueError, TypeError):
            pass

    result = {
        "intogen_total_cohorts": len(comp_entries),
        "intogen_aml_myeloid_cohorts": len(comp_aml),
        "driver_roles": roles,
        "driver_roles_in_aml": aml_roles,
        "cgc_gene": cgc_gene,
        "cgc_cancer_gene": cgc_cancer_gene,
        "detection_methods": sorted(methods),
        "boostdm_model_available": boostdm_available,
        "is_driver_gene": len(comp_entries) > 0,
        "best_qvalue_aml": min(qvalues) if qvalues else None,
    }

    # Summary classification
    if comp_aml:
        role_str = "/".join(sorted(set(aml_roles))) if aml_roles else "/".join(sorted(set(roles)))
        result["intogen_classification"] = (
            f"Driver gene in AML ({role_str}), "
            f"{len(comp_aml)} cohort(s), CGC={'Yes' if cgc_gene else 'No'}"
        )
    elif comp_entries:
        role_str = "/".join(sorted(set(roles)))
        result["intogen_classification"] = (
            f"Driver gene in other cancer types ({role_str}), "
            f"not significant in AML cohorts, CGC={'Yes' if cgc_gene else 'No'}"
        )
    else:
        result["intogen_classification"] = "Not in IntOGen driver gene compendium"

    return result


# Known BoostDM model availability per gene (verified 2026-03-27)
BOOSTDM_MODELS: dict[str, dict[str, Any]] = {
    "IDH2": {
        "available": True,
        "tumor_types": [
            "Any Cancer Type", "Solid cancer", "Non-solid cancer",
            "Myeloid", "Myeloid Neoplasm", "Acute Myeloid Leukemia",
        ],
        "has_aml_model": True,
    },
    "PTPN11": {
        "available": True,
        "tumor_types": [
            "Any Cancer Type", "Non-solid cancers", "Lymphoid neoplasms",
            "Non-Hodgkin Lymphoma", "Mature B-Cell Neoplasms",
            "Plasma Cell Myeloma",
        ],
        "has_aml_model": False,
    },
    "EZH2": {
        "available": True,
        "tumor_types": [
            "Solid cancers", "Brain/CNS tumors", "Miscellaneous Brain Tumors",
            "Non-solid cancers", "Lymphoid neoplasms", "Non-Hodgkin Lymphoma",
            "Diffuse Large B-Cell Lymphoma", "Malignant Lymphoma",
        ],
        "has_aml_model": False,
    },
    "DNMT3A": {
        "available": False,
        "tumor_types": [],
        "has_aml_model": False,
        "note": "No BoostDM model exists. IntOGen directs to observed mutations.",
    },
    "SETBP1": {
        "available": False,
        "tumor_types": [],
        "has_aml_model": False,
        "note": "No BoostDM model exists. IntOGen directs to observed mutations.",
    },
}


def _check_boostdm_availability(gene: str) -> dict:
    """Return known BoostDM model availability for a gene."""
    return BOOSTDM_MODELS.get(gene, {
        "available": False, "tumor_types": [], "has_aml_model": False,
    })


# ---------------------------------------------------------------------------
# Compile per-variant results
# ---------------------------------------------------------------------------

def compile_variant_result(
    variant: dict,
    catalog_result: dict,
    biomarker_result: dict,
    intogen_result: dict,
) -> dict:
    """Compile all CGI data sources into a single per-variant result."""
    # Overall driver classification combining all evidence
    classification = _overall_classification(
        variant, catalog_result, intogen_result,
    )

    return {
        "gene": variant["gene"],
        "protein_change": variant["protein_change"],
        "hgvs_c": variant["hgvs_c"],
        "vaf": variant["vaf"],
        "pathway": variant["pathway"],
        "prior_classification": variant["classification"],
        # CGI catalog
        "cgi_catalog": catalog_result,
        # Drug response biomarkers
        "drug_response": biomarker_result,
        # IntOGen driver gene data
        "intogen": intogen_result,
        # Overall CGI-derived classification
        "cgi_overall_classification": classification,
    }


def _overall_classification(
    variant: dict,
    catalog_result: dict,
    intogen_result: dict,
) -> dict:
    """Derive overall driver/passenger classification from CGI evidence."""
    gene = variant["gene"]
    is_catalog_validated = catalog_result.get("exact_match", False)
    is_codon_validated = catalog_result.get("codon_match", False)
    is_intogen_driver = intogen_result.get("is_driver_gene", False)
    boostdm = intogen_result.get("boostdm_model_available", {})
    has_boostdm = boostdm.get("available", False) if isinstance(boostdm, dict) else False

    if is_catalog_validated:
        verdict = "DRIVER (CGI validated oncogenic mutation)"
        confidence = "High"
        evidence = "Exact variant in CGI catalog of validated oncogenic mutations"
    elif is_codon_validated and is_intogen_driver:
        verdict = "DRIVER (codon-level evidence + IntOGen driver gene)"
        confidence = "High"
        evidence = (
            f"Same-residue substitutions validated in CGI; "
            f"{gene} is an IntOGen driver gene"
        )
    elif is_intogen_driver and has_boostdm:
        verdict = "PREDICTED DRIVER (IntOGen driver gene + BoostDM model exists)"
        confidence = "Moderate-High"
        evidence = (
            f"{gene} is an IntOGen driver gene with BoostDM ML model; "
            f"specific variant not in CGI catalog"
        )
    elif is_intogen_driver:
        verdict = "PREDICTED DRIVER (IntOGen driver gene, no BoostDM model)"
        confidence = "Moderate"
        evidence = f"{gene} is an IntOGen driver gene; no BoostDM ML model available"
    else:
        verdict = "UNCERTAIN (not in CGI/IntOGen databases)"
        confidence = "Low"
        evidence = "Variant and gene not found in CGI or IntOGen driver catalogs"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence_summary": evidence,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: dict) -> str:
    """Generate markdown report from compiled results."""
    ts = results["metadata"]["timestamp"]
    lines = [
        "# Cancer Genome Interpreter (CGI) Classification Report",
        "",
        f"**Generated:** {ts}",
        "**Patient:** MDS-IB2 / MDS-AML, 5 somatic driver mutations",
        "**Cancer type queried:** AML / MDS",
        "",
        "## Data Sources",
        "",
        "| Source | Description | Status |",
        "|--------|-------------|--------|",
        f"| CGI REST API | Full analysis via cancergenomeinterpreter.org | {results['metadata']['api_status']} |",
        f"| CGI Validated Mutations Catalog | {results['metadata']['catalog_entries']} curated oncogenic mutations | Downloaded |",
        f"| CGI Biomarkers Database | {results['metadata']['biomarker_entries']} drug response entries | Downloaded |",
        f"| IntOGen Driver Compendium | {results['metadata']['intogen_compendium_entries']} gene-cohort entries (2024-09-20) | Downloaded |",
        "| BoostDM | ML driver/passenger models (gene+tumor type specific) | Checked per gene |",
        "",
        "## CGI API Access",
        "",
    ]

    if results["metadata"]["api_status"] == "Requires registration":
        lines.extend([
            "The CGI REST API requires registration and an authentication token.",
            "To obtain access:",
            "",
            "1. Register at https://www.cancergenomeinterpreter.org (supports Gmail/Mozilla/Yahoo login)",
            "2. Request an API token from your profile page",
            "3. Set environment variables: `CGI_EMAIL` and `CGI_TOKEN`",
            "4. Re-run this script to get full CGI analysis (driver/passenger with BoostDM scores,",
            "   drug response matching, resistance detection)",
            "",
            "This report uses the publicly downloadable CGI catalogs and IntOGen data as fallback.",
            "",
        ])

    # Summary table
    lines.extend([
        "## Classification Summary",
        "",
        "| Gene | Variant | VAF | CGI Catalog | BoostDM Model | IntOGen Driver | Drug Response | Overall |",
        "|------|---------|-----|-------------|---------------|----------------|---------------|---------|",
    ])

    for vr in results["variants"]:
        gene = vr["gene"]
        change = vr["protein_change"]
        vaf = f"{vr['vaf']*100:.0f}%"
        catalog = "Validated" if vr["cgi_catalog"]["exact_match"] else (
            "Codon-level" if vr["cgi_catalog"]["codon_match"] else "Absent"
        )
        boostdm_info = vr["intogen"]["boostdm_model_available"]
        boostdm = "Yes" if (isinstance(boostdm_info, dict) and boostdm_info.get("available")) else "No"
        intogen = "Yes" if vr["intogen"]["is_driver_gene"] else "No"
        drugs = str(vr["drug_response"]["drug_response_matches"])
        overall = vr["cgi_overall_classification"]["verdict"].split("(")[0].strip()
        lines.append(f"| {gene} | {change} | {vaf} | {catalog} | {boostdm} | {intogen} | {drugs} | {overall} |")

    lines.append("")

    # Per-variant details
    lines.extend(["## Per-Variant Details", ""])

    for vr in results["variants"]:
        gene = vr["gene"]
        change = vr["protein_change"]
        lines.extend([
            f"### {gene} {change}",
            "",
            f"- **VAF:** {vr['vaf']*100:.0f}%",
            f"- **Pathway:** {vr['pathway']}",
            f"- **Prior classification:** {vr['prior_classification']}",
            "",
        ])

        # CGI catalog
        cat = vr["cgi_catalog"]
        lines.append("**CGI Validated Mutations Catalog:**")
        if cat["exact_match"]:
            entry = cat["exact_entries"][0]
            lines.extend([
                f"- Exact match: **Yes** (validated oncogenic driver)",
                f"- Cancer type: {entry.get('cancer_acronym', 'N/A')}",
                f"- Source: {entry.get('source', 'N/A')}",
                f"- Context: {entry.get('context', 'N/A')}",
                f"- References: {_format_refs(entry.get('reference', ''))}",
            ])
        elif cat["codon_match"]:
            lines.extend([
                f"- Exact match: No",
                f"- Codon-level match: **Yes** ({len(cat['codon_entries'])} other substitutions at same residue)",
                f"- Validated substitutions: {', '.join(cat.get('substitutions_at_codon', []))}",
                f"- Cancer types: {', '.join(cat.get('cancer_types_at_codon', []))}",
            ])
        else:
            lines.extend([
                f"- Exact match: No",
                f"- Codon-level match: No",
                f"- Gene has {cat['gene_total_catalog_entries']} other validated mutations in catalog",
            ])
        lines.append("")

        # IntOGen
        ig = vr["intogen"]
        lines.append("**IntOGen Driver Gene Status:**")
        lines.extend([
            f"- Classification: {ig['intogen_classification']}",
            f"- Total cohorts: {ig['intogen_total_cohorts']}",
            f"- AML/myeloid cohorts: {ig['intogen_aml_myeloid_cohorts']}",
            f"- Driver roles: {', '.join(ig['driver_roles']) if ig['driver_roles'] else 'N/A'}",
            f"- CGC gene: {'Yes' if ig['cgc_gene'] else 'No'}",
        ])
        if ig.get("best_qvalue_aml") is not None:
            lines.append(f"- Best q-value (AML): {ig['best_qvalue_aml']:.2e}")
        lines.append("")

        # BoostDM
        boostdm_info = ig["boostdm_model_available"]
        if isinstance(boostdm_info, dict):
            lines.append("**BoostDM ML Model:**")
            if boostdm_info.get("available"):
                lines.extend([
                    f"- Model available: **Yes**",
                    f"- Tumor types with models: {', '.join(boostdm_info.get('tumor_types', []))}",
                    f"- AML-specific model: {'Yes' if boostdm_info.get('has_aml_model') else 'No'}",
                ])
                if not boostdm_info.get("has_aml_model"):
                    lines.append(
                        "- Note: BoostDM score for AML not available; model exists for other tumor types"
                    )
            else:
                lines.extend([
                    f"- Model available: **No**",
                ])
                if boostdm_info.get("note"):
                    lines.append(f"- Note: {boostdm_info['note']}")
            lines.append("")

        # Drug response
        dr = vr["drug_response"]
        lines.append("**Drug Response Biomarkers:**")
        if dr["drug_response_matches"] > 0:
            lines.append(f"- Matches: {dr['drug_response_matches']}")
            lines.append(f"- Highest evidence: {dr['highest_evidence_level']}")
            if dr["has_fda_approved"]:
                lines.append("- **FDA-approved indication exists**")
            lines.append("")
            lines.append("| Drug | Association | Evidence | Tumor Type | Source |")
            lines.append("|------|-------------|----------|------------|--------|")
            for b in dr["biomarker_entries"]:
                drug = b["drug"]
                assoc = b["association"]
                evid = b["evidence_level"]
                tt = b["tumor_type"]
                src = b["source"][:50]
                resist = " **[RESISTANCE]**" if b.get("resistance_flag") else ""
                lines.append(f"| {drug} | {assoc}{resist} | {evid} | {tt} | {src} |")
        else:
            lines.append("- No drug response biomarkers found in CGI database")
        lines.append("")

        # Overall
        ov = vr["cgi_overall_classification"]
        lines.extend([
            "**Overall CGI Classification:**",
            f"- Verdict: **{ov['verdict']}**",
            f"- Confidence: {ov['confidence']}",
            f"- Evidence: {ov['evidence_summary']}",
            "",
            "---",
            "",
        ])

    # Clinical notes
    lines.extend([
        "## Clinical Notes",
        "",
        "### EZH2 V662A -- Tazemetostat Contraindication",
        "",
        "The CGI biomarkers database lists EZH2 gain-of-function mutations (Y641, A677, A692V) as",
        "responsive to tazemetostat (FDA-approved for follicular lymphoma). However, the patient's",
        "EZH2 V662A is a **loss-of-function** variant in the SET domain (Chase et al. 2020,",
        "PMID 32322039). Tazemetostat is **CONTRAINDICATED** for this variant -- inhibiting an",
        "already-impaired EZH2 would further suppress H3K27 methylation.",
        "",
        "### SETBP1 G870S -- Database Gap",
        "",
        "SETBP1 is absent from the CGI validated mutations catalog and has minimal representation",
        "in IntOGen (1 cohort in adrenocortical carcinoma, not myeloid). No BoostDM model exists.",
        "This reflects the relative rarity of SETBP1 mutations and their concentration in MDS/MPN",
        "overlap syndromes that are underrepresented in CGI's source databases (DoCM, OncoKB).",
        "SETBP1 G870S is a well-characterized SKI domain hotspot with established pathogenicity",
        "in the myeloid literature (Piazza et al. 2013, PMID 23832012).",
        "",
        "### PTPN11 E76Q -- Codon-Level Evidence",
        "",
        "PTPN11 E76Q is not in the CGI catalog, but E76K, E76A, E76G, and E76V are all validated",
        "as oncogenic at the same codon. All E76 substitutions activate SHP2 phosphatase through",
        "disruption of the N-SH2/PTP autoinhibitory interface. The E76Q substitution has the same",
        "biochemical mechanism and is classified as pathogenic in ClinVar.",
        "",
        "### IDH2 R140Q -- Strongest CGI Evidence",
        "",
        "IDH2 R140Q has the strongest CGI evidence: validated oncogenic mutation in the catalog",
        "(sourced from DoCM, Martelotto, OncoKB with 14+ PMIDs), plus drug response biomarkers",
        "for enasidenib (AG-221) and venetoclax. This is the only variant with a BoostDM model",
        "specifically trained for AML.",
        "",
        "## Methodology",
        "",
        "### CGI Classification Logic",
        "",
        "1. **CGI Validated Mutations Catalog**: 5,601 curated oncogenic mutations compiled from",
        "   DoCM, ClinVar, and OncoKB. Exact protein-level match = validated driver.",
        "2. **Codon-level matching**: If the exact substitution is absent but other substitutions",
        "   at the same residue are validated, this provides strong supporting evidence.",
        "3. **IntOGen Driver Gene Compendium**: Gene-level driver status across 4,478 cohort-gene",
        "   entries. Includes detection methods, driver role (Act/LoF), and CGC status.",
        "4. **BoostDM**: ML models trained per gene and tumor type for driver/passenger",
        "   classification. Available for IDH2 (including AML model), PTPN11, and EZH2.",
        "   Not available for DNMT3A or SETBP1.",
        "5. **Drug response biomarkers**: 1,170 curated biomarker-drug associations with evidence",
        "   levels (FDA guidelines, clinical trials, pre-clinical).",
        "",
        "### Limitations",
        "",
        "- CGI REST API not used (requires registration token). Full BoostDM per-variant scores",
        "  and OncodriveMut predictions are only available through the API.",
        "- CGI catalog last updated 2018-02-16. Newer validated mutations may be missing.",
        "- BoostDM scores for specific mutations require the API or web interface; this script",
        "  reports model availability only.",
        "- SETBP1 is poorly represented in all CGI data sources despite being a well-known",
        "  myeloid driver gene.",
        "",
        "### CGI API Registration",
        "",
        "To obtain full CGI analysis with per-variant BoostDM scores:",
        "",
        "1. Go to https://www.cancergenomeinterpreter.org",
        "2. Sign in via Google, Mozilla Persona, or Yahoo",
        "3. Navigate to your profile and request an API token",
        "4. Set environment variables:",
        "   ```",
        "   export CGI_EMAIL='your@email.com'",
        "   export CGI_TOKEN='your_token_here'",
        "   ```",
        "5. Re-run: `python mutation_profile/scripts/ai_research/cgi_classification.py`",
        "",
    ])

    return "\n".join(lines)


def _format_refs(ref_str: str) -> str:
    """Format CGI reference string for display (truncate long lists)."""
    refs = [r.strip() for r in ref_str.split("__") if r.strip()]
    pmids = [r for r in refs if r.startswith("PMID:")]
    if len(pmids) > 5:
        return f"{', '.join(pmids[:5])} (+{len(pmids)-5} more)"
    return ", ".join(refs) if refs else "N/A"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== CGI Classification for Patient Variants ===")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 1. Attempt CGI REST API
    api_result = attempt_cgi_api(PATIENT_VARIANTS)
    api_status = "Full results obtained" if api_result else "Requires registration"

    # 2. Download CGI catalog
    catalog_entries = download_cgi_catalog()

    # 3. Download CGI biomarkers
    biomarker_entries = download_cgi_biomarkers()

    # 4. Download IntOGen driver gene compendium
    compendium, unfiltered = download_intogen_drivers()

    # 5. Process each variant
    variant_results = []
    for v in PATIENT_VARIANTS:
        log.info("Processing %s %s...", v["gene"], v["protein_change"])

        catalog_result = match_catalog(catalog_entries, v)
        biomarker_result = match_biomarkers(biomarker_entries, v)
        intogen_result = match_intogen(compendium, unfiltered, v)

        compiled = compile_variant_result(
            v, catalog_result, biomarker_result, intogen_result,
        )
        variant_results.append(compiled)

        log.info(
            "  %s %s: catalog=%s, drugs=%d, intogen=%s",
            v["gene"], v["protein_change"],
            "validated" if catalog_result["exact_match"] else (
                "codon" if catalog_result["codon_match"] else "absent"
            ),
            biomarker_result["drug_response_matches"],
            "driver" if intogen_result["is_driver_gene"] else "absent",
        )

    # 6. Compile results
    results = {
        "metadata": {
            "timestamp": timestamp,
            "api_status": api_status,
            "api_result": api_result,
            "catalog_entries": len(catalog_entries),
            "biomarker_entries": len(biomarker_entries),
            "intogen_compendium_entries": len(compendium),
            "intogen_unfiltered_entries": len(unfiltered),
            "cancer_type": "AML / MDS-IB2",
            "patient": "5 somatic driver mutations + monosomy 7",
        },
        "variants": variant_results,
    }

    # 7. Save JSON results
    json_path = RESULTS_DIR / "cgi_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("JSON results saved to %s", json_path)

    # 8. Generate and save markdown report
    report = generate_report(results)
    report_path = RESULTS_DIR / "cgi_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Markdown report saved to %s", report_path)

    # 9. Summary
    log.info("")
    log.info("=== Summary ===")
    for vr in variant_results:
        log.info(
            "  %s %s: %s (confidence: %s)",
            vr["gene"], vr["protein_change"],
            vr["cgi_overall_classification"]["verdict"],
            vr["cgi_overall_classification"]["confidence"],
        )


if __name__ == "__main__":
    main()
