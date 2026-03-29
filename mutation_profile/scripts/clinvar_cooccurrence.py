#!/usr/bin/env python3
"""
clinvar_cooccurrence.py -- Search ClinVar for co-occurrence of patient's 4 target mutations.

Queries the NCBI ClinVar database via E-utilities API for:
  1. DNMT3A R882H
  2. IDH2 R140Q
  3. SETBP1 G870S
  4. PTPN11 E76Q

For each variant:
  - Count ClinVar entries (total submissions)
  - Retrieve clinical significance classifications
  - Get submitter details and conditions
  - Check for any cross-references to co-occurring mutations

Also searches for broader gene-level pathogenic/likely pathogenic variants
in hematological conditions to find any submissions mentioning multiple
target genes.

Inputs:
    - NCBI E-utilities API (remote)
    - NCBI_API_KEY from .env file

Outputs:
    - mutation_profile/results/cross_database/clinvar_cooccurrence.json
    - mutation_profile/results/cross_database/clinvar_cooccurrence_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/clinvar_cooccurrence.py

Runtime: ~30-60 seconds (API rate-limited)
Dependencies: requests, python-dotenv
"""

import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # mutation_profile/
REPO_ROOT = PROJECT_ROOT.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "cross_database"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load API key
# ---------------------------------------------------------------------------
load_dotenv(REPO_ROOT / ".env")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
if not NCBI_API_KEY:
    log.warning("NCBI_API_KEY not found in .env -- rate limited to 3 req/sec without key")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_VARIANTS = [
    {
        "gene": "DNMT3A",
        "protein_change": "R882H",
        "hgvs_p": "p.Arg882His",
        "hgvs_c": "c.2645G>A",
        "search_terms": [
            "DNMT3A[gene] AND R882H",
            "DNMT3A[gene] AND p.Arg882His",
            "DNMT3A[gene] AND c.2645G>A",
        ],
    },
    {
        "gene": "IDH2",
        "protein_change": "R140Q",
        "hgvs_p": "p.Arg140Gln",
        "hgvs_c": "c.419G>A",
        "search_terms": [
            "IDH2[gene] AND R140Q",
            "IDH2[gene] AND p.Arg140Gln",
            "IDH2[gene] AND c.419G>A",
        ],
    },
    {
        "gene": "SETBP1",
        "protein_change": "G870S",
        "hgvs_p": "p.Gly870Ser",
        "hgvs_c": "c.2608G>A",
        "search_terms": [
            "SETBP1[gene] AND G870S",
            "SETBP1[gene] AND p.Gly870Ser",
            "SETBP1[gene] AND c.2608G>A",
        ],
    },
    {
        "gene": "PTPN11",
        "protein_change": "E76Q",
        "hgvs_p": "p.Glu76Gln",
        "hgvs_c": "c.226G>C",
        "search_terms": [
            "PTPN11[gene] AND E76Q",
            "PTPN11[gene] AND p.Glu76Gln",
            "PTPN11[gene] AND c.226G>C",
        ],
    },
]

HEMATOLOGICAL_CONDITIONS = [
    "myelodysplastic",
    "acute myeloid leukemia",
    "AML",
    "MDS",
    "myeloproliferative",
    "myeloid",
]

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_DELAY = 0.35  # ~3 req/sec with API key


def _api_params() -> dict:
    """Common API parameters."""
    params: dict[str, str] = {}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def _rate_limit():
    """Respect NCBI rate limits."""
    time.sleep(REQUEST_DELAY)


def esearch(db: str, term: str, retmax: int = 500) -> dict:
    """Run an E-utilities esearch and return the parsed JSON result."""
    params = _api_params()
    params.update({
        "db": db,
        "term": term,
        "retmax": str(retmax),
        "retmode": "json",
    })
    _rate_limit()
    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def esummary(db: str, ids: list[str]) -> dict:
    """Run esummary for a list of IDs. Returns parsed JSON."""
    if not ids:
        return {}
    params = _api_params()
    params.update({
        "db": db,
        "id": ",".join(ids[:200]),  # Cap at 200 per request
        "retmode": "json",
    })
    _rate_limit()
    resp = requests.get(f"{EUTILS_BASE}/esummary.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def efetch_xml(db: str, ids: list[str], rettype: str = "clinvarset") -> str:
    """Run efetch returning XML for detailed ClinVar records."""
    if not ids:
        return ""
    params = _api_params()
    params.update({
        "db": db,
        "id": ",".join(ids[:50]),  # Smaller batch for XML
        "rettype": rettype,
        "retmode": "xml",
    })
    _rate_limit()
    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def search_variant_clinvar(variant: dict) -> dict:
    """Search ClinVar for a specific variant using multiple query strategies."""
    gene = variant["gene"]
    protein = variant["protein_change"]
    log.info(f"Searching ClinVar for {gene} {protein}...")

    all_ids: set[str] = set()
    search_results: list[dict] = []

    for term in variant["search_terms"]:
        clinvar_term = f'{term} AND "clinvar"[source]' if "[source]" not in term.lower() else term
        # Search clinvar database directly
        result = esearch("clinvar", term)
        esearch_result = result.get("esearchresult", {})
        count = int(esearch_result.get("count", 0))
        ids = esearch_result.get("idlist", [])
        all_ids.update(ids)
        search_results.append({
            "query": term,
            "count": count,
            "ids_returned": len(ids),
        })
        log.info(f"  Query '{term}': {count} results, {len(ids)} IDs")

    return {
        "gene": gene,
        "protein_change": protein,
        "hgvs_p": variant["hgvs_p"],
        "hgvs_c": variant["hgvs_c"],
        "search_results": search_results,
        "unique_clinvar_ids": sorted(all_ids),
        "total_unique_records": len(all_ids),
    }


def get_variant_details(clinvar_ids: list[str]) -> list[dict]:
    """Get detailed information for ClinVar variant IDs via esummary."""
    if not clinvar_ids:
        return []

    summaries = []
    # Process in batches of 200
    for i in range(0, len(clinvar_ids), 200):
        batch = clinvar_ids[i:i + 200]
        result = esummary("clinvar", batch)
        result_data = result.get("result", {})
        for uid in batch:
            if uid in result_data and uid != "uids":
                record = result_data[uid]
                summary = {
                    "uid": uid,
                    "title": record.get("title", ""),
                    "clinical_significance": record.get("clinical_significance", {}).get(
                        "description", ""
                    ),
                    "review_status": record.get("clinical_significance", {}).get(
                        "review_status", ""
                    ),
                    "gene_sort": record.get("gene_sort", ""),
                    "protein_change": record.get("protein_change", ""),
                    "variation_set": record.get("variation_set", []),
                    "trait_set": [],
                }
                # Extract trait/condition info
                for vs in record.get("variation_set", []):
                    for vn in vs.get("variation_name", []) if isinstance(vs, dict) else []:
                        pass
                # Extract supporting submissions info
                supp = record.get("supporting_submissions", {})
                if supp:
                    summary["supporting_submissions"] = supp

                summaries.append(summary)

    return summaries


def fetch_clinvar_xml_details(clinvar_ids: list[str]) -> list[dict]:
    """Fetch detailed XML records from ClinVar to extract submission-level data."""
    if not clinvar_ids:
        return []

    detailed_records = []
    # Process in batches of 20 for XML (heavier)
    for i in range(0, len(clinvar_ids), 20):
        batch = clinvar_ids[i:i + 20]
        log.info(f"  Fetching XML details for batch {i // 20 + 1} ({len(batch)} records)...")
        xml_text = efetch_xml("clinvar", batch, rettype="variation")
        if not xml_text:
            continue

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            # Try wrapping in root element if multiple top-level elements
            try:
                root = ET.fromstring(f"<root>{xml_text}</root>")
            except ET.ParseError:
                log.warning(f"  Failed to parse XML for batch starting at {i}")
                continue

        # Parse VariationArchive or ClinVarSet records
        for record in root.iter("VariationArchive"):
            rec_data = _parse_variation_archive(record)
            if rec_data:
                detailed_records.append(rec_data)

        for record in root.iter("ClinVarSet"):
            rec_data = _parse_clinvar_set(record)
            if rec_data:
                detailed_records.append(rec_data)

    return detailed_records


def _parse_variation_archive(record: ET.Element) -> dict | None:
    """Parse a VariationArchive XML element."""
    variation_id = record.get("VariationID", "")
    variation_name = record.get("VariationName", "")
    record_type = record.get("VariationType", "")

    submissions = []
    conditions = []
    co_occurring_variants = []

    # Extract clinical assertions (submissions)
    for assertion in record.iter("ClinicalAssertion"):
        submitter_el = assertion.find(".//ClinVarAccession")
        submitter = submitter_el.get("OrgName", "") if submitter_el is not None else ""
        significance_el = assertion.find(".//Description")
        significance = significance_el.text if significance_el is not None else ""

        # Check for co-occurring variants in ObservedIn sections
        for observed in assertion.iter("ObservedIn"):
            for co_occ in observed.iter("CoOccurrence"):
                co_occurring_variants.append(ET.tostring(co_occ, encoding="unicode"))

        # Check for observed data mentioning other variants
        for comment_el in assertion.iter("Comment"):
            if comment_el.text:
                text_lower = comment_el.text.lower()
                for gene in ["dnmt3a", "idh2", "setbp1", "ptpn11"]:
                    if gene in text_lower:
                        co_occurring_variants.append(
                            f"Comment mentions {gene}: {comment_el.text[:200]}"
                        )

        submissions.append({
            "submitter": submitter,
            "significance": significance,
        })

    # Extract conditions/traits
    for trait in record.iter("TraitMapping"):
        cond_name = trait.get("ClinicalAssertionTraitName", "")
        if cond_name:
            conditions.append(cond_name)

    for trait in record.iter("Trait"):
        name_el = trait.find("Name/ElementValue")
        if name_el is not None and name_el.text:
            conditions.append(name_el.text)

    if not variation_id:
        return None

    return {
        "variation_id": variation_id,
        "variation_name": variation_name,
        "variation_type": record_type,
        "submissions": submissions,
        "conditions": list(set(conditions)),
        "co_occurring_variants": co_occurring_variants,
        "num_submissions": len(submissions),
    }


def _parse_clinvar_set(record: ET.Element) -> dict | None:
    """Parse a ClinVarSet XML element."""
    ref_assertion = record.find("ReferenceClinVarAssertion")
    if ref_assertion is None:
        return None

    cv_acc = ref_assertion.find("ClinVarAccession")
    accession = cv_acc.get("Acc", "") if cv_acc is not None else ""

    title_el = ref_assertion.find(".//Title")
    title = title_el.text if title_el is not None else ""

    submissions = []
    conditions = []
    co_occurring_variants = []

    for assertion in record.iter("ClinVarAssertion"):
        submitter_el = assertion.find(".//ClinVarSubmissionID")
        submitter = submitter_el.get("submitter", "") if submitter_el is not None else ""

        sig_el = assertion.find(".//Description")
        significance = sig_el.text if sig_el is not None else ""

        for comment_el in assertion.iter("Comment"):
            if comment_el.text:
                text_lower = comment_el.text.lower()
                for gene in ["dnmt3a", "idh2", "setbp1", "ptpn11"]:
                    if gene in text_lower:
                        co_occurring_variants.append(
                            f"Comment mentions {gene}: {comment_el.text[:200]}"
                        )

        submissions.append({
            "submitter": submitter,
            "significance": significance,
        })

    for trait in record.iter("Trait"):
        name_el = trait.find("Name/ElementValue")
        if name_el is not None and name_el.text:
            conditions.append(name_el.text)

    return {
        "accession": accession,
        "title": title,
        "submissions": submissions,
        "conditions": list(set(conditions)),
        "co_occurring_variants": co_occurring_variants,
        "num_submissions": len(submissions),
    }


def search_gene_hematological(gene: str) -> dict:
    """Search ClinVar for pathogenic variants in a gene linked to hematological conditions."""
    log.info(f"Searching {gene} + hematological conditions in ClinVar...")

    results = {}
    for condition in ["myelodysplastic", "acute myeloid leukemia", "myeloid neoplasm"]:
        term = f'{gene}[gene] AND "{condition}" AND (pathogenic[clinical significance] OR likely pathogenic[clinical significance])'
        result = esearch("clinvar", term)
        esearch_result = result.get("esearchresult", {})
        count = int(esearch_result.get("count", 0))
        ids = esearch_result.get("idlist", [])
        results[condition] = {
            "count": count,
            "ids": ids,
        }
        log.info(f"  {gene} + '{condition}' (pathogenic/LP): {count} records")

    return results


def search_cooccurrence_mentions() -> dict:
    """Search ClinVar for any records mentioning multiple target genes together."""
    log.info("Searching for co-occurrence mentions across target genes...")

    gene_pairs = [
        ("DNMT3A", "IDH2"),
        ("DNMT3A", "SETBP1"),
        ("DNMT3A", "PTPN11"),
        ("IDH2", "SETBP1"),
        ("IDH2", "PTPN11"),
        ("SETBP1", "PTPN11"),
    ]

    pair_results = {}
    for g1, g2 in gene_pairs:
        # Search for records mentioning both genes
        term = f'{g1}[gene] AND {g2}[gene]'
        result = esearch("clinvar", term)
        esearch_result = result.get("esearchresult", {})
        count = int(esearch_result.get("count", 0))
        ids = esearch_result.get("idlist", [])
        pair_key = f"{g1}+{g2}"
        pair_results[pair_key] = {
            "count": count,
            "ids": ids,
        }
        log.info(f"  {pair_key}: {count} records")

    # Triple and quadruple searches
    triple_term = "DNMT3A[gene] AND IDH2[gene] AND SETBP1[gene]"
    result = esearch("clinvar", triple_term)
    triple_count = int(result.get("esearchresult", {}).get("count", 0))
    pair_results["DNMT3A+IDH2+SETBP1"] = {
        "count": triple_count,
        "ids": result.get("esearchresult", {}).get("idlist", []),
    }
    log.info(f"  Triple (DNMT3A+IDH2+SETBP1): {triple_count}")

    quad_term = "DNMT3A[gene] AND IDH2[gene] AND SETBP1[gene] AND PTPN11[gene]"
    result = esearch("clinvar", quad_term)
    quad_count = int(result.get("esearchresult", {}).get("count", 0))
    pair_results["DNMT3A+IDH2+SETBP1+PTPN11"] = {
        "count": quad_count,
        "ids": result.get("esearchresult", {}).get("idlist", []),
    }
    log.info(f"  Quadruple (all 4): {quad_count}")

    return pair_results


def search_clinvar_variation_api(gene: str, protein_change: str) -> dict:
    """Search the ClinVar Variation API (newer REST endpoint) for a specific variant."""
    log.info(f"Searching ClinVar Variation API for {gene} {protein_change}...")

    # Use the NCBI variation services endpoint
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    # Try the specific HGVS search
    term = f'"{gene}"[gene] AND "{protein_change}"'
    params = _api_params()
    params.update({
        "db": "clinvar",
        "term": term,
        "retmax": "100",
        "retmode": "json",
    })
    _rate_limit()
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    count = int(data.get("esearchresult", {}).get("count", 0))
    ids = data.get("esearchresult", {}).get("idlist", [])

    return {
        "query": term,
        "count": count,
        "ids": ids,
    }


def generate_report(results: dict) -> str:
    """Generate the human-readable markdown report."""
    lines = [
        "# ClinVar Co-occurrence Analysis",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Database:** ClinVar (NCBI)",
        f"**API:** E-utilities (esearch, esummary, efetch)",
        "",
        "## Purpose",
        "",
        "Search ClinVar for independent classifications of each patient variant and any",
        "evidence of co-occurring mutations from the target set:",
        "- DNMT3A R882H",
        "- IDH2 R140Q",
        "- SETBP1 G870S",
        "- PTPN11 E76Q",
        "",
        "ClinVar is primarily a variant classification database, not a patient-level",
        "database. Individual submissions sometimes include phenotype and co-occurring",
        "variant information, but this is inconsistent.",
        "",
        "---",
        "",
        "## Individual Variant Results",
        "",
    ]

    for var_result in results.get("individual_variants", []):
        gene = var_result["gene"]
        protein = var_result["protein_change"]
        total = var_result["total_unique_records"]

        lines.append(f"### {gene} {protein}")
        lines.append("")
        lines.append(f"**ClinVar records found:** {total}")
        lines.append("")

        # Search query breakdown
        lines.append("| Query | Results |")
        lines.append("|-------|---------|")
        for sr in var_result.get("search_results", []):
            lines.append(f"| `{sr['query']}` | {sr['count']} |")
        lines.append("")

        # Detailed records
        details = var_result.get("details", [])
        if details:
            lines.append(f"**Detailed records ({len(details)}):**")
            lines.append("")

            # Collect unique submitters and significances
            submitters = set()
            significances = set()
            conditions = set()
            for d in details:
                for sub in d.get("submissions", []):
                    if sub.get("submitter"):
                        submitters.add(sub["submitter"])
                    if sub.get("significance"):
                        significances.add(sub["significance"])
                for c in d.get("conditions", []):
                    conditions.add(c)

            if significances:
                lines.append(f"**Clinical significance:** {', '.join(sorted(significances))}")
                lines.append("")

            if submitters:
                lines.append(f"**Submitters ({len(submitters)}):**")
                for s in sorted(submitters):
                    lines.append(f"- {s}")
                lines.append("")

            if conditions:
                lines.append(f"**Associated conditions:**")
                for c in sorted(conditions):
                    lines.append(f"- {c}")
                lines.append("")

            # Co-occurring variant mentions
            co_occ = []
            for d in details:
                co_occ.extend(d.get("co_occurring_variants", []))
            if co_occ:
                lines.append(f"**Co-occurring variant mentions ({len(co_occ)}):**")
                for mention in co_occ:
                    lines.append(f"- {mention[:300]}")
                lines.append("")
            else:
                lines.append("**Co-occurring variant mentions:** None found in submission data")
                lines.append("")

        # Summary info
        summary_details = var_result.get("summary_details", [])
        if summary_details:
            sig_counts: dict[str, int] = {}
            for sd in summary_details:
                sig = sd.get("clinical_significance", "Unknown")
                if sig:
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1
            if sig_counts:
                lines.append("**Classification summary:**")
                lines.append("")
                lines.append("| Classification | Count |")
                lines.append("|---------------|-------|")
                for sig, cnt in sorted(sig_counts.items(), key=lambda x: -x[1]):
                    lines.append(f"| {sig} | {cnt} |")
                lines.append("")

        lines.append("---")
        lines.append("")

    # Gene-level hematological results
    lines.append("## Gene-Level Hematological Queries")
    lines.append("")
    lines.append("Pathogenic/likely pathogenic variants per gene in hematological conditions:")
    lines.append("")
    lines.append("| Gene | Condition | Pathogenic/LP Count |")
    lines.append("|------|-----------|-------------------|")

    for gene, conditions_data in results.get("gene_hematological", {}).items():
        for condition, data in conditions_data.items():
            lines.append(f"| {gene} | {condition} | {data['count']} |")
    lines.append("")

    # Co-occurrence search
    lines.append("---")
    lines.append("")
    lines.append("## Co-occurrence Searches")
    lines.append("")
    lines.append("ClinVar records matching multiple target genes simultaneously:")
    lines.append("")
    lines.append("| Gene Combination | ClinVar Records |")
    lines.append("|-----------------|-----------------|")

    cooccurrence = results.get("cooccurrence_searches", {})
    for pair_key in sorted(cooccurrence.keys(), key=lambda x: len(x)):
        data = cooccurrence[pair_key]
        lines.append(f"| {pair_key} | {data['count']} |")
    lines.append("")

    # Check if any pair results have actual records
    pairs_with_records = {k: v for k, v in cooccurrence.items() if v["count"] > 0}
    if pairs_with_records:
        lines.append("### Pair Records Found")
        lines.append("")
        for pair_key, data in pairs_with_records.items():
            lines.append(f"**{pair_key}:** {data['count']} records (IDs: {', '.join(data['ids'][:20])})")
            lines.append("")
    else:
        lines.append("No ClinVar records found matching multiple target genes simultaneously.")
        lines.append("This is expected: ClinVar indexes individual variants, not co-occurrence patterns.")
        lines.append("")

    # Interpretation
    lines.append("---")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")

    total_individual = sum(
        v["total_unique_records"] for v in results.get("individual_variants", [])
    )
    lines.append(f"- **Total individual variant records across all 4 variants:** {total_individual}")
    lines.append("")

    for var_result in results.get("individual_variants", []):
        gene = var_result["gene"]
        protein = var_result["protein_change"]
        total = var_result["total_unique_records"]
        if total > 0:
            lines.append(f"- **{gene} {protein}:** {total} independent ClinVar record(s) confirm this variant is catalogued")
        else:
            lines.append(f"- **{gene} {protein}:** No ClinVar records found -- variant may not be submitted to ClinVar or uses different nomenclature")
    lines.append("")

    quad_count = cooccurrence.get("DNMT3A+IDH2+SETBP1+PTPN11", {}).get("count", 0)
    triple_count = cooccurrence.get("DNMT3A+IDH2+SETBP1", {}).get("count", 0)

    lines.append(f"- **Quadruple co-occurrence (all 4 genes):** {quad_count} ClinVar records")
    lines.append(f"- **Triple co-occurrence (DNMT3A+IDH2+SETBP1):** {triple_count} ClinVar records")
    lines.append("")
    lines.append("ClinVar does not systematically track variant co-occurrence at the patient level.")
    lines.append("The absence of co-occurrence records in ClinVar is expected and does not provide")
    lines.append("evidence for or against co-occurrence frequency. The primary value of this search")
    lines.append("is confirming how many independent laboratories have classified each variant and")
    lines.append("whether any submitters have flagged multi-gene combinations.")
    lines.append("")

    # Methodology
    lines.append("---")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("1. Each variant searched using 3 query strategies: gene+protein shorthand, gene+HGVS protein, gene+HGVS coding")
    lines.append("2. Unique ClinVar IDs deduplicated across all query strategies")
    lines.append("3. Detailed records fetched via esummary (JSON) and efetch (XML) for submission-level data")
    lines.append("4. XML records parsed for co-occurring variant mentions in comments and ObservedIn sections")
    lines.append("5. Gene-pair, triple, and quadruple searches performed to find multi-gene ClinVar records")
    lines.append("6. Pathogenic/LP variants per gene queried against hematological condition filters")
    lines.append("")
    lines.append("**API:** NCBI E-utilities (esearch, esummary, efetch)")
    lines.append(f"**Rate limit:** {REQUEST_DELAY}s between requests (with API key)")
    lines.append("")

    return "\n".join(lines)


def main():
    """Run the full ClinVar co-occurrence analysis."""
    log.info("=" * 60)
    log.info("ClinVar Co-occurrence Analysis")
    log.info("=" * 60)

    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "ClinVar (NCBI)",
        "api": "E-utilities",
        "target_variants": [
            f"{v['gene']} {v['protein_change']}" for v in TARGET_VARIANTS
        ],
        "individual_variants": [],
        "gene_hematological": {},
        "cooccurrence_searches": {},
    }

    # Step 1: Search for each individual variant
    log.info("")
    log.info("STEP 1: Individual variant searches")
    log.info("-" * 40)

    for variant in TARGET_VARIANTS:
        var_result = search_variant_clinvar(variant)

        # Get summary details for found records
        if var_result["unique_clinvar_ids"]:
            log.info(f"  Fetching esummary details for {len(var_result['unique_clinvar_ids'])} records...")
            summary_details = get_variant_details(var_result["unique_clinvar_ids"])
            var_result["summary_details"] = summary_details

            # Get XML details for co-occurrence mentions
            log.info(f"  Fetching XML details for co-occurrence analysis...")
            xml_details = fetch_clinvar_xml_details(var_result["unique_clinvar_ids"])
            var_result["details"] = xml_details
        else:
            var_result["summary_details"] = []
            var_result["details"] = []

        results["individual_variants"].append(var_result)

    # Step 2: Gene-level hematological condition searches
    log.info("")
    log.info("STEP 2: Gene-level hematological condition searches")
    log.info("-" * 40)

    for variant in TARGET_VARIANTS:
        gene = variant["gene"]
        gene_results = search_gene_hematological(gene)
        results["gene_hematological"][gene] = gene_results

    # Step 3: Co-occurrence searches
    log.info("")
    log.info("STEP 3: Co-occurrence / multi-gene searches")
    log.info("-" * 40)

    cooccurrence = search_cooccurrence_mentions()
    results["cooccurrence_searches"] = cooccurrence

    # Step 4: For any pair results with records, fetch details
    log.info("")
    log.info("STEP 4: Fetching details for co-occurrence hits")
    log.info("-" * 40)

    for pair_key, data in cooccurrence.items():
        if data["count"] > 0 and data["ids"]:
            log.info(f"  Fetching details for {pair_key} ({data['count']} records)...")
            details = get_variant_details(data["ids"])
            data["details"] = details

            # Also fetch XML for deeper analysis
            xml_details = fetch_clinvar_xml_details(data["ids"][:50])
            data["xml_details"] = xml_details

    # Generate outputs
    log.info("")
    log.info("STEP 5: Writing results")
    log.info("-" * 40)

    # Clean results for JSON serialization (remove non-serializable items)
    json_path = RESULTS_DIR / "clinvar_cooccurrence.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"  JSON: {json_path}")

    report = generate_report(results)
    report_path = RESULTS_DIR / "clinvar_cooccurrence_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"  Report: {report_path}")

    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for var_result in results["individual_variants"]:
        gene = var_result["gene"]
        protein = var_result["protein_change"]
        total = var_result["total_unique_records"]
        log.info(f"  {gene} {protein}: {total} ClinVar records")

    quad_count = cooccurrence.get("DNMT3A+IDH2+SETBP1+PTPN11", {}).get("count", 0)
    log.info(f"  Quadruple co-occurrence: {quad_count} records")
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()
