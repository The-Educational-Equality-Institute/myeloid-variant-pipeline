#!/usr/bin/env python3
"""
pecan_stjude_search.py -- Search St. Jude PeCan / ProteinPaint for patient target mutations.

Queries the St. Jude Pediatric Cancer (PeCan) data portal and ProteinPaint API
for mutation data on the patient's 4 target genes. PeCan contains ~9,000+
pediatric cancer samples with whole-genome/exome sequencing data.

NOTE: PeCan is a PEDIATRIC cancer database. Our patient is an adult. Results
from this search are included for completeness of the cross-database survey
only, not for direct clinical comparison.

APIs attempted (in order of priority):
  1. ProteinPaint API (https://proteinpaint.stjude.org/api/) -- newer, more stable
  2. PeCan GraphQL API (https://pecan.stjude.cloud/api/graphql)
  3. PeCan REST search (https://pecan.stjude.cloud/api/search)

Target mutations:
  - DNMT3A R882H
  - IDH2 R140Q
  - SETBP1 G870S
  - PTPN11 E76Q

Outputs:
    - mutation_profile/results/cross_database/pecan_stjude_results.json
    - mutation_profile/results/cross_database/pecan_stjude_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/pecan_stjude_search.py

Runtime: ~30-60 seconds (network-dependent)
Dependencies: requests
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # mutation_profile/
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
# Configuration
# ---------------------------------------------------------------------------
TARGET_GENES = ["DNMT3A", "IDH2", "SETBP1", "PTPN11"]
EXACT_VARIANTS = {
    "DNMT3A": "R882H",
    "IDH2": "R140Q",
    "SETBP1": "G870S",
    "PTPN11": "E76Q",
}

# Myeloid disease terms to filter for in PeCan results
MYELOID_DISEASE_TERMS = {
    "aml", "acute myeloid leukemia", "mds", "myelodysplastic",
    "myeloproliferative", "mpn", "cmml", "jmml", "myeloid",
    "chronic myelomonocytic", "juvenile myelomonocytic",
}

# API endpoints
PROTEINPAINT_API = "https://proteinpaint.stjude.org/api/"
PECAN_GRAPHQL_API = "https://pecan.stjude.cloud/api/graphql"
PECAN_SEARCH_API = "https://pecan.stjude.cloud/api/search"

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 2  # seconds between API calls

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "mrna-hematology-research/1.0 (academic research)",
    "Accept": "application/json",
    "Content-Type": "application/json",
})


# ---------------------------------------------------------------------------
# ProteinPaint API queries
# ---------------------------------------------------------------------------

def query_proteinpaint_gene(gene: str) -> dict[str, Any]:
    """Query ProteinPaint API for a single gene lookup in the pediatric dataset."""
    payload = {
        "genome": "hg38",
        "dslabel": "pediatric",
        "querykey": "singlegenelookup",
        "gene": gene,
    }
    try:
        log.info("ProteinPaint: querying gene %s ...", gene)
        resp = SESSION.post(PROTEINPAINT_API, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() if resp.text.strip() else {}
        return {"status": "ok", "data": data, "http_status": resp.status_code}
    except requests.exceptions.HTTPError as exc:
        log.warning("ProteinPaint gene query for %s returned HTTP %s", gene, exc.response.status_code)
        return {"status": "http_error", "http_status": exc.response.status_code, "error": str(exc)}
    except requests.exceptions.RequestException as exc:
        log.warning("ProteinPaint gene query for %s failed: %s", gene, exc)
        return {"status": "error", "error": str(exc)}
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("ProteinPaint gene query for %s returned non-JSON: %s", gene, exc)
        return {"status": "parse_error", "error": str(exc), "raw_text": resp.text[:500]}


def query_proteinpaint_genelookup(gene: str) -> dict[str, Any]:
    """Alternative ProteinPaint query using genelookup key."""
    payload = {
        "genome": "hg38",
        "dslabel": "pediatric",
        "querykey": "genelookup",
        "input": gene,
    }
    try:
        log.info("ProteinPaint genelookup: querying %s ...", gene)
        resp = SESSION.post(PROTEINPAINT_API, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() if resp.text.strip() else {}
        return {"status": "ok", "data": data, "http_status": resp.status_code}
    except requests.exceptions.RequestException as exc:
        log.warning("ProteinPaint genelookup for %s failed: %s", gene, exc)
        return {"status": "error", "error": str(exc)}
    except (json.JSONDecodeError, ValueError) as exc:
        return {"status": "parse_error", "error": str(exc), "raw_text": resp.text[:500]}


def query_proteinpaint_mutations(gene: str) -> dict[str, Any]:
    """Query ProteinPaint for mutation data on a gene using the mds3 endpoint."""
    payload = {
        "genome": "hg38",
        "dslabel": "pediatric",
        "querykey": "snvindel",
        "gene": gene,
    }
    try:
        log.info("ProteinPaint snvindel: querying %s ...", gene)
        resp = SESSION.post(PROTEINPAINT_API, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() if resp.text.strip() else {}
        return {"status": "ok", "data": data, "http_status": resp.status_code}
    except requests.exceptions.RequestException as exc:
        log.warning("ProteinPaint snvindel for %s failed: %s", gene, exc)
        return {"status": "error", "error": str(exc)}
    except (json.JSONDecodeError, ValueError) as exc:
        return {"status": "parse_error", "error": str(exc), "raw_text": resp.text[:500]}


def query_proteinpaint_mds3(gene: str) -> dict[str, Any]:
    """Query ProteinPaint mds3 endpoint for mutation track data."""
    payload = {
        "genome": "hg38",
        "dslabel": "pediatric",
        "embedder": "none",
        "for": "mds3",
        "gene": gene,
    }
    try:
        log.info("ProteinPaint mds3: querying %s ...", gene)
        resp = SESSION.post(PROTEINPAINT_API, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() if resp.text.strip() else {}
        return {"status": "ok", "data": data, "http_status": resp.status_code}
    except requests.exceptions.RequestException as exc:
        log.warning("ProteinPaint mds3 for %s failed: %s", gene, exc)
        return {"status": "error", "error": str(exc)}
    except (json.JSONDecodeError, ValueError) as exc:
        return {"status": "parse_error", "error": str(exc), "raw_text": resp.text[:500]}


# ---------------------------------------------------------------------------
# PeCan GraphQL API queries
# ---------------------------------------------------------------------------

def query_pecan_graphql_gene(gene: str) -> dict[str, Any]:
    """Query PeCan GraphQL API for gene-level mutation data."""
    query = """
    {
        gene(name: "%s") {
            name
            mutations {
                protein_change
                disease
                count
            }
        }
    }
    """ % gene
    try:
        log.info("PeCan GraphQL: querying gene %s ...", gene)
        resp = SESSION.post(
            PECAN_GRAPHQL_API,
            json={"query": query},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json() if resp.text.strip() else {}
        return {"status": "ok", "data": data, "http_status": resp.status_code}
    except requests.exceptions.HTTPError as exc:
        log.warning("PeCan GraphQL for %s returned HTTP %s", gene, exc.response.status_code)
        return {"status": "http_error", "http_status": exc.response.status_code, "error": str(exc)}
    except requests.exceptions.RequestException as exc:
        log.warning("PeCan GraphQL for %s failed: %s", gene, exc)
        return {"status": "error", "error": str(exc)}
    except (json.JSONDecodeError, ValueError) as exc:
        return {"status": "parse_error", "error": str(exc), "raw_text": resp.text[:500]}


def query_pecan_graphql_variant(gene: str, variant: str) -> dict[str, Any]:
    """Query PeCan GraphQL API for a specific variant."""
    query = """
    {
        gene(name: "%s") {
            name
            mutations(protein_change: "%s") {
                protein_change
                disease
                count
                samples {
                    sample_id
                    disease
                    age_at_diagnosis
                }
            }
        }
    }
    """ % (gene, variant)
    try:
        log.info("PeCan GraphQL: querying %s %s ...", gene, variant)
        resp = SESSION.post(
            PECAN_GRAPHQL_API,
            json={"query": query},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json() if resp.text.strip() else {}
        return {"status": "ok", "data": data, "http_status": resp.status_code}
    except requests.exceptions.RequestException as exc:
        log.warning("PeCan GraphQL variant for %s %s failed: %s", gene, variant, exc)
        return {"status": "error", "error": str(exc)}
    except (json.JSONDecodeError, ValueError) as exc:
        return {"status": "parse_error", "error": str(exc), "raw_text": resp.text[:500]}


# ---------------------------------------------------------------------------
# PeCan REST search API
# ---------------------------------------------------------------------------

def query_pecan_search(gene: str) -> dict[str, Any]:
    """Query PeCan search endpoint for a gene."""
    try:
        log.info("PeCan search: querying %s ...", gene)
        resp = SESSION.get(
            PECAN_SEARCH_API,
            params={"q": gene},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json() if resp.text.strip() else {}
        return {"status": "ok", "data": data, "http_status": resp.status_code}
    except requests.exceptions.HTTPError as exc:
        log.warning("PeCan search for %s returned HTTP %s", gene, exc.response.status_code)
        return {"status": "http_error", "http_status": exc.response.status_code, "error": str(exc)}
    except requests.exceptions.RequestException as exc:
        log.warning("PeCan search for %s failed: %s", gene, exc)
        return {"status": "error", "error": str(exc)}
    except (json.JSONDecodeError, ValueError) as exc:
        return {"status": "parse_error", "error": str(exc), "raw_text": resp.text[:500]}


def query_pecan_search_variant(gene: str, variant: str) -> dict[str, Any]:
    """Query PeCan search endpoint for a specific variant."""
    search_term = f"{gene} {variant}"
    try:
        log.info("PeCan search: querying %s %s ...", gene, variant)
        resp = SESSION.get(
            PECAN_SEARCH_API,
            params={"q": search_term},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json() if resp.text.strip() else {}
        return {"status": "ok", "data": data, "http_status": resp.status_code}
    except requests.exceptions.RequestException as exc:
        log.warning("PeCan search for %s %s failed: %s", gene, variant, exc)
        return {"status": "error", "error": str(exc)}
    except (json.JSONDecodeError, ValueError) as exc:
        return {"status": "parse_error", "error": str(exc), "raw_text": resp.text[:500]}


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------

def extract_mutations_from_proteinpaint(data: dict) -> list[dict]:
    """Extract mutation entries from ProteinPaint response (multiple formats)."""
    mutations = []

    # ProteinPaint may return a list of mutation objects directly
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                mutations.append(item)
        return mutations

    # Or nested under various keys
    for key in ("mlst", "lst", "mutations", "items", "data", "hits", "results"):
        if key in data:
            items = data[key]
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        mutations.append(item)
                return mutations

    return mutations


def find_exact_variant_in_mutations(mutations: list[dict], gene: str, variant: str) -> list[dict]:
    """Search mutation list for exact protein change match."""
    matches = []
    variant_patterns = [
        f"p.{variant}",
        variant,
        f"{gene} {variant}",
        f"{gene} p.{variant}",
    ]
    for mut in mutations:
        for field in ("mname", "protein_change", "aachange", "amino_acid_change",
                       "hgvsp_short", "consequence", "name", "label"):
            val = str(mut.get(field, ""))
            if any(pat in val for pat in variant_patterns):
                matches.append(mut)
                break
    return matches


def extract_disease_info(mutations: list[dict]) -> dict[str, int]:
    """Count mutations by disease/diagnosis."""
    disease_counts: dict[str, int] = {}
    for mut in mutations:
        for field in ("disease", "diagnosis", "diagnosis_short", "cancer_type",
                       "dt", "sample_type"):
            disease = mut.get(field)
            if disease:
                disease = str(disease)
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
                break
    return disease_counts


def is_myeloid_disease(disease_str: str) -> bool:
    """Check if a disease name matches myeloid categories."""
    lower = disease_str.lower()
    return any(term in lower for term in MYELOID_DISEASE_TERMS)


# ---------------------------------------------------------------------------
# Main search orchestration
# ---------------------------------------------------------------------------

def search_all_apis() -> dict[str, Any]:
    """Run all API queries across all endpoints and genes."""
    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "St. Jude PeCan / ProteinPaint",
        "database_type": "pediatric",
        "note": "PeCan is a PEDIATRIC cancer database (~9,000+ samples). "
                "Our patient is an adult. Results are for cross-database "
                "completeness only.",
        "target_genes": TARGET_GENES,
        "exact_variants": EXACT_VARIANTS,
        "api_results": {},
        "summary": {},
    }

    api_available = {
        "proteinpaint": False,
        "pecan_graphql": False,
        "pecan_search": False,
    }

    # --- ProteinPaint API ---
    log.info("=" * 60)
    log.info("Phase 1: ProteinPaint API queries")
    log.info("=" * 60)

    pp_results: dict[str, Any] = {}
    for gene in TARGET_GENES:
        gene_results: dict[str, Any] = {"gene": gene, "variant": EXACT_VARIANTS[gene]}

        # Try singlegenelookup
        resp = query_proteinpaint_gene(gene)
        gene_results["singlegenelookup"] = resp
        if resp["status"] == "ok":
            api_available["proteinpaint"] = True
        time.sleep(REQUEST_DELAY)

        # Try genelookup
        resp = query_proteinpaint_genelookup(gene)
        gene_results["genelookup"] = resp
        if resp["status"] == "ok":
            api_available["proteinpaint"] = True
        time.sleep(REQUEST_DELAY)

        # Try snvindel
        resp = query_proteinpaint_mutations(gene)
        gene_results["snvindel"] = resp
        if resp["status"] == "ok":
            api_available["proteinpaint"] = True
            mutations = extract_mutations_from_proteinpaint(resp.get("data", {}))
            gene_results["total_mutations_found"] = len(mutations)
            exact_matches = find_exact_variant_in_mutations(
                mutations, gene, EXACT_VARIANTS[gene]
            )
            gene_results["exact_variant_matches"] = len(exact_matches)
            gene_results["exact_variant_details"] = exact_matches[:20]  # cap detail
            disease_breakdown = extract_disease_info(mutations)
            gene_results["disease_breakdown"] = disease_breakdown
            myeloid_count = sum(
                v for k, v in disease_breakdown.items() if is_myeloid_disease(k)
            )
            gene_results["myeloid_mutation_count"] = myeloid_count
        time.sleep(REQUEST_DELAY)

        # Try mds3
        resp = query_proteinpaint_mds3(gene)
        gene_results["mds3"] = resp
        if resp["status"] == "ok":
            api_available["proteinpaint"] = True
            mutations = extract_mutations_from_proteinpaint(resp.get("data", {}))
            if mutations and "total_mutations_found" not in gene_results:
                gene_results["total_mutations_found"] = len(mutations)
                exact_matches = find_exact_variant_in_mutations(
                    mutations, gene, EXACT_VARIANTS[gene]
                )
                gene_results["exact_variant_matches"] = len(exact_matches)
                gene_results["exact_variant_details"] = exact_matches[:20]
        time.sleep(REQUEST_DELAY)

        pp_results[gene] = gene_results

    results["api_results"]["proteinpaint"] = pp_results

    # --- PeCan GraphQL API ---
    log.info("=" * 60)
    log.info("Phase 2: PeCan GraphQL API queries")
    log.info("=" * 60)

    gql_results: dict[str, Any] = {}
    for gene in TARGET_GENES:
        gene_results = {"gene": gene, "variant": EXACT_VARIANTS[gene]}

        # Gene-level query
        resp = query_pecan_graphql_gene(gene)
        gene_results["gene_query"] = resp
        if resp["status"] == "ok":
            api_available["pecan_graphql"] = True
            gql_data = resp.get("data", {})
            if isinstance(gql_data, dict) and "gene" in gql_data:
                gene_data = gql_data["gene"]
                if gene_data and "mutations" in gene_data:
                    mutations = gene_data["mutations"]
                    gene_results["total_mutations"] = len(mutations) if mutations else 0
                    # Search for exact variant
                    variant = EXACT_VARIANTS[gene]
                    exact = [
                        m for m in (mutations or [])
                        if variant in str(m.get("protein_change", ""))
                    ]
                    gene_results["exact_variant_matches"] = len(exact)
                    gene_results["exact_variant_details"] = exact[:20]
                    # Disease breakdown
                    disease_counts: dict[str, int] = {}
                    for m in (mutations or []):
                        d = m.get("disease", "unknown")
                        disease_counts[d] = disease_counts.get(d, 0) + m.get("count", 1)
                    gene_results["disease_breakdown"] = disease_counts
        time.sleep(REQUEST_DELAY)

        # Variant-level query
        resp = query_pecan_graphql_variant(gene, EXACT_VARIANTS[gene])
        gene_results["variant_query"] = resp
        if resp["status"] == "ok":
            api_available["pecan_graphql"] = True
        time.sleep(REQUEST_DELAY)

        gql_results[gene] = gene_results

    results["api_results"]["pecan_graphql"] = gql_results

    # --- PeCan Search API ---
    log.info("=" * 60)
    log.info("Phase 3: PeCan REST search API queries")
    log.info("=" * 60)

    search_results: dict[str, Any] = {}
    for gene in TARGET_GENES:
        gene_results = {"gene": gene, "variant": EXACT_VARIANTS[gene]}

        resp = query_pecan_search(gene)
        gene_results["gene_search"] = resp
        if resp["status"] == "ok":
            api_available["pecan_search"] = True
        time.sleep(REQUEST_DELAY)

        resp = query_pecan_search_variant(gene, EXACT_VARIANTS[gene])
        gene_results["variant_search"] = resp
        if resp["status"] == "ok":
            api_available["pecan_search"] = True
        time.sleep(REQUEST_DELAY)

        search_results[gene] = gene_results

    results["api_results"]["pecan_search"] = search_results

    # --- Summary ---
    results["api_availability"] = api_available
    results["summary"] = build_summary(results)

    return results


def build_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Build a summary of findings across all APIs."""
    summary: dict[str, Any] = {
        "apis_reached": [],
        "apis_failed": [],
        "genes_found": [],
        "exact_variant_matches": {},
        "myeloid_hits": {},
        "total_unique_findings": 0,
    }

    avail = results.get("api_availability", {})
    for api_name, reachable in avail.items():
        if reachable:
            summary["apis_reached"].append(api_name)
        else:
            summary["apis_failed"].append(api_name)

    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        gene_found = False
        exact_count = 0
        myeloid_count = 0

        # Check ProteinPaint results
        pp = results.get("api_results", {}).get("proteinpaint", {}).get(gene, {})
        if pp.get("total_mutations_found", 0) > 0:
            gene_found = True
            exact_count = max(exact_count, pp.get("exact_variant_matches", 0))
            myeloid_count = max(myeloid_count, pp.get("myeloid_mutation_count", 0))

        # Check GraphQL results
        gql = results.get("api_results", {}).get("pecan_graphql", {}).get(gene, {})
        if gql.get("total_mutations", 0) > 0:
            gene_found = True
            exact_count = max(exact_count, gql.get("exact_variant_matches", 0))

        # Check search results
        search = results.get("api_results", {}).get("pecan_search", {}).get(gene, {})
        for key in ("gene_search", "variant_search"):
            resp = search.get(key, {})
            if resp.get("status") == "ok" and resp.get("data"):
                data = resp["data"]
                if isinstance(data, list) and len(data) > 0:
                    gene_found = True
                elif isinstance(data, dict) and data:
                    gene_found = True

        if gene_found:
            summary["genes_found"].append(gene)
        summary["exact_variant_matches"][f"{gene} {variant}"] = exact_count
        summary["myeloid_hits"][gene] = myeloid_count

    summary["total_unique_findings"] = sum(summary["exact_variant_matches"].values())
    return summary


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: dict[str, Any]) -> str:
    """Generate markdown report from search results."""
    ts = results.get("timestamp", "unknown")
    summary = results.get("summary", {})
    avail = results.get("api_availability", {})

    lines = [
        "# PeCan / St. Jude ProteinPaint -- Pediatric Cancer Database Search",
        "",
        f"**Generated:** {ts}",
        f"**Database:** St. Jude PeCan / ProteinPaint (~9,000+ pediatric cancer samples)",
        f"**Database URL:** https://pecan.stjude.cloud/",
        "",
        "> **IMPORTANT:** PeCan is a **PEDIATRIC** cancer database. Our patient is an",
        "> adult with MDS-AML. Findings from this search are included for completeness",
        "> of the cross-database survey only, not for direct clinical comparison.",
        "> Pediatric myeloid malignancies have different genomic landscapes than adult",
        "> disease (e.g., DNMT3A R882H is rare in pediatric AML but common in adult AML).",
        "",
        "---",
        "",
        "## Target Mutations",
        "",
        "| Gene | Exact Variant | Patient VAF |",
        "|------|--------------|-------------|",
        "| DNMT3A | R882H | 39% |",
        "| IDH2 | R140Q | 2% |",
        "| SETBP1 | G870S | 34% |",
        "| PTPN11 | E76Q | 29% |",
        "",
        "---",
        "",
        "## API Availability",
        "",
    ]

    for api_name in ("proteinpaint", "pecan_graphql", "pecan_search"):
        status = "Reachable" if avail.get(api_name) else "Unreachable / No data"
        lines.append(f"- **{api_name}:** {status}")
    lines.append("")

    apis_reached = summary.get("apis_reached", [])
    apis_failed = summary.get("apis_failed", [])
    if apis_reached:
        lines.append(f"APIs that returned data: {', '.join(apis_reached)}")
    if apis_failed:
        lines.append(f"APIs that failed or returned no data: {', '.join(apis_failed)}")
    lines.append("")

    # --- Per-gene results ---
    lines.extend(["---", "", "## Per-Gene Results", ""])

    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        lines.append(f"### {gene} ({variant})")
        lines.append("")

        # ProteinPaint
        pp = results.get("api_results", {}).get("proteinpaint", {}).get(gene, {})
        total_muts = pp.get("total_mutations_found", "N/A")
        exact_muts = pp.get("exact_variant_matches", "N/A")
        myeloid_muts = pp.get("myeloid_mutation_count", "N/A")
        lines.append(f"**ProteinPaint:** {total_muts} total mutations found, "
                      f"{exact_muts} exact {variant} matches, "
                      f"{myeloid_muts} myeloid")

        disease_bd = pp.get("disease_breakdown", {})
        if disease_bd:
            lines.append("")
            lines.append("Disease breakdown (ProteinPaint):")
            lines.append("")
            for disease, count in sorted(disease_bd.items(), key=lambda x: -x[1]):
                myeloid_flag = " **(myeloid)**" if is_myeloid_disease(disease) else ""
                lines.append(f"  - {disease}: {count}{myeloid_flag}")
            lines.append("")

        exact_details = pp.get("exact_variant_details", [])
        if exact_details:
            lines.append(f"Exact {variant} match details:")
            lines.append("")
            for detail in exact_details[:10]:
                lines.append(f"  - {json.dumps(detail, default=str)}")
            lines.append("")

        # GraphQL
        gql = results.get("api_results", {}).get("pecan_graphql", {}).get(gene, {})
        gql_total = gql.get("total_mutations", "N/A")
        gql_exact = gql.get("exact_variant_matches", "N/A")
        lines.append(f"**PeCan GraphQL:** {gql_total} total mutations, "
                      f"{gql_exact} exact {variant} matches")

        gql_disease = gql.get("disease_breakdown", {})
        if gql_disease:
            lines.append("")
            lines.append("Disease breakdown (GraphQL):")
            lines.append("")
            for disease, count in sorted(gql_disease.items(), key=lambda x: -x[1]):
                myeloid_flag = " **(myeloid)**" if is_myeloid_disease(disease) else ""
                lines.append(f"  - {disease}: {count}{myeloid_flag}")
            lines.append("")

        # Search
        search = results.get("api_results", {}).get("pecan_search", {}).get(gene, {})
        gene_search = search.get("gene_search", {})
        variant_search = search.get("variant_search", {})
        gs_status = gene_search.get("status", "N/A")
        vs_status = variant_search.get("status", "N/A")
        lines.append(f"**PeCan Search:** gene query={gs_status}, variant query={vs_status}")

        if gs_status == "ok" and gene_search.get("data"):
            data = gene_search["data"]
            if isinstance(data, list):
                lines.append(f"  Gene search returned {len(data)} results")
            elif isinstance(data, dict):
                lines.append(f"  Gene search returned data: {json.dumps(data, default=str)[:300]}")

        lines.extend(["", "---", ""])

    # --- Summary ---
    lines.extend(["## Summary", ""])

    exact_matches = summary.get("exact_variant_matches", {})
    lines.append("| Variant | Exact Matches in PeCan |")
    lines.append("|---------|----------------------|")
    for var_key, count in exact_matches.items():
        lines.append(f"| {var_key} | {count} |")
    lines.append("")

    total = summary.get("total_unique_findings", 0)
    genes_found = summary.get("genes_found", [])
    lines.append(f"**Genes with any mutation data:** {', '.join(genes_found) if genes_found else 'None'}")
    lines.append(f"**Total exact variant matches across all APIs:** {total}")
    lines.append("")

    # --- Co-occurrence note ---
    lines.extend([
        "## Co-occurrence Analysis",
        "",
        "PeCan does not provide a public API for per-patient co-occurrence queries.",
        "Even if individual mutations are found, it is not possible to determine",
        "whether any single pediatric patient carries 2+ of our target mutations",
        "through the API alone. Manual review via the PeCan web portal may be needed.",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "Pediatric myeloid malignancies differ substantially from adult disease:",
        "",
        "- **DNMT3A R882H** is rare in pediatric AML (<2%) but common in adult AML (~15-25%)",
        "- **IDH2 R140Q** is uncommon in pediatric AML (<3%) vs adult AML (~8-15%)",
        "- **SETBP1 G870S** is extremely rare in pediatric disease",
        "- **PTPN11 E76Q** is more common in pediatric disease (JMML, pediatric AML) than adult",
        "",
        "PTPN11 mutations (including E76Q) are notably enriched in juvenile myelomonocytic",
        "leukemia (JMML) and pediatric AML, making this the most likely gene to show",
        "hits in PeCan. The other three mutations are primarily adult-onset.",
        "",
        "Any co-occurrence of all four mutations in a pediatric patient would be",
        "extraordinary given the rarity of DNMT3A/IDH2 mutations in childhood.",
        "",
        "---",
        "",
        f"*Report generated {ts} by pecan_stjude_search.py*",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the PeCan / ProteinPaint search pipeline."""
    log.info("=" * 60)
    log.info("PeCan / St. Jude ProteinPaint Search")
    log.info("Target: %s", ", ".join(f"{g} {v}" for g, v in EXACT_VARIANTS.items()))
    log.info("=" * 60)

    results = search_all_apis()

    # Save JSON results
    json_path = RESULTS_DIR / "pecan_stjude_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("JSON results saved to %s", json_path)

    # Generate and save markdown report
    report = generate_report(results)
    report_path = RESULTS_DIR / "pecan_stjude_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Markdown report saved to %s", report_path)

    # Print summary
    summary = results.get("summary", {})
    apis_reached = summary.get("apis_reached", [])
    apis_failed = summary.get("apis_failed", [])
    exact_matches = summary.get("exact_variant_matches", {})

    log.info("=" * 60)
    log.info("SEARCH COMPLETE")
    log.info("APIs reached: %s", ", ".join(apis_reached) if apis_reached else "NONE")
    log.info("APIs failed:  %s", ", ".join(apis_failed) if apis_failed else "NONE")
    for var_key, count in exact_matches.items():
        log.info("  %s: %d exact matches", var_key, count)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
