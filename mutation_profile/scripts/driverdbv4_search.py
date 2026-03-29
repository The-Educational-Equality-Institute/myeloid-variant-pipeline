#!/usr/bin/env python3
"""
driverdbv4_search.py -- Search DriverDBv4 for patient's 4 target gene mutations.

DriverDBv4 (http://driverdb.tms.cmu.edu.tw/) is a cancer driver gene database
containing ~24,000 tumor samples with driver gene predictions from multiple
computational methods.

For each target gene (DNMT3A, IDH2, SETBP1, PTPN11):
  1. Query REST API for gene-level driver information
  2. Query mutation endpoint for specific variants (R882H, R140Q, G870S, E76Q)
  3. Check for AML/MDS/myeloid cancer type associations
  4. Extract co-occurrence data if available
  5. Fall back to web scraping if REST API is unavailable

Outputs:
    - mutation_profile/results/cross_database/driverdbv4_results.json
    - mutation_profile/results/cross_database/driverdbv4_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/driverdbv4_search.py

Runtime: ~30-90 seconds (network-dependent)
Dependencies: requests, beautifulsoup4
"""

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

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

MYELOID_KEYWORDS = [
    "aml", "acute myeloid leukemia",
    "mds", "myelodysplastic",
    "myeloproliferative", "mpn",
    "cmml", "chronic myelomonocytic",
    "myeloid", "leukemia",
]

BASE_URL = "http://driverdb.tms.cmu.edu.tw"
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 2.0  # seconds between requests

# Multiple API endpoint patterns to try
API_PATTERNS = [
    "{base}/driverdbv4/api/gene/{gene}",
    "{base}/driverdbv4/api/genes/{gene}",
    "{base}/driverdbv4/api/search/{gene}",
    "{base}/api/gene/{gene}",
    "{base}/api/v4/gene/{gene}",
]

MUTATION_API_PATTERNS = [
    "{base}/driverdbv4/api/mutation/{gene}",
    "{base}/driverdbv4/api/mutations/{gene}",
    "{base}/api/mutation/{gene}",
    "{base}/api/v4/mutation/{gene}",
]

WEB_PATTERNS = [
    "{base}/driverdbv4/gene.php?gene={gene}",
    "{base}/driverdbv4/gene/{gene}",
    "{base}/driverdbv4/search.php?query={gene}",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-bot/1.0",
    "Accept": "application/json, text/html, */*",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rate_limit():
    """Respect rate limits between requests."""
    time.sleep(REQUEST_DELAY)


def _try_get(url: str, expect_json: bool = True) -> dict | str | None:
    """Try a GET request; return parsed JSON, raw text, or None on failure."""
    try:
        log.debug("  GET %s", url)
        resp = SESSION.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        if expect_json:
            try:
                return resp.json()
            except (ValueError, requests.exceptions.JSONDecodeError):
                # Not JSON -- return text if it looks like HTML
                if resp.text.strip():
                    return resp.text
                return None
        return resp.text
    except requests.exceptions.RequestException as exc:
        log.debug("  Request failed: %s", exc)
        return None


def _try_api_endpoints(patterns: list[str], gene: str) -> tuple[dict | None, str | None]:
    """Try multiple API endpoint patterns; return first successful (data, url)."""
    for pattern in patterns:
        url = pattern.format(base=BASE_URL, gene=gene)
        _rate_limit()
        result = _try_get(url, expect_json=True)
        if result is not None and isinstance(result, dict):
            log.info("  API hit: %s", url)
            return result, url
    return None, None


def _scrape_gene_page(gene: str) -> dict[str, Any] | None:
    """
    Fall back to scraping the DriverDBv4 web interface for gene info.
    Returns extracted data dict or None.
    """
    if not HAS_BS4:
        log.warning("  beautifulsoup4 not installed -- cannot scrape web interface")
        return None

    for pattern in WEB_PATTERNS:
        url = pattern.format(base=BASE_URL, gene=gene)
        _rate_limit()
        html = _try_get(url, expect_json=False)
        if html is None or not isinstance(html, str):
            continue
        if len(html) < 500:
            continue  # Too small to be a real page

        log.info("  Web page retrieved: %s (%d bytes)", url, len(html))
        soup = BeautifulSoup(html, "html.parser")

        extracted: dict[str, Any] = {
            "source_url": url,
            "page_title": soup.title.string.strip() if soup.title and soup.title.string else None,
            "gene": gene,
            "tables": [],
            "cancer_types": [],
            "mutation_mentions": [],
            "is_driver": False,
            "methods_count": 0,
        }

        # Look for "driver" classification text
        page_text = soup.get_text(separator=" ", strip=True).lower()
        if f"{gene.lower()}" in page_text and "driver" in page_text:
            extracted["is_driver"] = True

        # Look for the exact variant mention
        variant = EXACT_VARIANTS[gene]
        variant_pattern = re.compile(
            rf"\b{re.escape(variant)}\b|"
            rf"p\.{re.escape(variant)}|"
            rf"{gene}.*{re.escape(variant)}",
            re.IGNORECASE,
        )
        for match in variant_pattern.finditer(soup.get_text(separator=" ")):
            start = max(0, match.start() - 80)
            end = min(len(soup.get_text(separator=" ")), match.end() + 80)
            context = soup.get_text(separator=" ")[start:end].strip()
            extracted["mutation_mentions"].append(context)

        # Extract table data
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                extracted["tables"].append(rows)

        # Look for cancer type mentions
        for kw in MYELOID_KEYWORDS:
            if kw in page_text:
                extracted["cancer_types"].append(kw)
        extracted["cancer_types"] = sorted(set(extracted["cancer_types"]))

        # Count prediction methods (DriverDBv4 aggregates multiple tools)
        method_keywords = [
            "mutsigcv", "oncodrivefml", "oncodriveclust", "e-driver",
            "activedriver", "drgap", "simon", "music",
        ]
        for method in method_keywords:
            if method in page_text:
                extracted["methods_count"] += 1

        if extracted["page_title"] or extracted["tables"] or extracted["mutation_mentions"]:
            return extracted

    return None


def _check_myeloid_association(data: dict | list | str) -> list[str]:
    """Check if any myeloid-related cancer types are mentioned in the data."""
    text = json.dumps(data).lower() if not isinstance(data, str) else data.lower()
    found = []
    for kw in MYELOID_KEYWORDS:
        if kw in text:
            found.append(kw)
    return sorted(set(found))


def _search_mutations_for_variant(gene: str) -> dict[str, Any]:
    """Query mutation endpoints for a specific gene's variant list."""
    result: dict[str, Any] = {
        "api_data": None,
        "api_url": None,
        "exact_variant_found": False,
        "variant_searched": EXACT_VARIANTS[gene],
        "all_variants_returned": [],
    }

    data, url = _try_api_endpoints(MUTATION_API_PATTERNS, gene)
    if data is not None:
        result["api_data"] = data
        result["api_url"] = url

        # Search for exact variant in the response
        data_str = json.dumps(data)
        variant = EXACT_VARIANTS[gene]
        if variant in data_str:
            result["exact_variant_found"] = True

        # Try to extract variant list from common response structures
        variants = []
        if isinstance(data, dict):
            for key in ["mutations", "variants", "data", "results", "items"]:
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            for vkey in ["aa_change", "protein_change", "hgvsp_short",
                                         "amino_acid_change", "mutation", "variant"]:
                                if vkey in item:
                                    variants.append(str(item[vkey]))
                        elif isinstance(item, str):
                            variants.append(item)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for vkey in ["aa_change", "protein_change", "hgvsp_short",
                                 "amino_acid_change", "mutation", "variant"]:
                        if vkey in item:
                            variants.append(str(item[vkey]))
                elif isinstance(item, str):
                    variants.append(item)

        result["all_variants_returned"] = variants[:50]  # Cap at 50

    return result


# ---------------------------------------------------------------------------
# Main query logic
# ---------------------------------------------------------------------------
def query_gene(gene: str) -> dict[str, Any]:
    """Query DriverDBv4 for a single gene. Returns structured results."""
    log.info("Querying DriverDBv4 for %s (%s)...", gene, EXACT_VARIANTS[gene])

    result: dict[str, Any] = {
        "gene": gene,
        "exact_variant": EXACT_VARIANTS[gene],
        "gene_api": {
            "success": False,
            "data": None,
            "url": None,
            "myeloid_associations": [],
        },
        "mutation_api": {
            "success": False,
            "data": None,
            "url": None,
            "exact_variant_found": False,
            "all_variants": [],
        },
        "web_scrape": {
            "success": False,
            "data": None,
        },
        "is_driver": None,
        "myeloid_associations": [],
        "exact_variant_found": False,
    }

    # --- Step 1: Gene-level API ---
    log.info("  Step 1: Trying gene-level API endpoints...")
    gene_data, gene_url = _try_api_endpoints(API_PATTERNS, gene)
    if gene_data is not None:
        result["gene_api"]["success"] = True
        result["gene_api"]["data"] = gene_data
        result["gene_api"]["url"] = gene_url
        result["gene_api"]["myeloid_associations"] = _check_myeloid_association(gene_data)
        log.info("  Gene API: success")
    else:
        log.info("  Gene API: no working endpoint found")

    # --- Step 2: Mutation-level API ---
    log.info("  Step 2: Trying mutation-level API endpoints...")
    mut_result = _search_mutations_for_variant(gene)
    if mut_result["api_data"] is not None:
        result["mutation_api"]["success"] = True
        result["mutation_api"]["data"] = mut_result["api_data"]
        result["mutation_api"]["url"] = mut_result["api_url"]
        result["mutation_api"]["exact_variant_found"] = mut_result["exact_variant_found"]
        result["mutation_api"]["all_variants"] = mut_result["all_variants_returned"]
        log.info("  Mutation API: success (exact variant found: %s)",
                 mut_result["exact_variant_found"])
    else:
        log.info("  Mutation API: no working endpoint found")

    # --- Step 3: Web scraping fallback ---
    if not result["gene_api"]["success"] and not result["mutation_api"]["success"]:
        log.info("  Step 3: Falling back to web scraping...")
        scrape_data = _scrape_gene_page(gene)
        if scrape_data is not None:
            result["web_scrape"]["success"] = True
            result["web_scrape"]["data"] = scrape_data
            log.info("  Web scrape: success")
        else:
            log.info("  Web scrape: no data extracted")
    elif not result["mutation_api"]["exact_variant_found"]:
        # Also try scraping if we didn't find the exact variant via API
        log.info("  Step 3: Scraping web page for variant details...")
        scrape_data = _scrape_gene_page(gene)
        if scrape_data is not None:
            result["web_scrape"]["success"] = True
            result["web_scrape"]["data"] = scrape_data

    # --- Consolidate findings ---
    # Check if gene is marked as driver
    if result["gene_api"]["success"] and isinstance(result["gene_api"]["data"], dict):
        gdata = result["gene_api"]["data"]
        for key in ["is_driver", "driver", "driver_gene", "classification"]:
            if key in gdata:
                result["is_driver"] = bool(gdata[key])
                break

    if result["web_scrape"]["success"] and result["web_scrape"]["data"]:
        if result["is_driver"] is None:
            result["is_driver"] = result["web_scrape"]["data"].get("is_driver", None)

    # Consolidate myeloid associations
    all_myeloid = set()
    if result["gene_api"]["myeloid_associations"]:
        all_myeloid.update(result["gene_api"]["myeloid_associations"])
    if result["mutation_api"]["success"]:
        all_myeloid.update(_check_myeloid_association(
            result["mutation_api"]["data"] or {}))
    if result["web_scrape"]["success"] and result["web_scrape"]["data"]:
        all_myeloid.update(
            result["web_scrape"]["data"].get("cancer_types", []))
    result["myeloid_associations"] = sorted(all_myeloid)

    # Consolidate exact variant found
    result["exact_variant_found"] = (
        result["mutation_api"]["exact_variant_found"]
        or (result["web_scrape"]["success"]
            and result["web_scrape"]["data"]
            and bool(result["web_scrape"]["data"].get("mutation_mentions")))
    )

    return result


def search_cooccurrence() -> dict[str, Any]:
    """
    Attempt to find co-occurrence data across target genes.
    DriverDBv4 may not directly support co-occurrence queries,
    but we check for any cross-gene links in the data.
    """
    log.info("Checking for co-occurrence data...")

    cooccurrence: dict[str, Any] = {
        "attempted": True,
        "data_available": False,
        "notes": [],
    }

    # Try a combined search endpoint if one exists
    combined_patterns = [
        "{base}/driverdbv4/api/cooccurrence/{genes}",
        "{base}/driverdbv4/api/search?genes={genes}",
        "{base}/api/cooccurrence?genes={genes}",
    ]
    genes_str = ",".join(TARGET_GENES)

    for pattern in combined_patterns:
        url = pattern.format(base=BASE_URL, genes=genes_str)
        _rate_limit()
        result = _try_get(url, expect_json=True)
        if result is not None and isinstance(result, dict):
            cooccurrence["data_available"] = True
            cooccurrence["data"] = result
            cooccurrence["url"] = url
            log.info("  Co-occurrence endpoint found: %s", url)
            return cooccurrence

    cooccurrence["notes"].append(
        "No dedicated co-occurrence API endpoint found. "
        "DriverDBv4 primarily provides per-gene driver predictions "
        "rather than patient-level co-occurrence data."
    )
    log.info("  No co-occurrence endpoint available")

    return cooccurrence


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    gene_results: dict[str, dict[str, Any]],
    cooccurrence: dict[str, Any],
    timestamp: str,
) -> str:
    """Generate a markdown report from the query results."""
    lines: list[str] = []
    lines.append("# DriverDBv4 Search Results")
    lines.append("")
    lines.append(f"**Date:** {timestamp}")
    lines.append(f"**Database:** DriverDBv4 (http://driverdb.tms.cmu.edu.tw/)")
    lines.append(f"**Database size:** ~24,000 tumor samples")
    lines.append(f"**Target genes:** {', '.join(TARGET_GENES)}")
    lines.append(f"**Exact variants:** {', '.join(f'{g} {v}' for g, v in EXACT_VARIANTS.items())}")
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append("| Gene | Variant | API Accessible | Exact Variant Found | Driver Gene | Myeloid Associations |")
    lines.append("|------|---------|----------------|---------------------|-------------|---------------------|")

    for gene in TARGET_GENES:
        r = gene_results[gene]
        api_ok = "Yes" if (r["gene_api"]["success"] or r["mutation_api"]["success"]) else "No"
        web_ok = " (web)" if r["web_scrape"]["success"] else ""
        variant_found = "Yes" if r["exact_variant_found"] else "No"
        driver = str(r["is_driver"]) if r["is_driver"] is not None else "Unknown"
        myeloid = ", ".join(r["myeloid_associations"]) if r["myeloid_associations"] else "None found"
        lines.append(
            f"| {gene} | {EXACT_VARIANTS[gene]} | {api_ok}{web_ok} | "
            f"{variant_found} | {driver} | {myeloid} |"
        )

    lines.append("")

    # --- Per-gene details ---
    lines.append("## Per-Gene Results")
    lines.append("")

    for gene in TARGET_GENES:
        r = gene_results[gene]
        lines.append(f"### {gene} ({EXACT_VARIANTS[gene]})")
        lines.append("")

        # API results
        if r["gene_api"]["success"]:
            lines.append(f"**Gene API:** Accessible at `{r['gene_api']['url']}`")
            if r["gene_api"]["myeloid_associations"]:
                lines.append(f"**Myeloid associations found:** {', '.join(r['gene_api']['myeloid_associations'])}")
        else:
            lines.append("**Gene API:** No working endpoint found")
        lines.append("")

        if r["mutation_api"]["success"]:
            lines.append(f"**Mutation API:** Accessible at `{r['mutation_api']['url']}`")
            lines.append(f"**Exact variant ({EXACT_VARIANTS[gene]}) found:** "
                         f"{'Yes' if r['mutation_api']['exact_variant_found'] else 'No'}")
            if r["mutation_api"]["all_variants"]:
                lines.append(f"**Variants returned:** {len(r['mutation_api']['all_variants'])}")
                # Show first few variants
                shown = r["mutation_api"]["all_variants"][:10]
                lines.append(f"**Sample variants:** {', '.join(shown)}")
        else:
            lines.append("**Mutation API:** No working endpoint found")
        lines.append("")

        if r["web_scrape"]["success"] and r["web_scrape"]["data"]:
            wd = r["web_scrape"]["data"]
            lines.append(f"**Web scrape:** Retrieved from `{wd.get('source_url', 'N/A')}`")
            if wd.get("page_title"):
                lines.append(f"**Page title:** {wd['page_title']}")
            if wd.get("is_driver"):
                lines.append(f"**Driver status:** Identified as driver gene")
            if wd.get("methods_count", 0) > 0:
                lines.append(f"**Prediction methods supporting:** {wd['methods_count']}")
            if wd.get("mutation_mentions"):
                lines.append(f"**Variant mentions on page:** {len(wd['mutation_mentions'])}")
                for i, mention in enumerate(wd["mutation_mentions"][:5]):
                    lines.append(f"  {i + 1}. ...{mention}...")
            if wd.get("cancer_types"):
                lines.append(f"**Myeloid terms on page:** {', '.join(wd['cancer_types'])}")
            if wd.get("tables"):
                lines.append(f"**Tables found:** {len(wd['tables'])}")
        lines.append("")

    # --- Co-occurrence ---
    lines.append("## Co-occurrence Analysis")
    lines.append("")
    if cooccurrence.get("data_available"):
        lines.append("Co-occurrence data was available from DriverDBv4:")
        lines.append(f"```json\n{json.dumps(cooccurrence.get('data', {}), indent=2)}\n```")
    else:
        lines.append("No dedicated co-occurrence endpoint was found in DriverDBv4.")
        if cooccurrence.get("notes"):
            for note in cooccurrence["notes"]:
                lines.append(f"- {note}")
    lines.append("")

    # --- Comparison with GENIE ---
    lines.append("## Comparison with GENIE v19.0 Results")
    lines.append("")
    lines.append("| Metric | GENIE v19.0 | DriverDBv4 |")
    lines.append("|--------|-------------|------------|")
    lines.append("| Database size | ~271,837 samples (27,585 myeloid) | ~24,000 tumor samples |")
    lines.append("| Focus | Pan-cancer clinical genomics | Driver gene predictions |")

    genes_found_api = sum(
        1 for g in TARGET_GENES
        if gene_results[g]["gene_api"]["success"] or gene_results[g]["mutation_api"]["success"]
    )
    genes_found_any = sum(
        1 for g in TARGET_GENES
        if (gene_results[g]["gene_api"]["success"]
            or gene_results[g]["mutation_api"]["success"]
            or gene_results[g]["web_scrape"]["success"])
    )
    variants_found = sum(1 for g in TARGET_GENES if gene_results[g]["exact_variant_found"])

    lines.append(f"| Target genes found | 4/4 | {genes_found_any}/4 (API: {genes_found_api}/4) |")
    lines.append(f"| Exact variants found | All 4 present | {variants_found}/4 |")
    lines.append("| Quadruple co-occurrence | 0 patients | N/A (no patient-level co-occurrence) |")
    lines.append("| Co-occurrence analysis | Full pairwise Fisher's exact | Not available |")
    lines.append("")

    # --- Limitations ---
    lines.append("## Limitations")
    lines.append("")
    lines.append("- DriverDBv4 focuses on driver gene *prediction* (aggregating multiple "
                 "computational methods) rather than patient-level mutation co-occurrence.")
    lines.append("- The database contains ~24,000 samples which is smaller than GENIE v19.0 "
                 "(271,837 samples).")
    lines.append("- DriverDBv4 may not separate myeloid-specific data from pan-cancer results.")
    lines.append("- API endpoints may change; the script tries multiple patterns to be robust.")
    lines.append("- Co-occurrence across genes requires patient-level data which DriverDBv4 "
                 "may not expose via its public interface.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Query DriverDBv4 for all target genes and generate results."""
    log.info("=" * 60)
    log.info("DriverDBv4 Search for Patient Mutation Profile")
    log.info("=" * 60)
    log.info("Target genes: %s", ", ".join(TARGET_GENES))
    log.info("Exact variants: %s",
             ", ".join(f"{g} {v}" for g, v in EXACT_VARIANTS.items()))
    log.info("")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    gene_results: dict[str, dict[str, Any]] = {}

    # Query each gene
    for gene in TARGET_GENES:
        gene_results[gene] = query_gene(gene)
        log.info("")

    # Check for co-occurrence data
    cooccurrence = search_cooccurrence()
    log.info("")

    # --- Build combined results ---
    results: dict[str, Any] = {
        "metadata": {
            "script": "driverdbv4_search.py",
            "database": "DriverDBv4",
            "database_url": "http://driverdb.tms.cmu.edu.tw/",
            "database_size": "~24,000 tumor samples",
            "timestamp": timestamp,
            "target_genes": TARGET_GENES,
            "exact_variants": EXACT_VARIANTS,
        },
        "gene_results": {},
        "cooccurrence": cooccurrence,
        "summary": {
            "genes_accessible_api": 0,
            "genes_accessible_any": 0,
            "exact_variants_found": 0,
            "driver_genes_confirmed": 0,
            "myeloid_associated_genes": 0,
        },
    }

    for gene in TARGET_GENES:
        r = gene_results[gene]

        # Serialize without raw API data for cleaner JSON
        serializable = {
            "gene": gene,
            "exact_variant": EXACT_VARIANTS[gene],
            "gene_api_accessible": r["gene_api"]["success"],
            "gene_api_url": r["gene_api"]["url"],
            "gene_api_myeloid": r["gene_api"]["myeloid_associations"],
            "mutation_api_accessible": r["mutation_api"]["success"],
            "mutation_api_url": r["mutation_api"]["url"],
            "exact_variant_found": r["exact_variant_found"],
            "mutation_api_variants_count": len(r["mutation_api"]["all_variants"]),
            "mutation_api_sample_variants": r["mutation_api"]["all_variants"][:20],
            "web_scrape_success": r["web_scrape"]["success"],
            "is_driver": r["is_driver"],
            "myeloid_associations": r["myeloid_associations"],
        }

        # Include web scrape details if available (without full table data)
        if r["web_scrape"]["success"] and r["web_scrape"]["data"]:
            wd = r["web_scrape"]["data"]
            serializable["web_scrape_details"] = {
                "source_url": wd.get("source_url"),
                "page_title": wd.get("page_title"),
                "is_driver": wd.get("is_driver"),
                "methods_count": wd.get("methods_count", 0),
                "mutation_mentions_count": len(wd.get("mutation_mentions", [])),
                "mutation_mentions": wd.get("mutation_mentions", [])[:5],
                "cancer_types": wd.get("cancer_types", []),
                "tables_count": len(wd.get("tables", [])),
            }

        results["gene_results"][gene] = serializable

        # Update summary counters
        if r["gene_api"]["success"] or r["mutation_api"]["success"]:
            results["summary"]["genes_accessible_api"] += 1
        if r["gene_api"]["success"] or r["mutation_api"]["success"] or r["web_scrape"]["success"]:
            results["summary"]["genes_accessible_any"] += 1
        if r["exact_variant_found"]:
            results["summary"]["exact_variants_found"] += 1
        if r["is_driver"] is True:
            results["summary"]["driver_genes_confirmed"] += 1
        if r["myeloid_associations"]:
            results["summary"]["myeloid_associated_genes"] += 1

    # --- Write JSON ---
    json_path = RESULTS_DIR / "driverdbv4_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("JSON results written to %s", json_path)

    # --- Write report ---
    report = generate_report(gene_results, cooccurrence, timestamp)
    report_path = RESULTS_DIR / "driverdbv4_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Markdown report written to %s", report_path)

    # --- Final summary ---
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("Genes accessible (API): %d/4", results["summary"]["genes_accessible_api"])
    log.info("Genes accessible (any): %d/4", results["summary"]["genes_accessible_any"])
    log.info("Exact variants found:   %d/4", results["summary"]["exact_variants_found"])
    log.info("Driver genes confirmed: %d/4", results["summary"]["driver_genes_confirmed"])
    log.info("Myeloid associations:   %d/4", results["summary"]["myeloid_associated_genes"])
    log.info("Co-occurrence data:     %s",
             "Available" if cooccurrence.get("data_available") else "Not available")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sys.exit(1)
    except Exception:
        log.exception("Unexpected error")
        sys.exit(1)
