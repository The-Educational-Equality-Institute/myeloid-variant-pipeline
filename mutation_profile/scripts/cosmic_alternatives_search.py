#!/usr/bin/env python3
"""
cosmic_alternatives_search.py -- Query free databases that partially overlap with
COSMIC data as a workaround while COSMIC academic access is blocked.

Databases queried:
  1. TCGA-LAML via cBioPortal   -- exact variant search in TCGA AML study
  2. My Cancer Genome            -- Vanderbilt curated variant database
  3. CIViC                       -- Clinical Interpretation of Variants in Cancer (GraphQL)
  4. OncoKB                      -- MSK variant annotation (public endpoint)
  5. Cancer Hotspots             -- MSK recurrent mutation hotspot database

Inputs:
    - Remote APIs only (no local data required)

Outputs:
    - mutation_profile/results/cross_database/cosmic_alternatives_results.json
    - mutation_profile/results/cross_database/cosmic_alternatives_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/cosmic_alternatives_search.py

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
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

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

# Protein change aliases for matching across databases
VARIANT_ALIASES = {
    "DNMT3A": ["R882H", "p.R882H", "p.Arg882His", "Arg882His"],
    "IDH2": ["R140Q", "p.R140Q", "p.Arg140Gln", "Arg140Gln"],
    "SETBP1": ["G870S", "p.G870S", "p.Gly870Ser", "Gly870Ser"],
    "PTPN11": ["E76Q", "p.E76Q", "p.Glu76Gln", "Glu76Gln"],
}

# Entrez gene IDs
ENTREZ_IDS = {
    "DNMT3A": 1788,
    "IDH2": 3418,
    "SETBP1": 26040,
    "PTPN11": 5781,
}

# API endpoints
CBIO_API_URL = "https://www.cbioportal.org/api"
CIVIC_GRAPHQL_URL = "https://civicdb.org/api/graphql"
ONCOKB_API_URL = "https://www.oncokb.org/api/v1"
CANCER_HOTSPOTS_API_URL = "https://www.cancerhotspots.org/api"
MY_CANCER_GENOME_URL = "https://www.mycancergenome.org"

REQUEST_DELAY = 2  # seconds between requests
REQUEST_TIMEOUT = 30  # seconds

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "cosmic_alternatives_search.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "mrna-hematology-research/1.0",
})


def _api_get(url: str, params: dict | None = None, headers: dict | None = None) -> dict | list | None:
    """GET request with error handling and delay."""
    time.sleep(REQUEST_DELAY)
    try:
        resp = SESSION.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        log.warning("GET %s failed: %s", url, e)
        return None


def _api_post(url: str, json_data: dict, headers: dict | None = None) -> dict | None:
    """POST request with error handling and delay."""
    time.sleep(REQUEST_DELAY)
    try:
        resp = SESSION.post(url, json=json_data, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        log.warning("POST %s failed: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# 1. TCGA-LAML via cBioPortal
# ---------------------------------------------------------------------------
def query_tcga_laml() -> dict[str, Any]:
    """Query TCGA-LAML study on cBioPortal for exact patient variants."""
    log.info("=" * 60)
    log.info("1. TCGA-LAML via cBioPortal")
    log.info("=" * 60)

    results: dict[str, Any] = {
        "database": "TCGA-LAML (cBioPortal)",
        "url": "https://www.cbioportal.org/study/summary?id=laml_tcga",
        "status": "error",
        "genes": {},
    }

    # Find the TCGA-LAML study ID
    studies = _api_get(f"{CBIO_API_URL}/studies")
    if not studies:
        log.error("Failed to fetch cBioPortal studies list")
        results["error"] = "Failed to fetch studies list"
        return results

    tcga_laml_ids = [
        s["studyId"] for s in studies
        if "laml" in s["studyId"].lower() and "tcga" in s["studyId"].lower()
    ]
    log.info("Found TCGA-LAML study IDs: %s", tcga_laml_ids)

    if not tcga_laml_ids:
        # Fallback to known ID
        tcga_laml_ids = ["laml_tcga"]
        log.info("Using fallback study ID: laml_tcga")

    study_id = tcga_laml_ids[0]

    # Get molecular profiles for mutations
    profiles = _api_get(f"{CBIO_API_URL}/studies/{study_id}/molecular-profiles")
    if not profiles:
        log.error("Failed to fetch molecular profiles for %s", study_id)
        results["error"] = "Failed to fetch molecular profiles"
        return results

    mut_profile_id = None
    for p in profiles:
        if p.get("molecularAlterationType") == "MUTATION_EXTENDED":
            mut_profile_id = p["molecularProfileId"]
            break

    if not mut_profile_id:
        log.error("No mutation profile found for %s", study_id)
        results["error"] = "No mutation profile found"
        return results

    log.info("Using molecular profile: %s", mut_profile_id)

    # Get sample list for the study
    sample_list_id = f"{study_id}_all"

    # Query each gene using the correct cBioPortal v1 endpoint
    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        entrez_id = ENTREZ_IDS[gene]
        log.info("Querying %s %s (Entrez %d)...", gene, variant, entrez_id)

        mutations = _api_get(
            f"{CBIO_API_URL}/molecular-profiles/{mut_profile_id}/mutations",
            params={
                "entrezGeneId": entrez_id,
                "sampleListId": sample_list_id,
                "projection": "DETAILED",
            },
        )

        gene_result: dict[str, Any] = {
            "total_mutations": 0,
            "exact_variant_found": False,
            "exact_variant_count": 0,
            "all_variants": [],
        }

        if mutations:
            gene_result["total_mutations"] = len(mutations)
            for m in mutations:
                aa_change = m.get("proteinChange", "")
                gene_result["all_variants"].append(aa_change)
                if any(alias in aa_change for alias in VARIANT_ALIASES[gene]):
                    gene_result["exact_variant_found"] = True
                    gene_result["exact_variant_count"] += 1

            # Deduplicate variant list for report
            gene_result["all_variants"] = sorted(set(gene_result["all_variants"]))

        log.info(
            "  %s: %d total mutations, exact %s found=%s (count=%d)",
            gene, gene_result["total_mutations"], variant,
            gene_result["exact_variant_found"], gene_result["exact_variant_count"],
        )
        results["genes"][gene] = gene_result

    results["status"] = "success"
    results["study_id"] = study_id
    results["molecular_profile_id"] = mut_profile_id
    return results


# ---------------------------------------------------------------------------
# 2. My Cancer Genome (Vanderbilt)
# ---------------------------------------------------------------------------
def query_my_cancer_genome() -> dict[str, Any]:
    """Check My Cancer Genome for each gene/variant."""
    log.info("=" * 60)
    log.info("2. My Cancer Genome (Vanderbilt)")
    log.info("=" * 60)

    results: dict[str, Any] = {
        "database": "My Cancer Genome (Vanderbilt)",
        "url": MY_CANCER_GENOME_URL,
        "status": "error",
        "genes": {},
    }

    # My Cancer Genome does not have a public REST API. We check known URL
    # patterns and see if pages exist for our variants.
    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        gene_lower = gene.lower()

        # Known URL pattern for gene pages
        gene_url = f"{MY_CANCER_GENOME_URL}/content/disease/acute-myeloid-leukemia-aml/{gene_lower}/"
        variant_url = f"{MY_CANCER_GENOME_URL}/content/alteration/{gene_lower}-{variant.lower()}/"

        gene_result: dict[str, Any] = {
            "gene_page_url": gene_url,
            "variant_page_url": variant_url,
            "gene_page_accessible": False,
            "variant_page_accessible": False,
        }

        time.sleep(REQUEST_DELAY)
        try:
            resp = SESSION.get(gene_url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            gene_result["gene_page_accessible"] = resp.status_code == 200
            gene_result["gene_page_status"] = resp.status_code
            log.info("  %s gene page: HTTP %d", gene, resp.status_code)
        except requests.exceptions.RequestException as e:
            log.warning("  %s gene page failed: %s", gene, e)
            gene_result["gene_page_status"] = str(e)

        time.sleep(REQUEST_DELAY)
        try:
            resp = SESSION.get(variant_url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            gene_result["variant_page_accessible"] = resp.status_code == 200
            gene_result["variant_page_status"] = resp.status_code
            log.info("  %s %s variant page: HTTP %d", gene, variant, resp.status_code)
        except requests.exceptions.RequestException as e:
            log.warning("  %s %s variant page failed: %s", gene, variant, e)
            gene_result["variant_page_status"] = str(e)

        results["genes"][gene] = gene_result

    results["status"] = "success"
    return results


# ---------------------------------------------------------------------------
# 3. CIViC (Clinical Interpretation of Variants in Cancer)
# ---------------------------------------------------------------------------
def query_civic() -> dict[str, Any]:
    """Query CIViC GraphQL API for each gene's variants."""
    log.info("=" * 60)
    log.info("3. CIViC (Clinical Interpretation of Variants in Cancer)")
    log.info("=" * 60)

    results: dict[str, Any] = {
        "database": "CIViC",
        "url": "https://civicdb.org",
        "status": "error",
        "genes": {},
    }

    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        log.info("Querying CIViC for %s...", gene)

        # CIViC GraphQL: use variantsTypeahead which works without auth
        query = """
{
  variantsTypeahead(queryTerm: "%s") {
    name
    id
    feature { name }
    singleVariantMolecularProfile {
      molecularProfileScore
    }
  }
}
""" % variant

        data = _api_post(CIVIC_GRAPHQL_URL, json_data={"query": query})

        gene_result: dict[str, Any] = {
            "total_variants": 0,
            "exact_variant_found": False,
            "exact_variant_score": None,
            "civic_variant_id": None,
            "all_variants": [],
        }

        if data and "data" in data and data["data"].get("variantsTypeahead"):
            nodes = data["data"]["variantsTypeahead"]
            gene_result["total_variants"] = len(nodes)

            for node in nodes:
                name = node.get("name", "")
                feature_name = node.get("feature", {}).get("name", "")
                score = None
                profile = node.get("singleVariantMolecularProfile")
                if profile:
                    score = profile.get("molecularProfileScore")

                gene_result["all_variants"].append({
                    "name": name,
                    "gene": feature_name,
                    "molecular_profile_score": score,
                    "civic_id": node.get("id"),
                })

                # Check for exact match (same gene + same variant)
                if feature_name == gene and any(alias in name for alias in VARIANT_ALIASES[gene]):
                    gene_result["exact_variant_found"] = True
                    gene_result["exact_variant_score"] = score
                    gene_result["civic_variant_id"] = node.get("id")
                    log.info("  MATCH: %s %s (score=%s, CIViC ID=%s)", gene, name, score, node.get("id"))

            log.info(
                "  %s: %d results in CIViC, exact %s found=%s",
                gene, gene_result["total_variants"], variant,
                gene_result["exact_variant_found"],
            )
        elif data and "errors" in data:
            log.warning("  CIViC GraphQL error for %s: %s", gene, data["errors"])
            gene_result["error"] = str(data["errors"])
        else:
            log.warning("  No data returned for %s", gene)
            gene_result["error"] = "No data returned"

        results["genes"][gene] = gene_result

    results["status"] = "success"
    return results


# ---------------------------------------------------------------------------
# 4. OncoKB (MSK)
# ---------------------------------------------------------------------------
def query_oncokb() -> dict[str, Any]:
    """Query OncoKB public endpoint for variant annotations."""
    log.info("=" * 60)
    log.info("4. OncoKB (MSK)")
    log.info("=" * 60)

    results: dict[str, Any] = {
        "database": "OncoKB",
        "url": "https://www.oncokb.org",
        "status": "error",
        "genes": {},
    }

    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        log.info("Querying OncoKB for %s %s...", gene, variant)

        # Try the public variants lookup endpoint
        gene_result: dict[str, Any] = {
            "gene_found": False,
            "variant_found": False,
            "oncogenic": None,
            "mutation_effect": None,
            "highest_sensitive_level": None,
            "highest_resistance_level": None,
            "all_variants": [],
        }

        # Query gene-level info
        gene_data = _api_get(
            f"{ONCOKB_API_URL}/utils/allAnnotatedVariants",
        )

        if gene_data:
            matching = [
                v for v in gene_data
                if v.get("gene", {}).get("hugoSymbol") == gene
                or v.get("hugoSymbol") == gene
                or (isinstance(v, dict) and v.get("gene") == gene)
            ]
            if matching:
                gene_result["gene_found"] = True
                gene_result["total_annotated_variants"] = len(matching)
                for v in matching:
                    v_name = v.get("variant", {}).get("name", "") if isinstance(v.get("variant"), dict) else v.get("alteration", "")
                    gene_result["all_variants"].append(v_name)
                    if any(alias in str(v_name) for alias in VARIANT_ALIASES[gene]):
                        gene_result["variant_found"] = True
                        gene_result["oncogenic"] = v.get("oncogenic", "")
                        gene_result["mutation_effect"] = v.get("mutationEffect", "")
                        gene_result["highest_sensitive_level"] = v.get("highestSensitiveLevel", "")
                        gene_result["highest_resistance_level"] = v.get("highestResistanceLevel", "")
                        log.info("  MATCH: %s %s - oncogenic=%s", gene, v_name, gene_result["oncogenic"])
                log.info(
                    "  %s: %d annotated variants, exact %s found=%s",
                    gene, len(matching), variant, gene_result["variant_found"],
                )
            else:
                log.info("  %s: no annotated variants found in OncoKB", gene)
        else:
            # Fallback: try the specific variant lookup
            log.info("  Bulk endpoint failed, trying specific variant lookup...")
            variant_data = _api_get(
                f"{ONCOKB_API_URL}/variants/lookup",
                params={"hugoSymbol": gene},
            )
            if variant_data:
                if isinstance(variant_data, list):
                    gene_result["gene_found"] = True
                    gene_result["total_annotated_variants"] = len(variant_data)
                    for v in variant_data:
                        v_name = v.get("name", "") or v.get("alteration", "")
                        gene_result["all_variants"].append(v_name)
                        if any(alias in str(v_name) for alias in VARIANT_ALIASES[gene]):
                            gene_result["variant_found"] = True
                            log.info("  MATCH: %s %s", gene, v_name)
                    log.info("  %s: %d variants from lookup", gene, len(variant_data))
                elif isinstance(variant_data, dict):
                    gene_result["gene_found"] = True
                    gene_result["note"] = "Single result returned"
                    log.info("  %s: single result returned", gene)
            else:
                gene_result["error"] = "API requires authentication token"
                log.warning("  OncoKB requires API token for %s", gene)

        # Trim variant list for JSON output (keep unique, sorted)
        gene_result["all_variants"] = sorted(set(str(v) for v in gene_result["all_variants"]))

        results["genes"][gene] = gene_result

    results["status"] = "success"
    return results


# ---------------------------------------------------------------------------
# 5. Cancer Hotspots (MSK)
# ---------------------------------------------------------------------------
def query_cancer_hotspots() -> dict[str, Any]:
    """Query Cancer Hotspots API for recurrent mutation data."""
    log.info("=" * 60)
    log.info("5. Cancer Hotspots (MSK)")
    log.info("=" * 60)

    results: dict[str, Any] = {
        "database": "Cancer Hotspots",
        "url": "https://www.cancerhotspots.org",
        "status": "error",
        "genes": {},
    }

    # Download all hotspots once and filter locally
    all_hotspots = _api_get(f"{CANCER_HOTSPOTS_API_URL}/hotspots/single")
    if all_hotspots:
        log.info("  Downloaded %d total cancer hotspot residues", len(all_hotspots))
    else:
        log.warning("  Failed to download cancer hotspots data")
        all_hotspots = []

    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        log.info("Querying Cancer Hotspots for %s %s...", gene, variant)

        gene_result: dict[str, Any] = {
            "gene_found": False,
            "variant_is_hotspot": False,
            "hotspot_residues": [],
            "variant_details": None,
        }

        # Filter hotspots for this gene
        hotspot_data = [h for h in all_hotspots if h.get("hugoSymbol") == gene]

        if hotspot_data:
            gene_result["gene_found"] = True
            gene_result["total_hotspot_residues"] = len(hotspot_data) if isinstance(hotspot_data, list) else 1

            if isinstance(hotspot_data, list):
                for h in hotspot_data:
                    residue = h.get("residue", "")
                    # Extract residue number from variant (e.g., "R882H" -> 882)
                    variant_residue = "".join(c for c in variant if c.isdigit())
                    hotspot_residue = "".join(c for c in str(residue) if c.isdigit())

                    gene_result["hotspot_residues"].append({
                        "residue": residue,
                        "type": h.get("type", ""),
                        "classification": h.get("classification", ""),
                    })

                    if variant_residue and hotspot_residue and variant_residue == hotspot_residue:
                        gene_result["variant_is_hotspot"] = True
                        gene_result["variant_details"] = {
                            "residue": residue,
                            "type": h.get("type", ""),
                            "classification": h.get("classification", ""),
                            "tumor_count": h.get("tumorCount", None),
                            "tumor_type_composition": h.get("tumorTypeComposition", None),
                        }
                        log.info("  HOTSPOT: %s residue %s", gene, residue)

            log.info(
                "  %s: %d hotspot residues, %s is hotspot=%s",
                gene, gene_result.get("total_hotspot_residues", 0),
                variant, gene_result["variant_is_hotspot"],
            )
        else:
            log.warning("  No hotspot data for %s (API may be unavailable)", gene)
            gene_result["error"] = "API returned no data or unavailable"

        results["genes"][gene] = gene_result

    results["status"] = "success"
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(all_results: dict[str, Any]) -> str:
    """Generate a markdown report summarizing all database results."""
    lines = [
        "# COSMIC Alternatives Search Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Background",
        "",
        "COSMIC (Catalogue of Somatic Mutations in Cancer) is the gold standard for somatic",
        "mutation frequency data, but academic access requires institutional application which",
        "is currently pending. This report summarizes data from free alternative databases that",
        "partially overlap with COSMIC's coverage.",
        "",
        "## Patient Variants Searched",
        "",
        "| Gene | Variant | Protein Change |",
        "|------|---------|----------------|",
    ]
    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        lines.append(f"| {gene} | {variant} | p.{variant} |")

    lines.extend(["", "---", ""])

    # 1. TCGA-LAML
    tcga = all_results.get("tcga_laml", {})
    lines.extend([
        "## 1. TCGA-LAML via cBioPortal",
        "",
        f"**Status:** {tcga.get('status', 'unknown')}",
        f"**Study:** {tcga.get('study_id', 'N/A')}",
        "",
    ])
    if tcga.get("status") == "success":
        lines.extend([
            "| Gene | Variant | Total Mutations | Exact Match | Count |",
            "|------|---------|-----------------|-------------|-------|",
        ])
        for gene in TARGET_GENES:
            g = tcga.get("genes", {}).get(gene, {})
            lines.append(
                f"| {gene} | {EXACT_VARIANTS[gene]} | {g.get('total_mutations', 0)} "
                f"| {'Yes' if g.get('exact_variant_found') else 'No'} "
                f"| {g.get('exact_variant_count', 0)} |"
            )
    else:
        lines.append(f"*Error: {tcga.get('error', 'Unknown error')}*")
    lines.extend(["", "---", ""])

    # 2. My Cancer Genome
    mcg = all_results.get("my_cancer_genome", {})
    lines.extend([
        "## 2. My Cancer Genome (Vanderbilt)",
        "",
        f"**Status:** {mcg.get('status', 'unknown')}",
        "",
        "My Cancer Genome is a curated clinical resource. It does not provide a public REST API,",
        "so we checked for accessible gene and variant pages.",
        "",
        "| Gene | Variant | Gene Page | Variant Page |",
        "|------|---------|-----------|--------------|",
    ])
    for gene in TARGET_GENES:
        g = mcg.get("genes", {}).get(gene, {})
        gene_ok = "Accessible" if g.get("gene_page_accessible") else f"HTTP {g.get('gene_page_status', '?')}"
        var_ok = "Accessible" if g.get("variant_page_accessible") else f"HTTP {g.get('variant_page_status', '?')}"
        lines.append(f"| {gene} | {EXACT_VARIANTS[gene]} | {gene_ok} | {var_ok} |")
    lines.extend(["", "---", ""])

    # 3. CIViC
    civic = all_results.get("civic", {})
    lines.extend([
        "## 3. CIViC (Clinical Interpretation of Variants in Cancer)",
        "",
        f"**Status:** {civic.get('status', 'unknown')}",
        "",
        "| Gene | Variant | Total Variants in CIViC | Exact Match | Molecular Profile Score |",
        "|------|---------|------------------------|-------------|------------------------|",
    ])
    for gene in TARGET_GENES:
        g = civic.get("genes", {}).get(gene, {})
        score = g.get("exact_variant_score", "N/A")
        lines.append(
            f"| {gene} | {EXACT_VARIANTS[gene]} | {g.get('total_variants', 0)} "
            f"| {'Yes' if g.get('exact_variant_found') else 'No'} "
            f"| {score} |"
        )
    lines.extend(["", "---", ""])

    # 4. OncoKB
    oncokb = all_results.get("oncokb", {})
    lines.extend([
        "## 4. OncoKB (MSK)",
        "",
        f"**Status:** {oncokb.get('status', 'unknown')}",
        "",
        "| Gene | Variant | Gene Found | Variant Found | Oncogenic | Mutation Effect |",
        "|------|---------|------------|---------------|-----------|-----------------|",
    ])
    for gene in TARGET_GENES:
        g = oncokb.get("genes", {}).get(gene, {})
        lines.append(
            f"| {gene} | {EXACT_VARIANTS[gene]} "
            f"| {'Yes' if g.get('gene_found') else 'No'} "
            f"| {'Yes' if g.get('variant_found') else 'No'} "
            f"| {g.get('oncogenic', 'N/A')} "
            f"| {g.get('mutation_effect', 'N/A')} |"
        )
    lines.extend(["", "---", ""])

    # 5. Cancer Hotspots
    hotspots = all_results.get("cancer_hotspots", {})
    lines.extend([
        "## 5. Cancer Hotspots (MSK)",
        "",
        f"**Status:** {hotspots.get('status', 'unknown')}",
        "",
        "| Gene | Variant | Gene in DB | Is Hotspot | Hotspot Residues |",
        "|------|---------|------------|------------|------------------|",
    ])
    for gene in TARGET_GENES:
        g = hotspots.get("genes", {}).get(gene, {})
        n_residues = g.get("total_hotspot_residues", 0)
        lines.append(
            f"| {gene} | {EXACT_VARIANTS[gene]} "
            f"| {'Yes' if g.get('gene_found') else 'No'} "
            f"| {'Yes' if g.get('variant_is_hotspot') else 'No'} "
            f"| {n_residues} |"
        )
    lines.extend(["", "---", ""])

    # Summary
    lines.extend([
        "## Summary",
        "",
        "### Variant Coverage Across Databases",
        "",
        "| Gene | Variant | TCGA-LAML | CIViC | OncoKB | Cancer Hotspots |",
        "|------|---------|-----------|-------|--------|-----------------|",
    ])
    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        tcga_hit = "Found" if tcga.get("genes", {}).get(gene, {}).get("exact_variant_found") else "Not found"
        civic_hit = "Found" if civic.get("genes", {}).get(gene, {}).get("exact_variant_found") else "Not found"
        oncokb_hit = "Found" if oncokb.get("genes", {}).get(gene, {}).get("variant_found") else "Not found"
        hotspot_hit = "Hotspot" if hotspots.get("genes", {}).get(gene, {}).get("variant_is_hotspot") else "Not hotspot"
        lines.append(f"| {gene} | {variant} | {tcga_hit} | {civic_hit} | {oncokb_hit} | {hotspot_hit} |")

    lines.extend([
        "",
        "### Limitations",
        "",
        "- **COSMIC access pending**: These databases partially overlap with COSMIC but none",
        "  provides the same depth of somatic mutation frequency data across cancer types.",
        "- **My Cancer Genome**: No public API; web scraping results may be incomplete.",
        "- **OncoKB**: Full API access requires an API token (free for academic use,",
        "  application at https://www.oncokb.org/apiAccess).",
        "- **Cancer Hotspots**: Based on large-scale sequencing studies but may not be",
        "  actively maintained.",
        "- **Co-occurrence data**: None of these databases provide the same co-occurrence",
        "  query capability as COSMIC. For co-occurrence, GENIE v19.0 and cBioPortal",
        "  remain the primary sources.",
        "",
        "### Next Steps",
        "",
        "1. Apply for COSMIC academic access through TEEI (institutional application)",
        "2. Apply for OncoKB API token (free academic tier)",
        "3. Check FinnGen and UK Biobank for population-level hematological cancer data",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run all COSMIC alternative database queries."""
    start_time = time.time()
    log.info("COSMIC Alternatives Search - starting")
    log.info("Target variants: %s", ", ".join(f"{g} {v}" for g, v in EXACT_VARIANTS.items()))

    all_results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_genes": TARGET_GENES,
        "exact_variants": EXACT_VARIANTS,
    }

    # Query each database
    all_results["tcga_laml"] = query_tcga_laml()
    all_results["my_cancer_genome"] = query_my_cancer_genome()
    all_results["civic"] = query_civic()
    all_results["oncokb"] = query_oncokb()
    all_results["cancer_hotspots"] = query_cancer_hotspots()

    # Save JSON results
    json_path = RESULTS_DIR / "cosmic_alternatives_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("JSON results saved to %s", json_path)

    # Generate and save report
    report = generate_report(all_results)
    report_path = RESULTS_DIR / "cosmic_alternatives_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Report saved to %s", report_path)

    elapsed = time.time() - start_time
    log.info("Completed in %.1f seconds", elapsed)

    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        hits = []
        if all_results["tcga_laml"].get("genes", {}).get(gene, {}).get("exact_variant_found"):
            hits.append("TCGA-LAML")
        if all_results["civic"].get("genes", {}).get(gene, {}).get("exact_variant_found"):
            hits.append("CIViC")
        if all_results["oncokb"].get("genes", {}).get(gene, {}).get("variant_found"):
            hits.append("OncoKB")
        if all_results["cancer_hotspots"].get("genes", {}).get(gene, {}).get("variant_is_hotspot"):
            hits.append("Hotspots")
        hit_str = ", ".join(hits) if hits else "none"
        log.info("  %s %s: found in %s", gene, variant, hit_str)


if __name__ == "__main__":
    main()
