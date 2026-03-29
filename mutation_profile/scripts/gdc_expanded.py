#!/usr/bin/env python3
"""
gdc_expanded.py -- Expanded GDC query across ALL hematological/myeloid projects.

The existing cross_database.py only queries TCGA-LAML with gene-level SSM counts
and cannot do per-case variant intersection. This script expands the search to
all GDC myeloid projects, searches for exact patient variants (R882H, R140Q,
G870S, E76Q), and performs donor-level co-occurrence analysis.

GDC projects queried:
  - TCGA-LAML (AML, ~200 cases)
  - TARGET-AML (pediatric AML, ~1000+ cases)
  - BEATAML1.0-COHORT (~672 cases)
  - BEATAML1.0-CRENOLANIB (~43 cases)
  - Any other hematological/myeloid projects discovered at runtime

Approach:
  1. Discover all GDC projects with hematological primary sites
  2. For each project, count total myeloid cases
  3. Query ssm_occurrences for each target gene + project
  4. Search for exact amino acid changes (R882H, R140Q, G870S, E76Q)
  5. Retrieve case-level mutation data to check co-occurrence
  6. Report unique patients beyond what GENIE covers

Outputs:
  - mutation_profile/results/cross_database/gdc_expanded.json
  - mutation_profile/results/cross_database/gdc_expanded_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/gdc_expanded.py

Runtime: ~2-5 minutes (network-dependent)
Dependencies: pandas, requests
"""

import json
import logging
import sys
import time
from collections import defaultdict
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
GDC_BASE = "https://api.gdc.cancer.gov"

TARGET_GENES = ["DNMT3A", "IDH2", "SETBP1", "PTPN11"]

# Exact patient variants we are searching for
# GRCh38 genomic coordinates for GDC queries (GDC uses GRCh38 natively)
EXACT_VARIANTS = {
    "DNMT3A": {
        "aa_change": "R882H",
        "hgvsp_short": "p.R882H",
        "chromosome": "chr2",
        "start": 25234373,
        "ref": "C",
        "alt": "T",
        "genomic_dna_change": "chr2:g.25234373C>T",  # GDC stores on minus strand
    },
    "IDH2": {
        "aa_change": "R140Q",
        "hgvsp_short": "p.R140Q",
        "chromosome": "chr15",
        "start": 90088702,
        "ref": "C",
        "alt": "T",
        "genomic_dna_change": "chr15:g.90088702C>T",  # GDC stores on minus strand
    },
    "SETBP1": {
        "aa_change": "G870S",
        "hgvsp_short": "p.G870S",
        "chromosome": "chr18",
        "start": 44951948,
        "ref": "G",
        "alt": "A",
        "genomic_dna_change": "chr18:g.44951948G>A",
    },
    "PTPN11": {
        "aa_change": "E76Q",
        "hgvsp_short": "p.E76Q",
        "chromosome": "chr12",
        "start": 112450406,
        "ref": "G",
        "alt": "C",
        "genomic_dna_change": "chr12:g.112450406G>C",
    },
}

# Entrez gene IDs for GDC gene-level queries
GENE_ENTREZ_IDS = {
    "DNMT3A": "ENSG00000119772",
    "IDH2": "ENSG00000182054",
    "SETBP1": "ENSG00000152217",
    "PTPN11": "ENSG00000179295",
}

# Known myeloid/hematological projects to always check
KNOWN_MYELOID_PROJECTS = [
    "TCGA-LAML",
    "TARGET-AML",
    "BEATAML1.0-COHORT",
    "BEATAML1.0-CRENOLANIB",
]

# Primary sites that indicate hematological malignancies
HEMATOLOGICAL_PRIMARY_SITES = {
    "hematopoietic and reticuloendothelial systems",
    "blood",
    "bone marrow",
    "hematopoietic system",
    "lymph nodes",
    "spleen",
}

# Coding consequence types (exclude silent, intronic, UTR)
CODING_CONSEQUENCES = [
    "missense_variant",
    "stop_gained",
    "frameshift_variant",
    "inframe_deletion",
    "inframe_insertion",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "start_lost",
    "stop_lost",
]

# Rate limit: seconds between API calls
API_DELAY = 0.4

# Retry configuration
MAX_RETRIES = 3
CONNECT_TIMEOUT = 15  # seconds for initial connection
READ_TIMEOUT = 45  # seconds for reading response


def check_gdc_connectivity() -> bool:
    """Pre-flight check: verify we can reach the GDC API."""
    log.info("Pre-flight connectivity check: %s/status", GDC_BASE)
    try:
        resp = requests.get(
            f"{GDC_BASE}/status",
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        if resp.status_code == 200:
            status = resp.json()
            log.info("  GDC API reachable. Status: %s", status.get("status", "unknown"))
            return True
        log.warning("  GDC API returned HTTP %d", resp.status_code)
        return False
    except requests.ConnectionError as e:
        log.error("  Cannot connect to GDC API: %s", e)
        log.error("  This may be caused by VPN/firewall settings or network issues.")
        log.error("  Check: curl -v --connect-timeout 15 https://api.gdc.cancer.gov/status")
        return False
    except requests.Timeout:
        log.error("  GDC API connection timed out after %ds.", CONNECT_TIMEOUT)
        log.error("  The API may be down or blocked by your network configuration.")
        return False
    except requests.RequestException as e:
        log.error("  GDC API pre-flight failed: %s", e)
        return False


def gdc_request(
    endpoint: str,
    params: dict | None = None,
    method: str = "GET",
    json_body: dict | None = None,
    timeout: int | None = None,
) -> dict | None:
    """Make a GDC API request with retry logic and rate limiting."""
    url = f"{GDC_BASE}/{endpoint.lstrip('/')}"
    if timeout is None:
        timeout = (CONNECT_TIMEOUT, READ_TIMEOUT)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if method == "POST":
                resp = requests.post(url, json=json_body, timeout=timeout)
            else:
                resp = requests.get(url, params=params, timeout=timeout)

            if resp.status_code == 200:
                time.sleep(API_DELAY)
                return resp.json()
            log.warning("  %s returned HTTP %d (attempt %d/%d): %s",
                        endpoint, resp.status_code, attempt, MAX_RETRIES, resp.text[:200])
        except requests.RequestException as e:
            log.warning("  Request error for %s (attempt %d/%d): %s",
                        endpoint, attempt, MAX_RETRIES, e)

        if attempt < MAX_RETRIES:
            backoff = 2 ** attempt
            log.info("  Retrying in %ds...", backoff)
            time.sleep(backoff)

    time.sleep(API_DELAY)
    return None


# ===================================================================
# Step 1: Discover all hematological/myeloid GDC projects
# ===================================================================
def discover_hematological_projects() -> list[dict]:
    """Find all GDC projects with hematological/myeloid primary sites."""
    log.info("=" * 70)
    log.info("STEP 1: Discovering hematological GDC projects")

    # Get all projects with their primary sites
    data = gdc_request(
        "projects",
        params={
            "size": 1000,
            "fields": "project_id,name,primary_site,disease_type,summary.case_count",
        },
    )

    if not data:
        log.warning("  Failed to fetch projects; using known list only")
        return [{"project_id": p} for p in KNOWN_MYELOID_PROJECTS]

    hits = data.get("data", {}).get("hits", [])
    log.info("  Total GDC projects found: %d", len(hits))

    heme_projects = []
    for proj in hits:
        pid = proj.get("project_id", "")
        primary_sites = proj.get("primary_site", [])
        if isinstance(primary_sites, str):
            primary_sites = [primary_sites]
        disease_types = proj.get("disease_type", [])
        if isinstance(disease_types, str):
            disease_types = [disease_types]
        name = proj.get("name", "")
        case_count = proj.get("summary", {}).get("case_count", 0) if isinstance(proj.get("summary"), dict) else 0

        # Check if this is a hematological project
        is_heme = False

        # Check primary site
        for site in primary_sites:
            if site and site.lower() in HEMATOLOGICAL_PRIMARY_SITES:
                is_heme = True
                break

        # Check disease type keywords
        heme_keywords = [
            "leukemia", "lymphoma", "myeloma", "myeloid", "myelodysplastic",
            "myeloproliferative", "aml", "all", "cll", "cml",
            "hematopoietic", "lymphoid",
        ]
        for dt in disease_types:
            if dt and any(kw in dt.lower() for kw in heme_keywords):
                is_heme = True
                break

        # Check project name
        if any(kw in name.lower() for kw in ["leukemia", "lymphoma", "myeloid", "aml"]):
            is_heme = True

        # Always include known projects
        if pid in KNOWN_MYELOID_PROJECTS:
            is_heme = True

        if is_heme:
            heme_projects.append({
                "project_id": pid,
                "name": name,
                "primary_site": primary_sites,
                "disease_type": disease_types,
                "case_count": case_count,
            })

    # Sort by case count descending
    heme_projects.sort(key=lambda x: x.get("case_count", 0), reverse=True)

    log.info("  Hematological projects found: %d", len(heme_projects))
    for p in heme_projects:
        log.info("    %s: %s (n=%d)", p["project_id"], p.get("name", ""), p.get("case_count", 0))

    return heme_projects


# ===================================================================
# Step 2: Count cases per project
# ===================================================================
def count_project_cases(project_id: str) -> int:
    """Get total case count for a GDC project."""
    filt = {
        "op": "=",
        "content": {
            "field": "project.project_id",
            "value": project_id,
        },
    }
    data = gdc_request(
        "cases",
        params={"filters": json.dumps(filt), "size": 0},
    )
    if data:
        return data.get("data", {}).get("pagination", {}).get("total", 0)
    return 0


# ===================================================================
# Step 3: Query gene-level SSM counts per project
# ===================================================================
def query_gene_ssms_for_project(project_id: str, gene: str) -> dict:
    """Query SSM occurrences for a specific gene in a specific project.

    Uses the /analysis/top_mutated_genes_by_project endpoint for gene-level
    counts, which is more reliable than filtering /ssms by gene symbol.
    Falls back to /cases with gene mutation filter.

    Returns dict with total SSM count.
    """
    # Strategy 1: Use /cases endpoint filtered by gene mutations
    # This is the most reliable way to count cases with mutations in a gene
    case_filter = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "project.project_id",
                    "value": project_id,
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "genes.symbol",
                    "value": gene,
                },
            },
        ],
    }

    data = gdc_request(
        "analysis/top_mutated_genes_by_project",
        params={
            "filters": json.dumps({
                "op": "=",
                "content": {
                    "field": "project.project_id",
                    "value": project_id,
                },
            }),
            "size": 500,
        },
    )

    total_ssms = 0
    if data:
        # Search gene list for our target gene
        for gene_entry in data.get("data", {}).get("hits", []):
            if gene_entry.get("symbol") == gene:
                total_ssms = gene_entry.get("_score", 0)
                break

    return {"gene": gene, "project": project_id, "total_ssms": total_ssms}


# ===================================================================
# Step 4: Search for exact variants via ssm_occurrences
# ===================================================================
def search_exact_variant_in_project(project_id: str, gene: str, aa_change: str) -> dict:
    """Search for a specific amino acid change in a project.

    Two-strategy approach:
    1. Search /ssms by genomic_dna_change (GRCh38 coordinates) — most reliable
    2. Fall back to /ssm_occurrences filtered by project + gene

    Returns case IDs and variant details.
    """
    variant_info = EXACT_VARIANTS.get(gene, {})
    genomic_change = variant_info.get("genomic_dna_change", "")

    # Strategy 1: Search by exact genomic coordinate (most reliable)
    if genomic_change:
        ssm_filter = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "genomic_dna_change",
                        "value": genomic_change,
                    },
                },
            ],
        }

        # First find the SSM ID for this exact variant
        ssm_data = gdc_request(
            "ssms",
            params={
                "filters": json.dumps(ssm_filter),
                "fields": "ssm_id,genomic_dna_change,consequence.transcript.gene.symbol,consequence.transcript.aa_change,occurrence.case.project.project_id,occurrence.case.case_id,occurrence.case.submitter_id",
                "size": 10,
            },
        )

        if ssm_data:
            hits = ssm_data.get("data", {}).get("hits", [])
            if hits:
                ssm = hits[0]
                ssm_id = ssm.get("ssm_id", "")
                log.info("      Found SSM ID %s for %s %s", ssm_id, gene, aa_change)

                # Now get all occurrences of this SSM filtered by project
                occ_filter = {
                    "op": "and",
                    "content": [
                        {
                            "op": "=",
                            "content": {
                                "field": "ssm.ssm_id",
                                "value": ssm_id,
                            },
                        },
                        {
                            "op": "=",
                            "content": {
                                "field": "case.project.project_id",
                                "value": project_id,
                            },
                        },
                    ],
                }

                fields = [
                    "case.case_id",
                    "case.submitter_id",
                    "case.project.project_id",
                    "ssm.ssm_id",
                    "ssm.genomic_dna_change",
                    "ssm.consequence.transcript.aa_change",
                    "ssm.consequence.transcript.gene.symbol",
                ]

                all_hits = []
                offset = 0
                page_size = 100

                while True:
                    data = gdc_request(
                        "ssm_occurrences",
                        params={
                            "filters": json.dumps(occ_filter),
                            "fields": ",".join(fields),
                            "size": page_size,
                            "from": offset,
                        },
                    )

                    if not data:
                        break

                    hits_page = data.get("data", {}).get("hits", [])
                    if not hits_page:
                        break

                    for hit in hits_page:
                        case = hit.get("case", {})
                        all_hits.append({
                            "case_id": case.get("case_id", ""),
                            "submitter_id": case.get("submitter_id", ""),
                            "project": case.get("project", {}).get("project_id", ""),
                            "ssm_id": ssm_id,
                            "genomic_dna_change": genomic_change,
                            "aa_change": aa_change,
                            "gene": gene,
                        })

                    total = data.get("data", {}).get("pagination", {}).get("total", 0)
                    offset += page_size
                    if offset >= total:
                        break

                return {
                    "gene": gene,
                    "project": project_id,
                    "aa_change": aa_change,
                    "genomic_dna_change": genomic_change,
                    "total_gene_ssm_occurrences": len(all_hits),
                    "exact_variant_matches": len(all_hits),
                    "cases": all_hits,
                }

    # Strategy 2: Fall back to gene-level ssm_occurrences with aa_change filtering
    ssm_filter = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "case.project.project_id",
                    "value": project_id,
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "ssm.consequence.transcript.gene.symbol",
                    "value": gene,
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "ssm.consequence.transcript.consequence_type",
                    "value": ["missense_variant"],
                },
            },
        ],
    }

    fields = [
        "case.case_id",
        "case.submitter_id",
        "case.project.project_id",
        "ssm.consequence.transcript.aa_change",
        "ssm.consequence.transcript.gene.symbol",
        "ssm.ssm_id",
        "ssm.genomic_dna_change",
    ]

    all_hits = []
    offset = 0
    page_size = 100

    while True:
        data = gdc_request(
            "ssm_occurrences",
            params={
                "filters": json.dumps(ssm_filter),
                "fields": ",".join(fields),
                "size": page_size,
                "from": offset,
            },
        )

        if not data:
            break

        hits = data.get("data", {}).get("hits", [])
        if not hits:
            break

        all_hits.extend(hits)
        total = data.get("data", {}).get("pagination", {}).get("total", 0)

        offset += page_size
        if offset >= total:
            break

    # Filter for exact amino acid change
    matching_cases = []
    for hit in all_hits:
        ssm = hit.get("ssm", {})
        consequences = ssm.get("consequence", [])
        case_info = hit.get("case", {})

        for cons in consequences:
            transcript = cons.get("transcript", {})
            aa = transcript.get("aa_change", "")
            gene_sym = transcript.get("gene", {}).get("symbol", "")

            if gene_sym == gene and aa_change in str(aa):
                matching_cases.append({
                    "case_id": case_info.get("case_id", ""),
                    "submitter_id": case_info.get("submitter_id", ""),
                    "project_id": case_info.get("project", {}).get("project_id", project_id),
                    "aa_change": aa,
                    "ssm_id": ssm.get("ssm_id", ""),
                    "genomic_dna_change": ssm.get("genomic_dna_change", ""),
                })
                break

    return {
        "gene": gene,
        "aa_change": aa_change,
        "project": project_id,
        "total_gene_ssm_occurrences": len(all_hits),
        "exact_variant_matches": len(matching_cases),
        "cases": matching_cases,
    }


# ===================================================================
# Step 5: Case-level co-occurrence analysis
# ===================================================================
def get_case_mutations(case_id: str) -> list[dict]:
    """Get all SSMs for a specific case."""
    filt = {
        "op": "=",
        "content": {
            "field": "case.case_id",
            "value": case_id,
        },
    }

    fields = [
        "ssm.consequence.transcript.gene.symbol",
        "ssm.consequence.transcript.aa_change",
        "ssm.consequence.transcript.consequence_type",
        "ssm.ssm_id",
    ]

    data = gdc_request(
        "ssm_occurrences",
        params={
            "filters": json.dumps(filt),
            "fields": ",".join(fields),
            "size": 500,
        },
    )

    if not data:
        return []

    mutations = []
    for hit in data.get("data", {}).get("hits", []):
        ssm = hit.get("ssm", {})
        for cons in ssm.get("consequence", []):
            transcript = cons.get("transcript", {})
            gene_sym = transcript.get("gene", {}).get("symbol", "")
            aa = transcript.get("aa_change", "")
            cons_type = transcript.get("consequence_type", "")
            if gene_sym in TARGET_GENES:
                mutations.append({
                    "gene": gene_sym,
                    "aa_change": aa,
                    "consequence_type": cons_type,
                    "ssm_id": ssm.get("ssm_id", ""),
                })

    return mutations


def analyze_cooccurrence(all_variant_results: list[dict]) -> dict:
    """Analyze donor-level co-occurrence across all exact variant matches.

    For each case that has at least one of our target variants, check
    what other target gene mutations they carry.
    """
    log.info("=" * 70)
    log.info("STEP 5: Donor-level co-occurrence analysis")

    # Collect all unique case IDs from exact variant matches
    case_to_variants: dict[str, dict] = {}  # case_id -> {gene: aa_change, ...}
    case_metadata: dict[str, dict] = {}

    for vr in all_variant_results:
        for case in vr.get("cases", []):
            cid = case["case_id"]
            if cid not in case_to_variants:
                case_to_variants[cid] = {}
                case_metadata[cid] = {
                    "submitter_id": case.get("submitter_id", ""),
                    "project_id": case.get("project_id", ""),
                }
            case_to_variants[cid][vr["gene"]] = case.get("aa_change", vr["aa_change"])

    log.info("  Unique cases with at least one exact variant: %d", len(case_to_variants))

    # For each case with exact variant matches, fetch full target-gene mutations
    enriched_cases = []
    for cid, initial_variants in case_to_variants.items():
        meta = case_metadata[cid]
        log.info("  Fetching mutations for case %s (%s)", meta["submitter_id"], meta["project_id"])

        all_mutations = get_case_mutations(cid)

        # Build gene -> list of variants
        gene_variants: dict[str, list[str]] = defaultdict(list)
        for mut in all_mutations:
            gene_variants[mut["gene"]].append(mut["aa_change"])

        # Count how many of our 4 target genes are mutated
        genes_hit = [g for g in TARGET_GENES if g in gene_variants]

        enriched_cases.append({
            "case_id": cid,
            "submitter_id": meta["submitter_id"],
            "project_id": meta["project_id"],
            "initial_exact_variants": initial_variants,
            "all_target_gene_mutations": dict(gene_variants),
            "n_target_genes_mutated": len(genes_hit),
            "target_genes_mutated": genes_hit,
        })

    # Sort by number of target genes mutated (most first)
    enriched_cases.sort(key=lambda x: x["n_target_genes_mutated"], reverse=True)

    # Summary statistics
    cooccurrence_counts = defaultdict(int)
    for ec in enriched_cases:
        n = ec["n_target_genes_mutated"]
        cooccurrence_counts[n] += 1

    # Check for quadruple
    quadruple_cases = [ec for ec in enriched_cases if ec["n_target_genes_mutated"] == 4]
    triple_cases = [ec for ec in enriched_cases if ec["n_target_genes_mutated"] == 3]
    double_cases = [ec for ec in enriched_cases if ec["n_target_genes_mutated"] == 2]

    log.info("  Co-occurrence summary:")
    log.info("    4 genes (quadruple): %d cases", len(quadruple_cases))
    log.info("    3 genes (triple):    %d cases", len(triple_cases))
    log.info("    2 genes (double):    %d cases", len(double_cases))
    log.info("    1 gene only:         %d cases", cooccurrence_counts.get(1, 0))

    return {
        "total_unique_cases_with_exact_variants": len(case_to_variants),
        "cooccurrence_distribution": dict(cooccurrence_counts),
        "quadruple_cases": quadruple_cases,
        "triple_cases": triple_cases,
        "double_cases": double_cases,
        "all_enriched_cases": enriched_cases,
    }


# ===================================================================
# Step 6: Alternative approach -- query cases endpoint with gene filters
# ===================================================================
def query_cases_with_gene_mutations(project_id: str) -> dict:
    """Query cases that have mutations in ANY of our 4 target genes.

    Uses the /cases endpoint with SSM gene filter to get per-case counts.
    This is a complementary approach to ssm_occurrences.
    """
    gene_case_counts = {}

    for gene in TARGET_GENES:
        filt = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "project.project_id",
                        "value": project_id,
                    },
                },
                {
                    "op": "=",
                    "content": {
                        "field": "ssm_occurrences.ssm.consequence.transcript.gene.symbol",
                        "value": gene,
                    },
                },
            ],
        }

        data = gdc_request(
            "cases",
            params={
                "filters": json.dumps(filt),
                "size": 0,
            },
        )

        if data:
            total = data.get("data", {}).get("pagination", {}).get("total", 0)
            gene_case_counts[gene] = total
        else:
            gene_case_counts[gene] = 0

    return gene_case_counts


# ===================================================================
# Main pipeline
# ===================================================================
def run_expanded_gdc_pipeline() -> dict:
    """Run the full expanded GDC query pipeline."""
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    log.info("=" * 70)
    log.info("GDC EXPANDED QUERY PIPELINE")
    log.info("Timestamp: %s", timestamp)
    log.info("=" * 70)

    # Step 1: Discover projects
    heme_projects = discover_hematological_projects()

    if not heme_projects:
        log.warning("No hematological projects found. Using fallback list.")
        heme_projects = [{"project_id": p, "name": p, "case_count": 0} for p in KNOWN_MYELOID_PROJECTS]

    # Step 2: For each project, get case counts and gene-level SSM data
    log.info("=" * 70)
    log.info("STEP 2-3: Per-project case counts and gene-level SSM queries")

    project_results = []
    for proj in heme_projects:
        pid = proj["project_id"]
        log.info("  --- Project: %s ---", pid)

        # Case count
        case_count = proj.get("case_count", 0)
        if case_count == 0:
            case_count = count_project_cases(pid)

        # Gene-level SSM counts
        gene_ssm_counts = {}
        for gene in TARGET_GENES:
            result = query_gene_ssms_for_project(pid, gene)
            gene_ssm_counts[gene] = result["total_ssms"]

        # Cases with gene mutations (complementary check)
        gene_case_counts = query_cases_with_gene_mutations(pid)

        project_results.append({
            "project_id": pid,
            "name": proj.get("name", ""),
            "primary_site": proj.get("primary_site", []),
            "disease_type": proj.get("disease_type", []),
            "total_cases": case_count,
            "gene_ssm_counts": gene_ssm_counts,
            "gene_case_counts": gene_case_counts,
            "has_any_target_gene": any(v > 0 for v in gene_case_counts.values()),
        })

        log.info("    Cases: %d | Gene SSMs: %s | Gene cases: %s",
                 case_count, gene_ssm_counts, gene_case_counts)

    # Step 4: Global exact variant search by genomic coordinates
    # This bypasses the per-project gene-level counts (which can return 0
    # due to GDC API field indexing quirks) and searches directly for each
    # variant's GRCh38 genomic_dna_change across ALL GDC projects at once.
    log.info("=" * 70)
    log.info("STEP 4: Global exact variant search by genomic coordinates")

    all_variant_results = []
    heme_project_ids = {pr["project_id"] for pr in project_results}

    for gene, variant_info in EXACT_VARIANTS.items():
        genomic_change = variant_info.get("genomic_dna_change", "")
        aa_change = variant_info["aa_change"]
        log.info("  Searching %s %s (%s) across ALL GDC ...", gene, aa_change, genomic_change)

        # Search /ssms by genomic_dna_change to find the SSM ID
        ssm_filter = {
            "op": "=",
            "content": {
                "field": "genomic_dna_change",
                "value": genomic_change,
            },
        }

        ssm_data = gdc_request(
            "ssms",
            params={
                "filters": json.dumps(ssm_filter),
                "fields": "ssm_id,genomic_dna_change,consequence.transcript.gene.symbol,consequence.transcript.aa_change",
                "size": 10,
            },
        )

        if not ssm_data or not ssm_data.get("data", {}).get("hits"):
            log.warning("    No SSM found for %s in GDC", genomic_change)
            all_variant_results.append({
                "gene": gene,
                "project": "ALL",
                "aa_change": aa_change,
                "genomic_dna_change": genomic_change,
                "total_gene_ssm_occurrences": 0,
                "exact_variant_matches": 0,
                "cases": [],
            })
            continue

        ssm = ssm_data["data"]["hits"][0]
        ssm_id = ssm.get("ssm_id", "")
        log.info("    Found SSM ID: %s", ssm_id)

        # Get ALL occurrences of this SSM across ALL projects
        occ_filter = {
            "op": "=",
            "content": {
                "field": "ssm.ssm_id",
                "value": ssm_id,
            },
        }

        fields = [
            "case.case_id",
            "case.submitter_id",
            "case.project.project_id",
            "ssm.ssm_id",
            "ssm.genomic_dna_change",
        ]

        all_hits = []
        page_offset = 0
        page_size = 100

        while True:
            data = gdc_request(
                "ssm_occurrences",
                params={
                    "filters": json.dumps(occ_filter),
                    "fields": ",".join(fields),
                    "size": page_size,
                    "from": page_offset,
                },
            )

            if not data:
                break

            hits_page = data.get("data", {}).get("hits", [])
            if not hits_page:
                break

            for hit in hits_page:
                case = hit.get("case", {})
                proj = case.get("project", {}).get("project_id", "")
                all_hits.append({
                    "case_id": case.get("case_id", ""),
                    "submitter_id": case.get("submitter_id", ""),
                    "project_id": proj,
                    "ssm_id": ssm_id,
                    "genomic_dna_change": genomic_change,
                    "aa_change": aa_change,
                    "gene": gene,
                    "is_hematological": proj in heme_project_ids,
                })

            total = data.get("data", {}).get("pagination", {}).get("total", 0)
            page_offset += page_size
            if page_offset >= total:
                break

        heme_hits = [h for h in all_hits if h["is_hematological"]]
        log.info("    Total occurrences: %d | Hematological: %d", len(all_hits), len(heme_hits))

        # Group by project
        by_project: dict[str, int] = defaultdict(int)
        for h in all_hits:
            by_project[h["project_id"]] += 1
        for proj_id, count in sorted(by_project.items(), key=lambda x: -x[1])[:10]:
            log.info("      %s: %d cases", proj_id, count)

        all_variant_results.append({
            "gene": gene,
            "project": "ALL",
            "aa_change": aa_change,
            "genomic_dna_change": genomic_change,
            "total_gene_ssm_occurrences": len(all_hits),
            "exact_variant_matches": len(all_hits),
            "hematological_matches": len(heme_hits),
            "cases": all_hits,
            "by_project": dict(by_project),
        })

    # Also search per-project for projects with gene-level hits (fallback)
    projects_with_hits = [pr for pr in project_results if pr["has_any_target_gene"]]
    if projects_with_hits:
        log.info("  Additionally checking %d projects with gene-level hits...", len(projects_with_hits))
        for pr in projects_with_hits:
            pid = pr["project_id"]
            for gene_name, vi in EXACT_VARIANTS.items():
                if pr["gene_case_counts"].get(gene_name, 0) > 0:
                    log.info("    Per-project: %s %s in %s", gene_name, vi["aa_change"], pid)

    # Step 5: Co-occurrence analysis
    cooccurrence = analyze_cooccurrence(all_variant_results)

    # Compile final results
    elapsed = time.time() - start_time

    # Summary tables
    total_heme_cases = sum(pr["total_cases"] for pr in project_results)
    projects_with_any_target = [pr for pr in project_results if pr["has_any_target_gene"]]

    # Aggregate exact variant counts across all projects
    variant_summary = {}
    for gene, vi in EXACT_VARIANTS.items():
        matching = [vr for vr in all_variant_results if vr["gene"] == gene]
        total_exact = sum(vr["exact_variant_matches"] for vr in matching)
        variant_summary[f"{gene} {vi['aa_change']}"] = {
            "total_exact_matches": total_exact,
            "by_project": {vr["project"]: vr["exact_variant_matches"] for vr in matching},
        }

    # GENIE overlap assessment
    genie_overlap_projects = {"TCGA-LAML"}  # TCGA-LAML is known to overlap with GENIE
    potentially_unique = []
    for ec in cooccurrence["all_enriched_cases"]:
        if ec["project_id"] not in genie_overlap_projects:
            potentially_unique.append(ec)

    results = {
        "timestamp": timestamp,
        "runtime_seconds": round(elapsed, 1),
        "pipeline": "gdc_expanded",
        "api_base": GDC_BASE,
        "target_genes": TARGET_GENES,
        "exact_variants_searched": {g: v["aa_change"] for g, v in EXACT_VARIANTS.items()},
        "projects_discovered": len(heme_projects),
        "projects_with_target_gene_mutations": len(projects_with_any_target),
        "total_hematological_cases": total_heme_cases,
        "project_details": project_results,
        "exact_variant_results": all_variant_results,
        "variant_summary": variant_summary,
        "cooccurrence_analysis": {
            "total_cases_with_exact_variants": cooccurrence["total_unique_cases_with_exact_variants"],
            "distribution": cooccurrence["cooccurrence_distribution"],
            "quadruple_matches": len(cooccurrence["quadruple_cases"]),
            "triple_matches": len(cooccurrence["triple_cases"]),
            "double_matches": len(cooccurrence["double_cases"]),
            "quadruple_cases": cooccurrence["quadruple_cases"],
            "triple_cases": cooccurrence["triple_cases"],
            "double_cases": cooccurrence["double_cases"],
        },
        "genie_overlap": {
            "known_overlap_projects": list(genie_overlap_projects),
            "potentially_unique_cases": len(potentially_unique),
            "potentially_unique_details": potentially_unique,
        },
        "all_enriched_cases": cooccurrence["all_enriched_cases"],
    }

    return results


def generate_report(results: dict) -> str:
    """Generate a markdown report from the results."""
    lines = [
        "# GDC Expanded Query Report",
        "",
        f"**Generated:** {results['timestamp']}",
        f"**Runtime:** {results['runtime_seconds']}s",
        f"**API:** {results['api_base']}",
        "",
        "## Objective",
        "",
        "Search ALL GDC hematological/myeloid projects for the patient's exact mutation",
        "combination: DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q.",
        "The existing cross_database.py only queries TCGA-LAML with gene-level counts.",
        "This expanded search covers all discoverable myeloid projects and performs",
        "variant-specific, case-level co-occurrence analysis.",
        "",
        "---",
        "",
        "## 1. Hematological Projects Discovered",
        "",
        f"**Total hematological projects:** {results['projects_discovered']}",
        f"**Total cases across all projects:** {results['total_hematological_cases']}",
        f"**Projects with target gene mutations:** {results['projects_with_target_gene_mutations']}",
        "",
        "| Project | Name | Cases | DNMT3A | IDH2 | SETBP1 | PTPN11 |",
        "|---------|------|------:|-------:|-----:|-------:|-------:|",
    ]

    for pr in results["project_details"]:
        gc = pr["gene_case_counts"]
        lines.append(
            f"| {pr['project_id']} | {pr.get('name', '')[:40]} | "
            f"{pr['total_cases']} | "
            f"{gc.get('DNMT3A', 0)} | "
            f"{gc.get('IDH2', 0)} | "
            f"{gc.get('SETBP1', 0)} | "
            f"{gc.get('PTPN11', 0)} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Exact Variant Search",
        "",
        "Searched for the patient's exact amino acid changes:",
        "",
        "| Variant | Total Exact Matches | By Project |",
        "|---------|--------------------:|------------|",
    ])

    for variant_name, vs in results["variant_summary"].items():
        by_proj = ", ".join(f"{p}: {c}" for p, c in vs["by_project"].items() if c > 0)
        if not by_proj:
            by_proj = "none"
        lines.append(f"| {variant_name} | {vs['total_exact_matches']} | {by_proj} |")

    cooc = results["cooccurrence_analysis"]
    lines.extend([
        "",
        "---",
        "",
        "## 3. Co-occurrence Analysis",
        "",
        f"**Cases with at least one exact variant:** {cooc['total_cases_with_exact_variants']}",
        "",
        "### Distribution of target gene co-mutations",
        "",
        "| # Target Genes Mutated | Cases |",
        "|:----------------------:|------:|",
    ])

    for n_genes in sorted(cooc["distribution"].keys(), reverse=True):
        count = cooc["distribution"][n_genes]
        label = {4: "Quadruple", 3: "Triple", 2: "Double", 1: "Single"}.get(n_genes, str(n_genes))
        lines.append(f"| {n_genes} ({label}) | {count} |")

    lines.extend([
        "",
        f"**Quadruple co-occurrence (all 4 genes):** {cooc['quadruple_matches']} cases",
        f"**Triple co-occurrence (3 of 4 genes):** {cooc['triple_matches']} cases",
        f"**Double co-occurrence (2 of 4 genes):** {cooc['double_matches']} cases",
        "",
    ])

    if cooc["quadruple_matches"] > 0:
        lines.extend(["### Quadruple match cases", ""])
        for qc in cooc["quadruple_cases"]:
            lines.append(f"- **{qc['submitter_id']}** ({qc['project_id']}): {qc['all_target_gene_mutations']}")
        lines.append("")

    if cooc["triple_matches"] > 0:
        lines.extend(["### Triple match cases", ""])
        for tc in cooc["triple_cases"]:
            lines.append(f"- **{tc['submitter_id']}** ({tc['project_id']}): "
                         f"genes={tc['target_genes_mutated']}, variants={tc['all_target_gene_mutations']}")
        lines.append("")

    if cooc["double_matches"] > 0:
        lines.extend(["### Double match cases", ""])
        for dc in cooc["double_cases"]:
            lines.append(f"- **{dc['submitter_id']}** ({dc['project_id']}): "
                         f"genes={dc['target_genes_mutated']}, variants={dc['all_target_gene_mutations']}")
        lines.append("")

    # GENIE overlap
    genie_ov = results["genie_overlap"]
    lines.extend([
        "---",
        "",
        "## 4. GENIE Overlap Assessment",
        "",
        f"**Known overlap projects:** {', '.join(genie_ov['known_overlap_projects'])}",
        "",
        "TCGA-LAML patients are also present in GENIE (via DFCI/MSK contributions).",
        "TARGET-AML and BEATAML patients are generally NOT in GENIE v19.0,",
        "making them potentially unique additions to the cross-database search.",
        "",
        f"**Potentially unique cases (not in GENIE):** {genie_ov['potentially_unique_cases']}",
        "",
    ])

    if genie_ov["potentially_unique_cases"] > 0:
        lines.extend(["### Potentially unique cases", ""])
        for uc in genie_ov["potentially_unique_details"]:
            lines.append(f"- **{uc['submitter_id']}** ({uc['project_id']}): "
                         f"{uc['n_target_genes_mutated']} target genes, "
                         f"variants={uc['all_target_gene_mutations']}")
        lines.append("")

    # Detailed case list
    all_cases = results.get("all_enriched_cases", [])
    if all_cases:
        lines.extend([
            "---",
            "",
            "## 5. All Cases with Exact Patient Variants",
            "",
            "| Case | Project | # Genes | Genes | Variants |",
            "|------|---------|:-------:|-------|----------|",
        ])
        for ec in all_cases:
            variants_str = "; ".join(f"{g}: {'/'.join(v)}" for g, v in ec["all_target_gene_mutations"].items())
            lines.append(
                f"| {ec['submitter_id']} | {ec['project_id']} | "
                f"{ec['n_target_genes_mutated']} | "
                f"{', '.join(ec['target_genes_mutated'])} | "
                f"{variants_str} |"
            )
        lines.append("")

    # Conclusion
    lines.extend([
        "---",
        "",
        "## 6. Conclusion",
        "",
    ])

    if cooc["quadruple_matches"] == 0:
        lines.extend([
            f"No cases with all 4 exact patient variants (DNMT3A R882H + IDH2 R140Q + "
            f"SETBP1 G870S + PTPN11 E76Q) were found across {results['projects_discovered']} "
            f"GDC hematological projects totaling {results['total_hematological_cases']} cases.",
            "",
            "This is consistent with the GENIE v19.0 finding of 0 matches across ~14,600 "
            "myeloid patients and the cross-database finding of 0 matches across ~10,249 "
            "deduplicated myeloid patients.",
            "",
        ])
    else:
        lines.extend([
            f"**{cooc['quadruple_matches']} case(s) found with all 4 target genes mutated.**",
            "This would be a significant finding requiring detailed follow-up.",
            "",
        ])

    total_new = genie_ov["potentially_unique_cases"]
    lines.extend([
        f"**New patients beyond GENIE:** {total_new} cases from non-overlapping GDC projects "
        f"carry at least one of the patient's exact variants.",
        "",
        "---",
        "",
        "*Generated by gdc_expanded.py*",
    ])

    return "\n".join(lines)


def main():
    log.info("Starting GDC expanded query pipeline")

    # Pre-flight connectivity check
    if not check_gdc_connectivity():
        log.error("=" * 70)
        log.error("FATAL: Cannot reach GDC API at %s", GDC_BASE)
        log.error("Possible causes:")
        log.error("  1. VPN is blocking outbound connections (check Proton VPN split tunneling)")
        log.error("  2. GDC API is temporarily down (check https://status.gdc.cancer.gov)")
        log.error("  3. Network/firewall configuration issue")
        log.error("Retry after fixing connectivity.")
        log.error("=" * 70)
        return 1

    results = run_expanded_gdc_pipeline()

    # Save JSON results
    json_path = RESULTS_DIR / "gdc_expanded.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("JSON results saved to %s", json_path)

    # Generate and save report
    report = generate_report(results)
    report_path = RESULTS_DIR / "gdc_expanded_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Report saved to %s", report_path)

    # Print summary
    cooc = results["cooccurrence_analysis"]
    print("\n" + "=" * 70)
    print("GDC EXPANDED QUERY -- SUMMARY")
    print("=" * 70)
    print(f"Hematological projects discovered: {results['projects_discovered']}")
    print(f"Total cases across all projects:   {results['total_hematological_cases']}")
    print(f"Projects with target gene hits:    {results['projects_with_target_gene_mutations']}")
    print()
    print("Exact variant matches:")
    for vname, vs in results["variant_summary"].items():
        print(f"  {vname}: {vs['total_exact_matches']} matches")
    print()
    print(f"Co-occurrence: quadruple={cooc['quadruple_matches']}, "
          f"triple={cooc['triple_matches']}, double={cooc['double_matches']}")
    print(f"Potentially unique (not in GENIE): {results['genie_overlap']['potentially_unique_cases']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
