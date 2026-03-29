#!/usr/bin/env python3
"""
mastermind_search.py -- Query Genomenon Mastermind for literature evidence of patient's mutations.

Mastermind indexes 27M+ genomic variants from full-text scientific literature,
providing article counts for specific gene/variant mentions and co-mentions.

For each of the patient's 4 target variants (DNMT3A R882H, IDH2 R140Q,
SETBP1 G870S, PTPN11 E76Q), this script:
  1. Queries gene-level article counts
  2. Queries variant-level article counts
  3. Searches for pairwise co-mentions (6 pairs)
  4. Searches for triple co-mentions (4 triples)
  5. Searches for quadruple co-mention

This validates the "unprecedented" rarity claim by checking whether the
quadruple combination has EVER been described together in published literature.

Inputs:
    - Genomenon Mastermind API (remote)

Outputs:
    - mutation_profile/results/cross_database/mastermind_results.json
    - mutation_profile/results/cross_database/mastermind_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/mastermind_search.py

Runtime: ~60-90 seconds (API rate-limited with 2s delays)
Dependencies: requests
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
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

MASTERMIND_BASE = "https://mastermind.genomenon.com/api/v2"
REQUEST_DELAY = 2.0  # 2-second delay between requests
REQUEST_TIMEOUT = 30  # 30-second timeout

# API token strategies to try in order
API_TOKEN_STRATEGIES = [
    {"api_token": "free"},
    {},  # no token
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get(url: str, params: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Make a GET request to Mastermind, trying multiple auth strategies.

    Returns parsed JSON on success, None on failure.
    """
    if params is None:
        params = {}

    for strategy in API_TOKEN_STRATEGIES:
        merged = {**params, **strategy}
        try:
            resp = requests.get(url, params=merged, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (401, 403):
                log.debug("Auth strategy %s returned %d, trying next", strategy, resp.status_code)
                continue
            # Other errors -- log and try next strategy
            log.warning("Mastermind returned %d for %s (strategy=%s)", resp.status_code, url, strategy)
            continue
        except requests.exceptions.Timeout:
            log.warning("Timeout for %s (strategy=%s)", url, strategy)
            continue
        except requests.exceptions.RequestException as exc:
            log.warning("Request error for %s: %s", url, exc)
            continue

    log.error("All auth strategies failed for %s", url)
    return None


def _delay() -> None:
    """Sleep between API requests to respect rate limits."""
    time.sleep(REQUEST_DELAY)


# ---------------------------------------------------------------------------
# Mastermind API queries
# ---------------------------------------------------------------------------
def query_gene(gene: str) -> dict[str, Any]:
    """Query Mastermind for gene-level article counts."""
    log.info("Querying gene-level: %s", gene)
    url = f"{MASTERMIND_BASE}/genes"
    result = _get(url, {"gene": gene})
    _delay()

    if result is None:
        return {"gene": gene, "status": "error", "article_count": None}

    article_count = result.get("article_count", result.get("articles_count"))
    if article_count is None:
        # Try extracting from different response shapes
        article_count = result.get("count") or result.get("total")

    return {
        "gene": gene,
        "status": "ok",
        "article_count": article_count,
        "raw_response": result,
    }


def query_variant(gene: str, variant: str) -> dict[str, Any]:
    """Query Mastermind for variant-level article counts."""
    log.info("Querying variant-level: %s %s", gene, variant)
    url = f"{MASTERMIND_BASE}/variants"
    result = _get(url, {"gene": gene, "variant": variant})
    _delay()

    if result is None:
        return {"gene": gene, "variant": variant, "status": "error", "article_count": None}

    article_count = result.get("article_count", result.get("articles_count"))
    if article_count is None:
        article_count = result.get("count") or result.get("total")

    # Try to extract article URLs or IDs if available
    articles = result.get("articles", result.get("pmids", []))

    return {
        "gene": gene,
        "variant": variant,
        "status": "ok",
        "article_count": article_count,
        "articles": articles if isinstance(articles, list) else [],
        "raw_response": result,
    }


def query_gene_pair(gene1: str, gene2: str) -> dict[str, Any]:
    """Query Mastermind for co-mention of two genes in literature."""
    log.info("Querying gene pair co-mention: %s + %s", gene1, gene2)

    # Strategy 1: Use the genes endpoint with multiple genes
    url = f"{MASTERMIND_BASE}/genes"
    result = _get(url, {"gene": f"{gene1},{gene2}"})
    _delay()

    if result is not None:
        article_count = result.get("article_count", result.get("articles_count"))
        if article_count is None:
            article_count = result.get("count") or result.get("total")
        return {
            "genes": [gene1, gene2],
            "status": "ok",
            "article_count": article_count,
            "raw_response": result,
        }

    return {
        "genes": [gene1, gene2],
        "status": "error",
        "article_count": None,
    }


def query_variant_pair(gene1: str, var1: str, gene2: str, var2: str) -> dict[str, Any]:
    """Query Mastermind for co-mention of two specific variants."""
    log.info("Querying variant pair: %s %s + %s %s", gene1, var1, gene2, var2)

    # Try variant endpoint with multiple gene+variant combos
    url = f"{MASTERMIND_BASE}/variants"

    # Strategy 1: comma-separated genes with variants
    result = _get(url, {"gene": f"{gene1},{gene2}", "variant": f"{var1},{var2}"})
    _delay()

    if result is not None:
        article_count = result.get("article_count", result.get("articles_count"))
        if article_count is None:
            article_count = result.get("count") or result.get("total")
        return {
            "genes": [gene1, gene2],
            "variants": [var1, var2],
            "status": "ok",
            "article_count": article_count,
            "raw_response": result,
        }

    return {
        "genes": [gene1, gene2],
        "variants": [var1, var2],
        "status": "error",
        "article_count": None,
    }


def query_multi_gene(genes: list[str]) -> dict[str, Any]:
    """Query Mastermind for co-mention of multiple genes."""
    label = " + ".join(genes)
    log.info("Querying multi-gene co-mention: %s", label)

    url = f"{MASTERMIND_BASE}/genes"
    result = _get(url, {"gene": ",".join(genes)})
    _delay()

    if result is not None:
        article_count = result.get("article_count", result.get("articles_count"))
        if article_count is None:
            article_count = result.get("count") or result.get("total")
        return {
            "genes": genes,
            "status": "ok",
            "article_count": article_count,
            "raw_response": result,
        }

    return {
        "genes": genes,
        "status": "error",
        "article_count": None,
    }


def query_multi_variant(gene_variant_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Query Mastermind for co-mention of multiple specific variants."""
    genes = [g for g, _ in gene_variant_pairs]
    variants = [v for _, v in gene_variant_pairs]
    label = " + ".join(f"{g} {v}" for g, v in gene_variant_pairs)
    log.info("Querying multi-variant co-mention: %s", label)

    url = f"{MASTERMIND_BASE}/variants"
    result = _get(url, {"gene": ",".join(genes), "variant": ",".join(variants)})
    _delay()

    if result is not None:
        article_count = result.get("article_count", result.get("articles_count"))
        if article_count is None:
            article_count = result.get("count") or result.get("total")
        return {
            "genes": genes,
            "variants": variants,
            "status": "ok",
            "article_count": article_count,
            "raw_response": result,
        }

    return {
        "genes": genes,
        "variants": variants,
        "status": "error",
        "article_count": None,
    }


# ---------------------------------------------------------------------------
# Mastermind URL builder (for manual verification links)
# ---------------------------------------------------------------------------
def mastermind_url(gene: str, variant: str | None = None) -> str:
    """Build a Mastermind web URL for manual verification."""
    if variant:
        return f"https://mastermind.genomenon.com/detail?gene={gene}&mutation={gene}:{variant}"
    return f"https://mastermind.genomenon.com/detail?gene={gene}"


def mastermind_multi_url(genes: list[str]) -> str:
    """Build a Mastermind web URL for multi-gene search."""
    gene_str = "%20".join(genes)
    return f"https://mastermind.genomenon.com/detail?gene={gene_str}"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis() -> dict[str, Any]:
    """Run the full Mastermind literature search."""
    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "Genomenon Mastermind",
        "database_url": "https://mastermind.genomenon.com/",
        "target_genes": TARGET_GENES,
        "exact_variants": EXACT_VARIANTS,
        "gene_level": {},
        "variant_level": {},
        "pairwise_gene_comentions": {},
        "pairwise_variant_comentions": {},
        "triple_gene_comentions": {},
        "triple_variant_comentions": {},
        "quadruple_gene_comention": None,
        "quadruple_variant_comention": None,
        "errors": [],
    }

    # -----------------------------------------------------------------------
    # Step 1: Gene-level queries
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 1: Gene-level article counts")
    log.info("=" * 60)

    for gene in TARGET_GENES:
        gene_result = query_gene(gene)
        results["gene_level"][gene] = {
            "article_count": gene_result.get("article_count"),
            "status": gene_result["status"],
            "url": mastermind_url(gene),
        }
        if gene_result["status"] == "error":
            results["errors"].append(f"Gene query failed: {gene}")

    # -----------------------------------------------------------------------
    # Step 2: Variant-level queries
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 2: Variant-level article counts")
    log.info("=" * 60)

    for gene, variant in EXACT_VARIANTS.items():
        var_result = query_variant(gene, variant)
        results["variant_level"][f"{gene}_{variant}"] = {
            "gene": gene,
            "variant": variant,
            "article_count": var_result.get("article_count"),
            "articles": var_result.get("articles", []),
            "status": var_result["status"],
            "url": mastermind_url(gene, variant),
        }
        if var_result["status"] == "error":
            results["errors"].append(f"Variant query failed: {gene} {variant}")

    # -----------------------------------------------------------------------
    # Step 3: Pairwise gene co-mentions (6 pairs)
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 3: Pairwise gene co-mentions")
    log.info("=" * 60)

    for g1, g2 in combinations(TARGET_GENES, 2):
        pair_key = f"{g1}+{g2}"
        pair_result = query_gene_pair(g1, g2)
        results["pairwise_gene_comentions"][pair_key] = {
            "genes": [g1, g2],
            "article_count": pair_result.get("article_count"),
            "status": pair_result["status"],
        }
        if pair_result["status"] == "error":
            results["errors"].append(f"Gene pair query failed: {pair_key}")

    # -----------------------------------------------------------------------
    # Step 4: Pairwise variant co-mentions (6 pairs)
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 4: Pairwise variant co-mentions")
    log.info("=" * 60)

    gene_list = list(EXACT_VARIANTS.keys())
    for g1, g2 in combinations(gene_list, 2):
        v1, v2 = EXACT_VARIANTS[g1], EXACT_VARIANTS[g2]
        pair_key = f"{g1}_{v1}+{g2}_{v2}"
        pair_result = query_variant_pair(g1, v1, g2, v2)
        results["pairwise_variant_comentions"][pair_key] = {
            "genes": [g1, g2],
            "variants": [v1, v2],
            "article_count": pair_result.get("article_count"),
            "status": pair_result["status"],
        }
        if pair_result["status"] == "error":
            results["errors"].append(f"Variant pair query failed: {pair_key}")

    # -----------------------------------------------------------------------
    # Step 5: Triple gene co-mentions (4 triples)
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 5: Triple gene co-mentions")
    log.info("=" * 60)

    for triple in combinations(TARGET_GENES, 3):
        triple_key = "+".join(triple)
        triple_result = query_multi_gene(list(triple))
        results["triple_gene_comentions"][triple_key] = {
            "genes": list(triple),
            "article_count": triple_result.get("article_count"),
            "status": triple_result["status"],
        }
        if triple_result["status"] == "error":
            results["errors"].append(f"Triple gene query failed: {triple_key}")

    # -----------------------------------------------------------------------
    # Step 6: Triple variant co-mentions (4 triples)
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 6: Triple variant co-mentions")
    log.info("=" * 60)

    for triple in combinations(gene_list, 3):
        pairs = [(g, EXACT_VARIANTS[g]) for g in triple]
        triple_key = "+".join(f"{g}_{v}" for g, v in pairs)
        triple_result = query_multi_variant(pairs)
        results["triple_variant_comentions"][triple_key] = {
            "genes": list(triple),
            "variants": [v for _, v in pairs],
            "article_count": triple_result.get("article_count"),
            "status": triple_result["status"],
        }
        if triple_result["status"] == "error":
            results["errors"].append(f"Triple variant query failed: {triple_key}")

    # -----------------------------------------------------------------------
    # Step 7: Quadruple gene co-mention
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 7: Quadruple co-mentions")
    log.info("=" * 60)

    quad_gene_result = query_multi_gene(TARGET_GENES)
    results["quadruple_gene_comention"] = {
        "genes": TARGET_GENES,
        "article_count": quad_gene_result.get("article_count"),
        "status": quad_gene_result["status"],
    }
    if quad_gene_result["status"] == "error":
        results["errors"].append("Quadruple gene query failed")

    # Quadruple variant co-mention
    quad_pairs = [(g, EXACT_VARIANTS[g]) for g in gene_list]
    quad_var_result = query_multi_variant(quad_pairs)
    results["quadruple_variant_comention"] = {
        "genes": gene_list,
        "variants": [v for _, v in quad_pairs],
        "article_count": quad_var_result.get("article_count"),
        "status": quad_var_result["status"],
    }
    if quad_var_result["status"] == "error":
        results["errors"].append("Quadruple variant query failed")

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _fmt_count(count: int | None) -> str:
    """Format an article count, handling None/error."""
    if count is None:
        return "N/A (query failed)"
    return f"{count:,}"


def generate_report(results: dict[str, Any]) -> str:
    """Generate a markdown report from Mastermind results."""
    lines: list[str] = []

    lines.append("# Genomenon Mastermind Literature Search")
    lines.append("")
    lines.append(f"**Date:** {results['timestamp']}")
    lines.append(f"**Database:** {results['database']}")
    lines.append(f"**URL:** {results['database_url']}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("Genomenon Mastermind indexes 27M+ genomic variants extracted from")
    lines.append("full-text scientific literature. This search queries article counts")
    lines.append("for the patient's 4 target mutations individually and in combination")
    lines.append("to assess whether this specific mutation profile has been described")
    lines.append("in published literature.")
    lines.append("")
    lines.append("**Patient variants:**")
    for gene, variant in EXACT_VARIANTS.items():
        lines.append(f"- {gene} {variant}")
    lines.append("")

    # --- Gene-level ---
    lines.append("## 1. Gene-Level Article Counts")
    lines.append("")
    lines.append("| Gene | Articles | Mastermind Link |")
    lines.append("|------|----------|-----------------|")
    for gene in TARGET_GENES:
        info = results["gene_level"].get(gene, {})
        count = _fmt_count(info.get("article_count"))
        url = info.get("url", mastermind_url(gene))
        lines.append(f"| {gene} | {count} | [link]({url}) |")
    lines.append("")

    # --- Variant-level ---
    lines.append("## 2. Variant-Level Article Counts")
    lines.append("")
    lines.append("| Gene | Variant | Articles | Mastermind Link |")
    lines.append("|------|---------|----------|-----------------|")
    for gene, variant in EXACT_VARIANTS.items():
        key = f"{gene}_{variant}"
        info = results["variant_level"].get(key, {})
        count = _fmt_count(info.get("article_count"))
        url = info.get("url", mastermind_url(gene, variant))
        lines.append(f"| {gene} | {variant} | {count} | [link]({url}) |")
    lines.append("")

    # --- Pairwise gene co-mentions ---
    lines.append("## 3. Pairwise Gene Co-Mentions")
    lines.append("")
    lines.append("Articles mentioning both genes together in the same publication.")
    lines.append("")
    lines.append("| Gene Pair | Articles |")
    lines.append("|-----------|----------|")
    for pair_key, info in results["pairwise_gene_comentions"].items():
        count = _fmt_count(info.get("article_count"))
        lines.append(f"| {pair_key} | {count} |")
    lines.append("")

    # --- Pairwise variant co-mentions ---
    lines.append("## 4. Pairwise Variant Co-Mentions")
    lines.append("")
    lines.append("Articles mentioning both specific variants together.")
    lines.append("")
    lines.append("| Variant Pair | Articles |")
    lines.append("|--------------|----------|")
    for pair_key, info in results["pairwise_variant_comentions"].items():
        count = _fmt_count(info.get("article_count"))
        display = pair_key.replace("_", " ").replace("+", " + ")
        lines.append(f"| {display} | {count} |")
    lines.append("")

    # --- Triple gene co-mentions ---
    lines.append("## 5. Triple Gene Co-Mentions")
    lines.append("")
    lines.append("Articles mentioning three of the four target genes together.")
    lines.append("")
    lines.append("| Triple | Articles |")
    lines.append("|--------|----------|")
    for triple_key, info in results["triple_gene_comentions"].items():
        count = _fmt_count(info.get("article_count"))
        lines.append(f"| {triple_key} | {count} |")
    lines.append("")

    # --- Triple variant co-mentions ---
    lines.append("## 6. Triple Variant Co-Mentions")
    lines.append("")
    lines.append("Articles mentioning three specific variants together.")
    lines.append("")
    lines.append("| Triple | Articles |")
    lines.append("|--------|----------|")
    for triple_key, info in results["triple_variant_comentions"].items():
        count = _fmt_count(info.get("article_count"))
        display = triple_key.replace("_", " ").replace("+", " + ")
        lines.append(f"| {display} | {count} |")
    lines.append("")

    # --- Quadruple ---
    lines.append("## 7. Quadruple Co-Mention")
    lines.append("")
    lines.append("**The critical question: has the combination of all 4 genes/variants")
    lines.append("ever been described together in published literature?**")
    lines.append("")

    quad_gene = results.get("quadruple_gene_comention", {})
    quad_var = results.get("quadruple_variant_comention", {})

    quad_gene_count = quad_gene.get("article_count")
    quad_var_count = quad_var.get("article_count")

    lines.append(f"- **All 4 genes (DNMT3A + IDH2 + SETBP1 + PTPN11):** {_fmt_count(quad_gene_count)} articles")
    lines.append(f"- **All 4 exact variants (R882H + R140Q + G870S + E76Q):** {_fmt_count(quad_var_count)} articles")
    lines.append("")

    # --- Interpretation ---
    lines.append("## 8. Interpretation")
    lines.append("")

    if quad_gene_count is not None and quad_gene_count == 0:
        lines.append("The quadruple combination of DNMT3A + IDH2 + SETBP1 + PTPN11 has")
        lines.append("**zero co-mentions** in the entire Mastermind-indexed literature corpus")
        lines.append("(27M+ variants from full-text articles). This confirms the combination")
        lines.append("is unprecedented in published literature.")
    elif quad_gene_count is not None and quad_gene_count > 0:
        lines.append(f"The quadruple gene combination was found in {quad_gene_count} article(s).")
        lines.append("Manual review of these articles is recommended to determine whether")
        lines.append("they describe co-occurrence in the same patient or merely discuss")
        lines.append("these genes in the same review/study context.")
    else:
        lines.append("The quadruple query could not be completed (API error).")
        lines.append("Manual verification via the Mastermind web interface is recommended.")
    lines.append("")

    if quad_var_count is not None and quad_var_count == 0:
        lines.append("At the exact variant level (R882H + R140Q + G870S + E76Q), there are")
        lines.append("**zero co-mentions**, reinforcing that this specific mutation profile")
        lines.append("has never been described in scientific literature.")
    elif quad_var_count is not None and quad_var_count > 0:
        lines.append(f"At the exact variant level, {quad_var_count} article(s) were found.")
        lines.append("These should be reviewed to determine clinical relevance.")
    lines.append("")

    # --- Errors ---
    if results["errors"]:
        lines.append("## Errors and Limitations")
        lines.append("")
        lines.append("The following queries encountered errors:")
        lines.append("")
        for err in results["errors"]:
            lines.append(f"- {err}")
        lines.append("")
        lines.append("Failed queries may indicate API access restrictions. The Mastermind")
        lines.append("free API tier has limited functionality. For complete results, consider")
        lines.append("manual verification at https://mastermind.genomenon.com/")
    lines.append("")

    # --- Methodology ---
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Database:** Genomenon Mastermind (https://mastermind.genomenon.com/)")
    lines.append("- **API version:** v2")
    lines.append("- **Query types:** Gene-level, variant-level, multi-gene co-mention")
    lines.append(f"- **Request delay:** {REQUEST_DELAY}s between requests")
    lines.append(f"- **Timeout:** {REQUEST_TIMEOUT}s per request")
    lines.append("- **Auth strategies:** api_token=free, no token (tried in order)")
    lines.append("- Article counts represent full-text literature mentions, not just")
    lines.append("  abstracts -- Mastermind parses the complete text of articles.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_results(results: dict[str, Any], report: str) -> None:
    """Save JSON results and markdown report."""
    json_path = RESULTS_DIR / "mastermind_results.json"
    report_path = RESULTS_DIR / "mastermind_report.md"

    # Strip raw_response from JSON output to keep it clean
    clean = json.loads(json.dumps(results, default=str))
    for gene_data in clean.get("gene_level", {}).values():
        gene_data.pop("raw_response", None)
    for var_data in clean.get("variant_level", {}).values():
        var_data.pop("raw_response", None)
    for pair_data in clean.get("pairwise_gene_comentions", {}).values():
        pair_data.pop("raw_response", None)
    for pair_data in clean.get("pairwise_variant_comentions", {}).values():
        pair_data.pop("raw_response", None)
    for triple_data in clean.get("triple_gene_comentions", {}).values():
        triple_data.pop("raw_response", None)
    for triple_data in clean.get("triple_variant_comentions", {}).values():
        triple_data.pop("raw_response", None)
    if clean.get("quadruple_gene_comention"):
        clean["quadruple_gene_comention"].pop("raw_response", None)
    if clean.get("quadruple_variant_comention"):
        clean["quadruple_variant_comention"].pop("raw_response", None)

    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    log.info("Saved JSON results to %s", json_path)

    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved markdown report to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the Mastermind literature search pipeline."""
    log.info("=" * 60)
    log.info("Genomenon Mastermind Literature Search")
    log.info("Target: %s", ", ".join(f"{g} {v}" for g, v in EXACT_VARIANTS.items()))
    log.info("=" * 60)

    start = time.time()

    results = run_analysis()
    report = generate_report(results)
    save_results(results, report)

    elapsed = time.time() - start
    n_errors = len(results["errors"])

    log.info("=" * 60)
    log.info("Complete in %.1f seconds", elapsed)
    if n_errors:
        log.warning("%d queries failed -- see report for details", n_errors)
    else:
        log.info("All queries succeeded")
    log.info("=" * 60)

    # Print summary to stdout
    print()
    print("=== SUMMARY ===")
    print()
    for gene, variant in EXACT_VARIANTS.items():
        gene_count = results["gene_level"].get(gene, {}).get("article_count")
        var_count = results["variant_level"].get(f"{gene}_{variant}", {}).get("article_count")
        print(f"  {gene:8s} gene:    {_fmt_count(gene_count):>10s} articles")
        print(f"  {gene:8s} {variant:5s}:  {_fmt_count(var_count):>10s} articles")

    print()
    quad_gene = results.get("quadruple_gene_comention", {}).get("article_count")
    quad_var = results.get("quadruple_variant_comention", {}).get("article_count")
    print(f"  Quadruple gene co-mention:    {_fmt_count(quad_gene)}")
    print(f"  Quadruple variant co-mention: {_fmt_count(quad_var)}")
    print()

    if n_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
