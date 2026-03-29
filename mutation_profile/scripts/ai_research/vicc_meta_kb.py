#!/usr/bin/env python3
"""
vicc_meta_kb.py -- Query VICC Meta-Knowledgebase for federated clinical evidence
across 6 knowledgebases: OncoKB, CIViC, CGI, JAX-CKB, MolecularMatch, PMKB.

The VICC (Variant Interpretation for Cancer Consortium) Meta-Knowledgebase
federates variant interpretations from multiple clinical knowledge sources,
enabling concordance analysis across databases.

Patient variants:
    1. DNMT3A R882H  (VAF 39%, pathogenic hotspot)
    2. IDH2  R140Q   (VAF 2%, pathogenic subclone)
    3. SETBP1 G870S  (VAF 34%, likely pathogenic)
    4. PTPN11 E76Q   (VAF 29%, pathogenic)
    5. EZH2  V662A   (VAF 59%, founder clone)

Data source:
    VICC Meta-Knowledgebase (https://search.cancervariants.org)
    REST API, no authentication required.

    Federated sources:
        - CIViC (Clinical Interpretation of Variants in Cancer)
        - OncoKB (Memorial Sloan Kettering)
        - CGI (Cancer Genome Interpreter)
        - JAX-CKB (Jackson Laboratory Clinical Knowledgebase)
        - PMKB (Precision Medicine Knowledgebase, Weill Cornell)
        - MolecularMatch

Outputs:
    - mutation_profile/results/ai_research/vicc_meta_kb.json
    - mutation_profile/results/ai_research/vicc_meta_kb_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/vicc_meta_kb.py

Runtime: ~30 seconds (REST queries with pagination and rate limiting)
Dependencies: requests
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VICC_API_BASE = "https://search.cancervariants.org/api/v1"
ASSOCIATIONS_URL = f"{VICC_API_BASE}/associations"
REQUEST_TIMEOUT = 30
RATE_LIMIT_DELAY = 0.5  # seconds between requests
PAGE_SIZE = 100  # max per request

# All 6 federated knowledgebases
KNOWN_SOURCES = {"CIVIC", "oncokb", "cgi", "jax", "pmkb", "molecularmatch"}

# Evidence level mapping (VICC unified levels)
EVIDENCE_LEVEL_MAP = {
    1: "A - Validated (FDA-approved / guidelines)",
    2: "B - Clinical (clinical trial / large study)",
    3: "C - Case Study (case reports / small series)",
    4: "D - Preclinical (in vivo / in vitro)",
    5: "E - Inferential (indirect / computational)",
}

EVIDENCE_LEVEL_LABELS = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}

# Patient variants to query
PATIENT_VARIANTS = [
    {"gene": "DNMT3A", "variant": "R882H", "vaf": "39%", "role": "pathogenic hotspot"},
    {"gene": "IDH2", "variant": "R140Q", "vaf": "2%", "role": "pathogenic subclone"},
    {"gene": "SETBP1", "variant": "G870S", "vaf": "34%", "role": "likely pathogenic"},
    {"gene": "PTPN11", "variant": "E76Q", "vaf": "29%", "role": "pathogenic"},
    {"gene": "EZH2", "variant": "V662A", "vaf": "59%", "role": "founder clone Pathogenic"},
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_associations(query: str) -> list[dict]:
    """Fetch all association hits for a query, handling pagination."""
    all_hits: list[dict] = []
    offset = 0

    while True:
        params = {"q": query, "size": PAGE_SIZE, "from": offset}
        resp = requests.get(ASSOCIATIONS_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        hits_block = data.get("hits", {})
        total = hits_block.get("total", 0)
        hits = hits_block.get("hits", [])

        all_hits.extend(hits)
        log.info("  Fetched %d/%d hits (offset=%d)", len(all_hits), total, offset)

        if len(all_hits) >= total or not hits:
            break

        offset += PAGE_SIZE
        time.sleep(RATE_LIMIT_DELAY)

    return all_hits


def extract_source(hit: dict) -> str:
    """Extract the primary source knowledgebase name from a hit."""
    evidence_list = hit.get("association", {}).get("evidence", [])
    for ev in evidence_list:
        src = ev.get("evidenceType", {}).get("sourceName", "")
        if src:
            return src.upper() if src == "CIVIC" else src
    # Fallback: infer from source_link
    source_link = hit.get("association", {}).get("source_link", "")
    if "civic" in source_link:
        return "CIVIC"
    if "ckb.jax" in source_link:
        return "jax"
    if "pmkb" in source_link:
        return "pmkb"
    if "oncokb" in source_link:
        return "oncokb"
    return "unknown"


def extract_pmids(hit: dict) -> list[int]:
    """Extract PubMed IDs from a hit's evidence and publication URLs."""
    pmids: list[int] = []
    assoc = hit.get("association", {})

    # From evidence.info.publications
    for ev in assoc.get("evidence", []):
        pubs = ev.get("info", {}).get("publications", [])
        for pub in pubs if isinstance(pubs, list) else [pubs]:
            if isinstance(pub, str) and "pubmed" in pub:
                parts = pub.rstrip("/").split("/")
                for part in reversed(parts):
                    if part.isdigit():
                        pmids.append(int(part))
                        break

    # From publication_url
    pub_url = assoc.get("publication_url", [])
    if isinstance(pub_url, str):
        pub_url = [pub_url]
    for url in pub_url:
        if isinstance(url, str) and "pubmed" in url:
            parts = url.rstrip("/").split("/")
            for part in reversed(parts):
                if part.isdigit():
                    pmids.append(int(part))
                    break

    return list(set(pmids))


def extract_drugs(hit: dict) -> list[dict]:
    """Extract drug/therapy information from a hit."""
    drugs: list[dict] = []
    contexts = hit.get("association", {}).get("environmentalContexts", [])
    for ctx in contexts:
        drug = {
            "name": ctx.get("term", ctx.get("description", "")),
            "id": ctx.get("id", ""),
        }
        approved = ctx.get("approved_countries", [])
        if approved:
            drug["approved_countries"] = approved
        usan = ctx.get("usan_stem", "")
        if usan:
            drug["class"] = usan
        if drug["name"]:
            drugs.append(drug)
    return drugs


def extract_diseases(hit: dict) -> list[dict]:
    """Extract disease/phenotype information from a hit."""
    diseases: list[dict] = []
    phenotypes = hit.get("association", {}).get("phenotypes", [])
    for pheno in phenotypes:
        disease = {
            "name": pheno.get("description", pheno.get("term", "")),
            "id": pheno.get("id", ""),
            "family": pheno.get("family", ""),
        }
        if disease["name"]:
            diseases.append(disease)
    return diseases


def parse_hit(hit: dict) -> dict:
    """Parse a single VICC Meta-KB hit into a structured record."""
    assoc = hit.get("association", {})
    source = extract_source(hit)
    evidence_list = assoc.get("evidence", [])

    # Evidence descriptions and types
    evidence_items = []
    for ev in evidence_list:
        evidence_items.append({
            "description": ev.get("description", ""),
            "source": ev.get("evidenceType", {}).get("sourceName", ""),
            "publications": ev.get("info", {}).get("publications", []),
        })

    # Variant info from features
    features = hit.get("features", [])
    variant_names = []
    gene_symbols = []
    for feat in features:
        gene = feat.get("geneSymbol", "")
        name = feat.get("name", "")
        if gene and gene not in gene_symbols:
            gene_symbols.append(gene)
        if name:
            variant_names.append(f"{gene} {name}" if gene else name)

    return {
        "source": source,
        "description": assoc.get("description", ""),
        "variant_name": assoc.get("variant_name", ""),
        "feature_names": hit.get("feature_names", ""),
        "response_type": assoc.get("response_type", ""),
        "evidence_label": assoc.get("evidence_label", ""),
        "evidence_level": assoc.get("evidence_level"),
        "oncogenic": assoc.get("oncogenic", ""),
        "source_link": assoc.get("source_link", ""),
        "drugs": extract_drugs(hit),
        "diseases": extract_diseases(hit),
        "pmids": extract_pmids(hit),
        "evidence_items": evidence_items,
        "gene_symbols": gene_symbols,
        "variant_names": variant_names,
    }


# ---------------------------------------------------------------------------
# Per-variant query and analysis
# ---------------------------------------------------------------------------

def is_variant_specific(hit: dict, gene: str, variant: str) -> bool:
    """Check if a hit is specific to the exact variant (not just gene-level)."""
    assoc = hit.get("association", {})
    variant_name_raw = assoc.get("variant_name", "")
    feature_names_raw = hit.get("feature_names", "")
    description = assoc.get("description", "") or ""

    # Normalize to strings (API sometimes returns lists)
    if isinstance(variant_name_raw, list):
        variant_name = " ".join(str(v) for v in variant_name_raw)
    else:
        variant_name = str(variant_name_raw) if variant_name_raw else ""

    if isinstance(feature_names_raw, list):
        feature_names = " ".join(str(f) for f in feature_names_raw)
    else:
        feature_names = str(feature_names_raw) if feature_names_raw else ""

    if isinstance(description, list):
        description = " ".join(str(d) for d in description)

    # Check variant name field for exact or partial match
    var_short = variant  # e.g. "R882H"
    var_pos = variant[:-1] if variant else ""  # e.g. "R882"

    if var_short.lower() in variant_name.lower():
        return True
    if var_short.lower() in feature_names.lower():
        return True
    if var_pos and var_pos.lower() in variant_name.lower():
        return True

    # Check description
    if var_short in description:
        return True

    return False


def classify_relevance(hit: dict, gene: str, variant: str) -> str:
    """Classify a hit as variant-specific, gene-level, or co-occurrence."""
    if is_variant_specific(hit, gene, variant):
        return "variant-specific"

    # Check if it mentions the gene in a co-occurrence context
    features = hit.get("features", [])
    genes_in_hit = [f.get("geneSymbol", "") for f in features]
    if len([g for g in genes_in_hit if g]) > 1:
        return "co-occurrence"

    return "gene-level"


def query_variant(gene: str, variant: str, vaf: str, role: str) -> dict:
    """Full query pipeline for one patient variant."""
    query = f"{gene} {variant}"
    log.info("Querying VICC Meta-KB for %s ...", query)

    result: dict[str, Any] = {
        "gene": gene,
        "variant": variant,
        "query": query,
        "vaf": vaf,
        "role": role,
        "total_hits": 0,
        "hits_by_source": {},
        "hits_by_level": {},
        "hits_by_relevance": {},
        "unique_drugs": [],
        "unique_diseases": [],
        "unique_pmids": [],
        "parsed_hits": [],
        "concordance": {},
    }

    try:
        raw_hits = fetch_associations(query)
    except requests.exceptions.RequestException as exc:
        log.error("API error for %s: %s", query, exc)
        result["error"] = str(exc)
        return result

    result["total_hits"] = len(raw_hits)
    time.sleep(RATE_LIMIT_DELAY)

    # Parse and classify all hits
    source_counter: Counter = Counter()
    level_counter: Counter = Counter()
    relevance_counter: Counter = Counter()
    all_drugs: dict[str, dict] = {}
    all_diseases: dict[str, dict] = {}
    all_pmids: set[int] = set()
    parsed: list[dict] = []

    # Track response types per source for concordance
    source_responses: dict[str, list[str]] = defaultdict(list)

    for hit in raw_hits:
        record = parse_hit(hit)
        relevance = classify_relevance(hit, gene, variant)
        record["relevance"] = relevance

        source_counter[record["source"]] += 1
        level_label = record.get("evidence_label", "?")
        level_counter[level_label] += 1
        relevance_counter[relevance] += 1

        for drug in record["drugs"]:
            all_drugs[drug["name"]] = drug
        for disease in record["diseases"]:
            all_diseases[disease["name"]] = disease
        for pmid in record["pmids"]:
            all_pmids.add(pmid)

        # Track response type per source for concordance
        if record["response_type"]:
            source_responses[record["source"]].append(record["response_type"])

        parsed.append(record)

    result["hits_by_source"] = dict(source_counter.most_common())
    result["hits_by_level"] = dict(level_counter.most_common())
    result["hits_by_relevance"] = dict(relevance_counter.most_common())
    result["unique_drugs"] = sorted(all_drugs.values(), key=lambda d: d["name"])
    result["unique_diseases"] = sorted(all_diseases.values(), key=lambda d: d["name"])
    result["unique_pmids"] = sorted(all_pmids)
    result["parsed_hits"] = parsed

    # Concordance analysis across sources
    concordance = analyze_concordance(source_responses, parsed)
    result["concordance"] = concordance

    log.info(
        "  %s %s: %d hits from %d sources, %d drugs, %d diseases, %d PMIDs",
        gene, variant, len(raw_hits), len(source_counter),
        len(all_drugs), len(all_diseases), len(all_pmids),
    )

    return result


def analyze_concordance(source_responses: dict[str, list[str]], parsed: list[dict]) -> dict:
    """Analyze concordance/discordance across knowledgebases."""
    concordance: dict[str, Any] = {
        "sources_represented": sorted(source_responses.keys()),
        "n_sources": len(source_responses),
        "response_types_by_source": {},
        "is_concordant": True,
        "discordant_notes": [],
    }

    # Summarize response types per source
    all_responses: set[str] = set()
    for source, responses in source_responses.items():
        unique_responses = sorted(set(r.lower().strip() for r in responses if r))
        concordance["response_types_by_source"][source] = unique_responses
        all_responses.update(unique_responses)

    # Check for discordance: conflicting response types across sources
    positive_terms = {"responsive", "sensitive", "sensitivity", "benefit"}
    negative_terms = {"resistant", "resistance", "poor outcome", "no benefit"}
    has_positive = bool(all_responses & positive_terms)
    has_negative = bool(all_responses & negative_terms)

    if has_positive and has_negative:
        concordance["is_concordant"] = False
        concordance["discordant_notes"].append(
            f"Mixed response signals: positive ({all_responses & positive_terms}) "
            f"vs negative ({all_responses & negative_terms})"
        )

    # Check evidence level agreement
    source_levels: dict[str, set[str]] = defaultdict(set)
    for rec in parsed:
        if rec.get("evidence_label"):
            source_levels[rec["source"]].add(rec["evidence_label"])
    concordance["evidence_levels_by_source"] = {
        s: sorted(lvls) for s, lvls in source_levels.items()
    }

    return concordance


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], timestamp: str) -> str:
    """Generate a comprehensive markdown report."""
    lines = [
        "# VICC Meta-Knowledgebase: Federated Clinical Evidence",
        "",
        f"**Generated:** {timestamp}",
        f"**Source:** [VICC Meta-Knowledgebase](https://search.cancervariants.org)",
        f"**API:** `{ASSOCIATIONS_URL}`",
        "",
        "## Overview",
        "",
        "The VICC (Variant Interpretation for Cancer Consortium) Meta-Knowledgebase",
        "federates variant interpretations from multiple clinical knowledge sources.",
        "This enables cross-database concordance analysis and surfaces evidence from",
        "databases not queried individually (JAX-CKB, PMKB, MolecularMatch, CGI).",
        "",
        "### Federated Sources",
        "",
        "| Source | Full Name | Type |",
        "|--------|-----------|------|",
        "| CIViC | Clinical Interpretation of Variants in Cancer | Community-curated, open access |",
        "| OncoKB | Memorial Sloan Kettering Precision Oncology KB | Expert-curated, tiered |",
        "| CGI | Cancer Genome Interpreter | Computational + curated |",
        "| JAX-CKB | Jackson Laboratory Clinical Knowledgebase | Literature-curated |",
        "| PMKB | Precision Medicine Knowledgebase (Weill Cornell) | Pathologist-curated |",
        "| MolecularMatch | MolecularMatch | Clinical trial matching |",
        "",
    ]

    # Aggregate stats
    total_hits = sum(r["total_hits"] for r in results)
    all_sources: set[str] = set()
    total_drugs: dict[str, dict] = {}
    total_pmids: set[int] = set()
    for r in results:
        all_sources.update(r.get("hits_by_source", {}).keys())
        for d in r["unique_drugs"]:
            total_drugs[d["name"]] = d
        total_pmids.update(r["unique_pmids"])

    lines.extend([
        "### Summary Statistics",
        "",
        f"- **Variants queried:** {len(results)}",
        f"- **Total association hits:** {total_hits}",
        f"- **Knowledge sources with data:** {len(all_sources)} ({', '.join(sorted(all_sources))})",
        f"- **Unique drugs/therapies:** {len(total_drugs)}",
        f"- **Unique PubMed citations:** {len(total_pmids)}",
        "",
    ])

    # Cross-variant source matrix
    lines.extend([
        "### Hits by Source and Variant",
        "",
        "| Variant | Total |",
    ])

    # Build header with all sources
    sorted_sources = sorted(all_sources)
    header = "| Variant | " + " | ".join(sorted_sources) + " | Total |"
    separator = "|---------|" + "|".join(["------" for _ in sorted_sources]) + "|-------|"
    lines[-2] = header
    lines[-1] = separator

    for r in results:
        row = f"| {r['gene']} {r['variant']} |"
        for src in sorted_sources:
            count = r.get("hits_by_source", {}).get(src, 0)
            row += f" {count} |"
        row += f" **{r['total_hits']}** |"
        lines.append(row)

    total_row = "| **Total** |"
    for src in sorted_sources:
        col_total = sum(r.get("hits_by_source", {}).get(src, 0) for r in results)
        total_row += f" **{col_total}** |"
    total_row += f" **{total_hits}** |"
    lines.append(total_row)
    lines.append("")

    # Per-variant sections
    for r in results:
        gene = r["gene"]
        variant = r["variant"]
        lines.extend([
            f"## {gene} {variant}",
            "",
            f"- **VAF:** {r['vaf']}",
            f"- **Role:** {r['role']}",
            f"- **Total hits:** {r['total_hits']}",
            f"- **Sources:** {', '.join(f'{k} ({v})' for k, v in r.get('hits_by_source', {}).items())}",
            f"- **Evidence levels:** {', '.join(f'{k} ({v})' for k, v in r.get('hits_by_level', {}).items())}",
            f"- **Relevance:** {', '.join(f'{k} ({v})' for k, v in r.get('hits_by_relevance', {}).items())}",
            "",
        ])

        if r.get("error"):
            lines.extend([f"**Error:** {r['error']}", ""])
            continue

        if r["total_hits"] == 0:
            lines.extend(["No associations found in VICC Meta-Knowledgebase.", ""])
            continue

        # Drugs
        if r["unique_drugs"]:
            lines.append("### Drugs/Therapies")
            lines.append("")
            lines.append("| Drug | ID | Approved | Class |")
            lines.append("|------|----|----------|-------|")
            for drug in r["unique_drugs"]:
                approved = ", ".join(drug.get("approved_countries", []))
                cls = drug.get("class", "")
                lines.append(f"| {drug['name']} | {drug.get('id', '')} | {approved or '-'} | {cls or '-'} |")
            lines.append("")

        # Diseases
        if r["unique_diseases"]:
            lines.append("### Diseases")
            lines.append("")
            disease_names = sorted(set(d["name"] for d in r["unique_diseases"]))
            for dname in disease_names:
                lines.append(f"- {dname}")
            lines.append("")

        # Concordance
        conc = r.get("concordance", {})
        if conc.get("n_sources", 0) > 0:
            lines.append("### Cross-Database Concordance")
            lines.append("")
            status = "CONCORDANT" if conc.get("is_concordant", True) else "DISCORDANT"
            lines.append(f"- **Status:** {status}")
            lines.append(f"- **Sources represented:** {', '.join(conc.get('sources_represented', []))}")
            lines.append("")

            if conc.get("response_types_by_source"):
                lines.append("| Source | Response Types |")
                lines.append("|--------|---------------|")
                for src, resps in sorted(conc["response_types_by_source"].items()):
                    lines.append(f"| {src} | {', '.join(resps) if resps else '-'} |")
                lines.append("")

            if conc.get("discordant_notes"):
                for note in conc["discordant_notes"]:
                    lines.append(f"- **Note:** {note}")
                lines.append("")

        # Key evidence from non-CIViC sources (JAX-CKB, PMKB, OncoKB, CGI)
        non_civic = [h for h in r["parsed_hits"] if h["source"] not in ("CIVIC",)]
        if non_civic:
            lines.append("### Key Evidence from Non-CIViC Sources")
            lines.append("")
            lines.append("These entries provide data not available from our separate CIViC query.")
            lines.append("")

            for src_name in ["oncokb", "jax", "pmkb", "cgi", "molecularmatch"]:
                src_hits = [h for h in non_civic if h["source"] == src_name]
                if not src_hits:
                    continue

                source_label = {
                    "oncokb": "OncoKB",
                    "jax": "JAX-CKB",
                    "pmkb": "PMKB",
                    "cgi": "CGI",
                    "molecularmatch": "MolecularMatch",
                }.get(src_name, src_name)

                lines.append(f"#### {source_label} ({len(src_hits)} entries)")
                lines.append("")

                for i, hit in enumerate(src_hits[:10], 1):
                    desc = hit["description"].strip()
                    if len(desc) > 400:
                        desc = desc[:397] + "..."
                    level = hit.get("evidence_label", "?")
                    drugs_str = ", ".join(d["name"] for d in hit["drugs"]) if hit["drugs"] else "none"
                    diseases_str = ", ".join(d["name"] for d in hit["diseases"]) if hit["diseases"] else "N/A"
                    pmid_str = ", ".join(str(p) for p in hit["pmids"]) if hit["pmids"] else "-"
                    link = hit.get("source_link", "")
                    link_str = f" ([link]({link}))" if link else ""

                    lines.append(f"{i}. **[Level {level}]** {desc}")
                    lines.append(f"   - Drugs: {drugs_str} | Disease: {diseases_str} | PMID: {pmid_str}{link_str}")
                    lines.append("")

                if len(src_hits) > 10:
                    lines.append(f"   *... and {len(src_hits) - 10} more entries*")
                    lines.append("")

        # PubMed citations
        if r["unique_pmids"]:
            lines.append("### PubMed Citations")
            lines.append("")
            for pmid in r["unique_pmids"][:20]:
                lines.append(f"- [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
            if len(r["unique_pmids"]) > 20:
                lines.append(f"- *... and {len(r['unique_pmids']) - 20} more*")
            lines.append("")

    # New data analysis: what VICC adds beyond CIViC alone
    lines.extend([
        "## Added Value: Data Beyond CIViC",
        "",
        "The primary value of the VICC Meta-KB query is surfacing evidence from",
        "JAX-CKB, PMKB, OncoKB, and CGI that was not captured by our separate",
        "CIViC GraphQL query.",
        "",
    ])

    for r in results:
        non_civic_count = sum(
            v for k, v in r.get("hits_by_source", {}).items() if k != "CIVIC"
        )
        civic_count = r.get("hits_by_source", {}).get("CIVIC", 0)
        if non_civic_count > 0:
            non_civic_drugs = set()
            for hit in r["parsed_hits"]:
                if hit["source"] != "CIVIC":
                    for d in hit["drugs"]:
                        non_civic_drugs.add(d["name"])
            lines.append(
                f"- **{r['gene']} {r['variant']}:** {non_civic_count} non-CIViC entries "
                f"(vs {civic_count} CIViC). "
                f"Additional drugs: {', '.join(sorted(non_civic_drugs)) if non_civic_drugs else 'none'}."
            )
    lines.append("")

    # Actionable drugs summary
    lines.extend([
        "## Actionable Drug-Variant Associations",
        "",
        "| Variant | Drug | Source | Level | Response | Disease |",
        "|---------|------|--------|-------|----------|---------|",
    ])

    for r in results:
        for hit in r["parsed_hits"]:
            if hit["drugs"] and hit.get("evidence_label") in ("A", "B"):
                for drug in hit["drugs"]:
                    disease_str = hit["diseases"][0]["name"] if hit["diseases"] else "N/A"
                    lines.append(
                        f"| {r['gene']} {r['variant']} | {drug['name']} | {hit['source']} "
                        f"| {hit['evidence_label']} | {hit.get('response_type', '-')} "
                        f"| {disease_str} |"
                    )
    lines.append("")

    # Methodology
    lines.extend([
        "## Methodology",
        "",
        "1. Queried VICC Meta-Knowledgebase REST API (`/api/v1/associations?q=GENE+VARIANT`)"
        " for each of 5 patient variants",
        "2. Paginated through all results (100 per page) with rate limiting",
        "3. Parsed each hit: extracted source KB, evidence level, drugs, diseases,"
        " response types, PubMed IDs",
        "4. Classified hits as variant-specific, gene-level, or co-occurrence based on"
        " variant name matching",
        "5. Performed cross-database concordance analysis on response types per source",
        "6. Identified non-CIViC evidence (JAX-CKB, PMKB, OncoKB, CGI) as added value"
        " beyond our separate CIViC query",
        "",
        "### Limitations",
        "",
        "- The VICC Meta-KB aggregates data from a 2018-2020 snapshot; some sources may"
        " have newer data not reflected here",
        "- MolecularMatch data may be limited due to commercial restrictions",
        "- The `q=` parameter performs text search, which may include gene-level hits"
        " (e.g., DNMT3A MUTATION) alongside variant-specific hits (DNMT3A R882H)",
        "- CGI associations may include computational predictions not validated clinically",
        "",
        "---",
        f"*Generated by vicc_meta_kb.py on {timestamp}*",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("VICC Meta-Knowledgebase Federated Query")
    log.info("=" * 60)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    all_results: list[dict] = []

    for pv in PATIENT_VARIANTS:
        result = query_variant(pv["gene"], pv["variant"], pv["vaf"], pv["role"])
        all_results.append(result)
        time.sleep(RATE_LIMIT_DELAY)

    # Save JSON
    output = {
        "timestamp": timestamp,
        "source": "VICC Meta-Knowledgebase",
        "url": ASSOCIATIONS_URL,
        "federated_sources": sorted(KNOWN_SOURCES),
        "variants_queried": len(all_results),
        "total_hits": sum(r["total_hits"] for r in all_results),
        "results": all_results,
    }

    json_path = RESULTS_DIR / "vicc_meta_kb.json"
    json_path.write_text(json.dumps(output, indent=2, default=str))
    log.info("Saved JSON results to %s", json_path)

    # Generate markdown report
    report = generate_report(all_results, timestamp)
    md_path = RESULTS_DIR / "vicc_meta_kb_report.md"
    md_path.write_text(report)
    log.info("Saved markdown report to %s", md_path)

    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    for r in all_results:
        n = r["total_hits"]
        sources = r.get("hits_by_source", {})
        drugs = len(r["unique_drugs"])
        pmids = len(r["unique_pmids"])
        log.info(
            "  %s %s: %d hits (%s), %d drugs, %d PMIDs",
            r["gene"], r["variant"], n,
            ", ".join(f"{k}:{v}" for k, v in sources.items()),
            drugs, pmids,
        )

    total = sum(r["total_hits"] for r in all_results)
    all_sources = set()
    for r in all_results:
        all_sources.update(r.get("hits_by_source", {}).keys())
    log.info("  TOTAL: %d hits across %d variants from %d sources", total, len(all_results), len(all_sources))


if __name__ == "__main__":
    main()
