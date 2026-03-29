#!/usr/bin/env python3
"""
civic_annotation.py -- Query CIViC (Clinical Interpretation of Variants in Cancer)
for evidence items on patient variants via GraphQL API.

CIViC is an open-access, community-driven knowledgebase of clinical interpretations
of variants in cancer, maintained at Washington University in St. Louis.

Patient variants:
    1. DNMT3A R882H  (VAF 39%, pathogenic hotspot)
    2. IDH2  R140Q   (VAF 2%, pathogenic subclone)
    3. SETBP1 G870S  (VAF 34%, likely pathogenic)
    4. PTPN11 E76Q   (VAF 29%, pathogenic)
    5. EZH2  V662A   (VAF 59%, founder clone)

Data source:
    CIViC GraphQL API (https://civicdb.org/api/graphql)
    No authentication required.

Outputs:
    - mutation_profile/results/ai_research/civic_annotations.json
    - mutation_profile/results/ai_research/civic_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/civic_annotation.py

Runtime: ~15 seconds (GraphQL queries with rate limiting)
Dependencies: requests
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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

CIVIC_GRAPHQL_URL = "https://civicdb.org/api/graphql"
REQUEST_TIMEOUT = 30
RATE_LIMIT_DELAY = 0.5  # seconds between requests

# Patient variants to query
PATIENT_VARIANTS = [
    {"gene": "DNMT3A", "variant": "R882H", "vaf": "39%", "role": "pathogenic hotspot"},
    {"gene": "IDH2", "variant": "R140Q", "vaf": "2%", "role": "pathogenic subclone"},
    {"gene": "SETBP1", "variant": "G870S", "vaf": "34%", "role": "likely pathogenic"},
    {"gene": "PTPN11", "variant": "E76Q", "vaf": "29%", "role": "pathogenic"},
    {"gene": "EZH2", "variant": "V662A", "vaf": "59%", "role": "founder clone Pathogenic"},
]

# Step 1: Find variant IDs by gene name and variant name
BROWSE_VARIANTS_QUERY = """
query BrowseVariants($variantName: String!, $featureName: String!) {
  browseVariants(variantName: $variantName, featureName: $featureName, first: 10) {
    totalCount
    nodes {
      id
      name
      featureName
    }
  }
}
"""

# Step 2: Get full variant detail with molecular profiles and evidence
VARIANT_DETAIL_QUERY = """
query VariantDetail($id: Int!) {
  variant(id: $id) {
    id
    name
    feature {
      id
      name
    }
    variantAliases
    molecularProfiles {
      nodes {
        id
        name
        evidenceItems(first: 100) {
          totalCount
          nodes {
            id
            evidenceType
            evidenceLevel
            evidenceDirection
            significance
            status
            description
            disease {
              name
              doid
            }
            therapies {
              id
              name
              ncitId
            }
            source {
              citation
              citationId
              sourceUrl
              sourceType
              title
            }
          }
        }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _post_graphql(query: str, variables: dict) -> dict:
    """Execute a GraphQL query against CIViC API."""
    payload = {"query": query, "variables": variables}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    resp = requests.post(CIVIC_GRAPHQL_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        log.warning("GraphQL errors: %s", data["errors"])
    return data.get("data", {})


def find_variant_id(gene: str, variant: str) -> list[dict]:
    """Find CIViC variant IDs matching gene + variant name."""
    data = _post_graphql(BROWSE_VARIANTS_QUERY, {"variantName": variant, "featureName": gene})
    browse = data.get("browseVariants", {})
    total = browse.get("totalCount", 0)
    nodes = browse.get("nodes", [])
    # Filter to exact match on feature name and variant name
    exact = [n for n in nodes if n["featureName"] == gene and n["name"] == variant]
    log.info("  browseVariants: %d total, %d exact matches for %s %s", total, len(exact), gene, variant)
    return exact


def get_variant_detail(variant_id: int) -> dict:
    """Retrieve full variant detail including molecular profiles and evidence."""
    data = _post_graphql(VARIANT_DETAIL_QUERY, {"id": variant_id})
    return data.get("variant", {})


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_evidence_item(ev: dict) -> dict | None:
    """Parse a single evidence item node into a flat dict. Returns None for rejected items."""
    status = ev.get("status", "")
    if status == "REJECTED":
        return None

    disease = ev.get("disease")
    disease_name = disease["name"] if disease else "N/A"
    disease_doid = disease.get("doid") if disease else None

    therapies = [t["name"] for t in (ev.get("therapies") or []) if t.get("name")]

    source = ev.get("source") or {}
    citation_id = source.get("citationId")
    source_type = source.get("sourceType", "")
    # Build PubMed ID from citationId when sourceType is PUBMED
    pmid = int(citation_id) if source_type == "PUBMED" and citation_id else None

    return {
        "evidence_id": ev.get("id"),
        "evidence_type": ev.get("evidenceType"),
        "evidence_level": ev.get("evidenceLevel"),
        "evidence_direction": ev.get("evidenceDirection"),
        "significance": ev.get("significance"),
        "status": status,
        "description": ev.get("description", ""),
        "disease": disease_name,
        "disease_doid": disease_doid,
        "therapies": therapies,
        "pmid": pmid,
        "citation": source.get("citation", ""),
        "source_title": source.get("title", ""),
        "source_url": source.get("sourceUrl", ""),
        "source_type": source_type,
    }


def query_variant(gene: str, variant: str, vaf: str, role: str) -> dict:
    """Full query pipeline for one patient variant."""
    result = {
        "gene": gene,
        "variant": variant,
        "query": f"{gene} {variant}",
        "vaf": vaf,
        "role": role,
        "civic_variant_ids": [],
        "molecular_profiles": [],
        "evidence_items": [],
        "summary": {
            "total_evidence": 0,
            "by_type": {},
            "by_level": {},
            "diseases": [],
            "drugs": [],
            "pmids": [],
        },
    }

    log.info("Querying CIViC for %s %s ...", gene, variant)

    # Step 1: Find variant ID(s)
    matches = find_variant_id(gene, variant)
    if not matches:
        log.info("  No CIViC variant found for %s %s", gene, variant)
        return result

    result["civic_variant_ids"] = [m["id"] for m in matches]
    time.sleep(RATE_LIMIT_DELAY)

    # Step 2: Get detail for each variant ID
    for match in matches:
        vid = match["id"]
        log.info("  Fetching variant detail for ID %d ...", vid)
        detail = get_variant_detail(vid)
        if not detail:
            continue

        for mp in detail.get("molecularProfiles", {}).get("nodes", []):
            mp_name = mp.get("name", "")
            evidence_nodes = mp.get("evidenceItems", {}).get("nodes", [])
            total_in_mp = mp.get("evidenceItems", {}).get("totalCount", 0)

            mp_record = {
                "id": mp.get("id"),
                "name": mp_name,
                "evidence_count": total_in_mp,
            }
            result["molecular_profiles"].append(mp_record)

            log.info("    MolecularProfile '%s': %d evidence items", mp_name, total_in_mp)

            for ev_node in evidence_nodes:
                item = parse_evidence_item(ev_node)
                if item is None:
                    continue
                item["molecular_profile"] = mp_name
                result["evidence_items"].append(item)

                # Aggregate summary
                etype = item["evidence_type"] or "UNKNOWN"
                elevel = item["evidence_level"] or "UNKNOWN"
                result["summary"]["by_type"][etype] = result["summary"]["by_type"].get(etype, 0) + 1
                result["summary"]["by_level"][elevel] = result["summary"]["by_level"].get(elevel, 0) + 1

                if item["disease"] != "N/A" and item["disease"] not in result["summary"]["diseases"]:
                    result["summary"]["diseases"].append(item["disease"])
                for drug in item["therapies"]:
                    if drug not in result["summary"]["drugs"]:
                        result["summary"]["drugs"].append(drug)
                if item["pmid"] and item["pmid"] not in result["summary"]["pmids"]:
                    result["summary"]["pmids"].append(item["pmid"])

        time.sleep(RATE_LIMIT_DELAY)

    result["summary"]["total_evidence"] = len(result["evidence_items"])
    return result


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_report(results: list[dict], timestamp: str) -> str:
    """Generate a markdown summary report."""
    lines = [
        "# CIViC Clinical Evidence Annotations",
        "",
        f"**Generated:** {timestamp}",
        f"**Source:** [CIViC - Clinical Interpretation of Variants in Cancer](https://civicdb.org)",
        f"**API:** GraphQL endpoint (`{CIVIC_GRAPHQL_URL}`)",
        "",
        "## Overview",
        "",
        "CIViC is an open-access, community-driven knowledgebase for clinical interpretation",
        "of variants in cancer. Evidence items are curated from peer-reviewed literature and",
        "classified by type (predictive, diagnostic, prognostic, predisposing, oncogenic,",
        "functional) and level (A-E, from validated to preclinical).",
        "",
        "### Evidence Level Key",
        "",
        "| Level | Name | Description |",
        "|-------|------|-------------|",
        "| A | Validated | FDA-approved or professional guidelines |",
        "| B | Clinical | Clinical trial or large clinical study |",
        "| C | Case Study | Case reports or small series |",
        "| D | Preclinical | In vivo or in vitro models |",
        "| E | Inferential | Indirect or computational evidence |",
        "",
    ]

    total_evidence = sum(r["summary"]["total_evidence"] for r in results)
    total_drugs = len({d for r in results for d in r["summary"]["drugs"]})
    total_pmids = len({p for r in results for p in r["summary"]["pmids"]})
    variants_with_evidence = sum(1 for r in results if r["summary"]["total_evidence"] > 0)

    lines.extend([
        "### Summary Statistics",
        "",
        f"- **Variants queried:** {len(results)}",
        f"- **Variants with evidence:** {variants_with_evidence}",
        f"- **Total evidence items:** {total_evidence}",
        f"- **Unique drugs/therapies:** {total_drugs}",
        f"- **Unique PubMed citations:** {total_pmids}",
        "",
    ])

    # Per-variant sections
    for r in results:
        gene = r["gene"]
        variant = r["variant"]
        vaf = r["vaf"]
        role = r["role"]
        n_evidence = r["summary"]["total_evidence"]
        variant_ids = r.get("civic_variant_ids", [])

        lines.extend([
            f"## {gene} {variant}",
            "",
            f"- **VAF:** {vaf}",
            f"- **Role:** {role}",
            f"- **CIViC variant ID(s):** {', '.join(str(v) for v in variant_ids) if variant_ids else 'not found'}",
            f"- **Evidence items:** {n_evidence}",
            "",
        ])

        if n_evidence == 0:
            lines.extend(["No evidence items found in CIViC for this specific variant.", ""])
            continue

        # Summary stats
        if r["summary"]["by_type"]:
            lines.append("### Evidence by Type")
            lines.append("")
            for etype, count in sorted(r["summary"]["by_type"].items()):
                lines.append(f"- **{etype}:** {count}")
            lines.append("")

        if r["summary"]["by_level"]:
            lines.append("### Evidence by Level")
            lines.append("")
            for elevel, count in sorted(r["summary"]["by_level"].items()):
                lines.append(f"- **{elevel}:** {count}")
            lines.append("")

        if r["summary"]["drugs"]:
            lines.append("### Associated Drugs/Therapies")
            lines.append("")
            for drug in sorted(r["summary"]["drugs"]):
                lines.append(f"- {drug}")
            lines.append("")

        if r["summary"]["diseases"]:
            lines.append("### Associated Diseases")
            lines.append("")
            for disease in sorted(r["summary"]["diseases"]):
                lines.append(f"- {disease}")
            lines.append("")

        # Evidence table
        lines.append("### Evidence Items")
        lines.append("")
        lines.append("| # | Type | Level | Direction | Significance | Disease | Drugs | PMID |")
        lines.append("|---|------|-------|-----------|-------------|---------|-------|------|")

        for i, ev in enumerate(r["evidence_items"], 1):
            drugs_str = ", ".join(ev["therapies"]) if ev["therapies"] else "-"
            pmid = ev.get("pmid")
            pmid_str = f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else "-"
            lines.append(
                f"| {i} | {ev['evidence_type'] or '-'} | {ev['evidence_level'] or '-'} "
                f"| {ev['evidence_direction'] or '-'} | {ev['significance'] or '-'} "
                f"| {ev['disease']} | {drugs_str} | {pmid_str} |"
            )
        lines.append("")

        # Key evidence descriptions (sorted by level priority)
        level_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        sorted_evidence = sorted(
            r["evidence_items"],
            key=lambda e: level_order.get(e.get("evidence_level", "E"), 5),
        )
        top_evidence = sorted_evidence[:5]

        if top_evidence:
            lines.append("### Key Evidence Descriptions")
            lines.append("")
            for ev in top_evidence:
                desc = ev["description"].strip() if ev["description"] else "No description."
                if len(desc) > 500:
                    desc = desc[:497] + "..."
                pmid = ev.get("pmid")
                pmid_ref = f" (PMID: {pmid})" if pmid else ""
                lines.append(
                    f"- **[{ev['evidence_type']}/{ev['evidence_level']}]** {desc}{pmid_ref}"
                )
            lines.append("")

    # Clinical relevance summary
    lines.extend([
        "## Clinical Relevance to Patient Profile",
        "",
        "### Actionable Findings (Level A/B Predictive Evidence)",
        "",
    ])

    actionable = []
    for r in results:
        for ev in r["evidence_items"]:
            if ev["evidence_type"] == "PREDICTIVE" and ev["evidence_level"] in ("A", "B"):
                drugs = ", ".join(ev["therapies"]) if ev["therapies"] else "therapy"
                actionable.append(
                    f"- **{r['gene']} {r['variant']}**: {ev['significance']} response to "
                    f"**{drugs}** in {ev['disease']} (Level {ev['evidence_level']})"
                )

    if actionable:
        lines.extend(actionable)
    else:
        lines.append("- No Level A/B predictive evidence found for patient variants.")
    lines.append("")

    # Prognostic findings
    lines.extend(["### Prognostic Findings", ""])
    prognostic = []
    for r in results:
        for ev in r["evidence_items"]:
            if ev["evidence_type"] == "PROGNOSTIC":
                pmid = ev.get("pmid")
                pmid_ref = f" (PMID: {pmid})" if pmid else ""
                prognostic.append(
                    f"- **{r['gene']} {r['variant']}**: {ev['significance']} in "
                    f"{ev['disease']} (Level {ev['evidence_level']}){pmid_ref}"
                )
    if prognostic:
        lines.extend(prognostic)
    else:
        lines.append("- No prognostic evidence found for patient variants.")
    lines.append("")

    # Diagnostic findings
    lines.extend(["### Diagnostic Findings", ""])
    diagnostic = []
    for r in results:
        for ev in r["evidence_items"]:
            if ev["evidence_type"] == "DIAGNOSTIC":
                pmid = ev.get("pmid")
                pmid_ref = f" (PMID: {pmid})" if pmid else ""
                diagnostic.append(
                    f"- **{r['gene']} {r['variant']}**: {ev['significance']} for "
                    f"{ev['disease']} (Level {ev['evidence_level']}){pmid_ref}"
                )
    if diagnostic:
        lines.extend(diagnostic)
    else:
        lines.append("- No diagnostic evidence found for patient variants.")
    lines.append("")

    lines.extend([
        "## Methodology",
        "",
        "1. Queried CIViC GraphQL API `browseVariants` for each of 5 patient variants by gene symbol and variant name",
        "2. Retrieved variant IDs and fetched full molecular profiles with `variant(id:)` detail query",
        "3. Extracted all non-rejected evidence items with associated diseases, therapies, and citations",
        "4. Classified evidence by type (predictive, diagnostic, prognostic, oncogenic, functional) and level (A-E)",
        "5. Aggregated unique drugs, diseases, and PubMed references",
        "",
        "---",
        f"*Generated by civic_annotation.py on {timestamp}*",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("CIViC Clinical Evidence Annotation")
    log.info("=" * 60)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    all_results = []

    for pv in PATIENT_VARIANTS:
        try:
            result = query_variant(pv["gene"], pv["variant"], pv["vaf"], pv["role"])
            all_results.append(result)
        except requests.exceptions.RequestException as exc:
            log.error("API error for %s %s: %s", pv["gene"], pv["variant"], exc)
            all_results.append({
                "gene": pv["gene"],
                "variant": pv["variant"],
                "query": f"{pv['gene']} {pv['variant']}",
                "vaf": pv["vaf"],
                "role": pv["role"],
                "civic_variant_ids": [],
                "molecular_profiles": [],
                "evidence_items": [],
                "summary": {
                    "total_evidence": 0,
                    "by_type": {},
                    "by_level": {},
                    "diseases": [],
                    "drugs": [],
                    "pmids": [],
                },
                "error": str(exc),
            })

    # Save JSON results
    output = {
        "timestamp": timestamp,
        "source": "CIViC GraphQL API",
        "url": CIVIC_GRAPHQL_URL,
        "civic_version": "2024+",
        "variants_queried": len(all_results),
        "total_evidence_items": sum(r["summary"]["total_evidence"] for r in all_results),
        "results": all_results,
    }

    json_path = RESULTS_DIR / "civic_annotations.json"
    json_path.write_text(json.dumps(output, indent=2, default=str))
    log.info("Saved JSON results to %s", json_path)

    # Generate markdown report
    report = generate_report(all_results, timestamp)
    md_path = RESULTS_DIR / "civic_report.md"
    md_path.write_text(report)
    log.info("Saved markdown report to %s", md_path)

    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("RESULTS SUMMARY")
    log.info("=" * 60)
    for r in all_results:
        n = r["summary"]["total_evidence"]
        drugs = len(r["summary"]["drugs"])
        pmids = len(r["summary"]["pmids"])
        vids = r.get("civic_variant_ids", [])
        log.info(
            "  %s %s: %d evidence items, %d drugs, %d citations (variant IDs: %s)",
            r["gene"], r["variant"], n, drugs, pmids, vids or "none",
        )
    total = sum(r["summary"]["total_evidence"] for r in all_results)
    log.info("  TOTAL: %d evidence items across %d variants", total, len(all_results))


if __name__ == "__main__":
    main()
