#!/usr/bin/env python3
"""
dgidb_interactions.py -- Drug-gene interaction analysis via DGIdb GraphQL API.

Queries the Drug-Gene Interaction Database (DGIdb) for all known drug interactions,
gene categories, interaction types, and supporting publications for the 5 patient genes.

Patient mutation profile (MDS-AML):
    1. DNMT3A R882H -- epigenetic regulator (DNA methyltransferase)
    2. IDH2 R140Q   -- metabolic enzyme (isocitrate dehydrogenase)
    3. SETBP1 G870S -- oncogene (SET binding protein, stabilises SET/PP2A inhibition)
    4. PTPN11 E76Q  -- signalling (SHP2 phosphatase, RAS/MAPK pathway)
    5. EZH2 V662A   -- epigenetic regulator (histone methyltransferase)

Inputs:
    - DGIdb GraphQL API (https://dgidb.org/api/graphql, no auth required)

Outputs:
    - mutation_profile/results/ai_research/dgidb_interactions.json
    - mutation_profile/results/ai_research/dgidb_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/dgidb_interactions.py

Runtime: ~5-10 seconds (single API call)
Dependencies: requests
"""

from __future__ import annotations

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
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # mutation_profile/
RESULTS_DIR = PROJECT_ROOT / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = RESULTS_DIR / "dgidb_interactions.json"
MD_OUTPUT = RESULTS_DIR / "dgidb_report.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patient genes
# ---------------------------------------------------------------------------
PATIENT_GENES = ["DNMT3A", "IDH2", "SETBP1", "PTPN11", "EZH2"]

DGIDB_URL = "https://dgidb.org/api/graphql"

GRAPHQL_QUERY = """
{
  genes(names: ["DNMT3A", "IDH2", "SETBP1", "PTPN11", "EZH2"]) {
    nodes {
      name
      longName
      geneCategoriesWithSources {
        name
      }
      interactions {
        drug {
          name
          conceptId
        }
        interactionTypes {
          type
          directionality
        }
        publications {
          pmid
        }
        interactionScore
        evidenceScore
        sources {
          sourceDbName
        }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# API query
# ---------------------------------------------------------------------------
def query_dgidb() -> dict[str, Any]:
    """Send GraphQL query to DGIdb and return parsed response."""
    log.info("Querying DGIdb GraphQL API for %d genes...", len(PATIENT_GENES))
    resp = requests.post(
        DGIDB_URL,
        json={"query": GRAPHQL_QUERY},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        log.error("GraphQL errors: %s", json.dumps(data["errors"], indent=2))
        sys.exit(1)
    return data


# ---------------------------------------------------------------------------
# Parse response into structured results
# ---------------------------------------------------------------------------
def parse_results(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse DGIdb GraphQL response into a clean results dict."""
    genes_data = raw.get("data", {}).get("genes", {}).get("nodes", [])
    results: dict[str, Any] = {
        "query_date": datetime.now(timezone.utc).isoformat(),
        "source": "DGIdb (Drug-Gene Interaction Database)",
        "api_url": DGIDB_URL,
        "patient_genes": PATIENT_GENES,
        "genes": {},
        "summary": {},
    }

    total_interactions = 0
    total_drugs = set()

    for gene_node in genes_data:
        gene_name = gene_node.get("name", "UNKNOWN")
        long_name = gene_node.get("longName", "")
        categories = [c["name"] for c in gene_node.get("geneCategoriesWithSources", [])]

        interactions_raw = gene_node.get("interactions", [])
        interactions = []
        gene_drugs = set()

        for node in interactions_raw:
            drug_info = node.get("drug", {})
            drug_name = drug_info.get("name", "UNKNOWN")
            drug_concept_id = drug_info.get("conceptId", "")

            int_types = node.get("interactionTypes", [])
            type_list = []
            for it in int_types:
                type_list.append({
                    "type": it.get("type", ""),
                    "directionality": it.get("directionality", ""),
                })

            pubs = node.get("publications", [])
            pmids = [p.get("pmid") for p in pubs if p.get("pmid")]

            score = node.get("interactionScore")
            sources = [s.get("sourceDbName", "") for s in node.get("sources", [])]

            evidence_score = node.get("evidenceScore")

            interactions.append({
                "drug": drug_name,
                "drug_concept_id": drug_concept_id,
                "interaction_types": type_list,
                "interaction_score": score,
                "evidence_score": evidence_score,
                "sources": sources,
                "pmids": pmids,
                "pmid_count": len(pmids),
            })
            gene_drugs.add(drug_name)
            total_drugs.add(drug_name)

        total_interactions += len(interactions)

        # Sort by score descending (None last)
        interactions.sort(
            key=lambda x: (x["interaction_score"] is not None, x["interaction_score"] or 0),
            reverse=True,
        )

        results["genes"][gene_name] = {
            "long_name": long_name,
            "gene_categories": categories,
            "interaction_count": len(interactions),
            "unique_drugs": len(gene_drugs),
            "interactions": interactions,
        }

    results["summary"] = {
        "genes_queried": len(PATIENT_GENES),
        "genes_found": len(genes_data),
        "total_interactions": total_interactions,
        "total_unique_drugs": len(total_drugs),
    }

    return results


# ---------------------------------------------------------------------------
# Generate markdown report
# ---------------------------------------------------------------------------
def generate_report(results: dict[str, Any]) -> str:
    """Generate a markdown report from DGIdb results."""
    lines: list[str] = []
    summary = results["summary"]

    lines.append("# DGIdb Drug-Gene Interaction Report")
    lines.append("")
    lines.append(f"**Query date:** {results['query_date']}")
    lines.append(f"**Source:** {results['source']}")
    lines.append(f"**API:** {results['api_url']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Genes queried | {summary['genes_queried']} |")
    lines.append(f"| Genes found | {summary['genes_found']} |")
    lines.append(f"| Total interactions | {summary['total_interactions']} |")
    lines.append(f"| Unique drugs | {summary['total_unique_drugs']} |")
    lines.append("")

    # Gene categories overview
    lines.append("## Gene Categories")
    lines.append("")
    lines.append("| Gene | Categories |")
    lines.append("|------|------------|")
    for gene_name in PATIENT_GENES:
        gene_data = results["genes"].get(gene_name)
        if gene_data:
            cats = ", ".join(gene_data["gene_categories"]) if gene_data["gene_categories"] else "None"
            lines.append(f"| {gene_name} | {cats} |")
        else:
            lines.append(f"| {gene_name} | *Not found in DGIdb* |")
    lines.append("")

    # Per-gene interaction tables
    for gene_name in PATIENT_GENES:
        gene_data = results["genes"].get(gene_name)
        if not gene_data:
            lines.append(f"## {gene_name}")
            lines.append("")
            lines.append("*No data returned from DGIdb.*")
            lines.append("")
            continue

        lines.append(f"## {gene_name} ({gene_data['long_name']})")
        lines.append("")
        lines.append(f"**Interactions:** {gene_data['interaction_count']} | "
                      f"**Unique drugs:** {gene_data['unique_drugs']} | "
                      f"**Categories:** {', '.join(gene_data['gene_categories']) or 'None'}")
        lines.append("")

        if not gene_data["interactions"]:
            lines.append("*No drug interactions found.*")
            lines.append("")
            continue

        lines.append("| Drug | Type | Directionality | Score | Sources | PMIDs |")
        lines.append("|------|------|----------------|-------|---------|-------|")

        for ix in gene_data["interactions"]:
            drug = ix["drug"]
            if ix["interaction_types"]:
                int_type = ix["interaction_types"][0].get("type", "n/a")
                direction = ix["interaction_types"][0].get("directionality", "n/a")
            else:
                int_type = "n/a"
                direction = "n/a"
            score = f"{ix['interaction_score']:.2f}" if ix["interaction_score"] is not None else "n/a"
            sources = ", ".join(ix["sources"][:3])
            if len(ix["sources"]) > 3:
                sources += f" (+{len(ix['sources']) - 3})"
            pmid_count = ix["pmid_count"]
            pmid_str = str(pmid_count) if pmid_count else "0"
            lines.append(f"| {drug} | {int_type} | {direction} | {score} | {sources} | {pmid_str} |")

        lines.append("")

    # Clinical relevance section
    lines.append("## Clinical Relevance to Patient Profile")
    lines.append("")
    lines.append("### Directly actionable targets")
    lines.append("")
    lines.append("- **IDH2 R140Q**: Enasidenib (AG-221) FDA-approved 2017 for relapsed/refractory "
                  "AML with IDH2 mutation")
    lines.append("- **PTPN11 E76Q**: SHP2 inhibitors (TNO155, RMC-4550) in Phase I/II clinical "
                  "trials for myeloid malignancies")
    lines.append("- **EZH2**: EZH2 inhibitors (tazemetostat/EPZ-6438) FDA-approved for follicular "
                  "lymphoma; investigational in myeloid malignancies")
    lines.append("")
    lines.append("### Limited direct targeting")
    lines.append("")
    lines.append("- **DNMT3A R882H**: No direct inhibitors; hypomethylating agents (azacitidine, "
                  "decitabine) target the pathway")
    lines.append("- **SETBP1 G870S**: No direct inhibitors exist; PP2A activators under investigation")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Data from DGIdb (https://dgidb.org). DGIdb aggregates drug-gene interaction data "
                  "from >30 source databases including DrugBank, PharmGKB, ChEMBL, CIViC, OncoKB, "
                  "and others.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Query DGIdb and save results."""
    log.info("Starting DGIdb drug-gene interaction analysis")
    start = time.time()

    raw = query_dgidb()
    results = parse_results(raw)

    # Save JSON
    with open(JSON_OUTPUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved JSON results to %s", JSON_OUTPUT)

    # Save markdown
    report = generate_report(results)
    with open(MD_OUTPUT, "w") as f:
        f.write(report)
    log.info("Saved markdown report to %s", MD_OUTPUT)

    # Print summary
    elapsed = time.time() - start
    s = results["summary"]
    log.info("DGIdb query complete in %.1fs", elapsed)
    log.info("  Genes found: %d/%d", s["genes_found"], s["genes_queried"])
    log.info("  Total interactions: %d", s["total_interactions"])
    log.info("  Unique drugs: %d", s["total_unique_drugs"])

    for gene_name in PATIENT_GENES:
        gene_data = results["genes"].get(gene_name)
        if gene_data:
            log.info("  %s: %d interactions, %d drugs, categories: %s",
                     gene_name, gene_data["interaction_count"],
                     gene_data["unique_drugs"],
                     gene_data["gene_categories"] or ["None"])
        else:
            log.info("  %s: not found in DGIdb", gene_name)


if __name__ == "__main__":
    main()
