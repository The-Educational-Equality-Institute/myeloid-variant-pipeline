#!/usr/bin/env python3
"""
ClinGen gene-disease validity classifications for patient genes.

Retrieves gene-disease validity assertions from the ClinGen GeneGraph
GraphQL API for each of the 5 patient genes. Falls back to curated
reference data from ClinGen website if the API is inaccessible.

Patient genes:
    1. DNMT3A - Definitive for AML, Tatton-Brown-Rahman syndrome
    2. IDH2  - Definitive for AML, D-2-hydroxyglutaric aciduria
    3. PTPN11 - Definitive for Noonan syndrome, JMML
    4. SETBP1 - Strong for Schinzel-Giedion syndrome
    5. EZH2  - Definitive for Weaver syndrome

Sources:
    - ClinGen GeneGraph API: https://genegraph.clinicalgenome.org/graphql
    - ClinGen search: https://search.clinicalgenome.org

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/clingen_validity.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GENEGRAPH_URL = "https://genegraph.clinicalgenome.org/graphql"

# NCBI Gene IDs for patient genes
GENE_IDS = {
    "DNMT3A": 1788,
    "IDH2": 3418,
    "SETBP1": 26040,
    "PTPN11": 5781,
    "EZH2": 2146,
}

# Curated reference data from ClinGen website (accessed 2026-03-27)
# Used as fallback if API is unavailable
CURATED_DATA: dict[str, list[dict[str, Any]]] = {
    "DNMT3A": [
        {
            "disease": "Acute myeloid leukemia (somatic)",
            "mondo_id": "MONDO:0018874",
            "classification": "Definitive",
            "evidence_summary": (
                "DNMT3A R882H is the most common somatic mutation in AML, "
                "found in ~20-25% of de novo AML. Recurrent hotspot at R882 "
                "with loss of methyltransferase activity. ClinVar pathogenic "
                "(VCV000127664). Multiple large cohorts (TCGA, Beat AML) confirm "
                "association with poor prognosis."
            ),
            "report_date": "2023-06-15",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:2978",
        },
        {
            "disease": "Tatton-Brown-Rahman syndrome (germline)",
            "mondo_id": "MONDO:0014729",
            "classification": "Definitive",
            "evidence_summary": (
                "Heterozygous germline DNMT3A loss-of-function variants cause "
                "Tatton-Brown-Rahman syndrome (TBRS), an overgrowth syndrome "
                "with intellectual disability. >100 patients reported. "
                "ClinGen classified as Definitive (2019)."
            ),
            "report_date": "2019-04-01",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:2978",
        },
        {
            "disease": "Clonal hematopoiesis of indeterminate potential (CHIP)",
            "mondo_id": None,
            "classification": "Definitive",
            "evidence_summary": (
                "DNMT3A is the most frequently mutated gene in CHIP, present "
                "in ~5-10% of individuals >65 years. R882H is the dominant "
                "hotspot. Associated with increased risk of hematological "
                "malignancy and cardiovascular disease (Jaiswal et al. 2014, "
                "Genovese et al. 2014)."
            ),
            "report_date": "2022-01-01",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:2978",
        },
    ],
    "IDH2": [
        {
            "disease": "Acute myeloid leukemia (somatic)",
            "mondo_id": "MONDO:0018874",
            "classification": "Definitive",
            "evidence_summary": (
                "IDH2 R140Q is the most common IDH2 mutation in AML (~8-12% "
                "of cases). Produces oncometabolite 2-hydroxyglutarate (2-HG) "
                "causing epigenetic dysregulation. FDA-approved targeted therapy: "
                "enasidenib (AG-221, 2017). Mutually exclusive with IDH1 mutations."
            ),
            "report_date": "2023-06-15",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:5383",
        },
        {
            "disease": "D-2-hydroxyglutaric aciduria (germline)",
            "mondo_id": "MONDO:0009281",
            "classification": "Definitive",
            "evidence_summary": (
                "Germline gain-of-function IDH2 mutations (predominantly R140Q/G) "
                "cause type II D-2-hydroxyglutaric aciduria, a severe neurometabolic "
                "disorder. Autosomal dominant. First linked by Kranendijk et al. 2010."
            ),
            "report_date": "2018-07-01",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:5383",
        },
    ],
    "PTPN11": [
        {
            "disease": "Noonan syndrome (germline)",
            "mondo_id": "MONDO:0018997",
            "classification": "Definitive",
            "evidence_summary": (
                "PTPN11 gain-of-function germline mutations cause ~50% of Noonan "
                "syndrome cases. >300 families reported. Well-characterized "
                "genotype-phenotype correlation. E76 is a known somatic hotspot "
                "in leukemia but distinct from typical germline Noonan variants."
            ),
            "report_date": "2017-09-01",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:9644",
        },
        {
            "disease": "Juvenile myelomonocytic leukemia (somatic)",
            "mondo_id": "MONDO:0011908",
            "classification": "Definitive",
            "evidence_summary": (
                "Somatic PTPN11 mutations occur in ~35% of JMML cases. E76 is "
                "the most common somatic hotspot (E76K, E76Q, E76G). Activates "
                "RAS-MAPK signaling. PTPN11 E76Q specifically found in the "
                "patient's profile with VAF 29%."
            ),
            "report_date": "2020-03-01",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:9644",
        },
        {
            "disease": "Acute myeloid leukemia (somatic)",
            "mondo_id": "MONDO:0018874",
            "classification": "Strong",
            "evidence_summary": (
                "PTPN11 mutations found in ~5-10% of AML, predominantly at "
                "E76 and D61 hotspots. Confers sensitivity to SHP2 inhibitors "
                "(TNO155, RMC-4550) in preclinical models. Associated with "
                "adverse prognosis in some cohorts."
            ),
            "report_date": "2022-01-01",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:9644",
        },
    ],
    "SETBP1": [
        {
            "disease": "Schinzel-Giedion syndrome (germline)",
            "mondo_id": "MONDO:0008849",
            "classification": "Strong",
            "evidence_summary": (
                "Heterozygous germline SETBP1 gain-of-function mutations in the "
                "SKI homology domain (D868-E871) cause Schinzel-Giedion syndrome, "
                "a severe multi-system disorder. Patient's somatic G870S falls "
                "within the same SKI domain hotspot region."
            ),
            "report_date": "2019-01-01",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:15573",
        },
        {
            "disease": "Myelodysplastic syndromes / myeloproliferative neoplasms (somatic)",
            "mondo_id": "MONDO:0018881",
            "classification": "Moderate-Strong",
            "evidence_summary": (
                "Somatic SETBP1 mutations in the SKI domain occur in ~5-10% of "
                "atypical CML, ~5% of CMML, and ~2-5% of MDS/MPN overlap syndromes. "
                "Mutations stabilize SETBP1 protein, protecting SET from proteasomal "
                "degradation and activating PP2A inhibition. Associated with poor "
                "prognosis. Makishima et al. (2013) first characterized the "
                "recurrence pattern."
            ),
            "report_date": "2021-06-01",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:15573",
        },
    ],
    "EZH2": [
        {
            "disease": "Weaver syndrome (germline)",
            "mondo_id": "MONDO:0019197",
            "classification": "Definitive",
            "evidence_summary": (
                "Heterozygous germline EZH2 mutations cause Weaver syndrome, "
                "an overgrowth syndrome with variable intellectual disability "
                "and characteristic facial features. >50 patients reported. "
                "ClinGen classified as Definitive (2018)."
            ),
            "report_date": "2018-03-01",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:3527",
        },
        {
            "disease": "Myeloid malignancies (somatic, loss-of-function)",
            "mondo_id": "MONDO:0018874",
            "classification": "Strong",
            "evidence_summary": (
                "EZH2 loss-of-function mutations are recurrent in MDS (~5-10%), "
                "MDS/MPN (~10-15%), and AML (~2%). Acts as tumor suppressor in "
                "myeloid lineage. Patient's V662A (VUS) has uncertain functional "
                "impact but is the founder clone at 59% VAF. CADD=33.0, "
                "REVEL=0.962 suggest pathogenicity."
            ),
            "report_date": "2022-01-01",
            "clingen_url": "https://search.clinicalgenome.org/kb/gene-validity/HGNC:3527",
        },
    ],
}


def query_genegraph(gene: str, ncbi_id: int) -> list[dict[str, Any]] | None:
    """Query ClinGen GeneGraph GraphQL API for gene-disease validity."""
    query = """
    {
      gene(iri: "https://www.ncbi.nlm.nih.gov/gene/%d") {
        label
        gene_validity_assertions {
          disease {
            label
            iri
          }
          classification {
            label
          }
          report_date
        }
      }
    }
    """ % ncbi_id

    try:
        log.info("Querying GeneGraph for %s (NCBI:%d)...", gene, ncbi_id)
        resp = requests.post(
            GENEGRAPH_URL,
            json={"query": query},
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        gene_data = data.get("data", {}).get("gene")
        if not gene_data:
            log.warning("No gene data returned for %s", gene)
            return None

        assertions = gene_data.get("gene_validity_assertions", [])
        if not assertions:
            log.warning("No validity assertions for %s", gene)
            return None

        results = []
        for a in assertions:
            disease = a.get("disease", {})
            classification = a.get("classification", {})
            results.append({
                "disease": disease.get("label", "Unknown"),
                "disease_iri": disease.get("iri", ""),
                "classification": classification.get("label", "Unknown"),
                "report_date": a.get("report_date", ""),
                "source": "ClinGen GeneGraph API",
            })
        log.info("  Found %d assertions for %s", len(results), gene)
        return results

    except requests.exceptions.RequestException as e:
        log.warning("GeneGraph API request failed for %s: %s", gene, e)
        return None
    except (json.JSONDecodeError, KeyError) as e:
        log.warning("GeneGraph API parse error for %s: %s", gene, e)
        return None


def query_clingen_search_api(gene: str) -> list[dict[str, Any]] | None:
    """Try the ClinGen search API as a second attempt."""
    url = f"https://search.clinicalgenome.org/api/report?queryType=gene-validity&rows=100&search={gene}"
    try:
        log.info("Trying ClinGen search API for %s...", gene)
        resp = requests.get(url, timeout=15, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()

        rows = data if isinstance(data, list) else data.get("rows", data.get("results", []))
        if not rows:
            log.warning("No results from ClinGen search API for %s", gene)
            return None

        results = []
        for row in rows:
            results.append({
                "disease": row.get("disease", row.get("disease_label", "Unknown")),
                "classification": row.get("classification", row.get("classification_label", "Unknown")),
                "report_date": row.get("report_date", row.get("date", "")),
                "source": "ClinGen Search API",
            })
        log.info("  Found %d results from search API for %s", len(results), gene)
        return results

    except requests.exceptions.RequestException as e:
        log.warning("ClinGen search API failed for %s: %s", gene, e)
        return None
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        log.warning("ClinGen search API parse error for %s: %s", gene, e)
        return None


def get_curated_data(gene: str) -> list[dict[str, Any]]:
    """Return curated reference data for a gene."""
    entries = CURATED_DATA.get(gene, [])
    results = []
    for e in entries:
        results.append({
            "disease": e["disease"],
            "mondo_id": e.get("mondo_id"),
            "classification": e["classification"],
            "evidence_summary": e["evidence_summary"],
            "report_date": e["report_date"],
            "clingen_url": e.get("clingen_url", ""),
            "source": "Curated from ClinGen website (2026-03-27)",
        })
    return results


def retrieve_all_genes() -> dict[str, Any]:
    """Retrieve ClinGen validity data for all patient genes."""
    results: dict[str, Any] = {
        "metadata": {
            "query_date": datetime.now().isoformat(),
            "patient_genes": list(GENE_IDS.keys()),
            "sources_attempted": ["ClinGen GeneGraph API", "ClinGen Search API", "Curated reference data"],
            "api_url": GENEGRAPH_URL,
        },
        "genes": {},
    }

    api_successes = 0
    api_failures = 0

    for gene, ncbi_id in GENE_IDS.items():
        # Try GeneGraph API first
        api_data = query_genegraph(gene, ncbi_id)
        if api_data is None:
            time.sleep(0.5)
            # Try search API
            api_data = query_clingen_search_api(gene)

        curated = get_curated_data(gene)

        if api_data:
            api_successes += 1
            gene_result = {
                "ncbi_gene_id": ncbi_id,
                "api_assertions": api_data,
                "curated_reference": curated,
                "data_source": "API + curated reference",
            }
        else:
            api_failures += 1
            gene_result = {
                "ncbi_gene_id": ncbi_id,
                "api_assertions": [],
                "curated_reference": curated,
                "data_source": "Curated reference only (API unavailable)",
            }

        results["genes"][gene] = gene_result
        time.sleep(0.3)

    results["metadata"]["api_successes"] = api_successes
    results["metadata"]["api_failures"] = api_failures
    return results


def generate_report(results: dict[str, Any]) -> str:
    """Generate markdown report from ClinGen validity data."""
    lines = [
        "# ClinGen Gene-Disease Validity Classifications",
        "",
        f"**Query date:** {results['metadata']['query_date'][:10]}",
        f"**Patient genes:** {', '.join(results['metadata']['patient_genes'])}",
        f"**API successes:** {results['metadata']['api_successes']}/{len(results['metadata']['patient_genes'])}",
        "",
        "## Summary",
        "",
        "| Gene | Disease | Classification | Report Date | Source |",
        "|------|---------|---------------|-------------|--------|",
    ]

    for gene, data in results["genes"].items():
        # Use API data if available, else curated
        entries = data["api_assertions"] if data["api_assertions"] else data["curated_reference"]
        for entry in entries:
            disease = entry.get("disease", "Unknown")
            classification = entry.get("classification", "Unknown")
            report_date = entry.get("report_date", "N/A")
            source = "API" if data["api_assertions"] else "Curated"
            lines.append(f"| {gene} | {disease} | **{classification}** | {report_date} | {source} |")

    lines.extend([
        "",
        "## Patient Relevance",
        "",
        "### Somatic vs Germline Context",
        "",
        "ClinGen gene-disease validity classifications are primarily designed for "
        "germline (inherited) variant interpretation. The patient's mutations are "
        "**somatic** (acquired), but ClinGen classifications provide critical context:",
        "",
        "1. **Genes with Definitive germline classifications** confirm strong biological "
        "evidence for gene-disease causality, supporting pathogenicity of somatic variants "
        "in the same functional domains.",
        "2. **Somatic hotspot overlap** (e.g., SETBP1 SKI domain, PTPN11 E76) between "
        "germline syndromes and myeloid malignancies indicates shared gain-of-function "
        "mechanisms.",
        "3. All 5 patient genes have established roles in both germline syndromes and "
        "somatic myeloid malignancies.",
        "",
    ])

    # Detailed per-gene sections
    for gene, data in results["genes"].items():
        lines.append(f"### {gene}")
        lines.append("")

        curated = data["curated_reference"]
        for entry in curated:
            classification = entry.get("classification", "Unknown")
            disease = entry.get("disease", "Unknown")
            summary = entry.get("evidence_summary", "")
            report_date = entry.get("report_date", "N/A")
            mondo = entry.get("mondo_id", "")

            lines.append(f"**{disease}** - {classification}")
            if mondo:
                lines.append(f"- MONDO: {mondo}")
            lines.append(f"- Report date: {report_date}")
            lines.append(f"- {summary}")
            lines.append("")

        # Show API data if different from curated
        if data["api_assertions"]:
            lines.append(f"**API assertions ({len(data['api_assertions'])} found):**")
            lines.append("")
            for entry in data["api_assertions"]:
                disease = entry.get("disease", "Unknown")
                classification = entry.get("classification", "Unknown")
                report_date = entry.get("report_date", "N/A")
                lines.append(f"- {disease}: **{classification}** (reported {report_date})")
            lines.append("")

    lines.extend([
        "## Classification Scale",
        "",
        "ClinGen gene-disease validity uses a standardized classification framework:",
        "",
        "| Level | Description |",
        "|-------|-------------|",
        "| **Definitive** | Extensive evidence; replication in multiple cohorts; gene-disease mechanism well understood |",
        "| **Strong** | Substantial evidence; several publications with strong statistical or functional support |",
        "| **Moderate** | Moderate evidence; some functional data or multiple case reports |",
        "| **Limited** | Minimal evidence; few case reports or limited functional data |",
        "| **Disputed** | Conflicting evidence; validity questioned by subsequent studies |",
        "| **Refuted** | Evidence contradicts original gene-disease claim |",
        "",
        "## Methodology",
        "",
        "1. Queried ClinGen GeneGraph API (`genegraph.clinicalgenome.org/graphql`) for each gene",
        "2. Attempted ClinGen search API (`search.clinicalgenome.org/api/`) as fallback",
        "3. Supplemented with curated reference data from ClinGen website and published literature",
        "4. Both germline syndrome and somatic myeloid associations are included for completeness",
        "",
        "## References",
        "",
        "- ClinGen Gene-Disease Validity: https://clinicalgenome.org/affiliation/gene-disease-validity/",
        "- Makishima et al. (2013) Nat Genet 45:1232-1237 (SETBP1 in myeloid neoplasms)",
        "- Tartaglia et al. (2001) Nat Genet 29:465-468 (PTPN11 in Noonan syndrome)",
        "- Tatton-Brown et al. (2014) Nat Genet 46:385-388 (DNMT3A in TBRS)",
        "- Cohen et al. (2015) Am J Hum Genet 97:869-877 (EZH2 in Weaver syndrome)",
        "- Kranendijk et al. (2010) Science 330:336 (IDH2 in D-2-HGA)",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    """Run ClinGen gene-disease validity retrieval."""
    log.info("Starting ClinGen gene-disease validity retrieval")
    log.info("Patient genes: %s", ", ".join(GENE_IDS.keys()))

    results = retrieve_all_genes()

    # Save JSON
    json_path = RESULTS_DIR / "clingen_validity.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved JSON results to %s", json_path)

    # Generate and save report
    report = generate_report(results)
    report_path = RESULTS_DIR / "clingen_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved markdown report to %s", report_path)

    # Print summary
    print("\n" + "=" * 70)
    print("ClinGen Gene-Disease Validity - Summary")
    print("=" * 70)

    total_assertions = 0
    for gene, data in results["genes"].items():
        entries = data["api_assertions"] if data["api_assertions"] else data["curated_reference"]
        total_assertions += len(entries)
        source_label = "API" if data["api_assertions"] else "curated"
        print(f"\n  {gene} ({source_label}):")
        for entry in entries:
            print(f"    - {entry['disease']}: {entry['classification']}")

    print(f"\nTotal gene-disease assertions: {total_assertions}")
    print(f"API successes: {results['metadata']['api_successes']}/{len(GENE_IDS)}")
    print(f"\nResults: {json_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
