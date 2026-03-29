#!/usr/bin/env python3
"""
open_targets_search.py -- Query Open Targets Platform for patient mutation context.

Queries Open Targets Platform GraphQL API (v4) for:
  - Gene-disease associations for AML/MDS
  - Known drugs and clinical candidates per gene (drugAndClinicalCandidates)
  - Somatic mutation evidence from curated sources
  - Pharmacogenomics data

Note: Open Targets Genetics (api.genetics.opentargets.org) was retired and merged
into the main Platform API. GWAS/variant data is now accessed via the Platform's
variant/credibleSets queries, but these focus on common germline variants (GWAS),
not somatic hotspot mutations. The patient's somatic variants (R882H, R140Q, etc.)
are not indexed in the GWAS variant catalog.

Patient mutations:
  - DNMT3A R882H (VAF 39%)
  - IDH2 R140Q (VAF 2%)
  - SETBP1 G870S (VAF 34%)
  - PTPN11 E76Q (VAF 29%)
  - EZH2 V662A (VAF 59%, Pathogenic)

Outputs:
    - mutation_profile/results/cross_database/open_targets_results.json
    - mutation_profile/results/cross_database/open_targets_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/open_targets_search.py

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
# Configuration
# ---------------------------------------------------------------------------
PLATFORM_API = "https://api.platform.opentargets.org/api/v4/graphql"

PATIENT_GENES = {
    "DNMT3A": {"ensembl_id": "ENSG00000119772", "mutation": "R882H", "vaf": 0.39},
    "IDH2": {"ensembl_id": "ENSG00000182054", "mutation": "R140Q", "vaf": 0.02},
    "SETBP1": {"ensembl_id": "ENSG00000152217", "mutation": "G870S", "vaf": 0.34},
    "PTPN11": {"ensembl_id": "ENSG00000179295", "mutation": "E76Q", "vaf": 0.29},
    "EZH2": {"ensembl_id": "ENSG00000106462", "mutation": "V662A", "vaf": 0.59},
}

MYELOID_DISEASES = {
    "EFO_0000222": "acute myeloid leukemia",
    "EFO_0000198": "myelodysplastic syndrome",
    "EFO_0005952": "myeloproliferative neoplasm",
    "MONDO_0018881": "myelodysplastic/myeloproliferative neoplasm",
    "EFO_0000574": "leukemia",
    "EFO_0002430": "myeloid neoplasm",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REQUEST_DELAY = 0.5


# ---------------------------------------------------------------------------
# GraphQL helper
# ---------------------------------------------------------------------------
def _gql(query: str, variables: dict | None = None, retries: int = 3) -> dict | None:
    """Execute a GraphQL query against Open Targets Platform with retry."""
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    for attempt in range(retries):
        try:
            resp = requests.post(
                PLATFORM_API,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "errors" in data:
                    log.warning("GraphQL errors: %s", data["errors"][0].get("message", ""))
                return data.get("data")
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                log.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
            else:
                log.warning("HTTP %d: %s", resp.status_code, resp.text[:200])
                if attempt < retries - 1:
                    time.sleep(1)
        except requests.RequestException as e:
            log.warning("Request failed (attempt %d/%d): %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(1)
    return None


# ---------------------------------------------------------------------------
# Platform queries
# ---------------------------------------------------------------------------
def query_gene_info(ensembl_id: str) -> dict | None:
    """Get basic gene/target information including pathways."""
    query = """
    query TargetInfo($id: String!) {
      target(ensemblId: $id) {
        id
        approvedSymbol
        approvedName
        biotype
        functionDescriptions
        pathways {
          pathway
          pathwayId
        }
        tractability {
          label
          modality
          value
        }
      }
    }
    """
    time.sleep(REQUEST_DELAY)
    return _gql(query, {"id": ensembl_id})


def query_disease_associations(ensembl_id: str) -> dict | None:
    """Get disease associations for a gene target."""
    query = """
    query DiseaseAssociations($id: String!, $size: Int!) {
      target(ensemblId: $id) {
        id
        approvedSymbol
        associatedDiseases(page: {size: $size, index: 0}) {
          count
          rows {
            disease {
              id
              name
            }
            score
            datasourceScores {
              id
              score
            }
          }
        }
      }
    }
    """
    time.sleep(REQUEST_DELAY)
    return _gql(query, {"id": ensembl_id, "size": 100})


def query_drugs(ensembl_id: str) -> dict | None:
    """Get drugs and clinical candidates targeting this gene."""
    query = """
    query DrugCandidates($id: String!) {
      target(ensemblId: $id) {
        id
        approvedSymbol
        drugAndClinicalCandidates {
          count
          rows {
            drug {
              id
              name
              drugType
              maximumClinicalStage
              mechanismsOfAction {
                uniqueActionTypes
                rows {
                  actionType
                  mechanismOfAction
                }
              }
            }
            diseases {
              diseaseFromSource
              disease {
                id
                name
              }
            }
            maxClinicalStage
            clinicalReports {
              id
              clinicalStage
              trialPhase
              trialOverallStatus
              url
            }
          }
        }
      }
    }
    """
    time.sleep(REQUEST_DELAY)
    return _gql(query, {"id": ensembl_id})


def query_evidence(ensembl_id: str, disease_id: str) -> dict | None:
    """Get evidence linking a gene to a disease (somatic mutation data, literature, etc.)."""
    query = """
    query Evidence($id: String!, $efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        evidences(ensemblIds: [$id], size: $size) {
          count
          rows {
            id
            score
            datasourceId
            datatypeId
            literature
          }
        }
      }
    }
    """
    time.sleep(REQUEST_DELAY)
    return _gql(query, {"id": ensembl_id, "efoId": disease_id, "size": 50})


def query_pharmacogenomics(ensembl_id: str) -> dict | None:
    """Get pharmacogenomics data for a target."""
    query = """
    query Pharmacogenomics($id: String!) {
      target(ensemblId: $id) {
        id
        approvedSymbol
        pharmacogenomics {
          variantRsId
          genotype
          genotypeAnnotationText
          phenotypeText
          pgxCategory
          evidenceLevel
          datasourceId
          studyId
          literature
          genotypeId
          variantFunctionalConsequenceId
          targetFromSourceId
          drugs {
            drugId
            drugFromSource
          }
        }
      }
    }
    """
    time.sleep(REQUEST_DELAY)
    return _gql(query, {"id": ensembl_id})


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
MYELOID_KEYWORDS = [
    "leukemia", "leukaemia", "myeloid", "myelodysplastic", "myeloproliferative",
    "mds", "aml", "myeloma", "lymphoma", "hematolog", "haematolog",
    "blood", "bone marrow", "neoplasm",
]


def _is_heme(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in MYELOID_KEYWORDS)


def extract_myeloid_associations(data: dict) -> tuple[list[dict], int]:
    """Filter disease associations to myeloid/hematological. Returns (list, total_count)."""
    target = (data or {}).get("target")
    if not target:
        return [], 0

    ad = target.get("associatedDiseases", {})
    total = ad.get("count", 0)
    results = []

    for row in ad.get("rows", []):
        disease = row.get("disease", {})
        name = disease.get("name", "")
        disease_id = disease.get("id", "")

        if _is_heme(name) or disease_id in MYELOID_DISEASES:
            ds_scores = {ds["id"]: ds["score"] for ds in row.get("datasourceScores", [])}
            results.append({
                "disease_id": disease_id,
                "disease_name": name,
                "overall_score": row.get("score", 0),
                "datasource_scores": ds_scores,
            })

    results.sort(key=lambda x: x["overall_score"], reverse=True)
    return results, total


def extract_drugs(data: dict) -> dict:
    """Extract drug and clinical candidate info from drugAndClinicalCandidates."""
    target = (data or {}).get("target")
    if not target:
        return {"count": 0, "all_drugs": [], "heme_drugs": []}

    dcc = target.get("drugAndClinicalCandidates", {})
    count = dcc.get("count", 0)
    all_drugs = []
    heme_drugs = []

    for row in dcc.get("rows", []):
        drug = row.get("drug", {})
        diseases = row.get("diseases") or []

        disease_names = []
        disease_ids = []
        for d in diseases:
            inner = d.get("disease")
            if inner:
                disease_names.append(inner.get("name", ""))
                disease_ids.append(inner.get("id", ""))
            elif d.get("diseaseFromSource"):
                disease_names.append(d["diseaseFromSource"])

        moa_rows = drug.get("mechanismsOfAction", {}).get("rows", [])
        mechanisms = [m.get("mechanismOfAction", "") for m in moa_rows]

        reports = row.get("clinicalReports") or []
        trial_info = []
        for r in reports[:5]:
            trial_info.append({
                "id": r.get("id"),
                "phase": r.get("trialPhase"),
                "status": r.get("trialOverallStatus"),
                "url": r.get("url"),
            })

        entry = {
            "drug_name": drug.get("name", ""),
            "drug_id": drug.get("id", ""),
            "drug_type": drug.get("drugType", ""),
            "max_clinical_stage": row.get("maxClinicalStage", ""),
            "drug_max_stage": drug.get("maximumClinicalStage", ""),
            "diseases": disease_names,
            "disease_ids": disease_ids,
            "mechanisms": mechanisms,
            "is_hematological": any(_is_heme(n) for n in disease_names),
            "clinical_reports_count": len(reports),
            "sample_trials": trial_info,
        }

        all_drugs.append(entry)
        if entry["is_hematological"]:
            heme_drugs.append(entry)

    return {"count": count, "all_drugs": all_drugs, "heme_drugs": heme_drugs}


def extract_evidence(data: dict) -> dict:
    """Summarize evidence for a gene-disease pair."""
    disease = (data or {}).get("disease")
    if not disease:
        return {"count": 0, "datasources": {}, "top_evidence": []}

    evidences = disease.get("evidences", {})
    rows = evidences.get("rows", [])
    ds_counts: dict[str, int] = {}
    for row in rows:
        ds = row.get("datasourceId", "unknown")
        ds_counts[ds] = ds_counts.get(ds, 0) + 1

    top = sorted(rows, key=lambda x: x.get("score", 0), reverse=True)[:10]
    return {
        "disease_name": disease.get("name", ""),
        "count": evidences.get("count", 0),
        "datasources": ds_counts,
        "top_evidence": [
            {
                "datasource": e.get("datasourceId"),
                "datatype": e.get("datatypeId"),
                "score": e.get("score"),
                "pmids": e.get("literature", []),
            }
            for e in top
        ],
    }


def extract_pharmacogenomics(data: dict) -> list[dict]:
    """Extract pharmacogenomics entries."""
    target = (data or {}).get("target")
    if not target:
        return []
    return target.get("pharmacogenomics") or []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_queries() -> dict:
    """Run all Open Targets Platform queries for patient genes."""
    results = {}

    for gene, info in PATIENT_GENES.items():
        log.info("=== %s (%s, VAF %.0f%%) ===", gene, info["mutation"], info["vaf"] * 100)
        gr: dict[str, Any] = {
            "gene": gene,
            "ensembl_id": info["ensembl_id"],
            "mutation": info["mutation"],
            "vaf": info["vaf"],
        }

        # 1. Gene info
        log.info("  [1/5] Gene info...")
        gi_data = query_gene_info(info["ensembl_id"])
        if gi_data and gi_data.get("target"):
            t = gi_data["target"]
            gr["gene_info"] = {
                "approved_name": t.get("approvedName"),
                "biotype": t.get("biotype"),
                "function_descriptions": t.get("functionDescriptions", []),
                "pathways": [
                    {"name": p.get("pathway"), "id": p.get("pathwayId")}
                    for p in (t.get("pathways") or [])
                ],
                "tractability": t.get("tractability", []),
            }
        else:
            gr["gene_info"] = None

        # 2. Disease associations
        log.info("  [2/5] Disease associations...")
        assoc_data = query_disease_associations(info["ensembl_id"])
        myeloid_assocs, total_assocs = extract_myeloid_associations(assoc_data)
        gr["myeloid_associations"] = myeloid_assocs
        gr["total_disease_associations"] = total_assocs
        log.info("    %d myeloid associations (of %d total)", len(myeloid_assocs), total_assocs)

        # 3. Drugs
        log.info("  [3/5] Drugs and clinical candidates...")
        drugs_data = query_drugs(info["ensembl_id"])
        drug_info = extract_drugs(drugs_data)
        gr["drugs"] = drug_info
        log.info("    %d total drug entries, %d hematological",
                 drug_info["count"], len(drug_info["heme_drugs"]))

        # 4. Evidence for AML and MDS
        log.info("  [4/5] Evidence for AML/MDS...")
        ev_results = {}
        for did, dname in [("EFO_0000222", "AML"), ("EFO_0000198", "MDS")]:
            ev_data = query_evidence(info["ensembl_id"], did)
            ev_results[dname] = extract_evidence(ev_data)
            log.info("    %s: %d evidence entries", dname, ev_results[dname]["count"])
        gr["evidence"] = ev_results

        # 5. Pharmacogenomics
        log.info("  [5/5] Pharmacogenomics...")
        pgx_data = query_pharmacogenomics(info["ensembl_id"])
        pgx = extract_pharmacogenomics(pgx_data)
        gr["pharmacogenomics"] = pgx
        log.info("    %d pharmacogenomics entries", len(pgx))

        results[gene] = gr

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(results: dict) -> str:
    """Generate markdown report."""
    lines = [
        "# Open Targets Platform Search Results",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**API:** Open Targets Platform v4 GraphQL",
        f"**Endpoint:** {PLATFORM_API}",
        "",
        "## Overview",
        "",
        "Open Targets aggregates evidence from GWAS, somatic mutations, known drug targets,",
        "clinical trials, and literature to identify and validate drug targets. It is **not** a",
        "patient-level co-occurrence database. It complements the co-occurrence search (GENIE,",
        "cBioPortal, Beat AML, etc.) with functional annotation, drug-target relationships,",
        "and quantified evidence linking genes to hematological malignancies.",
        "",
        "**Note:** The former Open Targets Genetics API (api.genetics.opentargets.org) has been",
        "retired and merged into the main Platform. GWAS variant data is now in the Platform's",
        "variant/credibleSets queries, but these index common germline variants, not somatic",
        "hotspot mutations like R882H or R140Q.",
        "",
        "### Patient Mutations Queried",
        "",
        "| Gene | Mutation | VAF | Ensembl ID |",
        "|------|----------|-----|------------|",
    ]
    for gene, info in PATIENT_GENES.items():
        lines.append(f"| {gene} | {info['mutation']} | {info['vaf']*100:.0f}% | {info['ensembl_id']} |")
    lines.append("")

    # Per-gene sections
    for gene, info in PATIENT_GENES.items():
        r = results.get(gene, {})
        lines.extend(["---", "", f"## {gene} ({info['mutation']}, VAF {info['vaf']*100:.0f}%)", ""])

        # Gene info
        gi = r.get("gene_info")
        if gi:
            lines.append(f"**Full name:** {gi.get('approved_name', 'N/A')}")
            funcs = gi.get("function_descriptions", [])
            if funcs:
                lines.append(f"**Function:** {funcs[0][:400]}")
            lines.append("")

            # Tractability
            tract = gi.get("tractability", [])
            if tract:
                sm = [t for t in tract if t.get("modality") == "SM" and t.get("value")]
                ab = [t for t in tract if t.get("modality") == "AB" and t.get("value")]
                if sm or ab:
                    lines.append("**Tractability:**")
                    if sm:
                        lines.append(f"  - Small molecule: {', '.join(t['label'] for t in sm)}")
                    if ab:
                        lines.append(f"  - Antibody: {', '.join(t['label'] for t in ab)}")
                    lines.append("")

        # Disease associations
        assocs = r.get("myeloid_associations", [])
        total = r.get("total_disease_associations", 0)
        lines.append(f"### Disease Associations ({len(assocs)} myeloid of {total} total)")
        lines.append("")
        if assocs:
            lines.append("| Disease | Overall Score | Top Datasources |")
            lines.append("|---------|:------------:|-----------------|")
            for a in assocs[:15]:
                ds = a.get("datasource_scores", {})
                top_ds = sorted(ds.items(), key=lambda x: x[1], reverse=True)[:3]
                ds_str = ", ".join(f"{k}: {v:.2f}" for k, v in top_ds) if top_ds else "N/A"
                lines.append(f"| {a['disease_name']} | {a['overall_score']:.3f} | {ds_str} |")
            lines.append("")
        else:
            lines.append("No myeloid-relevant disease associations found.")
            lines.append("")

        # Drugs
        drugs = r.get("drugs", {})
        heme_drugs = drugs.get("heme_drugs", [])
        all_count = drugs.get("count", 0)
        lines.append(f"### Drugs and Clinical Candidates ({all_count} total, {len(heme_drugs)} hematological)")
        lines.append("")

        if heme_drugs:
            lines.append("| Drug | Max Stage | Diseases | Mechanism |")
            lines.append("|------|-----------|----------|-----------|")
            seen = set()
            for d in heme_drugs:
                name = d["drug_name"]
                if name in seen:
                    continue
                seen.add(name)
                diseases_str = "; ".join(d["diseases"][:3])
                mech = d["mechanisms"][0] if d["mechanisms"] else "N/A"
                lines.append(f"| {name} | {d['max_clinical_stage']} | {diseases_str[:80]} | {mech} |")
            lines.append("")

            # Show clinical trial details if available
            for d in heme_drugs:
                if d["sample_trials"]:
                    lines.append(f"**{d['drug_name']}** clinical trials:")
                    for t in d["sample_trials"][:3]:
                        status = t.get("status", "N/A")
                        phase = t.get("phase", "N/A")
                        url = t.get("url", "")
                        lines.append(f"  - Phase {phase}, status: {status}" +
                                     (f" ([link]({url}))" if url else ""))
                    lines.append("")
        else:
            # Show all drugs if no heme-specific ones
            all_drugs = drugs.get("all_drugs", [])
            if all_drugs:
                lines.append("No hematological drug entries. Other drugs targeting this gene:")
                lines.append("")
                lines.append("| Drug | Max Stage | Diseases | Mechanism |")
                lines.append("|------|-----------|----------|-----------|")
                seen = set()
                for d in all_drugs[:10]:
                    name = d["drug_name"]
                    if name in seen:
                        continue
                    seen.add(name)
                    diseases_str = "; ".join(d["diseases"][:2])
                    mech = d["mechanisms"][0] if d["mechanisms"] else "N/A"
                    lines.append(f"| {name} | {d['max_clinical_stage']} | {diseases_str[:60]} | {mech} |")
                lines.append("")
            else:
                lines.append("No drug entries found in Open Targets for this gene.")
                lines.append("")

        # Evidence
        evidence = r.get("evidence", {})
        if evidence:
            lines.append("### Evidence Summary")
            lines.append("")
            for dname, ev in evidence.items():
                n = ev.get("count", 0)
                ds_counts = ev.get("datasources", {})
                lines.append(f"**{dname}:** {n} evidence entries")
                if ds_counts:
                    ds_str = ", ".join(f"{k} ({v})" for k, v in
                                       sorted(ds_counts.items(), key=lambda x: -x[1]))
                    lines.append(f"  - Datasources: {ds_str}")
                lines.append("")

        # Pharmacogenomics
        pgx = r.get("pharmacogenomics", [])
        if pgx:
            lines.append(f"### Pharmacogenomics ({len(pgx)} entries)")
            lines.append("")
            lines.append("| Variant | Drug(s) | Genotype | Category | Evidence |")
            lines.append("|---------|---------|----------|----------|----------|")
            seen = set()
            for p in pgx[:15]:
                drugs_list = p.get("drugs") or []
                drug_names = ", ".join(d.get("drugFromSource", "?") for d in drugs_list) or "N/A"
                key = (p.get("variantRsId"), drug_names)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"| {p.get('variantRsId', 'N/A')} | {drug_names} | "
                    f"{p.get('genotype', 'N/A')} | {p.get('pgxCategory', 'N/A')} | "
                    f"{p.get('evidenceLevel', 'N/A')} |"
                )
            lines.append("")

    # Summary
    lines.extend(["---", "", "## Summary", ""])

    # Collect all heme drugs across genes
    all_heme = []
    for gene in PATIENT_GENES:
        r = results.get(gene, {})
        for d in r.get("drugs", {}).get("heme_drugs", []):
            all_heme.append({"gene": gene, **d})

    if all_heme:
        lines.append("### All Hematological Drug-Target Relationships")
        lines.append("")
        lines.append("| Gene | Drug | Max Stage | Mechanism | Diseases |")
        lines.append("|------|------|-----------|-----------|----------|")
        seen = set()
        for d in all_heme:
            key = (d["gene"], d["drug_name"])
            if key in seen:
                continue
            seen.add(key)
            mech = d["mechanisms"][0] if d["mechanisms"] else "N/A"
            diseases_str = "; ".join(d["diseases"][:2])
            lines.append(f"| {d['gene']} | {d['drug_name']} | {d['max_clinical_stage']} | "
                         f"{mech} | {diseases_str[:60]} |")
        lines.append("")

    # Association score summary
    lines.append("### Gene-Disease Association Scores (AML/MDS)")
    lines.append("")
    lines.append("| Gene | Mutation | AML Score | MDS Score | Total Myeloid Associations |")
    lines.append("|------|----------|:---------:|:---------:|:--------------------------:|")
    for gene, info in PATIENT_GENES.items():
        r = results.get(gene, {})
        assocs = r.get("myeloid_associations", [])
        aml_score = next((a["overall_score"] for a in assocs
                          if "acute myeloid" in a.get("disease_name", "").lower()), 0)
        mds_score = next((a["overall_score"] for a in assocs
                          if "myelodysplastic" in a.get("disease_name", "").lower()), 0)
        lines.append(f"| {gene} | {info['mutation']} | {aml_score:.3f} | "
                     f"{mds_score:.3f} | {len(assocs)} |")
    lines.append("")

    lines.extend([
        "### Interpretation",
        "",
        "1. **Not a patient-level database.** Open Targets aggregates gene-disease and",
        "   gene-drug relationships from published literature, clinical trials, and curated",
        "   databases. It does not contain individual patient mutation profiles or co-occurrence.",
        "",
        "2. **Drug-target relationships** confirmed here complement the co-occurrence analysis",
        "   by identifying approved drugs and active clinical trials for each mutated gene.",
        "",
        "3. **Disease association scores** (0-1) quantify evidence strength linking each gene",
        "   to AML/MDS across multiple sources (somatic mutations, literature, known targets,",
        "   animal models, pathways).",
        "",
        "4. **Somatic mutation context:** Open Targets confirms these genes as established",
        "   cancer drivers in myeloid malignancies through its cancer_gene_census, intogen,",
        "   and eva_somatic datasources.",
        "",
        "5. **The Genetics portal** (formerly api.genetics.opentargets.org) has been retired",
        "   and merged into the Platform. GWAS data now focuses on common germline variants",
        "   and is not relevant for somatic hotspot mutations like R882H or R140Q.",
        "",
        "### Data Sources Integrated by Open Targets",
        "",
        "- ClinVar, EVA (somatic/germline variants)",
        "- ChEMBL (drugs and clinical trials)",
        "- Cancer Gene Census / IntOGen (cancer driver genes)",
        "- GWAS Catalog / UK Biobank (genetic associations)",
        "- UniProt, Reactome (protein function, pathways)",
        "- Europe PMC (text-mined literature associations)",
        "- PharmGKB (pharmacogenomics)",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    log.info("Starting Open Targets Platform search for patient mutations")
    log.info("Genes: %s", ", ".join(f"{g} {i['mutation']}" for g, i in PATIENT_GENES.items()))
    log.info("API: %s", PLATFORM_API)

    results = run_queries()

    # Save JSON
    output = {
        "query_date": datetime.now(timezone.utc).isoformat(),
        "api": PLATFORM_API,
        "note": (
            "Open Targets is a drug target validation platform, not a patient-level "
            "co-occurrence database. It complements co-occurrence searches with functional "
            "annotation, drug-target relationships, and genetic evidence."
        ),
        "patient_genes": {g: i for g, i in PATIENT_GENES.items()},
        "results": results,
    }

    json_path = RESULTS_DIR / "open_targets_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved JSON: %s", json_path)

    # Generate report
    report = generate_report(results)
    report_path = RESULTS_DIR / "open_targets_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved report: %s", report_path)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for gene, info in PATIENT_GENES.items():
        r = results.get(gene, {})
        n_assoc = len(r.get("myeloid_associations", []))
        n_drugs = r.get("drugs", {}).get("count", 0)
        n_heme = len(r.get("drugs", {}).get("heme_drugs", []))
        aml_ev = r.get("evidence", {}).get("AML", {}).get("count", 0)
        mds_ev = r.get("evidence", {}).get("MDS", {}).get("count", 0)
        n_pgx = len(r.get("pharmacogenomics", []))
        log.info("  %s %s: %d myeloid assocs, %d drugs (%d heme), %d/%d AML/MDS evidence, %d PGx",
                 gene, info["mutation"], n_assoc, n_drugs, n_heme, aml_ev, mds_ev, n_pgx)

    log.info("")
    log.info("Results: %s", json_path)
    log.info("Report:  %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
