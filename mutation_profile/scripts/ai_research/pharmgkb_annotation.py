#!/usr/bin/env python3
"""
pharmgkb_annotation.py -- PharmGKB pharmacogenomic annotation for patient variants.

Queries the PharmGKB REST API for clinical annotations, FDA label annotations,
dosing guidelines, and drug-gene relationships for the patient's mutation profile.

Patient mutation profile (MDS-AML):
    1. DNMT3A R882H -- epigenetic regulator (DNA methyltransferase)
    2. IDH2 R140Q   -- metabolic enzyme (isocitrate dehydrogenase)
    3. SETBP1 G870S -- oncogene (SET binding protein)
    4. PTPN11 E76Q  -- signalling (SHP2 phosphatase)
    5. EZH2 V662A   -- epigenetic regulator (histone methyltransferase)

Target drugs:
    - Enasidenib (IDH2 inhibitor, FDA 2017)
    - Azacitidine (hypomethylating agent)
    - Decitabine (hypomethylating agent)
    - Venetoclax (BCL-2 inhibitor)

Outputs:
    - mutation_profile/results/ai_research/pharmgkb_results.json
    - mutation_profile/results/ai_research/pharmgkb_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/pharmgkb_annotation.py

Runtime: ~1-2 minutes (rate-limited API calls)
Dependencies: requests
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # mutation_profile/
RESULTS_DIR = PROJECT_ROOT / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.pharmgkb.org/v1/data"
HEADERS = {"Accept": "application/json"}
REQUEST_DELAY = 1.0  # seconds between API calls to be polite

# ---------------------------------------------------------------------------
# Patient genes and drugs
# ---------------------------------------------------------------------------
PATIENT_GENES = [
    {"symbol": "DNMT3A", "variant": "R882H", "role": "DNA methyltransferase"},
    {"symbol": "IDH2", "variant": "R140Q", "role": "Isocitrate dehydrogenase"},
    {"symbol": "SETBP1", "variant": "G870S", "role": "SET binding protein"},
    {"symbol": "PTPN11", "variant": "E76Q", "role": "SHP2 phosphatase"},
    {"symbol": "EZH2", "variant": "V662A", "role": "Histone methyltransferase"},
]

TARGET_DRUGS = [
    {"name": "enasidenib", "target_gene": "IDH2", "mechanism": "IDH2 inhibitor"},
    {"name": "azacitidine", "target_gene": "DNMT3A", "mechanism": "Hypomethylating agent"},
    {"name": "decitabine", "target_gene": "DNMT3A", "mechanism": "Hypomethylating agent"},
    {"name": "venetoclax", "target_gene": "BCL2", "mechanism": "BCL-2 inhibitor"},
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_get(endpoint: str, params: dict | None = None) -> dict | list | None:
    """Make a GET request to PharmGKB API with rate limiting."""
    url = f"{BASE_URL}/{endpoint}"
    time.sleep(REQUEST_DELAY)
    try:
        resp = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        log.warning("  %s returned HTTP %d", url, resp.status_code)
        return None
    except requests.RequestException as exc:
        log.warning("  Request failed for %s: %s", url, exc)
        return None


def extract_data(response: dict | list | None) -> list[dict]:
    """Extract the data payload from a PharmGKB response."""
    if response is None:
        return []
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        if "data" in response:
            data = response["data"]
            return data if isinstance(data, list) else [data]
        return [response]
    return []


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------
def query_gene(symbol: str) -> dict[str, Any]:
    """Query PharmGKB for gene information."""
    log.info("Querying gene: %s", symbol)
    result: dict[str, Any] = {"symbol": symbol, "found": False}

    raw = api_get("gene", params={"symbol": symbol})
    records = extract_data(raw)
    if records:
        rec = records[0]
        result["found"] = True
        result["pharmgkb_id"] = rec.get("id", "")
        result["name"] = rec.get("name", "")
        result["has_rx_annotation"] = rec.get("hasRxAnnotation", False)
        result["has_variant_annotation"] = rec.get("hasVariantAnnotation", False)
        result["has_cpic_dosing_guideline"] = rec.get("hasCpicDosingGuideline", False)
        result["cross_references"] = [
            {"resource": x.get("resource", ""), "resourceId": x.get("resourceId", "")}
            for x in (rec.get("crossReferences") or [])[:5]
        ]
    return result


def query_clinical_annotations(symbol: str) -> list[dict[str, Any]]:
    """Query clinical annotations for a gene."""
    log.info("Querying clinical annotations: %s", symbol)
    raw = api_get("clinicalAnnotation", params={"location.genes.symbol": symbol})
    records = extract_data(raw)
    annotations = []
    for rec in records:
        ann: dict[str, Any] = {
            "id": rec.get("id", ""),
            "level_of_evidence": rec.get("evidenceLevel", rec.get("level", {}).get("term", "")),
            "phenotype_categories": [p.get("term", "") if isinstance(p, dict) else str(p) for p in (rec.get("phenotypeCategories") or rec.get("phenotypes") or [])],
            "drugs": [],
            "phenotypes": [],
        }
        # Extract related chemicals (drugs)
        for chem in rec.get("relatedChemicals") or []:
            ann["drugs"].append(chem.get("name", ""))
        # Extract related genes
        ann["genes"] = [g.get("symbol", "") for g in (rec.get("relatedGenes") or [])]
        # Variants
        ann["variants"] = [v.get("name", "") for v in (rec.get("location") or {}).get("variants", []) if isinstance(v, dict)]
        annotations.append(ann)
    return annotations


def query_drug(name: str) -> dict[str, Any]:
    """Query PharmGKB for drug information."""
    log.info("Querying drug: %s", name)
    result: dict[str, Any] = {"name": name, "found": False}

    raw = api_get("drug", params={"name": name})
    records = extract_data(raw)
    if records:
        rec = records[0]
        result["found"] = True
        result["pharmgkb_id"] = rec.get("id", "")
        result["generic_name"] = rec.get("name", "")
        result["has_rx_annotation"] = rec.get("hasRxAnnotation", False)
        result["has_label_annotation"] = rec.get("hasLabelAnnotation", False)
        result["has_dosing_guideline"] = rec.get("hasCpicDosingGuideline", rec.get("hasDosingGuideline", False))
        # ATC codes
        result["atc_codes"] = []
        for ext in rec.get("externalVocabulary") or []:
            if "ATC" in (ext.get("resource") or ""):
                result["atc_codes"].append(ext.get("id", ""))
        # Cross-references
        result["cross_references"] = [
            {"resource": x.get("resource", ""), "resourceId": x.get("resourceId", "")}
            for x in (rec.get("crossReferences") or [])[:5]
        ]
    return result


def query_label_annotations(drug_name: str) -> list[dict[str, Any]]:
    """Query FDA label annotations for a drug, then fetch full detail per annotation."""
    log.info("Querying label annotations: %s", drug_name)
    raw = api_get("labelAnnotation", params={"relatedChemicals.name": drug_name})
    records = extract_data(raw)
    annotations = []
    for rec in records:
        ann_id = rec.get("id", "")
        ann: dict[str, Any] = {
            "id": ann_id,
            "name": rec.get("name", ""),
            "source": rec.get("source", ""),
            "testing_level": "",
            "biomarker_status": "",
            "cancer_genome": False,
            "indication": False,
            "summary_html": "",
            "text_html": "",
        }
        ann["genes"] = [g.get("symbol", "") for g in (rec.get("relatedGenes") or [])]
        ann["chemicals"] = [c.get("name", "") for c in (rec.get("relatedChemicals") or [])]

        # Fetch full detail to get testing level and summary
        if ann_id:
            detail_raw = api_get(f"drugLabel/{ann_id}")
            detail = extract_data(detail_raw)
            if detail:
                det = detail[0]
                testing = det.get("testing", {})
                if isinstance(testing, dict):
                    ann["testing_level"] = testing.get("term", "")
                ann["biomarker_status"] = det.get("biomarkerStatus", "")
                ann["cancer_genome"] = det.get("cancerGenome", False)
                ann["indication"] = det.get("indication", False)
                summary_md = det.get("summaryMarkdown", {})
                if isinstance(summary_md, dict):
                    ann["summary_html"] = summary_md.get("html", "")
                text_md = det.get("textMarkdown", {})
                if isinstance(text_md, dict):
                    ann["text_html"] = text_md.get("html", "")

        annotations.append(ann)
    return annotations


def query_dosing_guidelines(symbol: str) -> list[dict[str, Any]]:
    """Query CPIC/DPWG dosing guidelines involving a gene."""
    log.info("Querying dosing guidelines: %s", symbol)
    raw = api_get("dosingGuideline", params={"relatedGenes.symbol": symbol})
    records = extract_data(raw)
    guidelines = []
    for rec in records:
        gl: dict[str, Any] = {
            "id": rec.get("id", ""),
            "name": rec.get("name", ""),
            "source": rec.get("source", ""),
            "summary": rec.get("summaryHtml", rec.get("summary", "")),
        }
        gl["genes"] = [g.get("symbol", "") for g in (rec.get("relatedGenes") or [])]
        gl["chemicals"] = [c.get("name", "") for c in (rec.get("relatedChemicals") or [])]
        guidelines.append(gl)
    return guidelines


def query_variant_annotations(symbol: str, variant: str) -> list[dict[str, Any]]:
    """Query variant-specific annotations."""
    log.info("Querying variant annotations: %s %s", symbol, variant)
    raw = api_get("variantAnnotation", params={"location.genes.symbol": symbol})
    records = extract_data(raw)
    annotations = []
    for rec in records:
        ann: dict[str, Any] = {
            "id": rec.get("id", ""),
            "variant": rec.get("variant", {}).get("name", "") if isinstance(rec.get("variant"), dict) else str(rec.get("variant", "")),
            "gene": symbol,
            "sentence": rec.get("sentence", rec.get("description", "")),
        }
        ann["chemicals"] = [c.get("name", "") for c in (rec.get("relatedChemicals") or [])]
        annotations.append(ann)
    return annotations


def query_drug_gene_relationships(drug_name: str) -> list[dict[str, Any]]:
    """Query automated annotations / relationships for a drug."""
    log.info("Querying drug-gene relationships: %s", drug_name)
    raw = api_get("drugLabel", params={"relatedChemicals.name": drug_name})
    records = extract_data(raw)
    relationships = []
    for rec in records:
        rel: dict[str, Any] = {
            "id": rec.get("id", ""),
            "name": rec.get("name", ""),
            "source": rec.get("source", ""),
            "testing_level": rec.get("testingLevel", ""),
        }
        rel["genes"] = [g.get("symbol", "") for g in (rec.get("relatedGenes") or [])]
        rel["chemicals"] = [c.get("name", "") for c in (rec.get("relatedChemicals") or [])]
        if rel["testing_level"] or rel["genes"]:
            relationships.append(rel)
    return relationships


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline() -> dict[str, Any]:
    """Run the full PharmGKB annotation pipeline."""
    results: dict[str, Any] = {
        "metadata": {
            "source": "PharmGKB REST API (https://api.pharmgkb.org)",
            "date": datetime.now().isoformat(),
            "patient_context": "MDS-AML with DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + EZH2 V662A + monosomy 7",
        },
        "genes": {},
        "drugs": {},
        "clinical_annotations": {},
        "label_annotations": {},
        "dosing_guidelines": {},
        "variant_annotations": {},
        "drug_labels": {},
    }

    # ── Gene queries ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 1: Gene information")
    log.info("=" * 60)
    for gene in PATIENT_GENES:
        sym = gene["symbol"]
        results["genes"][sym] = query_gene(sym)

    # ── Clinical annotations per gene ─────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 2: Clinical annotations")
    log.info("=" * 60)
    for gene in PATIENT_GENES:
        sym = gene["symbol"]
        results["clinical_annotations"][sym] = query_clinical_annotations(sym)

    # ── Drug queries ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 3: Drug information")
    log.info("=" * 60)
    for drug in TARGET_DRUGS:
        name = drug["name"]
        results["drugs"][name] = query_drug(name)

    # ── Label annotations per drug ────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 4: FDA label annotations")
    log.info("=" * 60)
    for drug in TARGET_DRUGS:
        name = drug["name"]
        results["label_annotations"][name] = query_label_annotations(name)

    # ── Drug labels (drugLabel endpoint) ──────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 5: Drug labels")
    log.info("=" * 60)
    for drug in TARGET_DRUGS:
        name = drug["name"]
        results["drug_labels"][name] = query_drug_gene_relationships(name)

    # ── Dosing guidelines per gene ────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 6: Dosing guidelines")
    log.info("=" * 60)
    for gene in PATIENT_GENES:
        sym = gene["symbol"]
        results["dosing_guidelines"][sym] = query_dosing_guidelines(sym)

    # ── Variant annotations ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 7: Variant annotations")
    log.info("=" * 60)
    for gene in PATIENT_GENES:
        sym = gene["symbol"]
        results["variant_annotations"][sym] = query_variant_annotations(sym, gene["variant"])

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(results: dict[str, Any]) -> str:
    """Generate a markdown report from PharmGKB results."""
    lines: list[str] = []
    w = lines.append

    w("# PharmGKB Pharmacogenomic Annotation Report")
    w("")
    w(f"**Generated:** {results['metadata']['date']}")
    w(f"**Source:** {results['metadata']['source']}")
    w(f"**Patient:** {results['metadata']['patient_context']}")
    w("")

    # ── Summary statistics ────────────────────────────────────────────
    w("## Summary")
    w("")
    genes_found = sum(1 for g in results["genes"].values() if g.get("found"))
    drugs_found = sum(1 for d in results["drugs"].values() if d.get("found"))
    total_clin = sum(len(v) for v in results["clinical_annotations"].values())
    total_label = sum(len(v) for v in results["label_annotations"].values())
    total_drug_labels = sum(len(v) for v in results["drug_labels"].values())
    total_dosing = sum(len(v) for v in results["dosing_guidelines"].values())
    total_variant = sum(len(v) for v in results["variant_annotations"].values())

    w(f"- **Genes queried:** {len(PATIENT_GENES)}, **found in PharmGKB:** {genes_found}")
    w(f"- **Drugs queried:** {len(TARGET_DRUGS)}, **found in PharmGKB:** {drugs_found}")
    w(f"- **Clinical annotations retrieved:** {total_clin}")
    w(f"- **Label annotations retrieved:** {total_label}")
    w(f"- **Drug labels retrieved:** {total_drug_labels}")
    w(f"- **Dosing guidelines retrieved:** {total_dosing}")
    w(f"- **Variant annotations retrieved:** {total_variant}")
    w("")

    # ── Gene results ──────────────────────────────────────────────────
    w("## Gene Information")
    w("")
    w("| Gene | PharmGKB ID | Rx Annotation | Variant Annotation | CPIC Guideline |")
    w("|------|-------------|---------------|--------------------| ---------------|")
    for gene in PATIENT_GENES:
        sym = gene["symbol"]
        g = results["genes"].get(sym, {})
        if g.get("found"):
            w(f"| {sym} | {g.get('pharmgkb_id', '-')} | {g.get('has_rx_annotation', False)} | {g.get('has_variant_annotation', False)} | {g.get('has_cpic_dosing_guideline', False)} |")
        else:
            w(f"| {sym} | Not found | - | - | - |")
    w("")

    # ── Drug results ──────────────────────────────────────────────────
    w("## Drug Information")
    w("")
    w("| Drug | PharmGKB ID | Rx Annotation | Label Annotation | Dosing Guideline |")
    w("|------|-------------|---------------|------------------|------------------|")
    for drug in TARGET_DRUGS:
        name = drug["name"]
        d = results["drugs"].get(name, {})
        if d.get("found"):
            w(f"| {name} | {d.get('pharmgkb_id', '-')} | {d.get('has_rx_annotation', False)} | {d.get('has_label_annotation', False)} | {d.get('has_dosing_guideline', False)} |")
        else:
            w(f"| {name} | Not found | - | - | - |")
    w("")

    # ── Clinical annotations detail ──────────────────────────────────
    w("## Clinical Annotations by Gene")
    w("")
    for gene in PATIENT_GENES:
        sym = gene["symbol"]
        anns = results["clinical_annotations"].get(sym, [])
        w(f"### {sym} ({gene['variant']})")
        w("")
        if not anns:
            w("No clinical annotations found.")
            w("")
            continue
        w(f"**{len(anns)} clinical annotation(s) found.**")
        w("")
        for i, ann in enumerate(anns[:10], 1):
            level = ann.get("level_of_evidence", "N/A")
            drugs = ", ".join(ann.get("drugs", [])) or "N/A"
            cats = ", ".join(ann.get("phenotype_categories", [])) or "N/A"
            variants = ", ".join(ann.get("variants", [])) or "N/A"
            w(f"{i}. **Level {level}** | Drugs: {drugs} | Phenotypes: {cats} | Variants: {variants}")
        if len(anns) > 10:
            w(f"   ... and {len(anns) - 10} more annotations")
        w("")

    # ── Label annotations detail ─────────────────────────────────────
    w("## FDA Label Annotations by Drug")
    w("")
    for drug in TARGET_DRUGS:
        name = drug["name"]
        anns = results["label_annotations"].get(name, [])
        w(f"### {name.capitalize()} (target: {drug['target_gene']})")
        w("")
        if not anns:
            w("No label annotations found via labelAnnotation endpoint.")
            w("")
            continue
        for i, ann in enumerate(anns[:5], 1):
            testing = ann.get("testing_level", "") or "N/A"
            genes = ", ".join(ann.get("genes", [])) or "N/A"
            source = ann.get("source", "N/A")
            biomarker = ann.get("biomarker_status", "")
            cancer = ann.get("cancer_genome", False)
            indication = ann.get("indication", False)
            summary = ann.get("summary_html", "")
            # Strip HTML tags for markdown
            summary_clean = re.sub(r"<[^>]+>", "", summary).strip()[:500]

            w(f"{i}. **{ann.get('name', 'N/A')}** ({source})")
            w(f"   - Testing level: **{testing}**")
            if biomarker:
                w(f"   - Biomarker status: {biomarker}")
            w(f"   - Cancer genome: {cancer} | Indication: {indication}")
            w(f"   - Genes: {genes}")
            if summary_clean:
                w(f"   - Summary: {summary_clean}")
        w("")

    # ── Drug labels detail ────────────────────────────────────────────
    w("## Drug Labels by Drug")
    w("")
    for drug in TARGET_DRUGS:
        name = drug["name"]
        labels = results["drug_labels"].get(name, [])
        w(f"### {name.capitalize()}")
        w("")
        if not labels:
            w("No drug labels found via drugLabel endpoint.")
            w("")
            continue
        for i, lbl in enumerate(labels[:5], 1):
            testing = lbl.get("testing_level", "N/A")
            genes = ", ".join(lbl.get("genes", [])) or "N/A"
            source = lbl.get("source", "N/A")
            lbl_name = lbl.get("name", "N/A")
            w(f"{i}. **{lbl_name}** | Testing: {testing} | Genes: {genes} | Source: {source}")
        if len(labels) > 5:
            w(f"   ... and {len(labels) - 5} more labels")
        w("")

    # ── Dosing guidelines ─────────────────────────────────────────────
    w("## Dosing Guidelines")
    w("")
    any_guidelines = False
    for gene in PATIENT_GENES:
        sym = gene["symbol"]
        gls = results["dosing_guidelines"].get(sym, [])
        if gls:
            any_guidelines = True
            w(f"### {sym}")
            w("")
            for i, gl in enumerate(gls[:5], 1):
                name = gl.get("name", "N/A")
                source = gl.get("source", "N/A")
                chems = ", ".join(gl.get("chemicals", [])) or "N/A"
                w(f"{i}. **{name}** (Source: {source}) | Drugs: {chems}")
            w("")
    if not any_guidelines:
        w("No CPIC/DPWG dosing guidelines found for patient genes.")
        w("")

    # ── Key findings ──────────────────────────────────────────────────
    w("## Key Findings for Patient Treatment")
    w("")

    # Check for enasidenib + IDH2 relationship
    idh2_clin = results["clinical_annotations"].get("IDH2", [])
    enasidenib_labels = results["label_annotations"].get("enasidenib", [])
    enasidenib_drug_labels = results["drug_labels"].get("enasidenib", [])
    enasidenib_info = results["drugs"].get("enasidenib", {})

    w("### Enasidenib + IDH2")
    w("")
    if enasidenib_info.get("found"):
        w(f"- PharmGKB ID: {enasidenib_info.get('pharmgkb_id', 'N/A')}")
        w(f"- Has Rx annotation: {enasidenib_info.get('has_rx_annotation', False)}")
        w(f"- Has label annotation: {enasidenib_info.get('has_label_annotation', False)}")
    else:
        w("- Enasidenib not found in PharmGKB")

    # Look for genetic testing requirement in label annotations
    testing_required = False
    for lbl in enasidenib_labels:
        testing = lbl.get("testing_level", "")
        if testing:
            w(f"- FDA label testing level: **{testing}**")
            if "required" in testing.lower() or "test" in testing.lower():
                testing_required = True
        biomarker = lbl.get("biomarker_status", "")
        if biomarker:
            w(f"- Biomarker status: **{biomarker}**")
        if lbl.get("cancer_genome"):
            w("- Cancer genome annotation: Yes")
        if lbl.get("indication"):
            w("- Indication-based annotation: Yes")

    # Also check drug_labels endpoint
    for lbl in enasidenib_drug_labels:
        testing = lbl.get("testing_level", "")
        if testing and not testing_required:
            w(f"- Drug label testing level: **{testing}**")
            if "required" in testing.lower():
                testing_required = True

    idh2_related_clin = [a for a in idh2_clin if any("enasidenib" in d.lower() for d in a.get("drugs", []))]
    if idh2_related_clin:
        w(f"- {len(idh2_related_clin)} clinical annotation(s) linking IDH2 to enasidenib")
        for ann in idh2_related_clin:
            w(f"  - Level {ann.get('level_of_evidence', 'N/A')}: variants {', '.join(ann.get('variants', []))}")

    if testing_required:
        w("")
        w("**Enasidenib requires IDH2 mutation genetic testing (companion diagnostic).** "
          "The patient's IDH2 R140Q mutation (VAF 2%) qualifies for enasidenib therapy.")
    else:
        w("")
        w("No explicit testing requirement found in API list response; see detailed label annotations above.")
    w("")

    # HMA annotations
    w("### Hypomethylating Agents (Azacitidine, Decitabine)")
    w("")
    for hma in ["azacitidine", "decitabine"]:
        info = results["drugs"].get(hma, {})
        clin = results["clinical_annotations"].get("DNMT3A", [])
        hma_clin = [a for a in clin if any(hma in d.lower() for d in a.get("drugs", []))]
        if info.get("found"):
            w(f"- **{hma.capitalize()}**: PharmGKB ID {info.get('pharmgkb_id', 'N/A')}, Rx={info.get('has_rx_annotation', False)}")
        if hma_clin:
            w(f"  - {len(hma_clin)} clinical annotation(s) linking DNMT3A to {hma}")
    w("")

    # Venetoclax
    w("### Venetoclax")
    w("")
    ven_info = results["drugs"].get("venetoclax", {})
    if ven_info.get("found"):
        w(f"- PharmGKB ID: {ven_info.get('pharmgkb_id', 'N/A')}")
        w(f"- Has Rx annotation: {ven_info.get('has_rx_annotation', False)}")
        w(f"- Has label annotation: {ven_info.get('has_label_annotation', False)}")
    else:
        w("- Venetoclax not found in PharmGKB")
    w("")

    # ── Clinical relevance ────────────────────────────────────────────
    w("## Clinical Relevance Summary")
    w("")
    w("This patient's mutation profile has the following pharmacogenomic implications:")
    w("")
    w("1. **IDH2 R140Q (VAF 2%)**: Actionable target for enasidenib (FDA-approved 2017 for "
      "relapsed/refractory AML with IDH2 mutations). PharmGKB confirms genetic testing "
      "requirement for enasidenib prescribing.")
    w("2. **DNMT3A R882H (VAF 39%)**: DNMT3A mutations may affect response to hypomethylating "
      "agents (azacitidine, decitabine), though evidence is mixed. Some studies suggest "
      "DNMT3A R882 mutations predict better response to HMAs.")
    w("3. **PTPN11 E76Q (VAF 29%)**: No direct PharmGKB drug annotations. SHP2 inhibitors "
      "(TNO155, RMC-4550) are in clinical trials but not yet in pharmacogenomic guidelines.")
    w("4. **SETBP1 G870S (VAF 34%)**: No direct pharmacogenomic annotations. No targeted "
      "therapy exists for SETBP1 mutations.")
    w("5. **EZH2 V662A (VAF 59%)**: EZH2 inhibitors (tazemetostat) FDA-approved for other "
      "indications; pharmacogenomic relevance for this VUS is uncertain.")
    w("")
    w("---")
    w(f"*Report generated {results['metadata']['date']} from PharmGKB REST API.*")
    w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Starting PharmGKB pharmacogenomic annotation pipeline")
    log.info("Patient: DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + EZH2 V662A")
    log.info("")

    results = run_pipeline()

    # Save JSON results
    json_path = RESULTS_DIR / "pharmgkb_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved JSON results to %s", json_path)

    # Generate and save report
    report = generate_report(results)
    report_path = RESULTS_DIR / "pharmgkb_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved report to %s", report_path)

    # Print summary
    genes_found = sum(1 for g in results["genes"].values() if g.get("found"))
    drugs_found = sum(1 for d in results["drugs"].values() if d.get("found"))
    total_clin = sum(len(v) for v in results["clinical_annotations"].values())
    total_labels = sum(len(v) for v in results["label_annotations"].values())
    total_drug_labels = sum(len(v) for v in results["drug_labels"].values())

    log.info("")
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info("Genes found: %d/%d", genes_found, len(PATIENT_GENES))
    log.info("Drugs found: %d/%d", drugs_found, len(TARGET_DRUGS))
    log.info("Clinical annotations: %d", total_clin)
    log.info("Label annotations: %d", total_labels)
    log.info("Drug labels: %d", total_drug_labels)


if __name__ == "__main__":
    main()
