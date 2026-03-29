#!/usr/bin/env python3
"""
Multi-source pathogenicity scoring for patient variants.

Queries AlphaMissense, CADD, REVEL, and other pre-computed databases
for pathogenicity predictions on the 5 patient somatic mutations.

Sources:
    - AlphaMissense: Pre-computed scores from Google DeepMind (AUROC 0.94)
    - CADD v1.7: Combined Annotation Dependent Depletion (Phred-scaled)
    - REVEL: Rare Exome Variant Ensemble Learner (ensemble of 13 tools)
    - ClinVar: Clinical significance from NCBI
    - gnomAD: Population allele frequencies

Patient variants:
    1. EZH2 V662A (c.1985T>C) - VAF 59%, founder clone
    2. DNMT3A R882H - VAF 39%, pathogenic hotspot
    3. SETBP1 G870S - VAF 34%, likely pathogenic
    4. PTPN11 E76Q - VAF 29%, pathogenic
    5. IDH2 R140Q - VAF 2%, pathogenic subclone

Usage:
    python -m mutation_profile.scripts.ai_research.pathogenicity_scores
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "ai_research" / "pathogenicity_scoring"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Patient variants ────────────────────────────────────────────────────────

@dataclass
class Variant:
    """A patient somatic variant with genomic coordinates."""
    gene: str
    protein_change: str
    hgvs_c: str
    vaf: float
    chromosome: str
    position: int       # GRCh38
    ref: str
    alt: str
    transcript: str
    # Scores filled in by queries
    alphamissense: float | None = None
    alphamissense_class: str | None = None
    cadd_phred: float | None = None
    cadd_raw: float | None = None
    revel: float | None = None
    clinvar_significance: str | None = None
    clinvar_review: str | None = None
    gnomad_af: float | None = None
    sift_score: float | None = None
    polyphen2_score: float | None = None
    extra: dict = field(default_factory=dict)


# GRCh38 coordinates for patient variants (verified via VariantValidator + ClinVar VCV003373649 (2026-03-28))
PATIENT_VARIANTS = [
    Variant(
        gene="EZH2", protein_change="V662A", hgvs_c="c.1985T>C",
        vaf=0.59, chromosome="7", position=148810377, ref="A", alt="G",
        transcript="NM_004456.5",
    ),
    Variant(
        gene="DNMT3A", protein_change="R882H", hgvs_c="c.2645G>A",
        vaf=0.39, chromosome="2", position=25234373, ref="C", alt="T",
        transcript="NM_022552.5",
    ),
    Variant(
        gene="SETBP1", protein_change="G870S", hgvs_c="c.2608G>A",
        vaf=0.34, chromosome="18", position=44951948, ref="G", alt="A",
        transcript="NM_015559.3",
    ),
    Variant(
        gene="PTPN11", protein_change="E76Q", hgvs_c="c.226G>C",
        vaf=0.29, chromosome="12", position=112450406, ref="G", alt="C",
        transcript="NM_002834.5",
    ),
    Variant(
        gene="IDH2", protein_change="R140Q", hgvs_c="c.419G>A",
        vaf=0.02, chromosome="15", position=90088702, ref="C", alt="T",
        transcript="NM_002168.4",
    ),
]

# GRCh37 (hg19) coordinates for myvariant.info queries
GRCH37_COORDS = {
    "EZH2": {"chrom": "7", "pos": 148507469, "ref": "A", "alt": "G"},
    "DNMT3A": {"chrom": "2", "pos": 25457242, "ref": "C", "alt": "T"},
    "SETBP1": {"chrom": "18", "pos": 42531913, "ref": "G", "alt": "A"},
    "PTPN11": {"chrom": "12", "pos": 112888210, "ref": "G", "alt": "C"},
    "IDH2": {"chrom": "15", "pos": 90631934, "ref": "C", "alt": "T"},
}


# ── AlphaMissense lookup ────────────────────────────────────────────────────

def query_alphamissense(variants: list[Variant]) -> None:
    """Look up AlphaMissense scores via the pre-computed TSV hosted on Zenodo.

    AlphaMissense provides genome-wide missense pathogenicity predictions.
    Score ranges: <0.34 = likely benign, 0.34-0.564 = ambiguous, >0.564 = likely pathogenic.
    """
    log.info("Querying AlphaMissense pre-computed scores...")

    # Use VEP region endpoint with AlphaMissense plugin
    for v in variants:
        try:
            region = f"{v.chromosome}:{v.position}:{v.position}/{v.alt}"
            url = (
                f"https://rest.ensembl.org/vep/human/region/{region}"
                f"?content-type=application/json"
                f"&variant_class=1"
            )
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list):
                    for entry in data:
                        for tc in entry.get("transcript_consequences", []):
                            am_score = tc.get("am_pathogenicity")
                            am_class = tc.get("am_class")
                            if am_score is not None:
                                v.alphamissense = float(am_score)
                                v.alphamissense_class = am_class
                                log.info(f"  {v.gene} {v.protein_change}: AM={am_score} ({am_class})")
                                break
                            # Also check cadd/revel while we're here
                            revel = tc.get("revel_score")
                            if revel is not None and v.revel is None:
                                v.revel = float(revel) if isinstance(revel, (int, float)) else float(str(revel).split(",")[0])
                            cadd = tc.get("cadd_phred")
                            if cadd is not None and v.cadd_phred is None:
                                v.cadd_phred = float(cadd)
                        if v.alphamissense is not None:
                            break
                if v.alphamissense is None:
                    log.warning(f"  {v.gene} {v.protein_change}: No AlphaMissense score in VEP response")
            else:
                log.warning(f"  {v.gene} {v.protein_change}: VEP API returned {resp.status_code}")
            time.sleep(0.5)  # Rate limit
        except Exception as e:
            log.error(f"  {v.gene} {v.protein_change}: AlphaMissense error: {e}")


# ── CADD lookup ─────────────────────────────────────────────────────────────

def query_cadd(variants: list[Variant]) -> None:
    """Look up CADD v1.7 Phred scores via the CADD web API.

    CADD integrates diverse annotations into a single deleteriousness score.
    Phred-scaled: 10 = top 10%, 20 = top 1%, 30 = top 0.1%.
    Threshold: >15-20 considered deleterious, >25-30 highly deleterious.
    """
    log.info("Querying CADD v1.7 scores...")

    for v in variants:
        if v.cadd_phred is not None:
            log.info(f"  {v.gene} {v.protein_change}: CADD already from VEP = {v.cadd_phred:.1f}")
            continue
        try:
            # CADD API v1.0 with SNV lookup format
            url = (
                f"https://cadd.gs.washington.edu/api/v1.0/"
                f"GRCh38-v1.7/{v.chromosome}:{v.position}"
            )
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                # Response is TSV, parse it
                lines = resp.text.strip().split("\n")
                for line in lines:
                    if line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 6:
                        chrom, pos, ref_allele, alt_allele = parts[0], parts[1], parts[2], parts[3]
                        if alt_allele == v.alt:
                            v.cadd_raw = float(parts[4]) if parts[4] != "NA" else None
                            v.cadd_phred = float(parts[5]) if parts[5] != "NA" else None
                            if v.cadd_phred is not None:
                                log.info(f"  {v.gene} {v.protein_change}: CADD Phred={v.cadd_phred:.1f}")
                            break
                if v.cadd_phred is None:
                    log.warning(f"  {v.gene} {v.protein_change}: No matching CADD score in response")
            else:
                log.warning(f"  {v.gene} {v.protein_change}: CADD API returned {resp.status_code}")
            time.sleep(1.0)  # Rate limit
        except Exception as e:
            log.error(f"  {v.gene} {v.protein_change}: CADD error: {e}")


# ── REVEL lookup ────────────────────────────────────────────────────────────

def query_revel_via_vep(variants: list[Variant]) -> None:
    """Look up REVEL scores — already collected from VEP region query in AlphaMissense step.

    REVEL: ensemble of 13 tools, specifically for rare missense variants.
    Score range: 0-1. Threshold: >0.5 suggested (sensitive), >0.75 (specific).
    """
    log.info("Checking REVEL scores (collected from VEP)...")

    for v in variants:
        if v.revel is not None:
            log.info(f"  {v.gene} {v.protein_change}: REVEL={v.revel:.4f} (from VEP)")
            continue
        # Try myvariant.info API (uses GRCh37 by default)
        try:
            grch37 = GRCH37_COORDS.get(v.gene, {})
            if not grch37:
                log.warning(f"  {v.gene}: No GRCh37 coordinates for myvariant.info")
                continue
            variant_id = f"chr{grch37['chrom']}:g.{grch37['pos']}{grch37['ref']}>{grch37['alt']}"
            url = f"https://myvariant.info/v1/variant/{variant_id}?fields=dbnsfp.revel.score,dbnsfp.cadd.phred,cadd.phred,cadd.rawscore"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                revel_data = data.get("dbnsfp", {}).get("revel", {})
                if isinstance(revel_data, dict):
                    score = revel_data.get("score")
                    if score is not None:
                        if isinstance(score, list):
                            v.revel = float(score[0])
                        elif isinstance(score, (int, float)):
                            v.revel = float(score)
                        else:
                            v.revel = float(str(score).strip("[]").split(",")[0])
                        log.info(f"  {v.gene} {v.protein_change}: REVEL={v.revel:.4f} (myvariant.info)")
                elif isinstance(revel_data, list) and revel_data:
                    score = revel_data[0].get("score") if isinstance(revel_data[0], dict) else revel_data[0]
                    if score is not None:
                        if isinstance(score, list):
                            v.revel = float(score[0])
                        elif isinstance(score, (int, float)):
                            v.revel = float(score)
                        else:
                            v.revel = float(str(score).strip("[]").split(",")[0])
                        log.info(f"  {v.gene} {v.protein_change}: REVEL={v.revel:.4f} (myvariant.info)")
                # Also get CADD from myvariant.info if missing
                if v.cadd_phred is None:
                    # Try direct cadd field first (more complete)
                    cadd_data = data.get("cadd", {})
                    if isinstance(cadd_data, dict) and "phred" in cadd_data:
                        v.cadd_phred = float(cadd_data["phred"])
                        v.cadd_raw = float(cadd_data.get("rawscore", 0))
                        log.info(f"  {v.gene} {v.protein_change}: CADD={v.cadd_phred:.1f} (myvariant.info)")
                    else:
                        # Fallback to dbnsfp.cadd
                        cadd_data = data.get("dbnsfp", {}).get("cadd", {})
                        if isinstance(cadd_data, dict):
                            phred = cadd_data.get("phred")
                            if phred is not None:
                                v.cadd_phred = float(phred)
                                log.info(f"  {v.gene} {v.protein_change}: CADD={v.cadd_phred:.1f} (myvariant.info dbnsfp)")
            if v.revel is None:
                log.warning(f"  {v.gene} {v.protein_change}: No REVEL score found")
            time.sleep(0.3)
        except Exception as e:
            log.error(f"  {v.gene} {v.protein_change}: REVEL fallback error: {e}")


# ── ClinVar lookup ──────────────────────────────────────────────────────────

def query_clinvar(variants: list[Variant]) -> None:
    """Look up ClinVar clinical significance via NCBI E-utilities."""
    log.info("Querying ClinVar...")

    for v in variants:
        try:
            # Search ClinVar by gene + variant
            search_term = f"{v.gene}[gene] AND {v.protein_change}"
            url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=clinvar&term={search_term}&retmode=json"
            )
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                log.warning(f"  {v.gene}: ClinVar search returned {resp.status_code}")
                continue

            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                log.warning(f"  {v.gene} {v.protein_change}: No ClinVar entries found")
                continue

            # Fetch summary for first result
            uid = id_list[0]
            summary_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=clinvar&id={uid}&retmode=json"
            )
            resp2 = requests.get(summary_url, timeout=30)
            if resp2.status_code == 200:
                summary = resp2.json()
                result = summary.get("result", {}).get(uid, {})
                clin_sig = result.get("clinical_significance", {})
                if isinstance(clin_sig, dict):
                    v.clinvar_significance = clin_sig.get("description", "")
                    v.clinvar_review = clin_sig.get("review_status", "")
                else:
                    v.clinvar_significance = str(clin_sig)
                log.info(f"  {v.gene} {v.protein_change}: ClinVar={v.clinvar_significance}")

            time.sleep(0.4)
        except Exception as e:
            log.error(f"  {v.gene} {v.protein_change}: ClinVar error: {e}")


# ── gnomAD lookup ───────────────────────────────────────────────────────────

def query_gnomad(variants: list[Variant]) -> None:
    """Look up population allele frequencies from gnomAD v4."""
    log.info("Querying gnomAD v4...")

    for v in variants:
        try:
            # gnomAD GraphQL API
            query = """
            query($variantId: String!, $dataset: DatasetId!) {
              variant(variantId: $variantId, dataset: $dataset) {
                exome {
                  ac
                  an
                  af
                }
                genome {
                  ac
                  an
                  af
                }
              }
            }
            """
            variant_id = f"{v.chromosome}-{v.position}-{v.ref}-{v.alt}"
            payload = {
                "query": query,
                "variables": {
                    "variantId": variant_id,
                    "dataset": "gnomad_r4",
                },
            }
            resp = requests.post(
                "https://gnomad.broadinstitute.org/api",
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                variant_data = data.get("data", {}).get("variant")
                if variant_data:
                    exome_af = (variant_data.get("exome") or {}).get("af")
                    genome_af = (variant_data.get("genome") or {}).get("af")
                    v.gnomad_af = exome_af or genome_af or 0.0
                    log.info(f"  {v.gene} {v.protein_change}: gnomAD AF={v.gnomad_af:.2e}")
                else:
                    v.gnomad_af = 0.0
                    log.info(f"  {v.gene} {v.protein_change}: Not found in gnomAD (somatic-only)")
            else:
                log.warning(f"  {v.gene}: gnomAD API returned {resp.status_code}")
            time.sleep(0.5)
        except Exception as e:
            log.error(f"  {v.gene} {v.protein_change}: gnomAD error: {e}")


# ── Ensembl VEP comprehensive ──────────────────────────────────────────────

def query_vep_comprehensive(variants: list[Variant]) -> None:
    """Query Ensembl VEP for SIFT, PolyPhen-2, and all available annotations."""
    log.info("Querying Ensembl VEP for SIFT/PolyPhen-2...")

    for v in variants:
        try:
            url = (
                f"https://rest.ensembl.org/vep/human/hgvs/"
                f"{v.transcript}:{v.hgvs_c}"
                f"?content-type=application/json"
                f"&SIFT=b&PolyPhen=b"
                f"&Conservation=1"
            )
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list):
                    for entry in data:
                        for tc in entry.get("transcript_consequences", []):
                            # SIFT
                            sift = tc.get("sift_score")
                            if sift is not None and v.sift_score is None:
                                v.sift_score = float(sift)
                                v.extra["sift_prediction"] = tc.get("sift_prediction", "")
                            # PolyPhen
                            pp = tc.get("polyphen_score")
                            if pp is not None and v.polyphen2_score is None:
                                v.polyphen2_score = float(pp)
                                v.extra["polyphen_prediction"] = tc.get("polyphen_prediction", "")
                            # Conservation
                            if "conservation" in tc:
                                v.extra["conservation"] = tc["conservation"]
                            if v.sift_score is not None and v.polyphen2_score is not None:
                                break
                        if v.sift_score is not None:
                            break
                log.info(
                    f"  {v.gene} {v.protein_change}: "
                    f"SIFT={v.sift_score} ({v.extra.get('sift_prediction', '?')}), "
                    f"PolyPhen={v.polyphen2_score} ({v.extra.get('polyphen_prediction', '?')})"
                )
            time.sleep(0.5)
        except Exception as e:
            log.error(f"  {v.gene} {v.protein_change}: VEP error: {e}")


# ── ACMG evidence aggregation ──────────────────────────────────────────────

def classify_acmg(v: Variant) -> dict[str, Any]:
    """Aggregate computational evidence into ACMG PP3/BP4 classification.

    PP3 (supporting pathogenic): multiple lines of computational evidence support
    BP4 (supporting benign): multiple lines of computational evidence support benign

    Thresholds based on ClinGen SVI recommendations (2019):
        - REVEL >= 0.644 → PP3_Moderate; >= 0.773 → PP3_Strong
        - CADD >= 25.3 → PP3_Supporting
        - AlphaMissense >= 0.564 → likely pathogenic
    """
    evidence = {"pathogenic": [], "benign": [], "uncertain": []}

    # AlphaMissense
    if v.alphamissense is not None:
        if v.alphamissense >= 0.564:
            evidence["pathogenic"].append(f"AlphaMissense={v.alphamissense:.3f} (likely pathogenic)")
        elif v.alphamissense <= 0.34:
            evidence["benign"].append(f"AlphaMissense={v.alphamissense:.3f} (likely benign)")
        else:
            evidence["uncertain"].append(f"AlphaMissense={v.alphamissense:.3f} (ambiguous)")

    # CADD
    if v.cadd_phred is not None:
        if v.cadd_phred >= 25.3:
            evidence["pathogenic"].append(f"CADD Phred={v.cadd_phred:.1f} (top 0.3%)")
        elif v.cadd_phred >= 15:
            evidence["pathogenic"].append(f"CADD Phred={v.cadd_phred:.1f} (top 3%)")
        else:
            evidence["benign"].append(f"CADD Phred={v.cadd_phred:.1f} (below threshold)")

    # REVEL
    if v.revel is not None:
        if v.revel >= 0.773:
            evidence["pathogenic"].append(f"REVEL={v.revel:.4f} (PP3_Strong)")
        elif v.revel >= 0.644:
            evidence["pathogenic"].append(f"REVEL={v.revel:.4f} (PP3_Moderate)")
        elif v.revel >= 0.5:
            evidence["pathogenic"].append(f"REVEL={v.revel:.4f} (PP3_Supporting)")
        elif v.revel <= 0.183:
            evidence["benign"].append(f"REVEL={v.revel:.4f} (BP4_Strong)")
        elif v.revel <= 0.29:
            evidence["benign"].append(f"REVEL={v.revel:.4f} (BP4_Moderate)")
        else:
            evidence["uncertain"].append(f"REVEL={v.revel:.4f} (indeterminate)")

    # SIFT
    if v.sift_score is not None:
        pred = v.extra.get("sift_prediction", "")
        if v.sift_score < 0.05:
            evidence["pathogenic"].append(f"SIFT={v.sift_score:.3f} ({pred})")
        else:
            evidence["benign"].append(f"SIFT={v.sift_score:.3f} ({pred})")

    # PolyPhen-2
    if v.polyphen2_score is not None:
        pred = v.extra.get("polyphen_prediction", "")
        if v.polyphen2_score >= 0.85:
            evidence["pathogenic"].append(f"PolyPhen-2={v.polyphen2_score:.3f} ({pred})")
        elif v.polyphen2_score >= 0.15:
            evidence["uncertain"].append(f"PolyPhen-2={v.polyphen2_score:.3f} ({pred})")
        else:
            evidence["benign"].append(f"PolyPhen-2={v.polyphen2_score:.3f} ({pred})")

    # Overall PP3/BP4 classification
    n_path = len(evidence["pathogenic"])
    n_benign = len(evidence["benign"])

    if n_path >= 3:
        classification = "PP3_Strong"
    elif n_path >= 2:
        classification = "PP3_Moderate"
    elif n_path >= 1 and n_benign == 0:
        classification = "PP3_Supporting"
    elif n_benign >= 3:
        classification = "BP4_Strong"
    elif n_benign >= 2:
        classification = "BP4_Moderate"
    else:
        classification = "Uncertain"

    return {
        "classification": classification,
        "evidence": evidence,
        "n_pathogenic": n_path,
        "n_benign": n_benign,
    }


# ── Report generation ──────────────────────────────────────────────────────

def generate_report(variants: list[Variant]) -> str:
    """Generate comprehensive pathogenicity scoring report."""
    lines = [
        "# Multi-Source Pathogenicity Scoring Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M')}",
        "**Patient:** MDS-IB2/MDS-AML, 5 somatic variants",
        "**Genome build:** GRCh38",
        "",
        "## Summary Table",
        "",
        "| Gene | Variant | VAF | AlphaMissense | CADD | REVEL | SIFT | PolyPhen-2 | ClinVar | gnomAD AF | ACMG PP3 |",
        "|------|---------|-----|---------------|------|-------|------|------------|---------|-----------|----------|",
    ]

    acmg_results = []
    for v in variants:
        acmg = classify_acmg(v)
        acmg_results.append((v, acmg))

        am = f"{v.alphamissense:.3f}" if v.alphamissense is not None else "N/A"
        cadd = f"{v.cadd_phred:.1f}" if v.cadd_phred is not None else "N/A"
        revel = f"{v.revel:.4f}" if v.revel is not None else "N/A"
        sift = f"{v.sift_score:.3f}" if v.sift_score is not None else "N/A"
        pp2 = f"{v.polyphen2_score:.3f}" if v.polyphen2_score is not None else "N/A"
        cv = v.clinvar_significance or "N/A"
        gnomad = f"{v.gnomad_af:.2e}" if v.gnomad_af is not None else "N/A"

        lines.append(
            f"| **{v.gene}** | {v.protein_change} | {v.vaf:.0%} | {am} | {cadd} | "
            f"{revel} | {sift} | {pp2} | {cv} | {gnomad} | {acmg['classification']} |"
        )

    lines.extend(["", "## Detailed Results", ""])

    for v, acmg in acmg_results:
        lines.extend([
            f"### {v.gene} {v.protein_change} (VAF {v.vaf:.0%})",
            "",
            f"- **Transcript:** {v.transcript}",
            f"- **HGVS coding:** {v.hgvs_c}",
            f"- **Genomic:** chr{v.chromosome}:{v.position} {v.ref}>{v.alt}",
            "",
        ])

        if acmg["evidence"]["pathogenic"]:
            lines.append("**Pathogenic evidence:**")
            for e in acmg["evidence"]["pathogenic"]:
                lines.append(f"- {e}")
            lines.append("")

        if acmg["evidence"]["benign"]:
            lines.append("**Benign evidence:**")
            for e in acmg["evidence"]["benign"]:
                lines.append(f"- {e}")
            lines.append("")

        if acmg["evidence"]["uncertain"]:
            lines.append("**Uncertain:**")
            for e in acmg["evidence"]["uncertain"]:
                lines.append(f"- {e}")
            lines.append("")

        lines.append(f"**ACMG PP3/BP4 classification: {acmg['classification']}**")
        lines.append(f"({acmg['n_pathogenic']} pathogenic, {acmg['n_benign']} benign predictors)")
        lines.extend(["", "---", ""])

    # Cross-reference with ESM-2 results
    lines.extend([
        "## Cross-Reference with ESM-2 LLR Scores",
        "",
        "| Gene | Variant | ESM-2 LLR | AlphaMissense | CADD | REVEL | Concordance |",
        "|------|---------|-----------|---------------|------|-------|-------------|",
    ])

    esm2_scores = {
        "EZH2": -3.18,
        "DNMT3A": -8.383,
        "SETBP1": -9.804,
        "PTPN11": -1.76,
        "IDH2": -1.20,
    }

    for v, acmg in acmg_results:
        esm2 = esm2_scores.get(v.gene)
        esm2_str = f"{esm2:.3f}" if esm2 else "N/A"
        am = f"{v.alphamissense:.3f}" if v.alphamissense is not None else "N/A"
        cadd = f"{v.cadd_phred:.1f}" if v.cadd_phred is not None else "N/A"
        revel = f"{v.revel:.4f}" if v.revel is not None else "N/A"

        # Check concordance
        scores_pathogenic = 0
        scores_total = 0
        if esm2 is not None and esm2 < -2.0:
            scores_pathogenic += 1
            scores_total += 1
        elif esm2 is not None:
            scores_total += 1
        if v.alphamissense is not None:
            scores_total += 1
            if v.alphamissense >= 0.564:
                scores_pathogenic += 1
        if v.cadd_phred is not None:
            scores_total += 1
            if v.cadd_phred >= 20:
                scores_pathogenic += 1
        if v.revel is not None:
            scores_total += 1
            if v.revel >= 0.5:
                scores_pathogenic += 1

        concordance = f"{scores_pathogenic}/{scores_total} pathogenic" if scores_total > 0 else "N/A"
        lines.append(f"| {v.gene} | {v.protein_change} | {esm2_str} | {am} | {cadd} | {revel} | {concordance} |")

    lines.extend([
        "",
        "## Methodology",
        "",
        "### Data Sources",
        "- **AlphaMissense** (Cheng et al., Science 2023): Deep learning pathogenicity predictor, AUROC 0.94",
        "- **CADD v1.7** (Rentzsch et al., NAR 2021): Combined annotation dependent depletion, Phred-scaled",
        "- **REVEL** (Ioannidis et al., AJHG 2016): Ensemble of 13 tools for rare missense variants",
        "- **ClinVar** (NCBI): Curated clinical significance from submitting labs",
        "- **gnomAD v4** (Karczewski et al., Nature 2020): Population allele frequencies",
        "- **SIFT** (Ng & Henikoff, Genome Res 2003): Sequence-based tolerance prediction",
        "- **PolyPhen-2** (Adzhubei et al., Nat Methods 2010): Structure/sequence-based prediction",
        "- **ESM-2** (Lin et al., Science 2023): Protein language model log-likelihood ratios",
        "",
        "### ACMG PP3/BP4 Classification",
        "Aggregated per ClinGen SVI recommendations (Pejaver et al., AJHG 2022):",
        "- PP3_Strong: >= 3 pathogenic predictors concordant",
        "- PP3_Moderate: >= 2 pathogenic predictors concordant",
        "- PP3_Supporting: >= 1 pathogenic predictor, none benign",
        "- BP4 (Strong/Moderate): majority benign predictors",
        "",
        "### Thresholds Applied",
        "| Tool | Pathogenic | Benign |",
        "|------|-----------|--------|",
        "| AlphaMissense | >= 0.564 | <= 0.34 |",
        "| CADD Phred | >= 25.3 (strong), >= 15 (supporting) | < 15 |",
        "| REVEL | >= 0.773 (strong), >= 0.644 (moderate), >= 0.5 (supporting) | <= 0.183 (strong), <= 0.29 (moderate) |",
        "| SIFT | < 0.05 (deleterious) | >= 0.05 (tolerated) |",
        "| PolyPhen-2 | >= 0.85 (probably damaging) | < 0.15 (benign) |",
        "| ESM-2 LLR | < -7.5 (PP3_Strong), < -2.0 (PP3_Supporting) | > 0 |",
        "",
    ])

    return "\n".join(lines)


# ── JSON export ─────────────────────────────────────────────────────────────

def export_json(variants: list[Variant]) -> dict:
    """Export all scores as structured JSON."""
    results = []
    for v in variants:
        acmg = classify_acmg(v)
        results.append({
            "gene": v.gene,
            "protein_change": v.protein_change,
            "hgvs_c": v.hgvs_c,
            "vaf": v.vaf,
            "coordinates": {
                "chromosome": v.chromosome,
                "position": v.position,
                "ref": v.ref,
                "alt": v.alt,
                "build": "GRCh38",
            },
            "transcript": v.transcript,
            "scores": {
                "alphamissense": v.alphamissense,
                "alphamissense_class": v.alphamissense_class,
                "cadd_phred": v.cadd_phred,
                "cadd_raw": v.cadd_raw,
                "revel": v.revel,
                "sift": v.sift_score,
                "sift_prediction": v.extra.get("sift_prediction"),
                "polyphen2": v.polyphen2_score,
                "polyphen2_prediction": v.extra.get("polyphen_prediction"),
            },
            "clinvar": {
                "significance": v.clinvar_significance,
                "review_status": v.clinvar_review,
            },
            "gnomad_af": v.gnomad_af,
            "acmg_pp3": acmg["classification"],
            "acmg_evidence": acmg["evidence"],
        })
    return {"variants": results, "generated": time.strftime("%Y-%m-%d %H:%M")}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Multi-Source Pathogenicity Scoring")
    log.info(f"Variants: {len(PATIENT_VARIANTS)}")
    log.info("=" * 60)

    variants = PATIENT_VARIANTS

    # Query all sources
    query_vep_comprehensive(variants)    # SIFT + PolyPhen-2
    query_alphamissense(variants)         # AlphaMissense via VEP
    query_cadd(variants)                  # CADD v1.7
    query_revel_via_vep(variants)         # REVEL via VEP
    query_clinvar(variants)               # ClinVar
    query_gnomad(variants)                # gnomAD v4

    # Generate report
    report = generate_report(variants)
    report_path = RESULTS_DIR / "pathogenicity_report.md"
    report_path.write_text(report)
    log.info(f"Report saved to {report_path}")

    # Export JSON
    json_data = export_json(variants)
    json_path = RESULTS_DIR / "pathogenicity_scores.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    log.info(f"JSON saved to {json_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PATHOGENICITY SCORING SUMMARY")
    print("=" * 80)
    print(f"{'Gene':<10} {'Variant':<10} {'AM':<8} {'CADD':<8} {'REVEL':<8} {'SIFT':<8} {'PP2':<8} {'ACMG'}")
    print("-" * 80)
    for v in variants:
        acmg = classify_acmg(v)
        am = f"{v.alphamissense:.3f}" if v.alphamissense is not None else "N/A"
        cadd = f"{v.cadd_phred:.1f}" if v.cadd_phred is not None else "N/A"
        revel = f"{v.revel:.4f}" if v.revel is not None else "N/A"
        sift = f"{v.sift_score:.3f}" if v.sift_score is not None else "N/A"
        pp2 = f"{v.polyphen2_score:.3f}" if v.polyphen2_score is not None else "N/A"
        print(f"{v.gene:<10} {v.protein_change:<10} {am:<8} {cadd:<8} {revel:<8} {sift:<8} {pp2:<8} {acmg['classification']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
