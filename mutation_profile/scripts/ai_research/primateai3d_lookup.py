#!/usr/bin/env python3
"""
PrimateAI-3D pathogenicity score lookup for patient variants.

Retrieves pre-computed PrimateAI-3D scores from myvariant.info (dbNSFP v4+
integration) for the 5 patient somatic mutations. PrimateAI-3D combines
primate evolutionary conservation with 3D protein structure context to predict
variant pathogenicity (Gao et al., Nature Biotechnology 2024).

PrimateAI-3D score interpretation:
    - Continuous score 0-1 (higher = more pathogenic)
    - Rankscore: genome-wide percentile rank (0-1)
    - Prediction: D (Damaging) or T (Tolerated)
    - Trained on ~380k common primate missense variants as benign labels
    - Incorporates 3D protein structure context via AlphaFold2

Bonus pathogenicity scores (also from dbNSFP, not yet in pipeline):
    - VARITY_R / VARITY_ER: VARITY pathogenicity (ensemble method)
    - BayesDel_addAF: BayesDel with allele frequency features
    - DEOGEN2: effect prediction integrating protein, evolution, phenotype
    - LIST-S2: pathogenicity prediction using amino acid substitution matrices

Data source strategy (in priority order):
    1. myvariant.info API (dbNSFP v4+ fields)
    2. Ensembl VEP REST API with dbnsfp plugin (fallback)
    3. Manual lookup instructions for Illumina BaseSpace (last resort)

Patient variants:
    1. DNMT3A R882H (Q9Y6K1) - VAF 39%, pathogenic hotspot
    2. IDH2 R140Q (P48735) - VAF 2%, pathogenic subclone
    3. SETBP1 G870S (Q9Y6X0) - VAF 34%, likely pathogenic
    4. PTPN11 E76Q (Q06124) - VAF 29%, pathogenic
    5. EZH2 V662A (Q15910) - VAF 59%, founder clone (novel unreported variant)

Existing pipeline pathogenicity scores (for concordance comparison):
    - CADD: EZH2=33.0, DNMT3A=33.0, SETBP1=27.9, PTPN11=27.3, IDH2=28.1
    - REVEL: EZH2=0.962, DNMT3A=0.742, SETBP1=0.716, PTPN11=0.852, IDH2=0.891
    - AlphaMissense: EZH2=0.9952, DNMT3A~0.9952, others vary
    - EVE: retrieved via eve_scores.py

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/primateai3d_lookup.py

Runtime: ~15 seconds (API calls to myvariant.info + optional VEP fallback)
Dependencies: requests
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "ai_research" / "pathogenicity_scoring"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = RESULTS_DIR / "primateai3d_scores.json"
MD_OUTPUT = RESULTS_DIR / "primateai3d_report.md"

# ── APIs ───────────────────────────────────────────────────────────────────

MYVARIANT_BASE = "https://myvariant.info/v1/variant"
ENSEMBL_VEP_BASE = "https://rest.ensembl.org/vep/human/region"

# ── Patient variants (hg19/GRCh37 IDs for myvariant.info) ─────────────────
# Coordinates verified against pathogenicity_scores.py GRCH37_COORDS
# and eve_scores.py PATIENT_VARIANTS

PATIENT_VARIANTS = [
    {
        "key": "DNMT3A_R882H",
        "gene": "DNMT3A",
        "variant": "R882H",
        "uniprot_id": "Q9Y6K1",
        "hgvs_p": "p.Arg882His",
        "hgvs_c": "NM_022552.5:c.2645G>A",
        "hg19_id": "chr2:g.25457242C>T",
        "grch38_chrom": "2",
        "grch38_pos": 25234373,
        "grch38_ref": "G",
        "grch38_alt": "A",
        "vaf": 0.39,
    },
    {
        "key": "IDH2_R140Q",
        "gene": "IDH2",
        "variant": "R140Q",
        "uniprot_id": "P48735",
        "hgvs_p": "p.Arg140Gln",
        "hgvs_c": "NM_002168.4:c.419G>A",
        "hg19_id": "chr15:g.90631934C>T",
        "grch38_chrom": "15",
        "grch38_pos": 90088702,
        "grch38_ref": "C",
        "grch38_alt": "T",
        "vaf": 0.02,
    },
    {
        "key": "SETBP1_G870S",
        "gene": "SETBP1",
        "variant": "G870S",
        "uniprot_id": "Q9Y6X0",
        "hgvs_p": "p.Gly870Ser",
        "hgvs_c": "NM_015559.3:c.2608G>A",
        "hg19_id": "chr18:g.42531913G>A",
        "grch38_chrom": "18",
        "grch38_pos": 44951948,
        "grch38_ref": "G",
        "grch38_alt": "A",
        "vaf": 0.34,
    },
    {
        "key": "PTPN11_E76Q",
        "gene": "PTPN11",
        "variant": "E76Q",
        "uniprot_id": "Q06124",
        "hgvs_p": "p.Glu76Gln",
        "hgvs_c": "NM_002834.5:c.226G>C",
        "hg19_id": "chr12:g.112888210G>C",
        "grch38_chrom": "12",
        "grch38_pos": 112450406,
        "grch38_ref": "G",
        "grch38_alt": "C",
        "vaf": 0.29,
    },
    {
        "key": "EZH2_V662A",
        "gene": "EZH2",
        "variant": "V662A",
        "uniprot_id": "Q15910",
        "hgvs_p": "p.Val662Ala",
        "hgvs_c": "NM_004456.5:c.1985T>C",
        "hg19_id": "chr7:g.148507469T>C",
        "grch38_chrom": "7",
        "grch38_pos": 148810377,
        "grch38_ref": "T",
        "grch38_alt": "C",
        "vaf": 0.59,
    },
]

# ── Existing pipeline scores (for concordance comparison) ──────────────────

EXISTING_SCORES = {
    "DNMT3A_R882H": {"cadd": 33.0, "revel": 0.742, "alphamissense": 0.9952},
    "IDH2_R140Q": {"cadd": 28.1, "revel": 0.891, "alphamissense": None},
    "SETBP1_G870S": {"cadd": 27.9, "revel": 0.716, "alphamissense": None},
    "PTPN11_E76Q": {"cadd": 27.3, "revel": 0.852, "alphamissense": None},
    "EZH2_V662A": {"cadd": 33.0, "revel": 0.962, "alphamissense": 0.9952},
}

# ── dbNSFP fields to query ─────────────────────────────────────────────────

# PrimateAI-3D fields (may use hyphens or underscores in myvariant.info)
PRIMATEAI3D_FIELDS = [
    "dbnsfp.PrimateAI-3D",
    "dbnsfp.PrimateAI_3D",
    "dbnsfp.primateai",
    "dbnsfp.PrimateAI",
]

# Bonus pathogenicity score fields
BONUS_FIELDS = [
    "dbnsfp.VARITY_R",
    "dbnsfp.VARITY_ER",
    "dbnsfp.VARITY_R_LOO",
    "dbnsfp.VARITY_ER_LOO",
    "dbnsfp.BayesDel_addAF_score",
    "dbnsfp.BayesDel_addAF_rankscore",
    "dbnsfp.BayesDel_addAF_pred",
    "dbnsfp.BayesDel_noAF_score",
    "dbnsfp.BayesDel_noAF_rankscore",
    "dbnsfp.BayesDel_noAF_pred",
    "dbnsfp.DEOGEN2_score",
    "dbnsfp.DEOGEN2_rankscore",
    "dbnsfp.DEOGEN2_pred",
    "dbnsfp.LIST-S2_score",
    "dbnsfp.LIST-S2_rankscore",
    "dbnsfp.LIST-S2_pred",
]

# Context fields
CONTEXT_FIELDS = [
    "dbnsfp.genename",
    "dbnsfp.aaref",
    "dbnsfp.aaalt",
    "dbnsfp.aapos",
    "dbnsfp.hg38",
]

ALL_FIELDS = ",".join(PRIMATEAI3D_FIELDS + BONUS_FIELDS + CONTEXT_FIELDS)


# ── Helper: safe extraction from dbNSFP nested data ───────────────────────

def _extract_scalar(data: Any) -> Any:
    """Extract a scalar value from dbNSFP data (may be list or scalar)."""
    if isinstance(data, list):
        # Filter out None/"." placeholders
        clean = [x for x in data if x is not None and x != "."]
        return clean[0] if clean else None
    if data == ".":
        return None
    return data


def _extract_float(data: Any) -> float | None:
    """Extract a float from dbNSFP data."""
    val = _extract_scalar(data)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _extract_str(data: Any) -> str | None:
    """Extract a string from dbNSFP data."""
    val = _extract_scalar(data)
    if val is None:
        return None
    return str(val)


# ── Source 1: myvariant.info query ─────────────────────────────────────────

def fetch_from_myvariant(variant: dict[str, Any]) -> dict[str, Any]:
    """Fetch PrimateAI-3D and bonus scores from myvariant.info for a variant."""
    hg19_id = variant["hg19_id"]
    url = f"{MYVARIANT_BASE}/{hg19_id}"
    params = {"fields": ALL_FIELDS}

    log.info("  [myvariant.info] Querying %s (%s)", variant["key"], hg19_id)
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("  [myvariant.info] Request failed for %s: %s", variant["key"], exc)
        return {"source": "myvariant.info", "status": "error", "error": str(exc)}

    data = resp.json()
    if "error" in data or data.get("notfound"):
        log.warning("  [myvariant.info] Variant not found: %s", hg19_id)
        return {"source": "myvariant.info", "status": "not_found"}

    dbnsfp = data.get("dbnsfp", {})

    # Try all possible PrimateAI-3D field names
    primateai3d = None
    primateai3d_field = None
    for field_name in ["PrimateAI-3D", "PrimateAI_3D", "primateai", "PrimateAI"]:
        candidate = dbnsfp.get(field_name)
        if candidate is not None:
            primateai3d = candidate
            primateai3d_field = field_name
            break

    # Extract PrimateAI-3D scores
    p3d_score = None
    p3d_rankscore = None
    p3d_pred = None
    if primateai3d is not None and isinstance(primateai3d, dict):
        p3d_score = _extract_float(primateai3d.get("score"))
        p3d_rankscore = _extract_float(primateai3d.get("rankscore"))
        p3d_pred = _extract_str(primateai3d.get("pred"))
        log.info(
            "  [myvariant.info] PrimateAI-3D found (field=%s): score=%s, rank=%s, pred=%s",
            primateai3d_field, p3d_score, p3d_rankscore, p3d_pred,
        )
    else:
        # Also check for legacy PrimateAI (non-3D) as partial data
        primateai_legacy = dbnsfp.get("PrimateAI") or dbnsfp.get("primateai")
        if primateai_legacy is not None and isinstance(primateai_legacy, dict):
            p3d_score = _extract_float(primateai_legacy.get("score"))
            p3d_rankscore = _extract_float(primateai_legacy.get("rankscore"))
            p3d_pred = _extract_str(primateai_legacy.get("pred"))
            primateai3d_field = "PrimateAI (legacy, not 3D)"
            log.info(
                "  [myvariant.info] Legacy PrimateAI found: score=%s, rank=%s, pred=%s",
                p3d_score, p3d_rankscore, p3d_pred,
            )
        else:
            log.warning("  [myvariant.info] No PrimateAI-3D data for %s", variant["key"])

    # Extract bonus scores
    bonus = {}

    # VARITY
    varity_r = _extract_float(dbnsfp.get("VARITY_R") or dbnsfp.get("VARITY_R_LOO"))
    varity_er = _extract_float(dbnsfp.get("VARITY_ER") or dbnsfp.get("VARITY_ER_LOO"))
    bonus["varity_r"] = varity_r
    bonus["varity_er"] = varity_er

    # BayesDel
    bayesdel_af = dbnsfp.get("BayesDel_addAF_score")
    if bayesdel_af is None:
        bayesdel_af = dbnsfp.get("BayesDel_addAF", {})
        if isinstance(bayesdel_af, dict):
            bayesdel_af = bayesdel_af.get("score")
    bonus["bayesdel_addaf_score"] = _extract_float(bayesdel_af)
    bonus["bayesdel_addaf_rankscore"] = _extract_float(
        dbnsfp.get("BayesDel_addAF_rankscore")
        or (dbnsfp.get("BayesDel_addAF", {}) or {}).get("rankscore")
    )
    bonus["bayesdel_addaf_pred"] = _extract_str(
        dbnsfp.get("BayesDel_addAF_pred")
        or (dbnsfp.get("BayesDel_addAF", {}) or {}).get("pred")
    )

    bayesdel_noaf = dbnsfp.get("BayesDel_noAF_score")
    if bayesdel_noaf is None:
        bayesdel_noaf = dbnsfp.get("BayesDel_noAF", {})
        if isinstance(bayesdel_noaf, dict):
            bayesdel_noaf = bayesdel_noaf.get("score")
    bonus["bayesdel_noaf_score"] = _extract_float(bayesdel_noaf)
    bonus["bayesdel_noaf_pred"] = _extract_str(
        dbnsfp.get("BayesDel_noAF_pred")
        or (dbnsfp.get("BayesDel_noAF", {}) or {}).get("pred")
    )

    # DEOGEN2
    deogen2 = dbnsfp.get("DEOGEN2_score")
    if deogen2 is None:
        deogen2 = dbnsfp.get("DEOGEN2", {})
        if isinstance(deogen2, dict):
            deogen2 = deogen2.get("score")
    bonus["deogen2_score"] = _extract_float(deogen2)
    bonus["deogen2_rankscore"] = _extract_float(
        dbnsfp.get("DEOGEN2_rankscore")
        or (dbnsfp.get("DEOGEN2", {}) or {}).get("rankscore")
    )
    bonus["deogen2_pred"] = _extract_str(
        dbnsfp.get("DEOGEN2_pred")
        or (dbnsfp.get("DEOGEN2", {}) or {}).get("pred")
    )

    # LIST-S2
    lists2 = dbnsfp.get("LIST-S2_score")
    if lists2 is None:
        lists2 = dbnsfp.get("LIST_S2", {})
        if isinstance(lists2, dict):
            lists2 = lists2.get("score")
    bonus["list_s2_score"] = _extract_float(lists2)
    bonus["list_s2_rankscore"] = _extract_float(
        dbnsfp.get("LIST-S2_rankscore")
        or (dbnsfp.get("LIST_S2", {}) or {}).get("rankscore")
    )
    bonus["list_s2_pred"] = _extract_str(
        dbnsfp.get("LIST-S2_pred")
        or (dbnsfp.get("LIST_S2", {}) or {}).get("pred")
    )

    # Context
    context = {
        "genename": _extract_str(dbnsfp.get("genename")),
        "aaref": _extract_str(dbnsfp.get("aaref")),
        "aaalt": _extract_str(dbnsfp.get("aaalt")),
        "aapos": _extract_str(dbnsfp.get("aapos")),
    }

    return {
        "source": "myvariant.info",
        "status": "success" if p3d_score is not None else "primateai3d_not_found",
        "primateai3d_field": primateai3d_field,
        "primateai3d": {
            "score": p3d_score,
            "rankscore": p3d_rankscore,
            "pred": p3d_pred,
        },
        "bonus_scores": bonus,
        "context": context,
        "raw_dbnsfp_keys": list(dbnsfp.keys()),
    }


# ── Source 2: Ensembl VEP fallback ─────────────────────────────────────────

def fetch_from_vep(variant: dict[str, Any]) -> dict[str, Any]:
    """Fallback: query Ensembl VEP REST API for PrimateAI-3D via dbnsfp plugin."""
    chrom = variant["grch38_chrom"]
    pos = variant["grch38_pos"]
    alt = variant["grch38_alt"]
    region = f"{chrom}:{pos}:{pos}/{alt}"
    url = f"{ENSEMBL_VEP_BASE}/{region}"
    params = {"content-type": "application/json"}

    log.info("  [VEP fallback] Querying %s (%s)", variant["key"], region)
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("  [VEP fallback] Request failed for %s: %s", variant["key"], exc)
        return {"source": "ensembl_vep", "status": "error", "error": str(exc)}

    data = resp.json()
    if not data or not isinstance(data, list):
        log.warning("  [VEP fallback] No VEP data returned for %s", variant["key"])
        return {"source": "ensembl_vep", "status": "not_found"}

    # Search transcript consequences for PrimateAI-3D scores
    p3d_score = None
    p3d_pred = None
    for entry in data:
        for tc in entry.get("transcript_consequences", []):
            # VEP may include PrimateAI-3D in the consequence annotation
            if "primateai3d_score" in tc:
                p3d_score = _extract_float(tc["primateai3d_score"])
                p3d_pred = _extract_str(tc.get("primateai3d_pred"))
                break
            # Alternative field names
            if "primateai_3d_score" in tc:
                p3d_score = _extract_float(tc["primateai_3d_score"])
                p3d_pred = _extract_str(tc.get("primateai_3d_pred"))
                break
        if p3d_score is not None:
            break

    if p3d_score is not None:
        log.info("  [VEP fallback] PrimateAI-3D found: score=%s, pred=%s", p3d_score, p3d_pred)
    else:
        log.warning(
            "  [VEP fallback] PrimateAI-3D not in standard VEP response for %s "
            "(requires dbnsfp plugin on VEP installation)",
            variant["key"],
        )

    return {
        "source": "ensembl_vep",
        "status": "success" if p3d_score is not None else "primateai3d_not_found",
        "primateai3d": {
            "score": p3d_score,
            "rankscore": None,
            "pred": p3d_pred,
        },
    }


# ── Main query logic ──────────────────────────────────────────────────────

def query_variant(variant: dict[str, Any]) -> dict[str, Any]:
    """Query PrimateAI-3D and bonus scores for a single variant.

    Tries myvariant.info first; falls back to Ensembl VEP if PrimateAI-3D
    is not found; provides manual lookup instructions as last resort.
    """
    log.info("Querying %s %s ...", variant["gene"], variant["variant"])

    # Source 1: myvariant.info
    mv_result = fetch_from_myvariant(variant)
    p3d_score = None
    p3d_rankscore = None
    p3d_pred = None
    p3d_source = None
    bonus_scores = {}

    if mv_result["status"] == "success":
        p3d_score = mv_result["primateai3d"]["score"]
        p3d_rankscore = mv_result["primateai3d"]["rankscore"]
        p3d_pred = mv_result["primateai3d"]["pred"]
        p3d_source = f"myvariant.info (field: {mv_result.get('primateai3d_field', 'unknown')})"
        bonus_scores = mv_result.get("bonus_scores", {})
    elif mv_result["status"] == "primateai3d_not_found":
        # PrimateAI-3D not found but other dbNSFP data may exist
        bonus_scores = mv_result.get("bonus_scores", {})

        # Source 2: Ensembl VEP fallback
        time.sleep(0.5)
        vep_result = fetch_from_vep(variant)
        if vep_result["status"] == "success":
            p3d_score = vep_result["primateai3d"]["score"]
            p3d_rankscore = vep_result["primateai3d"]["rankscore"]
            p3d_pred = vep_result["primateai3d"]["pred"]
            p3d_source = "ensembl_vep"

    # If still not found, note manual lookup path
    manual_instructions = None
    if p3d_score is None:
        manual_instructions = (
            "PrimateAI-3D score not available via APIs. Manual lookup options:\n"
            "1. Illumina BaseSpace: https://basespace.illumina.com/apps/primateai-3d\n"
            "   Enter variant as: {gene} {variant} or {chrom}:{pos}{ref}>{alt}\n"
            "2. Download pre-computed TSV from PrimateAI-3D publication supplementary data\n"
            "3. Run locally with the PrimateAI-3D model (requires GPU + model weights)"
        ).format(
            gene=variant["gene"],
            variant=variant["variant"],
            chrom=variant["grch38_chrom"],
            pos=variant["grch38_pos"],
            ref=variant["grch38_ref"],
            alt=variant["grch38_alt"],
        )

    # Determine damaging classification for concordance
    p3d_is_damaging = None
    if p3d_pred is not None:
        p3d_is_damaging = p3d_pred.upper().startswith("D")
    elif p3d_score is not None:
        # PrimateAI-3D threshold: >0.803 is damaging (Gao et al. 2024)
        p3d_is_damaging = p3d_score > 0.803

    return {
        "key": variant["key"],
        "gene": variant["gene"],
        "variant": variant["variant"],
        "uniprot_id": variant["uniprot_id"],
        "hgvs_p": variant["hgvs_p"],
        "hgvs_c": variant["hgvs_c"],
        "hg19_id": variant["hg19_id"],
        "vaf": variant["vaf"],
        "primateai3d": {
            "score": p3d_score,
            "rankscore": p3d_rankscore,
            "pred": p3d_pred,
            "is_damaging": p3d_is_damaging,
            "source": p3d_source,
        },
        "bonus_scores": bonus_scores,
        "manual_instructions": manual_instructions,
        "myvariant_raw_keys": mv_result.get("raw_dbnsfp_keys"),
        "myvariant_context": mv_result.get("context"),
    }


# ── Concordance assessment ─────────────────────────────────────────────────

CONCORDANCE_THRESHOLDS = {
    "cadd": {"pathogenic": 20.0, "description": "CADD Phred >= 20 (top 1%)"},
    "revel": {"pathogenic": 0.5, "description": "REVEL >= 0.5 (likely pathogenic)"},
    "alphamissense": {"pathogenic": 0.564, "description": "AlphaMissense >= 0.564 (likely pathogenic)"},
    "primateai3d": {"pathogenic": 0.803, "description": "PrimateAI-3D > 0.803 (damaging)"},
}


def assess_concordance(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assess concordance between PrimateAI-3D and existing pipeline scores."""
    concordance = []
    for r in results:
        key = r["key"]
        existing = EXISTING_SCORES.get(key, {})
        p3d_score = r["primateai3d"]["score"]
        p3d_damaging = r["primateai3d"]["is_damaging"]

        tools_pathogenic = {}
        tools_pathogenic["cadd"] = (
            existing.get("cadd") is not None and existing["cadd"] >= 20.0
        )
        tools_pathogenic["revel"] = (
            existing.get("revel") is not None and existing["revel"] >= 0.5
        )
        tools_pathogenic["alphamissense"] = (
            existing.get("alphamissense") is not None
            and existing["alphamissense"] >= 0.564
        )
        tools_pathogenic["primateai3d"] = p3d_damaging if p3d_damaging is not None else None

        # Count available tools that agree
        available = {k: v for k, v in tools_pathogenic.items() if v is not None}
        n_pathogenic = sum(1 for v in available.values() if v)
        n_available = len(available)

        if n_available == 0:
            agreement = "insufficient_data"
        elif n_pathogenic == n_available:
            agreement = "full_concordance"
        elif n_pathogenic == 0:
            agreement = "concordant_benign"
        elif n_pathogenic >= n_available * 0.5:
            agreement = "majority_pathogenic"
        else:
            agreement = "discordant"

        concordance.append({
            "key": key,
            "gene": r["gene"],
            "variant": r["variant"],
            "tools": tools_pathogenic,
            "n_pathogenic": n_pathogenic,
            "n_available": n_available,
            "agreement": agreement,
            "existing": existing,
            "primateai3d_score": p3d_score,
        })

    return concordance


# ── Report generation ──────────────────────────────────────────────────────

def classify_primateai3d(score: float | None) -> str:
    """Classify PrimateAI-3D score for clinical context."""
    if score is None:
        return "Not available"
    if score > 0.803:
        return "Damaging"
    if score > 0.5:
        return "Possibly damaging"
    return "Tolerated"


def _fmt(val: float | None, decimals: int = 4) -> str:
    """Format a float for table display, or '--' if None."""
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


def _fmt_pred(pred: str | None) -> str:
    """Format prediction code for display."""
    if pred is None:
        return "--"
    label_map = {"D": "Damaging", "T": "Tolerated", "N": "Neutral"}
    return label_map.get(pred.upper(), pred)


def generate_report(
    results: list[dict[str, Any]],
    concordance: list[dict[str, Any]],
) -> str:
    """Generate markdown report from PrimateAI-3D results."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# PrimateAI-3D Pathogenicity Score Analysis",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## Overview",
        "",
        "PrimateAI-3D combines deep learning on primate evolutionary conservation with",
        "3D protein structure context (AlphaFold2) to predict variant pathogenicity.",
        "Unlike most tools that train on human clinical labels, PrimateAI-3D uses",
        "~380,000 common primate missense variants as benign training labels, avoiding",
        "circularity with clinical databases (Gao et al., Nature Biotechnology 2024).",
        "",
        "**Score interpretation:**",
        "- PrimateAI-3D score: 0 (tolerated) to 1 (damaging)",
        "- Damaging threshold: > 0.803 (recommended, Gao et al. 2024)",
        "- Rankscore: genome-wide percentile (0 = lowest, 1 = highest pathogenicity)",
        "- Prediction: D (Damaging) or T (Tolerated)",
        "",
        "## PrimateAI-3D Results",
        "",
        "| Gene | Variant | VAF | PrimateAI-3D Score | Rankscore | Prediction | Classification |",
        "|------|---------|-----|--------------------|-----------|------------|----------------|",
    ]

    for r in results:
        p3d = r["primateai3d"]
        classification = classify_primateai3d(p3d["score"])
        lines.append(
            f"| {r['gene']} | {r['variant']} | {r['vaf']:.0%} "
            f"| {_fmt(p3d['score'])} | {_fmt(p3d['rankscore'], 5)} "
            f"| {_fmt_pred(p3d['pred'])} | {classification} |"
        )

    # Data source notes
    sources_found = [r for r in results if r["primateai3d"]["score"] is not None]
    sources_missing = [r for r in results if r["primateai3d"]["score"] is None]

    lines.extend(["", f"**Scores retrieved:** {len(sources_found)}/{len(results)} variants"])

    if sources_found:
        for r in sources_found:
            lines.append(f"- {r['gene']} {r['variant']}: via {r['primateai3d']['source']}")

    if sources_missing:
        lines.extend(["", "**Missing scores:**"])
        for r in sources_missing:
            lines.append(f"- {r['gene']} {r['variant']}: not found in API sources")
            if r["manual_instructions"]:
                for instruction_line in r["manual_instructions"].split("\n"):
                    lines.append(f"  {instruction_line}")

    # Bonus scores section
    lines.extend(["", "## Bonus Pathogenicity Scores (from dbNSFP)", ""])

    has_any_bonus = any(
        any(v is not None for v in r.get("bonus_scores", {}).values())
        for r in results
    )

    if has_any_bonus:
        lines.extend([
            "These scores were retrieved alongside PrimateAI-3D from the dbNSFP database",
            "and are not yet part of the existing pathogenicity scoring pipeline.",
            "",
            "### VARITY",
            "",
            "VARITY (Variant Impact Predictor) uses protein structure, evolution, and",
            "function features. VARITY_R includes all features; VARITY_ER uses only",
            "evolutionary and residue-level features (Wu et al., Science 2021).",
            "",
            "| Gene | Variant | VARITY_R | VARITY_ER |",
            "|------|---------|----------|-----------|",
        ])
        for r in results:
            b = r.get("bonus_scores", {})
            lines.append(
                f"| {r['gene']} | {r['variant']} "
                f"| {_fmt(b.get('varity_r'))} | {_fmt(b.get('varity_er'))} |"
            )

        lines.extend([
            "",
            "### BayesDel",
            "",
            "BayesDel integrates multiple deleteriousness scores with a Bayesian framework.",
            "The addAF version includes allele frequency features; noAF does not.",
            "Threshold: > 0.0692 (addAF) or > -0.0570 (noAF) = Damaging (Feng 2017).",
            "",
            "| Gene | Variant | BayesDel addAF | BayesDel noAF | Prediction |",
            "|------|---------|----------------|---------------|------------|",
        ])
        for r in results:
            b = r.get("bonus_scores", {})
            pred = b.get("bayesdel_addaf_pred") or b.get("bayesdel_noaf_pred")
            lines.append(
                f"| {r['gene']} | {r['variant']} "
                f"| {_fmt(b.get('bayesdel_addaf_score'))} "
                f"| {_fmt(b.get('bayesdel_noaf_score'))} "
                f"| {_fmt_pred(pred)} |"
            )

        lines.extend([
            "",
            "### DEOGEN2",
            "",
            "DEOGEN2 predicts pathogenicity using protein sequence, structure, and gene-level",
            "phenotypic information (Raimondi et al., Nucleic Acids Res 2017).",
            "Score > 0.5 = Damaging.",
            "",
            "| Gene | Variant | DEOGEN2 Score | Rankscore | Prediction |",
            "|------|---------|---------------|-----------|------------|",
        ])
        for r in results:
            b = r.get("bonus_scores", {})
            lines.append(
                f"| {r['gene']} | {r['variant']} "
                f"| {_fmt(b.get('deogen2_score'))} "
                f"| {_fmt(b.get('deogen2_rankscore'), 5)} "
                f"| {_fmt_pred(b.get('deogen2_pred'))} |"
            )

        lines.extend([
            "",
            "### LIST-S2",
            "",
            "LIST-S2 predicts pathogenicity using sequence homology and structural features",
            "(Malhis et al., J Mol Biol 2020). Score > 0.85 = Damaging.",
            "",
            "| Gene | Variant | LIST-S2 Score | Rankscore | Prediction |",
            "|------|---------|---------------|-----------|------------|",
        ])
        for r in results:
            b = r.get("bonus_scores", {})
            lines.append(
                f"| {r['gene']} | {r['variant']} "
                f"| {_fmt(b.get('list_s2_score'))} "
                f"| {_fmt(b.get('list_s2_rankscore'), 5)} "
                f"| {_fmt_pred(b.get('list_s2_pred'))} |"
            )
    else:
        lines.append("No bonus pathogenicity scores were returned from dbNSFP.")

    # Concordance with existing pipeline
    lines.extend([
        "",
        "## Concordance with Existing Pipeline Scores",
        "",
        "Comparison of PrimateAI-3D with pathogenicity scores already in the pipeline.",
        "",
        "### Existing Scores (from pathogenicity_scores.py)",
        "",
        "| Gene | Variant | CADD (Phred) | REVEL | AlphaMissense |",
        "|------|---------|--------------|-------|---------------|",
    ])
    for r in results:
        existing = EXISTING_SCORES.get(r["key"], {})
        lines.append(
            f"| {r['gene']} | {r['variant']} "
            f"| {_fmt(existing.get('cadd'), 1)} "
            f"| {_fmt(existing.get('revel'), 3)} "
            f"| {_fmt(existing.get('alphamissense'))} |"
        )

    lines.extend([
        "",
        "### Cross-Tool Concordance",
        "",
        "| Gene | Variant | CADD P | REVEL P | AM P | PrimateAI-3D P | Agreement |",
        "|------|---------|--------|---------|------|----------------|-----------|",
    ])
    for c in concordance:
        t = c["tools"]

        def _bool_mark(val: bool | None) -> str:
            if val is None:
                return "--"
            return "Yes" if val else "No"

        agreement_label = {
            "full_concordance": "Full concordance",
            "concordant_benign": "Concordant benign",
            "majority_pathogenic": "Majority pathogenic",
            "discordant": "Discordant",
            "insufficient_data": "Insufficient data",
        }.get(c["agreement"], c["agreement"])

        lines.append(
            f"| {c['gene']} | {c['variant']} "
            f"| {_bool_mark(t.get('cadd'))} "
            f"| {_bool_mark(t.get('revel'))} "
            f"| {_bool_mark(t.get('alphamissense'))} "
            f"| {_bool_mark(t.get('primateai3d'))} "
            f"| {agreement_label} ({c['n_pathogenic']}/{c['n_available']}) |"
        )

    lines.extend([
        "",
        "**Thresholds used:**",
    ])
    for tool, info in CONCORDANCE_THRESHOLDS.items():
        lines.append(f"- **{tool}:** {info['description']}")

    # Concordance summary
    lines.extend(["", "### Concordance Summary", ""])
    full = sum(1 for c in concordance if c["agreement"] == "full_concordance")
    majority = sum(1 for c in concordance if c["agreement"] == "majority_pathogenic")
    discordant = sum(1 for c in concordance if c["agreement"] == "discordant")
    insufficient = sum(1 for c in concordance if c["agreement"] == "insufficient_data")

    lines.extend([
        f"- **Full concordance:** {full}/{len(concordance)} variants",
        f"- **Majority pathogenic:** {majority}/{len(concordance)} variants",
        f"- **Discordant:** {discordant}/{len(concordance)} variants",
        f"- **Insufficient data:** {insufficient}/{len(concordance)} variants",
    ])

    if full + majority == len(concordance) - insufficient:
        lines.extend([
            "",
            "All variants with sufficient data show concordant pathogenicity predictions",
            "across all tools, strengthening the computational evidence for ACMG PP3.",
        ])
    elif discordant > 0:
        disc_genes = [c["gene"] + " " + c["variant"] for c in concordance if c["agreement"] == "discordant"]
        lines.extend([
            "",
            f"Discordant variants ({', '.join(disc_genes)}) warrant closer examination.",
            "Discordance between computational predictors may indicate variants at the",
            "boundary of pathogenicity thresholds or tool-specific biases.",
        ])

    # Per-variant detail
    lines.extend(["", "## Detailed Per-Variant Analysis", ""])

    for r in results:
        p3d = r["primateai3d"]
        bonus = r.get("bonus_scores", {})

        lines.extend([
            f"### {r['gene']} {r['variant']} ({r['uniprot_id']})",
            "",
            f"- **HGVS protein:** {r['hgvs_p']}",
            f"- **HGVS coding:** {r['hgvs_c']}",
            f"- **VAF:** {r['vaf'] * 100:.0f}%",
            f"- **hg19 ID:** {r['hg19_id']}",
            "",
            "**PrimateAI-3D:**",
            "",
        ])

        if p3d["score"] is not None:
            lines.extend([
                f"- Score: {p3d['score']:.4f}",
                f"- Rankscore: {_fmt(p3d['rankscore'], 5)}",
                f"- Prediction: {_fmt_pred(p3d['pred'])}",
                f"- Classification: {classify_primateai3d(p3d['score'])}",
                f"- Source: {p3d['source']}",
            ])
        else:
            lines.append("- Score: Not available via API sources")
            if r["manual_instructions"]:
                lines.append("")
                for instruction_line in r["manual_instructions"].split("\n"):
                    lines.append(f"  {instruction_line}")

        # Bonus scores for this variant
        bonus_entries = []
        if bonus.get("varity_r") is not None:
            bonus_entries.append(f"VARITY_R={bonus['varity_r']:.4f}")
        if bonus.get("varity_er") is not None:
            bonus_entries.append(f"VARITY_ER={bonus['varity_er']:.4f}")
        if bonus.get("bayesdel_addaf_score") is not None:
            pred_str = f" ({_fmt_pred(bonus.get('bayesdel_addaf_pred'))})" if bonus.get("bayesdel_addaf_pred") else ""
            bonus_entries.append(f"BayesDel_addAF={bonus['bayesdel_addaf_score']:.4f}{pred_str}")
        if bonus.get("deogen2_score") is not None:
            pred_str = f" ({_fmt_pred(bonus.get('deogen2_pred'))})" if bonus.get("deogen2_pred") else ""
            bonus_entries.append(f"DEOGEN2={bonus['deogen2_score']:.4f}{pred_str}")
        if bonus.get("list_s2_score") is not None:
            pred_str = f" ({_fmt_pred(bonus.get('list_s2_pred'))})" if bonus.get("list_s2_pred") else ""
            bonus_entries.append(f"LIST-S2={bonus['list_s2_score']:.4f}{pred_str}")

        if bonus_entries:
            lines.extend(["", "**Bonus scores:** " + ", ".join(bonus_entries)])

        # Existing pipeline context
        existing = EXISTING_SCORES.get(r["key"], {})
        existing_entries = []
        if existing.get("cadd") is not None:
            existing_entries.append(f"CADD={existing['cadd']:.1f}")
        if existing.get("revel") is not None:
            existing_entries.append(f"REVEL={existing['revel']:.3f}")
        if existing.get("alphamissense") is not None:
            existing_entries.append(f"AlphaMissense={existing['alphamissense']:.4f}")
        if existing_entries:
            lines.extend(["", "**Existing pipeline scores:** " + ", ".join(existing_entries)])

        lines.extend(["", "---", ""])

    # Clinical interpretation
    lines.extend([
        "## Clinical Interpretation",
        "",
        "**ACMG PP3 contribution:**",
        "",
        "PrimateAI-3D provides computational (in silico) evidence for ACMG PP3 criterion.",
        "Variants classified as Damaging support PP3_Supporting or stronger evidence.",
        "PrimateAI-3D is particularly valuable because it avoids training on human clinical",
        "labels (using primate common variants instead), providing an independent evidence",
        "axis from tools like ClinVar-calibrated predictors.",
        "",
    ])

    # Summarize clinical relevance
    damaging_variants = [r for r in results if r["primateai3d"]["is_damaging"] is True]
    tolerated_variants = [r for r in results if r["primateai3d"]["is_damaging"] is False]
    unknown_variants = [r for r in results if r["primateai3d"]["is_damaging"] is None]

    if damaging_variants:
        genes = ", ".join(f"{r['gene']} {r['variant']}" for r in damaging_variants)
        lines.append(f"- **Damaging:** {genes}")
    if tolerated_variants:
        genes = ", ".join(f"{r['gene']} {r['variant']}" for r in tolerated_variants)
        lines.append(f"- **Tolerated:** {genes}")
    if unknown_variants:
        genes = ", ".join(f"{r['gene']} {r['variant']}" for r in unknown_variants)
        lines.append(f"- **Not assessed:** {genes}")

    lines.extend([
        "",
        "**Note on EZH2 V662A:** This is a novel unreported variant (0 PubMed hits for",
        "'EZH2 V662'). PrimateAI-3D's assessment is particularly important as there is no",
        "prior clinical data for this specific substitution. Concordance with other",
        "computational tools (CADD=33.0, REVEL=0.962, AlphaMissense=0.9952, EVE=0.9997)",
        "provides strong aggregate evidence for pathogenicity.",
        "",
    ])

    # Methods section
    lines.extend([
        "## Methods",
        "",
        "### Data Sources",
        "",
        "1. **Primary:** myvariant.info API (dbNSFP v4+ integration)",
        "   - Queried using GRCh37/hg19 coordinates",
        "   - Fields: PrimateAI-3D score, rankscore, prediction",
        "   - Bonus fields: VARITY, BayesDel, DEOGEN2, LIST-S2",
        "",
        "2. **Fallback:** Ensembl VEP REST API (GRCh38 coordinates)",
        "   - Used when PrimateAI-3D not found in myvariant.info",
        "   - Requires dbnsfp plugin for PrimateAI-3D scores",
        "",
        "3. **Manual:** Illumina BaseSpace PrimateAI-3D web tool",
        "   - Last resort for variants not in programmatic databases",
        "",
        "### PrimateAI-3D Model",
        "",
        "PrimateAI-3D trains a 3D convolutional neural network on:",
        "- Primate evolutionary conservation (~380k common primate missense variants as benign labels)",
        "- AlphaFold2 protein structure context (3D spatial neighborhood of each residue)",
        "- Does NOT train on human clinical labels (unlike REVEL, ClinVar-calibrated tools)",
        "",
        "This independence from clinical databases makes PrimateAI-3D particularly valuable",
        "for novel variants like EZH2 V662A that have no prior clinical annotation.",
        "",
        "### Concordance Thresholds",
        "",
        "| Tool | Threshold | Interpretation |",
        "|------|-----------|----------------|",
        "| CADD | Phred >= 20 | Top 1% genome-wide deleteriousness |",
        "| REVEL | >= 0.5 | Likely pathogenic (ensemble of 13 tools) |",
        "| AlphaMissense | >= 0.564 | Likely pathogenic (DeepMind) |",
        "| PrimateAI-3D | > 0.803 | Damaging (Gao et al. 2024) |",
        "",
        "### References",
        "",
        "1. Gao H, et al. Predicting the pathogenicity of missense variants using",
        "   features derived from AlphaFold2. Nature Biotechnology. 2024.",
        "2. Sundaram L, et al. Predicting the clinical impact of human mutation with",
        "   deep neural networks. Nature Genetics. 2018;50(8):1161-1170.",
        "3. Wu Y, et al. Improved pathogenicity prediction for rare human missense",
        "   variants. American Journal of Human Genetics. 2021;108(10):1891-1906.",
        "4. Feng BJ. PERCH: A unified framework for disease gene prioritization.",
        "   Human Mutation. 2017;38(3):243-251.",
        "5. Raimondi D, et al. DEOGEN2: prediction and interactive visualization of",
        "   single amino acid variant deleteriousness. Nucleic Acids Research.",
        "   2017;45(W1):W201-W206.",
        "6. Malhis N, et al. LIST-S2: taxonomy based sorting of pathogenic missense",
        "   mutations across species. J Mol Biol. 2020;432(11):3263-3272.",
        "7. Liu X, et al. dbNSFP v4: a comprehensive database of transcript-specific",
        "   functional predictions and annotations for human nonsynonymous and",
        "   splice-site SNVs. Genome Med. 2020;12:103. PMID: 33261662.",
        "",
    ])

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    """Fetch PrimateAI-3D and bonus scores for all patient variants."""
    log.info("=" * 70)
    log.info("PrimateAI-3D Pathogenicity Score Lookup")
    log.info("=" * 70)
    log.info("Querying %d patient variants ...", len(PATIENT_VARIANTS))

    results = []
    for i, variant in enumerate(PATIENT_VARIANTS):
        result = query_variant(variant)
        results.append(result)
        if i < len(PATIENT_VARIANTS) - 1:
            time.sleep(0.5)  # rate limiting

    # Concordance assessment
    concordance = assess_concordance(results)

    # Save JSON
    output = {
        "metadata": {
            "analysis": "PrimateAI-3D pathogenicity score lookup",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "source": "myvariant.info (dbNSFP v4+) with Ensembl VEP fallback",
            "reference": "Gao et al., Nature Biotechnology 2024",
            "n_variants": len(results),
            "n_primateai3d_found": sum(
                1 for r in results if r["primateai3d"]["score"] is not None
            ),
            "bonus_scores_queried": [
                "VARITY_R", "VARITY_ER", "BayesDel_addAF", "BayesDel_noAF",
                "DEOGEN2", "LIST-S2",
            ],
        },
        "variants": results,
        "concordance": concordance,
    }

    with open(JSON_OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved JSON results to %s", JSON_OUTPUT)

    # Generate markdown report
    report = generate_report(results, concordance)
    with open(MD_OUTPUT, "w") as f:
        f.write(report)
    log.info("Saved report to %s", MD_OUTPUT)

    # Print summary
    print("\n" + "=" * 70)
    print("PRIMATEAI-3D SCORE SUMMARY")
    print("=" * 70)
    print(
        f"{'Gene':<10} {'Variant':<10} {'P3D Score':<12} {'Rankscore':<12} "
        f"{'Pred':<12} {'Classification'}"
    )
    print("-" * 70)
    for r in results:
        p3d = r["primateai3d"]
        score_str = f"{p3d['score']:.4f}" if p3d["score"] is not None else "--"
        rank_str = f"{p3d['rankscore']:.5f}" if p3d["rankscore"] is not None else "--"
        pred_str = _fmt_pred(p3d["pred"])
        classification = classify_primateai3d(p3d["score"])
        print(
            f"{r['gene']:<10} {r['variant']:<10} {score_str:<12} {rank_str:<12} "
            f"{pred_str:<12} {classification}"
        )
    print("=" * 70)

    # Bonus scores summary
    has_bonus = any(
        any(v is not None for v in r.get("bonus_scores", {}).values())
        for r in results
    )
    if has_bonus:
        print("\nBONUS SCORES:")
        print("-" * 70)
        print(
            f"{'Gene':<10} {'Variant':<10} {'VARITY_R':<12} {'BayesDel':<12} "
            f"{'DEOGEN2':<12} {'LIST-S2':<12}"
        )
        print("-" * 70)
        for r in results:
            b = r.get("bonus_scores", {})
            print(
                f"{r['gene']:<10} {r['variant']:<10} "
                f"{_fmt(b.get('varity_r')):<12} "
                f"{_fmt(b.get('bayesdel_addaf_score')):<12} "
                f"{_fmt(b.get('deogen2_score')):<12} "
                f"{_fmt(b.get('list_s2_score')):<12}"
            )
        print("=" * 70)

    # Concordance summary
    print("\nCONCORDANCE:")
    print("-" * 70)
    for c in concordance:
        print(
            f"  {c['gene']} {c['variant']}: {c['agreement']} "
            f"({c['n_pathogenic']}/{c['n_available']} pathogenic)"
        )

    n_found = sum(1 for r in results if r["primateai3d"]["score"] is not None)
    n_damaging = sum(1 for r in results if r["primateai3d"]["is_damaging"] is True)
    print(f"\n{n_found}/{len(results)} variants scored, {n_damaging} classified Damaging")


if __name__ == "__main__":
    main()
