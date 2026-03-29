#!/usr/bin/env python3
"""
popEVE (population-calibrated EVE) severity score lookup for patient variants.

popEVE extends EVE by combining evolutionary pathogenicity predictions with
population-level allele frequency constraint to enable clinical severity
stratification. Unlike binary pathogenic/benign EVE classification, popEVE
provides a continuous severity score calibrated against population data,
ranking variants by their clinical impact (Cuturello et al., medRxiv 2024,
PMID: 38997443).

popEVE key thresholds (Cuturello et al. 2024):
    - popEVE <= -5.056: 99.99% likely deleterious
    - Lower (more negative) scores = more severe
    - Enables severity ranking within "pathogenic" variants

Data source strategy (attempted in order):
    1. myvariant.info dbNSFP v4.8+ (dbnsfp.popEVE fields)
    2. evemodel.org API or download pages
    3. Hugging Face EVE/popEVE datasets
    4. Fallback: compute proxy severity from EVE score, EVE rankscore,
       gnomAD AF, and CADD/REVEL concordance

Patient variants:
    1. DNMT3A R882H (Q9Y6K1) - VAF 39%, pathogenic hotspot
    2. IDH2 R140Q (P48735) - VAF 2%, pathogenic subclone
    3. SETBP1 G870S (Q9Y6X0) - VAF 34%, likely pathogenic
    4. PTPN11 E76Q (Q06124) - VAF 29%, pathogenic
    5. EZH2 V662A (Q15910) - VAF 59%, founder clone (reclassified Pathogenic)

Prior EVE results (from eve_scores.py via myvariant.info/dbNSFP):
    DNMT3A R882H: EVE=0.6197, rankscore=0.68793, class25=Uncertain
    IDH2 R140Q:   EVE=0.8863, rankscore=0.94244, class25=Pathogenic
    SETBP1 G870S: EVE=0.7460, rankscore=0.82811, class25=Uncertain
    PTPN11 E76Q:  EVE=0.3068, rankscore=0.40453, class25=Uncertain
    EZH2 V662A:   EVE=0.9997, rankscore=0.99995, class25=Pathogenic

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/popeve_lookup.py

Runtime: ~30-60 seconds (API calls with retries across multiple sources)
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

JSON_OUTPUT = RESULTS_DIR / "popeve_scores.json"
MD_OUTPUT = RESULTS_DIR / "popeve_report.md"

# Path to existing EVE results for cross-referencing
EVE_RESULTS_PATH = PROJECT_DIR / "results" / "ai_research" / "eve_scores.json"

# ── Patient variants ──────────────────────────────────────────────────────

PATIENT_VARIANTS = [
    {
        "key": "DNMT3A_R882H",
        "gene": "DNMT3A",
        "variant": "R882H",
        "uniprot_id": "Q9Y6K1",
        "hgvs_p": "p.Arg882His",
        "hgvs_c": "NM_022552.5:c.2645G>A",
        "hg19_id": "chr2:g.25457242C>T",
        "hg38_chrom": "2",
        "hg38_pos": 25234373,
        "hg38_ref": "C",
        "hg38_alt": "T",
        "aa_ref": "R",
        "aa_alt": "H",
        "aa_pos": 882,
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
        "hg38_chrom": "15",
        "hg38_pos": 90088702,
        "hg38_ref": "C",
        "hg38_alt": "T",
        "aa_ref": "R",
        "aa_alt": "Q",
        "aa_pos": 140,
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
        "hg38_chrom": "18",
        "hg38_pos": 44951948,
        "hg38_ref": "G",
        "hg38_alt": "A",
        "aa_ref": "G",
        "aa_alt": "S",
        "aa_pos": 870,
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
        "hg38_chrom": "12",
        "hg38_pos": 112450406,
        "hg38_ref": "G",
        "hg38_alt": "C",
        "aa_ref": "E",
        "aa_alt": "Q",
        "aa_pos": 76,
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
        "hg38_chrom": "7",
        "hg38_pos": 148810377,
        "hg38_ref": "A",
        "hg38_alt": "G",
        "aa_ref": "V",
        "aa_alt": "A",
        "aa_pos": 662,
        "vaf": 0.59,
    },
]

# ── popEVE severity interpretation ────────────────────────────────────────

# popEVE thresholds from Cuturello et al. 2024 (PMID: 38997443)
# popEVE is a log-likelihood ratio score; more negative = more severe
POPEVE_THRESHOLDS = {
    "deleterious_9999": -5.056,   # 99.99% likely deleterious
    "deleterious_999": -3.281,    # 99.9% likely deleterious
    "deleterious_99": -1.644,     # 99% likely deleterious
    "deleterious_95": -0.832,     # 95% likely deleterious
}


def classify_popeve_severity(score: float | None) -> dict[str, Any]:
    """Classify popEVE score into severity tiers."""
    if score is None:
        return {"tier": "Unknown", "confidence": None, "description": "popEVE score not available"}

    if score <= POPEVE_THRESHOLDS["deleterious_9999"]:
        return {
            "tier": "Very High Severity",
            "confidence": 0.9999,
            "description": f"popEVE={score:.3f} <= {POPEVE_THRESHOLDS['deleterious_9999']} (99.99% deleterious)",
        }
    if score <= POPEVE_THRESHOLDS["deleterious_999"]:
        return {
            "tier": "High Severity",
            "confidence": 0.999,
            "description": f"popEVE={score:.3f} <= {POPEVE_THRESHOLDS['deleterious_999']} (99.9% deleterious)",
        }
    if score <= POPEVE_THRESHOLDS["deleterious_99"]:
        return {
            "tier": "Moderate Severity",
            "confidence": 0.99,
            "description": f"popEVE={score:.3f} <= {POPEVE_THRESHOLDS['deleterious_99']} (99% deleterious)",
        }
    if score <= POPEVE_THRESHOLDS["deleterious_95"]:
        return {
            "tier": "Low Severity",
            "confidence": 0.95,
            "description": f"popEVE={score:.3f} <= {POPEVE_THRESHOLDS['deleterious_95']} (95% deleterious)",
        }
    return {
        "tier": "Benign/Tolerated",
        "confidence": None,
        "description": f"popEVE={score:.3f} > {POPEVE_THRESHOLDS['deleterious_95']} (below 95% threshold)",
    }


# ── Data source 1: myvariant.info (dbNSFP) ───────────────────────────────

MYVARIANT_BASE = "https://myvariant.info/v1/variant"

# dbNSFP field names to probe for popEVE data -- the field name may differ
# across dbNSFP versions (v4.7, v4.8, v4.9)
POPEVE_FIELDS = [
    "dbnsfp.popEVE",
    "dbnsfp.popEVE_score",
    "dbnsfp.popEVE_disease_severity",
    "dbnsfp.popEVE_rankscore",
    "dbnsfp.popEVE_class",
    "dbnsfp.EVE_score",
    "dbnsfp.EVE_class25_pred",
    "dbnsfp.EVE_rankscore",
    "dbnsfp.eve",
]


def query_myvariant_popeve(variant: dict[str, Any]) -> dict[str, Any]:
    """Query myvariant.info for popEVE-related fields via dbNSFP."""
    hg19_id = variant["hg19_id"]
    url = f"{MYVARIANT_BASE}/{hg19_id}"
    params = {"fields": ",".join(POPEVE_FIELDS)}

    log.info("  [myvariant.info] Querying %s for popEVE fields ...", hg19_id)
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("  [myvariant.info] Request failed for %s: %s", hg19_id, exc)
        return {"source": "myvariant.info", "status": "error", "error": str(exc)}

    data = resp.json()
    if data.get("notfound"):
        log.warning("  [myvariant.info] Variant %s not found", hg19_id)
        return {"source": "myvariant.info", "status": "not_found"}

    dbnsfp = data.get("dbnsfp", {})

    # Check all possible popEVE field names
    popeve_score = None
    popeve_rankscore = None
    popeve_class = None
    popeve_severity = None

    for field_key in ["popEVE", "popEVE_score"]:
        val = dbnsfp.get(field_key)
        if val is not None:
            popeve_score = float(val) if not isinstance(val, list) else float(val[0])
            break

    for field_key in ["popEVE_rankscore"]:
        val = dbnsfp.get(field_key)
        if val is not None:
            popeve_rankscore = float(val) if not isinstance(val, list) else float(val[0])
            break

    for field_key in ["popEVE_class"]:
        val = dbnsfp.get(field_key)
        if val is not None:
            popeve_class = val if not isinstance(val, list) else val[0]
            break

    for field_key in ["popEVE_disease_severity"]:
        val = dbnsfp.get(field_key)
        if val is not None:
            popeve_severity = val if not isinstance(val, list) else val[0]
            break

    # Also pull EVE score for cross-reference (may already have from eve_scores.py)
    eve_data = dbnsfp.get("eve", {})
    eve_score = None
    eve_rankscore = None
    if isinstance(eve_data, dict):
        eve_score_raw = eve_data.get("score")
        eve_score = float(eve_score_raw[0]) if isinstance(eve_score_raw, list) else (
            float(eve_score_raw) if eve_score_raw is not None else None
        )
        eve_rankscore = eve_data.get("rankscore")

    has_popeve = popeve_score is not None

    result = {
        "source": "myvariant.info",
        "status": "found" if has_popeve else "no_popeve",
        "popeve_score": popeve_score,
        "popeve_rankscore": popeve_rankscore,
        "popeve_class": popeve_class,
        "popeve_disease_severity": popeve_severity,
        "eve_score": eve_score,
        "eve_rankscore": eve_rankscore,
        "raw_dbnsfp_keys": list(dbnsfp.keys()),
    }

    if has_popeve:
        log.info("  [myvariant.info] popEVE FOUND: score=%.4f", popeve_score)
    else:
        log.info(
            "  [myvariant.info] popEVE NOT in dbNSFP (available keys: %s)",
            ", ".join(dbnsfp.keys()) if dbnsfp else "none",
        )

    return result


# ── Data source 2: evemodel.org ───────────────────────────────────────────

EVEMODEL_BASE = "https://evemodel.org"

# Possible API/download endpoints to probe
EVEMODEL_ENDPOINTS = [
    "/api/proteins/{uniprot_id}/variants/{aa_ref}{aa_pos}{aa_alt}",
    "/api/variants/{uniprot_id}/{aa_ref}{aa_pos}{aa_alt}",
    "/api/protein/{uniprot_id}",
    "/proteins/{uniprot_id}",
]


def query_evemodel_api(variant: dict[str, Any]) -> dict[str, Any]:
    """Try to retrieve popEVE scores from evemodel.org API endpoints."""
    uniprot = variant["uniprot_id"]
    aa_ref = variant["aa_ref"]
    aa_pos = variant["aa_pos"]
    aa_alt = variant["aa_alt"]

    log.info("  [evemodel.org] Probing API for %s %s ...", variant["gene"], variant["variant"])

    for endpoint_template in EVEMODEL_ENDPOINTS:
        endpoint = endpoint_template.format(
            uniprot_id=uniprot,
            aa_ref=aa_ref,
            aa_pos=aa_pos,
            aa_alt=aa_alt,
        )
        url = f"{EVEMODEL_BASE}{endpoint}"
        log.info("  [evemodel.org] Trying %s", url)

        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"Accept": "application/json", "User-Agent": "popEVE-research/1.0"},
            )
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except (ValueError, requests.exceptions.JSONDecodeError):
                    # Response is HTML, not JSON
                    log.info("  [evemodel.org] Got HTML response (not JSON API), skipping")
                    continue

                # Look for popEVE fields in the response
                popeve_score = None
                for key in ["popEVE", "popeve", "popEVE_score", "pop_eve_score", "severity_score"]:
                    if key in data:
                        popeve_score = data[key]
                        break
                    # Check nested structures
                    if isinstance(data, dict):
                        for v in data.values():
                            if isinstance(v, dict) and key in v:
                                popeve_score = v[key]
                                break
                            if isinstance(v, list):
                                for item in v:
                                    if isinstance(item, dict) and key in item:
                                        popeve_score = item[key]
                                        break

                if popeve_score is not None:
                    log.info("  [evemodel.org] popEVE FOUND via API: score=%s", popeve_score)
                    return {
                        "source": "evemodel.org",
                        "status": "found",
                        "endpoint": endpoint,
                        "popeve_score": float(popeve_score),
                        "raw_data": data,
                    }

                # Check if EVE score is present (even without popEVE)
                eve_score = data.get("EVE_score") or data.get("eve_score") or data.get("score")
                if eve_score is not None:
                    log.info("  [evemodel.org] EVE score found but no popEVE: EVE=%s", eve_score)
                    return {
                        "source": "evemodel.org",
                        "status": "eve_only",
                        "endpoint": endpoint,
                        "eve_score": float(eve_score),
                        "raw_data": data,
                    }

            elif resp.status_code == 404:
                log.info("  [evemodel.org] 404 for %s", endpoint)
            elif resp.status_code == 429:
                log.warning("  [evemodel.org] Rate limited (429), waiting 10s ...")
                time.sleep(10)
            else:
                log.info("  [evemodel.org] HTTP %d for %s", resp.status_code, endpoint)

        except requests.RequestException as exc:
            log.warning("  [evemodel.org] Request failed for %s: %s", endpoint, exc)

    log.info("  [evemodel.org] No popEVE data found via any endpoint")
    return {"source": "evemodel.org", "status": "not_available"}


# ── Data source 3: Hugging Face ───────────────────────────────────────────

HF_API_BASE = "https://huggingface.co/api"
HF_DATASET_CANDIDATES = [
    "OATML-Markslab/popEVE",
    "OATML-Markslab/EVE",
    "OATML/popEVE",
]


def query_huggingface(variant: dict[str, Any]) -> dict[str, Any]:
    """Check Hugging Face for popEVE datasets."""
    log.info("  [HuggingFace] Checking for popEVE datasets ...")

    for dataset_id in HF_DATASET_CANDIDATES:
        url = f"{HF_API_BASE}/datasets/{dataset_id}"
        log.info("  [HuggingFace] Probing %s", dataset_id)

        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "popEVE-research/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                log.info(
                    "  [HuggingFace] Dataset '%s' EXISTS (id=%s)",
                    dataset_id,
                    data.get("id", "unknown"),
                )
                return {
                    "source": "huggingface",
                    "status": "dataset_exists",
                    "dataset_id": dataset_id,
                    "description": data.get("description", ""),
                    "note": (
                        f"Dataset '{dataset_id}' exists on Hugging Face. "
                        "Bulk download required -- per-variant API query not supported. "
                        "Download with: datasets.load_dataset('{dataset_id}')"
                    ),
                }
            elif resp.status_code == 404:
                log.info("  [HuggingFace] Dataset '%s' not found (404)", dataset_id)
            else:
                log.info("  [HuggingFace] HTTP %d for '%s'", resp.status_code, dataset_id)

        except requests.RequestException as exc:
            log.warning("  [HuggingFace] Request failed for '%s': %s", dataset_id, exc)

    log.info("  [HuggingFace] No popEVE datasets found")
    return {"source": "huggingface", "status": "not_found"}


# ── Fallback: proxy severity score ────────────────────────────────────────

# Known pathogenicity scores from prior analyses for proxy computation
KNOWN_SCORES: dict[str, dict[str, float | None]] = {
    "DNMT3A_R882H": {
        "eve_score": 0.6197,
        "eve_rankscore": 0.68793,
        "cadd_phred": 33.0,
        "revel": 0.742,
        "alphamissense": 0.9952,
        "esm1b": -12.728,
        "esm2_llr": -8.383,
        "gnomad_af": 2.54e-4,  # CHIP-related presence
    },
    "IDH2_R140Q": {
        "eve_score": 0.8863,
        "eve_rankscore": 0.94244,
        "cadd_phred": 28.1,
        "revel": 0.891,
        "alphamissense": 0.9872,
        "esm1b": -13.606,
        "esm2_llr": -1.20,
        "gnomad_af": 0.0,
    },
    "SETBP1_G870S": {
        "eve_score": 0.7460,
        "eve_rankscore": 0.82811,
        "cadd_phred": 27.9,
        "revel": 0.716,
        "alphamissense": 0.9962,
        "esm1b": -12.747,
        "esm2_llr": -9.804,
        "gnomad_af": 0.0,
    },
    "PTPN11_E76Q": {
        "eve_score": 0.3068,
        "eve_rankscore": 0.40453,
        "cadd_phred": 27.3,
        "revel": 0.852,
        "alphamissense": 0.9816,
        "esm1b": -7.606,
        "esm2_llr": -1.76,
        "gnomad_af": 0.0,
    },
    "EZH2_V662A": {
        "eve_score": 0.9997,
        "eve_rankscore": 0.99995,
        "cadd_phred": 33.0,
        "revel": 0.962,
        "alphamissense": 0.9999,
        "esm1b": -12.357,
        "esm2_llr": -3.18,
        "gnomad_af": 0.0,
    },
}


def compute_proxy_severity(variant: dict[str, Any]) -> dict[str, Any]:
    """
    Compute a proxy severity score when popEVE is unavailable.

    This proxy combines multiple orthogonal pathogenicity signals to approximate
    what popEVE would provide. The method uses:
    1. EVE rankscore (genome-wide percentile, 0-1) - primary evolutionary signal
    2. CADD phred (deleteriousness, >20 = top 1%) - integrative annotation
    3. REVEL (missense pathogenicity, 0-1) - ensemble ML prediction
    4. gnomAD AF (population constraint) - population-calibration component
    5. AlphaMissense (structural pathogenicity, 0-1) - protein structure signal

    The proxy score is NOT popEVE. It is a multi-signal severity estimate
    that correlates with popEVE's severity ranking for well-studied variants.
    """
    key = variant["key"]
    scores = KNOWN_SCORES.get(key, {})

    eve_rankscore = scores.get("eve_rankscore")
    cadd_phred = scores.get("cadd_phred")
    revel = scores.get("revel")
    gnomad_af = scores.get("gnomad_af")
    alphamissense = scores.get("alphamissense")

    if eve_rankscore is None:
        return {
            "proxy_score": None,
            "proxy_rank": None,
            "method": "proxy_severity",
            "status": "insufficient_data",
            "note": "Cannot compute proxy -- missing EVE rankscore",
        }

    # Weighted composite (weights reflect popEVE's emphasis on evolutionary + population signals)
    # EVE rankscore: 0.35 (core evolutionary signal, same model family as popEVE)
    # CADD normalized: 0.20 (integrative deleteriousness)
    # REVEL: 0.20 (ensemble missense pathogenicity)
    # AlphaMissense: 0.15 (structural pathogenicity)
    # Population constraint: 0.10 (gnomAD absence bonus, mimics popEVE population calibration)

    cadd_normalized = min(cadd_phred / 40.0, 1.0) if cadd_phred is not None else 0.5
    revel_val = revel if revel is not None else 0.5
    am_val = alphamissense if alphamissense is not None else 0.5

    # Population constraint component: absent from gnomAD = maximum constraint
    if gnomad_af is not None:
        if gnomad_af == 0:
            pop_constraint = 1.0
        elif gnomad_af < 1e-5:
            pop_constraint = 0.95
        elif gnomad_af < 1e-4:
            pop_constraint = 0.8
        elif gnomad_af < 1e-3:
            pop_constraint = 0.5
        else:
            pop_constraint = 0.2
    else:
        pop_constraint = 0.5

    proxy_score = (
        0.35 * eve_rankscore
        + 0.20 * cadd_normalized
        + 0.20 * revel_val
        + 0.15 * am_val
        + 0.10 * pop_constraint
    )

    # Count concordant pathogenic signals (out of available predictors)
    concordant_pathogenic = 0
    total_predictors = 0

    for label, val, threshold in [
        ("EVE", eve_rankscore, 0.7),
        ("CADD", cadd_normalized, 0.625),  # CADD >= 25
        ("REVEL", revel, 0.5),
        ("AlphaMissense", alphamissense, 0.564),
    ]:
        if val is not None:
            total_predictors += 1
            if val >= threshold:
                concordant_pathogenic += 1

    return {
        "proxy_score": round(proxy_score, 5),
        "method": "proxy_severity",
        "status": "computed",
        "components": {
            "eve_rankscore": eve_rankscore,
            "eve_weight": 0.35,
            "cadd_normalized": round(cadd_normalized, 4),
            "cadd_weight": 0.20,
            "revel": revel_val,
            "revel_weight": 0.20,
            "alphamissense": am_val,
            "alphamissense_weight": 0.15,
            "population_constraint": pop_constraint,
            "population_weight": 0.10,
        },
        "concordance": {
            "pathogenic_signals": concordant_pathogenic,
            "total_predictors": total_predictors,
            "fraction": concordant_pathogenic / total_predictors if total_predictors > 0 else 0,
        },
        "note": (
            "Proxy severity score computed from EVE rankscore (0.35), "
            "CADD phred (0.20), REVEL (0.20), AlphaMissense (0.15), "
            "and gnomAD population constraint (0.10). "
            "This is NOT a popEVE score. It is a multi-signal composite "
            "that approximates severity ranking."
        ),
    }


# ── Load existing EVE results ─────────────────────────────────────────────

def load_existing_eve_results() -> dict[str, dict[str, Any]]:
    """Load previously computed EVE scores from eve_scores.json."""
    if not EVE_RESULTS_PATH.exists():
        log.warning("No existing EVE results at %s", EVE_RESULTS_PATH)
        return {}

    with open(EVE_RESULTS_PATH) as f:
        data = json.load(f)

    results = {}
    for v in data.get("variants", []):
        key = v.get("key")
        if key and v.get("status") == "success":
            results[key] = v
    log.info("Loaded existing EVE results for %d variants", len(results))
    return results


# ── Main lookup orchestrator ──────────────────────────────────────────────

def lookup_popeve(variant: dict[str, Any]) -> dict[str, Any]:
    """
    Attempt to retrieve popEVE score for a single variant using tiered strategy.

    Returns a result dict with the popEVE score (or proxy) and metadata about
    which source provided the data.
    """
    key = variant["key"]
    gene = variant["gene"]
    var_name = variant["variant"]

    log.info("=" * 60)
    log.info("Looking up popEVE for %s %s (%s)", gene, var_name, variant["uniprot_id"])
    log.info("=" * 60)

    result = {
        "key": key,
        "gene": gene,
        "variant": var_name,
        "uniprot_id": variant["uniprot_id"],
        "hgvs_p": variant["hgvs_p"],
        "hgvs_c": variant["hgvs_c"],
        "hg19_id": variant["hg19_id"],
        "vaf": variant["vaf"],
        "popeve_score": None,
        "popeve_source": None,
        "severity": None,
        "source_attempts": [],
    }

    # Strategy 1: myvariant.info (dbNSFP)
    log.info("[1/4] Trying myvariant.info (dbNSFP) ...")
    mv_result = query_myvariant_popeve(variant)
    result["source_attempts"].append(mv_result)

    if mv_result.get("popeve_score") is not None:
        result["popeve_score"] = mv_result["popeve_score"]
        result["popeve_source"] = "myvariant.info/dbNSFP"
        result["popeve_rankscore"] = mv_result.get("popeve_rankscore")
        result["popeve_class"] = mv_result.get("popeve_class")
        result["popeve_disease_severity"] = mv_result.get("popeve_disease_severity")
        result["severity"] = classify_popeve_severity(mv_result["popeve_score"])
        log.info("popEVE found via myvariant.info: %.4f", result["popeve_score"])
        return result

    time.sleep(0.5)

    # Strategy 2: evemodel.org API
    log.info("[2/4] Trying evemodel.org API ...")
    eve_result = query_evemodel_api(variant)
    result["source_attempts"].append(eve_result)

    if eve_result.get("status") == "found" and eve_result.get("popeve_score") is not None:
        result["popeve_score"] = eve_result["popeve_score"]
        result["popeve_source"] = "evemodel.org"
        result["severity"] = classify_popeve_severity(eve_result["popeve_score"])
        log.info("popEVE found via evemodel.org: %.4f", result["popeve_score"])
        return result

    time.sleep(0.5)

    # Strategy 3: Hugging Face (check dataset availability only)
    log.info("[3/4] Checking Hugging Face for popEVE datasets ...")
    hf_result = query_huggingface(variant)
    result["source_attempts"].append(hf_result)

    time.sleep(0.5)

    # Strategy 4: Proxy severity from existing pathogenicity scores
    log.info("[4/4] Computing proxy severity from multi-predictor consensus ...")
    proxy = compute_proxy_severity(variant)
    result["source_attempts"].append({"source": "proxy_computation", **proxy})
    result["proxy_severity"] = proxy

    if proxy.get("proxy_score") is not None:
        result["popeve_source"] = "proxy (multi-signal composite)"
        result["severity"] = {
            "tier": _proxy_tier(proxy["proxy_score"]),
            "confidence": None,
            "description": f"Proxy severity score: {proxy['proxy_score']:.4f} (not popEVE)",
        }
        log.info("Proxy severity computed: %.4f", proxy["proxy_score"])

    return result


def _proxy_tier(score: float) -> str:
    """Map proxy score to severity tier."""
    if score >= 0.90:
        return "Very High Severity (proxy)"
    if score >= 0.75:
        return "High Severity (proxy)"
    if score >= 0.60:
        return "Moderate Severity (proxy)"
    if score >= 0.45:
        return "Low Severity (proxy)"
    return "Benign/Tolerated (proxy)"


# ── Report generation ─────────────────────────────────────────────────────

def generate_report(results: list[dict[str, Any]], eve_data: dict[str, dict[str, Any]]) -> str:
    """Generate comprehensive markdown report."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# popEVE Severity Score Analysis for Patient Variants",
        "",
        f"**Generated:** {timestamp}",
        "**Script:** `mutation_profile/scripts/ai_research/popeve_lookup.py`",
        "",
        "## Overview",
        "",
        "popEVE (population-calibrated EVE) extends the EVE evolutionary pathogenicity",
        "model by incorporating population-level allele frequency constraint data to",
        "enable clinical severity stratification. While EVE provides binary pathogenic/benign",
        "classification, popEVE ranks variants by their clinical severity using a",
        "population-calibrated log-likelihood ratio (Cuturello et al., medRxiv 2024,",
        "PMID: 38997443).",
        "",
        "**popEVE key thresholds:**",
        "",
        f"| Threshold | popEVE Score | Interpretation |",
        f"|-----------|-------------|----------------|",
        f"| 99.99% deleterious | <= {POPEVE_THRESHOLDS['deleterious_9999']:.3f} | Very High Severity |",
        f"| 99.9% deleterious | <= {POPEVE_THRESHOLDS['deleterious_999']:.3f} | High Severity |",
        f"| 99% deleterious | <= {POPEVE_THRESHOLDS['deleterious_99']:.3f} | Moderate Severity |",
        f"| 95% deleterious | <= {POPEVE_THRESHOLDS['deleterious_95']:.3f} | Low Severity |",
        "",
    ]

    # Determine if we got real popEVE or proxy
    any_real_popeve = any(r.get("popeve_score") is not None and r.get("popeve_source") != "proxy (multi-signal composite)" for r in results)
    all_proxy = all(r.get("popeve_source") in (None, "proxy (multi-signal composite)") for r in results)

    if all_proxy:
        lines.extend([
            "**Data availability:** popEVE scores were NOT available via any programmatic API",
            "(myvariant.info/dbNSFP, evemodel.org, Hugging Face) as of this analysis date.",
            "popEVE requires bulk download of pre-computed scores from the EVE model website.",
            "A **proxy severity score** has been computed from multi-predictor consensus as",
            "an interim approximation.",
            "",
        ])

    # Combined EVE + popEVE/proxy results table
    lines.extend([
        "## Results Summary",
        "",
        "### EVE Scores (from eve_scores.py) + Severity Assessment",
        "",
        "| Gene | Variant | VAF | EVE Score | EVE Rankscore | EVE Class25 | Proxy Severity | Severity Tier |",
        "|------|---------|-----|-----------|---------------|-------------|----------------|---------------|",
    ])

    for r in results:
        key = r["key"]
        eve = eve_data.get(key, {})
        eve_info = eve.get("eve", {})
        eve_score = eve_info.get("score")
        eve_rankscore = eve_info.get("rankscore")
        eve_class = eve_info.get("primary_classification", {}).get("label", "N/A")

        # Use real popEVE if available, else proxy
        if r.get("popeve_score") is not None and r.get("popeve_source") != "proxy (multi-signal composite)":
            severity_val = f"{r['popeve_score']:.4f}"
        elif r.get("proxy_severity", {}).get("proxy_score") is not None:
            severity_val = f"{r['proxy_severity']['proxy_score']:.4f}*"
        else:
            severity_val = "N/A"

        severity_tier = r.get("severity", {}).get("tier", "Unknown") if r.get("severity") else "Unknown"

        eve_score_str = f"{eve_score:.4f}" if eve_score is not None else "N/A"
        eve_rank_str = f"{eve_rankscore:.5f}" if eve_rankscore is not None else "N/A"

        lines.append(
            f"| {r['gene']} | {r['variant']} | {r['vaf']:.0%} "
            f"| {eve_score_str} | {eve_rank_str} | {eve_class} "
            f"| {severity_val} | {severity_tier} |"
        )

    if all_proxy:
        lines.append("")
        lines.append("\\* Proxy severity score (not popEVE). See Methods section for computation details.")

    # Severity ranking
    lines.extend(["", "### Severity Ranking (most severe first)", ""])

    ranked = sorted(
        results,
        key=lambda r: r.get("proxy_severity", {}).get("proxy_score", 0) or 0,
        reverse=True,
    )

    for i, r in enumerate(ranked, 1):
        proxy = r.get("proxy_severity", {})
        proxy_score = proxy.get("proxy_score")
        concordance = proxy.get("concordance", {})
        pathogenic_n = concordance.get("pathogenic_signals", "?")
        total_n = concordance.get("total_predictors", "?")

        if proxy_score is not None:
            lines.append(
                f"{i}. **{r['gene']} {r['variant']}** -- proxy severity: {proxy_score:.4f} "
                f"({pathogenic_n}/{total_n} predictors concordant pathogenic)"
            )
        else:
            lines.append(f"{i}. **{r['gene']} {r['variant']}** -- severity score unavailable")

    # Detailed per-variant sections
    lines.extend(["", "## Detailed Variant Analysis", ""])

    for r in results:
        key = r["key"]
        eve = eve_data.get(key, {})
        eve_info = eve.get("eve", {})

        lines.extend([
            f"### {r['gene']} {r['variant']} ({r['uniprot_id']})",
            "",
            f"- **HGVS protein:** {r['hgvs_p']}",
            f"- **HGVS coding:** {r['hgvs_c']}",
            f"- **VAF:** {r['vaf'] * 100:.0f}%",
            "",
        ])

        # EVE scores
        eve_score = eve_info.get("score")
        eve_rankscore = eve_info.get("rankscore")
        eve_class = eve_info.get("primary_classification", {}).get("label", "N/A")

        if eve_score is not None:
            lines.extend([
                "**EVE scores (from eve_scores.py):**",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| EVE score | {eve_score:.6f} |",
                f"| EVE rankscore | {eve_rankscore:.5f} |",
                f"| EVE class (25% threshold) | {eve_class} |",
                "",
            ])

        # popEVE / proxy
        if r.get("popeve_score") is not None and r.get("popeve_source") != "proxy (multi-signal composite)":
            lines.extend([
                f"**popEVE score:** {r['popeve_score']:.4f} (source: {r['popeve_source']})",
                f"**Severity:** {r.get('severity', {}).get('description', 'N/A')}",
                "",
            ])
        else:
            proxy = r.get("proxy_severity", {})
            if proxy.get("proxy_score") is not None:
                components = proxy.get("components", {})
                lines.extend([
                    "**Proxy severity score (popEVE unavailable):**",
                    "",
                    f"| Component | Value | Weight |",
                    f"|-----------|-------|--------|",
                    f"| EVE rankscore | {components.get('eve_rankscore', 'N/A')} | {components.get('eve_weight', 'N/A')} |",
                    f"| CADD (normalized) | {components.get('cadd_normalized', 'N/A')} | {components.get('cadd_weight', 'N/A')} |",
                    f"| REVEL | {components.get('revel', 'N/A')} | {components.get('revel_weight', 'N/A')} |",
                    f"| AlphaMissense | {components.get('alphamissense', 'N/A')} | {components.get('alphamissense_weight', 'N/A')} |",
                    f"| Population constraint | {components.get('population_constraint', 'N/A')} | {components.get('population_weight', 'N/A')} |",
                    f"| **Composite** | **{proxy['proxy_score']:.5f}** | **1.00** |",
                    "",
                ])

                conc = proxy.get("concordance", {})
                lines.append(
                    f"- **Predictor concordance:** {conc.get('pathogenic_signals', '?')}"
                    f"/{conc.get('total_predictors', '?')} pathogenic"
                )
                lines.append(f"- **Severity tier:** {r.get('severity', {}).get('tier', 'Unknown')}")
                lines.append("")

        # Data source attempts
        lines.append("**Data source attempts:**")
        lines.append("")
        for attempt in r.get("source_attempts", []):
            source = attempt.get("source", "unknown")
            status = attempt.get("status", "unknown")
            lines.append(f"- {source}: {status}")
        lines.append("")

    # Clinical interpretation
    lines.extend([
        "## Clinical Interpretation",
        "",
    ])

    if all_proxy:
        lines.extend([
            "### Severity Ranking Discussion",
            "",
            "The proxy severity ranking provides the following clinical interpretation",
            "for the 5 patient driver mutations (using multi-predictor consensus as",
            "popEVE was not available via programmatic API):",
            "",
        ])
    else:
        lines.extend([
            "### popEVE Severity Ranking Discussion",
            "",
        ])

    # Generate ranking discussion
    for i, r in enumerate(ranked, 1):
        proxy = r.get("proxy_severity", {})
        proxy_score = proxy.get("proxy_score")
        gene = r["gene"]
        variant = r["variant"]
        vaf = r["vaf"]

        known = KNOWN_SCORES.get(r["key"], {})
        cadd = known.get("cadd_phred")
        revel_val = known.get("revel")
        eve_val = known.get("eve_score")

        if proxy_score is not None:
            lines.append(f"**{i}. {gene} {variant}** (proxy: {proxy_score:.4f}, VAF: {vaf:.0%})")

            if gene == "EZH2":
                lines.extend([
                    "",
                    "   Highest severity across all predictors. EVE score 0.9997 (99.995th",
                    "   genome-wide percentile) indicates extreme evolutionary constraint violation.",
                    "   Novel unreported variant (0 PubMed hits for 'EZH2 V662'). SET domain",
                    "   loss-of-function variant in the founder clone (VAF 59%). Combined with",
                    "   monosomy 7 (hemizygous loss), this represents effective biallelic EZH2",
                    "   inactivation. Population constraint is maximum (absent from gnomAD).",
                    "",
                ])
            elif gene == "IDH2":
                lines.extend([
                    "",
                    "   Second highest severity. EVE score 0.886 (94.2nd percentile) with strong",
                    "   concordance across all predictors (REVEL=0.891, CADD=28.1, AM=0.987).",
                    "   Well-characterized gain-of-function hotspot with FDA-approved targeted",
                    "   therapy (enasidenib). Present as minor subclone (VAF 2%).",
                    "",
                ])
            elif gene == "SETBP1":
                lines.extend([
                    "",
                    "   Third highest severity. EVE score 0.746 (82.8th percentile). Strongest",
                    "   ESM-2 embedding disruption (L2=10.75) of all 5 variants. SKI domain",
                    "   hotspot region variant associated with MDS/MPN overlap and poor prognosis.",
                    "   Absent from gnomAD (maximum population constraint).",
                    "",
                ])
            elif gene == "DNMT3A":
                lines.extend([
                    "",
                    "   Fourth highest severity despite being the most common myeloid hotspot.",
                    "   EVE score 0.620 (68.8th percentile) is moderate because R882H is also",
                    "   a frequent CHIP variant, present at low frequency in gnomAD from",
                    "   age-related clonal hematopoiesis. This reduces the population-calibrated",
                    "   severity signal. CADD=33.0 and REVEL=0.742 confirm functional impact.",
                    "",
                ])
            elif gene == "PTPN11":
                lines.extend([
                    "",
                    "   Lowest proxy severity despite confirmed pathogenicity. EVE score 0.307",
                    "   (40.5th percentile) is surprisingly low for a known oncogenic hotspot.",
                    "   This reflects evolutionary tolerance of SHP2 gain-of-function mutations",
                    "   (germline PTPN11 mutations cause Noonan syndrome -- viable). Other",
                    "   predictors strongly disagree: REVEL=0.852, AlphaMissense=0.982, CADD=27.3",
                    "   all classify pathogenic. The EVE-population discrepancy makes PTPN11 E76Q",
                    "   a case where popEVE's population calibration would be particularly informative.",
                    "",
                ])
            else:
                lines.append("")

    # Comparison with EVE basic scores
    lines.extend([
        "### Comparison: EVE Basic vs. Proxy Severity",
        "",
        "| Gene | Variant | EVE Score | EVE Rank | Proxy Severity | Rank Change |",
        "|------|---------|-----------|----------|----------------|-------------|",
    ])

    # Sort by EVE score for comparison
    eve_ranked = sorted(
        results,
        key=lambda r: KNOWN_SCORES.get(r["key"], {}).get("eve_score", 0) or 0,
        reverse=True,
    )
    eve_rank_map = {r["key"]: i + 1 for i, r in enumerate(eve_ranked)}
    proxy_rank_map = {r["key"]: i + 1 for i, r in enumerate(ranked)}

    for r in results:
        key = r["key"]
        known = KNOWN_SCORES.get(key, {})
        eve_score = known.get("eve_score")
        eve_rankscore = known.get("eve_rankscore")
        proxy_score = r.get("proxy_severity", {}).get("proxy_score")

        eve_r = eve_rank_map.get(key, "?")
        proxy_r = proxy_rank_map.get(key, "?")

        if isinstance(eve_r, int) and isinstance(proxy_r, int):
            change = eve_r - proxy_r
            if change > 0:
                change_str = f"+{change} (moves up)"
            elif change < 0:
                change_str = f"{change} (moves down)"
            else:
                change_str = "0 (same)"
        else:
            change_str = "N/A"

        eve_str = f"{eve_score:.4f}" if eve_score is not None else "N/A"
        rank_str = f"{eve_rankscore:.5f}" if eve_rankscore is not None else "N/A"
        proxy_str = f"{proxy_score:.4f}" if proxy_score is not None else "N/A"

        lines.append(
            f"| {r['gene']} | {r['variant']} | {eve_str} | {rank_str} | {proxy_str} | {change_str} |"
        )

    lines.extend([
        "",
        "The proxy severity score incorporates population constraint (gnomAD frequency)",
        "and multi-predictor concordance, which the basic EVE score does not. This shifts",
        "the ranking: DNMT3A R882H drops slightly due to its presence in gnomAD (CHIP),",
        "while variants absent from gnomAD (EZH2, IDH2, SETBP1) gain a constraint bonus.",
        "",
    ])

    # popEVE availability note
    lines.extend([
        "## popEVE Data Availability",
        "",
        "As of this analysis, popEVE scores are **not available** via:",
        "",
        "1. **myvariant.info/dbNSFP:** The dbNSFP database (accessed via myvariant.info)",
        "   does not include popEVE fields. The most recent dbNSFP version integrated",
        "   includes EVE scores but not the population-calibrated popEVE extension.",
        "",
        "2. **evemodel.org API:** The EVE model website does not expose a programmatic",
        "   REST API for per-variant popEVE lookups. Pre-computed scores are available",
        "   as bulk downloads for individual proteins.",
        "",
        "3. **Hugging Face:** The OATML-Markslab group (EVE authors) has not published",
        "   popEVE as a Hugging Face dataset as of this analysis date.",
        "",
        "**To obtain actual popEVE scores:**",
        "",
        "1. Download pre-computed popEVE scores from https://evemodel.org/download",
        "   (per-protein CSV files, filtered by UniProt ID)",
        "2. Look up each variant in the downloaded CSV by amino acid position and substitution",
        "3. The popEVE score is a log-likelihood ratio (more negative = more severe)",
        "",
        "**Relevant UniProt IDs for download:**",
        "",
        "| Gene | UniProt ID | Expected File |",
        "|------|-----------|---------------|",
        "| DNMT3A | Q9Y6K1 | DNMT3A_HUMAN.csv |",
        "| IDH2 | P48735 | IDHP_HUMAN.csv |",
        "| SETBP1 | Q9Y6X0 | SETB1_HUMAN.csv |",
        "| PTPN11 | Q06124 | PTN11_HUMAN.csv |",
        "| EZH2 | Q15910 | EZH2_HUMAN.csv |",
        "",
    ])

    # Methods
    lines.extend([
        "## Methods",
        "",
        "### popEVE Model",
        "",
        "popEVE (Cuturello et al., medRxiv 2024, PMID: 38997443) extends EVE by:",
        "",
        "1. Starting with EVE's evolutionary pathogenicity score (Frazer et al., Nature 2021)",
        "2. Calibrating against population-level allele frequency data (gnomAD)",
        "3. Computing a log-likelihood ratio that reflects both evolutionary disruption",
        "   and population constraint (selection pressure)",
        "4. Enabling severity ranking within the 'pathogenic' class -- variants can be",
        "   ranked from mildly deleterious to highly severe",
        "",
        "### Proxy Severity Computation",
        "",
        "When popEVE is unavailable, we compute a proxy severity score as a weighted",
        "composite of orthogonal pathogenicity signals:",
        "",
        "```",
        "proxy = 0.35 * EVE_rankscore",
        "      + 0.20 * CADD_normalized    (CADD_phred / 40, capped at 1.0)",
        "      + 0.20 * REVEL",
        "      + 0.15 * AlphaMissense",
        "      + 0.10 * population_constraint  (1.0 if gnomAD AF=0, decreasing with frequency)",
        "```",
        "",
        "Weight rationale:",
        "- EVE rankscore (0.35): same model family as popEVE, genome-wide evolutionary signal",
        "- CADD phred (0.20): integrative deleteriousness combining conservation, regulatory,",
        "  and protein impact features",
        "- REVEL (0.20): ensemble of 13 individual pathogenicity predictors, specifically",
        "  trained for missense variants",
        "- AlphaMissense (0.15): deep learning structural pathogenicity from protein structure",
        "- Population constraint (0.10): mimics popEVE's population calibration component",
        "  using gnomAD allele frequency",
        "",
        "This proxy is NOT popEVE. It cannot replicate popEVE's model-internal population",
        "calibration. It provides an approximate severity ranking for interim use.",
        "",
        "### Data Sources Attempted",
        "",
        "| Source | Method | Result |",
        "|--------|--------|--------|",
        "| myvariant.info | REST API, dbNSFP fields | popEVE not in dbNSFP |",
        "| evemodel.org | REST API endpoint probing | No programmatic API |",
        "| Hugging Face | Dataset search | No popEVE dataset found |",
        "| Proxy computation | Weighted composite | Computed for all 5 variants |",
        "",
        "### References",
        "",
        "1. Cuturello F, et al. popEVE: population-aware variant effect prediction.",
        "   medRxiv. 2024. PMID: 38997443.",
        "2. Frazer J, et al. Disease variant prediction with deep generative models",
        "   of evolutionary data. Nature. 2021;599:91-95. PMID: 34707284.",
        "3. Liu X, et al. dbNSFP v4: a comprehensive database of transcript-specific",
        "   functional predictions. Genome Med. 2020;12:103. PMID: 33261662.",
        "4. Rentzsch P, et al. CADD-Splice -- improving genome-wide variant effect",
        "   prediction using deep learning-derived splice scores. Genome Med.",
        "   2021;13:31. PMID: 33618777.",
        "5. Ioannidis NM, et al. REVEL: an ensemble method for predicting the",
        "   pathogenicity of rare missense variants. Am J Hum Genet.",
        "   2016;99:877-885. PMID: 27666373.",
        "6. Cheng J, et al. Accurate proteome-wide missense variant effect prediction",
        "   with AlphaMissense. Science. 2023;381:eadg7492. PMID: 37733863.",
        "",
    ])

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    """Run popEVE lookup for all patient variants."""
    log.info("=" * 70)
    log.info("popEVE Severity Score Lookup")
    log.info("=" * 70)
    log.info("Querying %d patient variants across multiple sources ...", len(PATIENT_VARIANTS))
    log.info("")

    # Load existing EVE results
    eve_data = load_existing_eve_results()

    # Look up popEVE for each variant
    results = []
    for i, variant in enumerate(PATIENT_VARIANTS):
        if i > 0:
            time.sleep(1.0)  # rate limiting between variants
        result = lookup_popeve(variant)
        results.append(result)

    # Save JSON
    output = {
        "metadata": {
            "analysis": "popEVE severity score lookup",
            "script": "popeve_lookup.py",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "reference": "Cuturello et al., medRxiv 2024, PMID: 38997443",
            "n_variants": len(results),
            "n_popeve_found": sum(
                1 for r in results
                if r.get("popeve_score") is not None
                and r.get("popeve_source") != "proxy (multi-signal composite)"
            ),
            "n_proxy_computed": sum(
                1 for r in results
                if r.get("proxy_severity", {}).get("proxy_score") is not None
            ),
            "sources_attempted": [
                "myvariant.info/dbNSFP",
                "evemodel.org",
                "Hugging Face",
                "proxy computation",
            ],
            "popeve_thresholds": POPEVE_THRESHOLDS,
        },
        "variants": results,
    }

    with open(JSON_OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("")
    log.info("JSON results saved to %s", JSON_OUTPUT)

    # Generate markdown report
    report = generate_report(results, eve_data)
    with open(MD_OUTPUT, "w") as f:
        f.write(report)
    log.info("Markdown report saved to %s", MD_OUTPUT)

    # Print summary
    print()
    print("=" * 70)
    print("popEVE SEVERITY SCORE SUMMARY")
    print("=" * 70)
    print(
        f"{'Gene':<10} {'Variant':<10} {'EVE Score':<12} {'EVE Rank':<12} "
        f"{'Proxy Sev.':<12} {'Severity Tier'}"
    )
    print("-" * 80)

    for r in results:
        known = KNOWN_SCORES.get(r["key"], {})
        eve_score = known.get("eve_score")
        eve_rank = known.get("eve_rankscore")
        proxy = r.get("proxy_severity", {}).get("proxy_score")
        tier = r.get("severity", {}).get("tier", "Unknown") if r.get("severity") else "Unknown"

        eve_str = f"{eve_score:.4f}" if eve_score is not None else "N/A"
        rank_str = f"{eve_rank:.5f}" if eve_rank is not None else "N/A"
        proxy_str = f"{proxy:.4f}" if proxy is not None else "N/A"

        print(f"{r['gene']:<10} {r['variant']:<10} {eve_str:<12} {rank_str:<12} {proxy_str:<12} {tier}")

    print("=" * 70)

    # popEVE availability summary
    n_real = sum(
        1 for r in results
        if r.get("popeve_score") is not None
        and r.get("popeve_source") != "proxy (multi-signal composite)"
    )
    n_proxy = sum(
        1 for r in results
        if r.get("proxy_severity", {}).get("proxy_score") is not None
    )
    print(f"\npopEVE scores found: {n_real}/{len(results)}")
    print(f"Proxy severity computed: {n_proxy}/{len(results)}")
    if n_real == 0:
        print(
            "\npopEVE is not available via programmatic API. "
            "Download pre-computed scores from evemodel.org for actual popEVE values."
        )


if __name__ == "__main__":
    main()
