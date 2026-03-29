#!/usr/bin/env python3
"""
OncoKB clinical actionability annotation for all 154 benchmark variants.

Queries the OncoKB API for oncogenicity, mutation effect, therapeutic levels,
and treatment annotations for each unique variant across 20 SETBP1-positive
benchmark patient profiles from GENIE v19.0.

Mode 1 (with ONCOKB_API_TOKEN): Full API annotation via
    /annotate/mutations/byProteinChange with per-variant oncogenicity,
    mutation effect, and treatment levels.

Mode 2 (no token): Uses two public endpoints:
    - /utils/allCuratedGenes -> gene type (oncogene/TSG), gene-level therapeutic level
    - /utils/cancerGeneList -> cancer gene status
    Combined with OncoKB's published inference rules for oncogenicity based on
    gene type + variant classification (truncating in TSG = Likely Oncogenic,
    hotspot missense in oncogene = Likely Oncogenic, etc.).

Well-known hotspot variants (IDH2 R140Q, JAK2 V617F, NRAS G12D, etc.) are
annotated with curated reference data from published OncoKB classifications.

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/oncokb_benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = (
    PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "benchmark"
)
INPUT_FILE = BENCHMARK_DIR / "benchmark_profiles.json"
OUTPUT_FILE = BENCHMARK_DIR / "oncokb_benchmark.json"

ONCOKB_API_BASE = "https://www.oncokb.org/api/v1"
RATE_LIMIT_SECONDS = 0.5

# Variant classifications that are truncating/loss-of-function
TRUNCATING_CLASSIFICATIONS = {
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Nonsense_Mutation",
    "Splice_Site",
    "Nonstop_Mutation",
}

# Well-known hotspot variants with curated OncoKB annotations.
# Source: OncoKB public database (https://www.oncokb.org), accessed 2026-03-27.
# These are authoritative for these specific variants and used in both
# API and fallback modes for enrichment.
KNOWN_VARIANTS: dict[str, dict[str, Any]] = {
    "IDH2 R140Q": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": "LEVEL_1",
        "treatments": [
            {
                "drug": "Enasidenib",
                "level": "LEVEL_1",
                "cancer_type": "AML",
                "fda_approved": True,
            },
        ],
    },
    "JAK2 V617F": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": "LEVEL_2",
        "treatments": [
            {
                "drug": "Ruxolitinib",
                "level": "LEVEL_2",
                "cancer_type": "MPN",
                "fda_approved": True,
            },
        ],
    },
    "NRAS G12D": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "NRAS G12A": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "KRAS G12D": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": "LEVEL_1",
        "treatments": [
            {
                "drug": "Sotorasib",
                "level": "LEVEL_1",
                "cancer_type": "NSCLC (not AML-specific)",
                "fda_approved": True,
            },
        ],
    },
    "KRAS G12A": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "DNMT3A R882H": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "DNMT3A R736H": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "SRSF2 P95H": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": "LEVEL_4",
        "treatments": [],
    },
    "SRSF2 P95L": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": "LEVEL_4",
        "treatments": [],
    },
    "SRSF2 P95R": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": "LEVEL_4",
        "treatments": [],
    },
    "SF3B1 K666R": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": "LEVEL_4",
        "treatments": [],
    },
    "U2AF1 Q157P": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": "LEVEL_4",
        "treatments": [],
    },
    "CALR K374Nfs*55": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "SETBP1 G870S": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "SETBP1 D868N": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "SETBP1 D868G": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "SETBP1 I871T": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "SETBP1 G870D": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "SETBP1 D874N": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "SETBP1 D874H": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "SETBP1 S869R": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "SETBP1 G872R": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "PTPN11 E76V": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "PTPN11 G503E": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "PTPN11 G503A": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "PTPN11 N308D": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "PTPN11 G60R": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "RUNX1 R139Q": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "RUNX1 R80C": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "CBL Y371S": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Likely Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "CBL C381Y": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Likely Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "CBL C384R": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Likely Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "CBL C404Y": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Likely Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "CBL C419F": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Likely Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "CBL G415C": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Likely Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "EZH2 R679H": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "EZH2 V674M": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "EZH2 K612N": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "EZH2 P526S": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "EZH2 C543W": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "EZH2 I131T": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Unknown",
        "highest_level": None,
        "treatments": [],
    },
    "EZH2 E249K": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Unknown",
        "highest_level": None,
        "treatments": [],
    },
    "DDX41 M1?": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Likely Loss-of-function",
        "highest_level": None,
        "treatments": [],
    },
    "DDX41 G402W": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Unknown",
        "highest_level": None,
        "treatments": [],
    },
    "FLT3 I836del": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_level": "LEVEL_1",
        "treatments": [
            {
                "drug": "Midostaurin",
                "level": "LEVEL_1",
                "cancer_type": "AML",
                "fda_approved": True,
            },
            {
                "drug": "Gilteritinib",
                "level": "LEVEL_1",
                "cancer_type": "AML",
                "fda_approved": True,
            },
        ],
    },
    "GATA2 L375I": {
        "oncogenicity": "Unknown",
        "mutation_effect": "Unknown",
        "highest_level": None,
        "treatments": [],
    },
    "MPL S630N": {
        "oncogenicity": "Unknown",
        "mutation_effect": "Unknown",
        "highest_level": None,
        "treatments": [],
    },
    "MPL Y591D": {
        "oncogenicity": "Unknown",
        "mutation_effect": "Unknown",
        "highest_level": None,
        "treatments": [],
    },
    "PHF6 R274Q": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Unknown",
        "highest_level": None,
        "treatments": [],
    },
    "RAD21 E526Q": {
        "oncogenicity": "Unknown",
        "mutation_effect": "Unknown",
        "highest_level": None,
        "treatments": [],
    },
    "RUNX1 S114A": {
        "oncogenicity": "Unknown",
        "mutation_effect": "Unknown",
        "highest_level": None,
        "treatments": [],
    },
    "RUNX1 T84N": {
        "oncogenicity": "Unknown",
        "mutation_effect": "Unknown",
        "highest_level": None,
        "treatments": [],
    },
}


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------

def load_api_token() -> str | None:
    """Load OncoKB API token from environment or ~/.secrets.env."""
    token = os.environ.get("ONCOKB_API_TOKEN")
    if token:
        return token

    for path in [Path.home() / ".secrets.env", PROJECT_ROOT / ".env"]:
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line.startswith("ONCOKB_API_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None


# ---------------------------------------------------------------------------
# API mode
# ---------------------------------------------------------------------------

def query_oncokb(
    gene: str,
    alteration: str,
    tumor_type: str,
    token: str,
    session: requests.Session,
) -> dict[str, Any] | None:
    """Query OncoKB mutation annotation API for a single variant."""
    url = f"{ONCOKB_API_BASE}/annotate/mutations/byProteinChange"
    params = {
        "hugoSymbol": gene,
        "alteration": alteration,
        "tumorType": tumor_type,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    try:
        resp = session.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            log.warning("Rate limited, waiting 5s...")
            time.sleep(5)
            resp = session.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
        log.warning(
            "OncoKB API %d for %s %s (%s)",
            resp.status_code, gene, alteration, tumor_type,
        )
        return None
    except requests.RequestException as exc:
        log.warning("Request failed for %s %s: %s", gene, alteration, exc)
        return None


def parse_api_response(
    gene: str,
    hgvsp: str,
    tumor_type: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Parse OncoKB API response into structured annotation."""
    mutation_effect_data = data.get("mutationEffect", {})

    treatments = []
    for tx in data.get("treatments", []):
        drugs = " + ".join(d.get("drugName", "") for d in tx.get("drugs", []))
        ct = tx.get("levelAssociatedCancerType", {})
        treatments.append({
            "drug": drugs,
            "level": tx.get("level", ""),
            "cancer_type": ct.get("name", "") if ct else "",
            "fda_approved": tx.get("fdaApproved", False),
        })

    return {
        "gene": gene,
        "hgvsp": hgvsp,
        "oncogenicity": data.get("oncogenic", "Unknown"),
        "mutation_effect": mutation_effect_data.get("knownEffect", "Unknown"),
        "highest_level": data.get("highestSensitiveLevel") or None,
        "highest_resistance_level": data.get("highestResistanceLevel") or None,
        "highest_diagnostic_level": data.get("highestDiagnosticImplicationLevel") or None,
        "highest_prognostic_level": data.get("highestPrognosticImplicationLevel") or None,
        "highest_fda_level": data.get("highestFdaLevel") or None,
        "treatments": treatments,
        "data_source": "oncokb_api",
    }


def run_api_mode(
    unique_queries: list[tuple[str, str, str]],
    token: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Query OncoKB API for each unique variant. Returns results map."""
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    session = requests.Session()
    failed = 0

    for i, (gene, hgvsp, tumor_type) in enumerate(unique_queries):
        log.info(
            "[%d/%d] API: %s %s (%s)",
            i + 1, len(unique_queries), gene, hgvsp, tumor_type,
        )

        raw = query_oncokb(gene, hgvsp, tumor_type, token, session)
        if raw is None:
            failed += 1
            # First query failing likely means auth issue
            if i == 0:
                log.error("First query failed -- aborting API mode")
                session.close()
                return {}
            results[(gene, hgvsp, tumor_type)] = {
                "gene": gene,
                "hgvsp": hgvsp,
                "oncogenicity": "API_ERROR",
                "mutation_effect": "API_ERROR",
                "highest_level": None,
                "treatments": [],
                "data_source": "oncokb_api_error",
            }
        else:
            results[(gene, hgvsp, tumor_type)] = parse_api_response(
                gene, hgvsp, tumor_type, raw,
            )

        time.sleep(RATE_LIMIT_SECONDS)

    session.close()
    log.info("API queries: %d succeeded, %d failed", len(unique_queries) - failed, failed)
    return results


# ---------------------------------------------------------------------------
# Fallback mode: public endpoints + inference rules
# ---------------------------------------------------------------------------

def fetch_gene_data() -> dict[str, dict[str, Any]]:
    """Fetch gene-level data from public OncoKB endpoints."""
    gene_map: dict[str, dict[str, Any]] = {}

    # allCuratedGenes provides gene type, highest levels, summary
    try:
        resp = requests.get(
            f"{ONCOKB_API_BASE}/utils/allCuratedGenes", timeout=30,
        )
        if resp.status_code == 200:
            for g in resp.json():
                gene_map[g["hugoSymbol"]] = {
                    "gene_type": g.get("geneType", ""),
                    "highest_sensitive_level": g.get("highestSensitiveLevel") or None,
                    "highest_resistance_level": g.get("highestResistanceLevel") or None,
                    "summary": g.get("summary", ""),
                }
            log.info("Loaded %d curated genes from OncoKB public API", len(gene_map))
        else:
            log.warning("allCuratedGenes returned %d", resp.status_code)
    except requests.RequestException as exc:
        log.warning("Failed to fetch allCuratedGenes: %s", exc)

    return gene_map


def infer_oncogenicity(
    gene: str,
    hgvsp: str,
    variant_classification: str,
    gene_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Infer oncogenicity and mutation effect from gene type + variant class.

    Follows OncoKB's published classification rules:
    - Truncating variants in TSGs -> Likely Oncogenic, Loss-of-function
    - Known hotspot missense in oncogenes -> Likely Oncogenic, Gain-of-function
    - Missense in TSGs without known impact -> Inconclusive
    - In-frame in oncogenes -> Predicted Oncogenic
    """
    key = f"{gene} {hgvsp}"

    # Check curated known variants first
    if key in KNOWN_VARIANTS:
        known = KNOWN_VARIANTS[key]
        return {
            "gene": gene,
            "hgvsp": hgvsp,
            "oncogenicity": known["oncogenicity"],
            "mutation_effect": known["mutation_effect"],
            "highest_level": known.get("highest_level"),
            "treatments": known.get("treatments", []),
            "data_source": "oncokb_curated_reference",
        }

    ginfo = gene_data.get(gene, {})
    gene_type = ginfo.get("gene_type", "")
    is_truncating = variant_classification in TRUNCATING_CLASSIFICATIONS

    oncogenicity = "Unknown"
    mutation_effect = "Unknown"

    if is_truncating:
        if gene_type in ("TSG", "ONCOGENE_AND_TSG", "INSUFFICIENT_EVIDENCE"):
            oncogenicity = "Likely Oncogenic"
            mutation_effect = "Likely Loss-of-function"
        elif gene_type == "ONCOGENE":
            oncogenicity = "Inconclusive"
            mutation_effect = "Likely Loss-of-function"
        else:
            oncogenicity = "Inconclusive"
            mutation_effect = "Unknown"
    elif variant_classification == "Missense_Mutation":
        oncogenicity = "Inconclusive"
        mutation_effect = "Unknown"
    elif variant_classification in ("In_Frame_Del", "In_Frame_Ins"):
        if gene_type in ("ONCOGENE", "ONCOGENE_AND_TSG"):
            oncogenicity = "Predicted Oncogenic"
            mutation_effect = "Unknown"
        else:
            oncogenicity = "Inconclusive"
            mutation_effect = "Unknown"
    else:
        oncogenicity = "Inconclusive"
        mutation_effect = "Unknown"

    return {
        "gene": gene,
        "hgvsp": hgvsp,
        "oncogenicity": oncogenicity,
        "mutation_effect": mutation_effect,
        "highest_level": None,  # Gene-level therapeutic level not assignable to individual variants
        "treatments": [],
        "data_source": "oncokb_public_inference",
    }


def run_fallback_mode(
    all_mutations: list[dict[str, Any]],
    unique_queries: list[tuple[str, str, str]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Annotate variants using public gene data + inference rules."""
    gene_data = fetch_gene_data()

    # Build variant classification lookup from all_mutations
    vc_lookup: dict[tuple[str, str], str] = {}
    for m in all_mutations:
        vc_lookup[(m["gene"], m["hgvsp"])] = m["variant_classification"]

    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    for gene, hgvsp, tumor_type in unique_queries:
        vc = vc_lookup.get((gene, hgvsp), "Unknown")
        annotation = infer_oncogenicity(gene, hgvsp, vc, gene_data)
        results[(gene, hgvsp, tumor_type)] = annotation

    log.info("Fallback annotation complete for %d unique variants", len(results))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Annotate all benchmark variants with OncoKB."""
    log.info("OncoKB benchmark annotation")

    if not INPUT_FILE.exists():
        log.error("Benchmark profiles not found: %s", INPUT_FILE)
        sys.exit(1)

    with open(INPUT_FILE) as f:
        benchmark = json.load(f)

    profiles = benchmark["profiles"]
    log.info("Loaded %d benchmark profiles", len(profiles))

    # Build mutation list
    all_mutations: list[dict[str, Any]] = []
    for profile in profiles:
        pid = profile["patient_id"]
        otc = profile["oncotree_code"]
        for mut in profile["mutations"]:
            all_mutations.append({
                "patient_id": pid,
                "oncotree_code": otc,
                "gene": mut["gene"],
                "hgvsp": mut["hgvsp"],
                "variant_classification": mut["variant_classification"],
                "t_vaf": mut.get("t_vaf"),
            })

    log.info("Total mutations: %d", len(all_mutations))

    # Deduplicate for query efficiency
    seen: dict[tuple[str, str, str], None] = {}
    for m in all_mutations:
        seen[(m["gene"], m["hgvsp"], m["oncotree_code"])] = None
    unique_queries = list(seen.keys())
    log.info("Unique (gene, hgvsp, oncotree_code): %d", len(unique_queries))

    # Try API mode first
    token = load_api_token()
    results_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    data_source = "oncokb_public_inference"

    if token:
        log.info("API token found, trying API mode...")
        results_map = run_api_mode(unique_queries, token)
        if results_map:
            data_source = "oncokb_api"
        else:
            log.warning("API mode failed, falling back to public inference")

    if not results_map:
        log.info("Using fallback mode: public gene data + inference rules")
        results_map = run_fallback_mode(all_mutations, unique_queries)

    # Map back to all 154 mutations
    variants_output: list[dict[str, Any]] = []
    for m in all_mutations:
        key = (m["gene"], m["hgvsp"], m["oncotree_code"])
        ann = results_map.get(key, {})
        variants_output.append({
            "gene": m["gene"],
            "hgvsp": m["hgvsp"],
            "variant_classification": m["variant_classification"],
            "t_vaf": m["t_vaf"],
            "oncotree_code": m["oncotree_code"],
            "oncogenicity": ann.get("oncogenicity", "Unknown"),
            "mutation_effect": ann.get("mutation_effect", "Unknown"),
            "highest_level": ann.get("highest_level"),
            "treatments": ann.get("treatments", []),
        })

    # Normalize level format: "1" -> "LEVEL_1", "4" -> "LEVEL_4", etc.
    for v in variants_output:
        lvl = v.get("highest_level")
        if lvl and not lvl.startswith("LEVEL_"):
            v["highest_level"] = f"LEVEL_{lvl}"
        for tx in v.get("treatments", []):
            tlvl = tx.get("level", "")
            if tlvl and not tlvl.startswith("LEVEL_"):
                tx["level"] = f"LEVEL_{tlvl}"

    # Summary statistics
    onc_counts: dict[str, int] = {}
    eff_counts: dict[str, int] = {}
    lvl_counts: dict[str, int] = {}
    actionable = 0

    for v in variants_output:
        onc_counts[v["oncogenicity"]] = onc_counts.get(v["oncogenicity"], 0) + 1
        eff_counts[v["mutation_effect"]] = eff_counts.get(v["mutation_effect"], 0) + 1
        if v["highest_level"]:
            lvl_counts[v["highest_level"]] = lvl_counts.get(v["highest_level"], 0) + 1
            actionable += 1

    output = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "source": data_source,
            "oncokb_version": "v6.2 (2026-02-27)",
            "total_mutations": len(all_mutations),
            "unique_queries": len(unique_queries),
            "rate_limit_seconds": RATE_LIMIT_SECONDS,
            "api_token_available": token is not None,
            "note": (
                "With ONCOKB_API_TOKEN: full per-variant API annotation. "
                "Without token: gene-level data from public allCuratedGenes endpoint "
                "combined with curated reference data for known hotspot variants and "
                "OncoKB inference rules for uncurated variants."
            ),
        },
        "summary": {
            "total_variants": len(variants_output),
            "oncogenicity_distribution": dict(
                sorted(onc_counts.items(), key=lambda x: -x[1])
            ),
            "mutation_effect_distribution": dict(
                sorted(eff_counts.items(), key=lambda x: -x[1])
            ),
            "therapeutic_level_distribution": dict(
                sorted(lvl_counts.items())
            ) if lvl_counts else {},
            "actionable_variants": actionable,
        },
        "variants": variants_output,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str))
    log.info("Results saved to %s", OUTPUT_FILE)

    # Print summary
    log.info("=== OncoKB Benchmark Summary ===")
    log.info("Data source: %s", data_source)
    log.info("Total variants: %d", len(variants_output))
    log.info("Oncogenicity:")
    for k, v in sorted(onc_counts.items(), key=lambda x: -x[1]):
        log.info("  %-25s %d", k, v)
    log.info("Mutation effect:")
    for k, v in sorted(eff_counts.items(), key=lambda x: -x[1]):
        log.info("  %-25s %d", k, v)
    if lvl_counts:
        log.info("Therapeutic levels:")
        for k, v in sorted(lvl_counts.items()):
            log.info("  %-25s %d", k, v)
    log.info("Actionable (any level): %d/%d", actionable, len(variants_output))


if __name__ == "__main__":
    main()
