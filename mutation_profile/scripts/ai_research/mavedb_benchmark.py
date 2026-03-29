#!/usr/bin/env python3
"""
mavedb_benchmark.py -- Query MaveDB for deep mutational scanning data covering
benchmark genes and patient variants, providing PS3-level functional evidence.

Queries MaveDB API v1 for score sets covering target genes:
    PTPN11, DNMT3A, TP53, EZH2, ASXL1, TET2, RUNX1, CBL, NRAS, KRAS, JAK2,
    IDH2, SETBP1, NPM1, FLT3, SF3B1, SRSF2

For each gene with DMS data:
    - Downloads all gene-targeted score sets
    - Downloads full score CSVs
    - Looks up specific benchmark variant positions
    - Classifies ACMG PS3 evidence level

Patient variants (always included):
    DNMT3A R882H, IDH2 R140Q, SETBP1 G870S, PTPN11 E76Q, EZH2 V662A

Outputs:
    mutation_profile/results/ai_research/benchmark/mavedb_functional_evidence.json

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/mavedb_benchmark.py

Runtime: ~3-5 minutes (API calls with rate limiting)
Dependencies: requests
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = RESULTS_DIR / "mavedb_functional_evidence.json"

# ---------------------------------------------------------------------------
# MaveDB API config
# ---------------------------------------------------------------------------
MAVEDB_BASE = "https://api.mavedb.org/api/v1"
RATE_LIMIT_SLEEP = 1.5  # seconds between requests
REQUEST_TIMEOUT = 60

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "mrna-hematology-research/1.0 (academic; mailto:research@theeducationalequalityinstitute.org)",
    "Accept": "application/json",
})

# ---------------------------------------------------------------------------
# Amino acid mappings
# ---------------------------------------------------------------------------
AA_1TO3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}
AA_3TO1 = {v: k for k, v in AA_1TO3.items()}

# ---------------------------------------------------------------------------
# Target genes
# ---------------------------------------------------------------------------
TARGET_GENES = [
    "PTPN11", "DNMT3A", "TP53", "EZH2", "ASXL1", "TET2", "RUNX1",
    "CBL", "NRAS", "KRAS", "JAK2", "IDH2", "SETBP1", "NPM1", "FLT3",
    "SF3B1", "SRSF2",
]

# Patient variants (always look up)
PATIENT_VARIANTS = {
    "DNMT3A": {"variant": "R882H", "position": 882, "ref": "R", "alt": "H", "vaf": 0.39},
    "IDH2": {"variant": "R140Q", "position": 140, "ref": "R", "alt": "Q", "vaf": 0.02},
    "SETBP1": {"variant": "G870S", "position": 870, "ref": "G", "alt": "S", "vaf": 0.34},
    "PTPN11": {"variant": "E76Q", "position": 76, "ref": "E", "alt": "Q", "vaf": 0.29},
    "EZH2": {"variant": "V662A", "position": 662, "ref": "V", "alt": "A", "vaf": 0.59},
}

# Known myeloid hotspot variants to look up per gene (benchmark)
BENCHMARK_VARIANTS: dict[str, list[dict[str, Any]]] = {
    "TP53": [
        {"variant": "R175H", "position": 175, "ref": "R", "alt": "H"},
        {"variant": "R248W", "position": 248, "ref": "R", "alt": "W"},
        {"variant": "R273H", "position": 273, "ref": "R", "alt": "H"},
        {"variant": "R249S", "position": 249, "ref": "R", "alt": "S"},
        {"variant": "Y220C", "position": 220, "ref": "Y", "alt": "C"},
        {"variant": "G245S", "position": 245, "ref": "G", "alt": "S"},
        {"variant": "R282W", "position": 282, "ref": "R", "alt": "W"},
        {"variant": "C176Y", "position": 176, "ref": "C", "alt": "Y"},
    ],
    "KRAS": [
        {"variant": "G12D", "position": 12, "ref": "G", "alt": "D"},
        {"variant": "G12V", "position": 12, "ref": "G", "alt": "V"},
        {"variant": "G12C", "position": 12, "ref": "G", "alt": "C"},
        {"variant": "G13D", "position": 13, "ref": "G", "alt": "D"},
        {"variant": "Q61H", "position": 61, "ref": "Q", "alt": "H"},
        {"variant": "A146T", "position": 146, "ref": "A", "alt": "T"},
    ],
    "NRAS": [
        {"variant": "G12D", "position": 12, "ref": "G", "alt": "D"},
        {"variant": "G13D", "position": 13, "ref": "G", "alt": "D"},
        {"variant": "Q61R", "position": 61, "ref": "Q", "alt": "R"},
        {"variant": "Q61K", "position": 61, "ref": "Q", "alt": "K"},
    ],
    "JAK2": [
        {"variant": "V617F", "position": 617, "ref": "V", "alt": "F"},
    ],
    "DNMT3A": [
        {"variant": "R882H", "position": 882, "ref": "R", "alt": "H"},
        {"variant": "R882C", "position": 882, "ref": "R", "alt": "C"},
    ],
    "IDH2": [
        {"variant": "R140Q", "position": 140, "ref": "R", "alt": "Q"},
        {"variant": "R172K", "position": 172, "ref": "R", "alt": "K"},
    ],
    "PTPN11": [
        {"variant": "E76Q", "position": 76, "ref": "E", "alt": "Q"},
        {"variant": "E76K", "position": 76, "ref": "E", "alt": "K"},
        {"variant": "A72T", "position": 72, "ref": "A", "alt": "T"},
        {"variant": "G503E", "position": 503, "ref": "G", "alt": "E"},
    ],
    "EZH2": [
        {"variant": "V662A", "position": 662, "ref": "V", "alt": "A"},
        {"variant": "Y646N", "position": 646, "ref": "Y", "alt": "N"},
        {"variant": "A682G", "position": 682, "ref": "A", "alt": "G"},
    ],
    "ASXL1": [
        {"variant": "G646Wfs*12", "position": 646, "ref": "G", "alt": "fs"},
        {"variant": "R693*", "position": 693, "ref": "R", "alt": "*"},
    ],
    "RUNX1": [
        {"variant": "R80C", "position": 80, "ref": "R", "alt": "C"},
        {"variant": "R201Q", "position": 201, "ref": "R", "alt": "Q"},
        {"variant": "D198N", "position": 198, "ref": "D", "alt": "N"},
    ],
    "CBL": [
        {"variant": "C404Y", "position": 404, "ref": "C", "alt": "Y"},
        {"variant": "C381R", "position": 381, "ref": "C", "alt": "R"},
    ],
    "SETBP1": [
        {"variant": "G870S", "position": 870, "ref": "G", "alt": "S"},
        {"variant": "I871T", "position": 871, "ref": "I", "alt": "T"},
        {"variant": "D868N", "position": 868, "ref": "D", "alt": "N"},
    ],
    "SF3B1": [
        {"variant": "K700E", "position": 700, "ref": "K", "alt": "E"},
        {"variant": "K666N", "position": 666, "ref": "K", "alt": "N"},
    ],
    "SRSF2": [
        {"variant": "P95H", "position": 95, "ref": "P", "alt": "H"},
        {"variant": "P95L", "position": 95, "ref": "P", "alt": "L"},
    ],
    "TET2": [
        {"variant": "R1261C", "position": 1261, "ref": "R", "alt": "C"},
        {"variant": "H1382Y", "position": 1382, "ref": "H", "alt": "Y"},
    ],
    "NPM1": [
        {"variant": "W288Cfs*12", "position": 288, "ref": "W", "alt": "fs"},
    ],
    "FLT3": [
        {"variant": "D835Y", "position": 835, "ref": "D", "alt": "Y"},
        {"variant": "D835H", "position": 835, "ref": "D", "alt": "H"},
    ],
}

# Benchmark profiles from GENIE data
BENCHMARK_PROFILE_FILE = RESULTS_DIR / "benchmark_profiles.json"


def rate_limit() -> None:
    """Respect MaveDB rate limits."""
    time.sleep(RATE_LIMIT_SLEEP)


def search_scoresets(query: str) -> tuple[list[dict[str, Any]], int]:
    """Search MaveDB score sets via POST /score-sets/search.

    Returns (scoreset_list, total_count).
    """
    url = f"{MAVEDB_BASE}/score-sets/search"
    payload = {"text": query}
    log.info("POST %s body=%s", url, json.dumps(payload))
    try:
        resp = SESSION.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        log.info("  -> %d (%d bytes)", resp.status_code, len(resp.content))
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "scoreSets" in data:
                return data["scoreSets"], data.get("numScoreSets", len(data["scoreSets"]))
            if isinstance(data, dict) and "results" in data:
                return data["results"], len(data["results"])
            if isinstance(data, list):
                return data, len(data)
        return [], 0
    except Exception as exc:
        log.warning("Search failed for %r: %s", query, exc)
        return [], 0


def download_scores(urn: str) -> list[dict[str, Any]]:
    """Download variant scores CSV from a score set.

    Returns list of dicts with keys like hgvs_nt, hgvs_pro, score, etc.
    """
    url = f"{MAVEDB_BASE}/score-sets/{urn}/scores"
    log.info("GET %s", url)
    try:
        resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            log.warning("  -> %d for scores of %s", resp.status_code, urn)
            return []

        content_type = resp.headers.get("content-type", "")
        text = resp.text

        # MaveDB returns CSV
        if "csv" in content_type or text.startswith("accession,") or "hgvs" in text[:200]:
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
            log.info("  -> %d CSV rows from %s", len(rows), urn)
            return rows

        # JSON fallback
        try:
            data = resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "results" in data:
                return data["results"]
        except Exception:
            pass

        return []
    except Exception as exc:
        log.warning("Failed to download scores for %s: %s", urn, exc)
        return []


def extract_target_genes(ss: dict[str, Any]) -> list[str]:
    """Extract target gene names from a scoreset record."""
    targets = ss.get("targetGenes", ss.get("target_genes", []))
    names = []
    if isinstance(targets, list):
        for t in targets:
            name = t.get("name", "")
            if not name:
                tg = t.get("targetGene", t.get("target_gene", {}))
                if isinstance(tg, dict):
                    name = tg.get("name", "")
            if name:
                names.append(name)
    return names


def has_target_gene(ss: dict[str, Any], gene: str) -> bool:
    """Check if scoreset targets the specified gene."""
    targets = extract_target_genes(ss)
    gene_up = gene.upper()
    for t in targets:
        # Match gene name or gene name as part of target (e.g. "CBL Ubiquitin-associated domain")
        if gene_up == t.upper() or t.upper().startswith(gene_up + " ") or t.upper().startswith(gene_up + "_"):
            return True
    return False


def extract_scoreset_info(ss: dict[str, Any]) -> dict[str, Any]:
    """Extract relevant fields from a scoreset record."""
    targets = extract_target_genes(ss)
    doi_ids = ss.get("doiIdentifiers", ss.get("doi_identifiers", []))
    doi = ""
    if doi_ids and isinstance(doi_ids, list):
        d = doi_ids[0]
        doi = d.get("identifier", "") if isinstance(d, dict) else str(d)
    if not doi:
        pub = ss.get("primaryPublication", ss.get("primary_publication", {}))
        if isinstance(pub, dict):
            doi = pub.get("doi", pub.get("identifier", ""))

    return {
        "urn": ss.get("urn", ""),
        "title": ss.get("title", ""),
        "short_description": (ss.get("shortDescription", ss.get("short_description", "")) or "")[:500],
        "abstract": (ss.get("abstractText", ss.get("abstract_text", "")) or "")[:500],
        "method_text": (ss.get("methodText", ss.get("method_text", "")) or "")[:500],
        "target_genes": targets,
        "num_variants": ss.get("numVariants", ss.get("num_variants", None)),
        "published_date": ss.get("publishedDate", ss.get("published_date", "")),
        "creation_date": ss.get("creationDate", ss.get("creation_date", "")),
        "doi": doi,
        "licence": ss.get("licence", {}).get("shortName", "") if isinstance(ss.get("licence"), dict) else "",
    }


def parse_hgvs_position(hgvs_pro: str) -> tuple[str, int, str] | None:
    """Parse HGVS protein notation to extract (ref_aa_1letter, position, alt_aa_1letter).

    Handles both 3-letter (p.Arg882His) and 1-letter (p.R882H) formats.
    Returns None if unparseable.
    """
    if not hgvs_pro:
        return None

    # 3-letter format: p.Arg882His, p.Gly12Asp
    m = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", hgvs_pro)
    if m:
        ref_3 = m.group(1)
        pos = int(m.group(2))
        alt_3 = m.group(3)
        ref_1 = AA_3TO1.get(ref_3, "?")
        alt_1 = AA_3TO1.get(alt_3, "?")
        return ref_1, pos, alt_1

    # 1-letter format: p.R882H, p.G12D
    m = re.match(r"p\.([A-Z])(\d+)([A-Z*])", hgvs_pro)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)

    # 3-letter with Ter (stop): p.Arg693Ter -> R693*
    m = re.match(r"p\.([A-Z][a-z]{2})(\d+)(Ter)", hgvs_pro)
    if m:
        ref_3 = m.group(1)
        pos = int(m.group(2))
        ref_1 = AA_3TO1.get(ref_3, "?")
        return ref_1, pos, "*"

    return None


def find_variant_in_scores(
    scores: list[dict[str, Any]],
    position: int,
    ref_aa: str,
    alt_aa: str,
    gene: str = "",
) -> list[dict[str, Any]]:
    """Search scores for a specific variant at a protein position.

    For multi-gene score sets, filters by gene name in hgvs fields.
    Handles both 1-letter and 3-letter amino acid codes.
    """
    ref_3 = AA_1TO3.get(ref_aa, ref_aa)
    alt_3 = AA_1TO3.get(alt_aa, alt_aa)

    # Target patterns
    patterns_1 = [f"p.{ref_aa}{position}{alt_aa}"]
    patterns_3 = [f"p.{ref_3}{position}{alt_3}"]

    matches = []
    for row in scores:
        hgvs_pro = (row.get("hgvs_pro", "") or "").strip()
        if not hgvs_pro or hgvs_pro == "NA":
            continue

        # For multi-gene sets, filter by gene in hgvs_nt
        if gene:
            hgvs_nt = (row.get("hgvs_nt", "") or "").strip()
            if hgvs_nt and ":" in hgvs_nt:
                nt_gene = hgvs_nt.split(":")[0]
                if gene.upper() not in nt_gene.upper():
                    continue

        matched = False
        for pat in patterns_1 + patterns_3:
            if pat.lower() in hgvs_pro.lower():
                matched = True
                break

        if not matched:
            # Parse HGVS and check position + amino acids
            parsed = parse_hgvs_position(hgvs_pro)
            if parsed:
                p_ref, p_pos, p_alt = parsed
                if p_pos == position and p_ref == ref_aa and p_alt == alt_aa:
                    matched = True

        if matched:
            matches.append(row)

    return matches


def find_all_variants_at_position(
    scores: list[dict[str, Any]],
    position: int,
    gene: str = "",
) -> list[dict[str, Any]]:
    """Find all scored variants at a specific protein position."""
    matches = []
    for row in scores:
        hgvs_pro = (row.get("hgvs_pro", "") or "").strip()
        if not hgvs_pro or hgvs_pro == "NA":
            continue

        if gene:
            hgvs_nt = (row.get("hgvs_nt", "") or "").strip()
            if hgvs_nt and ":" in hgvs_nt:
                nt_gene = hgvs_nt.split(":")[0]
                if gene.upper() not in nt_gene.upper():
                    continue

        parsed = parse_hgvs_position(hgvs_pro)
        if parsed and parsed[1] == position:
            matches.append(row)

    return matches


def process_gene(gene: str) -> dict[str, Any]:
    """Search MaveDB for all DMS data for a given gene."""
    log.info("=" * 60)
    log.info("Processing gene: %s", gene)

    result: dict[str, Any] = {
        "gene": gene,
        "is_patient_gene": gene in PATIENT_VARIANTS,
        "search_hits": 0,
        "gene_targeted_scoresets": 0,
        "scoresets": [],
        "variant_lookups": [],
        "position_scans": [],
        "total_variant_hits": 0,
        "has_dms_data": False,
        "acmg_ps3_evidence": "N/A",
    }

    # Search MaveDB
    scoresets_raw, count = search_scoresets(gene)
    result["search_hits"] = count
    rate_limit()

    # Filter to gene-targeted scoresets
    targeted = []
    for ss in scoresets_raw:
        if has_target_gene(ss, gene):
            targeted.append(ss)

    result["gene_targeted_scoresets"] = len(targeted)
    result["has_dms_data"] = len(targeted) > 0

    if not targeted:
        log.info("  No gene-targeted score sets for %s", gene)
        return result

    log.info("  %d gene-targeted score sets for %s", len(targeted), gene)

    # Get benchmark variants for this gene
    variants_to_check = list(BENCHMARK_VARIANTS.get(gene, []))
    # Add patient variant if not already in benchmark list
    if gene in PATIENT_VARIANTS:
        pv = PATIENT_VARIANTS[gene]
        pv_key = f"{pv['ref']}{pv['position']}{pv['alt']}"
        existing_keys = [f"{v['ref']}{v['position']}{v['alt']}" for v in variants_to_check]
        if pv_key not in existing_keys:
            variants_to_check.insert(0, {
                "variant": pv["variant"],
                "position": pv["position"],
                "ref": pv["ref"],
                "alt": pv["alt"],
                "is_patient_variant": True,
            })

    # Process each targeted scoreset
    for ss in targeted:
        info = extract_scoreset_info(ss)
        urn = info["urn"]
        result["scoresets"].append(info)

        # Download scores
        scores = download_scores(urn)
        if not scores:
            log.info("  No scores downloaded for %s", urn)
            rate_limit()
            continue

        # Determine total gene-specific rows
        gene_scores = []
        # Check if multi-gene set by looking at hgvs_nt prefixes
        has_gene_prefix = False
        for row in scores[:20]:
            hgvs_nt = (row.get("hgvs_nt", "") or "")
            if ":" in hgvs_nt and hgvs_nt.split(":")[0] != "NA":
                has_gene_prefix = True
                break

        if has_gene_prefix:
            gene_scores = [
                r for r in scores
                if gene.upper() in ((r.get("hgvs_nt", "") or "").split(":")[0]).upper()
            ]
            log.info("  Multi-gene set: %d total rows, %d for %s", len(scores), len(gene_scores), gene)
        else:
            gene_scores = scores
            log.info("  Single-gene set: %d rows for %s", len(gene_scores), gene)

        scoreset_result = {
            "urn": urn,
            "title": info["title"],
            "total_score_rows": len(scores),
            "gene_specific_rows": len(gene_scores),
            "variant_hits": [],
            "position_context": [],
        }

        # Look up each variant
        for vinfo in variants_to_check:
            hits = find_variant_in_scores(
                gene_scores,
                vinfo["position"],
                vinfo["ref"],
                vinfo["alt"],
                gene=gene if has_gene_prefix else "",
            )

            if hits:
                for hit in hits:
                    score_val = hit.get("score", "")
                    try:
                        score_float = float(score_val) if score_val and score_val != "NA" else None
                    except (ValueError, TypeError):
                        score_float = None

                    variant_hit = {
                        "variant": vinfo["variant"],
                        "position": vinfo["position"],
                        "ref_aa": vinfo["ref"],
                        "alt_aa": vinfo["alt"],
                        "is_patient_variant": vinfo.get("is_patient_variant", gene in PATIENT_VARIANTS and vinfo["variant"] == PATIENT_VARIANTS[gene]["variant"]),
                        "hgvs_pro": hit.get("hgvs_pro", ""),
                        "hgvs_nt": hit.get("hgvs_nt", ""),
                        "score": score_float,
                        "score_raw": score_val,
                        "all_columns": {k: v for k, v in hit.items() if k not in ("accession",)},
                    }
                    scoreset_result["variant_hits"].append(variant_hit)
                    result["total_variant_hits"] += 1
                    log.info(
                        "    FOUND %s %s in %s: score=%s",
                        gene, vinfo["variant"], urn, score_val,
                    )

            # Get all variants at this position for context
            pos_variants = find_all_variants_at_position(
                gene_scores,
                vinfo["position"],
                gene=gene if has_gene_prefix else "",
            )
            if pos_variants:
                pos_context = {
                    "position": vinfo["position"],
                    "target_variant": vinfo["variant"],
                    "n_variants_at_position": len(pos_variants),
                    "variants": [],
                }
                for pv in pos_variants:
                    score_val = pv.get("score", "")
                    try:
                        score_float = float(score_val) if score_val and score_val != "NA" else None
                    except (ValueError, TypeError):
                        score_float = None
                    pos_context["variants"].append({
                        "hgvs_pro": pv.get("hgvs_pro", ""),
                        "score": score_float,
                    })
                scoreset_result["position_context"].append(pos_context)

        result["variant_lookups"].append(scoreset_result)
        rate_limit()

    # Classify ACMG PS3 evidence
    if result["total_variant_hits"] > 0:
        # Direct functional score found
        result["acmg_ps3_evidence"] = "PS3_Strong"
    elif result["has_dms_data"]:
        # DMS data exists for gene but specific variant not scored
        result["acmg_ps3_evidence"] = "PS3_Supporting"
    else:
        result["acmg_ps3_evidence"] = "N/A"

    return result


def load_benchmark_profiles() -> list[dict[str, Any]]:
    """Load benchmark profiles and extract unique missense variants per gene."""
    if not BENCHMARK_PROFILE_FILE.exists():
        log.info("No benchmark profiles file found at %s", BENCHMARK_PROFILE_FILE)
        return []

    with open(BENCHMARK_PROFILE_FILE) as f:
        data = json.load(f)

    profiles = data.get("profiles", [])
    variants_by_gene: dict[str, list[dict[str, Any]]] = {}

    for profile in profiles:
        for mut in profile.get("mutations", []):
            gene = mut.get("gene", "")
            hgvsp = mut.get("hgvsp", "")
            vc = mut.get("variant_classification", "")

            if gene not in TARGET_GENES:
                continue
            if vc != "Missense_Mutation":
                continue
            if not hgvsp:
                continue

            # Parse single-letter HGVS: R882H -> R, 882, H
            m = re.match(r"([A-Z])(\d+)([A-Z])", hgvsp)
            if not m:
                continue

            ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
            key = f"{ref}{pos}{alt}"

            if gene not in variants_by_gene:
                variants_by_gene[gene] = []

            existing_keys = [f"{v['ref']}{v['position']}{v['alt']}" for v in variants_by_gene[gene]]
            if key not in existing_keys:
                variants_by_gene[gene].append({
                    "variant": hgvsp,
                    "position": pos,
                    "ref": ref,
                    "alt": alt,
                    "source": "benchmark_profiles",
                })

    return variants_by_gene


def main() -> None:
    """Run MaveDB benchmark query for all target genes."""
    start_time = time.time()
    log.info("Starting MaveDB benchmark query")
    log.info("Target genes: %s", ", ".join(TARGET_GENES))

    # Load additional variants from benchmark profiles
    benchmark_extra = load_benchmark_profiles()
    if benchmark_extra:
        for gene, variants in benchmark_extra.items():
            if gene not in BENCHMARK_VARIANTS:
                BENCHMARK_VARIANTS[gene] = []
            existing_keys = [f"{v['ref']}{v['position']}{v['alt']}" for v in BENCHMARK_VARIANTS[gene]]
            for v in variants:
                key = f"{v['ref']}{v['position']}{v['alt']}"
                if key not in existing_keys:
                    BENCHMARK_VARIANTS[gene].append(v)
                    existing_keys.append(key)
        log.info(
            "Added %d extra variants from benchmark profiles across %d genes",
            sum(len(v) for v in benchmark_extra.values()),
            len(benchmark_extra),
        )

    results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "MaveDB API v1 (https://api.mavedb.org/)",
            "description": "Deep mutational scanning functional evidence for benchmark myeloid genes",
            "target_genes": TARGET_GENES,
            "patient_variants": {k: v["variant"] for k, v in PATIENT_VARIANTS.items()},
            "n_benchmark_variants_queried": sum(len(v) for v in BENCHMARK_VARIANTS.values()),
            "acmg_evidence_level": "PS3 (Functional Studies)",
            "methodology": [
                "POST /score-sets/search with gene name",
                "Filter to score sets with gene as named target",
                "Download full score CSV for each targeted score set",
                "Look up specific variant positions using HGVS protein notation",
                "Classify ACMG PS3 evidence based on direct variant score availability",
            ],
        },
        "summary": {
            "genes_with_dms_data": [],
            "genes_without_dms_data": [],
            "total_variant_hits": 0,
            "total_scoresets_found": 0,
            "patient_variant_coverage": {},
        },
        "gene_results": [],
    }

    # Process each gene
    for gene in TARGET_GENES:
        gene_result = process_gene(gene)
        results["gene_results"].append(gene_result)

        if gene_result["has_dms_data"]:
            results["summary"]["genes_with_dms_data"].append(gene)
        else:
            results["summary"]["genes_without_dms_data"].append(gene)

        results["summary"]["total_variant_hits"] += gene_result["total_variant_hits"]
        results["summary"]["total_scoresets_found"] += gene_result["gene_targeted_scoresets"]

        # Track patient variant coverage
        if gene in PATIENT_VARIANTS:
            pv = PATIENT_VARIANTS[gene]
            patient_hit = False
            patient_scores = []
            for lookup in gene_result.get("variant_lookups", []):
                for vh in lookup.get("variant_hits", []):
                    if vh.get("is_patient_variant"):
                        patient_hit = True
                        patient_scores.append({
                            "scoreset": lookup["urn"],
                            "score": vh["score"],
                            "hgvs_pro": vh["hgvs_pro"],
                        })
            results["summary"]["patient_variant_coverage"][gene] = {
                "variant": pv["variant"],
                "dms_data_exists": gene_result["has_dms_data"],
                "direct_score_found": patient_hit,
                "scores": patient_scores,
                "acmg_ps3": gene_result["acmg_ps3_evidence"],
            }

    elapsed = time.time() - start_time

    results["metadata"]["runtime_seconds"] = round(elapsed, 1)

    # Save results
    with open(JSON_OUTPUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", JSON_OUTPUT)

    # Print summary
    print("\n" + "=" * 70)
    print("MaveDB Benchmark Functional Evidence Query")
    print("=" * 70)
    print(f"\nRuntime: {elapsed:.1f}s")
    print(f"\nGenes with DMS data ({len(results['summary']['genes_with_dms_data'])}):")
    for gene in results["summary"]["genes_with_dms_data"]:
        gr = next(g for g in results["gene_results"] if g["gene"] == gene)
        print(f"  {gene:>10}: {gr['gene_targeted_scoresets']} score sets, {gr['total_variant_hits']} variant hits")
        for lookup in gr.get("variant_lookups", []):
            for vh in lookup.get("variant_hits", []):
                patient_tag = " [PATIENT]" if vh.get("is_patient_variant") else ""
                print(f"             {vh['variant']:>10} score={vh['score']}{patient_tag}  ({lookup['urn']})")

    print(f"\nGenes without DMS data ({len(results['summary']['genes_without_dms_data'])}):")
    for gene in results["summary"]["genes_without_dms_data"]:
        print(f"  {gene}")

    print(f"\nPatient variant coverage:")
    for gene, cov in results["summary"]["patient_variant_coverage"].items():
        status = "SCORED" if cov["direct_score_found"] else ("DMS exists" if cov["dms_data_exists"] else "No DMS")
        print(f"  {gene:>10} {cov['variant']:>8}: {status} (PS3: {cov['acmg_ps3']})")
        for sc in cov.get("scores", []):
            print(f"             score={sc['score']}  ({sc['scoreset']})")

    print(f"\nTotal: {results['summary']['total_scoresets_found']} score sets, "
          f"{results['summary']['total_variant_hits']} variant hits")
    print(f"\nOutput: {JSON_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
