#!/usr/bin/env python3
"""
Manual QA cross-check: re-query myvariant.info for 10 benchmark variants.

Loads benchmark_results.json, picks 10 well-known missense variants,
re-queries myvariant.info with fields=dbnsfp,cadd,clinvar,gnomad_exome,
and compares stored scores against fresh results. Flags discrepancies.

Variants selected:
  JAK2 V617F, SETBP1 G870S, DNMT3A R882H, IDH2 R140Q, NRAS G12D,
  EZH2 E249K, CBL C404Y, SRSF2 P95R, RUNX1 R80C, TET2 K1197E

Output: mutation_profile/results/ai_research/benchmark/qa_manual_crosscheck.md
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "ai_research" / "benchmark"
)
BENCHMARK_RESULTS = BENCHMARK_DIR / "benchmark_results.json"
OUTPUT_MD = BENCHMARK_DIR / "qa_manual_crosscheck.md"

MYVARIANT_URL = "https://myvariant.info/v1/variant"
MYVARIANT_FIELDS = "dbnsfp,cadd,clinvar,gnomad_exome"

# 10 target variants with their hg19 (GRCh37) genomic coordinates from GENIE
TARGETS = [
    {
        "gene": "JAK2",
        "hgvsp": "V617F",
        "chrom": "9",
        "pos": 5073770,
        "ref": "G",
        "alt": "T",
    },
    {
        "gene": "SETBP1",
        "hgvsp": "G870S",
        "chrom": "18",
        "pos": 42531913,
        "ref": "G",
        "alt": "A",
    },
    {
        "gene": "DNMT3A",
        "hgvsp": "R882H",
        "chrom": "2",
        "pos": 25457242,
        "ref": "C",
        "alt": "T",
    },
    {
        "gene": "IDH2",
        "hgvsp": "R140Q",
        "chrom": "15",
        "pos": 90631934,
        "ref": "C",
        "alt": "T",
    },
    {
        "gene": "NRAS",
        "hgvsp": "G12D",
        "chrom": "1",
        "pos": 115258747,
        "ref": "C",
        "alt": "T",
    },
    {
        "gene": "EZH2",
        "hgvsp": "E249K",
        "chrom": "7",
        "pos": 148523708,
        "ref": "C",
        "alt": "T",
    },
    {
        "gene": "CBL",
        "hgvsp": "C404Y",
        "chrom": "11",
        "pos": 119148991,
        "ref": "G",
        "alt": "A",
    },
    {
        "gene": "SRSF2",
        "hgvsp": "P95R",
        "chrom": "17",
        "pos": 74732959,
        "ref": "G",
        "alt": "C",
    },
    {
        "gene": "RUNX1",
        "hgvsp": "R80C",
        "chrom": "21",
        "pos": 36259172,
        "ref": "G",
        "alt": "A",
    },
    {
        "gene": "TET2",
        "hgvsp": "K1197E",
        "chrom": "4",
        "pos": 106164079,
        "ref": "A",
        "alt": "G",
    },
]


@dataclass
class ScoreComparison:
    """Holds stored vs fresh score for one field."""

    field: str
    stored: Any
    fresh: Any
    match: bool
    note: str = ""


def build_hg19_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"chr{chrom}:g.{pos}{ref}>{alt}"


def query_myvariant(hg19_id: str) -> dict:
    """Query myvariant.info for a single variant."""
    try:
        r = requests.get(
            f"{MYVARIANT_URL}/{hg19_id}",
            params={"fields": MYVARIANT_FIELDS},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.error("myvariant.info error for %s: %s", hg19_id, exc)
        return {}


def _extract_float(val: Any) -> float | None:
    """Safely extract a float from myvariant.info response."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list):
        return _extract_float(val[0]) if val else None
    if isinstance(val, str):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return None


def _unwrap(obj: Any) -> Any:
    """Unwrap single-element list."""
    if isinstance(obj, list):
        return obj[0] if obj else {}
    return obj


def extract_scores(data: dict) -> dict[str, Any]:
    """Extract the 8 key scores from a myvariant.info response."""
    dbnsfp = data.get("dbnsfp", {})
    if isinstance(dbnsfp, list):
        dbnsfp = dbnsfp[0] if dbnsfp else {}

    # AlphaMissense
    am = _unwrap(dbnsfp.get("alphamissense", {}))
    alphamissense = _extract_float(am.get("score") if isinstance(am, dict) else am)

    # EVE (field name in myvariant.info is "score", not "eve_score")
    eve_data = _unwrap(dbnsfp.get("eve", {}))
    eve_score = _extract_float(
        eve_data.get("score") if isinstance(eve_data, dict) else eve_data
    )

    # CADD (try top-level cadd first, then dbnsfp.cadd)
    cadd_top = data.get("cadd", {})
    cadd_dbnsfp = _unwrap(dbnsfp.get("cadd", {}))

    cadd_phred = None
    if isinstance(cadd_top, dict):
        cadd_phred = _extract_float(cadd_top.get("phred"))
    if cadd_phred is None and isinstance(cadd_dbnsfp, dict):
        cadd_phred = _extract_float(cadd_dbnsfp.get("phred"))

    # REVEL
    revel_data = _unwrap(dbnsfp.get("revel", {}))
    revel = _extract_float(
        revel_data.get("score") if isinstance(revel_data, dict) else revel_data
    )

    # SIFT
    sift_data = _unwrap(dbnsfp.get("sift", {}))
    sift_score = _extract_float(
        sift_data.get("score") if isinstance(sift_data, dict) else sift_data
    )

    # PolyPhen-2 HDIV
    pp2_data = _unwrap(dbnsfp.get("polyphen2", {}))
    if isinstance(pp2_data, dict):
        pp2_hdiv = pp2_data.get("hdiv", {})
        pp2_hdiv = _unwrap(pp2_hdiv)
        polyphen2 = _extract_float(
            pp2_hdiv.get("score") if isinstance(pp2_hdiv, dict) else pp2_hdiv
        )
    else:
        polyphen2 = None

    # gnomAD exome AF
    gnomad = data.get("gnomad_exome", {})
    gnomad_af = None
    if isinstance(gnomad, dict):
        af_data = gnomad.get("af", {})
        if isinstance(af_data, dict):
            gnomad_af = _extract_float(af_data.get("af"))
        else:
            gnomad_af = _extract_float(af_data)

    # ClinVar
    clinvar = data.get("clinvar", {})
    clinvar_sig = None
    clinvar_review = None
    clinvar_id = None
    if isinstance(clinvar, dict):
        rcv = clinvar.get("rcv", {})
        if isinstance(rcv, list):
            rcv = rcv[0] if rcv else {}
        if isinstance(rcv, dict):
            clinvar_sig = rcv.get("clinical_significance")
        clinvar_id = clinvar.get("variant_id")
        clinvar_review = clinvar.get("review_status") or (
            rcv.get("review_status") if isinstance(rcv, dict) else None
        )

    return {
        "cadd_phred": cadd_phred,
        "revel": revel,
        "alphamissense": alphamissense,
        "eve_score": eve_score,
        "sift_score": sift_score,
        "polyphen2_score": polyphen2,
        "gnomad_af": gnomad_af,
        "clinvar_classification": clinvar_sig,
        "clinvar_review_status": clinvar_review,
        "clinvar_id": clinvar_id,
    }


def find_stored_variant(
    variants: list[dict], gene: str, hgvsp: str
) -> dict | None:
    """Find the first matching variant in benchmark_results.json."""
    for v in variants:
        if v["gene"] == gene and v["hgvsp"] == hgvsp:
            return v
    return None


def compare_float(
    field: str, stored: Any, fresh: Any, tolerance: float = 0.01
) -> ScoreComparison:
    """Compare two float values within tolerance."""
    if stored is None and fresh is None:
        return ScoreComparison(field, stored, fresh, True, "both null")
    if stored is None and fresh is not None:
        return ScoreComparison(
            field, stored, fresh, False, "STORED=null but FRESH has value"
        )
    if stored is not None and fresh is None:
        return ScoreComparison(
            field, stored, fresh, False, "STORED has value but FRESH=null"
        )
    try:
        diff = abs(float(stored) - float(fresh))
        if diff <= tolerance:
            return ScoreComparison(field, stored, fresh, True, f"delta={diff:.6f}")
        return ScoreComparison(
            field,
            stored,
            fresh,
            False,
            f"DISCREPANCY delta={diff:.6f} > tol={tolerance}",
        )
    except (ValueError, TypeError):
        eq = str(stored) == str(fresh)
        return ScoreComparison(field, stored, fresh, eq, "string compare")


def compare_string(field: str, stored: Any, fresh: Any) -> ScoreComparison:
    """Compare two string/categorical values."""
    if stored is None and fresh is None:
        return ScoreComparison(field, stored, fresh, True, "both null")
    if stored is None and fresh is not None:
        return ScoreComparison(
            field, stored, fresh, False, "STORED=null but FRESH has value"
        )
    if stored is not None and fresh is None:
        return ScoreComparison(
            field, stored, fresh, False, "STORED has value but FRESH=null"
        )
    eq = str(stored).strip().lower() == str(fresh).strip().lower()
    return ScoreComparison(
        field, stored, fresh, eq, "match" if eq else "DISCREPANCY"
    )


def main() -> None:
    log.info("Loading benchmark_results.json")
    with open(BENCHMARK_RESULTS) as f:
        bench = json.load(f)
    variants = bench["variants"]
    log.info("Loaded %d variants from benchmark", len(variants))

    results: list[dict[str, Any]] = []
    total_checks = 0
    total_discrepancies = 0

    for target in TARGETS:
        gene = target["gene"]
        hgvsp = target["hgvsp"]
        hg19_id = build_hg19_id(
            target["chrom"], target["pos"], target["ref"], target["alt"]
        )
        label = f"{gene} {hgvsp}"
        log.info("--- %s (%s) ---", label, hg19_id)

        # Find stored
        stored_v = find_stored_variant(variants, gene, hgvsp)
        if stored_v is None:
            log.warning("  NOT FOUND in benchmark_results.json: %s", label)
            results.append(
                {
                    "variant": label,
                    "hg19_id": hg19_id,
                    "status": "NOT_FOUND_IN_BENCHMARK",
                    "comparisons": [],
                }
            )
            continue

        stored_scores = stored_v.get("scores", {})
        stored_clinvar = stored_v.get("clinvar", {})

        # Fresh query
        log.info("  Querying myvariant.info: %s", hg19_id)
        raw_response = query_myvariant(hg19_id)
        if not raw_response:
            log.error("  Empty response from myvariant.info for %s", hg19_id)
            results.append(
                {
                    "variant": label,
                    "hg19_id": hg19_id,
                    "status": "API_ERROR",
                    "comparisons": [],
                }
            )
            continue

        fresh = extract_scores(raw_response)

        # Compare numeric scores
        comparisons: list[ScoreComparison] = []
        for field in [
            "cadd_phred",
            "revel",
            "alphamissense",
            "sift_score",
            "polyphen2_score",
            "gnomad_af",
        ]:
            stored_val = stored_scores.get(field)
            fresh_val = fresh.get(field)
            # gnomad_af uses tighter tolerance
            tol = 1e-6 if field == "gnomad_af" else 0.01
            comp = compare_float(field, stored_val, fresh_val, tolerance=tol)
            comparisons.append(comp)
            total_checks += 1
            if not comp.match:
                total_discrepancies += 1
                log.warning(
                    "  DISCREPANCY %s: stored=%s fresh=%s (%s)",
                    field,
                    stored_val,
                    fresh_val,
                    comp.note,
                )
            else:
                log.info("  OK %s: stored=%s fresh=%s", field, stored_val, fresh_val)

        # Compare eve_score (not in stored scores, but useful to report)
        eve_comp = compare_float(
            "eve_score", stored_scores.get("eve_score"), fresh.get("eve_score")
        )
        comparisons.append(eve_comp)
        total_checks += 1
        if not eve_comp.match:
            total_discrepancies += 1

        # Compare ClinVar classification
        stored_clinvar_class = stored_clinvar.get("classification")
        fresh_clinvar_class = fresh.get("clinvar_classification")
        clinvar_comp = compare_string(
            "clinvar_classification", stored_clinvar_class, fresh_clinvar_class
        )
        comparisons.append(clinvar_comp)
        total_checks += 1
        if not clinvar_comp.match:
            total_discrepancies += 1
            log.warning(
                "  DISCREPANCY clinvar: stored=%s fresh=%s",
                stored_clinvar_class,
                fresh_clinvar_class,
            )

        results.append(
            {
                "variant": label,
                "hg19_id": hg19_id,
                "status": "COMPARED",
                "comparisons": [
                    {
                        "field": c.field,
                        "stored": c.stored,
                        "fresh": c.fresh,
                        "match": c.match,
                        "note": c.note,
                    }
                    for c in comparisons
                ],
                "raw_response_keys": list(raw_response.keys()),
            }
        )
        time.sleep(0.5)  # Rate limit

    # Generate markdown report
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# QA Manual Cross-Check: myvariant.info Re-Query",
        "",
        f"**Generated:** {now}",
        f"**Method:** Re-queried myvariant.info with fields={MYVARIANT_FIELDS}",
        f"**Variants checked:** {len(TARGETS)}",
        f"**Total field comparisons:** {total_checks}",
        f"**Discrepancies found:** {total_discrepancies}",
        "",
        "## Summary",
        "",
    ]

    if total_discrepancies == 0:
        lines.append(
            "All stored scores match fresh myvariant.info queries. "
            "No discrepancies detected."
        )
    else:
        lines.append(
            f"**{total_discrepancies} discrepancies** detected across "
            f"{total_checks} field comparisons."
        )
    lines.append("")

    # Summary table
    lines.append("## Per-Variant Results")
    lines.append("")
    lines.append(
        "| Variant | hg19 ID | Fields Checked | Matches | Discrepancies | Status |"
    )
    lines.append(
        "|---------|---------|----------------|---------|---------------|--------|"
    )
    for r in results:
        if r["status"] != "COMPARED":
            lines.append(
                f"| {r['variant']} | `{r['hg19_id']}` | - | - | - | {r['status']} |"
            )
            continue
        n_match = sum(1 for c in r["comparisons"] if c["match"])
        n_disc = sum(1 for c in r["comparisons"] if not c["match"])
        n_total = len(r["comparisons"])
        status = "PASS" if n_disc == 0 else f"**{n_disc} FLAGGED**"
        lines.append(
            f"| {r['variant']} | `{r['hg19_id']}` | {n_total} | {n_match} | {n_disc} | {status} |"
        )

    lines.append("")
    lines.append("## Detailed Comparisons")
    lines.append("")

    for r in results:
        if r["status"] != "COMPARED":
            continue
        lines.append(f"### {r['variant']} (`{r['hg19_id']}`)")
        lines.append("")
        lines.append(
            "| Field | Stored | Fresh | Match | Note |"
        )
        lines.append(
            "|-------|--------|-------|-------|------|"
        )
        for c in r["comparisons"]:
            stored_str = f"{c['stored']}" if c["stored"] is not None else "null"
            fresh_str = f"{c['fresh']}" if c["fresh"] is not None else "null"
            match_str = "YES" if c["match"] else "**NO**"
            lines.append(
                f"| {c['field']} | {stored_str} | {fresh_str} | {match_str} | {c['note']} |"
            )
        lines.append("")

    # Discrepancy details
    disc_variants = [
        r for r in results if any(not c["match"] for c in r.get("comparisons", []))
    ]
    if disc_variants:
        lines.append("## Discrepancy Analysis")
        lines.append("")
        for r in disc_variants:
            lines.append(f"### {r['variant']}")
            lines.append("")
            for c in r["comparisons"]:
                if not c["match"]:
                    lines.append(
                        f"- **{c['field']}**: stored={c['stored']}, "
                        f"fresh={c['fresh']} -- {c['note']}"
                    )
            lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "1. Loaded `benchmark_results.json` containing scores from the benchmark pipeline"
    )
    lines.append(
        "2. Selected 10 well-characterized missense hotspot variants across 10 genes"
    )
    lines.append(
        "3. Constructed hg19 (GRCh37) variant IDs from GENIE genomic coordinates"
    )
    lines.append(
        f"4. Re-queried myvariant.info API with fields={MYVARIANT_FIELDS}"
    )
    lines.append(
        "5. Compared 8 scores per variant: CADD, REVEL, AlphaMissense, EVE, SIFT, PolyPhen-2, gnomAD AF, ClinVar"
    )
    lines.append(
        "6. Numeric tolerance: 0.01 for scores, 1e-6 for gnomAD AF; exact match for ClinVar classification"
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Discrepancies may arise from: (a) myvariant.info database updates between benchmark run "
        "and this cross-check, (b) differences in field extraction logic (list unwrapping, "
        "transcript selection), (c) actual scoring errors in the benchmark pipeline."
    )

    report = "\n".join(lines) + "\n"
    OUTPUT_MD.write_text(report)
    log.info("Report written to %s", OUTPUT_MD)

    # Also save raw JSON for audit
    raw_out = BENCHMARK_DIR / "qa_manual_crosscheck.json"
    with open(raw_out, "w") as f:
        json.dump(
            {
                "metadata": {
                    "generated": now,
                    "total_checks": total_checks,
                    "total_discrepancies": total_discrepancies,
                    "myvariant_fields": MYVARIANT_FIELDS,
                },
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    log.info("Raw JSON written to %s", raw_out)

    # Exit code
    if total_discrepancies > 0:
        log.warning(
            "RESULT: %d/%d discrepancies found", total_discrepancies, total_checks
        )
    else:
        log.info("RESULT: All %d checks passed -- zero discrepancies", total_checks)


if __name__ == "__main__":
    main()
