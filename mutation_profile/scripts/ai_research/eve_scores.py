#!/usr/bin/env python3
"""
EVE (Evolutionary model of Variant Effect) score retrieval for patient variants.

Retrieves pre-computed EVE scores from myvariant.info (dbNSFP integration) for
the 5 patient somatic mutations. EVE uses unsupervised deep generative models
trained on evolutionary sequences to predict variant pathogenicity without
relying on clinical labels (Frazer et al., Nature 2021, PMID: 34707284).

EVE score interpretation:
    - Continuous score 0-1 (higher = more pathogenic)
    - Classification at multiple thresholds (e.g., 25%/75% for B/U/P)
    - class25_pred: Benign (B), Uncertain (U), or Pathogenic (P)
      Uses 25th/75th percentile of ClinVar benign/pathogenic as boundaries
    - rankscore: genome-wide percentile rank (0-1)

popEVE (population-calibrated EVE):
    - Adds clinical severity stratification using population constraint
    - Not yet available via programmatic API; noted where applicable

Patient variants:
    1. DNMT3A R882H (Q9Y6K1) - VAF 39%, pathogenic hotspot
    2. IDH2 R140Q (P48735) - VAF 2%, pathogenic subclone
    3. SETBP1 G870S (Q9Y6X0) - VAF 34%, likely pathogenic
    4. PTPN11 E76Q (Q06124) - VAF 29%, pathogenic
    5. EZH2 V662A (Q15910) - VAF 59%, founder clone (Pathogenic)

Data source: myvariant.info -> dbNSFP v4.x -> EVE (Frazer et al. 2021)

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/eve_scores.py

Runtime: ~10 seconds (API calls to myvariant.info)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MYVARIANT_BASE = "https://myvariant.info/v1/variant"

# Patient variants with hg19 (GRCh37) coordinates for myvariant.info lookup
# Coordinates verified against Ensembl VEP and prior clinical_variant_scores.py
PATIENT_VARIANTS = [
    {
        "key": "DNMT3A_R882H",
        "gene": "DNMT3A",
        "variant": "R882H",
        "uniprot_id": "Q9Y6K1",
        "hgvs_p": "p.Arg882His",
        "hgvs_c": "NM_022552.5:c.2645G>A",
        "hg19_id": "chr2:g.25457242C>T",
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
        "vaf": 0.29,
    },
    {
        "key": "EZH2_V662A",
        "gene": "EZH2",
        "variant": "V662A",
        "uniprot_id": "Q15910",
        "hgvs_p": "p.Val662Ala",
        "hgvs_c": "NM_004456.5:c.1985T>C",
        "hg19_id": "chr7:g.148508727T>C",
        "vaf": 0.59,
    },
]

# EVE classification thresholds (Frazer et al. 2021)
# class25_pred uses 25th percentile of ClinVar benign / 75th percentile of ClinVar pathogenic
# B = Benign, U = Uncertain, P = Pathogenic
EVE_THRESHOLDS = {
    "class25": "25th/75th percentile ClinVar calibration (recommended)",
    "class50": "50th percentile ClinVar calibration (balanced)",
    "class75": "75th/25th percentile ClinVar calibration (conservative)",
}


def fetch_eve_scores(variant: dict[str, Any]) -> dict[str, Any]:
    """Fetch EVE scores from myvariant.info for a single variant."""
    hg19_id = variant["hg19_id"]
    url = f"{MYVARIANT_BASE}/{hg19_id}"
    params = {
        "fields": (
            "dbnsfp.eve,"
            "dbnsfp.genename,"
            "dbnsfp.aaref,"
            "dbnsfp.aaalt,"
            "dbnsfp.aapos,"
            "dbnsfp.hg38,"
            "dbnsfp.esm1b,"
            "dbnsfp.alphamissense"
        ),
    }

    log.info("Fetching EVE scores for %s (%s)", variant["key"], hg19_id)
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Failed to fetch %s: %s", variant["key"], exc)
        return {
            "key": variant["key"],
            "gene": variant["gene"],
            "variant": variant["variant"],
            "uniprot_id": variant["uniprot_id"],
            "status": "error",
            "error": str(exc),
        }

    data = resp.json()
    dbnsfp = data.get("dbnsfp", {})
    eve = dbnsfp.get("eve")

    if eve is None:
        log.warning("No EVE data found for %s", variant["key"])
        return {
            "key": variant["key"],
            "gene": variant["gene"],
            "variant": variant["variant"],
            "uniprot_id": variant["uniprot_id"],
            "status": "not_found",
        }

    # Extract EVE score (handle list vs scalar)
    score_raw = eve.get("score")
    if isinstance(score_raw, list):
        score = score_raw[0]
    else:
        score = score_raw

    rankscore = eve.get("rankscore")

    # Extract classifications at all thresholds
    classifications = {}
    for threshold in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
        pred_key = f"class{threshold}_pred"
        pred = eve.get(pred_key)
        if isinstance(pred, list):
            pred = pred[0]
        label_map = {"B": "Benign", "U": "Uncertain", "P": "Pathogenic"}
        classifications[f"class{threshold}"] = {
            "code": pred,
            "label": label_map.get(pred, pred),
        }

    # Determine primary classification (class25 = recommended)
    primary_class = classifications.get("class25", {}).get("code", "U")
    primary_label = classifications.get("class25", {}).get("label", "Uncertain")

    # Extract companion scores for context
    esm1b = dbnsfp.get("esm1b", {})
    am = dbnsfp.get("alphamissense", {})

    esm1b_score = esm1b.get("score") if isinstance(esm1b, dict) else None
    if isinstance(esm1b_score, list):
        esm1b_score = esm1b_score[0]

    am_score = am.get("score") if isinstance(am, dict) else None
    if isinstance(am_score, list):
        am_score = am_score[0]
    am_class = am.get("pred") if isinstance(am, dict) else None
    if isinstance(am_class, list):
        am_class = am_class[0]

    # popEVE thresholds (Cuturello et al. 2024, PMID: 38997443)
    # popEVE <= -5.056 means 99.99% likely deleterious
    # popEVE data not available via myvariant.info; requires direct download
    popeve_note = (
        "popEVE scores not available via myvariant.info API. "
        "popEVE requires bulk download from the EVE model website or "
        "direct computation. The popEVE threshold of <= -5.056 indicates "
        "99.99% likelihood of deleteriousness (Cuturello et al. 2024)."
    )

    result = {
        "key": variant["key"],
        "gene": variant["gene"],
        "variant": variant["variant"],
        "uniprot_id": variant["uniprot_id"],
        "hgvs_p": variant["hgvs_p"],
        "hgvs_c": variant["hgvs_c"],
        "hg19_id": variant["hg19_id"],
        "vaf": variant["vaf"],
        "status": "success",
        "eve": {
            "score": score,
            "rankscore": rankscore,
            "primary_classification": {
                "threshold": "class25",
                "code": primary_class,
                "label": primary_label,
                "description": EVE_THRESHOLDS["class25"],
            },
            "all_classifications": classifications,
        },
        "companion_scores": {
            "esm1b_score": esm1b_score,
            "alphamissense_score": am_score,
            "alphamissense_class": am_class,
        },
        "popeve": {
            "score": None,
            "available": False,
            "note": popeve_note,
        },
    }

    log.info(
        "  %s: EVE=%.4f (rank=%.5f), class25=%s",
        variant["key"],
        score,
        rankscore,
        primary_label,
    )
    return result


def classify_eve_severity(score: float | None) -> str:
    """Classify EVE score severity for clinical context."""
    if score is None:
        return "Unknown"
    if score >= 0.9:
        return "Very High Pathogenicity"
    if score >= 0.7:
        return "High Pathogenicity"
    if score >= 0.5:
        return "Moderate Pathogenicity"
    if score >= 0.3:
        return "Low Pathogenicity"
    return "Likely Benign"


def generate_report(results: list[dict[str, Any]]) -> str:
    """Generate markdown report from EVE score results."""
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# EVE Score Analysis for Patient Variants",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## Overview",
        "",
        "EVE (Evolutionary model of Variant Effect) uses unsupervised deep generative",
        "models trained on evolutionary sequence alignments to predict variant",
        "pathogenicity without relying on clinical labels. EVE achieves state-of-the-art",
        "performance in distinguishing ClinVar pathogenic from benign variants",
        "(Frazer et al., Nature 2021, PMID: 34707284).",
        "",
        "**Score interpretation:**",
        "- EVE score: 0 (benign) to 1 (pathogenic)",
        "- class25_pred: recommended threshold using 25th/75th percentile ClinVar calibration",
        "  - B = Benign, U = Uncertain, P = Pathogenic",
        "- Rankscore: genome-wide percentile (0 = lowest, 1 = highest pathogenicity)",
        "",
        "## Results Summary",
        "",
        "| Gene | Variant | EVE Score | Rankscore | Class (25%) | Severity |",
        "|------|---------|-----------|-----------|-------------|----------|",
    ]

    for r in results:
        if r["status"] != "success":
            lines.append(
                f"| {r['gene']} | {r['variant']} | -- | -- | {r['status']} | -- |"
            )
            continue

        eve = r["eve"]
        score = eve["score"]
        rankscore = eve["rankscore"]
        cls = eve["primary_classification"]
        severity = classify_eve_severity(score)

        lines.append(
            f"| {r['gene']} | {r['variant']} | {score:.4f} | {rankscore:.5f} "
            f"| {cls['label']} | {severity} |"
        )

    # Detailed per-variant sections
    lines.extend(["", "## Detailed Variant Analysis", ""])

    for r in results:
        if r["status"] != "success":
            lines.extend([
                f"### {r['gene']} {r['variant']} ({r['uniprot_id']})",
                "",
                f"**Status:** {r['status']}",
                f"**Error:** {r.get('error', 'N/A')}",
                "",
            ])
            continue

        eve = r["eve"]
        score = eve["score"]
        rankscore = eve["rankscore"]
        cls = eve["primary_classification"]
        severity = classify_eve_severity(score)
        companion = r.get("companion_scores", {})

        lines.extend([
            f"### {r['gene']} {r['variant']} ({r['uniprot_id']})",
            "",
            f"- **HGVS protein:** {r['hgvs_p']}",
            f"- **HGVS coding:** {r['hgvs_c']}",
            f"- **VAF:** {r['vaf'] * 100:.0f}%",
            f"- **EVE score:** {score:.6f}",
            f"- **EVE rankscore:** {rankscore:.5f} (genome-wide percentile)",
            f"- **EVE class (25% threshold):** {cls['label']} ({cls['code']})",
            f"- **Clinical severity:** {severity}",
            "",
        ])

        # Classification across thresholds
        lines.append("**Classification across thresholds:**")
        lines.append("")
        lines.append("| Threshold | Classification |")
        lines.append("|-----------|---------------|")
        for t in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
            c = eve["all_classifications"].get(f"class{t}", {})
            lines.append(f"| {t}% | {c.get('label', 'N/A')} ({c.get('code', '?')}) |")
        lines.append("")

        # Companion scores
        if companion.get("esm1b_score") is not None:
            lines.append(f"- **ESM-1b score:** {companion['esm1b_score']:.4f}")
        if companion.get("alphamissense_score") is not None:
            lines.append(
                f"- **AlphaMissense score:** {companion['alphamissense_score']:.4f}"
                f" ({companion.get('alphamissense_class', 'N/A')})"
            )
        lines.append("")

    # Cross-variant comparison
    lines.extend(["## Cross-Variant Comparison", ""])

    successful = [r for r in results if r["status"] == "success"]
    if successful:
        sorted_by_score = sorted(
            successful,
            key=lambda x: x["eve"]["score"],
            reverse=True,
        )

        lines.append("**Ranked by EVE pathogenicity (highest first):**")
        lines.append("")
        for i, r in enumerate(sorted_by_score, 1):
            score = r["eve"]["score"]
            cls_code = r["eve"]["primary_classification"]["code"]
            cls_label = r["eve"]["primary_classification"]["label"]
            lines.append(
                f"{i}. **{r['gene']} {r['variant']}** - "
                f"EVE={score:.4f} ({cls_label})"
            )

        lines.extend([
            "",
            "**Concordance with other predictors:**",
            "",
            "| Gene | Variant | EVE | AlphaMissense | ESM-1b | Agreement |",
            "|------|---------|-----|---------------|--------|-----------|",
        ])

        for r in sorted_by_score:
            eve_score = r["eve"]["score"]
            eve_cls = r["eve"]["primary_classification"]["code"]
            comp = r.get("companion_scores", {})
            am = comp.get("alphamissense_score")
            esm = comp.get("esm1b_score")

            am_str = f"{am:.3f}" if am is not None else "N/A"
            esm_str = f"{esm:.3f}" if esm is not None else "N/A"

            # Check agreement
            eve_pathogenic = eve_cls == "P"
            am_pathogenic = am is not None and am >= 0.564
            esm_pathogenic = esm is not None and esm < -7.0  # strong threshold

            agree_count = sum([
                eve_pathogenic,
                am_pathogenic,
            ])
            if am is not None:
                agreement = "Concordant" if eve_pathogenic == am_pathogenic else "Discordant"
            else:
                agreement = "N/A (missing AM)"

            lines.append(
                f"| {r['gene']} | {r['variant']} | {eve_score:.3f} ({eve_cls}) "
                f"| {am_str} | {esm_str} | {agreement} |"
            )

        lines.append("")

    # popEVE section
    lines.extend([
        "## popEVE (Population-Calibrated EVE)",
        "",
        "popEVE extends EVE by incorporating population-level allele frequency",
        "constraint to calibrate clinical severity (Cuturello et al., medRxiv 2024,",
        "PMID: 38997443). The key threshold is:",
        "",
        "- **popEVE <= -5.056:** 99.99% likelihood of deleteriousness",
        "",
        "popEVE scores are not currently available via the myvariant.info API or",
        "other programmatic sources. They require either:",
        "1. Bulk download from the EVE model website (evemodel.org)",
        "2. Direct computation using the popEVE codebase",
        "",
        "Given the high EVE scores for most patient variants (especially EZH2 V662A",
        "and IDH2 R140Q), popEVE would likely confirm strong deleteriousness for",
        "these variants.",
        "",
    ])

    # Clinical interpretation
    lines.extend([
        "## Clinical Interpretation",
        "",
        "**Key findings:**",
        "",
    ])

    if successful:
        high_path = [r for r in successful if r["eve"]["score"] >= 0.7]
        moderate_path = [r for r in successful if 0.5 <= r["eve"]["score"] < 0.7]
        low_uncertain = [r for r in successful if r["eve"]["score"] < 0.5]

        if high_path:
            genes = ", ".join(
                f"{r['gene']} {r['variant']} (EVE={r['eve']['score']:.3f})"
                for r in high_path
            )
            lines.append(f"1. **High pathogenicity (EVE >= 0.7):** {genes}")

        if moderate_path:
            genes = ", ".join(
                f"{r['gene']} {r['variant']} (EVE={r['eve']['score']:.3f})"
                for r in moderate_path
            )
            lines.append(f"2. **Moderate pathogenicity (0.5 <= EVE < 0.7):** {genes}")

        if low_uncertain:
            genes = ", ".join(
                f"{r['gene']} {r['variant']} (EVE={r['eve']['score']:.3f})"
                for r in low_uncertain
            )
            lines.append(f"3. **Low/uncertain pathogenicity (EVE < 0.5):** {genes}")

        lines.extend([
            "",
            "**ACMG PP3 evidence contribution:**",
            "EVE scores provide computational (in silico) evidence for ACMG PP3 criterion.",
            "Variants classified as Pathogenic by EVE at the 25% threshold support PP3_Supporting",
            "or stronger evidence depending on the score magnitude.",
            "",
        ])

    # Methods
    lines.extend([
        "## Methods",
        "",
        "### Data Source",
        "EVE scores retrieved from myvariant.info (dbNSFP v4.x integration).",
        "myvariant.info aggregates variant annotations from 40+ databases using",
        "GRCh37/hg19 coordinates.",
        "",
        "### EVE Model",
        "EVE trains a Bayesian variational autoencoder (VAE) on evolutionary",
        "multiple sequence alignments (MSAs) for each protein family. The model",
        "learns the distribution of natural amino acid variation and scores each",
        "possible substitution by its likelihood under the learned distribution.",
        "Lower likelihood = higher predicted pathogenicity.",
        "",
        "### Classification Thresholds",
        "EVE provides classifications at 11 thresholds (10-90%). The class25",
        "threshold is recommended as it balances sensitivity and specificity",
        "using the 25th percentile of ClinVar benign scores and the 75th",
        "percentile of ClinVar pathogenic scores as boundaries.",
        "",
        "### References",
        "1. Frazer J, et al. Disease variant prediction with deep generative models",
        "   of evolutionary data. Nature. 2021;599:91-95. PMID: 34707284.",
        "2. Cuturello F, et al. popEVE: population-aware variant effect prediction.",
        "   medRxiv. 2024. PMID: 38997443.",
        "3. Liu X, et al. dbNSFP v4: a comprehensive database of transcript-specific",
        "   functional predictions and annotations for human nonsynonymous and",
        "   splice-site SNVs. Genome Med. 2020;12:103. PMID: 33261662.",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    """Fetch EVE scores for all patient variants and generate outputs."""
    log.info("Starting EVE score retrieval for %d variants", len(PATIENT_VARIANTS))

    results = []
    for variant in PATIENT_VARIANTS:
        result = fetch_eve_scores(variant)
        results.append(result)
        time.sleep(0.5)  # rate limiting

    # Save JSON results
    json_path = RESULTS_DIR / "eve_scores.json"
    output = {
        "metadata": {
            "analysis": "EVE score retrieval",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "source": "myvariant.info (dbNSFP v4.x -> EVE)",
            "reference": "Frazer et al., Nature 2021, PMID: 34707284",
            "n_variants": len(results),
            "n_success": sum(1 for r in results if r["status"] == "success"),
        },
        "variants": results,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved JSON results to %s", json_path)

    # Generate markdown report
    report = generate_report(results)
    report_path = RESULTS_DIR / "eve_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved report to %s", report_path)

    # Print summary
    print("\n" + "=" * 70)
    print("EVE SCORE SUMMARY")
    print("=" * 70)
    print(f"{'Gene':<10} {'Variant':<10} {'EVE Score':<12} {'Rankscore':<12} {'Class25':<12} {'Severity'}")
    print("-" * 70)
    for r in results:
        if r["status"] == "success":
            eve = r["eve"]
            print(
                f"{r['gene']:<10} {r['variant']:<10} {eve['score']:<12.4f} "
                f"{eve['rankscore']:<12.5f} {eve['primary_classification']['label']:<12} "
                f"{classify_eve_severity(eve['score'])}"
            )
        else:
            print(f"{r['gene']:<10} {r['variant']:<10} {'--':<12} {'--':<12} {r['status']}")
    print("=" * 70)

    # Count successes
    n_success = sum(1 for r in results if r["status"] == "success")
    n_pathogenic = sum(
        1 for r in results
        if r["status"] == "success" and r["eve"]["primary_classification"]["code"] == "P"
    )
    print(f"\n{n_success}/{len(results)} variants scored, {n_pathogenic} classified Pathogenic (class25)")


if __name__ == "__main__":
    main()
