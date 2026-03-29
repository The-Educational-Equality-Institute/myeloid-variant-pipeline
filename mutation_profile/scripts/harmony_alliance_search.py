#!/usr/bin/env python3
"""
harmony_alliance_search.py -- Document HARMONY Alliance query results and
attempt to access public data exploration tools.

HARMONY Alliance is a European big data platform for hematological malignancies
with 120,000+ patient datasets (2,796 AML with detailed molecular data).

Their 41-gene panel covers ALL 4 target genes:
    DNMT3A, IDH2, SETBP1, PTPN11

Access model:
    - Spotfire dashboards (AML, ALL) at https://spotfire.harmony-platform.eu/
    - These require a web browser (JavaScript/Spotfire WebPlayer)
    - No REST API for programmatic queries
    - Research questions can be submitted to office@harmony-alliance.eu

This script documents what we know from published HARMONY studies and
attempts to access any machine-readable endpoints.

Inputs:
    - Remote web pages (HARMONY platform, publications)

Outputs:
    - mutation_profile/results/cross_database/harmony_alliance_results.json
    - mutation_profile/results/cross_database/harmony_alliance_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/harmony_alliance_search.py

Runtime: ~15 seconds
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
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_GENES = ["DNMT3A", "IDH2", "SETBP1", "PTPN11"]
EXACT_VARIANTS = {
    "DNMT3A": "R882H",
    "IDH2": "R140Q",
    "SETBP1": "G870S",
    "PTPN11": "E76Q",
}

# HARMONY 41-gene panel (verified from Eurofins-Biomnis panel spec + ASH publications)
HARMONY_41_GENE_PANEL = [
    "ASXL1", "BCOR", "BCORL1", "BRAF", "CALR", "CBL", "CEBPA", "CSF3R",
    "DNMT3A", "ETNK1", "ETV6", "EZH2", "FLT3", "GATA2", "GNB1", "HRAS",
    "IDH1", "IDH2", "JAK2", "KIT", "KMT2A", "KRAS", "MPL", "NF1",
    "NPM1", "NRAS", "PHF6", "PPM1D", "PRPF8", "PTPN11", "RUNX1",
    "SETBP1", "SF3B1", "SRSF2", "STAG2", "TET2", "TP53", "UBA1",
    "U2AF1", "WT1", "ZRSR2",
]

REQUEST_DELAY = 2
REQUEST_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "harmony_alliance_search.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "mrna-hematology-research/1.0",
})


def check_panel_coverage() -> dict[str, Any]:
    """Verify that all 4 target genes are on the HARMONY 41-gene panel."""
    log.info("=" * 60)
    log.info("1. Panel Coverage Verification")
    log.info("=" * 60)

    coverage = {}
    all_covered = True
    for gene in TARGET_GENES:
        on_panel = gene in HARMONY_41_GENE_PANEL
        coverage[gene] = on_panel
        if on_panel:
            log.info("  %s: ON PANEL (41-gene)", gene)
        else:
            log.warning("  %s: NOT on panel", gene)
            all_covered = False

    return {
        "panel_size": len(HARMONY_41_GENE_PANEL),
        "all_target_genes_covered": all_covered,
        "gene_coverage": coverage,
        "panel_genes": HARMONY_41_GENE_PANEL,
    }


def check_spotfire_access() -> dict[str, Any]:
    """Try to access the Spotfire exploration dashboards."""
    log.info("=" * 60)
    log.info("2. Spotfire Dashboard Access Check")
    log.info("=" * 60)

    dashboards = {
        "AML": "https://spotfire.harmony-platform.eu/spotfire/wp/analysis?file=/Anonymous/AML",
        "ALL": "https://spotfire.harmony-platform.eu/spotfire/wp/analysis?file=/Anonymous/ALL",
    }

    results = {}
    for name, url in dashboards.items():
        time.sleep(REQUEST_DELAY)
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            results[name] = {
                "url": url,
                "status_code": resp.status_code,
                "accessible": resp.status_code == 200,
                "content_type": resp.headers.get("Content-Type", ""),
                "note": "Spotfire WebPlayer requires JavaScript browser — not queryable via REST API",
            }
            log.info("  %s dashboard: HTTP %d (%s)", name, resp.status_code, resp.headers.get("Content-Type", ""))
        except requests.exceptions.RequestException as e:
            results[name] = {
                "url": url,
                "status_code": None,
                "accessible": False,
                "error": str(e),
            }
            log.warning("  %s dashboard failed: %s", name, e)

    return results


def check_outcome_predictor() -> dict[str, Any]:
    """Check the AML outcome predictor tool."""
    log.info("=" * 60)
    log.info("3. AML Outcome Predictor")
    log.info("=" * 60)

    url = "https://taxonomy.harmony-platform.eu/AML_outcome_predictor/"
    time.sleep(REQUEST_DELAY)
    try:
        resp = SESSION.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        result = {
            "url": url,
            "status_code": resp.status_code,
            "accessible": resp.status_code == 200,
            "content_type": resp.headers.get("Content-Type", ""),
        }
        log.info("  Outcome predictor: HTTP %d", resp.status_code)
    except requests.exceptions.RequestException as e:
        result = {
            "url": url,
            "accessible": False,
            "error": str(e),
        }
        log.warning("  Outcome predictor failed: %s", e)

    return result


def compile_published_data() -> dict[str, Any]:
    """Compile mutation frequency data from published HARMONY studies."""
    log.info("=" * 60)
    log.info("4. Published HARMONY Mutation Data")
    log.info("=" * 60)

    # Data from published HARMONY Alliance studies (ASH abstracts, Blood, etc.)
    published = {
        "source": "HARMONY Alliance published studies (ASH 2021, Blood 2024)",
        "aml_cohort_size": 2796,
        "note": "2,796 AML patients with detailed molecular data (41-gene panel + cytogenetics)",
        "total_datasets": "120,000+",
        "total_available_for_analysis": "~121,000",
        "gene_frequencies_in_aml": {
            "DNMT3A": {
                "frequency_percent": "~25-30%",
                "note": "53% in NPM1-mutated subset (HARMONY ASH 2021)",
                "estimated_patients": "700-840",
            },
            "IDH2": {
                "frequency_percent": "~12-15%",
                "note": "24% in NPM1-mutated subset",
                "estimated_patients": "336-420",
            },
            "PTPN11": {
                "frequency_percent": "~5-10%",
                "note": "11.4% in NPM1-mutated subset",
                "estimated_patients": "140-280",
            },
            "SETBP1": {
                "frequency_percent": "~1-3%",
                "note": "More common in MDS/MPN than AML; SKI domain hotspot",
                "estimated_patients": "28-84",
            },
        },
        "expected_cooccurrence": {
            "DNMT3A_IDH2": {
                "note": "Well-known co-occurrence pair in AML",
                "estimated_patients": "100-200",
            },
            "DNMT3A_IDH2_PTPN11": {
                "note": "Triple combination; each gene independently mutated",
                "estimated_patients": "5-20",
            },
            "DNMT3A_IDH2_PTPN11_SETBP1": {
                "note": "Quadruple; extremely rare given SETBP1 rarity in AML",
                "estimated_patients": "0-1",
            },
        },
    }

    log.info("  AML cohort: %d patients with 41-gene panel", published["aml_cohort_size"])
    for gene, data in published["gene_frequencies_in_aml"].items():
        log.info("  %s: %s (%s est. patients)", gene, data["frequency_percent"], data["estimated_patients"])

    log.info("  Expected quadruple co-occurrence: %s patients",
             published["expected_cooccurrence"]["DNMT3A_IDH2_PTPN11_SETBP1"]["estimated_patients"])

    return published


def generate_report(results: dict[str, Any]) -> str:
    """Generate markdown report."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    panel = results["panel_coverage"]
    published = results["published_data"]

    report = f"""# HARMONY Alliance Search Report

**Generated:** {ts} UTC
**Database:** HARMONY Alliance (https://www.harmony-alliance.eu/)
**Total datasets:** {published['total_datasets']} blood cancer patients
**AML molecular cohort:** {published['aml_cohort_size']} patients with 41-gene panel
**Panel genes:** {panel['panel_size']} genes
**All target genes on panel:** {'Yes' if panel['all_target_genes_covered'] else 'No'}

---

## 1. Panel Coverage

**Critical finding:** All 4 target genes ARE on the HARMONY 41-gene panel.

| Gene | Variant | On HARMONY Panel | Estimated AML Frequency | Est. Patients (of 2,796) |
|------|---------|-----------------|------------------------|--------------------------|
"""
    for gene in TARGET_GENES:
        variant = EXACT_VARIANTS[gene]
        on_panel = "Yes" if panel["gene_coverage"][gene] else "No"
        freq = published["gene_frequencies_in_aml"][gene]["frequency_percent"]
        est = published["gene_frequencies_in_aml"][gene]["estimated_patients"]
        report += f"| {gene} | {variant} | {on_panel} | {freq} | {est} |\n"

    report += """
**Note:** The project's earlier intelligence report (2026-03-14) incorrectly stated
SETBP1 was NOT on the HARMONY panel. Web search (2026-03-27) confirmed SETBP1 IS
included in the 41-gene panel, verified via Eurofins-Biomnis panel specification and
ASH 2021 abstract data.

---

## 2. Expected Co-occurrence in HARMONY

Based on published gene frequencies in HARMONY's AML cohort:

| Combination | Estimated Patients | Notes |
|------------|-------------------|-------|
"""
    for combo, data in published["expected_cooccurrence"].items():
        report += f"| {combo.replace('_', ' + ')} | {data['estimated_patients']} | {data['note']} |\n"

    report += """
Under an independence model with 2,796 AML patients:
- P(DNMT3A) × P(IDH2) × P(PTPN11) × P(SETBP1) ≈ 0.275 × 0.135 × 0.075 × 0.02 = 0.0000557
- Expected: 2,796 × 0.0000557 ≈ **0.16 patients**

This means finding even **1 patient** with the quadruple in HARMONY would be notable.
The fact that 0 were found in ~33,000 patients across other databases is consistent.

---

## 3. Data Access

### Spotfire Dashboards (Public)

"""
    for name, data in results["spotfire_access"].items():
        status = f"HTTP {data.get('status_code', 'N/A')}"
        report += f"- **{name} Exploration Tool:** [{data['url']}]({data['url']}) — {status}\n"

    report += """
These dashboards are built with Spotfire WebPlayer and require a JavaScript-capable
browser. They allow cohort selection by age, gender, mutations, and cytogenetics.
However, they cannot be queried programmatically via REST API.

**To query HARMONY for our specific variants:**
- Use browser automation (Playwright) to interact with the AML Spotfire dashboard
- OR submit a research question to office@harmony-alliance.eu

### AML Outcome Predictor

"""
    pred = results["outcome_predictor"]
    report += f"- **URL:** [{pred.get('url', 'N/A')}]({pred.get('url', 'N/A')})\n"
    report += f"- **Accessible:** {'Yes' if pred.get('accessible') else 'No'}\n"

    report += """
---

## 4. Action Items

1. **Browser query (immediate):** Use Playwright to navigate the AML Spotfire dashboard
   and select DNMT3A + IDH2 + SETBP1 + PTPN11 mutations to see cohort size.

2. **Email query (high priority):** Update the draft email to HARMONY
   (docs/outreach/DRAFTS/email-harmony.md) to request the FULL quadruple query,
   since SETBP1 IS on the panel. The current draft incorrectly requests only the triple.

3. **Outcome predictor:** Test whether the AML outcome predictor accepts specific
   mutation inputs (could provide survival estimates for our combination).

---

## 5. HARMONY 41-Gene Panel

"""
    for i, gene in enumerate(HARMONY_41_GENE_PANEL):
        marker = " **←**" if gene in TARGET_GENES else ""
        report += f"{i+1}. {gene}{marker}\n"

    report += f"""
---

*Generated by harmony_alliance_search.py | {ts}*
"""
    return report


def main():
    """Run HARMONY Alliance search."""
    start = time.time()
    log.info("HARMONY Alliance Search - starting")
    log.info("Target variants: %s", ", ".join(f"{g} {v}" for g, v in EXACT_VARIANTS.items()))

    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "HARMONY Alliance",
        "url": "https://www.harmony-alliance.eu/",
    }

    results["panel_coverage"] = check_panel_coverage()
    results["spotfire_access"] = check_spotfire_access()
    results["outcome_predictor"] = check_outcome_predictor()
    results["published_data"] = compile_published_data()

    # Save JSON
    json_path = RESULTS_DIR / "harmony_alliance_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("JSON results saved to %s", json_path)

    # Generate report
    report = generate_report(results)
    report_path = RESULTS_DIR / "harmony_alliance_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Report saved to %s", report_path)

    elapsed = time.time() - start
    log.info("Completed in %.1f seconds", elapsed)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("  All 4 genes on panel: %s", results["panel_coverage"]["all_target_genes_covered"])
    log.info("  AML cohort size: %d", results["published_data"]["aml_cohort_size"])
    log.info("  Expected quadruple matches: ~0.16 (under independence)")
    log.info("  Access: Spotfire dashboards (browser) or email query")


if __name__ == "__main__":
    main()
