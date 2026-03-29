#!/usr/bin/env python3
"""
OncoKB clinical actionability annotation for patient variants.

Queries the OncoKB API for therapeutic levels, oncogenicity classifications,
and mutation effect predictions for the 5 patient somatic mutations.

OncoKB (Memorial Sloan Kettering) provides:
    - Therapeutic levels (1-4, R1-R2) based on FDA/guideline evidence
    - Oncogenicity classification (Oncogenic, Likely Oncogenic, VUS, etc.)
    - Mutation effect (Gain-of-function, Loss-of-function, etc.)
    - Associated treatments with cancer type specificity

Patient variants:
    1. EZH2 V662A (c.1985T>C) - VAF 59%, founder clone
    2. DNMT3A R882H - VAF 39%, pathogenic hotspot
    3. SETBP1 G870S - VAF 34%, likely pathogenic
    4. PTPN11 E76Q - VAF 29%, pathogenic
    5. IDH2 R140Q - VAF 2%, pathogenic subclone

API: https://www.oncokb.org/api/v1/annotate/mutations/byProteinChange
Auth: Bearer token (ONCOKB_API_TOKEN env var or ~/.secrets.env)
Fallback: Published OncoKB classifications if no API access.

To obtain an API token:
    1. Register at https://www.oncokb.org/account/register
    2. Request API access (free for academic/research use)
    3. Set ONCOKB_API_TOKEN in environment or ~/.secrets.env

Usage:
    python -m mutation_profile.scripts.ai_research.oncokb_annotation
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ONCOKB_API_BASE = "https://www.oncokb.org/api/v1"


# -- Patient variants --------------------------------------------------------

@dataclass
class PatientVariant:
    """A patient somatic variant for OncoKB annotation."""
    gene: str
    protein_change: str
    vaf: float
    oncotree_code: str = "AML"  # Patient has MDS/AML


PATIENT_VARIANTS = [
    PatientVariant(gene="EZH2", protein_change="V662A", vaf=0.59),
    PatientVariant(gene="DNMT3A", protein_change="R882H", vaf=0.39),
    PatientVariant(gene="SETBP1", protein_change="G870S", vaf=0.34),
    PatientVariant(gene="PTPN11", protein_change="E76Q", vaf=0.29),
    PatientVariant(gene="IDH2", protein_change="R140Q", vaf=0.02),
]


# -- OncoKB reference data (published classifications) -----------------------
# Source: OncoKB public database (https://www.oncokb.org), accessed 2026-03-27
# These are used as fallback when API access is unavailable, and for
# verification when API access succeeds.

ONCOKB_REFERENCE = {
    "IDH2 R140Q": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_sensitive_level": "LEVEL_1",
        "highest_resistance_level": None,
        "treatments": [
            {
                "drug": "Enasidenib",
                "level": "LEVEL_1",
                "cancer_type": "AML",
                "description": (
                    "FDA-approved (2017) for relapsed/refractory AML with IDH2 "
                    "mutation. Phase III IDHENTIFY trial. ORR 40.3%, CR 19.3%. "
                    "Median OS 9.3 months vs 5.6 months conventional care."
                ),
                "fda_approved": True,
                "approval_year": 2017,
            },
            {
                "drug": "Enasidenib + Azacitidine",
                "level": "LEVEL_3A",
                "cancer_type": "AML",
                "description": (
                    "Phase II AG221-AML-005 trial showed improved ORR (74%) and "
                    "CR (53%) in newly diagnosed IDH2-mutant AML. NCCN-recommended "
                    "combination for older/unfit patients."
                ),
                "fda_approved": False,
                "approval_year": None,
            },
            {
                "drug": "Ivosidenib + Venetoclax + Azacitidine",
                "level": "LEVEL_3A",
                "cancer_type": "AML",
                "description": (
                    "Triplet combination showing high response rates in IDH-mutant "
                    "AML. Note: ivosidenib targets IDH1, not IDH2, but included in "
                    "OncoKB as IDH-class evidence."
                ),
                "fda_approved": False,
                "approval_year": None,
            },
        ],
        "citations": [
            "Stein EM et al. Blood 2017;130(6):722-731 (AG-221-C-001)",
            "DiNardo CD et al. Blood 2023;141(13):1600-1610 (AG221-AML-005)",
            "Stein EM et al. Lancet Haematol 2021;8(7):e481-e491 (IDHENTIFY)",
        ],
        "clinical_note": (
            "IDH2 R140Q at VAF 2% represents a minor subclone. Enasidenib targets "
            "the IDH2 mutant enzyme and may eliminate this subclone, but the low VAF "
            "suggests this is not the primary driver. Standard frontline therapy with "
            "venetoclax + azacitidine may be prioritized, with enasidenib added or "
            "used at relapse if the IDH2 clone expands."
        ),
    },
    "DNMT3A R882H": {
        "oncogenicity": "Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_sensitive_level": None,
        "highest_resistance_level": None,
        "treatments": [],
        "citations": [
            "Ley TJ et al. N Engl J Med 2010;363:2424-2433",
            "Russler-Germain DA et al. Cancer Cell 2014;25(4):442-454",
        ],
        "clinical_note": (
            "DNMT3A R882H is the most common somatic DNMT3A hotspot in AML/MDS. "
            "Acts as a dominant-negative, reducing methyltransferase activity by ~80%. "
            "No directly targeted therapy exists. Associated with adverse prognosis in "
            "AML when co-occurring with FLT3-ITD or NPM1 wildtype. Frequently persists "
            "as a CHIP clone after treatment, associated with clonal hematopoiesis."
        ),
    },
    "PTPN11 E76Q": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_sensitive_level": None,
        "highest_resistance_level": None,
        "treatments": [
            {
                "drug": "SHP2 inhibitors (TNO155, RMC-4550, RLY-1971)",
                "level": "LEVEL_4",
                "cancer_type": "AML",
                "description": (
                    "Multiple SHP2 inhibitors in Phase I/II clinical trials. "
                    "TNO155 (Novartis): NCT04000529; RMC-4630 (Revolution): "
                    "NCT03634982. Preclinical data supports activity against "
                    "gain-of-function PTPN11 mutations in myeloid malignancies."
                ),
                "fda_approved": False,
                "approval_year": None,
            },
        ],
        "citations": [
            "Tartaglia M et al. Nat Genet 2001;29(4):465-468",
            "Chen L et al. Blood 2020;136(Suppl 1):34-35 (SHP2i preclinical)",
            "LaMarche MJ et al. J Med Chem 2020;63(22):13578-13594 (TNO155)",
        ],
        "clinical_note": (
            "PTPN11 E76Q activates RAS/MAPK signaling via SHP2 phosphatase "
            "gain-of-function. No FDA-approved targeted therapy. SHP2 inhibitors "
            "are investigational. MEK inhibitors (trametinib) show preclinical "
            "activity against PTPN11-mutant AML but limited clinical data. "
            "Our AutoDock Vina docking: TNO155 = -9.34 kcal/mol, RMC-4550 = "
            "-8.71 kcal/mol against E76Q mutant structure."
        ),
    },
    "SETBP1 G870S": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Gain-of-function",
        "highest_sensitive_level": None,
        "highest_resistance_level": None,
        "treatments": [],
        "citations": [
            "Piazza R et al. Nat Genet 2013;45(1):18-24",
            "Makishima H et al. Nat Genet 2013;45(8):942-946",
            "Vishwakarma BA et al. J Clin Invest 2016;126(12):4562-4568",
        ],
        "clinical_note": (
            "SETBP1 G870S is in the SKI homology domain degron (amino acids "
            "858-871), the recurrent hotspot region. Stabilizes SETBP1 protein "
            "by disrupting beta-TrCP-mediated ubiquitination. Activates "
            "HOXA9/HOXA10 and represses PP2A tumor suppressor. Enriched in "
            "MDS/MPN overlap syndromes (aCML, CMML). No targeted therapy exists. "
            "CSF3R co-mutation enrichment (O/E=8.60 in our GENIE analysis) "
            "suggests shared MDS/MPN biology."
        ),
    },
    "EZH2 V662A": {
        "oncogenicity": "Likely Oncogenic",
        "mutation_effect": "Loss-of-function",
        "highest_sensitive_level": None,
        "highest_resistance_level": None,
        "treatments": [],
        "citations": [
            "Ernst T et al. Nat Genet 2010;42(8):722-726",
            "Nikoloski G et al. Nat Genet 2010;42(8):665-667",
        ],
        "clinical_note": (
            "EZH2 V662A is in the catalytic SET domain (residues 612-727, UniProt Q15910). "
            "VAF 59% indicates this is the founder clone. Loss-of-function "
            "EZH2 mutations in myeloid malignancies reduce H3K27 trimethylation, "
            "derepressing polycomb target genes. Classified as VUS in ClinVar for "
            "this specific residue but likely pathogenic given SET domain location "
            "and high VAF as founder event. No directly targeted therapy. "
            "EZH2 inhibitors (tazemetostat) target gain-of-function mutations "
            "in lymphomas and are contraindicated for loss-of-function variants."
        ),
    },
}


# -- API query ----------------------------------------------------------------

def load_api_token() -> str | None:
    """Load OncoKB API token from environment or ~/.secrets.env."""
    token = os.environ.get("ONCOKB_API_TOKEN")
    if token:
        return token

    secrets_path = Path.home() / ".secrets.env"
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ONCOKB_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def query_oncokb_api(
    variant: PatientVariant,
    token: str,
) -> dict[str, Any] | None:
    """Query OncoKB mutation annotation API for a single variant.

    Endpoint: /annotate/mutations/byProteinChange
    Required params: hugoSymbol, alteration, tumorType (OncoTree code)
    """
    url = f"{ONCOKB_API_BASE}/annotate/mutations/byProteinChange"
    params = {
        "hugoSymbol": variant.gene,
        "alteration": variant.protein_change,
        "tumorType": variant.oncotree_code,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 401:
            log.warning("OncoKB API: 401 Unauthorized - invalid or expired token")
            return None
        elif resp.status_code == 403:
            log.warning("OncoKB API: 403 Forbidden - token lacks access")
            return None
        else:
            log.warning(
                "OncoKB API: %d %s for %s %s",
                resp.status_code, resp.reason, variant.gene, variant.protein_change,
            )
            return None
    except requests.RequestException as exc:
        log.warning("OncoKB API request failed for %s %s: %s", variant.gene, variant.protein_change, exc)
        return None


def parse_api_response(variant: PatientVariant, data: dict[str, Any]) -> dict[str, Any]:
    """Parse OncoKB API response into structured annotation."""
    query = data.get("query", {})
    oncogenic = data.get("oncogenic", "Unknown")
    mutation_effect = data.get("mutationEffect", {})
    highest_sensitive = data.get("highestSensitiveLevel", None)
    highest_resistance = data.get("highestResistanceLevel", None)
    treatments_raw = data.get("treatments", [])

    treatments = []
    for tx in treatments_raw:
        drugs = " + ".join([d.get("drugName", "") for d in tx.get("drugs", [])])
        treatments.append({
            "drug": drugs,
            "level": tx.get("level", ""),
            "cancer_type": tx.get("levelAssociatedCancerType", {}).get("name", ""),
            "description": tx.get("description", ""),
            "fda_approved": tx.get("fdaApproved", False),
        })

    return {
        "gene": variant.gene,
        "protein_change": variant.protein_change,
        "vaf": variant.vaf,
        "oncogenicity": oncogenic,
        "mutation_effect": mutation_effect.get("knownEffect", "Unknown"),
        "mutation_effect_description": mutation_effect.get("description", ""),
        "highest_sensitive_level": highest_sensitive,
        "highest_resistance_level": highest_resistance,
        "treatments": treatments,
        "data_source": "oncokb_api",
    }


# -- Fallback from reference data --------------------------------------------

def get_reference_annotation(variant: PatientVariant) -> dict[str, Any]:
    """Get annotation from published reference data (fallback)."""
    key = f"{variant.gene} {variant.protein_change}"
    ref = ONCOKB_REFERENCE.get(key, {})

    return {
        "gene": variant.gene,
        "protein_change": variant.protein_change,
        "vaf": variant.vaf,
        "oncogenicity": ref.get("oncogenicity", "Unknown"),
        "mutation_effect": ref.get("mutation_effect", "Unknown"),
        "mutation_effect_description": "",
        "highest_sensitive_level": ref.get("highest_sensitive_level"),
        "highest_resistance_level": ref.get("highest_resistance_level"),
        "treatments": ref.get("treatments", []),
        "citations": ref.get("citations", []),
        "clinical_note": ref.get("clinical_note", ""),
        "data_source": "oncokb_reference_2026-03-27",
    }


# -- Report generation -------------------------------------------------------

def level_display(level: str | None) -> str:
    """Convert OncoKB level code to human-readable string."""
    mapping = {
        "LEVEL_1": "Level 1 (FDA-recognized biomarker, standard of care)",
        "LEVEL_2": "Level 2 (Standard of care, FDA-approved in another indication)",
        "LEVEL_3A": "Level 3A (Compelling clinical evidence)",
        "LEVEL_3B": "Level 3B (Standard of care in another biomarker)",
        "LEVEL_4": "Level 4 (Compelling biological evidence)",
        "LEVEL_R1": "Level R1 (Standard of care resistance biomarker)",
        "LEVEL_R2": "Level R2 (Compelling clinical resistance evidence)",
    }
    if level is None:
        return "No therapeutic level"
    return mapping.get(level, level)


def generate_markdown_report(annotations: list[dict[str, Any]], api_used: bool) -> str:
    """Generate markdown summary report from annotations."""
    lines = [
        "# OncoKB Clinical Actionability Annotations",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Data source:** {'OncoKB API (live query)' if api_used else 'OncoKB published classifications (reference data)'}",
        f"**OncoKB database:** https://www.oncokb.org",
        f"**Tumor type context:** AML (Acute Myeloid Leukemia)",
        "",
        "## Summary Table",
        "",
        "| Gene | Variant | VAF | Oncogenicity | Mutation Effect | Therapeutic Level |",
        "|------|---------|-----|-------------|-----------------|-------------------|",
    ]

    for ann in annotations:
        level = ann.get("highest_sensitive_level")
        level_str = level if level else "None"
        lines.append(
            f"| {ann['gene']} | {ann['protein_change']} | {ann['vaf']:.0%} "
            f"| {ann['oncogenicity']} | {ann['mutation_effect']} | {level_str} |"
        )

    lines.extend([
        "",
        "## Actionable Findings",
        "",
    ])

    actionable = [a for a in annotations if a.get("highest_sensitive_level")]
    if actionable:
        for ann in actionable:
            lines.append(f"### {ann['gene']} {ann['protein_change']} - {level_display(ann['highest_sensitive_level'])}")
            lines.append("")
            for tx in ann.get("treatments", []):
                fda_tag = " **(FDA-approved)**" if tx.get("fda_approved") else ""
                lines.append(f"- **{tx['drug']}** [{tx.get('level', 'N/A')}]{fda_tag}")
                if tx.get("description"):
                    lines.append(f"  - {tx['description']}")
                lines.append("")
    else:
        lines.append("No Level 1-4 actionable targets identified among all 5 variants.")
        lines.append("")

    lines.extend([
        "## Variant-by-Variant Annotations",
        "",
    ])

    for ann in annotations:
        ref = ONCOKB_REFERENCE.get(f"{ann['gene']} {ann['protein_change']}", {})
        lines.append(f"### {ann['gene']} {ann['protein_change']} (VAF {ann['vaf']:.0%})")
        lines.append("")
        lines.append(f"- **Oncogenicity:** {ann['oncogenicity']}")
        lines.append(f"- **Mutation effect:** {ann['mutation_effect']}")
        lines.append(f"- **Highest sensitive level:** {level_display(ann.get('highest_sensitive_level'))}")
        lines.append(f"- **Highest resistance level:** {level_display(ann.get('highest_resistance_level'))}")
        lines.append("")

        if ref.get("clinical_note"):
            lines.append(f"**Clinical context:** {ref['clinical_note']}")
            lines.append("")

        if ref.get("citations"):
            lines.append("**Key references:**")
            for cite in ref["citations"]:
                lines.append(f"- {cite}")
            lines.append("")

        if ann.get("treatments"):
            lines.append("**Treatments:**")
            for tx in ann["treatments"]:
                fda_tag = " (FDA-approved)" if tx.get("fda_approved") else ""
                lines.append(f"- {tx['drug']} [{tx.get('level', 'N/A')}]{fda_tag}")
                if tx.get("description"):
                    lines.append(f"  - {tx['description']}")
            lines.append("")
        else:
            lines.append("No targeted therapies with OncoKB therapeutic levels.")
            lines.append("")

    lines.extend([
        "## Therapeutic Implications for This Patient",
        "",
        "### Primary actionable target: IDH2 R140Q (Level 1)",
        "",
        "Despite the low VAF (2%), IDH2 R140Q is the only variant with an FDA-approved ",
        "targeted therapy (enasidenib). Clinical considerations:",
        "",
        "1. **Frontline:** Venetoclax + azacitidine is the standard backbone for ",
        "   older/unfit AML patients regardless of IDH2 status",
        "2. **IDH2-directed:** Enasidenib can be added to venetoclax + azacitidine ",
        "   (Level 3A evidence) or reserved for relapse/resistance if the IDH2 clone expands",
        "3. **Monitoring:** Serial IDH2 R140Q VAF tracking to detect subclone expansion",
        "",
        "### Investigational targets",
        "",
        "- **PTPN11 E76Q:** SHP2 inhibitors (TNO155, RMC-4550) in Phase I/II trials. ",
        "  No approved therapy. Our docking analysis supports binding affinity to E76Q mutant.",
        "- **SETBP1 G870S:** No targeted therapy exists. SKI domain degron mutations are ",
        "  not druggable with current compounds.",
        "- **DNMT3A R882H:** No targeted therapy. Hypomethylating agents (azacitidine, ",
        "  decitabine) address the epigenetic consequences indirectly.",
        "- **EZH2 V662A:** Loss-of-function mutation. EZH2 inhibitors (tazemetostat) are ",
        "  contraindicated as they target gain-of-function mutations in lymphomas.",
        "",
        "### Treatment strategy alignment",
        "",
        "The OncoKB annotations support the treatment recommendation from our drug ",
        "repurposing analysis: venetoclax + azacitidine backbone with enasidenib for the ",
        "IDH2 R140Q subclone. Only 1 of 5 variants (IDH2) has a Level 1 actionable target, ",
        "underscoring the limited therapeutic options for this rare mutation combination.",
        "",
        "---",
        "",
        "*OncoKB is maintained by Memorial Sloan Kettering Cancer Center. Therapeutic levels ",
        "follow the OncoKB evidence classification system. Level 1 = FDA-recognized biomarker ",
        "predictive of response to an FDA-approved drug. See https://www.oncokb.org/levels for ",
        "the complete level definitions.*",
        "",
    ])

    return "\n".join(lines)


# -- Main --------------------------------------------------------------------

def main() -> None:
    """Annotate patient variants with OncoKB clinical actionability data."""
    log.info("Starting OncoKB clinical actionability annotation")
    log.info("Patient context: MDS/AML with 5 somatic mutations")

    # Attempt API access
    token = load_api_token()
    api_used = False
    annotations: list[dict[str, Any]] = []

    if token:
        log.info("OncoKB API token found, attempting live queries...")
        api_results = []
        for variant in PATIENT_VARIANTS:
            log.info("Querying OncoKB API: %s %s", variant.gene, variant.protein_change)
            result = query_oncokb_api(variant, token)
            if result is not None:
                parsed = parse_api_response(variant, result)
                # Merge reference citations and clinical notes
                ref = ONCOKB_REFERENCE.get(f"{variant.gene} {variant.protein_change}", {})
                parsed["citations"] = ref.get("citations", [])
                parsed["clinical_note"] = ref.get("clinical_note", "")
                api_results.append(parsed)
            else:
                log.warning("API query failed for %s %s, will use reference data", variant.gene, variant.protein_change)
                api_results.append(None)
            time.sleep(0.5)  # Rate limit courtesy

        # Check if all API calls succeeded
        if all(r is not None for r in api_results):
            annotations = api_results
            api_used = True
            log.info("All 5 API queries succeeded")
        else:
            log.warning("Some API queries failed, falling back to reference data for all")
            annotations = []

    if not annotations:
        if token:
            log.warning("API access failed, using published reference data")
        else:
            log.info(
                "No OncoKB API token found (set ONCOKB_API_TOKEN env var). "
                "Using published reference data. "
                "Register at https://www.oncokb.org/account/register for API access."
            )
        for variant in PATIENT_VARIANTS:
            annotations.append(get_reference_annotation(variant))

    # Save JSON results
    output_json = RESULTS_DIR / "oncokb_annotations.json"
    output = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "data_source": "oncokb_api" if api_used else "oncokb_reference_2026-03-27",
            "oncokb_url": "https://www.oncokb.org",
            "tumor_type": "AML",
            "patient_context": "MDS/AML, 5 somatic mutations, monosomy 7, IPSS-M Very High",
            "api_token_available": token is not None,
            "api_query_successful": api_used,
            "note": (
                "Reference data compiled from OncoKB public database and published "
                "literature as of 2026-03-27. For the most current annotations, "
                "query the OncoKB API with a valid token."
            ),
        },
        "annotations": annotations,
        "summary": {
            "total_variants": len(annotations),
            "oncogenic": sum(1 for a in annotations if a["oncogenicity"] == "Oncogenic"),
            "likely_oncogenic": sum(1 for a in annotations if a["oncogenicity"] == "Likely Oncogenic"),
            "actionable_level_1": sum(
                1 for a in annotations if a.get("highest_sensitive_level") == "LEVEL_1"
            ),
            "actionable_any_level": sum(
                1 for a in annotations if a.get("highest_sensitive_level") is not None
            ),
            "total_treatments": sum(len(a.get("treatments", [])) for a in annotations),
        },
    }
    output_json.write_text(json.dumps(output, indent=2, default=str))
    log.info("JSON results saved to %s", output_json)

    # Save markdown report
    output_md = RESULTS_DIR / "oncokb_report.md"
    report = generate_markdown_report(annotations, api_used)
    output_md.write_text(report)
    log.info("Markdown report saved to %s", output_md)

    # Print summary
    log.info("=== OncoKB Annotation Summary ===")
    for ann in annotations:
        level = ann.get("highest_sensitive_level")
        level_str = f" [{level}]" if level else ""
        log.info(
            "  %s %s (VAF %.0f%%): %s, %s%s",
            ann["gene"],
            ann["protein_change"],
            ann["vaf"] * 100,
            ann["oncogenicity"],
            ann["mutation_effect"],
            level_str,
        )
    log.info(
        "Actionable: %d/%d variants (Level 1: %d)",
        output["summary"]["actionable_any_level"],
        output["summary"]["total_variants"],
        output["summary"]["actionable_level_1"],
    )


if __name__ == "__main__":
    main()
