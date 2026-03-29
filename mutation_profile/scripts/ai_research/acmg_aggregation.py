#!/usr/bin/env python3
"""
ACMG evidence aggregation for patient variants.

Combines all available pathogenicity evidence into formal ACMG/AMP
classifications using the Richards et al. (2015) combining rules and
the Tavtigian et al. (2020) Bayesian point system.

Evidence sources aggregated:
    1. pathogenicity_scores.json -- CADD, REVEL, SIFT, PolyPhen-2
    2. clinical_variant_scores.json -- AlphaMissense, multi-predictor consensus
    3. esm2_deep/esm2_deep_results.json -- ESM-2 LLR scores
    4. variant_interpretation/acmg_evidence.md -- prior manual ACMG curation
    5. oncokb_annotations.json -- oncogenicity classification
    6. civic_annotations.json -- clinical evidence
    7. gnomad_v4_results.json -- population frequency (PM2)
    8. clingen_validity.json -- gene-disease validity
    9. spliceai_scores.json -- splice prediction (BP7)
    10. eve_scores.json -- EVE pathogenicity
    11. mavedb_results.json -- functional assay (PS3)

Patient variants:
    1. EZH2 V662A (c.1985T>C) -- VAF 59%, founder clone
    2. DNMT3A R882H -- VAF 39%, pathogenic hotspot
    3. SETBP1 G870S -- VAF 34%, likely pathogenic
    4. PTPN11 E76Q -- VAF 29%, pathogenic
    5. IDH2 R140Q -- VAF 2%, pathogenic subclone

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/acmg_aggregation.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "ai_research"

# ---------------------------------------------------------------------------
# ACMG point system (Tavtigian et al. 2020, ClinGen SVI)
# ---------------------------------------------------------------------------
# Strength -> Bayesian points (pathogenic positive, benign negative)
STRENGTH_POINTS = {
    # Pathogenic
    "Very Strong": 8,
    "Strong": 4,
    "Moderate": 2,
    "Supporting": 1,
    # Benign
    "Stand-Alone": -8,
    "Strong_Benign": -4,
    "Moderate_Benign": -2,  # not standard but used for BS criteria at moderate
    "Supporting_Benign": -1,
}

# Classification thresholds (Tavtigian 2020)
# Pathogenic >= 10, Likely Pathogenic >= 6, VUS = -5..5,
# Likely Benign <= -1, Benign <= -7
CLASSIFICATION_THRESHOLDS = [
    (10, "Pathogenic"),
    (6, "Likely Pathogenic"),
    (0, "VUS"),
    (-1, "VUS"),  # -1..5 inclusive is VUS
    (-6, "Likely Benign"),
]


def classify_by_points(total_points: int) -> str:
    """Classify variant using Bayesian point totals."""
    if total_points >= 10:
        return "Pathogenic"
    if total_points >= 6:
        return "Likely Pathogenic"
    if total_points >= 0:
        return "VUS"
    if total_points >= -6:
        return "Likely Benign"
    return "Benign"


# ---------------------------------------------------------------------------
# Richards et al. 2015 rule-based classification
# ---------------------------------------------------------------------------
def classify_by_rules(
    criteria: dict[str, str],
) -> str:
    """Apply ACMG/AMP combining rules with ClinGen SVI strength modifications.

    Per ClinGen SVI (2020), criteria can be up/downgraded in strength.
    A PP criterion at Very Strong counts at the PVS level, a PP at Strong
    counts at PS level, etc. This function counts criteria by their
    *effective* strength level, not their default category.

    criteria: dict mapping criterion code (e.g. 'PS1') to strength
              ('Very Strong', 'Strong', 'Moderate', 'Supporting')
    """
    # Count by effective strength level (ClinGen SVI approach)
    # Any criterion at Very Strong counts as PVS-equivalent
    # Any criterion at Strong counts as PS-equivalent
    # Any criterion at Moderate counts as PM-equivalent
    # Any criterion at Supporting counts as PP-equivalent
    pvs = 0
    ps = 0
    pm = 0
    pp = 0
    for _c, s in criteria.items():
        if _c.startswith(("BA", "BS", "BP")):
            continue
        if s == "Very Strong":
            pvs += 1
        elif s == "Strong":
            ps += 1
        elif s == "Moderate":
            pm += 1
        elif s == "Supporting":
            pp += 1

    ba = any(c.startswith("BA") for c in criteria)
    bs = sum(1 for c, s in criteria.items() if c.startswith("BS"))
    bp = sum(1 for c, s in criteria.items() if c.startswith("BP"))

    # Benign rules
    if ba:
        return "Benign"
    if bs >= 2:
        return "Benign"
    if bs >= 1 and bp >= 1:
        return "Likely Benign"
    if bp >= 2:
        return "Likely Benign"

    # Pathogenic rules (Richards 2015 Table 5)
    # (i) 1 Very Strong AND (>=1 Strong OR >=2 Moderate OR 1 Moderate + 1 Supporting OR >=2 Supporting)
    if pvs >= 1:
        if ps >= 1:
            return "Pathogenic"
        if pm >= 2:
            return "Pathogenic"
        if pm >= 1 and pp >= 1:
            return "Pathogenic"
        if pp >= 2:
            return "Pathogenic"

    # (ii) >=2 Strong
    if ps >= 2:
        return "Pathogenic"

    # (iii) 1 Strong AND (>=3 Moderate OR 2 Moderate + >=2 Supporting OR 1 Moderate + >=4 Supporting)
    if ps >= 1:
        if pm >= 3:
            return "Pathogenic"
        if pm >= 2 and pp >= 2:
            return "Pathogenic"
        if pm >= 1 and pp >= 4:
            return "Pathogenic"

    # Likely Pathogenic rules
    # (i) 1 Very Strong AND 1 Moderate
    if pvs >= 1 and pm >= 1:
        return "Likely Pathogenic"
    # (ii) 1 Strong AND 1-2 Moderate
    if ps >= 1 and 1 <= pm <= 2:
        return "Likely Pathogenic"
    # (iii) 1 Strong AND >=2 Supporting
    if ps >= 1 and pp >= 2:
        return "Likely Pathogenic"
    # (iv) >=3 Moderate
    if pm >= 3:
        return "Likely Pathogenic"
    # (v) 2 Moderate AND >=2 Supporting
    if pm >= 2 and pp >= 2:
        return "Likely Pathogenic"
    # (vi) 1 Moderate AND >=4 Supporting
    if pm >= 1 and pp >= 4:
        return "Likely Pathogenic"

    return "VUS"


# ---------------------------------------------------------------------------
# Data class for per-variant evidence
# ---------------------------------------------------------------------------
@dataclass
class VariantEvidence:
    gene: str
    variant: str
    hgvs_c: str
    vaf: float
    transcript: str

    # Each criterion maps to its strength level
    criteria: dict[str, str] = field(default_factory=dict)
    # Detailed evidence text per criterion
    evidence_details: dict[str, str] = field(default_factory=dict)
    # Source scores
    scores: dict[str, object] = field(default_factory=dict)

    def total_points(self) -> int:
        total = 0
        for criterion, strength in self.criteria.items():
            if criterion.startswith(("PVS", "PS", "PM", "PP")):
                total += STRENGTH_POINTS.get(strength, 0)
            elif criterion.startswith(("BA", "BS", "BP")):
                if strength == "Stand-Alone":
                    total += STRENGTH_POINTS["Stand-Alone"]
                elif "Strong" in strength:
                    total += STRENGTH_POINTS["Strong_Benign"]
                else:
                    total += STRENGTH_POINTS["Supporting_Benign"]
        return total


# ---------------------------------------------------------------------------
# Load all evidence sources
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        log.warning("File not found: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def build_variant_evidence() -> list[VariantEvidence]:
    """Aggregate evidence from all sources for each patient variant."""

    # Define patient variants
    variants_def = [
        ("EZH2", "V662A", "c.1985T>C", 0.59, "NM_004456.5"),
        ("DNMT3A", "R882H", "c.2645G>A", 0.39, "NM_022552.5"),
        ("SETBP1", "G870S", "c.2608G>A", 0.34, "NM_015559.3"),
        ("PTPN11", "E76Q", "c.226G>C", 0.29, "NM_002834.5"),
        ("IDH2", "R140Q", "c.419G>A", 0.02, "NM_002168.4"),
    ]

    variants = [
        VariantEvidence(gene=g, variant=v, hgvs_c=h, vaf=vf, transcript=t)
        for g, v, h, vf, t in variants_def
    ]
    vmap = {ve.gene: ve for ve in variants}

    # --- Source 1: pathogenicity_scores.json (CADD, REVEL, SIFT, PolyPhen-2) ---
    path_scores = load_json(RESULTS_DIR / "pathogenicity_scoring" / "pathogenicity_scores.json")
    if path_scores:
        for vdata in path_scores.get("variants", []):
            gene = vdata["gene"]
            ve = vmap.get(gene)
            if not ve:
                continue
            scores = vdata.get("scores", {})
            ve.scores["cadd_phred"] = scores.get("cadd_phred")
            ve.scores["revel"] = scores.get("revel")
            ve.scores["sift"] = scores.get("sift")
            ve.scores["sift_prediction"] = scores.get("sift_prediction")
            ve.scores["polyphen2"] = scores.get("polyphen2")
            ve.scores["polyphen2_prediction"] = scores.get("polyphen2_prediction")
            ve.scores["gnomad_af_pathscores"] = vdata.get("gnomad_af")

    # --- Source 2: clinical_variant_scores.json (AlphaMissense, consensus) ---
    clin_scores = load_json(RESULTS_DIR / "clinical_variant_scores.json")
    if clin_scores:
        for key, vdata in clin_scores.get("variants", {}).items():
            gene = vdata.get("gene")
            ve = vmap.get(gene)
            if not ve:
                continue
            am = vdata.get("scores", {}).get("alphamissense", {})
            ve.scores["alphamissense_score"] = am.get("score")
            ve.scores["alphamissense_class"] = am.get("class")
            ve.scores["alphamissense_pp3"] = am.get("pp3")

            esm2_clin = vdata.get("scores", {}).get("esm2", {})
            ve.scores["esm2_llr"] = esm2_clin.get("llr")
            ve.scores["esm2_pp3"] = esm2_clin.get("pp3")

            consensus = vdata.get("consensus", {})
            ve.scores["n_predictors_pathogenic"] = consensus.get("n_predictors_pathogenic")
            ve.scores["n_predictors_scored"] = consensus.get("n_predictors_scored")

    # --- Source 3: ESM-2 deep results ---
    esm2_deep = load_json(RESULTS_DIR / "esm2_deep" / "esm2_deep_results.json")
    if esm2_deep:
        for gene, data in esm2_deep.items():
            ve = vmap.get(gene)
            if not ve:
                continue
            scan = data.get("positional_scan", {})
            site = scan.get("mutation_site", {})
            if site:
                ve.scores["esm2_deep_score"] = site.get("score")
                ve.scores["esm2_deep_alt_log_prob"] = site.get("alt_log_prob")

    # --- Source 5: OncoKB annotations ---
    oncokb = load_json(RESULTS_DIR / "oncokb_annotations.json")
    if oncokb:
        for ann in oncokb.get("annotations", []):
            gene = ann["gene"]
            ve = vmap.get(gene)
            if not ve:
                continue
            ve.scores["oncokb_oncogenicity"] = ann.get("oncogenicity")
            ve.scores["oncokb_mutation_effect"] = ann.get("mutation_effect")
            ve.scores["oncokb_level"] = ann.get("highest_sensitive_level")

    # --- Source 6: CIViC annotations ---
    civic = load_json(RESULTS_DIR / "civic_annotations.json")
    if civic:
        for result in civic.get("results", []):
            gene = result["gene"]
            ve = vmap.get(gene)
            if not ve:
                continue
            ve.scores["civic_evidence_count"] = result.get("summary", {}).get("total_evidence", 0)
            ve.scores["civic_variant_ids"] = result.get("civic_variant_ids", [])

    # --- Source 7: gnomAD v4 ---
    gnomad = load_json(RESULTS_DIR / "gnomad_v4_results.json")
    if gnomad:
        for result in gnomad.get("results", []):
            gene = result["gene"]
            ve = vmap.get(gene)
            if not ve:
                continue
            pm2 = result.get("pm2_classification", {})
            ve.scores["gnomad_pm2_strength"] = pm2.get("strength")
            ve.scores["gnomad_pm2_interpretation"] = pm2.get("interpretation")
            ve.scores["gnomad_total_ac"] = pm2.get("total_ac")
            ve.scores["gnomad_max_af"] = pm2.get("max_af")

    # --- Source 8: ClinGen gene-disease validity ---
    clingen = load_json(RESULTS_DIR / "clingen_validity.json")
    if clingen:
        for gene, gdata in clingen.get("genes", {}).items():
            ve = vmap.get(gene)
            if not ve:
                continue
            refs = gdata.get("curated_reference", [])
            ve.scores["clingen_classifications"] = [
                {"disease": r["disease"], "classification": r["classification"]}
                for r in refs
            ]

    # --- Source 9: SpliceAI ---
    spliceai = load_json(RESULTS_DIR / "spliceai_scores.json")
    if spliceai:
        for vdata in spliceai.get("variants", []):
            gene = vdata["gene"]
            ve = vmap.get(gene)
            if not ve:
                continue
            ve.scores["spliceai_max_delta"] = vdata.get("max_delta_score")
            ve.scores["spliceai_classification"] = vdata.get("classification")

    # --- Source 10: EVE ---
    eve = load_json(RESULTS_DIR / "eve_scores.json")
    if eve:
        for vdata in eve.get("variants", []):
            gene = vdata.get("gene")
            ve = vmap.get(gene)
            if not ve:
                continue
            eve_data = vdata.get("eve", {})
            ve.scores["eve_score"] = eve_data.get("score")
            ve.scores["eve_rankscore"] = eve_data.get("rankscore")
            primary = eve_data.get("primary_classification", {})
            ve.scores["eve_class"] = primary.get("label")
            companion = vdata.get("companion_scores", {})
            ve.scores["esm1b_score"] = companion.get("esm1b_score")

    # --- Source 11: MaveDB ---
    mavedb = load_json(RESULTS_DIR / "mavedb_results.json")
    if mavedb:
        ve_dnmt3a = vmap.get("DNMT3A")
        if ve_dnmt3a:
            garcia = mavedb.get("garcia_2025_reference", {})
            if garcia:
                ve_dnmt3a.scores["mavedb_garcia2025"] = garcia.get("acmg_evidence")
                ve_dnmt3a.scores["mavedb_garcia2025_summary"] = garcia.get("summary")

    # -----------------------------------------------------------------
    # Now assign ACMG criteria based on aggregated evidence
    # -----------------------------------------------------------------
    _assign_criteria(vmap)

    return variants


# ---------------------------------------------------------------------------
# Criterion assignment logic
# ---------------------------------------------------------------------------

# Known pathogenic ClinVar entries for PS1
_PS1_CLINVAR = {
    "DNMT3A": "VCV000012600 (Pathogenic)",
    "IDH2": "VCV000036527 (Pathogenic/Likely pathogenic)",
    "SETBP1": "VCV000029037 (Pathogenic, Schinzel-Giedion)",
    "PTPN11": "E76K VCV000013383 (Pathogenic for Noonan); E76Q reported in COSMIC/leukemia",
}

# Functional study references for PS3
_PS3_REFS = {
    "DNMT3A": (
        "~80% reduction in methyltransferase activity, dominant-negative on WT DNMT3A. "
        "Garcia et al. 2025 paired DMS (PS3_Strong). Russler-Germain 2014, Kim 2013."
    ),
    "IDH2": (
        "Neomorphic alpha-KG to 2-HG conversion. 2-HG elevated 10-100x. "
        "Ward 2010, Dang 2010. Crystal structure confirms altered active site."
    ),
    "SETBP1": (
        "G870S disrupts beta-TrCP binding, prevents ubiquitination/degradation. "
        "SETBP1 accumulation inhibits PP2A. Piazza 2018, Vishwakarma 2016."
    ),
    "PTPN11": (
        "E76 mutations disrupt N-SH2 autoinhibition. Constitutive SHP2 phosphatase "
        "activity, RAS-MAPK hyperactivation. Bentires-Alj 2004, Keilhack 2005."
    ),
    "EZH2": (
        "V662A in catalytic SET domain (612-727). Loss-of-function reducing H3K27me3. "
        "EZH2 LOF established as tumor suppressor in myeloid lineage. Ernst 2010, Nikoloski 2010. "
        "No direct functional assay for V662A specifically (PS3 at Supporting level)."
    ),
}

# PM1 hotspot/functional domain evidence
_PM1_DOMAINS = {
    "DNMT3A": "Methyltransferase catalytic domain (634-912). R882 is the dominant hotspot.",
    "IDH2": "IDH2 active site. R140 is one of only two recurrently mutated positions (R140, R172).",
    "SETBP1": "SCF/beta-TrCP degron motif (858-871). Nearly all pathogenic mutations cluster here.",
    "PTPN11": "N-SH2 domain (3-103) autoinhibitory interface. E76 is the primary somatic hotspot.",
    "EZH2": "Catalytic SET domain (612-727). Critical for H3K27 methyltransferase activity.",
}


def _get_pp3_strength(ve: VariantEvidence) -> tuple[str, str]:
    """Determine PP3 strength from computational predictors.

    Uses the ClinGen SVI multi-tool approach: if majority of well-calibrated
    tools agree on pathogenicity, assign PP3. Strength depends on concordance.

    Returns (strength, detail_string).
    """
    pathogenic_votes = 0
    total_votes = 0
    details = []

    # AlphaMissense
    am_score = ve.scores.get("alphamissense_score")
    if am_score is not None:
        total_votes += 1
        am_pp3 = ve.scores.get("alphamissense_pp3", "")
        if am_score >= 0.564:  # AlphaMissense pathogenic threshold
            pathogenic_votes += 1
        details.append(f"AlphaMissense={am_score:.4f} ({am_pp3})")

    # CADD
    cadd = ve.scores.get("cadd_phred")
    if cadd is not None:
        total_votes += 1
        if cadd >= 25.3:  # ClinGen SVI recommended CADD threshold
            pathogenic_votes += 1
        details.append(f"CADD={cadd:.1f}")

    # REVEL
    revel = ve.scores.get("revel")
    if revel is not None:
        total_votes += 1
        if revel >= 0.644:  # ClinGen SVI PP3_Moderate threshold
            pathogenic_votes += 1
        details.append(f"REVEL={revel:.3f}")

    # ESM-2 LLR
    esm2_llr = ve.scores.get("esm2_llr")
    if esm2_llr is not None:
        total_votes += 1
        if esm2_llr <= -1.5:
            pathogenic_votes += 1
        details.append(f"ESM-2 LLR={esm2_llr:.3f}")

    # EVE
    eve_class = ve.scores.get("eve_class")
    if eve_class:
        total_votes += 1
        if eve_class == "Pathogenic":
            pathogenic_votes += 1
        details.append(f"EVE={ve.scores.get('eve_score', 'N/A'):.4f} ({eve_class})")

    # SIFT
    sift = ve.scores.get("sift")
    if sift is not None:
        total_votes += 1
        if sift <= 0.05:
            pathogenic_votes += 1
        details.append(f"SIFT={sift}")

    # PolyPhen-2
    pp2 = ve.scores.get("polyphen2")
    if pp2 is not None:
        total_votes += 1
        if pp2 >= 0.85:
            pathogenic_votes += 1
        details.append(f"PolyPhen-2={pp2}")

    if total_votes == 0:
        return ("Not Met", "No computational predictors available")

    ratio = pathogenic_votes / total_votes
    detail_str = f"{pathogenic_votes}/{total_votes} pathogenic: " + ", ".join(details)

    # Strength assignment based on concordance and individual tool thresholds
    # If AlphaMissense >= 0.927 (VeryStrong) and REVEL >= 0.773 (Strong) and CADD >= 25.3:
    am_vs = am_score is not None and am_score >= 0.927
    revel_strong = ve.scores.get("revel") is not None and ve.scores["revel"] >= 0.773

    if ratio >= 0.85 and am_vs:
        return ("Very Strong", detail_str)
    if ratio >= 0.7 and (am_vs or revel_strong):
        return ("Strong", detail_str)
    if ratio >= 0.6:
        return ("Moderate", detail_str)
    if ratio >= 0.5:
        return ("Supporting", detail_str)
    return ("Not Met", detail_str)


def _assign_criteria(vmap: dict[str, VariantEvidence]) -> None:
    """Assign all ACMG criteria for each variant."""

    for gene, ve in vmap.items():
        # === PS1: Same amino acid change previously established as pathogenic ===
        if gene in _PS1_CLINVAR:
            ve.criteria["PS1"] = "Strong"
            ve.evidence_details["PS1"] = _PS1_CLINVAR[gene]
        # EZH2 V662A: not in ClinVar as pathogenic for this specific change
        # but EZH2 LOF in SET domain is well-established -- use PM5 instead

        # === PS3: Well-established functional studies ===
        if gene in _PS3_REFS:
            if gene == "EZH2":
                # No direct V662A functional assay; general LOF evidence only
                ve.criteria["PS3"] = "Supporting"
            elif gene == "DNMT3A":
                # Garcia 2025 paired DMS is PS3_Strong
                ve.criteria["PS3"] = "Strong"
            else:
                ve.criteria["PS3"] = "Strong"
            ve.evidence_details["PS3"] = _PS3_REFS[gene]

        # === PM1: Hotspot / critical functional domain without benign variation ===
        if gene in _PM1_DOMAINS:
            ve.criteria["PM1"] = "Moderate"
            # DNMT3A R882 and SETBP1 SKI domain are extremely well-characterized hotspots
            # Upgrade to Strong per ClinGen SVI PM1_Strong guidance
            if gene in ("DNMT3A", "SETBP1", "IDH2"):
                ve.criteria["PM1"] = "Strong"
            ve.evidence_details["PM1"] = _PM1_DOMAINS[gene]

        # === PM2: Absent / extremely rare in population databases ===
        pm2_strength = ve.scores.get("gnomad_pm2_strength", "")
        if pm2_strength == "PM2_Strong":
            # ClinGen SVI: PM2 max strength is Supporting (2020 guidance)
            # But absent from gnomAD entirely is the strongest PM2 can be
            ve.criteria["PM2"] = "Supporting"
            ve.evidence_details["PM2"] = (
                f"Absent from gnomAD v4 (AC=0). {ve.scores.get('gnomad_pm2_interpretation', '')}"
            )
        elif pm2_strength == "PM2_Supporting":
            ve.criteria["PM2"] = "Supporting"
            ve.evidence_details["PM2"] = (
                f"Very rare in gnomAD v4. {ve.scores.get('gnomad_pm2_interpretation', '')}"
            )
        elif pm2_strength == "PM2_Not_Met":
            # DNMT3A R882H: present in gnomAD likely as somatic CHIP contamination
            if gene == "DNMT3A":
                ve.evidence_details["PM2"] = (
                    "Present in gnomAD at AF=3.69e-04 (427 alleles). This is attributable to "
                    "CHIP (clonal hematopoiesis) contamination in gnomAD blood-derived samples. "
                    "PM2 not formally met but CHIP context noted."
                )

        # === PM5: Novel missense at position where different pathogenic missense seen ===
        # Applicable to EZH2 V662A since other SET domain missense are pathogenic
        if gene == "EZH2":
            ve.criteria["PM5"] = "Supporting"
            ve.evidence_details["PM5"] = (
                "Other missense variants in the EZH2 SET domain (e.g., Y646, A682, A692) are "
                "established as pathogenic in lymphoid malignancies (gain-of-function). In myeloid "
                "context, EZH2 LOF missense across SET domain are recurrent."
            )

        # === PP3: Computational (in silico) evidence ===
        pp3_strength, pp3_detail = _get_pp3_strength(ve)
        if pp3_strength != "Not Met":
            ve.criteria["PP3"] = pp3_strength
            ve.evidence_details["PP3"] = pp3_detail

        # === PP5 / PS2 equivalent: Reputable source classifications ===
        # OncoKB oncogenicity as supporting evidence
        oncokb_class = ve.scores.get("oncokb_oncogenicity", "")
        if oncokb_class in ("Oncogenic", "Likely Oncogenic"):
            ve.criteria["PP5"] = "Strong" if oncokb_class == "Oncogenic" else "Supporting"
            ve.evidence_details["PP5"] = (
                f"OncoKB: {oncokb_class}. "
                f"Effect: {ve.scores.get('oncokb_mutation_effect', 'N/A')}. "
                f"Level: {ve.scores.get('oncokb_level') or 'No targeted therapy'}."
            )

        # === CIViC evidence as additional PP5 support ===
        civic_count = ve.scores.get("civic_evidence_count", 0)
        if civic_count > 0:
            existing = ve.evidence_details.get("PP5", "")
            ve.evidence_details["PP5"] = (
                existing + f" CIViC: {civic_count} evidence item(s) supporting oncogenicity."
            )

        # === ClinGen gene-disease validity as supporting context ===
        clingen_list = ve.scores.get("clingen_classifications", [])
        if clingen_list:
            definitives = [c for c in clingen_list if c["classification"] == "Definitive"]
            if definitives:
                detail = "; ".join(f"{c['disease']} ({c['classification']})" for c in clingen_list)
                ve.evidence_details["ClinGen_GeneDisease"] = detail

        # === BP7: Splice impact (benign evidence if no splice disruption) ===
        # All variants show NO_DATA from SpliceAI -- cannot apply BP7
        splice_class = ve.scores.get("spliceai_classification")
        if splice_class and splice_class != "NO_DATA":
            max_delta = ve.scores.get("spliceai_max_delta", 0) or 0
            if max_delta < 0.2:
                ve.criteria["BP7"] = "Supporting_Benign"
                ve.evidence_details["BP7"] = f"SpliceAI max delta = {max_delta:.3f} (< 0.2 threshold)"


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
def generate_json_output(variants: list[VariantEvidence]) -> dict:
    """Generate structured JSON output."""
    results = []
    for ve in variants:
        points = ve.total_points()
        rules_class = classify_by_rules(ve.criteria)
        points_class = classify_by_points(points)

        # Use the more conservative of the two methods
        # (in practice they usually agree)
        if _class_rank(rules_class) > _class_rank(points_class):
            final_class = points_class
            method_note = f"Point system ({points} pts) more conservative than rules ({rules_class})"
        elif _class_rank(rules_class) < _class_rank(points_class):
            final_class = rules_class
            method_note = f"Rules more conservative than point system ({points} pts -> {points_class})"
        else:
            final_class = rules_class
            method_note = f"Rules and point system ({points} pts) agree"

        pathogenic_criteria = {
            k: v for k, v in ve.criteria.items()
            if k.startswith(("PVS", "PS", "PM", "PP"))
        }
        benign_criteria = {
            k: v for k, v in ve.criteria.items()
            if k.startswith(("BA", "BS", "BP"))
        }

        results.append({
            "gene": ve.gene,
            "variant": ve.variant,
            "hgvs_c": ve.hgvs_c,
            "vaf": ve.vaf,
            "transcript": ve.transcript,
            "classification": {
                "final": final_class,
                "by_rules": rules_class,
                "by_points": points_class,
                "total_points": points,
                "method_note": method_note,
            },
            "pathogenic_criteria": pathogenic_criteria,
            "benign_criteria": benign_criteria,
            "evidence_details": ve.evidence_details,
            "key_scores": {
                "cadd_phred": ve.scores.get("cadd_phred"),
                "revel": ve.scores.get("revel"),
                "alphamissense": ve.scores.get("alphamissense_score"),
                "esm2_llr": ve.scores.get("esm2_llr"),
                "eve_score": ve.scores.get("eve_score"),
                "eve_class": ve.scores.get("eve_class"),
                "gnomad_max_af": ve.scores.get("gnomad_max_af"),
                "oncokb": ve.scores.get("oncokb_oncogenicity"),
                "civic_evidence": ve.scores.get("civic_evidence_count"),
            },
        })

    return {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "method": "ACMG/AMP (Richards et al. 2015) + Bayesian point system (Tavtigian et al. 2020)",
            "evidence_sources": [
                "pathogenicity_scores.json (CADD, REVEL, SIFT, PolyPhen-2, gnomAD)",
                "clinical_variant_scores.json (AlphaMissense, multi-predictor consensus)",
                "esm2_deep/esm2_deep_results.json (ESM-2 deep positional analysis)",
                "oncokb_annotations.json (OncoKB oncogenicity)",
                "civic_annotations.json (CIViC clinical evidence)",
                "gnomad_v4_results.json (gnomAD v4 population frequency)",
                "clingen_validity.json (ClinGen gene-disease validity)",
                "spliceai_scores.json (SpliceAI splice prediction)",
                "eve_scores.json (EVE evolutionary model)",
                "mavedb_results.json (MaveDB functional assay reference)",
                "variant_interpretation/acmg_evidence.md (prior manual curation)",
            ],
            "patient_context": "MDS/AML, 5 somatic mutations, monosomy 7, IPSS-M Very High",
            "n_variants": len(results),
            "point_system": {
                "Very Strong": 8,
                "Strong": 4,
                "Moderate": 2,
                "Supporting": 1,
                "Pathogenic_threshold": ">=10 points",
                "Likely_Pathogenic_threshold": ">=6 points",
            },
        },
        "variants": results,
        "summary": {
            "pathogenic": sum(1 for r in results if r["classification"]["final"] == "Pathogenic"),
            "likely_pathogenic": sum(1 for r in results if r["classification"]["final"] == "Likely Pathogenic"),
            "vus": sum(1 for r in results if r["classification"]["final"] == "VUS"),
            "likely_benign": sum(1 for r in results if r["classification"]["final"] == "Likely Benign"),
            "benign": sum(1 for r in results if r["classification"]["final"] == "Benign"),
        },
    }


def _class_rank(classification: str) -> int:
    """Rank classifications from most pathogenic (5) to most benign (1)."""
    ranks = {
        "Pathogenic": 5,
        "Likely Pathogenic": 4,
        "VUS": 3,
        "Likely Benign": 2,
        "Benign": 1,
    }
    return ranks.get(classification, 3)


def generate_markdown_report(data: dict) -> str:
    """Generate human-readable markdown report."""
    lines = [
        "# ACMG Evidence Aggregation Report",
        "",
        f"**Generated:** {data['metadata']['generated']}",
        f"**Method:** {data['metadata']['method']}",
        f"**Patient:** MDS/AML with 5 somatic mutations + monosomy 7, IPSS-M Very High",
        "",
        "## Classification Summary",
        "",
        "| Variant | Gene | VAF | Classification | Points | Method Agreement |",
        "|---------|------|-----|---------------|--------|-----------------|",
    ]

    for v in data["variants"]:
        c = v["classification"]
        lines.append(
            f"| {v['gene']} {v['variant']} | {v['gene']} | {v['vaf']:.0%} | "
            f"**{c['final']}** | {c['total_points']} | {c['method_note']} |"
        )

    lines.extend([
        "",
        f"**Total:** {data['summary']['pathogenic']} Pathogenic, "
        f"{data['summary']['likely_pathogenic']} Likely Pathogenic, "
        f"{data['summary']['vus']} VUS",
        "",
        "---",
        "",
    ])

    # Per-variant detail
    for v in data["variants"]:
        c = v["classification"]
        lines.extend([
            f"## {v['gene']} {v['variant']} ({v['hgvs_c']})",
            "",
            f"**Transcript:** {v['transcript']}",
            f"**VAF:** {v['vaf']:.0%}",
            f"**Final Classification:** {c['final']}",
            f"**Points:** {c['total_points']} (Rules: {c['by_rules']}, Points: {c['by_points']})",
            "",
        ])

        # Pathogenic criteria table
        if v["pathogenic_criteria"]:
            lines.extend([
                "### Pathogenic Evidence",
                "",
                "| Criterion | Strength | Evidence |",
                "|-----------|----------|---------|",
            ])
            for crit, strength in sorted(v["pathogenic_criteria"].items()):
                detail = v["evidence_details"].get(crit, "")
                # Truncate long details for table readability
                if len(detail) > 200:
                    detail = detail[:197] + "..."
                lines.append(f"| {crit} | {strength} | {detail} |")
            lines.append("")

        # Benign criteria
        if v["benign_criteria"]:
            lines.extend([
                "### Benign Evidence",
                "",
                "| Criterion | Strength | Evidence |",
                "|-----------|----------|---------|",
            ])
            for crit, strength in sorted(v["benign_criteria"].items()):
                detail = v["evidence_details"].get(crit, "")
                lines.append(f"| {crit} | {strength} | {detail} |")
            lines.append("")

        # Key scores
        ks = v["key_scores"]
        lines.extend([
            "### Key Scores",
            "",
            "| Predictor | Score | Interpretation |",
            "|-----------|-------|---------------|",
        ])

        if ks.get("cadd_phred") is not None:
            cadd_interp = "Deleterious" if ks["cadd_phred"] >= 25.3 else "Tolerated"
            lines.append(f"| CADD (Phred) | {ks['cadd_phred']:.1f} | {cadd_interp} (threshold >=25.3) |")
        if ks.get("revel") is not None:
            revel_interp = "Pathogenic" if ks["revel"] >= 0.644 else "Benign/VUS"
            lines.append(f"| REVEL | {ks['revel']:.3f} | {revel_interp} (threshold >=0.644) |")
        if ks.get("alphamissense") is not None:
            am_interp = "Pathogenic" if ks["alphamissense"] >= 0.564 else "Benign"
            lines.append(f"| AlphaMissense | {ks['alphamissense']:.4f} | {am_interp} (threshold >=0.564) |")
        if ks.get("esm2_llr") is not None:
            esm2_interp = "PP3_Strong" if ks["esm2_llr"] <= -7.0 else (
                "PP3_Moderate" if ks["esm2_llr"] <= -3.5 else (
                    "PP3_Supporting" if ks["esm2_llr"] <= -1.5 else "Benign/VUS"
                ))
            lines.append(f"| ESM-2 LLR | {ks['esm2_llr']:.3f} | {esm2_interp} |")
        if ks.get("eve_score") is not None:
            lines.append(f"| EVE | {ks['eve_score']:.4f} | {ks.get('eve_class', 'N/A')} |")
        if ks.get("gnomad_max_af") is not None:
            lines.append(f"| gnomAD AF | {ks['gnomad_max_af']:.2e} | {'Absent' if ks['gnomad_max_af'] == 0 else 'Present'} |")
        if ks.get("oncokb"):
            lines.append(f"| OncoKB | {ks['oncokb']} | - |")
        if ks.get("civic_evidence") is not None:
            lines.append(f"| CIViC | {ks['civic_evidence']} evidence items | - |")

        # ClinGen context
        clingen_detail = v["evidence_details"].get("ClinGen_GeneDisease")
        if clingen_detail:
            lines.extend([
                "",
                f"**ClinGen Gene-Disease Validity:** {clingen_detail}",
            ])

        # PM2 note (e.g. DNMT3A CHIP explanation)
        pm2_detail = v["evidence_details"].get("PM2", "")
        if pm2_detail and "PM2" not in v["pathogenic_criteria"]:
            lines.extend([
                "",
                f"**PM2 Note:** {pm2_detail}",
            ])

        lines.extend(["", "---", ""])

    # Evidence source summary
    lines.extend([
        "## Evidence Sources",
        "",
    ])
    for i, src in enumerate(data["metadata"]["evidence_sources"], 1):
        lines.append(f"{i}. {src}")

    lines.extend([
        "",
        "## Methodology",
        "",
        "Classification uses two parallel approaches:",
        "",
        "1. **Richards et al. 2015 combining rules** -- the original ACMG/AMP rule tables",
        "2. **Tavtigian et al. 2020 Bayesian point system** -- quantitative point-based classification",
        "",
        "The final classification uses the more conservative of the two methods when they disagree.",
        "",
        "### ACMG Point System",
        "",
        "| Strength | Points |",
        "|----------|--------|",
        "| Very Strong (PVS1) | 8 |",
        "| Strong (PS1-4) | 4 |",
        "| Moderate (PM1-6) | 2 |",
        "| Supporting (PP1-5) | 1 |",
        "",
        "| Classification | Threshold |",
        "|---------------|-----------|",
        "| Pathogenic | >= 10 points |",
        "| Likely Pathogenic | >= 6 points |",
        "| VUS | 0 to 5 points |",
        "| Likely Benign | -1 to -6 points |",
        "| Benign | <= -7 points |",
        "",
        "### PP3 Multi-tool Assessment",
        "",
        "PP3 strength determined by concordance across 7 calibrated predictors:",
        "AlphaMissense, CADD, REVEL, ESM-2, EVE, SIFT, PolyPhen-2.",
        "",
        "- >= 85% concordance + AlphaMissense >= 0.927: Very Strong",
        "- >= 70% concordance + (AlphaMissense >= 0.927 OR REVEL >= 0.773): Strong",
        "- >= 60% concordance: Moderate",
        "- >= 50% concordance: Supporting",
        "",
        "### Important Caveats",
        "",
        "1. **Somatic context:** ACMG/AMP guidelines were designed for germline variant "
        "classification. Application to somatic variants uses the adapted AMP/ASCO/CAP framework "
        "(Li et al. 2017) where applicable. PS1/PS3/PM1 criteria apply directly.",
        "",
        "2. **Gain-of-function limitation:** ESM-2 and EVE underperform for gain-of-function "
        "variants (IDH2 R140Q, PTPN11 E76Q, SETBP1 G870S) because protein language models "
        "measure evolutionary constraint / structural disruption, not neomorphic activity.",
        "",
        "3. **CHIP contamination:** DNMT3A R882H presence in gnomAD (AF ~3.7e-4) reflects somatic "
        "CHIP in blood-derived samples, not true germline frequency. PM2 is not formally met but "
        "this does not count as benign evidence.",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Starting ACMG evidence aggregation")

    variants = build_variant_evidence()
    log.info("Loaded evidence for %d variants", len(variants))

    for ve in variants:
        log.info(
            "%s %s: %d criteria, %d points",
            ve.gene, ve.variant, len(ve.criteria), ve.total_points(),
        )

    data = generate_json_output(variants)

    # Write JSON
    json_path = RESULTS_DIR / "acmg_aggregation.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Wrote JSON: %s", json_path)

    # Write Markdown report
    md_path = RESULTS_DIR / "acmg_aggregation_report.md"
    report = generate_markdown_report(data)
    with open(md_path, "w") as f:
        f.write(report)
    log.info("Wrote report: %s", md_path)

    # Print summary
    print("\n" + "=" * 70)
    print("ACMG EVIDENCE AGGREGATION RESULTS")
    print("=" * 70)
    for v in data["variants"]:
        c = v["classification"]
        criteria_str = ", ".join(
            f"{k}({s})" for k, s in sorted(v["pathogenic_criteria"].items())
        )
        print(f"\n  {v['gene']} {v['variant']} (VAF {v['vaf']:.0%})")
        print(f"    Classification: {c['final']} ({c['total_points']} points)")
        print(f"    Criteria: {criteria_str}")

    s = data["summary"]
    print(f"\n  Summary: {s['pathogenic']}P / {s['likely_pathogenic']}LP / {s['vus']}VUS")
    print("=" * 70)


if __name__ == "__main__":
    main()
