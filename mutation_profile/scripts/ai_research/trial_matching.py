#!/usr/bin/env python3
"""
trial_matching.py -- Systematic clinical trial matching for patient mutation profile.

Queries ClinicalTrials.gov API v2 for trials targeting the patient's 5 driver mutations
(DNMT3A R882H, IDH2 R140Q, SETBP1 G870S, PTPN11 E76Q, EZH2 V662A), monosomy 7,
and post-HSCT relapse scenarios.

Patient: 36M, post-allo-HSCT for MDS-AML, ELN 2022 adverse risk.

Inputs:
    - ClinicalTrials.gov API v2 (remote, no key required)

Outputs:
    - mutation_profile/results/ai_research/trial_matching.json
    - mutation_profile/results/ai_research/trial_matching_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/trial_matching.py

Runtime: ~1-3 minutes (API rate limits)
Dependencies: requests
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # mutation_profile/
RESULTS_DIR = PROJECT_ROOT / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = RESULTS_DIR / "trial_matching.json"
REPORT_OUTPUT = RESULTS_DIR / "trial_matching_report.md"

# ---------------------------------------------------------------------------
# ClinicalTrials.gov API v2 configuration
# ---------------------------------------------------------------------------
CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
CT_PAGE_SIZE = 50
CT_RATE_LIMIT_DELAY = 0.5  # seconds between requests

# ---------------------------------------------------------------------------
# Patient profile
# ---------------------------------------------------------------------------
PATIENT = {
    "age": 36,
    "sex": "MALE",
    "diagnosis": "MDS-AML (MDS-IB2 / MDS with increased blasts-2)",
    "eln_risk": "Adverse",
    "cytogenetics": "Monosomy 7 (45,XY,-7)",
    "hsct_status": "Post allo-HSCT (Nov 2023), >99% donor chimerism",
    "disease_status": "Complete morphological remission, MRD negative (March 2026)",
    "gvhd": "Mild chronic GvHD, topical steroids only",
    "mutations": {
        "DNMT3A": {"variant": "R882H", "vaf": 0.39, "classification": "Pathogenic"},
        "IDH2": {"variant": "R140Q", "vaf": 0.02, "classification": "Pathogenic (subclone)"},
        "SETBP1": {"variant": "G870S", "vaf": 0.34, "classification": "Likely pathogenic"},
        "PTPN11": {"variant": "E76Q", "vaf": 0.29, "classification": "Pathogenic"},
        "EZH2": {"variant": "V662A", "vaf": 0.59, "classification": "Pathogenic (biallelic)"},
    },
    "negative_markers": ["FLT3-ITD", "FLT3-TKD", "NPM1"],
}

# ---------------------------------------------------------------------------
# Search queries -- drug/target combinations to search for
# ---------------------------------------------------------------------------
SEARCH_QUERIES: list[dict[str, Any]] = [
    # IDH2 inhibitors
    {
        "category": "IDH2 Inhibitors",
        "target_mutation": "IDH2 R140Q",
        "rationale": "FDA-approved enasidenib targets IDH2 R140Q directly. Olutasidenib is dual IDH1/IDH2.",
        "queries": [
            {"terms": "enasidenib AND (AML OR MDS)", "label": "Enasidenib in AML/MDS"},
            {"terms": "olutasidenib AND (AML OR MDS)", "label": "Olutasidenib in AML/MDS"},
            {"terms": "vorasidenib AND (AML OR MDS OR leukemia)", "label": "Vorasidenib in AML/MDS"},
            {"terms": "IDH2 AND inhibitor AND (AML OR MDS)", "label": "IDH2 inhibitors in AML/MDS"},
        ],
    },
    # SHP2 inhibitors (PTPN11)
    {
        "category": "SHP2 Inhibitors (PTPN11)",
        "target_mutation": "PTPN11 E76Q",
        "rationale": "SHP2 gain-of-function drives RAS-MAPK signaling. Multiple inhibitors in Phase 1/2.",
        "queries": [
            {"terms": "TNO155 AND (AML OR MDS OR myeloid OR leukemia)", "label": "TNO155 in myeloid"},
            {"terms": "RMC-4550 AND (AML OR MDS OR myeloid OR leukemia)", "label": "RMC-4550 in myeloid"},
            {"terms": "JAB-3312 AND (AML OR MDS OR myeloid OR leukemia)", "label": "JAB-3312 in myeloid"},
            {"terms": "BBP-398 AND (AML OR MDS OR myeloid OR leukemia)", "label": "BBP-398 in myeloid"},
            {"terms": "SHP2 AND inhibitor AND (AML OR MDS OR myeloid)", "label": "SHP2 inhibitors in myeloid"},
            {"terms": "PTPN11 AND (AML OR MDS OR myeloid)", "label": "PTPN11-targeted trials"},
        ],
    },
    # EZH2 inhibitors
    {
        "category": "EZH2 Inhibitors",
        "target_mutation": "EZH2 V662A (biallelic inactivation)",
        "rationale": "EZH2 loss-of-function via monosomy 7 + V662A. Tazemetostat targets EZH2 gain-of-function. PRC2 modulators may apply.",
        "queries": [
            {"terms": "tazemetostat AND (AML OR MDS OR myeloid OR leukemia)", "label": "Tazemetostat in myeloid"},
            {"terms": "EZH2 AND (AML OR MDS OR myeloid)", "label": "EZH2-targeted myeloid trials"},
            {"terms": "PRC2 AND (AML OR MDS OR myeloid)", "label": "PRC2 modulator trials"},
        ],
    },
    # Venetoclax combinations post-HSCT
    {
        "category": "Venetoclax Combinations (Post-HSCT)",
        "target_mutation": "General AML/DNMT3A R882H",
        "rationale": "Venetoclax + HMA is standard for unfit AML. Post-HSCT relapse trials are critical.",
        "queries": [
            {"terms": "venetoclax AND transplant AND (AML OR MDS) AND relapse", "label": "Venetoclax post-HSCT relapse"},
            {"terms": "venetoclax AND (maintenance OR post-transplant) AND AML", "label": "Venetoclax maintenance post-HSCT"},
            {"terms": "venetoclax AND azacitidine AND (relapsed OR refractory) AND AML", "label": "Ven/Aza in R/R AML"},
        ],
    },
    # Menin inhibitors
    {
        "category": "Menin Inhibitors",
        "target_mutation": "General AML (emerging class)",
        "rationale": "Revumenib and ziftomenib target KMT2A/NPM1 AML. May have broader activity in IDH-mutated AML.",
        "queries": [
            {"terms": "revumenib AND (AML OR MDS OR leukemia)", "label": "Revumenib in AML"},
            {"terms": "ziftomenib AND (AML OR MDS OR leukemia)", "label": "Ziftomenib in AML"},
            {"terms": "menin AND inhibitor AND (AML OR MDS)", "label": "Menin inhibitors in AML"},
        ],
    },
    # HSCT relapse-specific
    {
        "category": "HSCT Relapse-Specific",
        "target_mutation": "Post-HSCT relapse/maintenance",
        "rationale": "Patient is in remission post-HSCT but at high risk of relapse (adverse cytogenetics, 5 driver mutations).",
        "queries": [
            {"terms": "allogeneic AND transplant AND relapse AND (AML OR MDS) AND (maintenance OR prevention)", "label": "Post-HSCT relapse prevention"},
            {"terms": "donor lymphocyte AND (AML OR MDS) AND relapse", "label": "DLI in AML/MDS relapse"},
            {"terms": "monosomy 7 AND (AML OR MDS) AND (clinical trial OR treatment)", "label": "Monosomy 7 AML trials"},
            {"terms": "MRD AND (AML OR MDS) AND post-transplant", "label": "MRD-guided post-HSCT"},
        ],
    },
    # DNMT3A-targeted
    {
        "category": "DNMT3A-Targeted / Hypomethylating",
        "target_mutation": "DNMT3A R882H",
        "rationale": "DNMT3A R882H is a founder mutation at 39% VAF. HMAs and novel DNMT3A-targeting agents.",
        "queries": [
            {"terms": "DNMT3A AND (AML OR MDS) AND (targeted OR inhibitor)", "label": "DNMT3A-targeted trials"},
            {"terms": "oral azacitidine AND (AML OR MDS) AND maintenance", "label": "Oral aza maintenance"},
            {"terms": "decitabine AND cedazuridine AND (AML OR MDS)", "label": "ASTX727 (oral decitabine)"},
        ],
    },
]

# Nordic/European countries to prioritize
PRIORITY_COUNTRIES = {
    "Norway", "Sweden", "Denmark", "Germany", "Belgium", "Netherlands",
    "Finland", "United Kingdom", "France", "Switzerland", "Austria",
}
TOP_PRIORITY_COUNTRIES = {"Norway", "Sweden", "Denmark", "Germany", "Belgium", "Netherlands"}


def query_ctgov(search_terms: str, page_token: str | None = None) -> dict:
    """Query ClinicalTrials.gov API v2."""
    params: dict[str, Any] = {
        "query.term": search_terms,
        "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ENROLLING_BY_INVITATION,ACTIVE_NOT_RECRUITING",
        "pageSize": CT_PAGE_SIZE,
        "fields": (
            "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,"
            "EnrollmentCount,StartDate,CompletionDate,"
            "BriefSummary,EligibilityCriteria,MinimumAge,MaximumAge,Sex,"
            "Condition,Intervention,LocationCountry,LocationFacility,LocationCity,"
            "StudyType,DesignPrimaryPurpose,LeadSponsorName,"
            "CollaboratorName,ResponsiblePartyInvestigatorFullName"
        ),
        "format": "json",
    }
    if page_token:
        params["pageToken"] = page_token

    try:
        resp = requests.get(CT_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  WARNING: API request failed for '{search_terms[:60]}...': {e}")
        return {"studies": [], "totalCount": 0}


def extract_study_info(study: dict) -> dict:
    """Extract relevant fields from a ClinicalTrials.gov study record."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    eligibility = proto.get("eligibilityModule", {})
    contacts = proto.get("contactsLocationsModule", {})
    desc = proto.get("descriptionModule", {})
    arms = proto.get("armsInterventionsModule", {})
    conditions = proto.get("conditionsModule", {})
    sponsor = proto.get("sponsorCollaboratorsModule", {})

    # Extract locations
    locations = contacts.get("locations", [])
    countries = set()
    facilities = []
    for loc in locations:
        country = loc.get("country", "")
        if country:
            countries.add(country)
        facility = loc.get("facility", "")
        city = loc.get("city", "")
        if facility or city:
            facilities.append({
                "facility": facility,
                "city": city,
                "country": country,
            })

    # Extract interventions
    interventions = []
    for intervention in arms.get("interventions", []):
        interventions.append({
            "name": intervention.get("name", ""),
            "type": intervention.get("type", ""),
            "description": intervention.get("description", "")[:300],
        })

    # Phase extraction
    phases = design.get("phases", [])
    phase_str = ", ".join(phases) if phases else "Not specified"

    # Eligibility criteria text
    criteria_text = eligibility.get("eligibilityCriteria", "")

    # Sponsor
    lead_sponsor = sponsor.get("leadSponsor", {}).get("name", "Unknown")

    return {
        "nct_id": ident.get("nctId", ""),
        "brief_title": ident.get("briefTitle", ""),
        "official_title": ident.get("officialTitle", ""),
        "overall_status": status.get("overallStatus", ""),
        "phase": phase_str,
        "enrollment": design.get("enrollmentInfo", {}).get("count"),
        "start_date": status.get("startDateStruct", {}).get("date", ""),
        "completion_date": status.get("completionDateStruct", {}).get("date", ""),
        "brief_summary": desc.get("briefSummary", "")[:500],
        "eligibility_criteria": criteria_text,
        "min_age": eligibility.get("minimumAge", ""),
        "max_age": eligibility.get("maximumAge", ""),
        "sex": eligibility.get("sex", "ALL"),
        "conditions": conditions.get("conditions", []),
        "interventions": interventions,
        "countries": sorted(countries),
        "facilities": facilities,
        "study_type": design.get("studyType", ""),
        "lead_sponsor": lead_sponsor,
        "url": f"https://clinicaltrials.gov/study/{ident.get('nctId', '')}",
    }


def score_eligibility(study: dict) -> dict[str, Any]:
    """Score a trial for patient eligibility (0-100) with detailed breakdown."""
    score = 50  # base score
    reasons: list[str] = []
    flags: list[str] = []
    criteria_text = study.get("eligibility_criteria", "").lower()
    title = (study.get("brief_title", "") + " " + study.get("official_title", "")).lower()
    summary = study.get("brief_summary", "").lower()
    combined_text = criteria_text + " " + title + " " + summary

    # --- Positive factors ---

    # Phase bonus
    phase = study.get("phase", "").upper()
    if "PHASE3" in phase.replace(" ", "") or "PHASE 3" in phase:
        score += 8
        reasons.append("+8: Phase 3 (most likely to lead to approval)")
    elif "PHASE2" in phase.replace(" ", "") or "PHASE 2" in phase:
        score += 5
        reasons.append("+5: Phase 2")
    elif "PHASE1" in phase.replace(" ", "") or "PHASE 1" in phase:
        score += 2
        reasons.append("+2: Phase 1")

    # Location bonus
    countries = set(study.get("countries", []))
    in_top_priority = countries & TOP_PRIORITY_COUNTRIES
    in_europe = countries & PRIORITY_COUNTRIES
    if in_top_priority:
        score += 15
        reasons.append(f"+15: Available in {', '.join(sorted(in_top_priority))}")
    elif in_europe:
        score += 8
        reasons.append(f"+8: Available in Europe ({', '.join(sorted(in_europe))})")

    # Recruiting status bonus
    status = study.get("overall_status", "")
    if status == "RECRUITING":
        score += 5
        reasons.append("+5: Actively recruiting")
    elif status == "NOT_YET_RECRUITING":
        score += 2
        reasons.append("+2: Not yet recruiting (may open soon)")

    # Mutation-specific targeting
    mutation_matches = []
    if any(term in combined_text for term in ["idh2", "idh-2", "enasidenib", "olutasidenib", "vorasidenib"]):
        mutation_matches.append("IDH2")
        score += 10
    if any(term in combined_text for term in ["ptpn11", "shp2", "shp-2", "tno155", "rmc-4550", "jab-3312", "bbp-398"]):
        mutation_matches.append("PTPN11/SHP2")
        score += 10
    if any(term in combined_text for term in ["ezh2", "tazemetostat", "prc2"]):
        mutation_matches.append("EZH2/PRC2")
        score += 8
    if any(term in combined_text for term in ["dnmt3a", "hypomethylat", "azacitidine", "decitabine"]):
        mutation_matches.append("DNMT3A/HMA")
        score += 5
    if any(term in combined_text for term in ["setbp1", "pp2a"]):
        mutation_matches.append("SETBP1/PP2A")
        score += 10
    if mutation_matches:
        reasons.append(f"+bonus: Targets {', '.join(mutation_matches)}")

    # AML/MDS specific
    if any(term in combined_text for term in ["acute myeloid", "aml", "myelodysplastic", "mds"]):
        score += 5
        reasons.append("+5: AML/MDS specific trial")

    # Post-transplant relevance
    if any(term in combined_text for term in ["post-transplant", "post transplant", "after transplant",
                                               "allogeneic", "hsct", "hct", "maintenance after"]):
        score += 8
        reasons.append("+8: Post-transplant relevant")

    # Relapse/refractory
    if any(term in combined_text for term in ["relapsed", "refractory", "r/r"]):
        score += 3
        reasons.append("+3: Relapsed/refractory eligible")

    # Monosomy 7 / adverse cytogenetics
    if any(term in combined_text for term in ["monosomy 7", "adverse cytogenetic", "adverse risk",
                                               "unfavorable", "high risk", "high-risk"]):
        score += 5
        reasons.append("+5: Includes adverse cytogenetics/monosomy 7")

    # --- Negative factors (exclusion risks) ---

    # Age exclusion
    min_age_str = study.get("min_age", "")
    max_age_str = study.get("max_age", "")
    if max_age_str:
        max_age_num = _parse_age(max_age_str)
        if max_age_num and max_age_num < 36:
            score -= 50
            flags.append("EXCLUDED: Max age < 36")
    if min_age_str:
        min_age_num = _parse_age(min_age_str)
        if min_age_num and min_age_num > 36:
            score -= 50
            flags.append("EXCLUDED: Min age > 36")

    # Sex exclusion
    sex = study.get("sex", "ALL")
    if sex == "FEMALE":
        score -= 50
        flags.append("EXCLUDED: Female only")

    # Specific exclusion patterns in criteria
    exclusion_patterns = [
        (r"prior\s+(?:allogeneic|allo)", "May exclude prior allo-HSCT recipients"),
        (r"no\s+prior\s+(?:stem\s+cell|transplant|hsct|hct)", "May exclude prior HSCT"),
        (r"exclude.*(?:prior|previous)\s+transplant", "May exclude prior transplant"),
        (r"flt3.*(?:mutation|positive|itd)", "May require FLT3 mutation (patient is FLT3-negative)"),
        (r"npm1.*(?:mutation|positive|mutated)", "May require NPM1 mutation (patient is NPM1-negative)"),
        (r"(?:kmt2a|mll)\s+(?:rearrange|transloc|fusion)", "May require KMT2A rearrangement (patient negative)"),
    ]

    # Check exclusion section specifically
    exclusion_section = ""
    if "exclusion" in criteria_text:
        exc_idx = criteria_text.index("exclusion")
        exclusion_section = criteria_text[exc_idx:]

    inclusion_section = ""
    if "inclusion" in criteria_text:
        inc_start = criteria_text.index("inclusion")
        inc_end = criteria_text.index("exclusion") if "exclusion" in criteria_text else len(criteria_text)
        inclusion_section = criteria_text[inc_start:inc_end]

    for pattern, note in exclusion_patterns:
        if re.search(pattern, exclusion_section):
            score -= 10
            flags.append(f"-10: {note} (in exclusion criteria)")
        elif re.search(pattern, inclusion_section):
            # If it's in inclusion as a requirement, check if patient meets it
            if "flt3" in pattern or "npm1" in pattern or "kmt2a" in pattern:
                score -= 15
                flags.append(f"-15: {note} (required, patient negative)")

    # IDH2 requirement check -- patient has IDH2 R140Q, this is a positive
    if re.search(r"idh2.*(?:mutation|positive|mutated|r140)", inclusion_section):
        score += 5
        reasons.append("+5: Requires IDH2 mutation (patient has R140Q)")

    # Newly diagnosed only (patient is post-HSCT, not newly diagnosed)
    if re.search(r"newly\s+diagnosed", inclusion_section) and not re.search(r"relapsed|refractory", inclusion_section):
        score -= 10
        flags.append("-10: Newly diagnosed only (patient is post-HSCT)")

    # ECOG/Karnofsky check
    if re.search(r"ecog.*[01]", inclusion_section) or re.search(r"karnofsky.*(?:8|9|10)0", inclusion_section):
        # Patient is Karnofsky 70-80%, ECOG ~1-2
        if re.search(r"ecog.*0(?:\s|$|,)", inclusion_section):
            score -= 5
            flags.append("-5: May require ECOG 0 (patient ~ECOG 1-2)")

    # Cap score
    score = max(0, min(100, score))

    return {
        "score": score,
        "reasons": reasons,
        "flags": flags,
        "mutation_matches": mutation_matches,
        "in_priority_country": bool(in_top_priority),
        "in_europe": bool(in_europe),
    }


def _parse_age(age_str: str) -> int | None:
    """Parse age string like '18 Years' into integer."""
    match = re.search(r"(\d+)", age_str)
    return int(match.group(1)) if match else None


def run_all_searches() -> dict[str, list[dict]]:
    """Run all search queries against ClinicalTrials.gov."""
    all_results: dict[str, list[dict]] = {}
    seen_ncts: set[str] = set()
    total_queries = sum(len(q["queries"]) for q in SEARCH_QUERIES)
    query_num = 0

    for category_config in SEARCH_QUERIES:
        category = category_config["category"]
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"Target: {category_config['target_mutation']}")
        print(f"{'='*60}")

        category_results: list[dict] = []

        for query_def in category_config["queries"]:
            query_num += 1
            terms = query_def["terms"]
            label = query_def["label"]
            print(f"\n  [{query_num}/{total_queries}] {label}")
            print(f"    Query: {terms}")

            data = query_ctgov(terms)
            studies = data.get("studies", [])
            total = data.get("totalCount", 0)
            print(f"    Found: {total} studies ({len(studies)} returned)")

            for study_raw in studies:
                study_info = extract_study_info(study_raw)
                nct_id = study_info["nct_id"]

                if nct_id in seen_ncts:
                    continue
                seen_ncts.add(nct_id)

                # Score eligibility
                eligibility = score_eligibility(study_info)
                study_info["eligibility_score"] = eligibility["score"]
                study_info["scoring_details"] = eligibility
                study_info["search_category"] = category
                study_info["search_label"] = label
                study_info["target_mutation"] = category_config["target_mutation"]
                study_info["category_rationale"] = category_config["rationale"]

                category_results.append(study_info)
                print(f"    + {nct_id}: score={eligibility['score']} | {study_info['brief_title'][:70]}")

            time.sleep(CT_RATE_LIMIT_DELAY)

        # Sort by eligibility score descending
        category_results.sort(key=lambda x: x["eligibility_score"], reverse=True)
        all_results[category] = category_results
        print(f"\n  Category total: {len(category_results)} unique trials")

    return all_results


def generate_report(results: dict[str, list[dict]], run_metadata: dict) -> str:
    """Generate a ranked markdown report from trial matching results."""
    lines: list[str] = []
    lines.append("# Clinical Trial Matching Report")
    lines.append(f"## Patient: DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + EZH2 V662A + Monosomy 7")
    lines.append("")
    lines.append(f"*Generated: {run_metadata['timestamp']}*")
    lines.append(f"*Source: ClinicalTrials.gov API v2*")
    lines.append(f"*Total unique trials found: {run_metadata['total_unique_trials']}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Patient summary
    lines.append("## Patient Summary")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Age/Sex | 36M |")
    lines.append("| Diagnosis | MDS-AML (MDS-IB2) |")
    lines.append("| ELN 2022 Risk | Adverse |")
    lines.append("| Cytogenetics | Monosomy 7 (45,XY,-7) |")
    lines.append("| HSCT Status | Post allo-HSCT (Nov 2023), >99% donor chimerism |")
    lines.append("| Disease Status | Complete remission, MRD negative |")
    lines.append("| GvHD | Mild chronic, topical steroids only |")
    lines.append("| Mutations | DNMT3A R882H (39%), EZH2 V662A (59%), SETBP1 G870S (34%), PTPN11 E76Q (29%), IDH2 R140Q (2%) |")
    lines.append("| Negative | FLT3-ITD, FLT3-TKD, NPM1 |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Top-ranked trials across all categories
    all_trials = []
    for category_trials in results.values():
        all_trials.extend(category_trials)
    all_trials.sort(key=lambda x: x["eligibility_score"], reverse=True)

    lines.append("## Top 20 Trials (All Categories, Ranked by Eligibility Score)")
    lines.append("")
    _write_trial_table(lines, all_trials[:20])
    lines.append("")
    lines.append("---")
    lines.append("")

    # Trials in priority countries
    nordic_trials = [t for t in all_trials if t["scoring_details"]["in_priority_country"]]
    if nordic_trials:
        lines.append("## Trials Available in Norway/Sweden/Denmark/Germany/Belgium/Netherlands")
        lines.append("")
        lines.append(f"*{len(nordic_trials)} trials found in priority countries*")
        lines.append("")
        _write_trial_table(lines, nordic_trials)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Per-category sections
    for category_config in SEARCH_QUERIES:
        category = category_config["category"]
        trials = results.get(category, [])
        lines.append(f"## {category}")
        lines.append("")
        lines.append(f"**Target:** {category_config['target_mutation']}")
        lines.append(f"**Rationale:** {category_config['rationale']}")
        lines.append(f"**Trials found:** {len(trials)}")
        lines.append("")

        if trials:
            _write_trial_table(lines, trials[:10])

            # Detailed cards for top 3
            lines.append("")
            lines.append("### Top Trials - Details")
            lines.append("")
            for trial in trials[:3]:
                _write_trial_card(lines, trial)
        else:
            lines.append("*No active trials found for this search.*")
            lines.append("")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Scoring methodology
    lines.append("## Scoring Methodology")
    lines.append("")
    lines.append("Each trial is scored 0-100 based on:")
    lines.append("")
    lines.append("| Factor | Points |")
    lines.append("|--------|--------|")
    lines.append("| Base score | 50 |")
    lines.append("| Phase 3 | +8 |")
    lines.append("| Phase 2 | +5 |")
    lines.append("| Phase 1 | +2 |")
    lines.append("| Available in NO/SE/DK/DE/BE/NL | +15 |")
    lines.append("| Available in wider Europe | +8 |")
    lines.append("| Actively recruiting | +5 |")
    lines.append("| Targets patient's mutation | +5 to +10 |")
    lines.append("| AML/MDS specific | +5 |")
    lines.append("| Post-transplant relevant | +8 |")
    lines.append("| Includes adverse cytogenetics | +5 |")
    lines.append("| Relapsed/refractory eligible | +3 |")
    lines.append("| Age exclusion | -50 |")
    lines.append("| Requires FLT3/NPM1/KMT2A (patient negative) | -15 |")
    lines.append("| Newly diagnosed only | -10 |")
    lines.append("| May exclude prior HSCT | -10 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Disclaimer: This report is for research purposes only. ")
    lines.append("Trial eligibility must be confirmed by the treating haematologist-oncologist.*")

    return "\n".join(lines)


def _write_trial_table(lines: list[str], trials: list[dict]) -> None:
    """Write a markdown table of trials."""
    lines.append("| Score | NCT ID | Phase | Status | Title | Countries | Mutations |")
    lines.append("|------:|--------|-------|--------|-------|-----------|-----------|")
    for t in trials:
        nct = t["nct_id"]
        phase = t["phase"].replace("PHASE", "Ph").replace(",", "/")
        status = t["overall_status"].replace("_", " ").title()
        title = t["brief_title"][:60]
        if len(t["brief_title"]) > 60:
            title += "..."
        countries = ", ".join(t["countries"][:5])
        if len(t["countries"]) > 5:
            countries += f" +{len(t['countries'])-5}"
        mutations = ", ".join(t["scoring_details"].get("mutation_matches", []))
        flags_str = ""
        if t["scoring_details"].get("flags"):
            flags_str = " **!**"
        lines.append(
            f"| {t['eligibility_score']} | [{nct}]({t['url']}) | {phase} | {status} | {title}{flags_str} | {countries} | {mutations} |"
        )


def _write_trial_card(lines: list[str], trial: dict) -> None:
    """Write a detailed trial card."""
    lines.append(f"#### [{trial['nct_id']}]({trial['url']}) - Score: {trial['eligibility_score']}/100")
    lines.append("")
    lines.append(f"**{trial['brief_title']}**")
    lines.append("")
    lines.append(f"- **Phase:** {trial['phase']}")
    lines.append(f"- **Status:** {trial['overall_status']}")
    lines.append(f"- **Sponsor:** {trial['lead_sponsor']}")
    lines.append(f"- **Enrollment:** {trial.get('enrollment', 'N/A')}")
    lines.append(f"- **Countries:** {', '.join(trial['countries'])}")

    # Priority country facilities
    priority_facilities = [
        f for f in trial.get("facilities", [])
        if f.get("country") in TOP_PRIORITY_COUNTRIES
    ]
    if priority_facilities:
        lines.append(f"- **Priority country sites:**")
        for fac in priority_facilities[:5]:
            lines.append(f"  - {fac.get('facility', 'N/A')}, {fac.get('city', '')}, {fac.get('country', '')}")

    # Interventions
    if trial.get("interventions"):
        lines.append(f"- **Interventions:**")
        for intv in trial["interventions"][:3]:
            lines.append(f"  - {intv['name']} ({intv['type']})")

    # Scoring details
    scoring = trial.get("scoring_details", {})
    if scoring.get("reasons"):
        lines.append(f"- **Scoring:**")
        for reason in scoring["reasons"]:
            lines.append(f"  - {reason}")
    if scoring.get("flags"):
        lines.append(f"- **Flags:**")
        for flag in scoring["flags"]:
            lines.append(f"  - {flag}")

    # Summary
    if trial.get("brief_summary"):
        summary = trial["brief_summary"][:300]
        if len(trial["brief_summary"]) > 300:
            summary += "..."
        lines.append(f"- **Summary:** {summary}")

    lines.append("")


def main() -> None:
    """Main entry point."""
    print("=" * 70)
    print("CLINICAL TRIAL MATCHING - ClinicalTrials.gov API v2")
    print(f"Patient: 36M, MDS-AML, 5 driver mutations + monosomy 7, post-HSCT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    results = run_all_searches()

    # Compute summary stats
    all_trials = []
    for category_trials in results.values():
        all_trials.extend(category_trials)
    all_trials.sort(key=lambda x: x["eligibility_score"], reverse=True)

    total_unique = len(all_trials)
    nordic_count = sum(1 for t in all_trials if t["scoring_details"]["in_priority_country"])
    europe_count = sum(1 for t in all_trials if t["scoring_details"]["in_europe"])
    high_score = sum(1 for t in all_trials if t["eligibility_score"] >= 70)

    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_unique_trials": total_unique,
        "trials_in_priority_countries": nordic_count,
        "trials_in_europe": europe_count,
        "trials_score_70_plus": high_score,
        "categories_searched": len(SEARCH_QUERIES),
        "total_queries": sum(len(q["queries"]) for q in SEARCH_QUERIES),
        "patient_mutations": list(PATIENT["mutations"].keys()),
        "api_version": "ClinicalTrials.gov API v2",
    }

    # Save JSON
    output = {
        "metadata": run_metadata,
        "patient": PATIENT,
        "results_by_category": {
            cat: [
                {k: v for k, v in t.items() if k != "eligibility_criteria"}
                for t in trials
            ]
            for cat, trials in results.items()
        },
        "all_trials_ranked": [
            {
                "nct_id": t["nct_id"],
                "eligibility_score": t["eligibility_score"],
                "brief_title": t["brief_title"],
                "phase": t["phase"],
                "overall_status": t["overall_status"],
                "countries": t["countries"],
                "search_category": t["search_category"],
                "target_mutation": t["target_mutation"],
                "scoring_details": t["scoring_details"],
                "url": t["url"],
            }
            for t in all_trials
        ],
    }

    JSON_OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nJSON saved: {JSON_OUTPUT}")

    # Generate and save report
    report = generate_report(results, run_metadata)
    REPORT_OUTPUT.write_text(report)
    print(f"Report saved: {REPORT_OUTPUT}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total unique trials: {total_unique}")
    print(f"Trials in NO/SE/DK/DE/BE/NL: {nordic_count}")
    print(f"Trials in wider Europe: {europe_count}")
    print(f"Trials scoring >= 70: {high_score}")
    print()

    per_category = {cat: len(trials) for cat, trials in results.items()}
    for cat, count in per_category.items():
        print(f"  {cat}: {count} trials")

    if all_trials:
        print(f"\nTop 5 trials:")
        for t in all_trials[:5]:
            countries_str = ", ".join(t["countries"][:3])
            print(f"  {t['eligibility_score']:3d} | {t['nct_id']} | {t['phase']:20s} | {countries_str:30s} | {t['brief_title'][:50]}")


if __name__ == "__main__":
    main()
