#!/usr/bin/env python3
"""
clinical_trial_search.py -- ClinicalTrials.gov search for patient-specific trials.

Targeted search using ClinicalTrials.gov API v2 for trials relevant to a patient
with DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + EZH2 V662A,
monosomy 7, ELN Adverse, IPSS-M Very High, post-allo-HSCT.

Focuses on 6 specific query categories:
  1. IDH2 AND AML (enasidenib, olutasidenib)
  2. SHP2 inhibitor AND myeloid (TNO155, RMC-4630, JAB-3312)
  3. venetoclax AND azacitidine AND IDH2 (combination trials)
  4. PTPN11 AND leukemia
  5. MDS AND monosomy 7
  6. myeloid AND enasidenib AND venetoclax (ENAVEN-AML)

Inputs:
    - ClinicalTrials.gov API v2 (remote, no key required)

Outputs:
    - mutation_profile/results/ai_research/clinical_trials.json
    - mutation_profile/results/ai_research/clinical_trials_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/clinical_trial_search.py

Runtime: ~30-90 seconds (API rate limits)
Dependencies: requests
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # mrna-hematology-research/
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = RESULTS_DIR / "clinical_trials.json"
REPORT_OUTPUT = RESULTS_DIR / "clinical_trials_report.md"

# ---------------------------------------------------------------------------
# ClinicalTrials.gov API v2 configuration
# ---------------------------------------------------------------------------
CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
CT_PAGE_SIZE = 50
CT_RATE_LIMIT_DELAY = 0.6  # seconds between requests

# ---------------------------------------------------------------------------
# Patient profile
# ---------------------------------------------------------------------------
PATIENT = {
    "age": 36,
    "sex": "MALE",
    "diagnosis": "MDS-AML (MDS-IB2)",
    "eln_risk": "Adverse",
    "ipssm": "Very High (2.976)",
    "cytogenetics": "Monosomy 7 (45,XY,-7)",
    "hsct_status": "Post allo-HSCT (Nov 2023)",
    "mutations": [
        {"gene": "EZH2", "variant": "V662A", "vaf": 0.59, "actionable": False,
         "note": "Loss-of-function, tazemetostat contraindicated"},
        {"gene": "DNMT3A", "variant": "R882H", "vaf": 0.39, "actionable": False,
         "note": "Founder clone, no direct inhibitor"},
        {"gene": "SETBP1", "variant": "G870S", "vaf": 0.34, "actionable": False,
         "note": "No targeted therapy exists"},
        {"gene": "PTPN11", "variant": "E76Q", "vaf": 0.29, "actionable": True,
         "note": "SHP2 inhibitors in Phase I/II"},
        {"gene": "IDH2", "variant": "R140Q", "vaf": 0.02, "actionable": True,
         "note": "Enasidenib FDA-approved 2017"},
    ],
    "key_actionable_targets": ["IDH2 R140Q (enasidenib)", "PTPN11 E76Q (SHP2 inhibitors)"],
}

# ---------------------------------------------------------------------------
# Search queries -- the 6 specified searches
# ---------------------------------------------------------------------------
SEARCH_QUERIES: list[dict[str, str]] = [
    {
        "id": "idh2_aml",
        "query": "IDH2 AND AML",
        "label": "IDH2 + AML (enasidenib, olutasidenib trials)",
        "target": "IDH2 R140Q",
        "rationale": "FDA-approved enasidenib targets IDH2 R140Q. Olutasidenib is dual IDH1/IDH2. "
                     "Patient has IDH2 R140Q subclone at 2% VAF -- actionable at relapse.",
    },
    {
        "id": "shp2_myeloid",
        "query": "SHP2 inhibitor AND myeloid",
        "label": "SHP2 inhibitors in myeloid (TNO155, RMC-4630, JAB-3312)",
        "target": "PTPN11 E76Q",
        "rationale": "PTPN11 E76Q is a gain-of-function SHP2 mutation driving RAS-MAPK signaling. "
                     "Multiple SHP2 inhibitors in Phase I/II: TNO155, RMC-4630 (now RMC-4550), JAB-3312, BBP-398.",
    },
    {
        "id": "ven_aza_idh2",
        "query": "venetoclax AND azacitidine AND IDH2",
        "label": "Venetoclax + azacitidine + IDH2 combination trials",
        "target": "IDH2 R140Q + backbone therapy",
        "rationale": "Ven/aza is standard-of-care backbone for AML. Adding IDH2 inhibitor to ven/aza "
                     "is the leading combination strategy for IDH2-mutated AML.",
    },
    {
        "id": "ptpn11_leukemia",
        "query": "PTPN11 AND leukemia",
        "label": "PTPN11-targeted leukemia trials",
        "target": "PTPN11 E76Q",
        "rationale": "Direct search for trials targeting PTPN11 mutations in any leukemia subtype.",
    },
    {
        "id": "mds_mono7",
        "query": "MDS AND monosomy 7",
        "label": "MDS with monosomy 7 trials",
        "target": "Monosomy 7 cytogenetics",
        "rationale": "Monosomy 7 defines adverse risk and may be specifically targeted in trial "
                     "eligibility criteria. Patient has monosomy 7 in 90% of metaphases at diagnosis.",
    },
    {
        "id": "enaven",
        "query": "myeloid AND enasidenib AND venetoclax",
        "label": "ENAVEN-AML and similar enasidenib + venetoclax trials",
        "target": "IDH2 R140Q + combination therapy",
        "rationale": "ENAVEN-AML (NCT05102370) is the key trial combining enasidenib with venetoclax "
                     "for IDH2-mutated AML. Triple combination with azacitidine is the optimal strategy.",
    },
]

# Countries to prioritize (patient is in Norway)
PRIORITY_COUNTRIES = {
    "Norway", "Sweden", "Denmark", "Germany", "Netherlands", "Belgium",
}
EUROPEAN_COUNTRIES = PRIORITY_COUNTRIES | {
    "Finland", "United Kingdom", "France", "Switzerland", "Austria",
    "Italy", "Spain", "Poland", "Czech Republic", "Ireland",
}

# ---------------------------------------------------------------------------
# API functions
# ---------------------------------------------------------------------------


def query_ctgov(
    search_terms: str,
    statuses: str = "RECRUITING,NOT_YET_RECRUITING,ENROLLING_BY_INVITATION,ACTIVE_NOT_RECRUITING",
) -> list[dict]:
    """Query ClinicalTrials.gov API v2 and return all matching studies.

    Handles pagination automatically.
    """
    all_studies: list[dict] = []
    page_token: str | None = None
    page = 0

    while True:
        params: dict[str, Any] = {
            "query.term": search_terms,
            "filter.overallStatus": statuses,
            "pageSize": CT_PAGE_SIZE,
            "fields": (
                "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,"
                "EnrollmentCount,StartDate,CompletionDate,"
                "BriefSummary,EligibilityCriteria,MinimumAge,MaximumAge,Sex,"
                "Condition,Intervention,LocationCountry,LocationFacility,LocationCity,"
                "StudyType,LeadSponsorName"
            ),
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = requests.get(CT_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  WARNING: API request failed: {e}")
            break

        studies = data.get("studies", [])
        all_studies.extend(studies)
        page += 1

        next_token = data.get("nextPageToken")
        if not next_token or not studies:
            break
        page_token = next_token
        time.sleep(CT_RATE_LIMIT_DELAY)

    return all_studies


def extract_study(study: dict) -> dict[str, Any]:
    """Extract relevant fields from a ClinicalTrials.gov API v2 study record."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    eligibility = proto.get("eligibilityModule", {})
    contacts = proto.get("contactsLocationsModule", {})
    desc = proto.get("descriptionModule", {})
    arms = proto.get("armsInterventionsModule", {})
    conditions = proto.get("conditionsModule", {})
    sponsor = proto.get("sponsorCollaboratorsModule", {})

    # Locations
    locations = contacts.get("locations", [])
    countries: set[str] = set()
    facilities: list[dict[str, str]] = []
    for loc in locations:
        country = loc.get("country", "")
        if country:
            countries.add(country)
        facility = loc.get("facility", "")
        city = loc.get("city", "")
        if facility or city:
            facilities.append({"facility": facility, "city": city, "country": country})

    # Interventions
    interventions: list[dict[str, str]] = []
    for intv in arms.get("interventions", []):
        interventions.append({
            "name": intv.get("name", ""),
            "type": intv.get("type", ""),
            "description": intv.get("description", "")[:300],
        })

    # Phase
    phases = design.get("phases", [])
    phase_str = ", ".join(phases) if phases else "Not specified"

    nct_id = ident.get("nctId", "")

    return {
        "nct_id": nct_id,
        "brief_title": ident.get("briefTitle", ""),
        "official_title": ident.get("officialTitle", ""),
        "overall_status": status_mod.get("overallStatus", ""),
        "phase": phase_str,
        "enrollment": design.get("enrollmentInfo", {}).get("count"),
        "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
        "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
        "brief_summary": desc.get("briefSummary", "")[:500],
        "eligibility_criteria": eligibility.get("eligibilityCriteria", ""),
        "min_age": eligibility.get("minimumAge", ""),
        "max_age": eligibility.get("maximumAge", ""),
        "sex": eligibility.get("sex", "ALL"),
        "conditions": conditions.get("conditions", []),
        "interventions": interventions,
        "countries": sorted(countries),
        "facilities": facilities,
        "lead_sponsor": sponsor.get("leadSponsor", {}).get("name", "Unknown"),
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
    }


def _parse_age(age_str: str) -> int | None:
    """Parse age string like '18 Years' to int."""
    m = re.search(r"(\d+)", age_str)
    return int(m.group(1)) if m else None


def score_trial(trial: dict) -> dict[str, Any]:
    """Score a trial 0-100 for relevance to the patient profile.

    Returns a dict with score, positive reasons, and warning flags.
    """
    score = 50
    reasons: list[str] = []
    flags: list[str] = []

    criteria = trial.get("eligibility_criteria", "").lower()
    title = (trial.get("brief_title", "") + " " + trial.get("official_title", "")).lower()
    summary = trial.get("brief_summary", "").lower()
    text = f"{criteria} {title} {summary}"

    # --- Phase ---
    phase = trial.get("phase", "").upper()
    if "PHASE3" in phase.replace(" ", "") or "PHASE 3" in phase:
        score += 8
        reasons.append("+8 Phase 3")
    elif "PHASE2" in phase.replace(" ", "") or "PHASE 2" in phase:
        score += 5
        reasons.append("+5 Phase 2")
    elif "PHASE1" in phase.replace(" ", "") or "PHASE 1" in phase:
        score += 2
        reasons.append("+2 Phase 1")

    # --- Geography ---
    countries = set(trial.get("countries", []))
    in_priority = countries & PRIORITY_COUNTRIES
    in_europe = countries & EUROPEAN_COUNTRIES
    if in_priority:
        score += 15
        reasons.append(f"+15 Available in {', '.join(sorted(in_priority))}")
    elif in_europe:
        score += 8
        reasons.append(f"+8 Available in Europe ({', '.join(sorted(in_europe))})")

    # --- Recruiting ---
    status = trial.get("overall_status", "")
    if status == "RECRUITING":
        score += 5
        reasons.append("+5 Actively recruiting")
    elif status == "NOT_YET_RECRUITING":
        score += 2
        reasons.append("+2 Not yet recruiting")

    # --- Mutation targeting ---
    mutation_hits: list[str] = []
    if any(t in text for t in ["idh2", "enasidenib", "olutasidenib", "vorasidenib"]):
        mutation_hits.append("IDH2")
        score += 10
    if any(t in text for t in ["ptpn11", "shp2", "shp-2", "tno155", "rmc-4550",
                                 "rmc-4630", "jab-3312", "bbp-398"]):
        mutation_hits.append("PTPN11/SHP2")
        score += 10
    if any(t in text for t in ["dnmt3a", "azacitidine", "decitabine", "hypomethylat"]):
        mutation_hits.append("DNMT3A/HMA")
        score += 5
    if any(t in text for t in ["setbp1", "pp2a"]):
        mutation_hits.append("SETBP1")
        score += 10
    if any(t in text for t in ["ezh2", "prc2"]):
        mutation_hits.append("EZH2")
        score += 8
    if mutation_hits:
        reasons.append(f"Targets: {', '.join(mutation_hits)}")

    # --- Disease match ---
    if any(t in text for t in ["acute myeloid", "aml", "myelodysplastic", "mds"]):
        score += 5
        reasons.append("+5 AML/MDS specific")

    # --- Post-transplant ---
    if any(t in text for t in ["post-transplant", "post transplant", "after transplant",
                                 "allogeneic", "hsct", "maintenance after"]):
        score += 8
        reasons.append("+8 Post-transplant relevant")

    # --- Relapsed/refractory ---
    if any(t in text for t in ["relapsed", "refractory", "r/r"]):
        score += 3
        reasons.append("+3 R/R eligible")

    # --- Adverse cytogenetics ---
    if any(t in text for t in ["monosomy 7", "adverse cytogenetic", "adverse risk",
                                 "unfavorable", "high-risk", "high risk"]):
        score += 5
        reasons.append("+5 Adverse cytogenetics/monosomy 7")

    # --- Venetoclax ---
    if "venetoclax" in text:
        score += 3
        reasons.append("+3 Venetoclax-based")

    # --- Negative factors ---
    max_age_str = trial.get("max_age", "")
    if max_age_str:
        max_age = _parse_age(max_age_str)
        if max_age and max_age < 36:
            score -= 50
            flags.append("EXCLUDED: Max age < 36")

    min_age_str = trial.get("min_age", "")
    if min_age_str:
        min_age = _parse_age(min_age_str)
        if min_age and min_age > 36:
            score -= 50
            flags.append("EXCLUDED: Min age > 36")

    if trial.get("sex") == "FEMALE":
        score -= 50
        flags.append("EXCLUDED: Female only")

    # Check inclusion/exclusion sections
    inc_section = ""
    exc_section = ""
    if "inclusion" in criteria:
        inc_start = criteria.index("inclusion")
        inc_end = criteria.index("exclusion") if "exclusion" in criteria else len(criteria)
        inc_section = criteria[inc_start:inc_end]
    if "exclusion" in criteria:
        exc_section = criteria[criteria.index("exclusion"):]

    # Prior HSCT exclusion
    if re.search(r"no\s+prior\s+(?:stem\s+cell|transplant|hsct|hct)", exc_section):
        score -= 10
        flags.append("-10 May exclude prior HSCT")

    # Requires FLT3/NPM1 (patient negative)
    for marker in ["flt3", "npm1"]:
        if re.search(rf"{marker}.*(?:mutation|positive|mutated)", inc_section):
            score -= 15
            flags.append(f"-15 Requires {marker.upper()} mutation (patient negative)")

    # IDH2 required = positive for patient
    if re.search(r"idh2.*(?:mutation|positive|mutated|r140)", inc_section):
        score += 5
        reasons.append("+5 Requires IDH2 mutation (patient has R140Q)")

    # Newly diagnosed only
    if re.search(r"newly\s+diagnosed", inc_section) and not re.search(r"relapsed|refractory", inc_section):
        score -= 10
        flags.append("-10 Newly diagnosed only (patient is post-HSCT)")

    score = max(0, min(100, score))

    return {
        "score": score,
        "reasons": reasons,
        "flags": flags,
        "mutation_hits": mutation_hits,
        "in_priority_country": bool(in_priority),
        "in_europe": bool(in_europe),
    }


# ---------------------------------------------------------------------------
# Main search and reporting
# ---------------------------------------------------------------------------


def run_searches() -> tuple[dict[str, list[dict]], dict[str, Any]]:
    """Run all 6 search queries and return deduplicated, scored results."""
    results_by_query: dict[str, list[dict]] = {}
    seen_ncts: set[str] = set()
    all_trials: list[dict] = []

    for i, sq in enumerate(SEARCH_QUERIES, 1):
        qid = sq["id"]
        query = sq["query"]
        label = sq["label"]
        print(f"\n[{i}/{len(SEARCH_QUERIES)}] {label}")
        print(f"  Query: {query}")

        raw_studies = query_ctgov(query)
        print(f"  API returned: {len(raw_studies)} studies")

        query_trials: list[dict] = []
        for study_raw in raw_studies:
            trial = extract_study(study_raw)
            nct_id = trial["nct_id"]
            if not nct_id:
                continue

            scoring = score_trial(trial)
            trial["eligibility_score"] = scoring["score"]
            trial["scoring"] = scoring
            trial["search_query_id"] = qid
            trial["search_label"] = label
            trial["target"] = sq["target"]

            if nct_id not in seen_ncts:
                seen_ncts.add(nct_id)
                all_trials.append(trial)

            query_trials.append(trial)
            status_icon = "+" if scoring["score"] >= 70 else " "
            print(f"  {status_icon} {nct_id} score={scoring['score']:3d} | {trial['brief_title'][:65]}")

        query_trials.sort(key=lambda t: t["eligibility_score"], reverse=True)
        results_by_query[qid] = query_trials
        print(f"  Subtotal: {len(query_trials)} trials ({len([t for t in query_trials if t['nct_id'] in seen_ncts])} unique so far)")

        time.sleep(CT_RATE_LIMIT_DELAY)

    # Rank all unique trials
    all_trials.sort(key=lambda t: t["eligibility_score"], reverse=True)

    # Summary stats
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_unique_trials": len(all_trials),
        "total_recruiting": sum(1 for t in all_trials if t["overall_status"] == "RECRUITING"),
        "trials_in_priority_countries": sum(1 for t in all_trials if t["scoring"]["in_priority_country"]),
        "trials_in_europe": sum(1 for t in all_trials if t["scoring"]["in_europe"]),
        "trials_score_70_plus": sum(1 for t in all_trials if t["eligibility_score"] >= 70),
        "trials_score_80_plus": sum(1 for t in all_trials if t["eligibility_score"] >= 80),
        "queries_executed": len(SEARCH_QUERIES),
        "api": "ClinicalTrials.gov API v2",
        "filter": "RECRUITING, NOT_YET_RECRUITING, ENROLLING_BY_INVITATION, ACTIVE_NOT_RECRUITING",
        "per_query_counts": {sq["id"]: len(results_by_query.get(sq["id"], [])) for sq in SEARCH_QUERIES},
    }

    return results_by_query, meta


def build_json_output(
    results_by_query: dict[str, list[dict]],
    all_trials_ranked: list[dict],
    meta: dict,
) -> dict:
    """Build the JSON output structure, stripping bulky eligibility criteria text."""

    def slim(trial: dict) -> dict:
        return {k: v for k, v in trial.items() if k != "eligibility_criteria"}

    return {
        "metadata": meta,
        "patient": PATIENT,
        "search_queries": SEARCH_QUERIES,
        "results_by_query": {qid: [slim(t) for t in trials] for qid, trials in results_by_query.items()},
        "all_trials_ranked": [
            {
                "rank": i + 1,
                "nct_id": t["nct_id"],
                "eligibility_score": t["eligibility_score"],
                "brief_title": t["brief_title"],
                "phase": t["phase"],
                "overall_status": t["overall_status"],
                "countries": t["countries"],
                "interventions": [iv["name"] for iv in t.get("interventions", [])],
                "search_query_id": t["search_query_id"],
                "target": t["target"],
                "scoring": t["scoring"],
                "url": t["url"],
            }
            for i, t in enumerate(all_trials_ranked)
        ],
    }


def generate_report(
    results_by_query: dict[str, list[dict]],
    all_trials_ranked: list[dict],
    meta: dict,
) -> str:
    """Generate a prioritized markdown report."""
    lines: list[str] = []

    lines.append("# Clinical Trial Search Report")
    lines.append("")
    lines.append(f"**Generated:** {meta['timestamp']}")
    lines.append(f"**Source:** ClinicalTrials.gov API v2")
    lines.append(f"**Total unique trials:** {meta['total_unique_trials']}")
    lines.append(f"**Actively recruiting:** {meta['total_recruiting']}")
    lines.append(f"**Score >= 70:** {meta['trials_score_70_plus']}")
    lines.append(f"**Score >= 80:** {meta['trials_score_80_plus']}")
    lines.append(f"**In NO/SE/DK/DE/NL/BE:** {meta['trials_in_priority_countries']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Patient summary
    lines.append("## Patient Profile")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Age/Sex | 36M |")
    lines.append("| Diagnosis | MDS-AML (MDS-IB2) |")
    lines.append("| ELN 2022 | Adverse |")
    lines.append("| IPSS-M | Very High (2.976) |")
    lines.append("| Cytogenetics | Monosomy 7 (45,XY,-7) |")
    lines.append("| HSCT | Post allo-HSCT Nov 2023, >99% donor chimerism |")
    lines.append("| Status | Complete remission, MRD negative |")
    lines.append("| Mutations | EZH2 V662A (59%), DNMT3A R882H (39%), SETBP1 G870S (34%), PTPN11 E76Q (29%), IDH2 R140Q (2%) |")
    lines.append("| Actionable | IDH2 R140Q (enasidenib), PTPN11 E76Q (SHP2 inhibitors) |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Top trials overall
    top_trials = [t for t in all_trials_ranked if t["eligibility_score"] >= 70]
    lines.append(f"## Priority Trials (Score >= 70): {len(top_trials)} trials")
    lines.append("")
    if top_trials:
        _write_table(lines, top_trials[:25])
    else:
        lines.append("*No trials scored >= 70.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Trials in priority countries
    priority_trials = [t for t in all_trials_ranked if t["scoring"]["in_priority_country"]]
    lines.append(f"## Trials in Norway/Sweden/Denmark/Germany/Netherlands/Belgium: {len(priority_trials)}")
    lines.append("")
    if priority_trials:
        _write_table(lines, priority_trials)
        lines.append("")
        # Detailed cards for priority country trials
        lines.append("### Detailed Trial Cards (Priority Countries)")
        lines.append("")
        for trial in priority_trials[:10]:
            _write_card(lines, trial)
    else:
        lines.append("*No trials found in priority countries.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-query sections
    for sq in SEARCH_QUERIES:
        qid = sq["id"]
        trials = results_by_query.get(qid, [])
        lines.append(f"## Query: {sq['label']}")
        lines.append("")
        lines.append(f"**Search:** `{sq['query']}`")
        lines.append(f"**Target:** {sq['target']}")
        lines.append(f"**Rationale:** {sq['rationale']}")
        lines.append(f"**Results:** {len(trials)} trials")
        lines.append("")

        if trials:
            _write_table(lines, trials[:15])
            lines.append("")
            # Top 3 detail cards
            for trial in trials[:3]:
                _write_card(lines, trial)
        else:
            lines.append("*No active/recruiting trials found.*")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Scoring methodology
    lines.append("## Scoring Methodology")
    lines.append("")
    lines.append("Base score 50. Modifiers:")
    lines.append("")
    lines.append("| Factor | Points |")
    lines.append("|--------|--------|")
    lines.append("| Phase 3 / Phase 2 / Phase 1 | +8 / +5 / +2 |")
    lines.append("| NO/SE/DK/DE/NL/BE site | +15 |")
    lines.append("| Wider Europe site | +8 |")
    lines.append("| Actively recruiting | +5 |")
    lines.append("| Targets IDH2 or PTPN11/SHP2 | +10 each |")
    lines.append("| Targets SETBP1 or EZH2 | +10 / +8 |")
    lines.append("| Targets DNMT3A/HMA | +5 |")
    lines.append("| AML/MDS specific | +5 |")
    lines.append("| Post-transplant relevant | +8 |")
    lines.append("| Adverse cytogenetics | +5 |")
    lines.append("| Venetoclax-based | +3 |")
    lines.append("| R/R eligible | +3 |")
    lines.append("| Requires IDH2 (patient has) | +5 |")
    lines.append("| Age excluded | -50 |")
    lines.append("| Requires FLT3/NPM1 (patient neg) | -15 |")
    lines.append("| Newly diagnosed only | -10 |")
    lines.append("| May exclude prior HSCT | -10 |")
    lines.append("")

    return "\n".join(lines)


def _write_table(lines: list[str], trials: list[dict]) -> None:
    """Write a compact markdown table of trials."""
    lines.append("| Score | NCT ID | Phase | Status | Title | Countries | Targets |")
    lines.append("|------:|--------|-------|--------|-------|-----------|---------|")
    for t in trials:
        nct = t["nct_id"]
        phase = t["phase"].replace("PHASE", "Ph").replace(",", "/")
        status = t["overall_status"].replace("_", " ").title()
        title = t["brief_title"][:55] + ("..." if len(t["brief_title"]) > 55 else "")
        countries = ", ".join(t["countries"][:4])
        if len(t["countries"]) > 4:
            countries += f" +{len(t['countries']) - 4}"
        targets = ", ".join(t["scoring"].get("mutation_hits", []))
        flag = " **!**" if t["scoring"].get("flags") else ""
        lines.append(
            f"| {t['eligibility_score']} | [{nct}]({t['url']}) | {phase} | {status} | {title}{flag} | {countries} | {targets} |"
        )


def _write_card(lines: list[str], trial: dict) -> None:
    """Write a detailed trial card."""
    lines.append(f"#### [{trial['nct_id']}]({trial['url']}) -- Score {trial['eligibility_score']}/100")
    lines.append("")
    lines.append(f"**{trial['brief_title']}**")
    lines.append("")
    lines.append(f"- Phase: {trial['phase']}")
    lines.append(f"- Status: {trial['overall_status']}")
    lines.append(f"- Sponsor: {trial['lead_sponsor']}")
    lines.append(f"- Enrollment: {trial.get('enrollment', 'N/A')}")
    lines.append(f"- Countries: {', '.join(trial['countries'])}")

    # Priority country facilities
    pf = [f for f in trial.get("facilities", []) if f.get("country") in PRIORITY_COUNTRIES]
    if pf:
        lines.append("- Priority sites:")
        for fac in pf[:5]:
            lines.append(f"  - {fac.get('facility', 'N/A')}, {fac.get('city', '')}, {fac.get('country', '')}")

    if trial.get("interventions"):
        lines.append("- Interventions:")
        for iv in trial["interventions"][:4]:
            lines.append(f"  - {iv['name']} ({iv['type']})")

    scoring = trial.get("scoring", {})
    if scoring.get("reasons"):
        lines.append("- Scoring: " + "; ".join(scoring["reasons"]))
    if scoring.get("flags"):
        lines.append("- Flags: " + "; ".join(scoring["flags"]))

    if trial.get("brief_summary"):
        s = trial["brief_summary"][:250]
        if len(trial["brief_summary"]) > 250:
            s += "..."
        lines.append(f"- Summary: {s}")

    lines.append("")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("CLINICAL TRIAL SEARCH -- ClinicalTrials.gov API v2")
    print("Patient: 36M, MDS-AML, 5 drivers + monosomy 7, post-HSCT")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    results_by_query, meta = run_searches()

    # Collect all unique trials, ranked
    seen: set[str] = set()
    all_ranked: list[dict] = []
    for sq in SEARCH_QUERIES:
        for t in results_by_query.get(sq["id"], []):
            if t["nct_id"] not in seen:
                seen.add(t["nct_id"])
                all_ranked.append(t)
    all_ranked.sort(key=lambda t: t["eligibility_score"], reverse=True)

    # Save JSON
    json_data = build_json_output(results_by_query, all_ranked, meta)
    JSON_OUTPUT.write_text(json.dumps(json_data, indent=2, ensure_ascii=False))
    print(f"\nJSON saved: {JSON_OUTPUT}")

    # Generate and save report
    report = generate_report(results_by_query, all_ranked, meta)
    REPORT_OUTPUT.write_text(report)
    print(f"Report saved: {REPORT_OUTPUT}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total unique trials: {meta['total_unique_trials']}")
    print(f"Actively recruiting: {meta['total_recruiting']}")
    print(f"Score >= 70: {meta['trials_score_70_plus']}")
    print(f"Score >= 80: {meta['trials_score_80_plus']}")
    print(f"In priority countries: {meta['trials_in_priority_countries']}")
    print(f"In Europe: {meta['trials_in_europe']}")
    print()
    for sq in SEARCH_QUERIES:
        count = len(results_by_query.get(sq["id"], []))
        print(f"  {sq['label']}: {count} trials")

    if all_ranked:
        print(f"\nTop 10 trials:")
        for t in all_ranked[:10]:
            c = ", ".join(t["countries"][:3])
            print(f"  {t['eligibility_score']:3d} | {t['nct_id']} | {t['phase'][:15]:15s} | {c:25s} | {t['brief_title'][:50]}")


if __name__ == "__main__":
    main()
