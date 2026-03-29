#!/usr/bin/env python3
"""
cbioportal_expanded.py -- Expanded cBioPortal search for patient mutation profile.

The existing cross_database.py queries cBioPortal but uses a limited set of
myeloid keywords and only checks for the quadruple co-occurrence. This script:

1. Lists ALL cBioPortal studies and filters with broader keywords including
   hematological, blood cancer, lymphoid (some mixed-lineage studies contain
   myeloid data)
2. For each myeloid study, queries mutations in DNMT3A, IDH2, SETBP1, PTPN11
3. Searches for exact hotspot variants: DNMT3A R882H, IDH2 R140Q, SETBP1 G870S,
   PTPN11 E76Q
4. Identifies patients with 2+ of the 4 target mutations
5. Checks for the quadruple combination
6. Tracks which studies overlap with GENIE
7. Reports total unique patients with deduplication notes

Inputs:
    - cBioPortal REST API (https://www.cbioportal.org/api)

Outputs:
    - mutation_profile/results/cross_database/cbioportal_expanded.json
    - mutation_profile/results/cross_database/cbioportal_expanded_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/cbioportal_expanded.py

Runtime: ~2-5 minutes (network-dependent, many studies)
Dependencies: requests
"""

import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
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
CBIO_API_URL = "https://www.cbioportal.org/api"

TARGET_GENES = ["DNMT3A", "IDH2", "SETBP1", "PTPN11"]

ENTREZ_IDS = {
    "DNMT3A": 1788,
    "IDH2": 3418,
    "SETBP1": 26040,
    "PTPN11": 5781,
}

# Exact hotspot variants we are searching for
HOTSPOT_VARIANTS = {
    "DNMT3A": {"protein_change": "R882H", "aliases": ["p.R882H", "R882H", "p.Arg882His"]},
    "IDH2": {"protein_change": "R140Q", "aliases": ["p.R140Q", "R140Q", "p.Arg140Gln"]},
    "SETBP1": {"protein_change": "G870S", "aliases": ["p.G870S", "G870S", "p.Gly870Ser"]},
    "PTPN11": {"protein_change": "E76Q", "aliases": ["p.E76Q", "E76Q", "p.Glu76Gln"]},
}

# Broader keyword set than cross_database.py
MYELOID_KEYWORDS = [
    "aml", "mds", "myeloid", "cmml", "mpn", "leukemia",
    "myeloproliferative", "myelodysplastic",
    "hematolog", "haematolog",  # catches hematological/haematological
    "blood cancer", "myeloma",
    "lymphoma",  # some pan-heme studies
    "acute_leukemia",
    "myelofibrosis",
    "polycythemia", "essential thrombocyth",
    "jmml",
]

# Studies known to overlap with GENIE v19.0
GENIE_OVERLAP_PATTERNS = [
    "genie",
    "msk_impact",
    "mskcc",  # MSK center
    "dfci",  # Dana-Farber
    "vicc",  # Vanderbilt
    "uhn",   # University Health Network
    "grcc",  # Gustave Roussy
    "jhu",   # Johns Hopkins
    "ohsu",  # Oregon Health
    "chop",  # Children's Hospital of Philadelphia
    "nki",   # Netherlands Cancer Institute
    "cruk",  # Cancer Research UK
    "wake",  # Wake Forest
    "yale",  # Yale
    "duke",  # Duke
]

# Coding mutation types (exclude silent/intronic/UTR)
CODING_MUTATION_TYPES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "cbioportal_expanded.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "Content-Type": "application/json",
})


def cbio_get(endpoint: str, params: dict | None = None) -> Any:
    """GET request to cBioPortal API with retry logic."""
    url = f"{CBIO_API_URL}{endpoint}"
    for attempt in range(3):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                log.warning("  Rate-limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            log.warning("  GET %s returned %d", endpoint, resp.status_code)
            return None
        except requests.RequestException as e:
            log.warning("  Request error (attempt %d): %s", attempt + 1, e)
            time.sleep(2)
    return None


def cbio_post(endpoint: str, payload: dict) -> Any:
    """POST request to cBioPortal API with retry logic."""
    url = f"{CBIO_API_URL}{endpoint}"
    for attempt in range(3):
        try:
            resp = SESSION.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                log.warning("  Rate-limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            log.warning("  POST %s returned %d: %s", endpoint, resp.status_code,
                        resp.text[:200] if resp.text else "")
            return None
        except requests.RequestException as e:
            log.warning("  Request error (attempt %d): %s", attempt + 1, e)
            time.sleep(2)
    return None


def is_coding_mutation(mutation_type: str) -> bool:
    """Check if a cBioPortal mutation type is a coding variant."""
    if not mutation_type:
        return False
    # Direct match
    if mutation_type in CODING_MUTATION_TYPES:
        return True
    # Partial match for variant naming inconsistencies
    mt_lower = mutation_type.lower()
    return any(t.lower() in mt_lower for t in [
        "missense", "nonsense", "frame_shift", "in_frame",
        "splice_site", "translation_start", "nonstop",
    ])


def matches_hotspot(gene: str, protein_change: str | None) -> bool:
    """Check if a mutation matches our exact hotspot variant."""
    if not protein_change or gene not in HOTSPOT_VARIANTS:
        return False
    pc = protein_change.strip()
    aliases = HOTSPOT_VARIANTS[gene]["aliases"]
    return any(alias in pc for alias in aliases)


def is_genie_overlap(study_id: str) -> bool:
    """Check if a study likely overlaps with GENIE."""
    sid_lower = study_id.lower()
    return any(pattern in sid_lower for pattern in GENIE_OVERLAP_PATTERNS)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def discover_myeloid_studies() -> list[dict]:
    """Fetch all cBioPortal studies and filter for myeloid/hematological."""
    log.info("Step 1: Discovering myeloid/hematological studies...")
    all_studies = cbio_get("/studies", params={"projection": "DETAILED"})
    if not all_studies:
        log.error("Failed to fetch studies list from cBioPortal")
        return []

    log.info("  Total studies on cBioPortal: %d", len(all_studies))

    myeloid_studies = []
    for study in all_studies:
        sid = study.get("studyId", "").lower()
        name = study.get("name", "").lower()
        desc = study.get("description", "").lower()
        combined = f"{sid} {name} {desc}"
        if any(kw in combined for kw in MYELOID_KEYWORDS):
            myeloid_studies.append(study)

    log.info("  Myeloid/hematological studies found: %d", len(myeloid_studies))
    return myeloid_studies


def get_study_sample_count(study_id: str) -> int:
    """Get real sample count for a study."""
    sl = cbio_get(f"/sample-lists/{study_id}_all")
    if sl and isinstance(sl, dict):
        return sl.get("sampleCount", 0)
    return 0


def query_study_mutations(study_id: str, study_name: str) -> dict:
    """Query a single study for target gene mutations and hotspot variants."""
    result = {
        "study_id": study_id,
        "name": study_name,
        "genie_overlap": is_genie_overlap(study_id),
        "per_gene_any_coding": {},
        "per_gene_hotspot": {},
        "patients_2plus": [],
        "patients_3plus": [],
        "patients_4_quad": [],
        "all_mutations_detail": [],
    }

    # Get molecular profiles
    profiles = cbio_get(f"/studies/{study_id}/molecular-profiles")
    if not profiles:
        result["error"] = "no molecular profiles"
        return result

    # Find mutation profile
    mut_profile_id = None
    for prof in profiles:
        if prof.get("molecularAlterationType") == "MUTATION_EXTENDED":
            mut_profile_id = prof["molecularProfileId"]
            break

    if not mut_profile_id:
        result["error"] = "no mutation profile"
        return result

    result["mutation_profile_id"] = mut_profile_id
    sample_list_id = f"{study_id}_all"

    # Track per-patient mutations
    # patient_id -> {gene -> [mutation_details]}
    patient_mutations: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for gene in TARGET_GENES:
        payload = {
            "entrezGeneIds": [ENTREZ_IDS[gene]],
            "sampleListId": sample_list_id,
        }
        mutations = cbio_post(
            f"/molecular-profiles/{mut_profile_id}/mutations/fetch?projection=DETAILED",
            payload,
        )

        coding_patients = set()
        hotspot_patients = set()

        if mutations:
            for m in mutations:
                mut_type = m.get("mutationType", "")
                if not is_coding_mutation(mut_type):
                    continue

                patient_id = m.get("patientId", m.get("uniquePatientKey", ""))
                if not patient_id:
                    continue

                coding_patients.add(patient_id)
                protein_change = m.get("proteinChange", "") or m.get("aminoAcidChange", "")
                is_hotspot = matches_hotspot(gene, protein_change)

                if is_hotspot:
                    hotspot_patients.add(patient_id)

                mut_detail = {
                    "gene": gene,
                    "protein_change": protein_change,
                    "mutation_type": mut_type,
                    "is_hotspot": is_hotspot,
                    "patient_id": patient_id,
                    "sample_id": m.get("sampleId", ""),
                    "vaf": m.get("tumorAltCount", None),
                    "keyword": m.get("keyword", ""),
                }
                patient_mutations[patient_id][gene].append(mut_detail)

        result["per_gene_any_coding"][gene] = len(coding_patients)
        result["per_gene_hotspot"][gene] = len(hotspot_patients)

        time.sleep(0.25)  # Rate limiting

    # Find patients with 2+ target genes mutated
    for patient_id, gene_muts in patient_mutations.items():
        genes_hit = set(gene_muts.keys())
        n_genes = len(genes_hit)

        if n_genes >= 2:
            patient_entry = {
                "patient_id": patient_id,
                "genes_mutated": sorted(genes_hit),
                "n_genes": n_genes,
                "mutations": [],
                "has_exact_hotspots": {},
            }
            for gene, muts in gene_muts.items():
                for m in muts:
                    patient_entry["mutations"].append(m)
                    if m["is_hotspot"]:
                        patient_entry["has_exact_hotspots"][gene] = m["protein_change"]

            result["patients_2plus"].append(patient_entry)
            if n_genes >= 3:
                result["patients_3plus"].append(patient_entry)
            if n_genes >= 4:
                result["patients_4_quad"].append(patient_entry)

    # Store all mutation details for patients with any target gene hit
    for patient_id, gene_muts in patient_mutations.items():
        for gene, muts in gene_muts.items():
            result["all_mutations_detail"].extend(muts)

    return result


def run_expanded_search() -> dict:
    """Run the full expanded cBioPortal search."""
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).isoformat()

    log.info("=" * 70)
    log.info("cBioPortal EXPANDED Search")
    log.info("Target: DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q")
    log.info("=" * 70)

    # Step 1: Discover studies
    myeloid_studies = discover_myeloid_studies()
    if not myeloid_studies:
        return {"error": "No myeloid studies found", "timestamp": timestamp}

    # Step 2: Get sample counts and log studies
    log.info("\nStep 2: Getting sample counts for %d studies...", len(myeloid_studies))
    for study in myeloid_studies:
        study["_sample_count"] = get_study_sample_count(study["studyId"])
        time.sleep(0.15)

    # Sort by sample count descending
    myeloid_studies.sort(key=lambda s: s.get("_sample_count", 0), reverse=True)

    for s in myeloid_studies:
        overlap_tag = " [GENIE-OVERLAP]" if is_genie_overlap(s["studyId"]) else ""
        log.info("  %s: %s (n=%d)%s",
                 s["studyId"], s.get("name", "?"), s.get("_sample_count", 0), overlap_tag)

    # Step 3: Query each study
    log.info("\nStep 3: Querying mutations across %d studies...", len(myeloid_studies))
    study_results = []
    all_patients_2plus = []
    all_patients_3plus = []
    all_patients_quad = []
    global_patient_genes: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    genie_overlap_studies = []
    non_genie_studies = []

    for i, study in enumerate(myeloid_studies):
        study_id = study["studyId"]
        study_name = study.get("name", study_id)
        sample_count = study.get("_sample_count", 0)

        log.info("\n  [%d/%d] %s (%d samples)...",
                 i + 1, len(myeloid_studies), study_id, sample_count)

        result = query_study_mutations(study_id, study_name)
        result["sample_count"] = sample_count
        study_results.append(result)

        if result.get("error"):
            log.warning("    Skipped: %s", result["error"])
            continue

        # Log per-gene counts
        log.info("    Any coding: %s", result["per_gene_any_coding"])
        log.info("    Exact hotspot: %s", result["per_gene_hotspot"])
        log.info("    Patients with 2+ genes: %d", len(result["patients_2plus"]))
        if result["patients_3plus"]:
            log.info("    Patients with 3+ genes: %d", len(result["patients_3plus"]))
        if result["patients_4_quad"]:
            log.info("    QUADRUPLE MATCH: %d patients!", len(result["patients_4_quad"]))

        # Accumulate
        all_patients_2plus.extend(result["patients_2plus"])
        all_patients_3plus.extend(result["patients_3plus"])
        all_patients_quad.extend(result["patients_4_quad"])

        # Track for deduplication
        for p in result.get("all_mutations_detail", []):
            pid = p["patient_id"]
            gene = p["gene"]
            global_patient_genes[pid][gene].append({
                "study": study_id,
                **p,
            })

        if result.get("genie_overlap"):
            genie_overlap_studies.append(study_id)
        else:
            non_genie_studies.append(study_id)

        time.sleep(0.2)  # Between studies

    # Step 4: Deduplication analysis
    log.info("\n" + "=" * 70)
    log.info("Step 4: Deduplication and cross-study analysis")

    # Count unique patients with 2+ target genes across all studies
    unique_2plus = {}
    unique_3plus = {}
    unique_quad = {}

    for pid, gene_muts in global_patient_genes.items():
        n_genes = len(gene_muts)
        if n_genes >= 2:
            # Collect all mutations for this patient
            all_muts = []
            studies_seen = set()
            hotspots_found = {}
            for gene, muts in gene_muts.items():
                for m in muts:
                    all_muts.append(m)
                    studies_seen.add(m.get("study", ""))
                    if m.get("is_hotspot"):
                        hotspots_found[gene] = m.get("protein_change", "")

            entry = {
                "patient_id": pid,
                "genes_mutated": sorted(gene_muts.keys()),
                "n_genes": n_genes,
                "studies": sorted(studies_seen),
                "hotspots_found": hotspots_found,
                "mutations": all_muts,
                "in_genie_study": any(is_genie_overlap(s) for s in studies_seen),
            }
            unique_2plus[pid] = entry
            if n_genes >= 3:
                unique_3plus[pid] = entry
            if n_genes >= 4:
                unique_quad[pid] = entry

    log.info("  Unique patients with 2+ target genes: %d", len(unique_2plus))
    log.info("  Unique patients with 3+ target genes: %d", len(unique_3plus))
    log.info("  Unique patients with 4 (quadruple): %d", len(unique_quad))

    # Count how many are in GENIE-overlap studies vs independent
    n_genie_2plus = sum(1 for p in unique_2plus.values() if p["in_genie_study"])
    n_independent_2plus = len(unique_2plus) - n_genie_2plus
    log.info("  Of 2+ patients: %d in GENIE-overlap studies, %d in independent studies",
             n_genie_2plus, n_independent_2plus)

    # Step 5: Detailed hotspot analysis
    log.info("\nStep 5: Exact hotspot variant analysis")
    hotspot_summary = {}
    for gene in TARGET_GENES:
        variant = HOTSPOT_VARIANTS[gene]["protein_change"]
        total_across_studies = sum(
            r.get("per_gene_hotspot", {}).get(gene, 0)
            for r in study_results if not r.get("error")
        )
        hotspot_summary[gene] = {
            "variant": variant,
            "total_patients_across_studies": total_across_studies,
        }
        log.info("  %s %s: %d patients (across all studies, before dedup)",
                 gene, variant, total_across_studies)

    elapsed = time.time() - start_time

    # Build final result
    final_result = {
        "timestamp": timestamp,
        "runtime_seconds": round(elapsed, 1),
        "api_url": CBIO_API_URL,
        "search_config": {
            "target_genes": TARGET_GENES,
            "hotspot_variants": {g: v["protein_change"] for g, v in HOTSPOT_VARIANTS.items()},
            "myeloid_keywords": MYELOID_KEYWORDS,
        },
        "studies_summary": {
            "total_cbioportal_studies": len(cbio_get("/studies") or []) if False else "see study_results",
            "myeloid_studies_found": len(myeloid_studies),
            "myeloid_studies_with_mutation_data": sum(
                1 for r in study_results if not r.get("error")
            ),
            "genie_overlap_studies": genie_overlap_studies,
            "non_genie_studies": non_genie_studies,
            "total_samples_across_studies": sum(
                s.get("_sample_count", 0) for s in myeloid_studies
            ),
        },
        "hotspot_summary": hotspot_summary,
        "cooccurrence_summary": {
            "unique_patients_2plus_genes": len(unique_2plus),
            "unique_patients_3plus_genes": len(unique_3plus),
            "unique_patients_quadruple": len(unique_quad),
            "of_2plus_in_genie_overlap": n_genie_2plus,
            "of_2plus_in_independent": n_independent_2plus,
        },
        "patients_2plus": list(unique_2plus.values()),
        "patients_3plus": list(unique_3plus.values()),
        "patients_quadruple": list(unique_quad.values()),
        "per_study_results": [
            {k: v for k, v in r.items() if k != "all_mutations_detail"}
            for r in study_results
        ],
    }

    return final_result


def generate_report(result: dict) -> str:
    """Generate markdown report from results."""
    lines = []
    lines.append("# cBioPortal Expanded Search Report")
    lines.append("")
    lines.append(f"**Generated:** {result.get('timestamp', 'unknown')}")
    lines.append(f"**Runtime:** {result.get('runtime_seconds', '?')} seconds")
    lines.append(f"**API:** {result.get('api_url', CBIO_API_URL)}")
    lines.append("")
    lines.append("## Search Target")
    lines.append("")
    lines.append("Patient mutation profile:")
    lines.append("- DNMT3A R882H (VAF 39%)")
    lines.append("- IDH2 R140Q (VAF 2%)")
    lines.append("- SETBP1 G870S (VAF 34%)")
    lines.append("- PTPN11 E76Q (VAF 29%)")
    lines.append("")

    # Studies summary
    studies = result.get("studies_summary", {})
    lines.append("## Studies Discovered")
    lines.append("")
    lines.append(f"- **Myeloid/hematological studies found:** {studies.get('myeloid_studies_found', '?')}")
    lines.append(f"- **Studies with mutation data:** {studies.get('myeloid_studies_with_mutation_data', '?')}")
    lines.append(f"- **Total samples across studies:** {studies.get('total_samples_across_studies', '?'):,}")
    lines.append(f"- **GENIE-overlap studies:** {len(studies.get('genie_overlap_studies', []))}")
    lines.append(f"- **Independent (non-GENIE) studies:** {len(studies.get('non_genie_studies', []))}")
    lines.append("")

    # GENIE overlap
    genie_studies = studies.get("genie_overlap_studies", [])
    if genie_studies:
        lines.append("### GENIE-Overlap Studies")
        lines.append("")
        lines.append("These studies share data with AACR GENIE and should be deduplicated:")
        lines.append("")
        for sid in genie_studies:
            lines.append(f"- `{sid}`")
        lines.append("")

    non_genie = studies.get("non_genie_studies", [])
    if non_genie:
        lines.append("### Independent Studies (not in GENIE)")
        lines.append("")
        for sid in non_genie:
            lines.append(f"- `{sid}`")
        lines.append("")

    # Hotspot variants
    lines.append("## Exact Hotspot Variant Counts")
    lines.append("")
    lines.append("| Gene | Variant | Patients (all studies, pre-dedup) |")
    lines.append("|------|---------|----------------------------------|")
    for gene in TARGET_GENES:
        hs = result.get("hotspot_summary", {}).get(gene, {})
        lines.append(f"| {gene} | {hs.get('variant', '?')} | {hs.get('total_patients_across_studies', 0)} |")
    lines.append("")

    # Co-occurrence
    cooc = result.get("cooccurrence_summary", {})
    lines.append("## Co-occurrence Results")
    lines.append("")
    lines.append(f"- **Patients with 2+ target genes mutated:** {cooc.get('unique_patients_2plus_genes', 0)}")
    lines.append(f"  - In GENIE-overlap studies: {cooc.get('of_2plus_in_genie_overlap', 0)}")
    lines.append(f"  - In independent studies: {cooc.get('of_2plus_in_independent', 0)}")
    lines.append(f"- **Patients with 3+ target genes mutated:** {cooc.get('unique_patients_3plus_genes', 0)}")
    lines.append(f"- **Patients with all 4 (quadruple):** {cooc.get('unique_patients_quadruple', 0)}")
    lines.append("")

    # Detail on 2+ patients
    patients_2plus = result.get("patients_2plus", [])
    if patients_2plus:
        lines.append("## Patients with 2+ Target Gene Mutations")
        lines.append("")
        lines.append("| Patient ID | Genes Mutated | N | Exact Hotspots | Studies | GENIE Overlap |")
        lines.append("|------------|---------------|---|----------------|---------|---------------|")
        for p in sorted(patients_2plus, key=lambda x: -x.get("n_genes", 0)):
            genes = ", ".join(p.get("genes_mutated", []))
            n = p.get("n_genes", 0)
            hotspots = ", ".join(f"{g}:{v}" for g, v in p.get("hotspots_found", {}).items()) or "none"
            studies_str = ", ".join(p.get("studies", []))
            genie = "Yes" if p.get("in_genie_study") else "No"
            pid = p.get("patient_id", "?")
            # Truncate long patient IDs
            if len(pid) > 30:
                pid = pid[:27] + "..."
            lines.append(f"| {pid} | {genes} | {n} | {hotspots} | {studies_str} | {genie} |")
        lines.append("")

    # Detail on 3+ patients
    patients_3plus = result.get("patients_3plus", [])
    if patients_3plus:
        lines.append("## Patients with 3+ Target Gene Mutations (Detailed)")
        lines.append("")
        for p in patients_3plus:
            pid = p.get("patient_id", "?")
            lines.append(f"### Patient: {pid}")
            lines.append("")
            lines.append(f"- **Genes:** {', '.join(p.get('genes_mutated', []))}")
            lines.append(f"- **Studies:** {', '.join(p.get('studies', []))}")
            lines.append(f"- **GENIE overlap:** {'Yes' if p.get('in_genie_study') else 'No'}")
            lines.append("")
            lines.append("| Gene | Protein Change | Type | Hotspot |")
            lines.append("|------|---------------|------|---------|")
            seen = set()
            for m in p.get("mutations", []):
                key = (m.get("gene"), m.get("protein_change"))
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"| {m.get('gene', '?')} | {m.get('protein_change', '?')} | "
                    f"{m.get('mutation_type', '?')} | {'YES' if m.get('is_hotspot') else 'no'} |"
                )
            lines.append("")

    # Quadruple
    patients_quad = result.get("patients_quadruple", [])
    if patients_quad:
        lines.append("## QUADRUPLE MATCH FOUND")
        lines.append("")
        for p in patients_quad:
            lines.append(f"**Patient {p.get('patient_id', '?')}**")
            for m in p.get("mutations", []):
                lines.append(f"- {m.get('gene')}: {m.get('protein_change')} ({m.get('mutation_type')})")
            lines.append("")
    else:
        lines.append("## Quadruple Match")
        lines.append("")
        lines.append("**0 patients found with all 4 target genes mutated (DNMT3A + IDH2 + SETBP1 + PTPN11).**")
        lines.append("")
        lines.append("This confirms the finding from cross_database.py and the broader GENIE v19.0 analysis.")
        lines.append("")

    # Per-study table
    per_study = result.get("per_study_results", [])
    if per_study:
        lines.append("## Per-Study Results")
        lines.append("")
        lines.append("| Study ID | Name | Samples | DNMT3A | IDH2 | SETBP1 | PTPN11 | 2+ | 3+ | 4 | GENIE |")
        lines.append("|----------|------|---------|--------|------|--------|--------|----|----|---|-------|")
        for r in per_study:
            if r.get("error"):
                lines.append(
                    f"| {r['study_id']} | {r.get('name', '?')[:40]} | "
                    f"{r.get('sample_count', '?')} | - | - | - | - | - | - | - | "
                    f"{'Yes' if r.get('genie_overlap') else 'No'} |"
                )
                continue
            gc = r.get("per_gene_any_coding", {})
            lines.append(
                f"| {r['study_id']} | {r.get('name', '?')[:40]} | "
                f"{r.get('sample_count', 0)} | "
                f"{gc.get('DNMT3A', 0)} | {gc.get('IDH2', 0)} | "
                f"{gc.get('SETBP1', 0)} | {gc.get('PTPN11', 0)} | "
                f"{len(r.get('patients_2plus', []))} | "
                f"{len(r.get('patients_3plus', []))} | "
                f"{len(r.get('patients_4_quad', []))} | "
                f"{'Yes' if r.get('genie_overlap') else 'No'} |"
            )
        lines.append("")

    # Deduplication notes
    lines.append("## Deduplication Notes")
    lines.append("")
    lines.append("1. **GENIE overlap:** Many cBioPortal myeloid studies are subsets of AACR GENIE. "
                 "Studies from MSK, DFCI, VICC, UHN, OHSU, JHU, and other GENIE contributing "
                 "centers likely contain patients already counted in our GENIE v19.0 analysis.")
    lines.append("2. **Cross-study duplication:** The same patient may appear in multiple cBioPortal "
                 "studies (e.g., a TCGA-AML patient may also be in a pan-cancer study).")
    lines.append("3. **Patient ID format:** cBioPortal patient IDs are study-specific. Without a "
                 "universal patient identifier, exact deduplication across studies is not possible "
                 "from API data alone.")
    lines.append("4. **Truly independent patients** are those in studies from centers NOT contributing "
                 "to GENIE (e.g., local institutional studies, international cohorts).")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"- **Keywords used:** {', '.join(MYELOID_KEYWORDS)}")
    lines.append("- **Variant filter:** Coding mutations only (missense, nonsense, frameshift, "
                 "splice site, in-frame indels)")
    lines.append("- **Hotspot matching:** Exact protein change match for R882H, R140Q, G870S, E76Q")
    lines.append("- **Co-occurrence:** Patient must have coding mutations in 2+ of the 4 target "
                 "genes within the same study")
    lines.append("- **API:** cBioPortal public REST API, no authentication")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run the expanded cBioPortal search and save results."""
    result = run_expanded_search()

    # Save JSON
    json_path = RESULTS_DIR / "cbioportal_expanded.json"
    with open(json_path, "w") as f:
        # Convert sets and other non-serializable types
        json.dump(result, f, indent=2, default=str)
    log.info("\nJSON results saved to: %s", json_path)

    # Generate and save report
    report = generate_report(result)
    report_path = RESULTS_DIR / "cbioportal_expanded_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Report saved to: %s", report_path)

    # Summary
    cooc = result.get("cooccurrence_summary", {})
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("  Myeloid studies queried: %d", result.get("studies_summary", {}).get("myeloid_studies_found", 0))
    log.info("  Patients with 2+ target genes: %d", cooc.get("unique_patients_2plus_genes", 0))
    log.info("  Patients with 3+ target genes: %d", cooc.get("unique_patients_3plus_genes", 0))
    log.info("  Patients with quadruple: %d", cooc.get("unique_patients_quadruple", 0))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
