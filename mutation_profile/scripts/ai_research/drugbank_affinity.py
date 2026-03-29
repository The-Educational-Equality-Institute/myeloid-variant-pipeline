#!/usr/bin/env python3
"""
drugbank_affinity.py -- Quantitative binding affinity data from ChEMBL + Open Targets.

Queries the ChEMBL REST API for Ki, IC50, Kd values for drugs targeting the patient's
mutant proteins, and Open Targets for drug-disease-gene associations.

Patient mutation profile (MDS-AML):
    1. PTPN11 E76Q  -- SHP2 phosphatase (RAS/MAPK pathway)
    2. IDH2 R140Q   -- isocitrate dehydrogenase (metabolic)
    3. EZH2 V662A   -- histone methyltransferase (PRC2)
    4. DNMT3A R882H  -- DNA methyltransferase (epigenetic)

Drug-target pairs queried:
    - PTPN11: TNO155, RMC-4550, SHP099
    - IDH2:   Enasidenib, Ivosidenib, Vorasidenib
    - EZH2:   Tazemetostat, EPZ-6438, GSK126
    - DNMT3A: Azacitidine, Decitabine

Inputs:
    - ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data/, no auth required)
    - Open Targets Platform API (https://api.platform.opentargets.org/, no auth required)

Outputs:
    - mutation_profile/results/ai_research/drugbank_affinity.json
    - mutation_profile/results/ai_research/drugbank_affinity_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/drugbank_affinity.py

Runtime: ~2-5 minutes (many sequential API calls with rate limiting)
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # mutation_profile/
RESULTS_DIR = PROJECT_ROOT / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = RESULTS_DIR / "drugbank_affinity.json"
MD_OUTPUT = RESULTS_DIR / "drugbank_affinity_report.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChEMBL API
# ---------------------------------------------------------------------------
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_HEADERS = {"Accept": "application/json"}

# Open Targets GraphQL API
OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# ---------------------------------------------------------------------------
# Drug-target pairs to query
# ---------------------------------------------------------------------------
DRUG_TARGET_PAIRS: dict[str, dict[str, Any]] = {
    "PTPN11": {
        "uniprot": "Q06124",
        "ensembl_gene": "ENSG00000179295",
        "mutation": "E76Q",
        "drugs": {
            "TNO155": {"synonyms": ["TNO155", "CHEMBL4594384"]},
            "RMC-4550": {"synonyms": ["RMC-4550", "RMC4550"]},
            "SHP099": {"synonyms": ["SHP099", "SHP-099", "CHEMBL4078582"]},
        },
    },
    "IDH2": {
        "uniprot": "P48735",
        "ensembl_gene": "ENSG00000182054",
        "mutation": "R140Q",
        "drugs": {
            "Enasidenib": {"synonyms": ["Enasidenib", "AG-221", "IDHIFA", "CHEMBL3989896"]},
            "Ivosidenib": {"synonyms": ["Ivosidenib", "AG-120", "TIBSOVO", "CHEMBL4297085"]},
            "Vorasidenib": {"synonyms": ["Vorasidenib", "AG-881", "CHEMBL4594314"]},
        },
    },
    "EZH2": {
        "uniprot": "Q15910",
        "ensembl_gene": "ENSG00000106462",
        "mutation": "V662A",
        "drugs": {
            "Tazemetostat": {"synonyms": ["Tazemetostat", "EPZ-6438", "EPZ6438", "TAZVERIK", "CHEMBL3545267"]},
            "GSK126": {"synonyms": ["GSK126", "GSK-126", "GSK2816126", "CHEMBL3360203"]},
            "EPZ005687": {"synonyms": ["EPZ005687", "EPZ-005687", "CHEMBL2216882"]},
        },
    },
    "DNMT3A": {
        "uniprot": "Q9Y6K1",
        "ensembl_gene": "ENSG00000119772",
        "mutation": "R882H",
        "drugs": {
            "Azacitidine": {"synonyms": ["Azacitidine", "5-Azacytidine", "Vidaza", "CHEMBL1489"]},
            "Decitabine": {"synonyms": ["Decitabine", "5-Aza-2-deoxycytidine", "Dacogen", "CHEMBL1201129"]},
        },
    },
}

# Activity types we want (binding / functional assays)
ACTIVITY_TYPES = {"IC50", "Ki", "Kd", "EC50", "Potency"}


def _build_session() -> requests.Session:
    """Build a requests session with retry logic for flaky APIs."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _build_session()


def rate_limit() -> None:
    """ChEMBL asks for reasonable rate limiting."""
    time.sleep(0.5)


# ---------------------------------------------------------------------------
# 1. ChEMBL: resolve molecule CHEMBL ID from name/synonym
# ---------------------------------------------------------------------------
def chembl_search_molecule(name: str) -> dict[str, Any] | None:
    """Search ChEMBL for a molecule by name. Returns first match or None."""
    url = f"{CHEMBL_BASE}/molecule/search.json"
    params = {"q": name, "limit": 5}
    rate_limit()
    try:
        resp = SESSION.get(url, params=params, headers=CHEMBL_HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        molecules = data.get("molecules", [])
        if molecules:
            return molecules[0]
    except Exception as e:
        log.warning("ChEMBL molecule search failed for %s: %s", name, e)
    return None


def chembl_get_molecule(chembl_id: str) -> dict[str, Any] | None:
    """Get molecule details by ChEMBL ID."""
    url = f"{CHEMBL_BASE}/molecule/{chembl_id}.json"
    rate_limit()
    try:
        resp = SESSION.get(url, headers=CHEMBL_HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("ChEMBL molecule fetch failed for %s: %s", chembl_id, e)
    return None


# Known ChEMBL molecule IDs (fallback if API is down)
KNOWN_CHEMBL_MOLECULES = {
    "TNO155": "CHEMBL4594384",
    "RMC-4550": "CHEMBL4545691",
    "SHP099": "CHEMBL4078582",
    "Enasidenib": "CHEMBL3989896",
    "Ivosidenib": "CHEMBL4297085",
    "Vorasidenib": "CHEMBL4594314",
    "Tazemetostat": "CHEMBL3545267",
    "GSK126": "CHEMBL3360203",
    "EPZ005687": "CHEMBL2216882",
    "Azacitidine": "CHEMBL1489",
    "Decitabine": "CHEMBL1201129",
}


def resolve_molecule_chembl_id(drug_name: str, synonyms: list[str]) -> str | None:
    """Try to resolve a ChEMBL ID for a drug via search or direct ID lookup.

    Falls back to hardcoded IDs if the API is unavailable.
    """
    # First check if any synonym is already a CHEMBL ID
    for syn in synonyms:
        if syn.startswith("CHEMBL"):
            mol = chembl_get_molecule(syn)
            if mol:
                return mol.get("molecule_chembl_id", syn)

    # Search by each synonym
    for syn in synonyms:
        mol = chembl_search_molecule(syn)
        if mol and mol.get("molecule_chembl_id"):
            return mol["molecule_chembl_id"]

    # Fallback to known IDs
    fallback = KNOWN_CHEMBL_MOLECULES.get(drug_name)
    if fallback:
        log.info("    Using hardcoded ChEMBL molecule ID: %s", fallback)
    return fallback


# ---------------------------------------------------------------------------
# 2. ChEMBL: resolve target CHEMBL ID from UniProt
# ---------------------------------------------------------------------------
# Known ChEMBL target IDs (fallback if API is down)
KNOWN_CHEMBL_TARGETS = {
    "Q06124": "CHEMBL3717",   # PTPN11/SHP2
    "P48735": "CHEMBL2096906",  # IDH2
    "Q15910": "CHEMBL2189110",  # EZH2
    "Q9Y6K1": "CHEMBL1795126",  # DNMT3A
}


def chembl_resolve_target(uniprot_id: str) -> str | None:
    """Resolve a ChEMBL target ID from a UniProt accession.

    Falls back to hardcoded IDs if the API is unavailable.
    """
    url = f"{CHEMBL_BASE}/target.json"
    params = {
        "target_components__accession": uniprot_id,
        "limit": 5,
    }
    rate_limit()
    try:
        resp = SESSION.get(url, params=params, headers=CHEMBL_HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        targets = data.get("targets", [])
        # Prefer SINGLE PROTEIN targets
        for t in targets:
            if t.get("target_type") == "SINGLE PROTEIN":
                return t["target_chembl_id"]
        if targets:
            return targets[0]["target_chembl_id"]
    except Exception as e:
        log.warning("ChEMBL target resolution failed for %s: %s", uniprot_id, e)

    # Fallback to known IDs
    fallback = KNOWN_CHEMBL_TARGETS.get(uniprot_id)
    if fallback:
        log.info("  Using hardcoded ChEMBL target ID: %s", fallback)
    return fallback


# ---------------------------------------------------------------------------
# 3. ChEMBL: get bioactivity data for a molecule against a target
# ---------------------------------------------------------------------------
def chembl_get_activities(
    molecule_chembl_id: str,
    target_chembl_id: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch bioactivity data (IC50, Ki, Kd, etc.) for a molecule, optionally filtered by target."""
    url = f"{CHEMBL_BASE}/activity.json"
    params: dict[str, Any] = {
        "molecule_chembl_id": molecule_chembl_id,
        "limit": 100,
    }
    if target_chembl_id:
        params["target_chembl_id"] = target_chembl_id

    all_activities: list[dict[str, Any]] = []
    rate_limit()
    try:
        resp = SESSION.get(url, params=params, headers=CHEMBL_HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        activities = data.get("activities", [])
        all_activities.extend(activities)

        # Page through if needed (up to 500 results)
        page_meta = data.get("page_meta", {})
        next_url = page_meta.get("next")
        pages = 0
        while next_url and pages < 4:
            rate_limit()
            resp = SESSION.get(
                f"https://www.ebi.ac.uk{next_url}",
                headers=CHEMBL_HEADERS,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            all_activities.extend(data.get("activities", []))
            next_url = data.get("page_meta", {}).get("next")
            pages += 1

    except Exception as e:
        log.warning("ChEMBL activity fetch failed for %s: %s", molecule_chembl_id, e)

    return all_activities


def filter_binding_activities(activities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter activities to only binding/potency assays with numeric values."""
    filtered = []
    for act in activities:
        std_type = act.get("standard_type", "")
        if std_type not in ACTIVITY_TYPES:
            continue
        val = act.get("standard_value")
        if val is None:
            continue
        try:
            float(val)
        except (ValueError, TypeError):
            continue
        filtered.append({
            "assay_chembl_id": act.get("assay_chembl_id"),
            "assay_description": act.get("assay_description", ""),
            "assay_type": act.get("assay_type", ""),
            "type": std_type,
            "value": float(val),
            "units": act.get("standard_units", ""),
            "relation": act.get("standard_relation", "="),
            "target_chembl_id": act.get("target_chembl_id", ""),
            "target_pref_name": act.get("target_pref_name", ""),
            "pchembl_value": act.get("pchembl_value"),
            "document_chembl_id": act.get("document_chembl_id", ""),
            "doi": act.get("document_journal", ""),
            "year": act.get("document_year"),
        })
    return filtered


# ---------------------------------------------------------------------------
# 4. ChEMBL: get all activities for a target (for selectivity)
# ---------------------------------------------------------------------------
def chembl_get_target_activities(
    molecule_chembl_id: str,
) -> list[dict[str, Any]]:
    """Fetch ALL activities for a molecule (not filtered by target) to assess selectivity."""
    return chembl_get_activities(molecule_chembl_id, target_chembl_id=None)


def compute_selectivity(
    on_target_activities: list[dict[str, Any]],
    all_activities: list[dict[str, Any]],
    target_chembl_id: str | None,
) -> dict[str, Any]:
    """Compute selectivity metrics from on-target vs all activities."""
    if not target_chembl_id:
        return {"selectivity_ratio": None, "off_targets_tested": 0, "off_target_hits": []}

    # Gather unique targets from all activities
    target_values: dict[str, list[float]] = {}
    for act in all_activities:
        t_id = act.get("target_chembl_id", "")
        t_name = act.get("target_pref_name", "")
        val = act.get("value")
        if val and act.get("type") in {"IC50", "Ki", "Kd"}:
            key = f"{t_id}|{t_name}"
            if key not in target_values:
                target_values[key] = []
            target_values[key].append(val)

    on_target_key = None
    for key in target_values:
        if key.startswith(target_chembl_id):
            on_target_key = key
            break

    on_target_median = None
    if on_target_key and target_values[on_target_key]:
        vals = sorted(target_values[on_target_key])
        on_target_median = vals[len(vals) // 2]

    off_target_hits = []
    for key, vals in target_values.items():
        if key == on_target_key:
            continue
        t_id, t_name = key.split("|", 1)
        median_val = sorted(vals)[len(vals) // 2]
        off_target_hits.append({
            "target_chembl_id": t_id,
            "target_name": t_name,
            "median_value_nM": median_val,
            "n_assays": len(vals),
        })

    selectivity_ratio = None
    if on_target_median and off_target_hits:
        best_off = min(h["median_value_nM"] for h in off_target_hits) if off_target_hits else None
        if best_off and best_off > 0:
            selectivity_ratio = best_off / on_target_median

    return {
        "on_target_median_nM": on_target_median,
        "selectivity_ratio": selectivity_ratio,
        "off_targets_tested": len(off_target_hits),
        "off_target_hits": sorted(off_target_hits, key=lambda x: x["median_value_nM"])[:10],
    }


# ---------------------------------------------------------------------------
# 5. ChEMBL: clinical trial data
# ---------------------------------------------------------------------------
def chembl_get_clinical_data(molecule_chembl_id: str) -> dict[str, Any]:
    """Get max clinical trial phase and indication data for a molecule."""
    mol = chembl_get_molecule(molecule_chembl_id)
    if not mol:
        return {"max_phase": None, "first_approval": None, "indications": []}

    max_phase = mol.get("max_phase")
    first_approval = mol.get("first_approval")

    # Get drug indications
    url = f"{CHEMBL_BASE}/drug_indication.json"
    params = {"molecule_chembl_id": molecule_chembl_id, "limit": 20}
    indications = []
    rate_limit()
    try:
        resp = SESSION.get(url, params=params, headers=CHEMBL_HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for ind in data.get("drug_indications", []):
            indications.append({
                "mesh_heading": ind.get("mesh_heading", ""),
                "efo_term": ind.get("efo_term", ""),
                "max_phase_for_ind": ind.get("max_phase_for_ind"),
            })
    except Exception as e:
        log.warning("ChEMBL indication fetch failed for %s: %s", molecule_chembl_id, e)

    return {
        "max_phase": max_phase,
        "first_approval": first_approval,
        "indications": indications,
    }


# ---------------------------------------------------------------------------
# 6. Open Targets: drug-disease-gene associations
# ---------------------------------------------------------------------------
def query_open_targets(ensembl_gene_id: str, gene_symbol: str) -> list[dict[str, Any]]:
    """Query Open Targets Platform API for drug-disease-gene associations.

    Uses the v4 GraphQL schema with drugAndClinicalCandidates field.
    """
    query = """
    query DrugAssociations($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        approvedSymbol
        drugAndClinicalCandidates {
          count
          rows {
            id
            maxClinicalStage
            drug {
              id
              name
              maximumClinicalStage
              mechanismsOfAction {
                rows {
                  mechanismOfAction
                  actionType
                }
              }
            }
            diseases {
              diseaseFromSource
              disease {
                id
                name
              }
            }
          }
        }
      }
    }
    """
    variables = {"ensemblId": ensembl_gene_id}
    rate_limit()

    try:
        resp = SESSION.post(
            OPENTARGETS_URL,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("errors"):
            log.warning("Open Targets errors for %s: %s", gene_symbol, data["errors"][0].get("message", ""))
            return []

        target_data = data.get("data", {}).get("target")
        if not target_data:
            log.warning("Open Targets: no data for %s (%s)", gene_symbol, ensembl_gene_id)
            return []

        candidates = target_data.get("drugAndClinicalCandidates") or {}
        rows = candidates.get("rows", [])
        results = []
        for row in rows:
            drug = row.get("drug") or {}
            moa_rows = (drug.get("mechanismsOfAction") or {}).get("rows", [])
            moa_text = "; ".join(
                r.get("mechanismOfAction", "") for r in moa_rows if r.get("mechanismOfAction")
            )
            diseases = row.get("diseases", [])
            disease_names = []
            for d in diseases:
                disease_obj = d.get("disease") or {}
                name = disease_obj.get("name") or d.get("diseaseFromSource") or ""
                if name:
                    disease_names.append(name)

            results.append({
                "drug_id": drug.get("id", ""),
                "drug_name": drug.get("name", ""),
                "max_clinical_stage": row.get("maxClinicalStage", ""),
                "mechanism_of_action": moa_text,
                "diseases": disease_names,
                "trial_phase": row.get("maxClinicalStage", ""),
                "trial_status": "",
            })

        if results:
            log.info("  Open Targets: found %d drug associations for %s", len(results), gene_symbol)
        return results

    except Exception as e:
        log.warning("Open Targets query failed for %s: %s", gene_symbol, e)
        return []


# ---------------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline() -> dict[str, Any]:
    """Run the full binding affinity pipeline."""
    results: dict[str, Any] = {
        "metadata": {
            "query_date": datetime.now(timezone.utc).isoformat(),
            "sources": ["ChEMBL REST API", "Open Targets Platform API"],
            "patient_mutations": {
                gene: info["mutation"] for gene, info in DRUG_TARGET_PAIRS.items()
            },
        },
        "targets": {},
    }

    for gene, gene_info in DRUG_TARGET_PAIRS.items():
        log.info("=" * 60)
        log.info("Processing %s (%s)", gene, gene_info["mutation"])
        log.info("=" * 60)

        # Resolve ChEMBL target ID
        target_chembl_id = chembl_resolve_target(gene_info["uniprot"])
        log.info("  Target ChEMBL ID: %s", target_chembl_id)

        # Query Open Targets
        log.info("  Querying Open Targets for %s...", gene)
        ot_results = query_open_targets(gene_info["ensembl_gene"], gene)

        target_result: dict[str, Any] = {
            "gene": gene,
            "mutation": gene_info["mutation"],
            "uniprot": gene_info["uniprot"],
            "target_chembl_id": target_chembl_id,
            "drugs": {},
            "open_targets": ot_results,
        }

        for drug_name, drug_info in gene_info["drugs"].items():
            log.info("  Processing drug: %s", drug_name)

            # Resolve molecule ChEMBL ID
            mol_id = resolve_molecule_chembl_id(drug_name, drug_info["synonyms"])
            log.info("    Molecule ChEMBL ID: %s", mol_id)

            drug_result: dict[str, Any] = {
                "drug_name": drug_name,
                "molecule_chembl_id": mol_id,
                "on_target_activities": [],
                "all_binding_activities": [],
                "selectivity": {},
                "clinical": {},
            }

            if mol_id:
                # Get on-target activities
                if target_chembl_id:
                    log.info("    Fetching on-target activities...")
                    raw_activities = chembl_get_activities(mol_id, target_chembl_id)
                    on_target = filter_binding_activities(raw_activities)
                    drug_result["on_target_activities"] = on_target
                    log.info("    Found %d on-target binding data points", len(on_target))

                # Get all activities for selectivity
                log.info("    Fetching all activities for selectivity...")
                all_raw = chembl_get_target_activities(mol_id)
                all_binding = filter_binding_activities(all_raw)
                drug_result["all_binding_activities"] = all_binding
                log.info("    Found %d total binding data points", len(all_binding))

                # Compute selectivity
                drug_result["selectivity"] = compute_selectivity(
                    drug_result["on_target_activities"],
                    all_binding,
                    target_chembl_id,
                )

                # Clinical trial data
                log.info("    Fetching clinical data...")
                drug_result["clinical"] = chembl_get_clinical_data(mol_id)

            target_result["drugs"][drug_name] = drug_result

        results["targets"][gene] = target_result

    return results


# ---------------------------------------------------------------------------
# 8. Report generation
# ---------------------------------------------------------------------------
def summarize_activities(activities: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Group activities by type and return values."""
    by_type: dict[str, list[float]] = {}
    for act in activities:
        t = act["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(act["value"])
    return by_type


def format_value_range(values: list[float]) -> str:
    """Format a list of values as median (range)."""
    if not values:
        return "N/A"
    values = sorted(values)
    median = values[len(values) // 2]
    if len(values) == 1:
        return f"{median:.1f} nM (n=1)"
    return f"{median:.1f} nM (range: {values[0]:.1f}-{values[-1]:.1f}, n={len(values)})"


def generate_report(results: dict[str, Any]) -> str:
    """Generate markdown report from results."""
    lines = [
        "# Drug-Target Binding Affinity Report",
        "",
        f"**Query date:** {results['metadata']['query_date']}",
        f"**Sources:** {', '.join(results['metadata']['sources'])}",
        "**Patient:** Henrik - MDS-AML post allo-HSCT (complete remission, MRD negative)",
        "",
        "---",
        "",
        "## Summary",
        "",
        "Quantitative binding data (Ki, IC50, Kd) for drugs targeting the patient's mutant proteins,",
        "collected from ChEMBL published assays. Open Targets provides drug-disease-gene associations.",
        "",
    ]

    # Summary table
    lines.extend([
        "### Pharmacology Overview",
        "",
        "| Gene | Mutation | Drug | IC50 (nM) | Ki (nM) | Kd (nM) | Max Phase | Selectivity |",
        "|------|----------|------|-----------|---------|---------|-----------|-------------|",
    ])

    for gene, target_data in results["targets"].items():
        mutation = target_data["mutation"]
        for drug_name, drug_data in target_data["drugs"].items():
            on_target = drug_data.get("on_target_activities", [])
            by_type = summarize_activities(on_target)

            ic50_str = format_value_range(by_type.get("IC50", []))
            ki_str = format_value_range(by_type.get("Ki", []))
            kd_str = format_value_range(by_type.get("Kd", []))

            clinical = drug_data.get("clinical", {})
            max_phase = clinical.get("max_phase", "?")
            if max_phase is None:
                max_phase = "?"

            sel = drug_data.get("selectivity", {})
            sel_ratio = sel.get("selectivity_ratio")
            sel_str = f"{sel_ratio:.1f}x" if sel_ratio else "N/A"

            lines.append(
                f"| {gene} | {mutation} | {drug_name} | {ic50_str} | {ki_str} | {kd_str} | {max_phase} | {sel_str} |"
            )

    lines.extend(["", "---", ""])

    # Detailed sections per gene
    for gene, target_data in results["targets"].items():
        mutation = target_data["mutation"]
        lines.extend([
            f"## {gene} ({mutation})",
            "",
            f"**UniProt:** {target_data['uniprot']}",
            f"**ChEMBL target:** {target_data.get('target_chembl_id', 'N/A')}",
            "",
        ])

        for drug_name, drug_data in target_data["drugs"].items():
            mol_id = drug_data.get("molecule_chembl_id", "N/A")
            lines.extend([
                f"### {drug_name}",
                "",
                f"**ChEMBL molecule:** {mol_id}",
                "",
            ])

            # On-target binding data
            on_target = drug_data.get("on_target_activities", [])
            if on_target:
                by_type = summarize_activities(on_target)
                lines.append("**On-target binding data:**")
                lines.append("")
                lines.append("| Assay Type | Value | Units | Relation | Assay Description |")
                lines.append("|------------|-------|-------|----------|-------------------|")
                for act in sorted(on_target, key=lambda x: x["value"]):
                    desc = act.get("assay_description", "")[:80]
                    lines.append(
                        f"| {act['type']} | {act['value']:.2f} | {act['units']} | {act['relation']} | {desc} |"
                    )
                lines.append("")

                # Summary statistics
                for atype, vals in by_type.items():
                    vals_sorted = sorted(vals)
                    median = vals_sorted[len(vals_sorted) // 2]
                    lines.append(f"- **{atype}:** median {median:.1f} nM (n={len(vals)}, range {vals_sorted[0]:.1f}-{vals_sorted[-1]:.1f})")
                lines.append("")
            else:
                lines.append("*No on-target binding data found in ChEMBL.*")
                lines.append("")

            # Check all binding activities for general data
            all_binding = drug_data.get("all_binding_activities", [])
            if all_binding and not on_target:
                lines.append("**General binding data (all targets):**")
                lines.append("")
                by_type_all = summarize_activities(all_binding)
                for atype, vals in by_type_all.items():
                    vals_sorted = sorted(vals)
                    median = vals_sorted[len(vals_sorted) // 2]
                    lines.append(f"- **{atype}:** median {median:.1f} nM (n={len(vals)}, range {vals_sorted[0]:.1f}-{vals_sorted[-1]:.1f})")
                lines.append("")

            # Selectivity
            sel = drug_data.get("selectivity", {})
            off_targets = sel.get("off_target_hits", [])
            if off_targets:
                lines.append("**Selectivity (top off-targets):**")
                lines.append("")
                lines.append("| Off-Target | Median IC50/Ki (nM) | N Assays |")
                lines.append("|------------|---------------------|----------|")
                for ot in off_targets[:5]:
                    lines.append(
                        f"| {ot['target_name']} | {ot['median_value_nM']:.1f} | {ot['n_assays']} |"
                    )
                lines.append("")
                sel_ratio = sel.get("selectivity_ratio")
                if sel_ratio:
                    lines.append(f"**Selectivity ratio (best off-target / on-target):** {sel_ratio:.1f}x")
                    lines.append("")

            # Clinical data
            clinical = drug_data.get("clinical", {})
            max_phase = clinical.get("max_phase")
            if max_phase is not None:
                lines.append(f"**Max clinical phase:** {max_phase}")
                first_approval = clinical.get("first_approval")
                if first_approval:
                    lines.append(f"**First approval:** {first_approval}")
                indications = clinical.get("indications", [])
                if indications:
                    lines.append("")
                    lines.append("**Indications:**")
                    for ind in indications:
                        heading = ind.get("mesh_heading") or ind.get("efo_term") or "Unknown"
                        phase = ind.get("max_phase_for_ind", "?")
                        lines.append(f"- {heading} (phase {phase})")
                lines.append("")

        # Open Targets associations
        ot_results = target_data.get("open_targets", [])
        if ot_results:
            lines.extend([
                f"### Open Targets - {gene} Drug-Disease Associations",
                "",
                "| Drug | Diseases | Clinical Stage | Mechanism |",
                "|------|---------|----------------|-----------|",
            ])
            seen = set()
            for row in ot_results:
                drug_name_ot = row.get("drug_name", "")
                if drug_name_ot in seen:
                    continue
                seen.add(drug_name_ot)
                diseases = row.get("diseases", [])
                disease_str = ", ".join(diseases[:3])
                if len(diseases) > 3:
                    disease_str += f" (+{len(diseases)-3})"
                moa = row.get("mechanism_of_action", "")[:60]
                stage = row.get("max_clinical_stage", "?")
                lines.append(
                    f"| {drug_name_ot} | {disease_str} | {stage} | {moa} |"
                )
            lines.append("")

        lines.extend(["---", ""])

    # Clinical relevance section
    lines.extend([
        "## Clinical Relevance to Patient Profile",
        "",
        "### Key Findings",
        "",
        "1. **IDH2 R140Q (2% VAF subclone):** Enasidenib is FDA-approved for relapsed/refractory AML with IDH2 mutation.",
        "   The subclonal nature (2% VAF) means the IDH2-mutant clone may not drive relapse, but enasidenib",
        "   remains the most clinically validated targeted option.",
        "",
        "2. **PTPN11 E76Q (29% VAF):** SHP2 inhibitors (TNO155, RMC-4550) are in Phase I/II clinical trials.",
        "   PTPN11 confers venetoclax resistance via RAS pathway activation. SHP2 inhibition could overcome this.",
        "",
        "3. **EZH2 V662A (59% VAF, biallelic loss):** EZH2 loss-of-function in myeloid malignancies is distinct",
        "   from gain-of-function in lymphoma. Standard EZH2 inhibitors (tazemetostat) would further suppress",
        "   residual EZH2 activity and are likely CONTRAINDICATED. DNMT inhibitors may be more appropriate",
        "   for EZH2-loss contexts.",
        "",
        "4. **DNMT3A R882H (39% VAF):** Hypomethylating agents (azacitidine, decitabine) target the pathway",
        "   but do not specifically inhibit the mutant protein. Their mechanism is DNA incorporation, not",
        "   direct enzyme inhibition.",
        "",
        "### Important Caveat: EZH2 Loss-of-Function",
        "",
        "The patient's EZH2 V662A is a loss-of-function mutation (disrupted SET domain + monosomy 7 = biallelic loss).",
        "EZH2 inhibitors were developed for gain-of-function mutations (e.g., Y641N in lymphoma). Using EZH2 inhibitors",
        "on a loss-of-function background could worsen disease. The binding data above reflects activity against wild-type",
        "EZH2, not the therapeutic context for this patient.",
        "",
        "---",
        "",
        "*Data from ChEMBL (https://www.ebi.ac.uk/chembl/) and Open Targets (https://platform.opentargets.org/).*",
        "*This report is for research purposes only and does not constitute medical advice.*",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def main() -> int:
    log.info("Starting drug-target binding affinity pipeline")
    log.info("Querying ChEMBL and Open Targets APIs...")

    results = run_pipeline()

    # Count results
    total_on_target = 0
    total_all = 0
    total_ot = 0
    for gene, target_data in results["targets"].items():
        for drug_name, drug_data in target_data["drugs"].items():
            total_on_target += len(drug_data.get("on_target_activities", []))
            total_all += len(drug_data.get("all_binding_activities", []))
        total_ot += len(target_data.get("open_targets", []))

    log.info("Results: %d on-target data points, %d total binding, %d Open Targets associations",
             total_on_target, total_all, total_ot)

    # Save JSON
    with open(JSON_OUTPUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("JSON saved: %s", JSON_OUTPUT)

    # Generate and save report
    report = generate_report(results)
    with open(MD_OUTPUT, "w") as f:
        f.write(report)
    log.info("Report saved: %s", MD_OUTPUT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
