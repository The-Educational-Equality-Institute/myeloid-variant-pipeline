#!/usr/bin/env python3
"""
check_database_freshness.py -- Check pinned database versions against current releases.

Reads mutation_profile/database_versions.yaml and queries each database's
version API to detect when a pinned version has become outdated. Prints a
summary table with FRESH/STALE/UNKNOWN status for each database.

For databases with programmatic version APIs (gnomAD, GDC, OncoKB, cBioPortal,
UniProt), the script queries the live endpoint and compares against the pinned
version. For databases without version APIs (ClinVar, CIViC, HARMONY), it
checks staleness based on days since last access against configured thresholds.

This implements the genomics convention where:
  - Static downloads (GENIE, COSMIC) are pinned to release versions
  - Continuously-updated APIs (ClinVar, cBioPortal) are pinned to access dates
  - Both types get sample counts as integrity checksums

Inputs:
    - mutation_profile/database_versions.yaml

Outputs:
    Prints freshness report to stdout. Optionally writes JSON report.

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/check_database_freshness.py
    python mutation_profile/scripts/check_database_freshness.py --json
    python mutation_profile/scripts/check_database_freshness.py --strict  # exit 1 if any stale

Runtime: ~30-60 seconds (network queries with timeouts)
Dependencies: requests, pyyaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # mutation_profile/
MANIFEST_PATH = PROJECT_DIR / "database_versions.yaml"
RESULTS_DIR = PROJECT_DIR / "results"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 15  # seconds
RATE_LIMIT_DELAY = 0.5  # seconds between API calls


def load_manifest() -> dict[str, Any]:
    """Load and parse the database_versions.yaml manifest."""
    if not MANIFEST_PATH.exists():
        log.error("Manifest not found: %s", MANIFEST_PATH)
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = yaml.safe_load(f)

    if not manifest or "databases" not in manifest:
        log.error("Invalid manifest: missing 'databases' key")
        sys.exit(1)

    return manifest


# ---------------------------------------------------------------------------
# Version check functions (one per database with a programmatic API)
# ---------------------------------------------------------------------------


def check_gdc_version(db_config: dict) -> dict:
    """Check GDC data release version via /status endpoint."""
    try:
        resp = requests.get(
            "https://api.gdc.cancer.gov/status",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        current = data.get("data_release", "unknown")
        pinned = db_config.get("data_release", "unknown")

        # Extract version number from string like "Data Release 42.0 - March 20, 2026"
        current_short = current
        if "Data Release" in current:
            parts = current.split(" - ")[0]
            current_short = parts.replace("Data Release ", "").strip()

        return {
            "current_version": current_short,
            "pinned_version": str(pinned),
            "match": str(pinned) in current,
            "raw_response": current,
        }
    except Exception as e:
        return {"error": str(e), "current_version": None}


def check_gnomad_version(db_config: dict) -> dict:
    """Check gnomAD version via GraphQL API."""
    query = '{"query": "{ meta { gnomadVersion } }"}'
    try:
        resp = requests.post(
            "https://gnomad.broadinstitute.org/api",
            headers={"Content-Type": "application/json"},
            data=query,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        current = data.get("data", {}).get("meta", {}).get("gnomadVersion", "unknown")
        pinned = db_config.get("version", "unknown")

        return {
            "current_version": current,
            "pinned_version": pinned,
            "match": current == pinned,
        }
    except Exception as e:
        return {"error": str(e), "current_version": None}


def check_cbioportal_version(db_config: dict) -> dict:
    """Check cBioPortal API info for database version and study count."""
    results = {}
    try:
        # Get API info (includes dbVersion)
        resp = requests.get(
            "https://www.cbioportal.org/api/info",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results["db_version"] = data.get("dbVersion", "unknown")
        results["portal_version"] = data.get("portalVersion", "unknown")
    except Exception as e:
        results["info_error"] = str(e)

    try:
        # Count studies as a growth indicator
        resp = requests.get(
            "https://www.cbioportal.org/api/studies",
            headers={"Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        studies = resp.json()
        results["total_studies"] = len(studies)
        pinned_count = db_config.get("key_counts", {}).get(
            "myeloid_studies_queried", 0
        )
        results["pinned_myeloid_studies"] = pinned_count
        results["note"] = (
            f"Total studies: {len(studies)} (pinned myeloid subset: {pinned_count})"
        )
    except Exception as e:
        results["studies_error"] = str(e)

    return results


def check_oncokb_version(db_config: dict) -> dict:
    """Check OncoKB data version via /info endpoint (requires auth token)."""
    import os

    token = os.environ.get("ONCOKB_API_TOKEN", "")
    if not token:
        # Try loading from secrets file
        secrets_path = Path.home() / ".secrets.env"
        if secrets_path.exists():
            for line in secrets_path.read_text().splitlines():
                if line.startswith("ONCOKB_API_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip("'\"")
                    break

    if not token:
        return {
            "error": "ONCOKB_API_TOKEN not set; cannot check OncoKB version",
            "current_version": None,
        }

    try:
        resp = requests.get(
            "https://www.oncokb.org/api/v1/info",
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        current = data.get("dataVersion", {})
        if isinstance(current, dict):
            current = current.get("version", "unknown")
        pinned = db_config.get("version", "unknown")

        return {
            "current_version": str(current),
            "pinned_version": str(pinned),
            "match": str(current) == str(pinned),
        }
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            return {"error": "OncoKB API token invalid or expired", "current_version": None}
        return {"error": str(e), "current_version": None}
    except Exception as e:
        return {"error": str(e), "current_version": None}


def check_uniprot_version(db_config: dict) -> dict:
    """Check UniProt release version via response header."""
    try:
        # Fetch a known entry and check the release header
        resp = requests.head(
            "https://rest.uniprot.org/uniprotkb/Q9Y6K1.json",
            timeout=REQUEST_TIMEOUT,
        )
        current = resp.headers.get("X-UniProt-Release", "unknown")
        pinned = db_config.get("version", "unknown")

        return {
            "current_version": current,
            "pinned_version": pinned,
            "match": current == pinned,
        }
    except Exception as e:
        return {"error": str(e), "current_version": None}


def check_genie_version(db_config: dict) -> dict:
    """Check GENIE version via Synapse (requires synapseclient + auth)."""
    try:
        import synapseclient

        syn = synapseclient.Synapse()
        syn.login(silent=True)

        # List children of the GENIE project to find release folders
        children = list(syn.getChildren("syn7222066", includeTypes=["folder"]))
        release_folders = [
            c["name"]
            for c in children
            if c["name"].startswith("Release ") or "public" in c["name"].lower()
        ]

        # Sort to find latest
        release_folders.sort()
        latest = release_folders[-1] if release_folders else "unknown"
        pinned = db_config.get("version", "unknown")

        return {
            "current_version": latest,
            "pinned_version": pinned,
            "all_releases": release_folders[-5:],  # Last 5
            "match": pinned in latest,
        }
    except ImportError:
        return {"error": "synapseclient not installed", "current_version": None}
    except Exception as e:
        return {"error": str(e), "current_version": None}


def check_clinvar_freshness(db_config: dict) -> dict:
    """Check ClinVar latest release date via FTP Last-Modified header."""
    try:
        resp = requests.head(
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarFullRelease_00-latest.xml.gz",
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        last_modified = resp.headers.get("Last-Modified", "unknown")
        return {
            "last_release": last_modified,
            "pinned_access_date": db_config.get("accessed"),
            "note": "ClinVar has no version; last FTP release date shown",
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Staleness calculation
# ---------------------------------------------------------------------------


def days_since_access(accessed_str: str | None) -> int | None:
    """Calculate days since the access date string (YYYY-MM-DD)."""
    if not accessed_str:
        return None
    try:
        accessed_date = datetime.strptime(accessed_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        delta = datetime.now(timezone.utc) - accessed_date
        return delta.days
    except (ValueError, TypeError):
        return None


def evaluate_staleness(
    db_key: str,
    db_config: dict,
    thresholds: dict,
    version_check_result: dict | None,
) -> dict:
    """Evaluate whether a database entry is stale.

    Returns a dict with:
        status: FRESH | STALE | UNKNOWN | UPDATED
        reason: human-readable explanation
        days_since_access: int or None
        version_outdated: bool or None
    """
    accessed = db_config.get("accessed")
    days = days_since_access(accessed)
    threshold = thresholds.get(db_key, 365)

    result = {
        "database": db_key,
        "name": db_config.get("name", db_key),
        "pinned_version": db_config.get("version") or db_config.get("data_release"),
        "accessed": accessed,
        "days_since_access": days,
        "threshold_days": threshold,
        "status": "UNKNOWN",
        "reason": "",
        "version_check": version_check_result,
    }

    # Check if version API detected an update
    if version_check_result:
        if version_check_result.get("error"):
            result["reason"] = f"Version check failed: {version_check_result['error']}"
        elif version_check_result.get("match") is False:
            current = version_check_result.get("current_version", "?")
            pinned = version_check_result.get("pinned_version", "?")
            result["status"] = "UPDATED"
            result["reason"] = f"New version available: {current} (pinned: {pinned})"
            result["version_outdated"] = True
            return result
        elif version_check_result.get("match") is True:
            result["version_outdated"] = False

    # Check staleness by access date
    if accessed is None:
        result["status"] = "UNKNOWN"
        result["reason"] = "No access date recorded (never accessed or pending)"
        return result

    if days is None:
        result["status"] = "UNKNOWN"
        result["reason"] = f"Could not parse access date: {accessed}"
        return result

    if days > threshold:
        result["status"] = "STALE"
        result["reason"] = f"Last accessed {days} days ago (threshold: {threshold})"
    else:
        remaining = threshold - days
        if version_check_result and version_check_result.get("match") is True:
            result["status"] = "FRESH"
            result["reason"] = f"Version confirmed current; {remaining} days until threshold"
        else:
            result["status"] = "FRESH"
            result["reason"] = f"Accessed {days} days ago; {remaining} days until threshold"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_checks(manifest: dict) -> list[dict]:
    """Run all freshness checks and return results."""
    databases = manifest["databases"]
    thresholds = manifest.get("staleness_thresholds", {})
    results = []

    # Map of databases that have programmatic version checks
    version_checkers = {
        "gdc": check_gdc_version,
        "gnomad": check_gnomad_version,
        "cbioportal": check_cbioportal_version,
        "oncokb": check_oncokb_version,
        "uniprot": check_uniprot_version,
        "clinvar": check_clinvar_freshness,
        # genie requires synapseclient auth; attempted but may fail
        "genie": check_genie_version,
    }

    for db_key, db_config in databases.items():
        log.info("Checking %s ...", db_config.get("name", db_key))

        # Run version check if available
        version_result = None
        if db_key in version_checkers:
            try:
                version_result = version_checkers[db_key](db_config)
            except Exception as e:
                version_result = {"error": str(e)}
            time.sleep(RATE_LIMIT_DELAY)

        # Evaluate staleness
        eval_result = evaluate_staleness(db_key, db_config, thresholds, version_result)
        results.append(eval_result)

    return results


def format_citation_block(manifest: dict) -> str:
    """Generate a formatted citation block for manuscripts/reports.

    This produces the standard genomics citation format:
        "Database vX.Y (accessed YYYY-MM-DD, N samples)"
    """
    lines = [
        "DATABASE CITATIONS (copy into Methods section)",
        "=" * 55,
        "",
    ]
    for db_key, db_config in manifest["databases"].items():
        citation = db_config.get("citation")
        if citation:
            lines.append(f"  {db_config.get('name', db_key)}:")
            lines.append(f"    {citation}")
            lines.append("")
    return "\n".join(lines)


def print_report(results: list[dict], manifest: dict) -> int:
    """Print a human-readable freshness report. Returns count of stale/updated databases."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print()
    print("=" * 78)
    print(f"  DATABASE FRESHNESS REPORT  |  {now}")
    print("=" * 78)
    print()

    # Summary table
    header = f"{'Database':<28} {'Version':<16} {'Accessed':<12} {'Days':<6} {'Status':<10}"
    print(header)
    print("-" * len(header))

    stale_count = 0
    updated_count = 0

    for r in results:
        name = r["name"][:27]
        version = str(r.get("pinned_version") or "-")[:15]
        accessed = str(r.get("accessed") or "never")[:11]
        days = str(r.get("days_since_access") or "-")
        status = r["status"]

        # Status indicator
        if status == "FRESH":
            indicator = "FRESH"
        elif status == "STALE":
            indicator = "** STALE **"
            stale_count += 1
        elif status == "UPDATED":
            indicator = "!! UPDATED !!"
            updated_count += 1
        else:
            indicator = "? UNKNOWN"

        print(f"{name:<28} {version:<16} {accessed:<12} {days:<6} {indicator}")

    print()
    print("-" * 78)

    # Detail section for non-FRESH databases
    flagged = [r for r in results if r["status"] not in ("FRESH",)]
    if flagged:
        print()
        print("DETAILS:")
        print()
        for r in flagged:
            print(f"  [{r['status']}] {r['name']}")
            print(f"    {r['reason']}")
            vc = r.get("version_check")
            if vc and not vc.get("error"):
                current = vc.get("current_version")
                if current:
                    print(f"    Current upstream version: {current}")
            print()

    # Citation block
    print()
    print(format_citation_block(manifest))

    # Summary
    total = len(results)
    fresh = sum(1 for r in results if r["status"] == "FRESH")
    stale = stale_count
    updated = updated_count
    unknown = total - fresh - stale - updated

    print()
    print(f"SUMMARY: {total} databases | {fresh} fresh | {stale} stale | {updated} updated | {unknown} unknown")
    print()

    if updated > 0:
        print(
            "ACTION REQUIRED: Databases with UPDATED status have new versions available."
        )
        print(
            "Re-run affected analyses, update database_versions.yaml, and verify results."
        )
        print()

    if stale > 0:
        print(
            "WARNING: Databases with STALE status exceed their configured staleness threshold."
        )
        print(
            "Consider re-querying these databases before manuscript submission."
        )
        print()

    return stale_count + updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Check database version freshness against pinned manifest."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write JSON report to mutation_profile/results/database_freshness.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any database is stale or updated",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network checks; only evaluate staleness by access date",
    )
    parser.add_argument(
        "--citations-only",
        action="store_true",
        help="Print only the citation block for manuscript use",
    )
    args = parser.parse_args()

    manifest = load_manifest()

    if args.citations_only:
        print(format_citation_block(manifest))
        return

    if args.skip_network:
        # Evaluate staleness without any network calls
        databases = manifest["databases"]
        thresholds = manifest.get("staleness_thresholds", {})
        results = []
        for db_key, db_config in databases.items():
            eval_result = evaluate_staleness(db_key, db_config, thresholds, None)
            results.append(eval_result)
    else:
        results = run_checks(manifest)

    flagged_count = print_report(results, manifest)

    if args.json:
        output_path = RESULTS_DIR / "database_freshness.json"
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(MANIFEST_PATH),
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info("JSON report written to %s", output_path)

    if args.strict and flagged_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
