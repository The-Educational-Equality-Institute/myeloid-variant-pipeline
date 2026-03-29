#!/usr/bin/env python3
"""
verify_trial_status.py -- Verify clinical trial statuses across all project files.

Scans .md, .py, and .json files for NCT IDs, extracts any claimed statuses from
surrounding context, queries ClinicalTrials.gov API v2 for current statuses, and
flags stale/changed entries.

ClinicalTrials.gov API v2 details:
    - Base: https://clinicaltrials.gov/api/v2/studies
    - No API key required.
    - Rate limits: Undocumented but empirically ~3 req/s is safe. The API returns
      HTTP 429 if exceeded. We use batch queries (OR syntax) to minimize requests.
    - Batch: Up to ~30 NCT IDs per query via OR syntax in query.term.
    - Direct endpoint: /api/v2/studies/{nctId} for single study.
    - Key fields: OverallStatus, StatusVerifiedDate, LastUpdatePostDate.
    - Status enum: RECRUITING, ACTIVE_NOT_RECRUITING, NOT_YET_RECRUITING,
      ENROLLING_BY_INVITATION, COMPLETED, TERMINATED, WITHDRAWN, SUSPENDED,
      UNKNOWN.

Alternative data sources investigated:
    - AACT (aact.ctti-clinicaltrials.org): Free PostgreSQL dump or pipe-delimited
      flat files, updated daily. Download ~4 GB, provides all CT.gov fields locally.
      Useful for offline analysis but overkill for status verification. The API v2
      approach is faster and lighter for this use case.
    - Monitoring services: No free tools exist that alert on individual trial status
      changes. The best approach is this script on a weekly cron.

Date-stamping convention adopted in this script:
    NCT IDs in reports should use the format:
        NCT03728335 (Recruiting, verified 2026-03-28)
    The verify report shows when each status was last checked.

Storage recommendation:
    Trial status data should be stored SEPARATELY from canonical reference files
    (PATIENT_PROFILE.md, COMPLETE_FINDINGS_REPORT.md) because:
    1. Statuses change every few weeks; canonical files should be stable.
    2. Git history becomes noisy if stable research files change weekly.
    3. The clinical_trials.json already serves as the structured data store.
    4. This script produces a separate verification report that timestamps each check.
    The canonical files should reference trial IDs and point to the verification
    report for current status. Only update canonical files when a status change is
    clinically significant (e.g., a top trial closes enrollment).

Inputs:
    - All .md, .py, .json files in the project (scanned for NCT IDs)
    - ClinicalTrials.gov API v2 (remote, no key)

Outputs:
    - mutation_profile/results/ai_research/trial_status_verification.json
    - mutation_profile/results/ai_research/trial_status_verification.md
    - Exit code 0 if all statuses match, 1 if any are stale

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/verify_trial_status.py

    # Verify only the 33 trials from clinical_trial_search.py:
    python mutation_profile/scripts/ai_research/verify_trial_status.py --tracked-only

    # Check specific NCT IDs:
    python mutation_profile/scripts/ai_research/verify_trial_status.py --nct NCT03728335 NCT05441514

    # JSON-only output (for cron/CI):
    python mutation_profile/scripts/ai_research/verify_trial_status.py --json-only

    # Dry run (scan files but don't query API):
    python mutation_profile/scripts/ai_research/verify_trial_status.py --dry-run

Cron setup (weekly, Sunday 06:00):
    0 6 * * 0 cd ~/projects/helse/mrna-hematology-research && \
        ~/projects/helse/.venv/bin/python \
        mutation_profile/scripts/ai_research/verify_trial_status.py \
        --tracked-only >> logs/trial_status_cron.log 2>&1

Runtime: ~2-10 seconds depending on NCT ID count (batch queries).
Dependencies: requests (in shared venv)
"""

from __future__ import annotations

import argparse
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

JSON_OUTPUT = RESULTS_DIR / "trial_status_verification.json"
REPORT_OUTPUT = RESULTS_DIR / "trial_status_verification.md"
CLINICAL_TRIALS_JSON = RESULTS_DIR / "clinical_trials.json"

# ---------------------------------------------------------------------------
# ClinicalTrials.gov API v2 configuration
# ---------------------------------------------------------------------------
CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
CT_BATCH_SIZE = 25  # NCT IDs per batch query (safe limit for URL length)
CT_RATE_LIMIT_DELAY = 0.4  # seconds between batch requests
CT_REQUEST_TIMEOUT = 30  # seconds

# ---------------------------------------------------------------------------
# Status normalization
# ---------------------------------------------------------------------------
# ClinicalTrials.gov API v2 uses UPPER_SNAKE_CASE. Markdown files use various
# human-readable forms. This map normalizes claimed statuses to API format.
STATUS_NORMALIZE: dict[str, str] = {
    # API canonical forms
    "RECRUITING": "RECRUITING",
    "ACTIVE_NOT_RECRUITING": "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING": "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION": "ENROLLING_BY_INVITATION",
    "COMPLETED": "COMPLETED",
    "TERMINATED": "TERMINATED",
    "WITHDRAWN": "WITHDRAWN",
    "SUSPENDED": "SUSPENDED",
    "UNKNOWN": "UNKNOWN",
    # Human-readable forms found in markdown
    "RECRUITING": "RECRUITING",
    "ACTIVE NOT RECRUITING": "ACTIVE_NOT_RECRUITING",
    "ACTIVE, NOT RECRUITING": "ACTIVE_NOT_RECRUITING",
    "NOT YET RECRUITING": "NOT_YET_RECRUITING",
    "ENROLLING BY INVITATION": "ENROLLING_BY_INVITATION",
    # Title case forms
    "RECRUITING": "RECRUITING",
    "ACTIVE NOT RECRUITING": "ACTIVE_NOT_RECRUITING",
}

# Regex patterns that capture status text near an NCT ID
STATUS_PATTERNS = [
    # "NCT03728335 (Recruiting)" or "NCT03728335 (Active Not Recruiting)"
    r"(NCT\d{8})\s*\(([^)]*(?:Recruit|Active|Complet|Terminat|Withdraw|Suspend|Enroll)[^)]*)\)",
    # "| Recruiting |" in markdown tables near NCT ID
    r"(NCT\d{8}).*?\|\s*(Recruiting|Active[_ ]Not[_ ]Recruiting|Not[_ ]Yet[_ ]Recruiting|"
    r"Completed|Terminated|Withdrawn|Suspended|Enrolling[_ ]By[_ ]Invitation)\s*\|",
    # "Status: RECRUITING" near NCT ID
    r"(NCT\d{8}).*?(?:status|Status|STATUS)[:\s]+([A-Z_]+(?:RECRUITING|COMPLETED|TERMINATED|WITHDRAWN|SUSPENDED))",
    # "Ph2, Recruiting" comma-separated
    r"(NCT\d{8}).*?Ph\d[^,]*,\s*(Recruiting|Active Not Recruiting|Completed|Terminated)",
    # Reverse: status before NCT ID in tables
    r"\|\s*(Recruiting|Active[_ ]Not[_ ]Recruiting|Not[_ ]Yet[_ ]Recruiting|"
    r"Completed|Terminated|Withdrawn|Suspended)\s*\|.*?(NCT\d{8})",
]

# Just the NCT ID pattern for extraction
NCT_PATTERN = re.compile(r"NCT\d{8}")

# Files/directories to skip when scanning
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "chroma_db"}
SCAN_EXTENSIONS = {".md", ".py", ".json"}


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------


def scan_files_for_nct_ids(
    root: Path,
    extensions: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Scan project files for NCT IDs and extract claimed statuses.

    Returns a dict mapping NCT ID -> list of occurrences, each with:
        - file: relative file path
        - line_number: 1-based line number
        - context: surrounding text (trimmed)
        - claimed_status: normalized status string or None
    """
    if extensions is None:
        extensions = SCAN_EXTENSIONS

    nct_occurrences: dict[str, list[dict[str, Any]]] = {}

    for ext in extensions:
        for filepath in root.rglob(f"*{ext}"):
            # Skip excluded directories
            if any(skip in filepath.parts for skip in SKIP_DIRS):
                continue

            try:
                text = filepath.read_text(errors="replace")
            except (OSError, PermissionError):
                continue

            lines = text.split("\n")
            for line_num, line in enumerate(lines, 1):
                for match in NCT_PATTERN.finditer(line):
                    nct_id = match.group()
                    claimed = _extract_claimed_status(nct_id, line, lines, line_num)

                    occurrence = {
                        "file": str(filepath.relative_to(root)),
                        "line_number": line_num,
                        "context": line.strip()[:200],
                        "claimed_status": claimed,
                    }

                    if nct_id not in nct_occurrences:
                        nct_occurrences[nct_id] = []
                    nct_occurrences[nct_id].append(occurrence)

    return nct_occurrences


def _extract_claimed_status(
    nct_id: str,
    line: str,
    all_lines: list[str],
    line_num: int,
) -> str | None:
    """Try to extract a claimed trial status from the line context."""
    # Build a context window: current line + 1 line before/after
    context_lines = []
    if line_num > 1:
        context_lines.append(all_lines[line_num - 2])
    context_lines.append(line)
    if line_num < len(all_lines):
        context_lines.append(all_lines[line_num])
    context = " ".join(context_lines)

    for pattern in STATUS_PATTERNS:
        for m in re.finditer(pattern, context, re.IGNORECASE):
            groups = m.groups()
            # Determine which group is the NCT ID and which is the status
            found_nct = None
            found_status = None
            for g in groups:
                if g and re.match(r"NCT\d{8}$", g):
                    found_nct = g
                elif g:
                    found_status = g

            if found_nct == nct_id and found_status:
                normalized = _normalize_status(found_status)
                if normalized:
                    return normalized

    return None


def _normalize_status(raw: str) -> str | None:
    """Normalize a raw status string to API format."""
    cleaned = raw.strip().upper().replace("_", " ").replace(",", "")
    # Direct lookup
    if cleaned in STATUS_NORMALIZE:
        return STATUS_NORMALIZE[cleaned]
    # Partial matching
    if "RECRUITING" in cleaned and "NOT" not in cleaned and "ACTIVE" not in cleaned:
        return "RECRUITING"
    if "ACTIVE" in cleaned and "NOT" in cleaned and "RECRUITING" in cleaned:
        return "ACTIVE_NOT_RECRUITING"
    if "NOT YET" in cleaned:
        return "NOT_YET_RECRUITING"
    if "COMPLET" in cleaned:
        return "COMPLETED"
    if "TERMINAT" in cleaned:
        return "TERMINATED"
    if "WITHDRAW" in cleaned:
        return "WITHDRAWN"
    if "SUSPEND" in cleaned:
        return "SUSPENDED"
    if "ENROLLING" in cleaned:
        return "ENROLLING_BY_INVITATION"
    return None


# ---------------------------------------------------------------------------
# ClinicalTrials.gov API v2 queries
# ---------------------------------------------------------------------------


def fetch_trial_statuses(
    nct_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Batch-query ClinicalTrials.gov API v2 for current trial statuses.

    Uses OR syntax to query up to CT_BATCH_SIZE IDs per request.

    Returns dict mapping NCT ID -> {
        overall_status, status_verified_date, last_update_date,
        brief_title, phase
    }
    """
    results: dict[str, dict[str, Any]] = {}
    batches = [
        nct_ids[i : i + CT_BATCH_SIZE]
        for i in range(0, len(nct_ids), CT_BATCH_SIZE)
    ]

    for batch_num, batch in enumerate(batches, 1):
        query = " OR ".join(batch)
        params = {
            "query.term": query,
            "pageSize": CT_BATCH_SIZE + 5,  # slight buffer
            "fields": (
                "NCTId,OverallStatus,StatusVerifiedDate,"
                "LastUpdatePostDate,BriefTitle,Phase"
            ),
            "format": "json",
        }

        try:
            resp = requests.get(CT_API_BASE, params=params, timeout=CT_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  WARNING: Batch {batch_num} API request failed: {e}")
            for nct_id in batch:
                results[nct_id] = {
                    "overall_status": "API_ERROR",
                    "status_verified_date": None,
                    "last_update_date": None,
                    "brief_title": None,
                    "phase": None,
                    "error": str(e),
                }
            continue

        # Parse response
        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})

            nct_id = ident.get("nctId", "")
            if not nct_id:
                continue

            phases = design.get("phases", [])
            phase_str = ", ".join(phases) if phases else "Not specified"

            results[nct_id] = {
                "overall_status": status_mod.get("overallStatus", "UNKNOWN"),
                "status_verified_date": status_mod.get("statusVerifiedDate"),
                "last_update_date": (
                    status_mod.get("lastUpdatePostDateStruct", {}).get("date")
                ),
                "brief_title": ident.get("briefTitle", ""),
                "phase": phase_str,
                "error": None,
            }

        # Mark any batch IDs not returned by API
        returned_ids = {
            study.get("protocolSection", {})
            .get("identificationModule", {})
            .get("nctId", "")
            for study in data.get("studies", [])
        }
        for nct_id in batch:
            if nct_id not in returned_ids and nct_id not in results:
                results[nct_id] = {
                    "overall_status": "NOT_FOUND",
                    "status_verified_date": None,
                    "last_update_date": None,
                    "brief_title": None,
                    "phase": None,
                    "error": "NCT ID not returned by API (may be invalid or removed)",
                }

        if batch_num < len(batches):
            time.sleep(CT_RATE_LIMIT_DELAY)

    return results


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------


def compare_statuses(
    occurrences: dict[str, list[dict[str, Any]]],
    api_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare claimed statuses against API results.

    Returns a list of verification records, one per unique NCT ID.
    """
    records: list[dict[str, Any]] = []

    for nct_id in sorted(occurrences.keys()):
        api = api_results.get(nct_id, {})
        actual_status = api.get("overall_status", "NOT_QUERIED")

        # Collect all unique claimed statuses for this NCT ID
        claimed_statuses: set[str] = set()
        files_with_claims: list[dict] = []
        for occ in occurrences[nct_id]:
            if occ["claimed_status"]:
                claimed_statuses.add(occ["claimed_status"])
                files_with_claims.append(occ)

        # Determine if stale
        is_stale = False
        stale_reason = None
        if claimed_statuses and actual_status not in ("NOT_QUERIED", "API_ERROR", "NOT_FOUND"):
            if actual_status not in claimed_statuses:
                is_stale = True
                stale_reason = (
                    f"Claimed {', '.join(sorted(claimed_statuses))} "
                    f"but actual is {actual_status}"
                )

        # Count occurrences across files
        files_mentioned = sorted(
            set(occ["file"] for occ in occurrences[nct_id])
        )

        records.append({
            "nct_id": nct_id,
            "actual_status": actual_status,
            "claimed_statuses": sorted(claimed_statuses) if claimed_statuses else [],
            "is_stale": is_stale,
            "stale_reason": stale_reason,
            "status_verified_date": api.get("status_verified_date"),
            "last_update_date": api.get("last_update_date"),
            "brief_title": api.get("brief_title"),
            "phase": api.get("phase"),
            "api_error": api.get("error"),
            "occurrence_count": len(occurrences[nct_id]),
            "files_mentioned": files_mentioned,
            "files_with_status_claims": [
                {"file": o["file"], "line": o["line_number"], "claimed": o["claimed_status"]}
                for o in files_with_claims
            ],
        })

    return records


def generate_json_report(
    records: list[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON output."""
    return {
        "metadata": meta,
        "summary": {
            "total_nct_ids": len(records),
            "with_status_claims": sum(1 for r in records if r["claimed_statuses"]),
            "stale": sum(1 for r in records if r["is_stale"]),
            "api_errors": sum(1 for r in records if r["api_error"]),
            "not_found": sum(1 for r in records if r["actual_status"] == "NOT_FOUND"),
            "status_distribution": _count_statuses(records),
        },
        "stale_trials": [r for r in records if r["is_stale"]],
        "all_trials": records,
    }


def _count_statuses(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count actual status distribution."""
    counts: dict[str, int] = {}
    for r in records:
        s = r["actual_status"]
        counts[s] = counts.get(s, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def generate_markdown_report(
    records: list[dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    """Generate a markdown verification report."""
    lines: list[str] = []
    stale = [r for r in records if r["is_stale"]]
    with_claims = [r for r in records if r["claimed_statuses"]]
    errors = [r for r in records if r["api_error"]]

    lines.append("# Clinical Trial Status Verification Report")
    lines.append("")
    lines.append(f"**Verified:** {meta['timestamp']}")
    lines.append(f"**Source:** ClinicalTrials.gov API v2 (batch query)")
    lines.append(f"**Total NCT IDs scanned:** {len(records)}")
    lines.append(f"**With status claims in files:** {len(with_claims)}")
    lines.append(f"**Stale (status changed):** {len(stale)}")
    lines.append(f"**API errors:** {len(errors)}")
    lines.append("")

    if meta.get("mode"):
        lines.append(f"**Mode:** {meta['mode']}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Stale trials section (most important)
    if stale:
        lines.append(f"## STALE STATUSES: {len(stale)} trials need updating")
        lines.append("")
        lines.append("| NCT ID | Claimed | Actual | CT.gov Verified | Files |")
        lines.append("|--------|---------|--------|-----------------|-------|")
        for r in stale:
            claimed = ", ".join(r["claimed_statuses"])
            files = ", ".join(r["files_mentioned"][:3])
            if len(r["files_mentioned"]) > 3:
                files += f" +{len(r['files_mentioned']) - 3}"
            verified = r["status_verified_date"] or "N/A"
            lines.append(
                f"| [{r['nct_id']}](https://clinicaltrials.gov/study/{r['nct_id']}) "
                f"| {claimed} | **{r['actual_status']}** | {verified} | {files} |"
            )
        lines.append("")
        lines.append("### Stale Trial Details")
        lines.append("")
        for r in stale:
            lines.append(f"#### {r['nct_id']}: {r['brief_title'] or 'Unknown'}")
            lines.append("")
            lines.append(f"- **Stale reason:** {r['stale_reason']}")
            lines.append(f"- **Phase:** {r['phase'] or 'N/A'}")
            lines.append(f"- **CT.gov verified:** {r['status_verified_date'] or 'N/A'}")
            lines.append(f"- **Last update:** {r['last_update_date'] or 'N/A'}")
            lines.append("- **Files with stale claim:**")
            for claim in r["files_with_status_claims"]:
                lines.append(f"  - `{claim['file']}` line {claim['line']} (claimed: {claim['claimed']})")
            lines.append("")
    else:
        lines.append("## All statuses are current")
        lines.append("")
        lines.append("No stale statuses detected.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # All tracked trials with status claims
    lines.append(f"## All Trials With Status Claims ({len(with_claims)})")
    lines.append("")
    if with_claims:
        lines.append("| NCT ID | Title | Claimed | Actual | Match | CT.gov Verified |")
        lines.append("|--------|-------|---------|--------|-------|-----------------|")
        for r in with_claims:
            claimed = ", ".join(r["claimed_statuses"])
            title = (r["brief_title"] or "Unknown")[:50]
            if len(r.get("brief_title", "") or "") > 50:
                title += "..."
            match_icon = "OK" if not r["is_stale"] else "STALE"
            verified = r["status_verified_date"] or "N/A"
            lines.append(
                f"| [{r['nct_id']}](https://clinicaltrials.gov/study/{r['nct_id']}) "
                f"| {title} | {claimed} | {r['actual_status']} | {match_icon} | {verified} |"
            )
        lines.append("")
    else:
        lines.append("No status claims found in project files.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Status distribution
    status_dist = _count_statuses(records)
    lines.append("## Status Distribution (All Scanned NCT IDs)")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|------:|")
    for status, count in status_dist.items():
        lines.append(f"| {status} | {count} |")
    lines.append("")

    lines.append("---")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("1. Scanned all `.md`, `.py`, `.json` files for `NCT\\d{8}` pattern")
    lines.append("2. Extracted status claims from surrounding context using regex")
    lines.append("3. Queried ClinicalTrials.gov API v2 in batches of 25 NCT IDs")
    lines.append("4. Compared claimed vs actual statuses")
    lines.append("5. Flagged any mismatches as stale")
    lines.append("")
    lines.append("### Date-stamping convention")
    lines.append("")
    lines.append("Trial references in reports should use this format:")
    lines.append("```")
    lines.append("NCT03728335 (Recruiting, verified 2026-03-28)")
    lines.append("```")
    lines.append("")
    lines.append("### Alternative data sources")
    lines.append("")
    lines.append("- **AACT database** (aact.ctti-clinicaltrials.org): Daily PostgreSQL")
    lines.append("  dumps and pipe-delimited flat files. Free, no registration. Contains")
    lines.append("  all CT.gov data. Useful for bulk offline analysis but overkill for")
    lines.append("  status verification. Download is ~4 GB compressed.")
    lines.append("- **ClinicalTrials.gov API v2**: No API key needed. Batch queries with")
    lines.append("  OR syntax handle up to ~30 IDs per request. Rate limit is empirically")
    lines.append("  ~3 requests/second. Ideal for weekly status checks.")
    lines.append("- **Trial monitoring services**: No free tools exist for automated")
    lines.append("  individual trial status change alerts. This script on weekly cron")
    lines.append("  is the recommended approach.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tracked trials (from clinical_trial_search.py output)
# ---------------------------------------------------------------------------


def get_tracked_nct_ids() -> set[str]:
    """Load NCT IDs from the clinical_trials.json produced by clinical_trial_search.py."""
    if not CLINICAL_TRIALS_JSON.exists():
        print(f"  WARNING: {CLINICAL_TRIALS_JSON} not found, falling back to full scan")
        return set()

    data = json.loads(CLINICAL_TRIALS_JSON.read_text())
    nct_ids: set[str] = set()
    for trial in data.get("all_trials_ranked", []):
        nct_id = trial.get("nct_id", "")
        if nct_id:
            nct_ids.add(nct_id)
    return nct_ids


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify clinical trial statuses across project files."
    )
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="Only verify trials from clinical_trials.json (the 33 tracked trials).",
    )
    parser.add_argument(
        "--nct",
        nargs="+",
        help="Verify specific NCT IDs (space-separated).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON only, no markdown report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files but do not query ClinicalTrials.gov API.",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("CLINICAL TRIAL STATUS VERIFICATION")
    print(f"Timestamp: {timestamp}")
    print("=" * 70)

    # Step 1: Scan files
    print("\n[1/3] Scanning project files for NCT IDs...")
    occurrences = scan_files_for_nct_ids(PROJECT_ROOT)
    all_nct_ids = set(occurrences.keys())
    print(f"  Found {len(all_nct_ids)} unique NCT IDs across project files")

    claims_count = sum(
        1 for nct_id in occurrences
        if any(occ["claimed_status"] for occ in occurrences[nct_id])
    )
    print(f"  {claims_count} NCT IDs have status claims in file context")

    # Step 2: Determine which IDs to verify
    mode = "full_scan"
    if args.nct:
        target_ids = sorted(set(args.nct))
        mode = f"specific ({len(target_ids)} IDs)"
        # Filter occurrences to only requested IDs
        occurrences = {k: v for k, v in occurrences.items() if k in target_ids}
        # Add any requested IDs not found in files
        for nct_id in target_ids:
            if nct_id not in occurrences:
                occurrences[nct_id] = []
    elif args.tracked_only:
        tracked = get_tracked_nct_ids()
        if tracked:
            mode = f"tracked_only ({len(tracked)} trials)"
            # Keep only tracked IDs but preserve all their occurrences
            occurrences = {k: v for k, v in occurrences.items() if k in tracked}
            print(f"  Filtered to {len(occurrences)} tracked trials")
        else:
            mode = "full_scan (tracked filter failed, using all)"

    nct_ids_to_query = sorted(occurrences.keys())
    print(f"\n[2/3] Querying ClinicalTrials.gov API v2 for {len(nct_ids_to_query)} trials...")

    # Step 3: Query API
    if args.dry_run:
        print("  DRY RUN -- skipping API queries")
        api_results = {
            nct_id: {
                "overall_status": "NOT_QUERIED",
                "status_verified_date": None,
                "last_update_date": None,
                "brief_title": None,
                "phase": None,
                "error": "dry-run",
            }
            for nct_id in nct_ids_to_query
        }
    else:
        batch_count = (len(nct_ids_to_query) + CT_BATCH_SIZE - 1) // CT_BATCH_SIZE
        print(f"  {batch_count} batch request(s) needed (batch size: {CT_BATCH_SIZE})")
        api_results = fetch_trial_statuses(nct_ids_to_query)
        print(f"  Received status for {sum(1 for v in api_results.values() if not v.get('error'))} trials")

    # Step 4: Compare and report
    print(f"\n[3/3] Comparing claimed vs actual statuses...")
    records = compare_statuses(occurrences, api_results)

    stale_count = sum(1 for r in records if r["is_stale"])
    print(f"  Stale statuses: {stale_count}")

    meta = {
        "timestamp": timestamp,
        "mode": mode,
        "total_scanned": len(records),
        "api": "ClinicalTrials.gov API v2",
        "batch_size": CT_BATCH_SIZE,
        "project_root": str(PROJECT_ROOT),
    }

    # Save JSON
    json_report = generate_json_report(records, meta)
    JSON_OUTPUT.write_text(json.dumps(json_report, indent=2, ensure_ascii=False))
    print(f"\n  JSON: {JSON_OUTPUT}")

    # Save markdown
    if not args.json_only:
        md_report = generate_markdown_report(records, meta)
        REPORT_OUTPUT.write_text(md_report)
        print(f"  Report: {REPORT_OUTPUT}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total NCT IDs verified: {len(records)}")
    print(f"  With status claims: {sum(1 for r in records if r['claimed_statuses'])}")
    print(f"  Stale: {stale_count}")
    print(f"  API errors: {sum(1 for r in records if r['api_error'])}")

    if stale_count > 0:
        print(f"\n  STALE TRIALS:")
        for r in records:
            if r["is_stale"]:
                print(f"    {r['nct_id']}: {r['stale_reason']}")

    # Status distribution
    dist = _count_statuses(records)
    print(f"\n  Status distribution:")
    for status, count in dist.items():
        print(f"    {status}: {count}")

    # Exit code: 1 if any stale
    if stale_count > 0:
        print(f"\nExit code 1: {stale_count} stale status(es) detected.")
        sys.exit(1)
    else:
        print("\nExit code 0: All statuses current.")


if __name__ == "__main__":
    main()
