#!/usr/bin/env python3
"""
ezh2_literature_search.py -- EZH2 V662A and EZH2 loss-of-function literature search.

Uses the Semantic Scholar Academic Graph API to search for publications about
EZH2 in MDS/AML, with specific focus on:
- EZH2 V662A (patient variant, Pathogenic at 59% VAF)
- EZH2 loss-of-function mutations in myeloid malignancies
- PRC2 complex disruption in MDS/AML
- EZH2 and monosomy 7 co-occurrence

Also searches for landmark papers (Ernst 2010, Nikoloski 2010, Bejar 2011).

Inputs:
    - Semantic Scholar API (remote, no auth required)

Outputs:
    - mutation_profile/results/ai_research/ezh2_literature.json
    - mutation_profile/results/ai_research/ezh2_literature_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/ezh2_literature_search.py

Runtime: ~2-5 minutes (rate-limited API calls)
Dependencies: requests, python-dotenv
"""

import json
import os
import time
import datetime
from collections import Counter, defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research"
JSON_OUTPUT = OUTPUT_DIR / "ezh2_literature.json"
REPORT_OUTPUT = OUTPUT_DIR / "ezh2_literature_report.md"

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
load_dotenv(ENV_PATH)
S2_API_KEY = os.getenv("S2_API_KEY", "")

BASE_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,authors,year,citationCount,abstract,externalIds,venue,referenceCount,fieldsOfStudy"
RATE_LIMIT_SLEEP = 3.5  # seconds between bulk requests (conservative to avoid 429s)
RATE_LIMIT_SLEEP_SEARCH = 5.0  # seconds between regular search requests (stricter limits)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "mrna-hematology-research/1.0 (academic; mailto:research@theeducationalequalityinstitute.org)",
})
if S2_API_KEY:
    SESSION.headers["x-api-key"] = S2_API_KEY
    print("Using Semantic Scholar API key (higher rate limits)")
else:
    print("No S2_API_KEY found -- using public rate limits")

# ---------------------------------------------------------------------------
# Search queries
# ---------------------------------------------------------------------------
SEARCH_QUERIES = {
    "EZH2 V662A": "EZH2 V662A",
    "EZH2 V662": "EZH2 V662 mutation",
    "EZH2 loss-of-function myeloid": "EZH2 loss of function myeloid",
    "EZH2 mutation MDS AML": "EZH2 mutation MDS AML",
    "PRC2 myeloid malignancy": "PRC2 myeloid malignancy",
    "EZH2 monosomy 7": "EZH2 monosomy 7",
    "EZH2 SET domain missense": "EZH2 SET domain missense",
}

# Landmark papers -- look up by DOI first (most reliable), fallback to search
LANDMARK_PAPERS = {
    "Ernst 2010 Nature Genetics": {
        "doi": "10.1038/ng.596",
        "fallback_query": "Ernst EZH2 inactivating mutations myeloid Nature Genetics 2010",
        "expected_title": "Inactivating mutations of the histone methyltransferase gene EZH2 in myeloid disorders",
    },
    "Nikoloski 2010 Nature Genetics": {
        "doi": "10.1038/ng.620",
        "fallback_query": "Nikoloski EZH2 somatic mutations myelodysplastic 2010",
        "expected_title": "Somatic mutations of the histone methyltransferase gene EZH2 in myelodysplastic syndromes",
    },
    "Bejar 2011 JCO": {
        "doi": "10.1200/JCO.2011.35.5693",
        "fallback_query": "Bejar clinical effect somatic mutations prognosis myelodysplastic 2011",
        "expected_title": "Clinical effect of point mutations in myelodysplastic syndromes",
    },
}

MAX_RESULTS_PER_QUERY = 100


def api_get(url: str, params: dict | None = None) -> dict | None:
    """Make a GET request with rate limiting and retries."""
    for attempt in range(5):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited (429). Waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"  API error {resp.status_code}: {resp.text[:200]}", flush=True)
            return None
        except requests.RequestException as e:
            print(f"  Request error: {e}", flush=True)
            if attempt < 4:
                time.sleep(10)
    return None


def search_papers(query: str, limit: int = MAX_RESULTS_PER_QUERY) -> list[dict]:
    """Search Semantic Scholar for papers matching a query."""
    all_papers = []
    continuation_token = None

    while len(all_papers) < limit:
        params = {
            "query": query,
            "fields": PAPER_FIELDS,
        }
        if continuation_token:
            params["token"] = continuation_token

        data = api_get(f"{BASE_URL}/paper/search/bulk", params)
        time.sleep(RATE_LIMIT_SLEEP)

        if not data or "data" not in data:
            break

        papers = data["data"]
        if not papers:
            break

        all_papers.extend(papers)

        continuation_token = data.get("token")
        if not continuation_token:
            break

    return all_papers[:limit]


def lookup_paper_by_doi(doi: str) -> dict | None:
    """Look up a specific paper by DOI."""
    params = {"fields": PAPER_FIELDS}
    data = api_get(f"{BASE_URL}/paper/DOI:{doi}", params)
    time.sleep(RATE_LIMIT_SLEEP)
    return data if data and data.get("paperId") else None


def search_paper_by_title(query: str, limit: int = 10) -> list[dict]:
    """Search for specific papers using the regular search endpoint (better for known papers)."""
    params = {
        "query": query,
        "fields": PAPER_FIELDS,
        "limit": limit,
    }
    data = api_get(f"{BASE_URL}/paper/search", params)
    time.sleep(RATE_LIMIT_SLEEP_SEARCH)
    if data and "data" in data:
        return data["data"]
    return []


def extract_author_info(papers: list[dict]) -> dict:
    """Extract author publication counts."""
    author_counts = Counter()
    author_ids = {}
    author_papers = defaultdict(list)

    for paper in papers:
        if not paper.get("authors"):
            continue
        for author in paper["authors"]:
            name = author.get("name", "Unknown")
            aid = author.get("authorId")
            if name and name != "Unknown":
                author_counts[name] += 1
                if aid:
                    author_ids[name] = aid
                author_papers[name].append({
                    "title": paper.get("title", ""),
                    "year": paper.get("year"),
                    "citations": paper.get("citationCount", 0),
                })

    return {"counts": author_counts, "ids": author_ids, "papers": dict(author_papers)}


def compute_publication_trends(papers: list[dict]) -> dict[int, int]:
    """Count papers per year."""
    year_counts = Counter()
    for p in papers:
        year = p.get("year")
        if year and isinstance(year, int) and 1990 <= year <= 2026:
            year_counts[year] += 1
    return dict(sorted(year_counts.items()))


def format_paper_entry(paper: dict, rank: int | None = None) -> str:
    """Format a single paper entry for markdown."""
    title = paper.get("title", "Untitled")
    year = paper.get("year", "N/A")
    citations = paper.get("citationCount", 0)
    doi = ""
    ext_ids = paper.get("externalIds") or {}
    if ext_ids.get("DOI"):
        doi = f" | DOI: [{ext_ids['DOI']}](https://doi.org/{ext_ids['DOI']})"
    venue = paper.get("venue", "")
    venue_str = f" | *{venue}*" if venue else ""

    authors = paper.get("authors", [])
    if authors:
        author_names = [a.get("name", "") for a in authors[:5]]
        if len(authors) > 5:
            author_names.append(f"+ {len(authors) - 5} more")
        author_str = ", ".join(author_names)
    else:
        author_str = "Unknown authors"

    prefix = f"{rank}. " if rank else "- "
    lines = [f"{prefix}**{title}** ({year}, {citations} citations{doi}{venue_str})"]
    lines.append(f"   {author_str}")

    abstract = paper.get("abstract", "")
    if abstract:
        truncated = abstract[:400] + "..." if len(abstract) > 400 else abstract
        lines.append(f"   > {truncated}")

    return "\n".join(lines)


def deduplicate_papers(all_papers: list[dict]) -> list[dict]:
    """Deduplicate papers by paperId."""
    seen = set()
    unique = []
    for p in all_papers:
        pid = p.get("paperId")
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(p)
    return unique


def generate_report(
    query_results: dict[str, list[dict]],
    landmark_results: dict[str, list[dict]],
    unique_papers: list[dict],
    run_timestamp: str,
) -> str:
    """Generate the full markdown report."""
    lines = [
        "# EZH2 Literature Search -- Semantic Scholar",
        "",
        f"**Generated:** {run_timestamp}",
        "**API:** Semantic Scholar Academic Graph API v1",
        "**Focus:** EZH2 V662A, EZH2 loss-of-function in MDS/AML, PRC2 in myeloid malignancies",
        "",
        "**Patient context:** EZH2 V662A Pathogenic at 59% VAF (highest clonal fraction among all 5 variants),",
        "co-occurring with DNMT3A R882H (39%), SETBP1 G870S (34%), PTPN11 E76Q (29%),",
        "IDH2 R140Q (2%), and monosomy 7.",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        "| Query | Papers Found | Top Paper (citations) |",
        "|---|---|---|",
    ]

    for query_name, papers in query_results.items():
        if papers:
            top = max(papers, key=lambda p: p.get("citationCount", 0))
            top_title = top.get("title", "N/A")[:55]
            top_str = f"{top_title}... ({top.get('citationCount', 0)})"
        else:
            top_str = "No results"
        lines.append(f"| {query_name} | {len(papers)} | {top_str} |")

    lines.extend([
        "",
        f"**Total unique papers across all queries:** {len(unique_papers)}",
        "",
        "---",
        "",
    ])

    # Landmark papers section
    lines.append("## Landmark Papers")
    lines.append("")
    lines.append("Key foundational papers on EZH2 in myeloid malignancies:")
    lines.append("")

    for landmark_name, papers in landmark_results.items():
        lines.append(f"### {landmark_name}")
        if papers:
            # Show the best match (first result, usually most relevant)
            lines.append(format_paper_entry(papers[0], rank=None))
            if len(papers) > 1:
                lines.append("")
                lines.append(f"   *({len(papers) - 1} additional results for this search)*")
        else:
            lines.append("- Not found in Semantic Scholar search results.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Per-query results
    lines.append("## Per-Query Results")
    lines.append("")

    for query_name, papers in query_results.items():
        query_str = SEARCH_QUERIES.get(query_name, query_name)
        lines.append(f"### {query_name}")
        lines.append(f"**Query:** `{query_str}` | **Results:** {len(papers)} papers")
        lines.append("")

        if not papers:
            lines.append("No papers found.")
            lines.append("")
            continue

        # Publication trend
        trends = compute_publication_trends(papers)
        if trends:
            lines.append("**Publication trend:**")
            lines.append("")
            lines.append("| Year | Papers |")
            lines.append("|---|---|")
            for year, count in trends.items():
                bar = "#" * min(count, 40)
                lines.append(f"| {year} | {count} {bar} |")
            lines.append("")

        # Top 5 most-cited
        sorted_papers = sorted(papers, key=lambda p: p.get("citationCount", 0), reverse=True)
        lines.append("**Top 5 most-cited:**")
        lines.append("")
        for i, paper in enumerate(sorted_papers[:5], 1):
            lines.append(format_paper_entry(paper, rank=i))
            lines.append("")

        # Key authors
        author_info = extract_author_info(papers)
        top_authors = author_info["counts"].most_common(10)
        if top_authors:
            lines.append("**Key authors:**")
            lines.append("")
            lines.append("| Author | Papers |")
            lines.append("|---|---|")
            for name, count in top_authors:
                lines.append(f"| {name} | {count} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Global top 15
    lines.append("## Top 15 Most-Cited Papers (All Queries)")
    lines.append("")
    global_sorted = sorted(unique_papers, key=lambda p: p.get("citationCount", 0), reverse=True)
    for i, paper in enumerate(global_sorted[:15], 1):
        lines.append(format_paper_entry(paper, rank=i))
        lines.append("")

    lines.append("---")
    lines.append("")

    # Global author analysis
    lines.append("## Key Research Groups")
    lines.append("")
    global_author_info = extract_author_info(unique_papers)
    top_global_authors = global_author_info["counts"].most_common(20)
    if top_global_authors:
        lines.append("| Rank | Author | Papers |")
        lines.append("|---|---|---|")
        for rank, (name, count) in enumerate(top_global_authors, 1):
            lines.append(f"| {rank} | {name} | {count} |")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Overall publication trend
    lines.append("## Overall Publication Trend")
    lines.append("")
    global_trends = compute_publication_trends(unique_papers)
    if global_trends:
        lines.append("| Year | Papers |")
        lines.append("|---|---|")
        for year, count in global_trends.items():
            bar = "#" * min(count, 50)
            lines.append(f"| {year} | {count} {bar} |")
        lines.append("")

    lines.append("---")
    lines.append("")

    # EZH2 V662A specific analysis
    lines.append("## EZH2 V662A Variant-Specific Findings")
    lines.append("")
    v662a_papers = query_results.get("EZH2 V662A", [])
    v662_papers = query_results.get("EZH2 V662", [])
    set_domain = query_results.get("EZH2 SET domain missense", [])

    if v662a_papers:
        lines.append(f"- **Direct V662A hits:** {len(v662a_papers)} papers mention EZH2 V662A specifically")
    else:
        lines.append("- **Direct V662A hits:** 0 -- this variant has no dedicated literature")

    if v662_papers:
        lines.append(f"- **V662 position hits:** {len(v662_papers)} papers reference the V662 position")
    else:
        lines.append("- **V662 position hits:** 0 -- position 662 not specifically studied")

    if set_domain:
        lines.append(f"- **SET domain missense papers:** {len(set_domain)} papers discuss SET domain missense variants")

    lines.extend([
        "",
        "**Clinical significance:** EZH2 V662 lies in the SET domain (methyltransferase catalytic domain).",
        "V662A at 59% VAF is the dominant clone. Loss-of-function EZH2 mutations are well-established",
        "in MDS/AML as tumor suppressors (Ernst et al. 2010, Nikoloski et al. 2010).",
        "The high VAF suggests this is an early/founding event.",
        "",
        "---",
        "",
    ])

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Data source:** Semantic Scholar Academic Graph API v1")
    lines.append(f"- **Rate limiting:** {RATE_LIMIT_SLEEP}s between API calls")
    lines.append(f"- **Max results per query:** {MAX_RESULTS_PER_QUERY}")
    lines.append(f"- **Search queries:** {len(SEARCH_QUERIES)} topic queries + {len(LANDMARK_PAPERS)} landmark paper lookups (DOI-based)")
    lines.append("- **Deduplication:** Papers appearing in multiple queries counted once in global stats")
    lines.append("")

    return "\n".join(lines)


def paper_to_json(paper: dict) -> dict:
    """Convert a paper to a clean JSON-serializable dict."""
    ext_ids = paper.get("externalIds") or {}
    return {
        "paperId": paper.get("paperId"),
        "title": paper.get("title"),
        "authors": [a.get("name") for a in (paper.get("authors") or [])],
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "citationCount": paper.get("citationCount"),
        "abstract": paper.get("abstract"),
        "doi": ext_ids.get("DOI"),
        "pmid": ext_ids.get("PubMed"),
        "fieldsOfStudy": paper.get("fieldsOfStudy"),
    }


def main():
    """Run the EZH2 literature search pipeline."""
    start_time = time.time()
    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("EZH2 Literature Search -- Semantic Scholar", flush=True)
    print(f"Started: {run_timestamp}", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Run topic searches
    query_results: dict[str, list[dict]] = {}
    all_papers: list[dict] = []

    for query_name, query_str in SEARCH_QUERIES.items():
        print(f"\nSearching: {query_name} -> '{query_str}'", flush=True)
        papers = search_papers(query_str)
        query_results[query_name] = papers
        all_papers.extend(papers)
        print(f"  Found {len(papers)} papers", flush=True)

    # Step 2: Look up landmark papers by DOI (reliable), fallback to search
    landmark_results: dict[str, list[dict]] = {}

    print("\n--- Landmark paper lookups ---", flush=True)
    for landmark_name, info in LANDMARK_PAPERS.items():
        print(f"\nLooking up: {landmark_name} (DOI: {info['doi']})", flush=True)
        paper = lookup_paper_by_doi(info["doi"])
        if paper:
            landmark_results[landmark_name] = [paper]
            all_papers.append(paper)
            print(f"  Found: {paper.get('title', 'N/A')[:80]} ({paper.get('year')}, {paper.get('citationCount', 0)} citations)", flush=True)
        else:
            print(f"  DOI lookup failed, trying search: '{info['fallback_query']}'", flush=True)
            papers = search_paper_by_title(info["fallback_query"], limit=5)
            landmark_results[landmark_name] = papers
            all_papers.extend(papers)
            if papers:
                top = papers[0]
                print(f"  Best match: {top.get('title', 'N/A')[:80]} ({top.get('year')}, {top.get('citationCount', 0)} citations)", flush=True)
            else:
                print("  No results found", flush=True)

    # Step 3: Deduplicate
    unique_papers = deduplicate_papers(all_papers)
    print(f"\nTotal unique papers: {len(unique_papers)}", flush=True)

    # Step 4: Generate report
    print("\nGenerating report...", flush=True)
    report = generate_report(query_results, landmark_results, unique_papers, run_timestamp)
    REPORT_OUTPUT.write_text(report, encoding="utf-8")
    print(f"Report saved to: {REPORT_OUTPUT}", flush=True)

    # Step 5: Save JSON
    json_data = {
        "metadata": {
            "generated": run_timestamp,
            "total_unique_papers": len(unique_papers),
            "queries": list(SEARCH_QUERIES.keys()),
            "landmark_searches": list(LANDMARK_PAPERS.keys()),
        },
        "query_results": {
            name: [paper_to_json(p) for p in papers]
            for name, papers in query_results.items()
        },
        "landmark_results": {
            name: [paper_to_json(p) for p in papers]
            for name, papers in landmark_results.items()
        },
        "all_unique_papers": [paper_to_json(p) for p in unique_papers],
    }

    JSON_OUTPUT.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON saved to: {JSON_OUTPUT}", flush=True)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
