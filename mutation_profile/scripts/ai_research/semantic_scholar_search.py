#!/usr/bin/env python3
"""
semantic_scholar_search.py -- Semantic Scholar API integration for literature synthesis.

Uses the Semantic Scholar Academic Graph API (https://api.semanticscholar.org/graph/v1/)
to search for papers about each patient variant, build citation networks, identify
key research groups, and find the most-cited papers.

No API key required for basic access (rate limited to 100 requests per 5 minutes).

Inputs:
    - Semantic Scholar API (remote, no auth required)

Outputs:
    - mutation_profile/results/ai_research/literature_synthesis/semantic_scholar_analysis.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/semantic_scholar_search.py

Runtime: ~2-4 minutes (rate-limited API calls)
Dependencies: requests
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
OUTPUT_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "literature_synthesis"
OUTPUT_FILE = OUTPUT_DIR / "semantic_scholar_analysis.md"

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
load_dotenv(ENV_PATH)
S2_API_KEY = os.getenv("S2_API_KEY", "")  # Optional: apply at https://www.semanticscholar.org/product/api#api-key-form

BASE_URL = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,authors,year,citationCount,abstract,externalIds,venue,referenceCount,fieldsOfStudy"
CITATION_FIELDS = "paperId,title,authors,year,citationCount"
RATE_LIMIT_SLEEP = 3.5  # seconds between requests (conservative to avoid 429s)

# Shared session with proper headers
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "mrna-hematology-research/1.0 (academic; mailto:research@theeducationalequalityinstitute.org)",
})
if S2_API_KEY:
    SESSION.headers["x-api-key"] = S2_API_KEY
    print(f"Using Semantic Scholar API key (higher rate limits)")
else:
    print("No S2_API_KEY found in .env -- using public rate limits (100 req/5min)")

# ---------------------------------------------------------------------------
# Search queries per variant
# ---------------------------------------------------------------------------
VARIANT_QUERIES = {
    "DNMT3A R882H": "DNMT3A R882H myeloid",
    "IDH2 R140Q": "IDH2 R140Q AML MDS",
    "SETBP1 G870S": "SETBP1 G870S myeloid",
    "PTPN11 E76Q": "PTPN11 E76Q leukemia",
    "SETBP1+IDH2 co-occurrence": "SETBP1 IDH2 co-occurrence",
    "Quadruple mutation": "quadruple mutation myeloid",
}

MAX_RESULTS_PER_QUERY = 100


def api_get(url: str, params: dict | None = None) -> dict | None:
    """Make a GET request to Semantic Scholar API with rate limiting and retries."""
    for attempt in range(5):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
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
    """Search Semantic Scholar for papers matching a query.

    Uses the bulk search endpoint (paper/search/bulk) which returns up to 1000
    results per call with token-based pagination and separate rate limits from
    the regular search endpoint.
    """
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

        # Bulk endpoint uses token-based pagination
        continuation_token = data.get("token")
        if not continuation_token:
            break

    return all_papers[:limit]


def get_paper_citations(paper_id: str, limit: int = 100) -> list[dict]:
    """Get papers that cite a given paper."""
    params = {"fields": CITATION_FIELDS, "limit": limit}
    data = api_get(f"{BASE_URL}/paper/{paper_id}/citations", params)
    time.sleep(RATE_LIMIT_SLEEP)
    if data and data.get("data"):
        return [c["citingPaper"] for c in data["data"] if c and c.get("citingPaper")]
    return []


def get_paper_references(paper_id: str, limit: int = 100) -> list[dict]:
    """Get papers referenced by a given paper."""
    params = {"fields": CITATION_FIELDS, "limit": limit}
    data = api_get(f"{BASE_URL}/paper/{paper_id}/references", params)
    time.sleep(RATE_LIMIT_SLEEP)
    if data and data.get("data"):
        return [r["citedPaper"] for r in data["data"] if r and r.get("citedPaper")]
    return []


def extract_author_info(papers: list[dict]) -> dict:
    """Extract author publication counts and identify key research groups."""
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

    return {
        "counts": author_counts,
        "ids": author_ids,
        "papers": dict(author_papers),
    }


def build_citation_network(variant_papers: dict[str, list[dict]]) -> dict:
    """Build a citation network between the top papers across variants.

    For the top 5 most-cited papers per variant, fetch their citations and
    references to find cross-variant citation connections.
    """
    # Collect top papers across all variants
    top_papers = {}
    for variant, papers in variant_papers.items():
        sorted_p = sorted(papers, key=lambda p: p.get("citationCount", 0), reverse=True)
        for paper in sorted_p[:3]:
            pid = paper.get("paperId")
            if pid:
                top_papers[pid] = {
                    "variant": variant,
                    "title": paper.get("title", ""),
                    "citations": paper.get("citationCount", 0),
                }

    all_paper_ids = set(top_papers.keys())
    # Also collect all paper IDs from the full results for cross-referencing
    all_result_ids = set()
    for papers in variant_papers.values():
        for p in papers:
            if p.get("paperId"):
                all_result_ids.add(p["paperId"])

    connections = []
    print(f"\nBuilding citation network for {len(top_papers)} top papers...")

    for pid, info in top_papers.items():
        # Get citations of this paper
        citing = get_paper_citations(pid, limit=20)
        for cp in citing:
            cpid = cp.get("paperId")
            if cpid and cpid in all_result_ids and cpid != pid:
                connections.append({
                    "from": info["title"],
                    "from_variant": info["variant"],
                    "to": cp.get("title", ""),
                    "type": "cited_by",
                })

        # Get references from this paper
        refs = get_paper_references(pid, limit=20)
        for rp in refs:
            rpid = rp.get("paperId")
            if rpid and rpid in all_result_ids and rpid != pid:
                connections.append({
                    "from": info["title"],
                    "from_variant": info["variant"],
                    "to": rp.get("title", ""),
                    "type": "references",
                })

    return {
        "top_papers": top_papers,
        "connections": connections,
        "cross_variant_links": len(connections),
    }


def compute_publication_trends(papers: list[dict]) -> dict[int, int]:
    """Count papers per year."""
    year_counts = Counter()
    for p in papers:
        year = p.get("year")
        if year and isinstance(year, int) and 1990 <= year <= 2026:
            year_counts[year] += 1
    return dict(sorted(year_counts.items()))


def format_paper_entry(paper: dict, rank: int | None = None) -> str:
    """Format a single paper entry for the markdown report."""
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
        truncated = abstract[:300] + "..." if len(abstract) > 300 else abstract
        lines.append(f"   > {truncated}")

    return "\n".join(lines)


def generate_report(
    variant_papers: dict[str, list[dict]],
    citation_network: dict,
    run_timestamp: str,
) -> str:
    """Generate the full markdown analysis report."""
    lines = [
        "# Semantic Scholar Literature Analysis",
        "",
        f"**Generated:** {run_timestamp}",
        f"**API:** Semantic Scholar Academic Graph API v1",
        f"**Queries:** {len(VARIANT_QUERIES)} variant-specific searches",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        "| Variant Query | Papers Found | Most-Cited (citations) |",
        "|---|---|---|",
    ]

    # Summary table
    all_papers = []
    for variant, query in VARIANT_QUERIES.items():
        papers = variant_papers.get(variant, [])
        all_papers.extend(papers)
        if papers:
            top = max(papers, key=lambda p: p.get("citationCount", 0))
            top_str = f"{top.get('title', 'N/A')[:60]}... ({top.get('citationCount', 0)})"
        else:
            top_str = "N/A"
        lines.append(f"| {variant} | {len(papers)} | {top_str} |")

    # Deduplicate papers by paperId
    seen_ids = set()
    unique_papers = []
    for p in all_papers:
        pid = p.get("paperId")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique_papers.append(p)

    lines.extend([
        "",
        f"**Total unique papers across all queries:** {len(unique_papers)}",
        "",
        "---",
        "",
    ])

    # Per-variant detailed results
    lines.append("## Per-Variant Results")
    lines.append("")

    for variant, query in VARIANT_QUERIES.items():
        papers = variant_papers.get(variant, [])
        lines.append(f"### {variant}")
        lines.append(f"**Query:** `{query}` | **Results:** {len(papers)} papers")
        lines.append("")

        if not papers:
            lines.append("No papers found for this query.")
            lines.append("")
            continue

        # Publication trend
        trends = compute_publication_trends(papers)
        if trends:
            lines.append("**Publication trend (papers per year):**")
            lines.append("")
            lines.append("| Year | Papers |")
            lines.append("|---|---|")
            for year, count in trends.items():
                bar = "#" * count
                lines.append(f"| {year} | {count} {bar} |")
            lines.append("")

        # Top 5 most-cited
        sorted_papers = sorted(papers, key=lambda p: p.get("citationCount", 0), reverse=True)
        lines.append("**Top 5 most-cited papers:**")
        lines.append("")
        for i, paper in enumerate(sorted_papers[:5], 1):
            lines.append(format_paper_entry(paper, rank=i))
            lines.append("")

        # Key authors for this variant
        author_info = extract_author_info(papers)
        top_authors = author_info["counts"].most_common(10)
        if top_authors:
            lines.append("**Key authors (by paper count in results):**")
            lines.append("")
            lines.append("| Author | Papers | S2 Author ID |")
            lines.append("|---|---|---|")
            for name, count in top_authors:
                aid = author_info["ids"].get(name, "N/A")
                aid_link = f"[{aid}](https://www.semanticscholar.org/author/{aid})" if aid and aid != "N/A" else "N/A"
                lines.append(f"| {name} | {count} | {aid_link} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Global top 10 most-cited papers
    lines.append("## Top 10 Most-Cited Papers (All Variants)")
    lines.append("")
    global_sorted = sorted(unique_papers, key=lambda p: p.get("citationCount", 0), reverse=True)
    for i, paper in enumerate(global_sorted[:10], 1):
        lines.append(format_paper_entry(paper, rank=i))
        lines.append("")

    lines.append("---")
    lines.append("")

    # Global author analysis
    lines.append("## Key Research Groups (All Variants)")
    lines.append("")
    global_author_info = extract_author_info(unique_papers)
    top_global_authors = global_author_info["counts"].most_common(20)
    if top_global_authors:
        lines.append("| Rank | Author | Total Papers | S2 Author ID |")
        lines.append("|---|---|---|---|")
        for rank, (name, count) in enumerate(top_global_authors, 1):
            aid = global_author_info["ids"].get(name, "N/A")
            aid_link = f"[{aid}](https://www.semanticscholar.org/author/{aid})" if aid and aid != "N/A" else "N/A"
            lines.append(f"| {rank} | {name} | {count} | {aid_link} |")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Citation network
    lines.append("## Citation Network")
    lines.append("")
    lines.append(f"**Cross-variant citation links found:** {citation_network['cross_variant_links']}")
    lines.append("")

    connections = citation_network.get("connections", [])
    if connections:
        # Deduplicate connections
        seen_conns = set()
        unique_conns = []
        for c in connections:
            key = (c["from"][:50], c["to"][:50], c["type"])
            if key not in seen_conns:
                seen_conns.add(key)
                unique_conns.append(c)

        lines.append(f"**Unique citation connections:** {len(unique_conns)}")
        lines.append("")
        lines.append("| Source Paper | Variant | Relationship | Target Paper |")
        lines.append("|---|---|---|---|")
        for c in unique_conns[:30]:  # Show top 30
            from_short = c["from"][:60] + ("..." if len(c["from"]) > 60 else "")
            to_short = c["to"][:60] + ("..." if len(c["to"]) > 60 else "")
            lines.append(f"| {from_short} | {c['from_variant']} | {c['type']} | {to_short} |")
        if len(unique_conns) > 30:
            lines.append(f"| ... | ... | ... | *{len(unique_conns) - 30} more connections* |")
        lines.append("")
    else:
        lines.append("No direct citation connections found between top papers across variant queries.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Global publication trend
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

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Data source:** Semantic Scholar Academic Graph API v1")
    lines.append(f"- **Rate limiting:** {RATE_LIMIT_SLEEP}s sleep between API calls")
    lines.append(f"- **Max results per query:** {MAX_RESULTS_PER_QUERY}")
    lines.append("- **Citation network:** Top 3 most-cited papers per variant, cross-referenced against all results")
    lines.append("- **Author clustering:** Based on co-occurrence in search results (not institutional affiliation)")
    lines.append("- **Deduplication:** Papers appearing in multiple variant searches counted once in global stats")
    lines.append("")

    return "\n".join(lines)


def main():
    """Run the full Semantic Scholar literature analysis pipeline."""
    start_time = time.time()
    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("Semantic Scholar Literature Analysis", flush=True)
    print(f"Started: {run_timestamp}", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Search for papers per variant
    variant_papers: dict[str, list[dict]] = {}

    for variant, query in VARIANT_QUERIES.items():
        print(f"\nSearching: {variant} -> '{query}'", flush=True)
        papers = search_papers(query)
        variant_papers[variant] = papers
        print(f"  Found {len(papers)} papers", flush=True)

    # Step 2: Build citation network
    citation_network = build_citation_network(variant_papers)

    # Step 3: Generate report
    print("\nGenerating report...")
    report = generate_report(variant_papers, citation_network, run_timestamp)

    OUTPUT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {OUTPUT_FILE}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print("=" * 70)

    # Save raw data as JSON for downstream use
    raw_output = OUTPUT_DIR / "semantic_scholar_raw.json"
    raw_data = {}
    for variant, papers in variant_papers.items():
        raw_data[variant] = [
            {
                "paperId": p.get("paperId"),
                "title": p.get("title"),
                "year": p.get("year"),
                "citationCount": p.get("citationCount"),
                "doi": (p.get("externalIds") or {}).get("DOI"),
                "venue": p.get("venue"),
                "authors": [a.get("name") for a in (p.get("authors") or [])],
            }
            for p in papers
        ]
    raw_output.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Raw data saved to: {raw_output}")


if __name__ == "__main__":
    main()
