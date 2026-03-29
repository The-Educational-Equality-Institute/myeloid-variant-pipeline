#!/usr/bin/env python3
"""
harmony_spotfire_query.py -- Use Playwright to query HARMONY Alliance's
Spotfire AML Data Exploration Tool for mutation co-occurrence data.

The Spotfire dashboard at:
  https://spotfire.harmony-platform.eu/spotfire/wp/analysis?file=/Anonymous/AML

allows interactive cohort selection by mutations, cytogenetics, age, gender.
This script automates that interaction to search for our target variants.

Outputs:
    - mutation_profile/results/cross_database/harmony_spotfire_results.json
    - mutation_profile/results/cross_database/harmony_spotfire_report.md
    - mutation_profile/results/cross_database/screenshots/harmony_*.png

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/harmony_spotfire_query.py

Runtime: ~60-120 seconds (page load + interactions)
Dependencies: playwright
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # mutation_profile/
RESULTS_DIR = PROJECT_ROOT / "results" / "cross_database"
SCREENSHOT_DIR = RESULTS_DIR / "screenshots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SPOTFIRE_AML_URL = "https://spotfire.harmony-platform.eu/spotfire/wp/analysis?file=/Anonymous/AML"
TARGET_GENES = ["DNMT3A", "IDH2", "SETBP1", "PTPN11"]
EXACT_VARIANTS = {
    "DNMT3A": "R882H",
    "IDH2": "R140Q",
    "SETBP1": "G870S",
    "PTPN11": "E76Q",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "harmony_spotfire_query.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def take_screenshot(page, name: str) -> str:
    """Take a screenshot and return the path."""
    path = SCREENSHOT_DIR / f"harmony_{name}.png"
    page.screenshot(path=str(path), full_page=True)
    log.info("  Screenshot saved: %s", path.name)
    return str(path)


def main():
    start = time.time()
    log.info("HARMONY Spotfire AML Dashboard Query")
    log.info("URL: %s", SPOTFIRE_AML_URL)
    log.info("Target: %s", ", ".join(f"{g} {v}" for g, v in EXACT_VARIANTS.items()))

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": SPOTFIRE_AML_URL,
        "screenshots": [],
        "findings": {},
        "page_content": {},
        "errors": [],
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        page = context.new_page()

        # ---------------------------------------------------------------
        # Step 1: Load the dashboard
        # ---------------------------------------------------------------
        log.info("=" * 60)
        log.info("Step 1: Loading Spotfire AML dashboard...")
        log.info("=" * 60)

        try:
            page.goto(SPOTFIRE_AML_URL, timeout=60000, wait_until="networkidle")
            log.info("  Page loaded (networkidle)")
        except PlaywrightTimeout:
            log.warning("  Page load timed out at 60s, continuing anyway...")

        # Wait extra time for Spotfire to initialize
        log.info("  Waiting 15s for Spotfire WebPlayer to initialize...")
        page.wait_for_timeout(15000)

        # Take initial screenshot
        ss = take_screenshot(page, "01_initial_load")
        results["screenshots"].append(ss)

        # ---------------------------------------------------------------
        # Step 2: Explore page structure
        # ---------------------------------------------------------------
        log.info("=" * 60)
        log.info("Step 2: Exploring page structure...")
        log.info("=" * 60)

        # Get page title
        title = page.title()
        log.info("  Page title: %s", title)
        results["page_content"]["title"] = title

        # Check for iframes (Spotfire often loads in iframe)
        iframes = page.frames
        log.info("  Found %d frames", len(iframes))
        results["page_content"]["frame_count"] = len(iframes)

        for i, frame in enumerate(iframes):
            log.info("  Frame %d: %s", i, frame.url[:100])

        # Look for Spotfire-specific elements
        spotfire_elements = {
            "sfx_page": page.query_selector_all("[class*='sfx']"),
            "sf_element": page.query_selector_all("[class*='sf-element']"),
            "spotfire": page.query_selector_all("[class*='spotfire']"),
            "filter": page.query_selector_all("[class*='filter']"),
            "checkbox": page.query_selector_all("input[type='checkbox']"),
            "dropdown": page.query_selector_all("select"),
            "button": page.query_selector_all("button"),
            "input": page.query_selector_all("input"),
        }

        for name, elements in spotfire_elements.items():
            count = len(elements)
            if count > 0:
                log.info("  Found %d '%s' elements", count, name)
            results["page_content"][f"{name}_count"] = count

        # Get all visible text to understand the dashboard layout
        try:
            body_text = page.inner_text("body", timeout=5000)
            # Truncate but save key parts
            results["page_content"]["body_text_preview"] = body_text[:5000]
            log.info("  Body text length: %d chars", len(body_text))

            # Search for gene names in text
            for gene in TARGET_GENES:
                if gene in body_text:
                    log.info("  Found '%s' in page text!", gene)
                    # Find context around the gene name
                    idx = body_text.find(gene)
                    context_text = body_text[max(0, idx - 100):idx + 100]
                    results["findings"][f"{gene}_in_text"] = context_text.strip()
                else:
                    log.info("  '%s' NOT found in page text", gene)
        except Exception as e:
            log.warning("  Could not get body text: %s", e)

        # ---------------------------------------------------------------
        # Step 3: Try interacting with Spotfire controls
        # ---------------------------------------------------------------
        log.info("=" * 60)
        log.info("Step 3: Searching for mutation filter controls...")
        log.info("=" * 60)

        # Look for any clickable elements with mutation-related text
        mutation_keywords = ["mutation", "gene", "DNMT3A", "IDH2", "SETBP1", "PTPN11",
                            "Molecular", "Genetic", "NGS", "Panel", "Filter"]

        for keyword in mutation_keywords:
            try:
                elements = page.query_selector_all(f"//*[contains(text(), '{keyword}')]")
                if elements:
                    log.info("  Found %d elements with text '%s'", len(elements), keyword)
                    for el in elements[:3]:
                        tag = el.evaluate("el => el.tagName")
                        text = el.evaluate("el => el.textContent.substring(0, 100)")
                        log.info("    <%s>: %s", tag, text.strip())
                        results["findings"][f"element_{keyword}"] = {
                            "count": len(elements),
                            "sample_tag": tag,
                            "sample_text": text.strip()[:200],
                        }
            except Exception as e:
                pass

        # Try each frame
        for i, frame in enumerate(iframes):
            if i == 0:
                continue  # Skip main frame, already checked
            log.info("  Checking frame %d: %s", i, frame.url[:80])
            try:
                frame_text = frame.inner_text("body", timeout=5000)
                if len(frame_text) > 50:
                    log.info("    Frame %d text length: %d", i, len(frame_text))
                    results["page_content"][f"frame_{i}_text"] = frame_text[:3000]
                    for gene in TARGET_GENES:
                        if gene in frame_text:
                            log.info("    Found '%s' in frame %d!", gene, i)
                            idx = frame_text.find(gene)
                            results["findings"][f"{gene}_in_frame_{i}"] = frame_text[max(0, idx-100):idx+100].strip()
            except Exception:
                pass

        # ---------------------------------------------------------------
        # Step 4: Try to find and interact with Spotfire filters
        # ---------------------------------------------------------------
        log.info("=" * 60)
        log.info("Step 4: Attempting to find Spotfire filter panels...")
        log.info("=" * 60)

        # Spotfire uses specific CSS classes for filter panels
        spotfire_selectors = [
            ".sf-element-filter-panel",
            ".sf-element-checkbox-list",
            ".sf-element-list-box",
            ".sf-element-dropdown-list",
            "[class*='FilterPanel']",
            "[class*='filterPanel']",
            "[class*='filter-panel']",
            "[data-testid*='filter']",
            ".sfx_page_0",
            ".sfx_page_1",
            "#filterPanel",
            ".sfc-filter",
        ]

        for selector in spotfire_selectors:
            try:
                elements = page.query_selector_all(selector)
                if elements:
                    log.info("  Found %d elements for '%s'", len(elements), selector)
                    results["findings"][f"selector_{selector}"] = len(elements)
            except Exception:
                pass

        # Take final screenshot
        ss = take_screenshot(page, "02_explored")
        results["screenshots"].append(ss)

        # ---------------------------------------------------------------
        # Step 5: Get all links/navigation on the page
        # ---------------------------------------------------------------
        log.info("=" * 60)
        log.info("Step 5: Extracting page navigation and tabs...")
        log.info("=" * 60)

        try:
            links = page.query_selector_all("a[href]")
            link_data = []
            for link in links[:50]:
                href = link.get_attribute("href")
                text = link.inner_text()
                if text.strip():
                    link_data.append({"href": href, "text": text.strip()[:100]})
                    log.info("  Link: %s -> %s", text.strip()[:50], href[:80] if href else "")
            results["page_content"]["links"] = link_data
        except Exception as e:
            log.warning("  Could not extract links: %s", e)

        # Try clicking on tabs if they exist
        try:
            tabs = page.query_selector_all("[role='tab'], .sf-element-page-tab, [class*='tab']")
            if tabs:
                log.info("  Found %d tab-like elements", len(tabs))
                for j, tab in enumerate(tabs[:10]):
                    text = tab.inner_text()
                    log.info("    Tab %d: %s", j, text.strip()[:50])
                    results["findings"][f"tab_{j}"] = text.strip()[:100]
        except Exception:
            pass

        # ---------------------------------------------------------------
        # Step 6: Dump full HTML structure for analysis
        # ---------------------------------------------------------------
        log.info("=" * 60)
        log.info("Step 6: Saving page HTML structure...")
        log.info("=" * 60)

        try:
            html = page.content()
            html_path = RESULTS_DIR / "screenshots" / "harmony_page.html"
            with open(html_path, "w") as f:
                f.write(html)
            log.info("  Full HTML saved to %s (%d bytes)", html_path.name, len(html))
            results["page_content"]["html_size"] = len(html)
        except Exception as e:
            log.warning("  Could not save HTML: %s", e)

        browser.close()

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    elapsed = time.time() - start

    json_path = RESULTS_DIR / "harmony_spotfire_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("JSON results saved to %s", json_path)

    # Generate report
    report = generate_report(results, elapsed)
    report_path = RESULTS_DIR / "harmony_spotfire_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Report saved to %s", report_path)

    log.info("Completed in %.1f seconds", elapsed)


def generate_report(results: dict, elapsed: float) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    report = f"""# HARMONY Spotfire Dashboard Query Results

**Generated:** {ts} UTC
**URL:** {results['url']}
**Runtime:** {elapsed:.1f} seconds
**Screenshots:** {len(results['screenshots'])} saved

---

## Page Structure

- **Title:** {results['page_content'].get('title', 'N/A')}
- **Frames:** {results['page_content'].get('frame_count', 'N/A')}
- **HTML size:** {results['page_content'].get('html_size', 'N/A')} bytes

## Elements Found

"""
    for key, val in results['page_content'].items():
        if key.endswith('_count') and val > 0:
            report += f"- **{key}:** {val}\n"

    report += "\n## Findings\n\n"

    if results['findings']:
        for key, val in results['findings'].items():
            report += f"### {key}\n```\n{val}\n```\n\n"
    else:
        report += "No mutation-related controls found in the initial page load.\n\n"

    report += """## Body Text Preview

```
"""
    preview = results['page_content'].get('body_text_preview', 'N/A')
    report += preview[:2000]
    report += """
```

## Screenshots

"""
    for ss in results['screenshots']:
        report += f"- `{Path(ss).name}`\n"

    report += f"""
---

## Interpretation

The Spotfire WebPlayer dashboard requires JavaScript execution and may use
proprietary Spotfire protocols that are not standard HTTP. The initial page load
and exploration reveals the dashboard structure, which can guide manual browser
interaction or more targeted automation.

**Next steps:**
1. Review screenshots to understand the dashboard layout
2. If mutation filters are visible, create targeted Playwright interactions
3. If the dashboard requires specific user interaction (clicks, scrolls), adapt the script
4. As fallback, send the updated email to office@harmony-alliance.eu

---

*Generated by harmony_spotfire_query.py | {ts}*
"""
    return report


if __name__ == "__main__":
    main()
