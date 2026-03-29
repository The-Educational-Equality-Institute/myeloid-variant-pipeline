#!/usr/bin/env python3
"""
harmony_spotfire_interact.py -- Interact with HARMONY Spotfire AML dashboard
to query co-occurrence of DNMT3A + IDH2 + SETBP1 + PTPN11.

Strategy:
1. Load the AML dashboard (12,041 patients)
2. In the Filters panel (right side), find each gene filter
3. Click "True" checkbox for each target gene to filter for patients carrying mutations
4. Record the patient count after each filter applied
5. Screenshot after each step

The Filters panel on the right has gene-level True/False/Unknown checkboxes.
We need to click "True" for DNMT3A, then IDH2, then SETBP1, then PTPN11
to progressively narrow down to co-occurrence.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "cross_database"
SCREENSHOT_DIR = RESULTS_DIR / "screenshots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

TARGET_GENES = ["DNMT3A", "IDH2", "SETBP1", "PTPN11"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "harmony_spotfire_interact.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SPOTFIRE_AML_URL = "https://spotfire.harmony-platform.eu/spotfire/wp/analysis?file=/Anonymous/AML"


def screenshot(page, name: str) -> str:
    path = SCREENSHOT_DIR / f"harmony_{name}.png"
    page.screenshot(path=str(path), full_page=False)
    log.info("  Screenshot: %s", path.name)
    return str(path)


def get_patient_count(page) -> str:
    """Try to extract the patient count from the dashboard header."""
    try:
        # The header shows "12,041 AML Patients"
        text = page.inner_text("body", timeout=3000)
        for line in text.split("\n"):
            if "AML Patients" in line or "AML patients" in line:
                return line.strip()
            if "patients" in line.lower() and any(c.isdigit() for c in line):
                return line.strip()
        # Try to find the count element directly
        elements = page.query_selector_all("//*[contains(text(), 'AML Patients')]")
        if elements:
            return elements[0].inner_text().strip()
    except Exception:
        pass
    return "unknown"


def find_and_click_gene_filter(page, gene: str) -> bool:
    """Find a gene filter in the right panel and click 'True' to select only mutated patients."""
    log.info("  Looking for %s filter...", gene)

    # The filter panel has sections like:
    #   DNMT3A
    #   ☐ False
    #   ☑ True
    #   ☐ Unknown
    # We need to find the gene label, then click the "True" checkbox nearby

    # Strategy 1: Find the gene text span, then find the nearby "True" checkbox
    try:
        # Find all elements containing the gene name
        gene_elements = page.query_selector_all(f"//*[contains(text(), '{gene}')]")
        log.info("    Found %d elements with '%s'", len(gene_elements), gene)

        for el in gene_elements:
            # Check if this is in the filter panel (right side)
            try:
                parent = el.evaluate("""el => {
                    let p = el;
                    for (let i = 0; i < 10; i++) {
                        p = p.parentElement;
                        if (!p) return null;
                        if (p.className && (p.className.includes('filter') || p.className.includes('Filter')))
                            return {class: p.className, tag: p.tagName};
                    }
                    return null;
                }""")
                if parent:
                    log.info("    Found %s in filter panel: %s", gene, parent)
            except Exception:
                pass
    except Exception as e:
        log.warning("    Error finding %s elements: %s", gene, e)

    # Strategy 2: Use the Spotfire filter structure
    # Spotfire checkbox filters have a specific DOM structure
    # Look for sf-element-checkbox-list items
    try:
        # The right filter panel has sections. Each gene section has:
        # A header span with the gene name
        # Checkboxes with "True", "False", "Unknown" labels
        # We want to find the section for this gene, then click "True"

        # First, scroll the filter panel to find the gene
        filter_panels = page.query_selector_all("[class*='FilterPanel'], [class*='filter-panel'], .sf-element-filter-panel")
        log.info("    Found %d filter panels", len(filter_panels))

        # Try clicking directly on the gene name text to expand/focus it
        gene_span = page.query_selector(f"span:has-text('{gene}')")
        if gene_span:
            # Get the bounding box to see if it's in the filter area (right side of screen)
            box = gene_span.bounding_box()
            if box and box['x'] > 800:  # Right side of 1920px viewport
                log.info("    Found %s in filter area at x=%d, y=%d", gene, box['x'], box['y'])

                # Look for "True" text near this element
                # The True checkbox should be below the gene name
                # Try finding sibling/child elements
                parent_section = gene_span.evaluate("""el => {
                    let p = el.parentElement;
                    while (p && p.tagName !== 'BODY') {
                        // Look for the filter section container
                        if (p.className && (
                            p.className.includes('sfx_') ||
                            p.className.includes('sf-element') ||
                            p.className.includes('sfc-')
                        )) {
                            return {
                                class: p.className.substring(0, 100),
                                html: p.innerHTML.substring(0, 500),
                                tag: p.tagName
                            };
                        }
                        p = p.parentElement;
                    }
                    return null;
                }""")
                if parent_section:
                    log.info("    Parent section: %s <%s>", parent_section['class'][:50], parent_section['tag'])

                # Try to find the True checkbox within a reasonable distance
                # Look for checkboxes below the gene name
                true_elements = page.query_selector_all(f"//*[contains(text(), 'True')]")
                for true_el in true_elements:
                    true_box = true_el.bounding_box()
                    if true_box and box:
                        # Check if this "True" is close to and below the gene name
                        x_close = abs(true_box['x'] - box['x']) < 100
                        y_below = 0 < (true_box['y'] - box['y']) < 80
                        if x_close and y_below:
                            log.info("    Found 'True' checkbox for %s at y=%d (gene at y=%d)",
                                    gene, true_box['y'], box['y'])
                            true_el.click()
                            log.info("    CLICKED 'True' for %s!", gene)
                            page.wait_for_timeout(3000)  # Wait for filter to apply
                            return True
            else:
                log.info("    %s span at x=%d (not in filter area)", gene, box['x'] if box else -1)
    except Exception as e:
        log.warning("    Strategy 2 error: %s", e)

    # Strategy 3: Try scrolling the filter panel and looking for gene checkboxes
    try:
        # Find the scrollable filter container on the right side
        # and scroll down to find the gene
        filter_area = page.query_selector("[class*='filter-panel'], [class*='FilterPanel']")
        if filter_area:
            # Scroll within the filter area
            for scroll_y in range(0, 2000, 200):
                filter_area.evaluate(f"el => el.scrollTop = {scroll_y}")
                page.wait_for_timeout(500)

                # Check if gene appeared
                gene_span = page.query_selector(f"span:has-text('{gene}')")
                if gene_span:
                    box = gene_span.bounding_box()
                    if box and box['x'] > 800:
                        log.info("    Found %s after scrolling to %d", gene, scroll_y)
                        break
    except Exception as e:
        log.warning("    Strategy 3 error: %s", e)

    log.warning("    Could not find/click filter for %s", gene)
    return False


def main():
    start = time.time()
    log.info("HARMONY Spotfire Interactive Query")
    log.info("=" * 60)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": SPOTFIRE_AML_URL,
        "steps": [],
        "screenshots": [],
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        )
        page = context.new_page()

        # Load dashboard
        log.info("Loading Spotfire AML dashboard...")
        try:
            page.goto(SPOTFIRE_AML_URL, timeout=60000, wait_until="networkidle")
        except PlaywrightTimeout:
            log.warning("Page load timed out, continuing...")

        log.info("Waiting 20s for Spotfire to initialize...")
        page.wait_for_timeout(20000)

        # Initial state
        count = get_patient_count(page)
        log.info("Initial patient count: %s", count)
        ss = screenshot(page, "interact_00_initial")
        results["screenshots"].append(ss)
        results["steps"].append({
            "step": 0,
            "action": "Initial load",
            "patient_count": count,
            "screenshot": ss,
        })

        # First, let's map out the filter panel structure
        log.info("=" * 60)
        log.info("Mapping filter panel structure...")
        log.info("=" * 60)

        # Get the full text of the body to find all gene names and their positions
        body_text = page.inner_text("body", timeout=5000)
        log.info("Full body text:\n%s", body_text[:3000])

        # Find all the filter section headers (gene names in filter panel)
        filter_spans = page.query_selector_all("span")
        gene_positions = {}
        for span in filter_spans:
            try:
                text = span.inner_text().strip()
                box = span.bounding_box()
                if text in ["DNMT3A", "IDH2", "SETBP1", "PTPN11", "NPM1", "TP53",
                           "ASXL1", "BCOR", "EZH2", "FLT3_ITD", "FLT3_TKD",
                           "NRAS", "TET2", "SRSF2", "RUNX1", "IDH1", "KIT",
                           "WT1", "STAG2", "KRAS", "CEBPA", "CSF3R", "SF3B1",
                           "U2AF1", "ZRSR2", "PHF6", "CBL", "CALR", "JAK2",
                           "MPL", "BRAF", "HRAS", "GNB1", "NF1", "BCORL1",
                           "ETNK1", "ETV6", "GATA2", "KMT2A", "PPM1D", "PRPF8",
                           "UBA1"]:
                    if box:
                        gene_positions[text] = {"x": box['x'], "y": box['y'], "w": box['width'], "h": box['height']}
                        if box['x'] > 700:  # Filter panel is on right side
                            log.info("  Filter gene: %s at (%d, %d)", text, box['x'], box['y'])
            except Exception:
                pass

        results["gene_positions"] = gene_positions

        # Now try to interact with each target gene filter
        log.info("=" * 60)
        log.info("Applying gene mutation filters...")
        log.info("=" * 60)

        for i, gene in enumerate(TARGET_GENES):
            log.info("")
            log.info("--- Filter %d: %s ---", i + 1, gene)

            # Re-scan gene positions after each filter (layout may shift)
            for span in page.query_selector_all("span"):
                try:
                    text = span.inner_text().strip()
                    box = span.bounding_box()
                    if text == gene and box and box['x'] > 700:
                        gene_positions[gene] = {"x": box['x'], "y": box['y'], "w": box['width'], "h": box['height']}
                except Exception:
                    pass

            # If gene is off-screen (y > 1000), scroll the filter panel first
            if gene in gene_positions and gene_positions[gene]['y'] > 950:
                log.info("  %s is at y=%d (off-screen), scrolling filter panel...", gene, gene_positions[gene]['y'])
                # Scroll the filter panel down
                page.evaluate("""() => {
                    // Find the main filter panel scrollable container
                    const containers = document.querySelectorAll('[class*="filter"], [class*="Filter"]');
                    for (const c of containers) {
                        const rect = c.getBoundingClientRect();
                        if (rect.x > 700 && c.scrollHeight > c.clientHeight + 10) {
                            c.scrollTop += 400;
                            return true;
                        }
                    }
                    // Fallback: scroll all sf-element containers on right side
                    const sfElements = document.querySelectorAll('.sf-element');
                    for (const el of sfElements) {
                        const rect = el.getBoundingClientRect();
                        if (rect.x > 700 && el.scrollHeight > el.clientHeight + 50) {
                            el.scrollTop += 400;
                            return true;
                        }
                    }
                    return false;
                }""")
                page.wait_for_timeout(1000)

                # Re-find the gene position after scrolling
                for span in page.query_selector_all("span"):
                    try:
                        text = span.inner_text().strip()
                        box = span.bounding_box()
                        if text == gene and box and box['x'] > 700:
                            gene_positions[gene] = {"x": box['x'], "y": box['y'], "w": box['width'], "h": box['height']}
                            log.info("  %s now at y=%d after scroll", gene, box['y'])
                    except Exception:
                        pass

            if gene in gene_positions and gene_positions[gene]['x'] > 700:
                pos = gene_positions[gene]
                log.info("  Gene %s found in filter panel at (%d, %d)", gene, pos['x'], pos['y'])

                # The checkbox structure in Spotfire:
                # Each gene section has the gene name, then checkboxes below it
                # Look for checkboxes/clickable items near this position
                # The "True" option is typically the second item after the gene name

                # Try to find and click "True" near this gene
                # Get all elements in the vicinity
                nearby_elements = page.evaluate("""(pos) => {
                    const results = [];
                    const allElements = document.querySelectorAll('*');
                    for (const el of allElements) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width === 0 || rect.height === 0) continue;
                        const xClose = Math.abs(rect.x - pos.x) < 120;
                        const yBelow = rect.y > pos.y && rect.y < pos.y + 80;
                        if (xClose && yBelow) {
                            let cn = '';
                            try { cn = typeof el.className === 'string' ? el.className.substring(0, 80) : ''; } catch(e) {}
                            results.push({
                                tag: el.tagName,
                                class: cn,
                                text: el.textContent ? el.textContent.substring(0, 50).trim() : '',
                                x: Math.round(rect.x),
                                y: Math.round(rect.y),
                                w: Math.round(rect.width),
                                h: Math.round(rect.height),
                            });
                        }
                    }
                    return results;
                }""", pos)

                log.info("  Found %d elements near %s:", len(nearby_elements), gene)
                for el in nearby_elements[:15]:
                    log.info("    <%s> '%s' at (%d,%d) %s", el['tag'], el['text'][:30], el['x'], el['y'], el['class'][:40])

                # Spotfire checkbox filters: all items start CHECKED.
                # To show only True, we need to UNCHECK "False" and "Unknown".
                # Use element.click() (not mouse.click) to handle off-screen elements.
                clicked = False

                # Find the "False" and "Unknown" filter items for this gene using DOM queries
                # The gene name is at the known position. We find the filter items using
                # Spotfire's DOM structure: the gene label and its filter items are siblings.
                filter_items = page.query_selector_all(".sf-element-filter-item")
                gene_y = pos['y']

                for item in filter_items:
                    try:
                        box = item.bounding_box()
                        if not box:
                            continue
                        text = item.inner_text().strip()
                        # Match filter items close to this gene's y position
                        x_match = abs(box['x'] - pos['x']) < 120
                        y_below = 0 < (box['y'] - gene_y) < 80

                        if x_match and y_below:
                            if text == 'False':
                                log.info("  Unchecking 'False' for %s at (%d, %d)", gene, box['x'], box['y'])
                                item.click(force=True)
                                page.wait_for_timeout(2000)
                                clicked = True
                            elif text == 'Unknown':
                                log.info("  Unchecking 'Unknown' for %s at (%d, %d)", gene, box['x'], box['y'])
                                item.click(force=True)
                                page.wait_for_timeout(2000)
                    except Exception as e:
                        log.warning("    Error clicking filter item: %s", e)

                page.wait_for_timeout(2000)  # Wait for filter to fully apply

                if not clicked:
                    # Try clicking the gene name itself to see if it toggles
                    log.info("  No 'True' found near %s, trying to click gene name", gene)
                    page.mouse.click(pos['x'] + 5, pos['y'] + 5)
                    page.wait_for_timeout(2000)

                # Check new patient count
                count = get_patient_count(page)
                log.info("  Patient count after %s filter: %s", gene, count)
                ss = screenshot(page, f"interact_{i+1:02d}_{gene}")
                results["screenshots"].append(ss)
                results["steps"].append({
                    "step": i + 1,
                    "gene": gene,
                    "action": f"Filter by {gene}=True",
                    "clicked": clicked,
                    "patient_count": count,
                    "screenshot": ss,
                })
            else:
                log.warning("  %s not found in filter panel — may need scrolling", gene)

                # Try scrolling the filter panel to find it
                # The filter panel is the rightmost column
                log.info("  Attempting to scroll filter panel to find %s...", gene)

                # Find the filter panel container
                scroll_container = page.evaluate("""() => {
                    const panels = document.querySelectorAll('[class*="filter"], [class*="Filter"]');
                    for (const p of panels) {
                        if (p.scrollHeight > p.clientHeight && p.getBoundingClientRect().x > 700) {
                            return {
                                class: p.className.substring(0, 80),
                                scrollHeight: p.scrollHeight,
                                clientHeight: p.clientHeight,
                                x: Math.round(p.getBoundingClientRect().x),
                            };
                        }
                    }
                    return null;
                }""")

                if scroll_container:
                    log.info("  Found scrollable filter at x=%d (scroll: %d/%d)",
                            scroll_container['x'], scroll_container['scrollHeight'], scroll_container['clientHeight'])

                    # Scroll through the filter panel
                    for scroll_pos in range(0, scroll_container['scrollHeight'], 100):
                        page.evaluate(f"""() => {{
                            const panels = document.querySelectorAll('[class*="filter"], [class*="Filter"]');
                            for (const p of panels) {{
                                if (p.scrollHeight > p.clientHeight && p.getBoundingClientRect().x > 700) {{
                                    p.scrollTop = {scroll_pos};
                                    break;
                                }}
                            }}
                        }}""")
                        page.wait_for_timeout(300)

                        # Check if gene appeared
                        gene_span = page.query_selector(f"span:has-text('{gene}')")
                        if gene_span:
                            gbox = gene_span.bounding_box()
                            if gbox and gbox['x'] > 700:
                                log.info("  Found %s at scroll position %d, y=%d", gene, scroll_pos, gbox['y'])
                                # Now look for True checkbox nearby
                                nearby = page.evaluate("""(pos) => {
                                    const results = [];
                                    const all = document.querySelectorAll('*');
                                    for (const el of all) {
                                        const r = el.getBoundingClientRect();
                                        if (r.width === 0) continue;
                                        const xClose = Math.abs(r.x - pos.x) < 120;
                                        const yBelow = r.y > pos.y && r.y < pos.y + 80;
                                        if (xClose && yBelow) {
                                            results.push({
                                                text: (el.textContent || '').substring(0,50).trim(),
                                                x: Math.round(r.x),
                                                y: Math.round(r.y),
                                            });
                                        }
                                    }
                                    return results;
                                }""", {"x": gbox['x'], "y": gbox['y']})

                                for nel in nearby:
                                    if 'True' in nel.get('text', ''):
                                        log.info("  Clicking True for %s at (%d,%d)", gene, nel['x'], nel['y'])
                                        page.mouse.click(nel['x'] + 5, nel['y'] + 5)
                                        page.wait_for_timeout(3000)
                                        break
                                break

                count = get_patient_count(page)
                ss = screenshot(page, f"interact_{i+1:02d}_{gene}")
                results["screenshots"].append(ss)
                results["steps"].append({
                    "step": i + 1,
                    "gene": gene,
                    "action": f"Scroll+filter by {gene}=True",
                    "patient_count": count,
                    "screenshot": ss,
                })

        # Final screenshot
        ss = screenshot(page, "interact_final")
        results["screenshots"].append(ss)

        browser.close()

    # Save results
    elapsed = time.time() - start
    json_path = RESULTS_DIR / "harmony_spotfire_interact_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", json_path)
    log.info("Completed in %.1f seconds", elapsed)

    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("FILTER RESULTS SUMMARY")
    log.info("=" * 60)
    for step in results["steps"]:
        log.info("  Step %d (%s): %s patients",
                step["step"], step.get("gene", "initial"), step["patient_count"])


if __name__ == "__main__":
    main()
