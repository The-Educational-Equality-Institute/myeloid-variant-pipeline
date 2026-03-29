#!/usr/bin/env python3
"""
Amazon Bedrock Nova Pro — Independent Statistical Methodology Review
=====================================================================

Sends the mutation co-occurrence analysis reports to Amazon Bedrock Nova Pro
for independent statistical methodology critique. Validates p-values, effect
sizes, multiple testing corrections, and overall methodology.

Patient context (ArcherDx VariantPlex Myeloid panel, BM 18.09.2023):
  - DNMT3A R882H  (VAF 39%, Pathogenic hotspot)
  - SETBP1 G870S  (VAF 34%, Likely pathogenic)
  - PTPN11 E76Q   (VAF 29%, Pathogenic)
  - IDH2   R140Q  (VAF 2%,  Pathogenic subclone)
  - EZH2   V662A  (VAF 59%, Pathogenic)
  Cytogenetics: 45,XY,-7[9]/46,XY[1] (monosomy 7 in 90% of metaphases)
  Diagnosis: MDS-IB2 / MDS-AML, ELN 2022 Adverse

Model: us.amazon.nova-pro-v1:0 (via InvokeModel API — NOT Converse)
Region: us-east-1
AWS profile: default
Cost: ~$0.0008/1K input tokens, ~$0.0032/1K output tokens

Usage:
  python mutation_profile/scripts/bedrock_statistical_review.py --review
  python mutation_profile/scripts/bedrock_statistical_review.py --review-section methodology
  python mutation_profile/scripts/bedrock_statistical_review.py --checklist
  python mutation_profile/scripts/bedrock_statistical_review.py --compare
  python mutation_profile/scripts/bedrock_statistical_review.py --dry-run --review
  python mutation_profile/scripts/bedrock_statistical_review.py --verbose --review

Author: Henrik Roine / TEEI
Date: 2026-03-22
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MRNA_ROOT = Path(__file__).resolve().parent.parent.parent  # mrna-hematology-research/
RESULTS_DIR = MRNA_ROOT / "mutation_profile" / "results" / "bedrock_review"
MUTATION_RESULTS = MRNA_ROOT / "mutation_profile" / "results"

# Source files for review (in priority order)
SOURCE_FILES = {
    "cross_database": MUTATION_RESULTS / "CROSS_DATABASE_CONSOLIDATION.md",
    "final_report": MUTATION_RESULTS / "FINAL_cooccurrence_report.md",
    "fisher_exact": MUTATION_RESULTS / "fisher_exact_mutual_exclusivity.md",
    "athena_summary": MUTATION_RESULTS / "athena" / "SUMMARY.md",
    "executive_summary": MRNA_ROOT / "output" / "reports" / "EXECUTIVE_SUMMARY.md",
    "genie_deep": MUTATION_RESULTS / "GENIE_DEEP_ANALYSIS.md",
    "panel_adjusted": MUTATION_RESULTS / "panel_adjusted_statistics.json",
    "age_stratification": MUTATION_RESULTS / "age_stratification_triples.md",
}

# ---------------------------------------------------------------------------
# AWS / Model config
# ---------------------------------------------------------------------------
DEFAULT_REGION = "us-east-1"
DEFAULT_PROFILE = "default"
MODEL_ID = "us.amazon.nova-pro-v1:0"
COST_PER_1K_INPUT = 0.0008
COST_PER_1K_OUTPUT = 0.0032

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
# Review sections — structured prompts
# ---------------------------------------------------------------------------
REVIEW_SECTIONS = {
    "methodology": {
        "title": "Methodology Review",
        "prompt": (
            "You are a biostatistician with expertise in cancer genomics and "
            "population-level mutation co-occurrence analysis. Review the methodology "
            "of this genomic co-occurrence study. Evaluate:\n\n"
            "1. SAMPLE SELECTION: Is the choice of databases appropriate? Are inclusion/"
            "exclusion criteria clearly defined? Is there ascertainment bias from "
            "panel-sequenced vs. WES/WGS cohorts?\n\n"
            "2. DEDUPLICATION: Is the cross-database deduplication rigorous? Are "
            "overlap estimates reasonable? Could double-counting inflate or deflate "
            "co-occurrence rates?\n\n"
            "3. STATISTICAL TESTS: Are Fisher's exact tests appropriate for these "
            "sample sizes and contingency tables? Are one-sided vs. two-sided tests "
            "correctly applied for mutual exclusivity vs. co-occurrence?\n\n"
            "4. MULTIPLE TESTING CORRECTIONS: Is Benjamini-Hochberg (FDR) correctly "
            "applied? Should Bonferroni also be reported? How many effective tests "
            "are being performed?\n\n"
            "5. POTENTIAL BIASES: Panel coverage bias (not all panels cover all 4 "
            "genes), VAF thresholds, hypermutation filtering, center-specific artifacts "
            "(e.g., UCHI intronic variants).\n\n"
            "6. METHODOLOGICAL WEAKNESSES: What are the most important limitations? "
            "What additional analyses would strengthen the findings?\n\n"
            "Be specific. Cite exact numbers from the report when identifying issues."
        ),
        "sources": ["cross_database", "final_report", "fisher_exact"],
    },
    "p-values": {
        "title": "P-value and Effect Size Validation",
        "prompt": (
            "You are a statistician reviewing p-values and effect sizes from a genomic "
            "co-occurrence analysis. For each statistical claim in the document:\n\n"
            "1. P-VALUE APPROPRIATENESS: Are the statistical tests (Fisher's exact, "
            "Poisson, Clopper-Pearson) appropriate for the data type and sample sizes?\n\n"
            "2. EFFECT SIZE VALIDATION: Are odds ratios and observed/expected ratios "
            "correctly calculated from the contingency tables provided? Check the "
            "arithmetic where 2x2 tables are given.\n\n"
            "3. CONFIDENCE INTERVALS: Are confidence intervals provided where needed? "
            "Are Clopper-Pearson upper bounds correctly computed for zero-event data?\n\n"
            "4. MULTIPLE TESTING: The study tests 6 pairwise combinations, 4 triple "
            "combinations, and multiple cross-database comparisons. Is the multiple "
            "testing burden adequately addressed? Which findings survive correction "
            "and which do not?\n\n"
            "5. SPECIFIC CLAIMS TO VERIFY:\n"
            "   a) IDH2+SETBP1 in IPSS-M: OR=0.22, p=0.070 (one-sided depletion)\n"
            "   b) IDH2+PTPN11 in IPSS-M: 0 observed, 2.23 expected, p=0.100\n"
            "   c) PTPN11+SETBP1 in GENIE: OR=6.37, p=2.5e-13\n"
            "   d) DNMT3A+IDH2 in GENIE: OR=8.51, p=1.1e-97\n"
            "   e) Population frequency estimate: <1 in 1,000,000\n\n"
            "For each, state whether the test, value, and interpretation are correct."
        ),
        "sources": ["cross_database", "fisher_exact"],
    },
    "effect-sizes": {
        "title": "Effect Size and Clinical Significance Review",
        "prompt": (
            "You are an epidemiologist reviewing effect sizes in a mutation "
            "co-occurrence study. Evaluate:\n\n"
            "1. ODDS RATIOS: Are the reported ORs (ranging from 0.22 to 8.5) "
            "clinically meaningful? How do they compare to known co-occurrence "
            "patterns in myeloid genomics literature?\n\n"
            "2. OBSERVED/EXPECTED RATIOS: The study uses O/E ratios as primary "
            "effect measures. Is this appropriate? Are there better alternatives "
            "(e.g., mutual information, phi coefficient)?\n\n"
            "3. POPULATION FREQUENCY ESTIMATES: Four methods are used to estimate "
            "the quadruple combination frequency (direct observation, independence "
            "assumption, conditional probability, Poisson). Evaluate each method's "
            "assumptions and validity. Is the <1 in 1,000,000 claim justified?\n\n"
            "4. AGE-ADJUSTED RARITY: The study notes the patient was 33 at diagnosis "
            "vs. median cohort age of 72. How should age be incorporated into the "
            "rarity estimate? Is the age dimension properly quantified?\n\n"
            "5. CLINICAL RELEVANCE: Do the statistical findings translate to "
            "meaningful clinical rarity? Is this profile genuinely unprecedented, "
            "or could it be an artifact of limited database coverage?"
        ),
        "sources": ["cross_database", "age_stratification"],
    },
    "conclusions": {
        "title": "Conclusion Validation",
        "prompt": (
            "You are a peer reviewer for a hematology journal. Given the data "
            "presented, evaluate whether the conclusions are justified:\n\n"
            "1. CLAIM: 'Zero patients carry the quadruple hotspot combination "
            "DNMT3A R882H + IDH2 R140Q + PTPN11 E76Q + SETBP1 G870S across "
            "~275,000 unique patients in 6 independent databases.'\n"
            "   - Is this properly supported by the data?\n"
            "   - Are the database sizes and coverage correctly stated?\n"
            "   - Could ascertainment bias explain the absence?\n\n"
            "2. CLAIM: 'The critical bottleneck is the DNMT3A+IDH2+SETBP1 triple, "
            "which has never been observed with hotspot variants.'\n"
            "   - Is the bottleneck analysis convincing?\n"
            "   - Are alternative explanations considered?\n\n"
            "3. CLAIM: 'Estimated population frequency: <1 in 1,000,000.'\n"
            "   - Is this supported by the four estimation methods?\n"
            "   - What are the confidence bounds?\n"
            "   - Are there alternative explanations?\n\n"
            "4. CLAIM: 'IDH2 and SETBP1 belong to different genomic subgroups "
            "(mutual exclusivity).'\n"
            "   - Is this supported across databases?\n"
            "   - The study notes disease-context dependence (depleted in MDS, "
            "     weakly enriched in broader myeloid). Is this nuance adequately "
            "     addressed?\n\n"
            "5. OVERALL: What additional analyses would you require before "
            "publication? What are the strongest and weakest aspects of this study?"
        ),
        "sources": ["cross_database", "athena_summary"],
    },
}

STROBE_CHECKLIST_PROMPT = (
    "You are a methodologist evaluating an observational genomic study against "
    "the STROBE checklist for cross-sectional studies. The study analyzes mutation "
    "co-occurrence across multiple cancer genomic databases.\n\n"
    "Evaluate each STROBE item (1-22) as: ADEQUATE, PARTIAL, MISSING, or N/A.\n\n"
    "For each item rated PARTIAL or MISSING, provide specific recommendations.\n\n"
    "Additional checklist items specific to genomic co-occurrence studies:\n"
    "- Panel coverage: Are gene panel differences across databases addressed?\n"
    "- Variant classification: Are hotspot vs. non-hotspot vs. VUS distinctions clear?\n"
    "- Deduplication: Is cross-database patient overlap handled?\n"
    "- Hypermutation: Are hypermutated samples identified and excluded?\n"
    "- Multiple testing: Is the FDR correction appropriate for the number of tests?\n"
    "- Sample size adequacy: For zero-event findings, is the sample size sufficient "
    "  to rule out the combination at meaningful frequency thresholds?\n"
    "- Bias assessment: Ascertainment bias (academic center enrichment), "
    "  survivorship bias, detection bias (panel coverage gaps).\n\n"
    "Format your response as a structured checklist with ratings and comments "
    "for each item."
)

COMPARE_PROMPT = (
    "You are a data auditor. Compare the claims in the executive summary against "
    "the raw data provided. For each factual claim (patient counts, p-values, "
    "odds ratios, frequencies), verify whether it matches the underlying data.\n\n"
    "Flag any discrepancies, inconsistencies, or claims that cannot be verified "
    "from the provided data. Categorize each finding as:\n"
    "- VERIFIED: Claim matches data exactly\n"
    "- MINOR DISCREPANCY: Small numerical difference (<5%)\n"
    "- MAJOR DISCREPANCY: Significant numerical difference (>5%)\n"
    "- UNVERIFIABLE: Claim cannot be checked from provided data\n"
    "- INCONSISTENT: Claim contradicts data from another section\n\n"
    "Present findings in a structured table."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_source_content(source_keys: list[str]) -> str:
    """Load and concatenate content from source files."""
    parts = []
    for key in source_keys:
        path = SOURCE_FILES.get(key)
        if path is None:
            log.warning("Unknown source key: %s", key)
            continue
        if not path.exists():
            log.warning("Source file not found: %s", path)
            continue
        content = path.read_text(encoding="utf-8")
        parts.append(f"--- BEGIN {key.upper()} ({path.name}) ---\n{content}\n--- END {key.upper()} ---\n")
        log.info("Loaded %s (%d chars)", path.name, len(content))
    return "\n".join(parts)


def count_tokens_approx(text: str) -> int:
    """Rough token count (~4 chars per token for English/markdown)."""
    return len(text) // 4


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD."""
    return (input_tokens / 1000 * COST_PER_1K_INPUT) + (output_tokens / 1000 * COST_PER_1K_OUTPUT)


def get_bedrock_client(region: str = DEFAULT_REGION, profile: str = DEFAULT_PROFILE):
    """Create a Bedrock Runtime client with the specified AWS profile."""
    import boto3

    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("bedrock-runtime", region_name=region)
    log.info("Bedrock client created (region=%s, profile=%s)", region, profile)
    return client


def invoke_nova_pro(
    client,
    prompt: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> dict:
    """
    Invoke Amazon Nova Pro via InvokeModel API.

    Returns dict with keys: text, input_tokens, output_tokens, cost, latency_s.
    """
    body = json.dumps({
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    })

    input_tokens_approx = count_tokens_approx(prompt)
    log.info(
        "Invoking Nova Pro: ~%d input tokens, max %d output tokens",
        input_tokens_approx,
        max_tokens,
    )

    t0 = time.monotonic()
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    latency = time.monotonic() - t0

    result = json.loads(response["body"].read())
    output_text = result["output"]["message"]["content"][0]["text"]

    # Token usage from response if available, else approximate
    usage = result.get("usage", {})
    in_tok = usage.get("inputTokens", input_tokens_approx)
    out_tok = usage.get("outputTokens", count_tokens_approx(output_text))
    cost = estimate_cost(in_tok, out_tok)

    log.info(
        "Response received: %d input, %d output tokens, $%.4f, %.1fs",
        in_tok, out_tok, cost, latency,
    )

    return {
        "text": output_text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost": cost,
        "latency_s": round(latency, 2),
    }


def save_json(data: dict, path: Path) -> None:
    """Write JSON to file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved: %s", path)


def save_markdown(content: str, path: Path) -> None:
    """Write markdown to file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Review commands
# ---------------------------------------------------------------------------

def run_full_review(
    client,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Run all four review sections and compile into a single report."""
    metadata = {
        "model": MODEL_ID,
        "region": DEFAULT_REGION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sections": {},
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "total_latency_s": 0.0,
    }

    report_parts = [
        "# Independent Statistical Methodology Review",
        "",
        f"**Model:** {MODEL_ID}",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "**Purpose:** Independent critique of mutation co-occurrence analysis methodology",
        "",
        "---",
        "",
    ]

    for section_key, section_cfg in REVIEW_SECTIONS.items():
        log.info("=== Section: %s ===", section_cfg["title"])
        source_content = load_source_content(section_cfg["sources"])

        if not source_content.strip():
            log.error("No source content found for section %s", section_key)
            continue

        full_prompt = (
            f"{section_cfg['prompt']}\n\n"
            f"--- STUDY DATA ---\n\n{source_content}"
        )

        if dry_run:
            log.info("[DRY RUN] Would send %d chars to Nova Pro", len(full_prompt))
            print(f"\n{'='*60}")
            print(f"SECTION: {section_cfg['title']}")
            print(f"{'='*60}")
            print(f"Prompt length: {len(full_prompt)} chars (~{count_tokens_approx(full_prompt)} tokens)")
            print(f"Sources: {', '.join(section_cfg['sources'])}")
            print(f"\nPrompt preview (first 500 chars):\n{full_prompt[:500]}...")
            continue

        result = invoke_nova_pro(client, full_prompt, max_tokens=4096, temperature=0.1)

        # Save individual section JSON
        section_filename = {
            "methodology": "methodology_review.json",
            "p-values": "pvalue_validation.json",
            "effect-sizes": "effect_size_review.json",
            "conclusions": "conclusion_validation.json",
        }.get(section_key, f"{section_key}_review.json")

        save_json(
            {
                "section": section_key,
                "title": section_cfg["title"],
                "sources": section_cfg["sources"],
                "prompt": section_cfg["prompt"],
                "response": result["text"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "cost": result["cost"],
                "latency_s": result["latency_s"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            RESULTS_DIR / section_filename,
        )

        # Accumulate metadata
        metadata["sections"][section_key] = {
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost": result["cost"],
            "latency_s": result["latency_s"],
        }
        metadata["total_input_tokens"] += result["input_tokens"]
        metadata["total_output_tokens"] += result["output_tokens"]
        metadata["total_cost"] += result["cost"]
        metadata["total_latency_s"] += result["latency_s"]

        # Add to report
        report_parts.extend([
            f"## {section_cfg['title']}",
            "",
            result["text"],
            "",
            "---",
            "",
        ])

        if verbose:
            print(f"\n{'='*60}")
            print(f"SECTION: {section_cfg['title']}")
            print(f"{'='*60}")
            print(result["text"])

    if not dry_run:
        # Add cost summary to report
        report_parts.extend([
            "## Review Metadata",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Model | {MODEL_ID} |",
            f"| Total input tokens | {metadata['total_input_tokens']:,} |",
            f"| Total output tokens | {metadata['total_output_tokens']:,} |",
            f"| Total cost | ${metadata['total_cost']:.4f} |",
            f"| Total latency | {metadata['total_latency_s']:.1f}s |",
            f"| Timestamp | {metadata['timestamp']} |",
            "",
        ])

        save_markdown("\n".join(report_parts), RESULTS_DIR / "STATISTICAL_REVIEW.md")
        save_json(metadata, RESULTS_DIR / "review_metadata.json")

        log.info(
            "Full review complete: %d sections, $%.4f total cost",
            len(metadata["sections"]),
            metadata["total_cost"],
        )

    return metadata


def run_section_review(
    client,
    section: str,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Run a single review section."""
    if section not in REVIEW_SECTIONS:
        log.error(
            "Unknown section: %s. Available: %s",
            section,
            ", ".join(REVIEW_SECTIONS.keys()),
        )
        sys.exit(1)

    section_cfg = REVIEW_SECTIONS[section]
    log.info("Running section: %s", section_cfg["title"])

    source_content = load_source_content(section_cfg["sources"])
    if not source_content.strip():
        log.error("No source content found for section %s", section)
        sys.exit(1)

    full_prompt = (
        f"{section_cfg['prompt']}\n\n"
        f"--- STUDY DATA ---\n\n{source_content}"
    )

    if dry_run:
        print(f"\n{'='*60}")
        print(f"SECTION: {section_cfg['title']}")
        print(f"{'='*60}")
        print(f"Prompt length: {len(full_prompt)} chars (~{count_tokens_approx(full_prompt)} tokens)")
        print(f"Sources: {', '.join(section_cfg['sources'])}")
        print(f"\nFull prompt:\n{full_prompt}")
        return {}

    result = invoke_nova_pro(client, full_prompt, max_tokens=4096, temperature=0.1)

    section_filename = {
        "methodology": "methodology_review.json",
        "p-values": "pvalue_validation.json",
        "effect-sizes": "effect_size_review.json",
        "conclusions": "conclusion_validation.json",
    }.get(section, f"{section}_review.json")

    save_json(
        {
            "section": section,
            "title": section_cfg["title"],
            "sources": section_cfg["sources"],
            "prompt": section_cfg["prompt"],
            "response": result["text"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost": result["cost"],
            "latency_s": result["latency_s"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        RESULTS_DIR / section_filename,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"SECTION: {section_cfg['title']}")
        print(f"{'='*60}")
        print(result["text"])

    log.info("Section review complete: $%.4f", result["cost"])
    return result


def run_strobe_checklist(
    client,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Run STROBE checklist evaluation."""
    log.info("Running STROBE checklist evaluation")

    source_content = load_source_content(["cross_database", "final_report", "fisher_exact"])
    if not source_content.strip():
        log.error("No source content found for STROBE checklist")
        sys.exit(1)

    full_prompt = (
        f"{STROBE_CHECKLIST_PROMPT}\n\n"
        f"--- STUDY DATA ---\n\n{source_content}"
    )

    if dry_run:
        print(f"\n{'='*60}")
        print("STROBE CHECKLIST")
        print(f"{'='*60}")
        print(f"Prompt length: {len(full_prompt)} chars (~{count_tokens_approx(full_prompt)} tokens)")
        print(f"\nFull prompt:\n{full_prompt}")
        return {}

    result = invoke_nova_pro(client, full_prompt, max_tokens=4096, temperature=0.1)

    save_json(
        {
            "section": "strobe_checklist",
            "title": "STROBE Checklist Evaluation",
            "prompt": STROBE_CHECKLIST_PROMPT,
            "response": result["text"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost": result["cost"],
            "latency_s": result["latency_s"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        RESULTS_DIR / "strobe_checklist.json",
    )

    if verbose:
        print(f"\n{'='*60}")
        print("STROBE CHECKLIST")
        print(f"{'='*60}")
        print(result["text"])

    log.info("STROBE checklist complete: $%.4f", result["cost"])
    return result


def run_compare(
    client,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Compare executive summary claims against raw data files."""
    log.info("Running claim comparison audit")

    # Load both the summary and raw data sources
    source_content = load_source_content([
        "cross_database",
        "athena_summary",
        "fisher_exact",
        "panel_adjusted",
    ])
    if not source_content.strip():
        log.error("No source content found for comparison")
        sys.exit(1)

    full_prompt = (
        f"{COMPARE_PROMPT}\n\n"
        f"--- STUDY DATA ---\n\n{source_content}"
    )

    if dry_run:
        print(f"\n{'='*60}")
        print("CLAIM COMPARISON AUDIT")
        print(f"{'='*60}")
        print(f"Prompt length: {len(full_prompt)} chars (~{count_tokens_approx(full_prompt)} tokens)")
        print(f"\nFull prompt:\n{full_prompt}")
        return {}

    result = invoke_nova_pro(client, full_prompt, max_tokens=4096, temperature=0.1)

    save_json(
        {
            "section": "claim_comparison",
            "title": "Claim vs. Raw Data Comparison",
            "prompt": COMPARE_PROMPT,
            "response": result["text"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost": result["cost"],
            "latency_s": result["latency_s"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        RESULTS_DIR / "claim_comparison.json",
    )

    if verbose:
        print(f"\n{'='*60}")
        print("CLAIM COMPARISON AUDIT")
        print(f"{'='*60}")
        print(result["text"])

    log.info("Comparison audit complete: $%.4f", result["cost"])
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Amazon Bedrock Nova Pro — Statistical Methodology Review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --review                    Full 4-section review\n"
            "  %(prog)s --review-section methodology Single section review\n"
            "  %(prog)s --checklist                  STROBE checklist\n"
            "  %(prog)s --compare                    Claims vs. raw data\n"
            "  %(prog)s --dry-run --review           Show prompts only\n"
            "  %(prog)s --verbose --review           Print full responses\n"
        ),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--review",
        action="store_true",
        help="Run full statistical review (all 4 sections)",
    )
    mode.add_argument(
        "--review-section",
        choices=list(REVIEW_SECTIONS.keys()),
        metavar="SECTION",
        help=f"Review a specific section: {', '.join(REVIEW_SECTIONS.keys())}",
    )
    mode.add_argument(
        "--checklist",
        action="store_true",
        help="Run STROBE checklist evaluation",
    )
    mode.add_argument(
        "--compare",
        action="store_true",
        help="Compare summary claims against raw data files",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show prompts without calling API",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full model responses to stdout",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"AWS profile (default: {DEFAULT_PROFILE})",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Print source file availability
    log.info("Source file availability:")
    for key, path in SOURCE_FILES.items():
        status = "OK" if path.exists() else "MISSING"
        log.info("  %-20s %s  %s", key, status, path.name)

    # Create client (skip for dry run)
    client = None
    if not args.dry_run:
        try:
            client = get_bedrock_client(region=args.region, profile=args.profile)
        except Exception as e:
            log.error("Failed to create Bedrock client: %s", e)
            sys.exit(1)

    # Dispatch
    if args.review:
        result = run_full_review(client, dry_run=args.dry_run, verbose=args.verbose)
        if not args.dry_run:
            print(f"\nFull review saved to: {RESULTS_DIR / 'STATISTICAL_REVIEW.md'}")
            print(f"Total cost: ${result.get('total_cost', 0):.4f}")

    elif args.review_section:
        result = run_section_review(
            client,
            section=args.review_section,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if not args.dry_run:
            print(f"\nSection review saved to: {RESULTS_DIR}")
            print(f"Cost: ${result.get('cost', 0):.4f}")

    elif args.checklist:
        result = run_strobe_checklist(client, dry_run=args.dry_run, verbose=args.verbose)
        if not args.dry_run:
            print(f"\nSTROBE checklist saved to: {RESULTS_DIR / 'strobe_checklist.json'}")
            print(f"Cost: ${result.get('cost', 0):.4f}")

    elif args.compare:
        result = run_compare(client, dry_run=args.dry_run, verbose=args.verbose)
        if not args.dry_run:
            print(f"\nComparison audit saved to: {RESULTS_DIR / 'claim_comparison.json'}")
            print(f"Cost: ${result.get('cost', 0):.4f}")


if __name__ == "__main__":
    main()
