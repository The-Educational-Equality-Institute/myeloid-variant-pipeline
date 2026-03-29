#!/usr/bin/env python3
"""
evo2_variant_scoring.py -- Evo 2 (40B) DNA-level variant pathogenicity scoring.

Scores 5 patient somatic mutations using the Evo 2 DNA foundation model
via the NVIDIA NIM API. Evo 2 is a 40-billion-parameter genomic language model
trained on 9.3 trillion nucleotides spanning prokaryotic and eukaryotic genomes,
providing DNA-level variant effect prediction that captures regulatory, splicing,
and coding context simultaneously.

This provides a fundamentally different signal from protein-level ESM-2 scoring:
  - ESM-2 operates on amino acid sequences (protein context only)
  - Evo 2 operates on raw DNA nucleotide sequences (full genomic context)
  - DNA-level models can detect effects invisible to protein models:
    regulatory element disruption, codon usage bias, splicing signals,
    secondary structure at the DNA level, and transcription factor binding motifs

Pipeline:
  1. Fetch 10 kb genomic context around each variant from GRCh38 (Ensembl REST API)
  2. Construct reference and alternate sequences (substitution at center)
  3. Score both sequences via NVIDIA NIM Evo 2 40B log-likelihood endpoint
  4. Compute delta log-likelihood as pathogenicity signal
  5. Save intermediate results after each variant
  6. Generate JSON and markdown reports with ESM-2 comparison

Patient variants (GRCh38, verified against myvariant.info and gnomAD v4):
    1. EZH2 V662A   - chr7:148810377 A>G  - VAF 59%, founder clone
    2. DNMT3A R882H  - chr2:25234373 C>T   - VAF 39%, pathogenic hotspot
    3. SETBP1 G870S  - chr18:44951948 G>A  - VAF 34%, likely pathogenic
    4. PTPN11 E76Q   - chr12:112450406 G>C - VAF 29%, pathogenic
    5. IDH2 R140Q    - chr15:90088702 C>T  - VAF 2%, pathogenic subclone

Data sources:
    - Ensembl REST API (GRCh38 reference genome)
    - NVIDIA NIM Evo 2 40B (https://build.nvidia.com)

Outputs:
    - mutation_profile/results/ai_research/evo2_variant_scoring/evo2_scores.json
    - mutation_profile/results/ai_research/evo2_variant_scoring/evo2_variant_report.md
    - mutation_profile/results/ai_research/evo2_variant_scoring/intermediate_*.json (per-variant)

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/evo2_variant_scoring.py

Runtime: ~2-5 minutes (API calls with rate limiting)
Dependencies: requests
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "ai_research" / "evo2_variant_scoring"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = RESULTS_DIR / "evo2_scores.json"
MD_OUTPUT = RESULTS_DIR / "evo2_variant_report.md"

# ── Configuration ──────────────────────────────────────────────────────────

CONTEXT_WINDOW = 5000  # 5 kb on each side = 10 kb total context
ENSEMBL_REST = "https://rest.ensembl.org"
NIM_ENDPOINT = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b"
NIM_ENDPOINT_ALT = "https://integrate.api.nvidia.com/v1/biology/arc/evo2-40b/loglikelihood"

MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds
RATE_LIMIT_SLEEP = 3   # seconds between API calls

# ── Patient variants (GRCh38, verified from gnomad_v4_query.py and pathogenicity_scores.py) ──

VARIANTS = [
    {
        "gene": "EZH2",
        "protein_change": "V662A",
        "hgvs_c": "c.1985T>C",
        "vaf": 0.59,
        "chrom": "7",
        "pos": 148810377,
        "ref": "A",
        "alt": "G",
        "transcript": "NM_004456.5",
        "note": "PRC2 chromatin remodeling. SET domain LoF. Founder clone. Novel unreported variant.",
    },
    {
        "gene": "DNMT3A",
        "protein_change": "R882H",
        "hgvs_c": "c.2645G>A",
        "vaf": 0.39,
        "chrom": "2",
        "pos": 25234373,
        "ref": "C",
        "alt": "T",
        "transcript": "NM_022552.5",
        "note": "DNA methyltransferase. Hotspot in AML/MDS. Disrupts catalytic domain.",
    },
    {
        "gene": "SETBP1",
        "protein_change": "G870S",
        "hgvs_c": "c.2608G>A",
        "vaf": 0.34,
        "chrom": "18",
        "pos": 44951948,
        "ref": "G",
        "alt": "A",
        "transcript": "NM_015559.3",
        "note": "SKI homology domain. Disrupts SCF-beta-TrCP degron motif.",
    },
    {
        "gene": "PTPN11",
        "protein_change": "E76Q",
        "hgvs_c": "c.226G>C",
        "vaf": 0.29,
        "chrom": "12",
        "pos": 112450406,
        "ref": "G",
        "alt": "C",
        "transcript": "NM_002834.5",
        "note": "N-SH2 domain. Disrupts autoinhibitory interface. GoF.",
    },
    {
        "gene": "IDH2",
        "protein_change": "R140Q",
        "hgvs_c": "c.419G>A",
        "vaf": 0.02,
        "chrom": "15",
        "pos": 90088702,
        "ref": "C",
        "alt": "T",
        "transcript": "NM_002168.4",
        "note": "Neomorphic GoF. Produces 2-HG oncometabolite. FDA-approved enasidenib.",
    },
]

# ── ESM-2 reference scores (for cross-model comparison) ──────────────────

ESM2_SCORES = {
    "SETBP1": {"llr": -9.804, "acmg": "PP3_Strong", "embedding_disruption": "L2=10.75, cosine=0.41"},
    "DNMT3A": {"llr": -8.383, "acmg": "PP3_Strong", "embedding_disruption": "L2=6.23, cosine=0.62"},
    "EZH2":   {"llr": -3.18,  "acmg": "PP3",        "embedding_disruption": "N/A (added post-analysis)"},
    "PTPN11": {"llr": -1.76,  "acmg": "PP3",        "embedding_disruption": "L2=3.41, cosine=0.78"},
    "IDH2":   {"llr": -1.20,  "acmg": "PP3",        "embedding_disruption": "L2=2.89, cosine=0.82"},
}


# ── Ensembl reference genome fetching ─────────────────────────────────────

def fetch_genomic_context(chrom: str, pos: int, context: int = CONTEXT_WINDOW) -> str | None:
    """Fetch genomic context from GRCh38 reference via Ensembl REST API.

    Retrieves context bp upstream and downstream of the variant position,
    returning a (2*context + 1) bp sequence with the variant position at center.
    """
    start = pos - context
    end = pos + context
    url = f"{ENSEMBL_REST}/sequence/region/human/{chrom}:{start}..{end}:1"
    headers = {"Content-Type": "text/plain"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 200:
                seq = resp.text.strip().upper()
                expected_len = end - start + 1
                if len(seq) != expected_len:
                    log.warning(
                        "Unexpected sequence length for %s:%d: got %d, expected %d",
                        chrom, pos, len(seq), expected_len,
                    )
                return seq
            if resp.status_code == 429:
                wait = INITIAL_BACKOFF * (2 ** attempt)
                log.warning("Ensembl rate limit (429), retrying in %.1fs ...", wait)
                time.sleep(wait)
                continue
            log.error("Ensembl API returned HTTP %d for %s:%d", resp.status_code, chrom, pos)
            return None
        except requests.RequestException as exc:
            wait = INITIAL_BACKOFF * (2 ** attempt)
            log.error("Ensembl request failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    log.error("All %d attempts failed for %s:%d", MAX_RETRIES, chrom, pos)
    return None


def verify_ref_allele(sequence: str, ref_allele: str, context: int = CONTEXT_WINDOW) -> bool:
    """Verify that the reference allele at the center of the fetched sequence matches expectations."""
    center_idx = context  # 0-based index of the variant position
    if center_idx >= len(sequence):
        log.error("Center index %d exceeds sequence length %d", center_idx, len(sequence))
        return False
    actual = sequence[center_idx]
    if actual != ref_allele:
        log.error(
            "Reference allele mismatch at center: expected %s, got %s (context window may be on wrong strand)",
            ref_allele, actual,
        )
        return False
    return True


def create_alt_sequence(ref_sequence: str, alt_allele: str, context: int = CONTEXT_WINDOW) -> str:
    """Create alternate sequence by substituting the variant allele at the center position."""
    center_idx = context
    return ref_sequence[:center_idx] + alt_allele + ref_sequence[center_idx + 1:]


# ── NVIDIA NIM Evo 2 API ─────────────────────────────────────────────────

def get_api_key() -> str | None:
    """Retrieve NVIDIA API key from environment."""
    return os.environ.get("NVIDIA_API_KEY")


def score_sequence(
    sequence: str,
    api_key: str,
    endpoint: str = NIM_ENDPOINT,
) -> dict[str, Any] | None:
    """Score a DNA sequence using NVIDIA NIM Evo 2 40B log-likelihood endpoint.

    Returns the API response containing per-nucleotide or aggregate log-likelihood scores.
    Implements exponential backoff for rate limits and transient failures.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "sequence": sequence,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 422:
                # Try alternative payload format
                alt_payload = {
                    "model": "arc/evo2-40b",
                    "input": sequence,
                }
                resp2 = requests.post(
                    endpoint,
                    headers=headers,
                    json=alt_payload,
                    timeout=120,
                )
                if resp2.status_code == 200:
                    return resp2.json()
                # Try the alternative endpoint
                if endpoint != NIM_ENDPOINT_ALT:
                    log.info("Primary endpoint returned 422, trying alternative endpoint ...")
                    resp3 = requests.post(
                        NIM_ENDPOINT_ALT,
                        headers=headers,
                        json=payload,
                        timeout=120,
                    )
                    if resp3.status_code == 200:
                        return resp3.json()
                    # Also try alt payload on alt endpoint
                    resp4 = requests.post(
                        NIM_ENDPOINT_ALT,
                        headers=headers,
                        json=alt_payload,
                        timeout=120,
                    )
                    if resp4.status_code == 200:
                        return resp4.json()
                log.error(
                    "Evo 2 API returned 422 (Unprocessable Entity). "
                    "Sequence length: %d. Response: %s",
                    len(sequence), resp.text[:500],
                )
                return None

            if resp.status_code in (429, 500, 502, 503, 504):
                wait = INITIAL_BACKOFF * (2 ** attempt)
                log.warning(
                    "Evo 2 API returned HTTP %d (attempt %d/%d), retrying in %.1fs ...",
                    resp.status_code, attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            if resp.status_code == 401:
                log.error("NVIDIA API key is invalid or expired (HTTP 401)")
                return None

            if resp.status_code == 403:
                log.error("NVIDIA API key lacks permission for Evo 2 (HTTP 403)")
                return None

            log.error(
                "Evo 2 API returned HTTP %d: %s",
                resp.status_code, resp.text[:500],
            )
            return None

        except requests.Timeout:
            wait = INITIAL_BACKOFF * (2 ** attempt)
            log.warning(
                "Evo 2 API timeout (attempt %d/%d), retrying in %.1fs ...",
                attempt + 1, MAX_RETRIES, wait,
            )
            time.sleep(wait)
        except requests.RequestException as exc:
            wait = INITIAL_BACKOFF * (2 ** attempt)
            log.error(
                "Evo 2 API request failed (attempt %d/%d): %s",
                attempt + 1, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

    log.error("All %d attempts failed for Evo 2 scoring", MAX_RETRIES)
    return None


def extract_log_likelihood(response: dict[str, Any]) -> float | None:
    """Extract aggregate log-likelihood score from Evo 2 API response.

    The API response format may vary; this function handles multiple possible
    response structures.
    """
    # Direct log-likelihood field
    if "loglikelihood" in response:
        return float(response["loglikelihood"])

    # Nested under results
    if "results" in response:
        results = response["results"]
        if isinstance(results, dict) and "loglikelihood" in results:
            return float(results["loglikelihood"])
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "loglikelihood" in first:
                return float(first["loglikelihood"])

    # Log-likelihood under data
    if "data" in response:
        data = response["data"]
        if isinstance(data, dict) and "loglikelihood" in data:
            return float(data["loglikelihood"])
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "loglikelihood" in first:
                return float(first["loglikelihood"])

    # Sum of per-nucleotide log-likelihoods
    if "loglikelihoods" in response:
        ll_values = response["loglikelihoods"]
        if isinstance(ll_values, list) and ll_values:
            return sum(float(x) for x in ll_values)

    if "per_token_loglikelihoods" in response:
        ll_values = response["per_token_loglikelihoods"]
        if isinstance(ll_values, list) and ll_values:
            return sum(float(x) for x in ll_values)

    # Score field (alternative naming)
    if "score" in response:
        return float(response["score"])

    # Mean log-likelihood
    if "mean_loglikelihood" in response:
        return float(response["mean_loglikelihood"])

    log.warning("Could not extract log-likelihood from response keys: %s", list(response.keys()))
    return None


# ── Scoring pipeline ──────────────────────────────────────────────────────

def score_variant(
    variant: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    """Score a single variant: fetch context, create ref/alt, call Evo 2, compute delta.

    Returns a result dict with scores and metadata.
    """
    gene = variant["gene"]
    chrom = variant["chrom"]
    pos = variant["pos"]
    ref = variant["ref"]
    alt = variant["alt"]
    result: dict[str, Any] = {
        "gene": gene,
        "protein_change": variant["protein_change"],
        "hgvs_c": variant["hgvs_c"],
        "vaf": variant["vaf"],
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "transcript": variant["transcript"],
        "note": variant["note"],
        "context_window_bp": CONTEXT_WINDOW * 2 + 1,
        "status": "pending",
    }

    # Step 1: Fetch genomic context
    log.info("  Fetching %d bp genomic context from GRCh38 ...", CONTEXT_WINDOW * 2 + 1)
    ref_sequence = fetch_genomic_context(chrom, pos, CONTEXT_WINDOW)
    if ref_sequence is None:
        result["status"] = "failed_sequence_fetch"
        result["error"] = f"Could not fetch genomic context for {chrom}:{pos}"
        return result

    result["sequence_length"] = len(ref_sequence)
    log.info("  Fetched %d bp sequence", len(ref_sequence))

    # Step 2: Verify reference allele
    if not verify_ref_allele(ref_sequence, ref, CONTEXT_WINDOW):
        # Try complement -- gene may be on minus strand
        complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
        ref_comp = complement.get(ref, "")
        alt_comp = complement.get(alt, "")
        if ref_comp and verify_ref_allele(ref_sequence, ref_comp, CONTEXT_WINDOW):
            log.info("  Gene on minus strand: using complement alleles %s>%s -> %s>%s", ref, alt, ref_comp, alt_comp)
            ref = ref_comp
            alt = alt_comp
            result["strand_note"] = f"Minus strand: original {variant['ref']}>{variant['alt']}, used complement {ref}>{alt}"
        else:
            actual_base = ref_sequence[CONTEXT_WINDOW] if CONTEXT_WINDOW < len(ref_sequence) else "?"
            result["status"] = "failed_ref_verification"
            result["error"] = (
                f"Reference allele mismatch: expected {ref} (or complement {ref_comp}), "
                f"got {actual_base} at {chrom}:{pos}"
            )
            log.error("  %s", result["error"])
            return result

    log.info("  Reference allele verified: %s at center position", ref)

    # Step 3: Create alternate sequence
    alt_sequence = create_alt_sequence(ref_sequence, alt, CONTEXT_WINDOW)
    result["ref_base_at_center"] = ref
    result["alt_base_at_center"] = alt

    # Step 4: Score reference sequence
    log.info("  Scoring reference sequence (%d bp) ...", len(ref_sequence))
    ref_response = score_sequence(ref_sequence, api_key)
    if ref_response is None:
        result["status"] = "failed_ref_scoring"
        result["error"] = "Evo 2 API failed for reference sequence"
        return result

    ref_ll = extract_log_likelihood(ref_response)
    if ref_ll is None:
        result["status"] = "failed_ref_extraction"
        result["error"] = "Could not extract log-likelihood from reference response"
        result["ref_response_keys"] = list(ref_response.keys())
        result["ref_response_sample"] = {k: str(v)[:200] for k, v in ref_response.items()}
        return result

    result["ref_loglikelihood"] = ref_ll
    log.info("  Reference log-likelihood: %.4f", ref_ll)

    # Rate limit between ref and alt scoring
    time.sleep(RATE_LIMIT_SLEEP)

    # Step 5: Score alternate sequence
    log.info("  Scoring alternate sequence (%d bp) ...", len(alt_sequence))
    alt_response = score_sequence(alt_sequence, api_key)
    if alt_response is None:
        result["status"] = "failed_alt_scoring"
        result["error"] = "Evo 2 API failed for alternate sequence"
        return result

    alt_ll = extract_log_likelihood(alt_response)
    if alt_ll is None:
        result["status"] = "failed_alt_extraction"
        result["error"] = "Could not extract log-likelihood from alternate response"
        result["alt_response_keys"] = list(alt_response.keys())
        result["alt_response_sample"] = {k: str(v)[:200] for k, v in alt_response.items()}
        return result

    result["alt_loglikelihood"] = alt_ll
    log.info("  Alternate log-likelihood: %.4f", alt_ll)

    # Step 6: Compute delta log-likelihood
    delta_ll = alt_ll - ref_ll
    result["delta_loglikelihood"] = delta_ll
    result["status"] = "success"

    # Interpretation: negative delta = alt is less likely = more damaging
    if delta_ll < -5.0:
        result["interpretation"] = "Strongly damaging"
    elif delta_ll < -2.0:
        result["interpretation"] = "Moderately damaging"
    elif delta_ll < -0.5:
        result["interpretation"] = "Mildly damaging"
    elif delta_ll < 0.5:
        result["interpretation"] = "Neutral / uncertain"
    else:
        result["interpretation"] = "Potentially beneficial / stabilizing"

    log.info(
        "  Delta log-likelihood: %.4f (%s)",
        delta_ll, result["interpretation"],
    )

    return result


def save_intermediate(result: dict[str, Any], index: int) -> None:
    """Save intermediate result for a single variant (crash recovery)."""
    path = RESULTS_DIR / f"intermediate_{index:02d}_{result['gene']}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    log.info("  Intermediate result saved: %s", path.name)


# ── Report generation ─────────────────────────────────────────────────────

def generate_markdown(results: list[dict[str, Any]]) -> str:
    """Generate markdown report with Evo 2 scores and ESM-2 comparison."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    lines = [
        "# Evo 2 (40B) DNA-Level Variant Pathogenicity Scoring",
        "",
        f"**Generated:** {ts}",
        "**Model:** Evo 2 40B (Arc Institute, via NVIDIA NIM)",
        "**Reference genome:** GRCh38 (Ensembl REST API)",
        f"**Context window:** {CONTEXT_WINDOW * 2 + 1:,} bp ({CONTEXT_WINDOW:,} bp flanking each side)",
        f"**Variants scored:** {len(successful)}/{len(results)} successful",
        "",
        "## Method",
        "",
        "Evo 2 is a 40-billion-parameter DNA foundation model trained on 9.3 trillion",
        "nucleotides from the OpenGenome2 dataset, spanning prokaryotic and eukaryotic",
        "genomes (Brixi et al., 2025). Unlike protein-level models (ESM-2), Evo 2",
        "operates on raw DNA sequences and can capture:",
        "",
        "- Codon usage bias and synonymous constraint",
        "- Local DNA secondary structure effects",
        "- Regulatory element disruption (promoters, enhancers, silencers)",
        "- Splicing signal perturbation (even for coding variants)",
        "- Transcription factor binding site disruption",
        "- Nucleosome positioning signals",
        "",
        "**Scoring approach:** For each variant, 10 kb of genomic context is fetched from",
        "GRCh38. Reference and alternate sequences are scored for log-likelihood by Evo 2.",
        "The delta log-likelihood (alt - ref) quantifies the variant's impact on genomic",
        "sequence plausibility: negative values indicate the variant makes the sequence",
        "less likely under the model, suggesting a damaging effect.",
        "",
    ]

    # Summary table
    if successful:
        lines += [
            "## Results Summary",
            "",
            "| Rank | Gene | Variant | Delta LL | Interpretation | Patient VAF |",
            "|------|------|---------|----------|----------------|-------------|",
        ]

        ranked = sorted(successful, key=lambda r: r["delta_loglikelihood"])
        for rank, r in enumerate(ranked, 1):
            lines.append(
                f"| {rank} | {r['gene']} | {r['protein_change']} ({r['hgvs_c']}) "
                f"| {r['delta_loglikelihood']:.4f} | {r['interpretation']} | {r['vaf']:.0%} |"
            )
        lines.append("")

    # Detailed results
    lines += ["## Detailed Results", ""]

    for r in results:
        gene = r["gene"]
        lines += [
            f"### {gene} {r['protein_change']} ({r['hgvs_c']})",
            "",
            f"- **Genomic position (GRCh38):** chr{r['chrom']}:{r['pos']:,}",
            f"- **Allele change:** {r['ref']}>{r['alt']}",
            f"- **Transcript:** {r['transcript']}",
            f"- **Patient VAF:** {r['vaf']:.0%}",
            f"- **Functional note:** {r['note']}",
            "",
        ]

        if r["status"] == "success":
            lines += [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Reference log-likelihood | {r['ref_loglikelihood']:.4f} |",
                f"| Alternate log-likelihood | {r['alt_loglikelihood']:.4f} |",
                f"| **Delta log-likelihood** | **{r['delta_loglikelihood']:.4f}** |",
                f"| Interpretation | {r['interpretation']} |",
                f"| Context window | {r.get('sequence_length', 'N/A'):,} bp |",
                "",
            ]
            if r.get("strand_note"):
                lines.append(f"*Note: {r['strand_note']}*")
                lines.append("")
        else:
            lines += [
                f"**Status:** {r['status']}",
                f"**Error:** {r.get('error', 'Unknown error')}",
                "",
            ]

        lines += ["---", ""]

    # ESM-2 cross-model comparison
    lines += [
        "## Cross-Model Comparison: Evo 2 (DNA) vs ESM-2 (Protein)",
        "",
        "This comparison highlights differences between DNA-level and protein-level",
        "variant effect prediction. Discordance between the two models may indicate",
        "variants where the DNA context carries additional pathogenicity signals beyond",
        "the protein-level amino acid substitution effect.",
        "",
        "| Gene | Variant | Evo 2 Delta LL | Evo 2 Rank | ESM-2 LLR | ESM-2 ACMG | ESM-2 Rank |",
        "|------|---------|----------------|------------|-----------|------------|------------|",
    ]

    esm2_ranked = sorted(ESM2_SCORES.items(), key=lambda x: x[1]["llr"])
    esm2_rank_map = {gene: rank + 1 for rank, (gene, _) in enumerate(esm2_ranked)}

    if successful:
        evo2_ranked = sorted(successful, key=lambda r: r["delta_loglikelihood"])
        evo2_rank_map = {r["gene"]: rank + 1 for rank, r in enumerate(evo2_ranked)}
    else:
        evo2_rank_map = {}

    for gene, esm2 in sorted(ESM2_SCORES.items(), key=lambda x: x[1]["llr"]):
        evo2_result = next((r for r in results if r["gene"] == gene), None)
        evo2_dll = f"{evo2_result['delta_loglikelihood']:.4f}" if evo2_result and evo2_result["status"] == "success" else "N/A"
        evo2_rank = str(evo2_rank_map.get(gene, "N/A"))
        esm2_rank = str(esm2_rank_map[gene])
        protein_change = next((v["protein_change"] for v in VARIANTS if v["gene"] == gene), "?")
        lines.append(
            f"| {gene} | {protein_change} | {evo2_dll} | {evo2_rank} | {esm2['llr']:.3f} | {esm2['acmg']} | {esm2_rank} |"
        )

    lines += [
        "",
        "### Interpretation",
        "",
        "**ESM-2 protein-level ranking (most to least damaging):**",
        "",
    ]
    for rank, (gene, scores) in enumerate(esm2_ranked, 1):
        protein_change = next((v["protein_change"] for v in VARIANTS if v["gene"] == gene), "?")
        lines.append(f"{rank}. {gene} {protein_change}: LLR={scores['llr']:.3f} ({scores['acmg']})")
    lines.append("")

    if successful:
        lines += [
            "**Evo 2 DNA-level ranking (most to least damaging):**",
            "",
        ]
        for rank, r in enumerate(evo2_ranked, 1):
            lines.append(
                f"{rank}. {r['gene']} {r['protein_change']}: "
                f"delta_LL={r['delta_loglikelihood']:.4f} ({r['interpretation']})"
            )
        lines.append("")

        # Rank concordance analysis
        concordant = 0
        discordant_pairs = []
        for gene in evo2_rank_map:
            if gene in esm2_rank_map:
                if evo2_rank_map[gene] == esm2_rank_map[gene]:
                    concordant += 1
                else:
                    discordant_pairs.append({
                        "gene": gene,
                        "evo2_rank": evo2_rank_map[gene],
                        "esm2_rank": esm2_rank_map[gene],
                        "rank_shift": evo2_rank_map[gene] - esm2_rank_map[gene],
                    })

        total_ranked = min(len(evo2_rank_map), len(esm2_rank_map))
        lines += [
            "**Rank concordance:**",
            f"- {concordant}/{total_ranked} variants have identical ranking between models",
            "",
        ]

        if discordant_pairs:
            lines += ["**Rank discordances (Evo 2 rank - ESM-2 rank):**", ""]
            for d in sorted(discordant_pairs, key=lambda x: abs(x["rank_shift"]), reverse=True):
                shift_dir = "more damaging" if d["rank_shift"] < 0 else "less damaging"
                protein_change = next((v["protein_change"] for v in VARIANTS if v["gene"] == d["gene"]), "?")
                lines.append(
                    f"- {d['gene']} {protein_change}: Evo 2 rank {d['evo2_rank']} vs ESM-2 rank {d['esm2_rank']} "
                    f"(shift {d['rank_shift']:+d}, {shift_dir} at DNA level)"
                )
            lines.append("")

            lines += [
                "Rank discordance may indicate variants where:",
                "- DNA-level regulatory context contributes to pathogenicity (higher Evo 2 damage)",
                "- Protein-level structural disruption dominates the effect (higher ESM-2 damage)",
                "- Codon usage, splicing signals, or local DNA structure modulate variant impact",
                "",
            ]

    # Failed variants
    if failed:
        lines += [
            "## Failed Variants",
            "",
        ]
        for r in failed:
            lines.append(f"- **{r['gene']} {r['protein_change']}:** {r['status']} -- {r.get('error', 'Unknown')}")
        lines.append("")

    # Methodology notes
    lines += [
        "## Methodology Notes",
        "",
        "1. **Genomic context:** 10 kb of flanking sequence was fetched from GRCh38 via Ensembl REST API",
        "2. **Strand handling:** Reference alleles are verified against GRCh38; if the gene is on the",
        "   minus strand, complement alleles are used automatically",
        "3. **Scoring:** Both reference and alternate sequences are scored independently by Evo 2 40B",
        "4. **Delta calculation:** delta_LL = alt_LL - ref_LL (negative = more damaging)",
        "5. **Context window:** 10,001 bp (5 kb upstream + variant + 5 kb downstream), well within",
        f"   Evo 2's maximum context length of 131,072 bp",
        "",
        "## References",
        "",
        "- Brixi G, Durairaj J, et al. Genome modeling and design across all domains of life",
        "  with Evo 2. *bioRxiv* (2025). doi:10.1101/2025.02.18.638918",
        "- Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a",
        "  language model. *Science* 379, 1123-1130 (2023). [ESM-2 reference]",
        "",
        "---",
        "",
        "*Analysis performed using Evo 2 40B via NVIDIA NIM API with GRCh38 reference coordinates.*",
        f"*Genomic context: {CONTEXT_WINDOW * 2 + 1:,} bp per variant.*",
        "*Coordinates verified against gnomAD v4 and myvariant.info in prior analyses.*",
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("Evo 2 (40B) DNA-Level Variant Pathogenicity Scoring")
    log.info("=" * 70)

    # Check API key
    api_key = get_api_key()
    if not api_key:
        log.error("")
        log.error("NVIDIA_API_KEY environment variable is not set.")
        log.error("")
        log.error("To get an API key:")
        log.error("  1. Go to https://build.nvidia.com")
        log.error("  2. Create a free account (1,000 free API credits)")
        log.error("  3. Navigate to the Evo 2 model page:")
        log.error("     https://build.nvidia.com/arc/evo2-40b")
        log.error("  4. Click 'Get API Key' and copy the key")
        log.error("  5. Set it in your environment:")
        log.error("     export NVIDIA_API_KEY='nvapi-...'")
        log.error("  6. Or add to ~/.secrets.env:")
        log.error("     NVIDIA_API_KEY=nvapi-...")
        log.error("")
        sys.exit(1)

    log.info("NVIDIA API key: %s...%s (%d chars)", api_key[:8], api_key[-4:], len(api_key))
    log.info("Scoring %d patient variants ...", len(VARIANTS))
    log.info("Context window: %d bp (%d kb flanking each side)", CONTEXT_WINDOW * 2 + 1, CONTEXT_WINDOW // 1000)
    log.info("")

    results: list[dict[str, Any]] = []
    start_time = time.monotonic()

    for i, variant in enumerate(VARIANTS):
        gene = variant["gene"]
        log.info("[%d/%d] %s %s (%s:%d %s>%s)",
                 i + 1, len(VARIANTS), gene, variant["protein_change"],
                 variant["chrom"], variant["pos"], variant["ref"], variant["alt"])

        result = score_variant(variant, api_key)
        results.append(result)

        # Save intermediate result after each variant
        save_intermediate(result, i)

        # Rate limit between variants
        if i < len(VARIANTS) - 1:
            log.info("  Rate limiting: sleeping %ds ...", RATE_LIMIT_SLEEP)
            time.sleep(RATE_LIMIT_SLEEP)

    elapsed = time.monotonic() - start_time

    # Save JSON
    output = {
        "metadata": {
            "script": "evo2_variant_scoring.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "Evo 2 40B (Arc Institute)",
            "api_endpoint": NIM_ENDPOINT,
            "genome_build": "GRCh38",
            "context_window_bp": CONTEXT_WINDOW * 2 + 1,
            "n_variants": len(VARIANTS),
            "n_successful": sum(1 for r in results if r["status"] == "success"),
            "elapsed_seconds": round(elapsed, 1),
        },
        "results": results,
        "esm2_reference_scores": ESM2_SCORES,
    }
    JSON_OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    log.info("JSON results saved to %s", JSON_OUTPUT)

    # Save markdown report
    report = generate_markdown(results)
    MD_OUTPUT.write_text(report)
    log.info("Markdown report saved to %s", MD_OUTPUT)

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY (%.1f seconds elapsed)", elapsed)
    log.info("=" * 70)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    if successful:
        ranked = sorted(successful, key=lambda r: r["delta_loglikelihood"])
        log.info("Evo 2 DNA-level rankings (most to least damaging):")
        for rank, r in enumerate(ranked, 1):
            log.info(
                "  %d. %s %s: delta_LL=%.4f (%s)",
                rank, r["gene"], r["protein_change"],
                r["delta_loglikelihood"], r["interpretation"],
            )

    if failed:
        log.info("")
        log.info("Failed variants:")
        for r in failed:
            log.info("  - %s %s: %s", r["gene"], r["protein_change"], r["status"])

    log.info("")
    log.info("Results: %s", JSON_OUTPUT)
    log.info("Report:  %s", MD_OUTPUT)


if __name__ == "__main__":
    main()
