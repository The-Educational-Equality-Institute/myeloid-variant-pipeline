#!/usr/bin/env python3
"""
esm2_benchmark_scoring.py -- ESM-2 masked marginal scoring for 154 benchmark variants.

Reads benchmark_profiles.json (20 SETBP1-positive myeloid patients, 154 total mutations),
filters to missense variants, and scores each using ESM-2 (facebook/esm2_t33_650M_UR50D).

Method:
    For each missense variant:
    1. Fetch the UniProt canonical sequence for the gene
    2. Parse HGVSp to extract wildtype AA, position, mutant AA
    3. Mask the target position, run ESM-2 inference
    4. Compute LLR = log P(mutant) - log P(wildtype)

ACMG PP3 interpretation thresholds (Brandes et al. 2023):
    LLR < -7.0  : PP3_Strong
    LLR < -3.5  : PP3_Moderate
    LLR < -1.5  : PP3_Supporting
    LLR >= -1.5 : Benign/VUS

Outputs:
    mutation_profile/results/ai_research/benchmark/esm2_benchmark_scores.json

Runtime: ~2-3 minutes on GPU (86 missense variants)
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "results" / "ai_research" / "benchmark"
DEFAULT_INPUT = BENCHMARK_DIR / "benchmark_profiles.json"
DEFAULT_OUTPUT = BENCHMARK_DIR / "esm2_benchmark_scores.json"

# ---------------------------------------------------------------------------
# UniProt accessions for all 34 target genes
# ---------------------------------------------------------------------------

UNIPROT_MAP = {
    "ASXL1": "Q8IXJ9",
    "TET2": "Q6N021",
    "SRSF2": "Q01130",
    "SF3B1": "O75533",
    "RUNX1": "Q01196",
    "TP53": "P04637",
    "FLT3": "P36888",
    "NPM1": "P06748",
    "NRAS": "P01111",
    "KRAS": "P01116",
    "CBL": "P22681",
    "EZH2": "Q15910",
    "U2AF1": "Q01081",
    "STAG2": "Q8N3U4",
    "BCOR": "Q6W2J9",
    "BCORL1": "Q5H9F3",
    "DDX41": "Q9UJV9",
    "DNMT3A": "Q9Y6K1",
    "IDH1": "O75874",
    "IDH2": "P48735",
    "PTPN11": "Q06124",
    "JAK2": "O60674",
    "CALR": "P27797",
    "MPL": "P40238",
    "PHF6": "Q8IWS0",
    "WT1": "P19544",
    "CEBPA": "P49715",
    "GATA2": "P23769",
    "ZRSR2": "Q15696",
    "RAD21": "O60216",
    "SMC1A": "Q14683",
    "SMC3": "Q9UQE7",
    "SETBP1": "Q9Y6X0",
    "CSF3R": "Q99062",
}

# Three-letter to one-letter amino acid conversion
AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

# ACMG PP3 thresholds
ACMG_THRESHOLDS = [
    (-7.0, "PP3_Strong"),
    (-3.5, "PP3_Moderate"),
    (-1.5, "PP3_Supporting"),
]


def acmg_classify(llr: float) -> str:
    """Classify LLR into ACMG PP3 tier."""
    for threshold, label in ACMG_THRESHOLDS:
        if llr < threshold:
            return label
    return "Benign/VUS"


# ---------------------------------------------------------------------------
# HGVSp parsing
# ---------------------------------------------------------------------------

def parse_hgvsp(hgvsp: str):
    """
    Parse HGVSp string to extract wildtype AA, position, mutant AA.

    Handles formats:
        "G870S"           -> (G, 870, S)
        "p.G870S"         -> (G, 870, S)
        "p.Val617Phe"     -> (V, 617, F)
        "Val617Phe"       -> (V, 617, F)

    Returns (wt_aa, position, mut_aa) or None if not parseable as missense.
    """
    s = hgvsp.strip()
    if s.startswith("p."):
        s = s[2:]

    # Try three-letter format first: e.g. Val617Phe
    m = re.match(r"^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$", s)
    if m:
        wt3, pos, mut3 = m.group(1), int(m.group(2)), m.group(3)
        wt1 = AA3_TO_1.get(wt3)
        mut1 = AA3_TO_1.get(mut3)
        if wt1 and mut1:
            return wt1, pos, mut1

    # Try one-letter format: e.g. G870S
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", s)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)

    return None


# ---------------------------------------------------------------------------
# UniProt sequence fetching with caching
# ---------------------------------------------------------------------------

_seq_cache: dict[str, str] = {}


def fetch_uniprot_sequence(uniprot_id: str) -> str:
    """Fetch canonical protein sequence from UniProt REST API (cached)."""
    if uniprot_id in _seq_cache:
        return _seq_cache[uniprot_id]

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    seq = "".join(lines[1:])  # skip FASTA header
    _seq_cache[uniprot_id] = seq
    return seq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ESM-2 benchmark scoring")
    parser.add_argument("--input", type=str, default="", help="Input profiles JSON")
    parser.add_argument("--output", type=str, default="", help="Output scores JSON")
    args = parser.parse_args()

    INPUT_FILE = Path(args.input) if args.input else DEFAULT_INPUT
    OUTPUT_FILE = Path(args.output) if args.output else DEFAULT_OUTPUT

    print("=" * 70)
    print(f"ESM-2 Benchmark Scoring: {INPUT_FILE.name}")
    print("=" * 70)

    # Load benchmark data
    with open(INPUT_FILE) as f:
        data = json.load(f)

    # Extract missense variants (deduplicated by gene+hgvsp for scoring,
    # but we keep all occurrences for output mapping)
    all_missense = []
    unique_variants = {}  # key: (gene, hgvsp) -> parsed info

    for profile in data["profiles"]:
        patient_id = profile["patient_id"]
        for mut in profile["mutations"]:
            if mut["variant_classification"] != "Missense_Mutation":
                continue
            gene = mut["gene"]
            hgvsp = mut["hgvsp"]
            parsed = parse_hgvsp(hgvsp)
            if parsed is None:
                print(f"  WARNING: Could not parse HGVSp '{hgvsp}' for {gene}, skipping")
                continue
            wt_aa, pos, mut_aa = parsed
            key = (gene, hgvsp)
            all_missense.append({
                "patient_id": patient_id,
                "gene": gene,
                "hgvsp": hgvsp,
                "wt_aa": wt_aa,
                "position": pos,
                "mut_aa": mut_aa,
                "t_vaf": mut.get("t_vaf"),
            })
            if key not in unique_variants:
                unique_variants[key] = (gene, hgvsp, wt_aa, pos, mut_aa)

    print(f"\nTotal missense mutations: {len(all_missense)}")
    print(f"Unique variants to score: {len(unique_variants)}")

    # Fetch all required UniProt sequences
    genes_needed = sorted(set(v[0] for v in unique_variants.values()))
    print(f"\nFetching UniProt sequences for {len(genes_needed)} genes...")
    sequences = {}
    for gene in genes_needed:
        uid = UNIPROT_MAP.get(gene)
        if not uid:
            print(f"  WARNING: No UniProt ID for {gene}, skipping")
            continue
        try:
            seq = fetch_uniprot_sequence(uid)
            sequences[gene] = seq
            print(f"  {gene} ({uid}): {len(seq)} residues")
        except Exception as e:
            print(f"  ERROR fetching {gene} ({uid}): {e}")

    # Load ESM-2 model
    print(f"\nLoading ESM-2 model (facebook/esm2_t33_650M_UR50D)...")
    t0 = time.time()

    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    print(f"  Model loaded on {device} in {time.time() - t0:.1f}s")

    # Score each unique variant
    print(f"\nScoring {len(unique_variants)} unique variants...")
    scores = {}  # key: (gene, hgvsp) -> llr
    scored = 0
    errors = 0
    t_start = time.time()

    sorted_variants = sorted(unique_variants.values(), key=lambda x: (x[0], x[3]))

    for gene, hgvsp, wt_aa, pos, mut_aa in sorted_variants:
        key = (gene, hgvsp)

        if gene not in sequences:
            print(f"  [{scored+errors+1}/{len(sorted_variants)}] SKIP {gene} {hgvsp} - no sequence")
            errors += 1
            continue

        seq = sequences[gene]

        # Validate position and wildtype AA
        if pos < 1 or pos > len(seq):
            print(f"  [{scored+errors+1}/{len(sorted_variants)}] SKIP {gene} {hgvsp} - pos {pos} out of range (len={len(seq)})")
            errors += 1
            continue

        actual_aa = seq[pos - 1]
        if actual_aa != wt_aa:
            print(f"  [{scored+errors+1}/{len(sorted_variants)}] WARN {gene} {hgvsp} - expected {wt_aa} at pos {pos}, got {actual_aa}")
            # Try to proceed anyway -- the clinical numbering might differ from UniProt

        # ESM-2 masked marginal scoring
        try:
            batch_data = [("protein", seq)]
            _, _, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)

            # Mask the target position (tokens has BOS at index 0, so position maps to index=pos)
            masked = tokens.clone()
            masked[0, pos] = alphabet.mask_idx

            with torch.no_grad():
                logits = model(masked)["logits"]

            log_probs = torch.log_softmax(logits[0, pos], dim=-1)
            llr = (log_probs[alphabet.get_idx(mut_aa)] - log_probs[alphabet.get_idx(wt_aa)]).item()

            scores[key] = llr
            scored += 1
            classification = acmg_classify(llr)

            print(f"  [{scored+errors}/{len(sorted_variants)}] {gene} {hgvsp}: LLR={llr:.3f} ({classification})")

        except Exception as e:
            print(f"  [{scored+errors+1}/{len(sorted_variants)}] ERROR {gene} {hgvsp}: {e}")
            errors += 1

        # Save intermediate results every 20 variants
        if (scored + errors) % 20 == 0:
            _save_intermediate(all_missense, scores, scored, errors, t_start)

    elapsed = time.time() - t_start
    print(f"\nScoring complete: {scored} scored, {errors} errors in {elapsed:.1f}s")

    # Build final output
    output = _build_output(all_missense, scores, data, scored, errors, elapsed)

    # Save final results
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Print summary table
    _print_summary(output)


def _save_intermediate(all_missense, scores, scored, errors, t_start):
    """Save intermediate results checkpoint."""
    elapsed = time.time() - t_start
    output = _build_output(all_missense, scores, None, scored, errors, elapsed, partial=True)
    tmp_file = BENCHMARK_DIR / "esm2_benchmark_scores_partial.json"
    with open(tmp_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"    [checkpoint: {scored} scored, {errors} errors, {elapsed:.1f}s elapsed]")


def _build_output(all_missense, scores, data, scored, errors, elapsed, partial=False):
    """Build the output JSON structure."""
    variants_out = []
    for entry in all_missense:
        key = (entry["gene"], entry["hgvsp"])
        llr = scores.get(key)
        rec = {
            "patient_id": entry["patient_id"],
            "gene": entry["gene"],
            "hgvsp": entry["hgvsp"],
            "wt_aa": entry["wt_aa"],
            "mut_aa": entry["mut_aa"],
            "position": entry["position"],
            "t_vaf": entry["t_vaf"],
            "esm2_llr": round(llr, 3) if llr is not None else None,
            "acmg_pp3": acmg_classify(llr) if llr is not None else None,
        }
        variants_out.append(rec)

    # Deduplicated variant list (unique gene+hgvsp)
    seen = set()
    unique_out = []
    for v in variants_out:
        key = (v["gene"], v["hgvsp"])
        if key not in seen and v["esm2_llr"] is not None:
            seen.add(key)
            unique_out.append({
                "gene": v["gene"],
                "hgvsp": v["hgvsp"],
                "wt_aa": v["wt_aa"],
                "mut_aa": v["mut_aa"],
                "position": v["position"],
                "esm2_llr": v["esm2_llr"],
                "acmg_pp3": v["acmg_pp3"],
            })
    unique_out.sort(key=lambda x: x["esm2_llr"])

    # Classification distribution
    class_counts = {}
    for v in unique_out:
        c = v["acmg_pp3"]
        class_counts[c] = class_counts.get(c, 0) + 1

    output = {
        "metadata": {
            "description": "ESM-2 masked marginal LLR scores for benchmark missense variants",
            "model": "facebook/esm2_t33_650M_UR50D",
            "method": "Masked marginal: LLR = log P(mut|context) - log P(wt|context)",
            "source": "benchmark_profiles.json (20 SETBP1+ myeloid patients, GENIE v19.0)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_mutations_in_benchmark": sum(len(p["mutations"]) for p in (data["profiles"] if data else [])) if data else None,
            "missense_mutations": len(all_missense),
            "unique_variants_scored": scored,
            "scoring_errors": errors,
            "runtime_seconds": round(elapsed, 1),
            "partial": partial,
            "acmg_thresholds": {
                "PP3_Strong": "LLR < -7.0",
                "PP3_Moderate": "-7.0 <= LLR < -3.5",
                "PP3_Supporting": "-3.5 <= LLR < -1.5",
                "Benign/VUS": "LLR >= -1.5",
            },
            "classification_distribution": class_counts,
        },
        "variants": unique_out,
        "per_patient_variants": variants_out,
    }
    return output


def _print_summary(output):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("SUMMARY: ESM-2 Benchmark Scoring Results")
    print("=" * 70)

    variants = output["variants"]
    print(f"\n{'Gene':<10} {'Variant':<12} {'LLR':>8} {'Classification':<16}")
    print("-" * 50)
    for v in variants:
        print(f"{v['gene']:<10} {v['hgvsp']:<12} {v['esm2_llr']:>8.3f} {v['acmg_pp3']:<16}")

    print(f"\n{'Classification':<20} {'Count':>5}")
    print("-" * 28)
    dist = output["metadata"]["classification_distribution"]
    for cls in ["PP3_Strong", "PP3_Moderate", "PP3_Supporting", "Benign/VUS"]:
        if cls in dist:
            print(f"{cls:<20} {dist[cls]:>5}")

    total = sum(dist.values())
    pathogenic = sum(v for k, v in dist.items() if k != "Benign/VUS")
    print(f"\nTotal unique variants scored: {total}")
    print(f"Computationally pathogenic (PP3_Supporting or higher): {pathogenic} ({100*pathogenic/total:.1f}%)")


if __name__ == "__main__":
    main()
