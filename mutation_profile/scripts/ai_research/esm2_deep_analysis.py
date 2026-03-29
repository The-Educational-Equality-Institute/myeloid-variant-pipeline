#!/usr/bin/env python3
"""
esm2_deep_analysis.py -- ESM-2 deep variant analysis with attention and embedding maps.

Goes beyond basic masked-marginal scoring to perform:
  1. Full positional scanning (+/-50 residues): log-likelihood landscape of all 20 AAs
  2. Attention analysis: which residues attend to the mutation site
  3. Embedding comparison: cosine similarity between wildtype and mutant representations
  4. Context-dependent scoring: epistatic interactions between co-occurring mutations

Patient mutations:
    DNMT3A R882H  (UniProt Q9Y6K1)
    IDH2 R140Q    (UniProt P48735)
    SETBP1 G870S  (UniProt Q9Y6X0)
    PTPN11 E76Q   (UniProt Q06124)

Inputs:
    - UniProt REST API (remote, fetches canonical protein sequences)
    - HuggingFace model: facebook/esm2_t33_650M_UR50D (cached locally)

Outputs:
    - mutation_profile/results/ai_research/esm2_deep/positional_scanning_*.json
    - mutation_profile/results/ai_research/esm2_deep/attention_analysis_*.json
    - mutation_profile/results/ai_research/esm2_deep/embedding_comparison.json
    - mutation_profile/results/ai_research/esm2_deep/epistatic_interactions.json
    - mutation_profile/results/ai_research/esm2_deep/esm2_deep_summary.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/esm2_deep_analysis.py

Runtime: ~3-5 minutes (GPU), ~15-20 minutes (CPU)
Dependencies: torch, transformers, numpy, requests
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
WINDOW_SIZE = 200        # residues on each side for scoring windows
SCAN_RADIUS = 50         # residues on each side for positional scanning
DEVICE = None            # set in main()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "ai_research" / "esm2_deep"

# The 4 patient mutations (EZH2 excluded -- only the 4 patient-confirmed ones)
VARIANTS = [
    {
        "gene": "DNMT3A", "variant": "R882H", "uniprot_id": "Q9Y6K1",
        "uniprot_pos": 882, "ref_aa": "R", "alt_aa": "H",
        "note": "Hotspot in AML/MDS, disrupts methyltransferase catalytic domain",
    },
    {
        "gene": "IDH2", "variant": "R140Q", "uniprot_id": "P48735",
        "uniprot_pos": 140, "ref_aa": "R", "alt_aa": "Q",
        "note": "Gain-of-function neomorphic, produces 2-HG oncometabolite",
    },
    {
        "gene": "SETBP1", "variant": "G870S", "uniprot_id": "Q9Y6X0",
        "uniprot_pos": 870, "ref_aa": "G", "alt_aa": "S",
        "note": "SKI homology domain, disrupts SCF-beta-TrCP degron motif",
    },
    {
        "gene": "PTPN11", "variant": "E76Q", "uniprot_id": "Q06124",
        "uniprot_pos": 76, "ref_aa": "E", "alt_aa": "Q",
        "note": "N-SH2 domain, disrupts autoinhibitory interface with PTP domain",
    },
]

# Standard amino acids (1-letter codes)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Sequence retrieval
# ---------------------------------------------------------------------------

_seq_cache: dict[str, str] = {}

def fetch_uniprot_sequence(uniprot_id: str) -> str:
    """Fetch canonical protein sequence from UniProt, with caching."""
    if uniprot_id in _seq_cache:
        return _seq_cache[uniprot_id]
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    print(f"    Fetching {uniprot_id} from UniProt...", end=" ", flush=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    seq = "".join(lines[1:])
    print(f"OK ({len(seq)} residues)")
    _seq_cache[uniprot_id] = seq
    return seq


def extract_window(sequence: str, pos_1based: int, window: int = WINDOW_SIZE) -> tuple[str, int]:
    """Extract subsequence window. Returns (subseq, pos_in_subseq_0based)."""
    idx = pos_1based - 1
    start = max(0, idx - window)
    end = min(len(sequence), idx + window + 1)
    return sequence[start:end], idx - start


# ---------------------------------------------------------------------------
# 1. POSITIONAL SCANNING
# ---------------------------------------------------------------------------

def positional_scan(
    model: EsmForMaskedLM,
    tokenizer: AutoTokenizer,
    full_sequence: str,
    mutation_pos_1based: int,
    ref_aa: str,
    alt_aa: str,
    scan_radius: int = SCAN_RADIUS,
) -> dict:
    """
    For each position in [mut_pos - scan_radius, mut_pos + scan_radius],
    mask it, predict log-probs for all 20 AAs, compute conservation metrics.
    """
    print("    Running positional scan...", flush=True)
    idx = mutation_pos_1based - 1  # 0-based in full sequence

    # Define scan range in full-sequence coords
    scan_start = max(0, idx - scan_radius)
    scan_end = min(len(full_sequence), idx + scan_radius + 1)

    # Build a window large enough for context (use WINDOW_SIZE around center of scan)
    center = (scan_start + scan_end) // 2
    win_start = max(0, center - WINDOW_SIZE)
    win_end = min(len(full_sequence), center + WINDOW_SIZE + 1)
    window_seq = full_sequence[win_start:win_end]

    # Get all AA token IDs
    aa_token_ids = [tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS]

    scan_results = []
    total_positions = scan_end - scan_start

    for abs_pos in range(scan_start, scan_end):
        rel_pos = abs_pos - win_start  # position in window
        wt_aa = window_seq[rel_pos]

        # Skip non-standard residues
        if wt_aa not in AMINO_ACIDS:
            continue

        # Mask this position
        masked = list(window_seq)
        masked[rel_pos] = tokenizer.mask_token
        masked_str = "".join(masked)

        inputs = tokenizer(masked_str, return_tensors="pt", padding=False, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits[0]

        token_pos = rel_pos + 1  # +1 for CLS
        log_probs = F.log_softmax(logits[token_pos], dim=-1)

        # Extract log-probs for all 20 AAs
        aa_log_probs = {aa: log_probs[tid].item() for aa, tid in zip(AMINO_ACIDS, aa_token_ids)}

        wt_log_prob = aa_log_probs.get(wt_aa, float("-inf"))
        wt_rank = sorted(aa_log_probs.values(), reverse=True).index(wt_log_prob) + 1

        # Entropy at this position (measure of conservation)
        probs = torch.exp(log_probs[aa_token_ids])
        probs = probs / probs.sum()  # renormalize over 20 AAs
        entropy = -(probs * torch.log2(probs + 1e-10)).sum().item()

        entry = {
            "pos_1based": abs_pos + 1,
            "wt_aa": wt_aa,
            "wt_log_prob": round(wt_log_prob, 4),
            "wt_rank": wt_rank,
            "entropy": round(entropy, 4),
            "is_mutation_site": (abs_pos == idx),
            "aa_log_probs": {aa: round(v, 4) for aa, v in aa_log_probs.items()},
        }

        # If this is the mutation site, add mutation-specific info
        if abs_pos == idx:
            alt_log_prob = aa_log_probs.get(alt_aa, float("-inf"))
            alt_rank = sorted(aa_log_probs.values(), reverse=True).index(alt_log_prob) + 1
            entry["alt_aa"] = alt_aa
            entry["alt_log_prob"] = round(alt_log_prob, 4)
            entry["alt_rank"] = alt_rank
            entry["score"] = round(alt_log_prob - wt_log_prob, 4)

        scan_results.append(entry)

        done = abs_pos - scan_start + 1
        if done % 20 == 0 or done == total_positions:
            print(f"      {done}/{total_positions} positions scanned", flush=True)

    # Summarize conservation
    entropies = [r["entropy"] for r in scan_results]
    wt_probs = [r["wt_log_prob"] for r in scan_results]

    # Most conserved positions (lowest entropy)
    sorted_by_entropy = sorted(scan_results, key=lambda x: x["entropy"])
    most_conserved = sorted_by_entropy[:10]

    # Least conserved (highest entropy)
    least_conserved = sorted_by_entropy[-5:]

    mut_site = [r for r in scan_results if r["is_mutation_site"]][0]

    return {
        "scan_range": f"{scan_start + 1}-{scan_end}",
        "positions_scanned": len(scan_results),
        "mutation_site": mut_site,
        "most_conserved_positions": most_conserved,
        "least_conserved_positions": least_conserved,
        "entropy_stats": {
            "mean": round(np.mean(entropies), 4),
            "std": round(np.std(entropies), 4),
            "min": round(np.min(entropies), 4),
            "max": round(np.max(entropies), 4),
            "mutation_site_entropy": mut_site["entropy"],
        },
        "all_positions": scan_results,
    }


# ---------------------------------------------------------------------------
# 2. ATTENTION ANALYSIS
# ---------------------------------------------------------------------------

def attention_analysis(
    model: EsmForMaskedLM,
    tokenizer: AutoTokenizer,
    full_sequence: str,
    mutation_pos_1based: int,
    ref_aa: str,
) -> dict:
    """
    Extract attention weights and identify residues that attend to the mutation site.
    Uses the wildtype sequence (unmasked) to capture natural attention patterns.
    """
    print("    Running attention analysis...", flush=True)

    window_seq, pos_in_win = extract_window(full_sequence, mutation_pos_1based)

    inputs = tokenizer(window_seq, return_tensors="pt", padding=False, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (batch, heads, seq_len, seq_len) per layer
    # Stack into (layers, heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions)[:, 0]  # remove batch dim
    n_layers, n_heads, seq_len, _ = attentions.shape

    # Token position of mutation site (+1 for CLS)
    mut_token_pos = pos_in_win + 1

    # -- Attention TO the mutation site (which positions attend to it) --
    # attentions[:, :, :, mut_token_pos] = attention from all positions to mut site
    attn_to_mut = attentions[:, :, :, mut_token_pos]  # (layers, heads, seq_len)
    # Average over all heads and layers
    avg_attn_to_mut = attn_to_mut.mean(dim=(0, 1)).cpu().numpy()  # (seq_len,)

    # -- Attention FROM the mutation site (what does the mut site attend to) --
    attn_from_mut = attentions[:, :, mut_token_pos, :]  # (layers, heads, seq_len)
    avg_attn_from_mut = attn_from_mut.mean(dim=(0, 1)).cpu().numpy()  # (seq_len,)

    # Map token positions back to sequence positions
    # Tokens: [CLS] aa1 aa2 ... aaN [EOS]
    # seq position i -> token position i+1
    n_residues = len(window_seq)

    # Top residues attending TO mutation site
    to_scores = []
    for i in range(n_residues):
        tok_pos = i + 1
        to_scores.append({
            "seq_pos_in_window": i,
            "seq_pos_1based": i + (mutation_pos_1based - pos_in_win),
            "residue": window_seq[i],
            "attention_score": float(avg_attn_to_mut[tok_pos]),
            "distance_from_mutation": abs(i - pos_in_win),
        })
    top_attending_to = sorted(to_scores, key=lambda x: x["attention_score"], reverse=True)[:20]

    # Top residues the mutation site attends FROM
    from_scores = []
    for i in range(n_residues):
        tok_pos = i + 1
        from_scores.append({
            "seq_pos_in_window": i,
            "seq_pos_1based": i + (mutation_pos_1based - pos_in_win),
            "residue": window_seq[i],
            "attention_score": float(avg_attn_from_mut[tok_pos]),
            "distance_from_mutation": abs(i - pos_in_win),
        })
    top_attending_from = sorted(from_scores, key=lambda x: x["attention_score"], reverse=True)[:20]

    # Layer-wise analysis: which layers focus most on the mutation site
    layer_attention_to_mut = []
    for layer_idx in range(n_layers):
        layer_attn = attentions[layer_idx, :, :, mut_token_pos]  # (heads, seq_len)
        avg_layer = layer_attn.mean().item()
        max_head = layer_attn.mean(dim=1).max().item()
        layer_attention_to_mut.append({
            "layer": layer_idx,
            "avg_attention_to_mut": round(avg_layer, 6),
            "max_head_attention": round(max_head, 6),
        })

    # Find long-range interactions (>20 residues away, high attention)
    long_range_to = [r for r in top_attending_to if r["distance_from_mutation"] > 20]
    long_range_from = [r for r in top_attending_from if r["distance_from_mutation"] > 20]

    # Clean up for JSON serialization
    for lst in [top_attending_to, top_attending_from, long_range_to, long_range_from]:
        for r in lst:
            r["attention_score"] = round(r["attention_score"], 6)

    return {
        "window_length": n_residues,
        "mutation_pos_in_window": pos_in_win,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "top_residues_attending_to_mutation": top_attending_to,
        "top_residues_mutation_attends_to": top_attending_from,
        "long_range_contacts_to_mutation": long_range_to,
        "long_range_contacts_from_mutation": long_range_from,
        "layer_attention_profile": layer_attention_to_mut,
    }


# ---------------------------------------------------------------------------
# 3. EMBEDDING COMPARISON
# ---------------------------------------------------------------------------

def embedding_comparison(
    model: EsmForMaskedLM,
    tokenizer: AutoTokenizer,
    full_sequence: str,
    mutation_pos_1based: int,
    ref_aa: str,
    alt_aa: str,
) -> dict:
    """
    Compare wildtype vs mutant sequence embeddings:
      - Cosine similarity of full-sequence (CLS) embeddings
      - Cosine similarity of per-residue embeddings at the mutation site
      - L2 distance of per-residue embeddings
      - Embedding shift profile across the window
    """
    print("    Running embedding comparison...", flush=True)

    wt_window, pos_in_win = extract_window(full_sequence, mutation_pos_1based)

    # Create mutant window
    mut_window = list(wt_window)
    assert mut_window[pos_in_win] == ref_aa
    mut_window[pos_in_win] = alt_aa
    mut_window = "".join(mut_window)

    # We need hidden states, not logits. EsmForMaskedLM wraps EsmModel.
    # We can get hidden states via output_hidden_states=True
    def get_embeddings(seq: str):
        inputs = tokenizer(seq, return_tensors="pt", padding=False, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Last hidden state: (1, seq_len, hidden_dim)
        last_hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)
        return last_hidden

    wt_emb = get_embeddings(wt_window)   # (seq_len, hidden_dim)
    mut_emb = get_embeddings(mut_window)  # (seq_len, hidden_dim)

    # CLS token embedding (position 0)
    wt_cls = wt_emb[0]
    mut_cls = mut_emb[0]
    cls_cosine = F.cosine_similarity(wt_cls.unsqueeze(0), mut_cls.unsqueeze(0)).item()

    # Mutation-site embedding (position pos_in_win + 1 for CLS offset)
    mut_tok_pos = pos_in_win + 1
    wt_site = wt_emb[mut_tok_pos]
    mut_site = mut_emb[mut_tok_pos]
    site_cosine = F.cosine_similarity(wt_site.unsqueeze(0), mut_site.unsqueeze(0)).item()
    site_l2 = torch.norm(wt_site - mut_site, p=2).item()

    # Mean embedding (average over all token positions, excluding CLS and EOS)
    wt_mean = wt_emb[1:-1].mean(dim=0)
    mut_mean = mut_emb[1:-1].mean(dim=0)
    mean_cosine = F.cosine_similarity(wt_mean.unsqueeze(0), mut_mean.unsqueeze(0)).item()

    # Per-residue cosine similarity profile (how much does each position's embedding change?)
    n_residues = len(wt_window)
    residue_shifts = []
    for i in range(n_residues):
        tok_i = i + 1  # CLS offset
        cos_sim = F.cosine_similarity(wt_emb[tok_i].unsqueeze(0), mut_emb[tok_i].unsqueeze(0)).item()
        l2_dist = torch.norm(wt_emb[tok_i] - mut_emb[tok_i], p=2).item()
        residue_shifts.append({
            "pos_in_window": i,
            "pos_1based": i + (mutation_pos_1based - pos_in_win),
            "residue": wt_window[i],
            "cosine_similarity": round(cos_sim, 6),
            "l2_distance": round(l2_dist, 4),
            "is_mutation_site": (i == pos_in_win),
        })

    # Sort by most affected (lowest cosine similarity)
    most_affected = sorted(residue_shifts, key=lambda x: x["cosine_similarity"])[:15]

    # Compute how far the embedding shift propagates
    cos_sims = [r["cosine_similarity"] for r in residue_shifts]
    l2_dists = [r["l2_distance"] for r in residue_shifts]

    return {
        "cls_cosine_similarity": round(cls_cosine, 6),
        "mutation_site_cosine_similarity": round(site_cosine, 6),
        "mutation_site_l2_distance": round(site_l2, 4),
        "mean_embedding_cosine_similarity": round(mean_cosine, 6),
        "most_affected_residues": most_affected,
        "embedding_shift_stats": {
            "mean_cosine": round(np.mean(cos_sims), 6),
            "min_cosine": round(np.min(cos_sims), 6),
            "std_cosine": round(np.std(cos_sims), 6),
            "mean_l2": round(np.mean(l2_dists), 4),
            "max_l2": round(np.max(l2_dists), 4),
        },
        "window_length": n_residues,
    }


# ---------------------------------------------------------------------------
# 4. CONTEXT-DEPENDENT SCORING (epistasis)
# ---------------------------------------------------------------------------

def context_dependent_scoring(
    model: EsmForMaskedLM,
    tokenizer: AutoTokenizer,
    variants: list[dict],
    sequences: dict[str, str],
) -> dict:
    """
    For each mutation, score it:
      (a) in isolation (on its own protein)
      (b) Check if co-occurring mutations on the SAME protein change each other's scores
      (c) For mutations on DIFFERENT proteins, simulate epistasis by measuring how
          the presence of one mutation's embedding signature compares.

    Since these 4 mutations are on 4 different proteins, true cis-epistasis is not
    possible. Instead we:
      - Score each mutation in its wildtype context (baseline)
      - Create a "context embedding signature" for each mutation
      - Measure embedding-space distances between the 4 mutation signatures
        to identify which mutations produce the most similar structural perturbations
    """
    print("    Running context-dependent scoring...", flush=True)

    # First: score each mutation in isolation (masked marginal)
    isolation_scores = {}
    mutation_embeddings = {}

    for v in variants:
        gene = v["gene"]
        full_seq = sequences[v["uniprot_id"]]
        window_seq, pos_in_win = extract_window(full_seq, v["uniprot_pos"])

        # --- Masked marginal score ---
        masked = list(window_seq)
        masked[pos_in_win] = tokenizer.mask_token
        masked_str = "".join(masked)

        inputs = tokenizer(masked_str, return_tensors="pt", padding=False, truncation=True)
        inputs = {k: v_tensor.to(DEVICE) for k, v_tensor in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0]
        tok_pos = pos_in_win + 1
        log_probs = F.log_softmax(logits[tok_pos], dim=-1)

        ref_tid = tokenizer.convert_tokens_to_ids(v["ref_aa"])
        alt_tid = tokenizer.convert_tokens_to_ids(v["alt_aa"])
        score = log_probs[alt_tid].item() - log_probs[ref_tid].item()

        # Rank of mutant AA among all 20
        aa_tids = [tokenizer.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS]
        all_scores = [(aa, log_probs[tid].item()) for aa, tid in zip(AMINO_ACIDS, aa_tids)]
        all_scores_sorted = sorted(all_scores, key=lambda x: x[1], reverse=True)
        alt_rank = next(i + 1 for i, (aa, _) in enumerate(all_scores_sorted) if aa == v["alt_aa"])

        isolation_scores[gene] = {
            "score": round(score, 4),
            "ref_log_prob": round(log_probs[ref_tid].item(), 4),
            "alt_log_prob": round(log_probs[alt_tid].item(), 4),
            "alt_rank_of_20": alt_rank,
            "all_aa_scores": {aa: round(s, 4) for aa, s in all_scores_sorted},
        }

        # --- Mutation embedding signature ---
        # Compute difference between mutant and wildtype embeddings at the mutation site
        wt_inputs = tokenizer(window_seq, return_tensors="pt", padding=False, truncation=True)
        wt_inputs = {k: v_tensor.to(DEVICE) for k, v_tensor in wt_inputs.items()}

        mut_window = list(window_seq)
        mut_window[pos_in_win] = v["alt_aa"]
        mut_window_str = "".join(mut_window)
        mut_inputs = tokenizer(mut_window_str, return_tensors="pt", padding=False, truncation=True)
        mut_inputs = {k: v_tensor.to(DEVICE) for k, v_tensor in mut_inputs.items()}

        with torch.no_grad():
            wt_out = model(**wt_inputs, output_hidden_states=True)
            mut_out = model(**mut_inputs, output_hidden_states=True)

        wt_emb = wt_out.hidden_states[-1][0, tok_pos]
        mut_emb = mut_out.hidden_states[-1][0, tok_pos]
        delta = mut_emb - wt_emb  # mutation signature vector
        mutation_embeddings[gene] = delta.cpu()

        print(f"      {gene} {v['variant']}: score={score:.4f}, rank={alt_rank}/20", flush=True)

    # --- Cross-mutation embedding comparison ---
    # Pairwise cosine similarity of mutation signature vectors
    genes = list(mutation_embeddings.keys())
    n_genes = len(genes)
    pairwise_cosine = {}
    pairwise_l2 = {}

    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            g1, g2 = genes[i], genes[j]
            d1 = mutation_embeddings[g1].float()
            d2 = mutation_embeddings[g2].float()
            cos = F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
            l2 = torch.norm(d1 - d2, p=2).item()
            pair_key = f"{g1}-{g2}"
            pairwise_cosine[pair_key] = round(cos, 6)
            pairwise_l2[pair_key] = round(l2, 4)

    # --- Magnitude of each mutation's embedding perturbation ---
    perturbation_magnitudes = {}
    for gene, delta in mutation_embeddings.items():
        mag = torch.norm(delta.float(), p=2).item()
        perturbation_magnitudes[gene] = round(mag, 4)

    return {
        "isolation_scores": isolation_scores,
        "perturbation_magnitudes": perturbation_magnitudes,
        "pairwise_mutation_signature_cosine": pairwise_cosine,
        "pairwise_mutation_signature_l2": pairwise_l2,
        "interpretation": (
            "Mutations on different proteins cannot exhibit direct cis-epistasis. "
            "Instead, we compare their embedding perturbation signatures. "
            "High cosine similarity between two mutation signatures suggests they "
            "induce structurally analogous changes in their respective protein contexts, "
            "which may indicate convergent functional impact. "
            "Large perturbation magnitude indicates greater structural disruption."
        ),
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(all_results: dict, elapsed: float) -> str:
    """Generate a comprehensive markdown report."""
    lines = []
    lines.append("# ESM-2 Deep Variant Analysis Report\n")
    lines.append(f"**Model:** `{MODEL_NAME}`  ")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}  ")
    lines.append(f"**Runtime:** {elapsed:.1f}s  ")
    lines.append(f"**Analyses:** Positional scanning, attention weights, embedding comparison, context-dependent scoring\n")

    # ---- Overview Table ----
    lines.append("## Summary Table\n")
    lines.append("| Gene | Variant | Score | Rank/20 | Site Entropy | CLS Cosine | Site Cosine | Perturb. Mag. |")
    lines.append("|------|---------|-------|---------|-------------|------------|-------------|---------------|")

    for gene_key, data in all_results.items():
        v = data["variant_info"]
        scan = data["positional_scan"]["mutation_site"]
        emb = data["embedding_comparison"]
        ctx = data["context_dependent"]["isolation_scores"][v["gene"]]
        mag = data["context_dependent"]["perturbation_magnitudes"][v["gene"]]
        lines.append(
            f"| {v['gene']} | {v['variant']} | {ctx['score']:.4f} | "
            f"{ctx['alt_rank_of_20']} | {scan['entropy']:.4f} | "
            f"{emb['cls_cosine_similarity']:.6f} | "
            f"{emb['mutation_site_cosine_similarity']:.6f} | {mag:.4f} |"
        )

    # ---- Per-gene Deep Analysis ----
    for gene_key, data in all_results.items():
        v = data["variant_info"]
        lines.append(f"\n---\n## {v['gene']} {v['variant']}\n")
        lines.append(f"**UniProt:** {v['uniprot_id']} | **Position:** {v['uniprot_pos']} | **{v['note']}**\n")

        # Positional scan
        scan = data["positional_scan"]
        mut_site = scan["mutation_site"]
        lines.append("### Positional Scan\n")
        lines.append(f"- **Scan range:** positions {scan['scan_range']} ({scan['positions_scanned']} residues)")
        lines.append(f"- **Mutation site entropy:** {mut_site['entropy']:.4f} bits (lower = more conserved)")
        lines.append(f"- **Region mean entropy:** {scan['entropy_stats']['mean']:.4f} +/- {scan['entropy_stats']['std']:.4f}")
        lines.append(f"- **Wildtype ({v['ref_aa']}) log-prob:** {mut_site['wt_log_prob']:.4f} (rank {mut_site['wt_rank']}/20)")
        if "alt_rank" in mut_site:
            lines.append(f"- **Mutant ({v['alt_aa']}) log-prob:** {mut_site['alt_log_prob']:.4f} (rank {mut_site['alt_rank']}/20)")
            lines.append(f"- **Score:** {mut_site['score']:.4f}")

        lines.append("\n**Most conserved positions nearby** (lowest entropy):\n")
        lines.append("| Position | Residue | Entropy | WT Log-Prob | WT Rank |")
        lines.append("|----------|---------|---------|-------------|---------|")
        for p in scan["most_conserved_positions"][:10]:
            marker = " **MUT**" if p["is_mutation_site"] else ""
            lines.append(f"| {p['pos_1based']} | {p['wt_aa']}{marker} | {p['entropy']:.4f} | {p['wt_log_prob']:.4f} | {p['wt_rank']} |")

        # Attention
        attn = data["attention_analysis"]
        lines.append("\n### Attention Analysis\n")
        lines.append(f"- **Model:** {attn['n_layers']} layers, {attn['n_heads']} heads per layer")

        lines.append("\n**Top residues attending TO mutation site** (averaged over all layers/heads):\n")
        lines.append("| Position | Residue | Attention | Distance |")
        lines.append("|----------|---------|-----------|----------|")
        for r in attn["top_residues_attending_to_mutation"][:10]:
            lines.append(f"| {r['seq_pos_1based']} | {r['residue']} | {r['attention_score']:.6f} | {r['distance_from_mutation']} |")

        lines.append("\n**Top residues mutation site attends TO:**\n")
        lines.append("| Position | Residue | Attention | Distance |")
        lines.append("|----------|---------|-----------|----------|")
        for r in attn["top_residues_mutation_attends_to"][:10]:
            lines.append(f"| {r['seq_pos_1based']} | {r['residue']} | {r['attention_score']:.6f} | {r['distance_from_mutation']} |")

        if attn["long_range_contacts_to_mutation"]:
            lines.append("\n**Long-range contacts (>20 residues) attending to mutation:**\n")
            for r in attn["long_range_contacts_to_mutation"][:5]:
                lines.append(f"- Position {r['seq_pos_1based']} ({r['residue']}): attention={r['attention_score']:.6f}, distance={r['distance_from_mutation']}")

        # Embedding
        emb = data["embedding_comparison"]
        lines.append("\n### Embedding Comparison (Wildtype vs Mutant)\n")
        lines.append(f"- **CLS (global) cosine similarity:** {emb['cls_cosine_similarity']:.6f}")
        lines.append(f"- **Mutation site cosine similarity:** {emb['mutation_site_cosine_similarity']:.6f}")
        lines.append(f"- **Mutation site L2 distance:** {emb['mutation_site_l2_distance']:.4f}")
        lines.append(f"- **Mean embedding cosine similarity:** {emb['mean_embedding_cosine_similarity']:.6f}")
        lines.append(f"- **Mean L2 across all residues:** {emb['embedding_shift_stats']['mean_l2']:.4f}")
        lines.append(f"- **Max L2 (most shifted residue):** {emb['embedding_shift_stats']['max_l2']:.4f}")

        lines.append("\n**Most affected residues** (by embedding shift):\n")
        lines.append("| Position | Residue | Cosine Sim | L2 Distance | Is Mut Site |")
        lines.append("|----------|---------|------------|-------------|-------------|")
        for r in emb["most_affected_residues"][:10]:
            lines.append(f"| {r['pos_1based']} | {r['residue']} | {r['cosine_similarity']:.6f} | {r['l2_distance']:.4f} | {'YES' if r['is_mutation_site'] else ''} |")

    # ---- Cross-mutation context analysis ----
    ctx = list(all_results.values())[0]["context_dependent"]
    lines.append("\n---\n## Cross-Mutation Context Analysis\n")

    lines.append("### Isolation Scores (confirmed)\n")
    lines.append("| Gene | Variant | Score | Rank/20 |")
    lines.append("|------|---------|-------|---------|")
    for gene, iso in ctx["isolation_scores"].items():
        lines.append(f"| {gene} | - | {iso['score']:.4f} | {iso['alt_rank_of_20']} |")

    lines.append("\n### Perturbation Magnitudes\n")
    lines.append("| Gene | Embedding Perturbation (L2) | Interpretation |")
    lines.append("|------|-----------------------------|----------------|")
    sorted_mag = sorted(ctx["perturbation_magnitudes"].items(), key=lambda x: x[1], reverse=True)
    for gene, mag in sorted_mag:
        interp = "Large structural disruption" if mag > 10 else ("Moderate disruption" if mag > 5 else "Mild disruption")
        lines.append(f"| {gene} | {mag:.4f} | {interp} |")

    lines.append("\n### Pairwise Mutation Signature Similarity\n")
    lines.append("High cosine similarity = mutations induce structurally analogous perturbations.\n")
    lines.append("| Pair | Cosine Similarity | L2 Distance | Interpretation |")
    lines.append("|------|-------------------|-------------|----------------|")
    sorted_pairs = sorted(ctx["pairwise_mutation_signature_cosine"].items(), key=lambda x: x[1], reverse=True)
    for pair, cos in sorted_pairs:
        l2 = ctx["pairwise_mutation_signature_l2"][pair]
        if cos > 0.3:
            interp = "Structurally similar perturbation"
        elif cos > 0.0:
            interp = "Weakly similar perturbation"
        elif cos > -0.3:
            interp = "Orthogonal perturbation"
        else:
            interp = "Opposing perturbation"
        lines.append(f"| {pair} | {cos:.6f} | {l2:.4f} | {interp} |")

    # ---- Clinical Interpretation ----
    lines.append("\n---\n## Clinical Interpretation\n")
    lines.append(
        "This deep analysis extends the basic ESM-2 pathogenicity scoring with "
        "three additional dimensions:\n"
    )
    lines.append(
        "1. **Positional scanning** reveals the evolutionary conservation landscape "
        "around each mutation. Mutations at highly conserved positions (low entropy) "
        "are more likely to be functionally consequential.\n"
    )
    lines.append(
        "2. **Attention analysis** identifies residues that are functionally coupled "
        "to the mutation site in the model's learned representation. Long-range "
        "attention contacts may correspond to allosteric communication pathways.\n"
    )
    lines.append(
        "3. **Embedding comparison** quantifies the global and local structural "
        "impact of each mutation. A low cosine similarity at the mutation site "
        "indicates the model perceives a large structural change.\n"
    )
    lines.append(
        "4. **Cross-mutation analysis** compares the perturbation signatures of all "
        "4 mutations. Since they occur on different proteins, we cannot measure "
        "direct epistasis, but we can identify which mutations induce structurally "
        "analogous changes -- potentially indicating convergent pathogenic mechanisms.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global DEVICE
    start_time = time.time()

    print("=" * 70)
    print("ESM-2 DEEP VARIANT ANALYSIS")
    print(f"Model: {MODEL_NAME}")
    print(f"Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Device
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"\nGPU: {gpu_name} ({free_mem:.1f} GB free)")
    else:
        DEVICE = torch.device("cpu")
        print("\nWARNING: No GPU, using CPU")

    # Load model (use eager attention to support output_attentions)
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForMaskedLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model = model.to(DEVICE)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} parameters")

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory used by model: {used:.2f} GB")

    # Fetch all sequences upfront
    print("\nFetching protein sequences...")
    sequences = {}
    for v in VARIANTS:
        seq = fetch_uniprot_sequence(v["uniprot_id"])
        sequences[v["uniprot_id"]] = seq
        # Verify reference
        actual = seq[v["uniprot_pos"] - 1]
        assert actual == v["ref_aa"], f"{v['gene']}: expected {v['ref_aa']} at pos {v['uniprot_pos']}, got {actual}"
        print(f"    {v['gene']}: verified {v['ref_aa']} at position {v['uniprot_pos']}")

    # Run all analyses
    all_results = {}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for v in VARIANTS:
        gene = v["gene"]
        variant = v["variant"]
        print(f"\n{'='*70}")
        print(f"  ANALYZING: {gene} {variant}")
        print(f"{'='*70}")

        full_seq = sequences[v["uniprot_id"]]

        # 1. Positional scan
        t0 = time.time()
        scan = positional_scan(model, tokenizer, full_seq, v["uniprot_pos"], v["ref_aa"], v["alt_aa"])
        print(f"    Positional scan done in {time.time()-t0:.1f}s")

        # 2. Attention analysis
        t0 = time.time()
        attn = attention_analysis(model, tokenizer, full_seq, v["uniprot_pos"], v["ref_aa"])
        print(f"    Attention analysis done in {time.time()-t0:.1f}s")

        # 3. Embedding comparison
        t0 = time.time()
        emb = embedding_comparison(model, tokenizer, full_seq, v["uniprot_pos"], v["ref_aa"], v["alt_aa"])
        print(f"    Embedding comparison done in {time.time()-t0:.1f}s")

        # Clear GPU cache between variants
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_results[gene] = {
            "variant_info": v,
            "positional_scan": scan,
            "attention_analysis": attn,
            "embedding_comparison": emb,
            "context_dependent": None,  # filled after all variants processed
        }

    # 4. Context-dependent scoring (needs all sequences)
    print(f"\n{'='*70}")
    print("  CROSS-MUTATION CONTEXT ANALYSIS")
    print(f"{'='*70}")
    t0 = time.time()
    ctx = context_dependent_scoring(model, tokenizer, VARIANTS, sequences)
    print(f"    Context analysis done in {time.time()-t0:.1f}s")

    # Attach to all results
    for gene in all_results:
        all_results[gene]["context_dependent"] = ctx

    # Save results
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    # 1. Full JSON (without huge positional scan all_positions to keep file manageable)
    json_output = {}
    for gene, data in all_results.items():
        gene_data = {
            "variant_info": data["variant_info"],
            "positional_scan": {
                k: v for k, v in data["positional_scan"].items() if k != "all_positions"
            },
            "attention_analysis": data["attention_analysis"],
            "embedding_comparison": data["embedding_comparison"],
        }
        json_output[gene] = gene_data
    json_output["cross_mutation_context"] = ctx
    json_output["metadata"] = {
        "model": MODEL_NAME,
        "date": datetime.now().isoformat(),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "runtime_seconds": round(elapsed, 1),
        "scan_radius": SCAN_RADIUS,
        "window_size": WINDOW_SIZE,
    }

    json_path = RESULTS_DIR / "esm2_deep_results.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"  JSON: {json_path}")

    # 2. Full positional scan data (separate file, can be large)
    scan_data = {}
    for gene, data in all_results.items():
        scan_data[gene] = data["positional_scan"]["all_positions"]
    scan_path = RESULTS_DIR / "positional_scan_full.json"
    with open(scan_path, "w") as f:
        json.dump(scan_data, f, indent=2)
    print(f"  Positional scan data: {scan_path}")

    # 3. Markdown report
    report = generate_report(all_results, elapsed)
    md_path = RESULTS_DIR / "esm2_deep_report.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"  Report: {md_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\n{'Gene':<10} {'Variant':<10} {'Score':>10} {'Rank':>6} {'Entropy':>10} {'CLS Cos':>10} {'Perturb':>10}")
    print("-" * 66)
    for gene, data in all_results.items():
        v = data["variant_info"]
        scan = data["positional_scan"]["mutation_site"]
        emb = data["embedding_comparison"]
        iso = ctx["isolation_scores"][gene]
        mag = ctx["perturbation_magnitudes"][gene]
        print(
            f"{gene:<10} {v['variant']:<10} {iso['score']:>10.4f} "
            f"{iso['alt_rank_of_20']:>5}/20 {scan['entropy']:>10.4f} "
            f"{emb['cls_cosine_similarity']:>10.6f} {mag:>10.4f}"
        )

    print(f"\nTotal runtime: {elapsed:.1f}s")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
