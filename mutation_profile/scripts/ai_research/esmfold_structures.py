#!/usr/bin/env python3
"""
esmfold_structures.py -- ESM-2 structural disruption analysis via contact prediction.

Predicts structural impact of patient mutations using ESM-2 contact prediction
and per-residue embedding analysis.

Background:
    ESMFold (Meta's structure prediction model) requires OpenFold as a dependency,
    which fails to build in this environment. Instead, we use ESM-2's built-in
    contact prediction head (trained on the same representations) to assess
    structural disruption. This is a well-validated approach: ESM-2 contact
    predictions achieve >80% precision on long-range contacts (Lin et al. 2023).

Method:
    For each mutation, we extract a fragment (~100-150 residues) centered on the
    mutation site and compute:
    1. Contact maps for wildtype and mutant fragments
    2. Per-residue representation cosine similarity (embedding disruption)
    3. Contact difference maps showing gained/lost contacts
    4. Local contact density changes around the mutation site
    5. Pseudo-pLDDT from attention-weighted contact confidence

Mutations analyzed:
    1. DNMT3A R882H  (UniProt Q9Y6K1, pos 882) - catalytic domain hotspot
    2. IDH2 R140Q    (UniProt P48735, pos 140) - neomorphic gain-of-function
    3. SETBP1 G870S  (UniProt Q9Y6X0, pos 870) - SKI domain degron
    4. PTPN11 E76Q   (UniProt Q06124, pos 76)  - N-SH2 autoinhibition

Inputs:
    - UniProt REST API (remote, fetches canonical protein sequences)
    - HuggingFace model: facebook/esm2_t33_650M_UR50D (cached locally)

Outputs:
    - mutation_profile/results/ai_research/structural_analysis/*_contact_map_wt.png
    - mutation_profile/results/ai_research/structural_analysis/*_contact_map_mut.png
    - mutation_profile/results/ai_research/structural_analysis/*_contact_diff.png
    - mutation_profile/results/ai_research/structural_analysis/*_disruption_scores.png
    - mutation_profile/results/ai_research/structural_analysis/structural_summary.json
    - mutation_profile/results/ai_research/structural_analysis/structural_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/esmfold_structures.py

Runtime: ~3-5 minutes (GPU), ~15 minutes (CPU)
Dependencies: torch, numpy, matplotlib, requests, esm
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import requests
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "ai_research" / "esmfold_structures"

# Fragment size: 100-150 residues centered on mutation
# Kept short to maximize contact map resolution and fit GPU memory
FRAGMENT_RADIUS = 60  # residues on each side = ~121 residue fragments

# Contact analysis parameters
CONTACT_THRESHOLD = 0.5    # probability threshold for "confident contact"
LOCAL_WINDOW = 12          # residues around mutation for local density

# Mutations to analyze (4 core mutations, excluding EZH2 V662A from this analysis)
MUTATIONS = [
    {
        "gene": "DNMT3A",
        "variant": "R882H",
        "uniprot_id": "Q9Y6K1",
        "position": 882,
        "ref_aa": "R",
        "alt_aa": "H",
        "domain": "Methyltransferase catalytic domain",
        "mechanism": "Loss of methyltransferase activity, dominant-negative on WT DNMT3A",
    },
    {
        "gene": "IDH2",
        "variant": "R140Q",
        "uniprot_id": "P48735",
        "position": 140,
        "ref_aa": "R",
        "alt_aa": "Q",
        "domain": "Isocitrate binding site",
        "mechanism": "Neomorphic: gains 2-hydroxyglutarate production",
    },
    {
        "gene": "SETBP1",
        "variant": "G870S",
        "uniprot_id": "Q9Y6X0",
        "position": 870,
        "ref_aa": "G",
        "alt_aa": "S",
        "domain": "SKI homology domain (degron motif)",
        "mechanism": "Disrupts beta-TrCP degron, stabilizes protein",
    },
    {
        "gene": "PTPN11",
        "variant": "E76Q",
        "uniprot_id": "Q06124",
        "position": 76,
        "ref_aa": "E",
        "alt_aa": "Q",
        "domain": "N-SH2 domain (autoinhibitory interface)",
        "mechanism": "Disrupts N-SH2/PTP interface, constitutive activation",
    },
]


# ---------------------------------------------------------------------------
# Sequence retrieval
# ---------------------------------------------------------------------------

def fetch_uniprot_sequence(uniprot_id: str) -> str:
    """Fetch canonical protein sequence from UniProt."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    print(f"    Fetching {uniprot_id} from UniProt...", end=" ", flush=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    seq = "".join(lines[1:])
    print(f"OK ({len(seq)} residues)")
    return seq


def extract_fragment(sequence: str, position_1based: int, radius: int = FRAGMENT_RADIUS):
    """
    Extract a fragment centered on the mutation site.

    Returns:
        (fragment_seq, mutation_index_in_fragment, start_pos_1based, end_pos_1based)
    """
    idx = position_1based - 1  # 0-based
    start = max(0, idx - radius)
    end = min(len(sequence), idx + radius + 1)
    fragment = sequence[start:end]
    mut_idx = idx - start
    return fragment, mut_idx, start + 1, end


def make_mutant_fragment(wt_fragment: str, mut_idx: int, alt_aa: str) -> str:
    """Create mutant version of a fragment by substituting one residue."""
    fragment_list = list(wt_fragment)
    fragment_list[mut_idx] = alt_aa
    return "".join(fragment_list)


# ---------------------------------------------------------------------------
# ESM-2 inference
# ---------------------------------------------------------------------------

def run_esm2_analysis(model, alphabet, sequence: str, device: torch.device):
    """
    Run ESM-2 on a sequence and extract:
    - Contact map (L x L probability matrix)
    - Per-residue representations (L x D matrix)
    - Attention maps for pseudo-confidence

    Returns dict with 'contacts', 'representations', 'attentions'.
    """
    batch_converter = alphabet.get_batch_converter()
    data = [("seq", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    # Contact map: (L, L) probabilities
    contacts = results["contacts"][0].cpu().numpy()

    # Per-residue representations from final layer: (L, 1280)
    representations = results["representations"][33][0, 1:-1].cpu().numpy()  # strip BOS/EOS

    # Attention maps for confidence estimation
    # Shape: (layers, heads, L+2, L+2) -- we use the last layer
    attentions = results["attentions"][0].cpu().numpy()  # all layers for this sequence

    return {
        "contacts": contacts,
        "representations": representations,
        "attentions": attentions,
    }


def compute_pseudo_plddt(attentions, seq_len: int) -> np.ndarray:
    """
    Compute pseudo-pLDDT scores from attention weights.

    ESM-2 attention patterns correlate with structural confidence. We compute
    per-residue scores by averaging attention entropy across heads in the
    final layers. Lower entropy = more confident structural environment.

    The scores are normalized to 0-100 scale to match pLDDT convention.
    """
    # Use last 4 layers, all heads
    # attentions shape: (layers, heads, L+2, L+2)
    n_layers = attentions.shape[0]
    last_layers = attentions[max(0, n_layers - 4):n_layers]  # last 4 layers

    # For each residue, compute mean attention concentration (inverse entropy)
    # across heads and layers. Offset by 1 for BOS token.
    plddt_scores = np.zeros(seq_len)

    for i in range(seq_len):
        token_idx = i + 1  # skip BOS
        attn_profiles = last_layers[:, :, token_idx, 1:seq_len + 1]  # (layers, heads, L)
        # Normalize each attention distribution
        attn_profiles = attn_profiles / (attn_profiles.sum(axis=-1, keepdims=True) + 1e-10)
        # Entropy per head per layer
        entropy = -np.sum(attn_profiles * np.log(attn_profiles + 1e-10), axis=-1)
        # Max entropy for uniform distribution
        max_entropy = np.log(seq_len)
        # Inverse normalized entropy = confidence
        confidence = 1.0 - (entropy.mean() / max_entropy)
        plddt_scores[i] = confidence * 100.0

    return plddt_scores


def compute_contact_disruption(wt_contacts, mut_contacts, mut_idx: int, local_window: int = LOCAL_WINDOW):
    """
    Quantify how the mutation disrupts the contact map.

    Returns dict with:
    - diff_map: mut - wt contact probabilities
    - lost_contacts: number of contacts lost (above threshold in WT, below in mutant)
    - gained_contacts: number of contacts gained
    - local_density_wt: contact density near mutation site (WT)
    - local_density_mut: contact density near mutation site (mutant)
    - max_disruption: maximum absolute change in any contact involving the mutation site
    - disruption_score: aggregate measure of structural disruption (0-1 scale)
    """
    diff = mut_contacts - wt_contacts
    L = wt_contacts.shape[0]

    # Count lost and gained contacts (long-range only, |i-j| > 5)
    lost = 0
    gained = 0
    for i in range(L):
        for j in range(i + 6, L):
            if wt_contacts[i, j] > CONTACT_THRESHOLD and mut_contacts[i, j] < CONTACT_THRESHOLD:
                lost += 1
            if wt_contacts[i, j] < CONTACT_THRESHOLD and mut_contacts[i, j] > CONTACT_THRESHOLD:
                gained += 1

    # Local contact density around mutation site
    local_start = max(0, mut_idx - local_window)
    local_end = min(L, mut_idx + local_window + 1)

    # Contacts from the mutation site to all other residues (long-range)
    wt_site_contacts = wt_contacts[mut_idx, :]
    mut_site_contacts = mut_contacts[mut_idx, :]

    # Mask out short-range contacts (|i-j| <= 5)
    mask = np.abs(np.arange(L) - mut_idx) > 5
    wt_lr = wt_site_contacts[mask]
    mut_lr = mut_site_contacts[mask]

    local_density_wt = (wt_lr > CONTACT_THRESHOLD).sum()
    local_density_mut = (mut_lr > CONTACT_THRESHOLD).sum()

    # Maximum disruption at the mutation site
    site_diff = np.abs(diff[mut_idx, :])
    site_diff_lr = site_diff[mask]
    max_disruption = site_diff_lr.max() if len(site_diff_lr) > 0 else 0.0

    # Aggregate disruption score: RMS of contact probability changes at mutation site
    rms_disruption = np.sqrt(np.mean(site_diff_lr ** 2)) if len(site_diff_lr) > 0 else 0.0

    # Normalize to 0-1 scale (empirically, RMS > 0.1 is significant)
    disruption_score = min(1.0, rms_disruption / 0.15)

    return {
        "diff_map": diff,
        "lost_contacts": int(lost),
        "gained_contacts": int(gained),
        "local_density_wt": int(local_density_wt),
        "local_density_mut": int(local_density_mut),
        "density_change": int(local_density_mut) - int(local_density_wt),
        "max_disruption": float(max_disruption),
        "rms_disruption": float(rms_disruption),
        "disruption_score": float(disruption_score),
    }


def compute_embedding_disruption(wt_repr, mut_repr, mut_idx: int, local_window: int = LOCAL_WINDOW):
    """
    Measure embedding-level disruption by comparing WT and mutant representations.

    Returns:
    - Per-residue cosine similarity between WT and mutant
    - Local disruption around the mutation site
    - Global disruption across the full fragment
    """
    L = wt_repr.shape[0]

    # Per-residue cosine similarity
    cosine_sim = np.zeros(L)
    for i in range(L):
        wt_vec = wt_repr[i]
        mut_vec = mut_repr[i]
        cos = np.dot(wt_vec, mut_vec) / (np.linalg.norm(wt_vec) * np.linalg.norm(mut_vec) + 1e-10)
        cosine_sim[i] = cos

    # Local disruption (around mutation site)
    local_start = max(0, mut_idx - local_window)
    local_end = min(L, mut_idx + local_window + 1)
    local_disruption = 1.0 - cosine_sim[local_start:local_end].mean()

    # Global disruption
    global_disruption = 1.0 - cosine_sim.mean()

    # Disruption at mutation site itself
    site_disruption = 1.0 - cosine_sim[mut_idx]

    return {
        "cosine_similarity": cosine_sim,
        "site_disruption": float(site_disruption),
        "local_disruption": float(local_disruption),
        "global_disruption": float(global_disruption),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_contact_maps(wt_contacts, mut_contacts, diff_map, mut_idx,
                      gene, variant, frag_start, output_dir: Path):
    """Plot wildtype, mutant, and difference contact maps side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    L = wt_contacts.shape[0]
    extent = [frag_start, frag_start + L, frag_start + L, frag_start]
    mut_pos_abs = frag_start + mut_idx

    # Wildtype
    im0 = axes[0].imshow(wt_contacts, cmap="Blues", vmin=0, vmax=1, extent=extent)
    axes[0].axhline(y=mut_pos_abs, color="red", linewidth=0.5, alpha=0.7)
    axes[0].axvline(x=mut_pos_abs, color="red", linewidth=0.5, alpha=0.7)
    axes[0].set_title(f"{gene} Wildtype", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Residue position")
    axes[0].set_ylabel("Residue position")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Contact prob.")

    # Mutant
    im1 = axes[1].imshow(mut_contacts, cmap="Blues", vmin=0, vmax=1, extent=extent)
    axes[1].axhline(y=mut_pos_abs, color="red", linewidth=0.5, alpha=0.7)
    axes[1].axvline(x=mut_pos_abs, color="red", linewidth=0.5, alpha=0.7)
    axes[1].set_title(f"{gene} {variant}", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residue position")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Contact prob.")

    # Difference
    max_abs = max(0.1, np.abs(diff_map).max())
    im2 = axes[2].imshow(diff_map, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, extent=extent)
    axes[2].axhline(y=mut_pos_abs, color="black", linewidth=0.5, alpha=0.7)
    axes[2].axvline(x=mut_pos_abs, color="black", linewidth=0.5, alpha=0.7)
    axes[2].set_title("Difference (Mut - WT)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Residue position")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Delta prob.")

    fig.suptitle(f"{gene} {variant} - ESM-2 Contact Prediction", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = output_dir / f"{gene}_{variant}_contact_maps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_plddt_comparison(wt_plddt, mut_plddt, mut_idx, gene, variant,
                          frag_start, output_dir: Path):
    """Plot pseudo-pLDDT scores for wildtype vs mutant."""
    fig, ax = plt.subplots(figsize=(12, 4))

    L = len(wt_plddt)
    positions = np.arange(frag_start, frag_start + L)
    mut_pos_abs = frag_start + mut_idx

    ax.plot(positions, wt_plddt, color="#2166ac", linewidth=1.2, alpha=0.8, label="Wildtype")
    ax.plot(positions, mut_plddt, color="#b2182b", linewidth=1.2, alpha=0.8, label=f"Mutant ({variant})")

    # Mark mutation site
    ax.axvline(x=mut_pos_abs, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.annotate(f"{variant}\n(pos {mut_pos_abs})",
                xy=(mut_pos_abs, max(wt_plddt[mut_idx], mut_plddt[mut_idx])),
                xytext=(15, 10), textcoords="offset points",
                fontsize=9, arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # Shade local window
    local_start = max(frag_start, mut_pos_abs - LOCAL_WINDOW)
    local_end = min(frag_start + L, mut_pos_abs + LOCAL_WINDOW + 1)
    ax.axvspan(local_start, local_end, alpha=0.08, color="gray")

    ax.set_xlabel("Residue position", fontsize=10)
    ax.set_ylabel("Pseudo-pLDDT", fontsize=10)
    ax.set_title(f"{gene} {variant} - Pseudo-pLDDT (Attention-derived confidence)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = output_dir / f"{gene}_{variant}_pseudo_plddt.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_embedding_disruption(cosine_sim, mut_idx, gene, variant,
                              frag_start, output_dir: Path):
    """Plot per-residue embedding disruption (1 - cosine similarity)."""
    fig, ax = plt.subplots(figsize=(12, 4))

    L = len(cosine_sim)
    positions = np.arange(frag_start, frag_start + L)
    mut_pos_abs = frag_start + mut_idx
    disruption = 1.0 - cosine_sim

    # Color bars by disruption level
    colors = np.where(disruption > 0.05, "#b2182b",
                      np.where(disruption > 0.02, "#ef8a62", "#4393c3"))

    ax.bar(positions, disruption, width=1.0, color=colors, alpha=0.8)
    ax.axvline(x=mut_pos_abs, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

    ax.annotate(f"{variant}\n(pos {mut_pos_abs})",
                xy=(mut_pos_abs, disruption[mut_idx]),
                xytext=(15, 10), textcoords="offset points",
                fontsize=9, arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_xlabel("Residue position", fontsize=10)
    ax.set_ylabel("Embedding disruption\n(1 - cosine similarity)", fontsize=10)
    ax.set_title(f"{gene} {variant} - Per-residue Embedding Disruption",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    path = output_dir / f"{gene}_{variant}_embedding_disruption.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_combined_summary(all_results, output_dir: Path):
    """Create a combined 4-panel summary figure for all mutations."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    for idx, result in enumerate(all_results):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        gene = result["gene"]
        variant = result["variant"]
        cosine_sim = result["embedding"]["cosine_similarity"]
        mut_idx = result["mut_idx"]
        frag_start = result["frag_start"]

        L = len(cosine_sim)
        positions = np.arange(frag_start, frag_start + L)
        mut_pos_abs = frag_start + mut_idx
        disruption = 1.0 - cosine_sim

        colors = np.where(disruption > 0.05, "#b2182b",
                          np.where(disruption > 0.02, "#ef8a62", "#4393c3"))
        ax.bar(positions, disruption, width=1.0, color=colors, alpha=0.8)
        ax.axvline(x=mut_pos_abs, color="black", linewidth=1.5, linestyle="--", alpha=0.5)

        # Annotation with key metrics
        contact_info = result["contacts"]
        embed_info = result["embedding"]
        text = (f"Site: {embed_info['site_disruption']:.3f}\n"
                f"Local: {embed_info['local_disruption']:.3f}\n"
                f"Contacts lost: {contact_info['lost_contacts']}")
        ax.text(0.97, 0.95, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(f"{gene} {variant}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Residue position", fontsize=9)
        ax.set_ylabel("Embedding disruption", fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("ESM-2 Structural Disruption Analysis - All Mutations",
                 fontsize=14, fontweight="bold", y=0.98)

    path = output_dir / "combined_disruption_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# PDB output (pseudo-structure from contact predictions)
# ---------------------------------------------------------------------------

def write_pseudo_pdb(sequence: str, plddt_scores: np.ndarray, frag_start: int,
                     label: str, output_path: Path):
    """
    Write a pseudo-PDB file with CA-only coordinates arranged linearly,
    with B-factors set to pseudo-pLDDT scores.

    This is not a real 3D structure but allows visualization tools (PyMOL, ChimeraX)
    to color by confidence using B-factor coloring, which is the standard pLDDT
    visualization approach used by AlphaFold and ESMFold.
    """
    with open(output_path, "w") as f:
        f.write(f"REMARK   ESM-2 pseudo-structure for {label}\n")
        f.write(f"REMARK   B-factors = pseudo-pLDDT scores (0-100)\n")
        f.write(f"REMARK   Coordinates are LINEAR (not real 3D structure)\n")
        f.write(f"REMARK   Use B-factor coloring to visualize confidence\n")

        aa_3letter = {
            "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
            "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
            "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
            "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
        }

        for i, (aa, plddt) in enumerate(zip(sequence, plddt_scores)):
            resname = aa_3letter.get(aa, "UNK")
            resnum = frag_start + i
            # Linear arrangement along x-axis, 3.8A spacing (CA-CA distance)
            x = i * 3.8
            y = 0.0
            z = 0.0
            f.write(
                f"ATOM  {i+1:5d}  CA  {resname} A{resnum:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}           C\n"
            )
        f.write("END\n")


# ---------------------------------------------------------------------------
# Structural impact classification
# ---------------------------------------------------------------------------

def classify_structural_impact(contact_result, embedding_result, wt_plddt, mut_plddt, mut_idx):
    """
    Classify the structural impact of a mutation based on combined metrics.

    Returns (severity, description) tuple.
    """
    site_disruption = embedding_result["site_disruption"]
    local_disruption = embedding_result["local_disruption"]
    contacts_lost = contact_result["lost_contacts"]
    disruption_score = contact_result["disruption_score"]
    plddt_drop = wt_plddt[mut_idx] - mut_plddt[mut_idx]

    # Composite score (weighted)
    composite = (
        0.30 * min(1.0, site_disruption / 0.10) +   # site embedding change
        0.25 * min(1.0, local_disruption / 0.05) +   # local embedding change
        0.20 * disruption_score +                      # contact map disruption
        0.15 * min(1.0, contacts_lost / 5.0) +        # lost contacts
        0.10 * min(1.0, max(0, plddt_drop) / 10.0)    # confidence drop
    )

    if composite > 0.7:
        severity = "HIGH"
        desc = "Severe structural disruption predicted"
    elif composite > 0.4:
        severity = "MODERATE"
        desc = "Moderate structural disruption predicted"
    elif composite > 0.2:
        severity = "LOW"
        desc = "Mild structural perturbation predicted"
    else:
        severity = "MINIMAL"
        desc = "Minimal structural impact predicted"

    return severity, desc, round(composite, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    print("=" * 72)
    print("ESM-2 Structural Disruption Analysis")
    print("(Fallback for ESMFold: contact prediction + embedding analysis)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("\nWARNING: No GPU detected, using CPU")

    # Load ESM-2 model (with contact prediction head)
    print("\nLoading ESM-2 (esm2_t33_650M_UR50D) with contact head...")
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used by model: {alloc:.2f} GB")

    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pdb_dir = RESULTS_DIR / "pdb_files"
    pdb_dir.mkdir(exist_ok=True)

    # Process each mutation
    all_results = []
    print(f"\nAnalyzing {len(MUTATIONS)} mutations...")
    print("-" * 72)

    for mut in MUTATIONS:
        gene = mut["gene"]
        variant = mut["variant"]
        print(f"\n{'='*60}")
        print(f"  {gene} {variant} ({mut['domain']})")
        print(f"{'='*60}")

        # 1. Fetch sequence and extract fragment
        full_seq = fetch_uniprot_sequence(mut["uniprot_id"])
        actual_ref = full_seq[mut["position"] - 1]
        assert actual_ref == mut["ref_aa"], (
            f"Reference mismatch: expected {mut['ref_aa']} at pos {mut['position']}, got {actual_ref}"
        )
        print(f"    Reference verified: {mut['ref_aa']} at position {mut['position']}")

        wt_frag, mut_idx, frag_start, frag_end = extract_fragment(full_seq, mut["position"])
        mut_frag = make_mutant_fragment(wt_frag, mut_idx, mut["alt_aa"])

        print(f"    Fragment: residues {frag_start}-{frag_end} ({len(wt_frag)} aa)")
        print(f"    Mutation at fragment index {mut_idx}")
        ctx_left = wt_frag[max(0, mut_idx-5):mut_idx]
        ctx_right = wt_frag[mut_idx+1:mut_idx+6]
        print(f"    Context: ...{ctx_left}[{mut['ref_aa']}->{mut['alt_aa']}]{ctx_right}...")

        # 2. Run ESM-2 on wildtype and mutant
        print(f"    Running ESM-2 on wildtype fragment...", end=" ", flush=True)
        t0 = time.time()
        wt_analysis = run_esm2_analysis(model, alphabet, wt_frag, device)
        print(f"({time.time()-t0:.1f}s)")

        print(f"    Running ESM-2 on mutant fragment...", end=" ", flush=True)
        t0 = time.time()
        mut_analysis = run_esm2_analysis(model, alphabet, mut_frag, device)
        print(f"({time.time()-t0:.1f}s)")

        # 3. Compute pseudo-pLDDT
        print(f"    Computing pseudo-pLDDT scores...")
        wt_plddt = compute_pseudo_plddt(wt_analysis["attentions"], len(wt_frag))
        mut_plddt = compute_pseudo_plddt(mut_analysis["attentions"], len(mut_frag))

        plddt_drop = wt_plddt[mut_idx] - mut_plddt[mut_idx]
        print(f"    Pseudo-pLDDT at mutation site: WT={wt_plddt[mut_idx]:.1f}, "
              f"Mut={mut_plddt[mut_idx]:.1f} (delta={plddt_drop:+.1f})")
        print(f"    Mean pseudo-pLDDT: WT={wt_plddt.mean():.1f}, Mut={mut_plddt.mean():.1f}")

        # 4. Contact disruption analysis
        print(f"    Analyzing contact map disruption...")
        contact_result = compute_contact_disruption(
            wt_analysis["contacts"], mut_analysis["contacts"], mut_idx
        )
        print(f"    Long-range contacts lost: {contact_result['lost_contacts']}, "
              f"gained: {contact_result['gained_contacts']}")
        print(f"    Local contact density: WT={contact_result['local_density_wt']}, "
              f"Mut={contact_result['local_density_mut']} "
              f"(delta={contact_result['density_change']:+d})")
        print(f"    Contact disruption score: {contact_result['disruption_score']:.3f}")

        # 5. Embedding disruption analysis
        print(f"    Analyzing embedding disruption...")
        embedding_result = compute_embedding_disruption(
            wt_analysis["representations"], mut_analysis["representations"], mut_idx
        )
        print(f"    Embedding disruption at site: {embedding_result['site_disruption']:.4f}")
        print(f"    Local disruption (+/-{LOCAL_WINDOW} residues): "
              f"{embedding_result['local_disruption']:.4f}")
        print(f"    Global disruption: {embedding_result['global_disruption']:.4f}")

        # 6. Classify structural impact
        severity, desc, composite = classify_structural_impact(
            contact_result, embedding_result, wt_plddt, mut_plddt, mut_idx
        )
        print(f"    --> Structural impact: {severity} ({desc})")
        print(f"    --> Composite score: {composite:.4f}")

        # 7. Generate plots
        print(f"    Generating plots...")
        contact_plot = plot_contact_maps(
            wt_analysis["contacts"], mut_analysis["contacts"],
            contact_result["diff_map"], mut_idx,
            gene, variant, frag_start, RESULTS_DIR
        )
        plddt_plot = plot_plddt_comparison(
            wt_plddt, mut_plddt, mut_idx, gene, variant, frag_start, RESULTS_DIR
        )
        embed_plot = plot_embedding_disruption(
            embedding_result["cosine_similarity"], mut_idx,
            gene, variant, frag_start, RESULTS_DIR
        )

        # 8. Write pseudo-PDB files
        wt_pdb_path = pdb_dir / f"{gene}_{variant}_wildtype.pdb"
        mut_pdb_path = pdb_dir / f"{gene}_{variant}_mutant.pdb"
        write_pseudo_pdb(wt_frag, wt_plddt, frag_start,
                         f"{gene} WT (pos {frag_start}-{frag_end})", wt_pdb_path)
        write_pseudo_pdb(mut_frag, mut_plddt, frag_start,
                         f"{gene} {variant} (pos {frag_start}-{frag_end})", mut_pdb_path)
        print(f"    PDB files: {wt_pdb_path.name}, {mut_pdb_path.name}")

        # Store results
        result = {
            "gene": gene,
            "variant": variant,
            "uniprot_id": mut["uniprot_id"],
            "position": mut["position"],
            "domain": mut["domain"],
            "mechanism": mut["mechanism"],
            "fragment_start": frag_start,
            "fragment_end": frag_end,
            "fragment_length": len(wt_frag),
            "mut_idx": mut_idx,
            "frag_start": frag_start,
            "pseudo_plddt": {
                "wt_at_site": round(float(wt_plddt[mut_idx]), 2),
                "mut_at_site": round(float(mut_plddt[mut_idx]), 2),
                "plddt_drop": round(float(plddt_drop), 2),
                "wt_mean": round(float(wt_plddt.mean()), 2),
                "mut_mean": round(float(mut_plddt.mean()), 2),
            },
            "contacts": {
                "lost_contacts": contact_result["lost_contacts"],
                "gained_contacts": contact_result["gained_contacts"],
                "local_density_wt": contact_result["local_density_wt"],
                "local_density_mut": contact_result["local_density_mut"],
                "density_change": contact_result["density_change"],
                "max_disruption": round(contact_result["max_disruption"], 4),
                "rms_disruption": round(contact_result["rms_disruption"], 4),
                "disruption_score": round(contact_result["disruption_score"], 4),
            },
            "embedding": {
                "site_disruption": round(embedding_result["site_disruption"], 4),
                "local_disruption": round(embedding_result["local_disruption"], 4),
                "global_disruption": round(embedding_result["global_disruption"], 4),
                "cosine_similarity": embedding_result["cosine_similarity"],
            },
            "structural_impact": {
                "severity": severity,
                "description": desc,
                "composite_score": composite,
            },
            "files": {
                "contact_map": str(contact_plot),
                "plddt_plot": str(plddt_plot),
                "embedding_plot": str(embed_plot),
                "wt_pdb": str(wt_pdb_path),
                "mut_pdb": str(mut_pdb_path),
            },
        }
        all_results.append(result)

        # Clear GPU cache between mutations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combined summary plot
    print(f"\nGenerating combined summary plot...")
    summary_plot = plot_combined_summary(all_results, RESULTS_DIR)

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n{'='*72}")
    print("Saving results...")

    # JSON (strip numpy arrays for serialization)
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k != "embedding"}
        jr["embedding"] = {
            k: v for k, v in r["embedding"].items() if k != "cosine_similarity"
        }
        json_results.append(jr)

    json_path = RESULTS_DIR / "esmfold_structural_analysis.json"
    output_json = {
        "metadata": {
            "method": "ESM-2 contact prediction + embedding disruption analysis",
            "model": "esm2_t33_650M_UR50D (with contact prediction head)",
            "fallback_note": "ESMFold requires OpenFold which failed to install; "
                            "using ESM-2 contact maps and embeddings instead",
            "date": datetime.now().isoformat(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "fragment_radius": FRAGMENT_RADIUS,
            "contact_threshold": CONTACT_THRESHOLD,
            "local_window": LOCAL_WINDOW,
            "runtime_seconds": round(elapsed, 1),
        },
        "mutations": json_results,
    }
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  JSON: {json_path}")

    # Markdown summary
    md_path = RESULTS_DIR / "esmfold_structural_summary.md"
    with open(md_path, "w") as f:
        f.write("# ESM-2 Structural Disruption Analysis\n\n")
        f.write(f"**Method:** ESM-2 contact prediction + embedding disruption analysis  \n")
        f.write(f"**Model:** `esm2_t33_650M_UR50D` (with contact prediction head)  \n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}  \n")
        f.write(f"**Runtime:** {elapsed:.1f}s  \n\n")

        f.write("## Background\n\n")
        f.write("ESMFold (Meta's protein structure prediction model) requires OpenFold as a dependency, "
                "which could not be installed in this environment. As a validated fallback, we use "
                "ESM-2's built-in contact prediction head and per-residue embeddings to assess "
                "structural disruption from mutations.\n\n")
        f.write("ESM-2 contact predictions achieve >80% precision on long-range contacts "
                "(Lin et al., Science 2023). The approach compares wildtype and mutant fragments "
                "(~121 residues centered on each mutation site) across three axes:\n\n")
        f.write("1. **Contact map disruption** -- gained/lost residue-residue contacts\n")
        f.write("2. **Embedding disruption** -- cosine distance between WT and mutant representations\n")
        f.write("3. **Pseudo-pLDDT** -- attention-derived confidence scores\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Gene | Variant | Impact | Composite | Site Disruption | "
                "Contacts Lost | pLDDT Drop |\n")
        f.write("|------|---------|--------|-----------|-----------------|"
                "---------------|------------|\n")
        for r in all_results:
            f.write(
                f"| {r['gene']} | {r['variant']} "
                f"| **{r['structural_impact']['severity']}** "
                f"| {r['structural_impact']['composite_score']:.4f} "
                f"| {r['embedding']['site_disruption']:.4f} "
                f"| {r['contacts']['lost_contacts']} "
                f"| {r['pseudo_plddt']['plddt_drop']:+.1f} |\n"
            )

        f.write("\n## Detailed Results\n\n")
        for r in all_results:
            f.write(f"### {r['gene']} {r['variant']}\n\n")
            f.write(f"- **Domain:** {r['domain']}\n")
            f.write(f"- **Mechanism:** {r['mechanism']}\n")
            f.write(f"- **Fragment:** residues {r['fragment_start']}-{r['fragment_end']} "
                    f"({r['fragment_length']} aa)\n")
            f.write(f"- **Structural impact:** {r['structural_impact']['severity']} "
                    f"({r['structural_impact']['description']})\n")
            f.write(f"- **Composite score:** {r['structural_impact']['composite_score']:.4f}\n\n")

            f.write("**Pseudo-pLDDT:**\n\n")
            f.write(f"| Metric | Wildtype | Mutant | Delta |\n")
            f.write(f"|--------|----------|--------|-------|\n")
            f.write(f"| At mutation site | {r['pseudo_plddt']['wt_at_site']:.1f} "
                    f"| {r['pseudo_plddt']['mut_at_site']:.1f} "
                    f"| {r['pseudo_plddt']['plddt_drop']:+.1f} |\n")
            f.write(f"| Fragment mean | {r['pseudo_plddt']['wt_mean']:.1f} "
                    f"| {r['pseudo_plddt']['mut_mean']:.1f} "
                    f"| {r['pseudo_plddt']['wt_mean'] - r['pseudo_plddt']['mut_mean']:+.1f} |\n\n")

            f.write("**Contact disruption:**\n\n")
            f.write(f"- Long-range contacts lost: {r['contacts']['lost_contacts']}\n")
            f.write(f"- Long-range contacts gained: {r['contacts']['gained_contacts']}\n")
            f.write(f"- Local density (WT): {r['contacts']['local_density_wt']}, "
                    f"(Mut): {r['contacts']['local_density_mut']} "
                    f"(delta: {r['contacts']['density_change']:+d})\n")
            f.write(f"- RMS disruption: {r['contacts']['rms_disruption']:.4f}\n\n")

            f.write("**Embedding disruption:**\n\n")
            f.write(f"- At mutation site: {r['embedding']['site_disruption']:.4f}\n")
            f.write(f"- Local (+/-{LOCAL_WINDOW} residues): {r['embedding']['local_disruption']:.4f}\n")
            f.write(f"- Global: {r['embedding']['global_disruption']:.4f}\n\n")

        f.write("## Interpretation\n\n")
        f.write("### Composite scoring\n\n")
        f.write("The composite score (0-1) integrates five metrics with the following weights:\n\n")
        f.write("- Site embedding disruption (30%)\n")
        f.write("- Local embedding disruption (25%)\n")
        f.write("- Contact map disruption score (20%)\n")
        f.write("- Number of lost contacts (15%)\n")
        f.write("- Pseudo-pLDDT drop (10%)\n\n")
        f.write("| Score Range | Classification |\n")
        f.write("|-------------|---------------|\n")
        f.write("| > 0.7 | HIGH -- Severe structural disruption |\n")
        f.write("| 0.4 - 0.7 | MODERATE -- Moderate structural disruption |\n")
        f.write("| 0.2 - 0.4 | LOW -- Mild structural perturbation |\n")
        f.write("| < 0.2 | MINIMAL -- Minimal structural impact |\n\n")

        f.write("### Clinical correlation\n\n")
        f.write("These structural disruption predictions should be interpreted alongside:\n\n")
        f.write("- ESM-2 variant pathogenicity scores (from `esm2_variant_scoring.py`)\n")
        f.write("- Known functional mechanisms of each mutation\n")
        f.write("- Co-occurrence patterns from GENIE analysis\n\n")
        f.write("Note that gain-of-function mutations (like IDH2 R140Q) may show lower "
                "structural disruption scores because the protein acquires a new activity "
                "rather than losing structural integrity.\n")

    print(f"  Markdown: {md_path}")

    # Final console summary
    print(f"\n{'='*72}")
    print("STRUCTURAL DISRUPTION ANALYSIS COMPLETE")
    print(f"{'='*72}")
    print(f"\n{'Gene':<10} {'Variant':<10} {'Impact':<10} {'Composite':>10} "
          f"{'SiteDisrupt':>12} {'Contacts':>10} {'pLDDT':>8}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['gene']:<10} {r['variant']:<10} "
              f"{r['structural_impact']['severity']:<10} "
              f"{r['structural_impact']['composite_score']:>10.4f} "
              f"{r['embedding']['site_disruption']:>12.4f} "
              f"{r['contacts']['lost_contacts']:>10d} "
              f"{r['pseudo_plddt']['plddt_drop']:>+8.1f}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results: {RESULTS_DIR}")
    print(f"\nGenerated files:")
    print(f"  - Contact map plots (4)")
    print(f"  - Pseudo-pLDDT plots (4)")
    print(f"  - Embedding disruption plots (4)")
    print(f"  - Combined summary plot")
    print(f"  - PDB files: {pdb_dir}")
    print(f"  - JSON: {json_path}")
    print(f"  - Summary: {md_path}")


if __name__ == "__main__":
    main()
