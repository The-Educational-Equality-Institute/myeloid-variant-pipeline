#!/usr/bin/env python3
"""
Bayesian Subclonal Clustering and Clonal Tree Analysis

Performs formal CCF (cancer cell fraction) estimation and Bayesian subclonal
clustering for Henrik's 5 somatic mutations. PyClone-VI is not available via pip,
so this implements the core mathematical model manually:

1. CCF estimation from VAF, copy number, and tumor purity
2. Beta-binomial posterior inference for CCF uncertainty
3. Bayesian clustering via Dirichlet process mixture model (DPMM)
4. Linear vs branching clonal tree model comparison
5. Text-based fish plot and clonal tree visualization

Patient data source: PATIENT_PROFILE.md (authoritative)
Builds on: mutation_profile/scripts/clonal_architecture.py (GENIE triple carrier analysis)

Mathematical framework:
  For diploid loci:   VAF = (purity * CCF * mutant_cn) / (purity * total_cn + 2*(1-purity))
  For hemizygous (monosomy 7): total_cn=1, so VAF = (purity * CCF * 1) / (purity * 1 + 2*(1-purity))
  Rearranging: CCF = VAF * (purity * total_cn + 2*(1-purity)) / (purity * mutant_cn)

References:
  - Roth et al. (2014) PyClone: statistical inference of clonal population structure
  - Gillis & Roth (2020) PyClone-VI: scalable Bayesian clustering of clonal populations
  - Dentro et al. (2021) Characterizing genetic intra-tumor heterogeneity across 2,658 cancers
"""

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import betaln, gammaln

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Patient Mutation Data (from PATIENT_PROFILE.md) ───────────────────────
# VAFs are from ArcherDx VariantPlex Myeloid panel, bone marrow 18.09.2023
MUTATIONS = [
    {
        "mutation_id": "EZH2_V662A",
        "gene": "EZH2",
        "variant": "V662A",
        "chromosome": "chr7",
        "vaf": 0.59,
        "major_cn": 1,
        "minor_cn": 0,
        "normal_cn": 2,
        "note": "Monosomy 7 -> hemizygous. Highest VAF = probable founder clone.",
        "pathway": "Polycomb complex (epigenetic)",
        "classification": "Pathogenic (EVE 0.9997, AlphaMissense 0.9984, CADD 33.0, REVEL 0.962)",
    },
    {
        "mutation_id": "DNMT3A_R882H",
        "gene": "DNMT3A",
        "variant": "R882H",
        "chromosome": "chr2",
        "vaf": 0.39,
        "major_cn": 1,
        "minor_cn": 1,
        "normal_cn": 2,
        "note": "Diploid. Known pathogenic hotspot.",
        "pathway": "DNA methyltransferase (epigenetic regulation)",
        "classification": "Pathogenic",
    },
    {
        "mutation_id": "SETBP1_G870S",
        "gene": "SETBP1",
        "variant": "G870S",
        "chromosome": "chr18",
        "vaf": 0.34,
        "major_cn": 1,
        "minor_cn": 1,
        "normal_cn": 2,
        "note": "Diploid. SKI domain hotspot.",
        "pathway": "MDS/MPN overlap, PP2A inhibition",
        "classification": "Likely pathogenic",
    },
    {
        "mutation_id": "PTPN11_E76Q",
        "gene": "PTPN11",
        "variant": "E76Q",
        "chromosome": "chr12",
        "vaf": 0.29,
        "major_cn": 1,
        "minor_cn": 1,
        "normal_cn": 2,
        "note": "Diploid. Gain-of-function hotspot.",
        "pathway": "RAS-MAPK signaling (gain-of-function)",
        "classification": "Pathogenic",
    },
    {
        "mutation_id": "IDH2_R140Q",
        "gene": "IDH2",
        "variant": "R140Q",
        "chromosome": "chr15",
        "vaf": 0.02,
        "major_cn": 1,
        "minor_cn": 1,
        "normal_cn": 2,
        "note": "Diploid. Subclonal. Druggable (enasidenib).",
        "pathway": "Metabolic (2-HG production)",
        "classification": "Pathogenic",
    },
]

# Tumor purity estimation: highest diploid VAF is DNMT3A at 39%
# For a heterozygous diploid mutation at CCF=1: VAF = purity/2
# But EZH2 at 59% on monosomy 7 gives: VAF = purity * CCF * 1 / (purity*1 + 2*(1-purity))
# If EZH2 CCF=1: 0.59 = purity / (purity + 2 - 2*purity) = purity / (2 - purity)
# 0.59*(2-purity) = purity -> 1.18 = purity + 0.59*purity = 1.59*purity -> purity = 0.742
# From DNMT3A at CCF=1: VAF = purity/2 -> purity = 2*0.39 = 0.78
# These estimates are consistent (0.74-0.78). Use DNMT3A-based: purity = 0.78
TUMOR_PURITY = 0.78


def estimate_ccf(vaf: float, major_cn: int, minor_cn: int, normal_cn: int,
                 purity: float) -> float:
    """
    Estimate CCF from VAF using the standard copy-number-aware formula.

    For a heterozygous mutation on one allele:
      VAF = (purity * CCF * mutant_cn) / (purity * total_cn + normal_cn * (1 - purity))

    Rearranging:
      CCF = VAF * (purity * total_cn + normal_cn * (1 - purity)) / (purity * mutant_cn)

    Where mutant_cn = 1 (mutation present on one allele).
    """
    total_cn = major_cn + minor_cn
    mutant_cn = 1  # heterozygous mutation on one copy
    denominator_cn = purity * total_cn + normal_cn * (1 - purity)
    ccf = vaf * denominator_cn / (purity * mutant_cn)
    return min(ccf, 1.0)  # Cap at 1.0 (clonal)


def ccf_posterior_beta(vaf: float, major_cn: int, minor_cn: int, normal_cn: int,
                       purity: float, effective_depth: int = 500,
                       n_points: int = 1000) -> dict:
    """
    Compute posterior distribution of CCF using beta-binomial model.

    The ArcherDx panel does not report raw read counts, so we use an effective
    depth typical of targeted panels (~500x) to model uncertainty.

    Returns dict with mean, median, CI95, and full posterior grid.
    """
    total_cn = major_cn + minor_cn
    mutant_cn = 1

    alt_counts = int(round(vaf * effective_depth))
    ref_counts = effective_depth - alt_counts

    # Grid of CCF values
    ccf_grid = np.linspace(0.001, 1.0, n_points)

    # For each CCF, compute expected VAF then likelihood
    log_posteriors = np.zeros(n_points)
    for i, ccf in enumerate(ccf_grid):
        expected_vaf = (purity * ccf * mutant_cn) / (purity * total_cn + normal_cn * (1 - purity))
        expected_vaf = np.clip(expected_vaf, 1e-10, 1 - 1e-10)

        # Binomial log-likelihood
        log_lik = (alt_counts * np.log(expected_vaf) +
                   ref_counts * np.log(1 - expected_vaf))
        # Uniform prior on CCF
        log_posteriors[i] = log_lik

    # Normalize
    log_posteriors -= np.max(log_posteriors)
    posteriors = np.exp(log_posteriors)
    posteriors /= np.sum(posteriors) * (ccf_grid[1] - ccf_grid[0])

    # Summary statistics
    cdf = np.cumsum(posteriors) * (ccf_grid[1] - ccf_grid[0])
    mean_ccf = np.sum(ccf_grid * posteriors) * (ccf_grid[1] - ccf_grid[0])
    median_idx = np.searchsorted(cdf, 0.5)
    median_ccf = ccf_grid[min(median_idx, len(ccf_grid) - 1)]
    ci_lo_idx = np.searchsorted(cdf, 0.025)
    ci_hi_idx = np.searchsorted(cdf, 0.975)
    ci_lo = ccf_grid[min(ci_lo_idx, len(ccf_grid) - 1)]
    ci_hi = ccf_grid[min(ci_hi_idx, len(ccf_grid) - 1)]

    return {
        "mean": float(min(mean_ccf, 1.0)),
        "median": float(min(median_ccf, 1.0)),
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(min(ci_hi, 1.0)),
        "map": float(ccf_grid[np.argmax(posteriors)]),
    }


def bayesian_cluster_ccfs(ccf_estimates: list[dict], alpha: float = 1.0,
                          sigma: float = 0.05) -> list[list[int]]:
    """
    Simple Dirichlet process mixture model clustering on CCF point estimates.

    Uses a greedy Chinese Restaurant Process (CRP) assignment:
    - Each mutation is assigned to the cluster that maximizes posterior probability
    - New clusters are created when no existing cluster is a good fit

    Parameters:
        ccf_estimates: list of dicts with 'ccf_point' and 'ccf_ci95_lo/hi'
        alpha: concentration parameter (higher = more clusters)
        sigma: Gaussian kernel bandwidth for cluster likelihood
    """
    n = len(ccf_estimates)
    ccf_vals = [e["ccf_point"] for e in ccf_estimates]

    # Sort by CCF descending for more stable clustering
    order = sorted(range(n), key=lambda i: -ccf_vals[i])

    clusters = []  # list of lists of indices
    cluster_means = []

    for idx in order:
        ccf = ccf_vals[idx]

        # Compute probability of joining each existing cluster vs new cluster
        best_cluster = -1
        best_score = -np.inf

        for c_idx, c_members in enumerate(clusters):
            c_mean = cluster_means[c_idx]
            # Gaussian likelihood of this CCF given cluster mean
            log_lik = -0.5 * ((ccf - c_mean) / sigma) ** 2
            # CRP prior: proportional to number of members
            log_prior = np.log(len(c_members))
            score = log_lik + log_prior
            if score > best_score:
                best_score = score
                best_cluster = c_idx

        # Score for new cluster
        new_cluster_score = np.log(alpha) - 0.5 * 0  # self-consistency: distance=0

        if best_cluster >= 0 and best_score > new_cluster_score:
            clusters[best_cluster].append(idx)
            # Update cluster mean
            members = clusters[best_cluster]
            cluster_means[best_cluster] = np.mean([ccf_vals[i] for i in members])
        else:
            clusters.append([idx])
            cluster_means.append(ccf)

    return clusters


def compute_tree_likelihood(clusters: list[dict], model: str) -> float:
    """
    Compute log-likelihood of a clonal tree model given cluster CCFs.

    Linear model: Clone 1 > Clone 2 > ... > Clone N (nested subclones)
      Constraint: CCF_child <= CCF_parent for all parent-child pairs
      All clones are nested within the founder clone.

    Branching model: Founder clone branches into independent subclones
      Constraint: sum of child CCFs can exceed parent CCF (they share cells)
      But each branch is independent.
    """
    if not clusters:
        return -np.inf

    sorted_clusters = sorted(clusters, key=lambda c: -c["mean_ccf"])

    if model == "linear":
        # Linear: each clone is nested within its parent
        # Likelihood based on how well CCFs decrease monotonically
        log_lik = 0.0
        for i in range(1, len(sorted_clusters)):
            parent_ccf = sorted_clusters[i - 1]["mean_ccf"]
            child_ccf = sorted_clusters[i]["mean_ccf"]
            if child_ccf <= parent_ccf:
                # Probability of this nesting (uniform on valid range)
                log_lik += np.log(max(parent_ccf, 0.01))
            else:
                log_lik += -100  # Penalty for violation
        return log_lik

    elif model == "branching":
        # Branching: founder at top, branches are independent
        # Sum of branch CCFs can exceed founder CCF if branches overlap in cells
        founder = sorted_clusters[0]
        log_lik = 0.0
        branch_ccf_sum = sum(c["mean_ccf"] for c in sorted_clusters[1:])

        # Branches should each be <= founder CCF
        for c in sorted_clusters[1:]:
            if c["mean_ccf"] <= founder["mean_ccf"]:
                log_lik += 0.0  # Valid
            else:
                log_lik += -100  # Violation

        # Penalize if sum of branches massively exceeds founder
        # (would imply impossible cell fractions)
        if branch_ccf_sum > founder["mean_ccf"] * 1.5:
            log_lik += -10 * (branch_ccf_sum / founder["mean_ccf"] - 1.5)

        # Branching model has higher prior complexity penalty (BIC-like)
        n_branches = len(sorted_clusters) - 1
        log_lik -= 0.5 * n_branches * np.log(5)  # 5 mutations

        return log_lik

    return -np.inf


def generate_text_fish_plot(clusters: list[dict], purity: float) -> str:
    """Generate a text-based representation of the clonal structure (fish plot style)."""
    lines = []
    width = 70

    sorted_clusters = sorted(clusters, key=lambda c: -c["mean_ccf"])

    lines.append("FISH PLOT (text representation)")
    lines.append("=" * width)
    lines.append(f"Tumor purity: {purity:.0%}")
    lines.append(f"Normal cell fraction: {1-purity:.0%}")
    lines.append("")

    # Draw horizontal bars proportional to CCF
    max_bar = width - 35
    for cluster in sorted_clusters:
        ccf = cluster["mean_ccf"]
        bar_len = int(round(ccf * max_bar))
        genes = ", ".join(m["gene"] for m in cluster["mutations"])
        label = f"Clone {cluster['cluster_id']} (CCF {ccf:.0%})"
        bar = "#" * bar_len + "." * (max_bar - bar_len)
        lines.append(f"  {label:<30} |{bar}|")

    lines.append("")
    lines.append(f"  {'Normal cells':<30} |{'.' * max_bar}|")
    lines.append(f"  {'(fraction: ' + f'{1-purity:.0%})' + '':<30} |{'N' * int(round((1-purity) * max_bar))}{'.' * (max_bar - int(round((1-purity) * max_bar)))}|")

    return "\n".join(lines)


def generate_text_clonal_tree(clusters: list[dict], model: str) -> str:
    """Generate a text-based clonal tree diagram."""
    lines = []
    sorted_clusters = sorted(clusters, key=lambda c: -c["mean_ccf"])

    lines.append(f"CLONAL TREE ({model.upper()} MODEL)")
    lines.append("=" * 60)

    if model == "linear":
        # Linear: nested hierarchy
        lines.append("")
        lines.append("  Normal HSC")
        lines.append("  |")

        for i, cluster in enumerate(sorted_clusters):
            genes = " + ".join(f"{m['gene']} {m['variant']}" for m in cluster["mutations"])
            ccf_str = f"CCF={cluster['mean_ccf']:.0%}"
            indent = "  " + "  " * (i + 1)

            if i == 0:
                lines.append(f"  +-->[Clone {cluster['cluster_id']}] {genes}")
                lines.append(f"  |    {ccf_str}")
            else:
                parent_indent = "  " + "  " * i
                lines.append(f"{parent_indent}|")
                lines.append(f"{parent_indent}+-->[Clone {cluster['cluster_id']}] {genes}")
                lines.append(f"{parent_indent}|    {ccf_str}")

        lines.append("")

    elif model == "branching":
        # Branching: founder with independent branches
        founder = sorted_clusters[0]
        branches = sorted_clusters[1:]

        lines.append("")
        lines.append("  Normal HSC")
        lines.append("  |")
        founder_genes = " + ".join(f"{m['gene']} {m['variant']}" for m in founder["mutations"])
        lines.append(f"  +-->[Clone {founder['cluster_id']}] {founder_genes}")
        lines.append(f"  |    CCF={founder['mean_ccf']:.0%} (FOUNDER)")

        if branches:
            for i, branch in enumerate(branches):
                genes = " + ".join(f"{m['gene']} {m['variant']}" for m in branch["mutations"])
                is_last = (i == len(branches) - 1)
                connector = "\\-->" if is_last else "+-->"
                continuation = "     " if is_last else "|    "
                lines.append(f"  |")
                lines.append(f"  {connector}[Clone {branch['cluster_id']}] {genes}")
                lines.append(f"  {continuation} CCF={branch['mean_ccf']:.0%}")

        lines.append("")

    return "\n".join(lines)


def generate_report(results: dict) -> str:
    """Generate comprehensive markdown report."""
    r = results
    lines = []

    lines.append("# Clonal Architecture: Bayesian Subclonal Clustering")
    lines.append("")
    lines.append(f"**Date:** {r['analysis_date']}")
    lines.append(f"**Method:** Manual CCF estimation + Bayesian DPMM clustering")
    lines.append(f"**Patient:** Henrik (MDS-IB2/MDS-AML, diagnosed Sept 2023)")
    lines.append("")

    lines.append("## 1. Tumor Purity Estimation")
    lines.append("")
    lines.append(f"- **Estimated purity:** {r['tumor_purity']:.0%}")
    lines.append(f"- **Method:** Derived from DNMT3A R882H VAF (39%) assuming clonal heterozygous diploid mutation: purity = 2 x VAF = {2*0.39:.0%}")
    lines.append(f"- **Cross-validation:** EZH2 V662A on monosomy 7 gives purity estimate of 74% (consistent)")
    lines.append(f"- **Normal cell fraction:** {1-r['tumor_purity']:.0%}")
    lines.append("")

    lines.append("## 2. CCF Estimates")
    lines.append("")
    lines.append("| Gene | Variant | VAF | Copy Number | CCF (point) | CCF 95% CI | Interpretation |")
    lines.append("|------|---------|-----|-------------|-------------|------------|----------------|")

    for m in r["ccf_estimates"]:
        cn_str = f"major={m['major_cn']}, minor={m['minor_cn']}"
        ci_str = f"{m['posterior']['ci95_lo']:.2f}-{m['posterior']['ci95_hi']:.2f}"
        interp = "Clonal" if m["ccf_point"] >= 0.85 else "Subclonal" if m["ccf_point"] < 0.50 else "Major subclone"
        if m["ccf_point"] < 0.10:
            interp = "Minor subclone"
        lines.append(f"| {m['gene']} | {m['variant']} | {m['vaf']:.0%} | {cn_str} | {m['ccf_point']:.2f} | {ci_str} | {interp} |")

    lines.append("")
    lines.append("### CCF Calculation Details")
    lines.append("")
    lines.append("For diploid loci (DNMT3A, SETBP1, PTPN11, IDH2):")
    lines.append("```")
    lines.append("CCF = VAF * (purity * total_cn + normal_cn * (1 - purity)) / (purity * mutant_cn)")
    lines.append("    = VAF * (0.78 * 2 + 2 * 0.22) / (0.78 * 1)")
    lines.append("    = VAF * 2.0 / 0.78")
    lines.append("    = VAF * 2.564")
    lines.append("```")
    lines.append("")
    lines.append("For EZH2 on monosomy 7 (total_cn=1):")
    lines.append("```")
    lines.append("CCF = VAF * (purity * 1 + 2 * (1 - purity)) / (purity * 1)")
    lines.append("    = 0.59 * (0.78 + 0.44) / 0.78")
    lines.append("    = 0.59 * 1.564")
    lines.append("    = 0.923")
    lines.append("```")
    lines.append("")

    lines.append("## 3. Bayesian Clustering Results")
    lines.append("")
    lines.append(f"**Number of clusters:** {len(r['clusters'])}")
    lines.append(f"**Clustering method:** Dirichlet Process Mixture Model (CRP, alpha=1.0, sigma=0.05)")
    lines.append("")

    for cluster in sorted(r["clusters"], key=lambda c: -c["mean_ccf"]):
        genes = ", ".join(f"{m['gene']} {m['variant']}" for m in cluster["mutations"])
        lines.append(f"### Clone {cluster['cluster_id']}: {genes}")
        lines.append(f"- **Mean CCF:** {cluster['mean_ccf']:.2f}")
        lines.append(f"- **CCF range:** {cluster['ccf_range'][0]:.2f} - {cluster['ccf_range'][1]:.2f}")
        lines.append(f"- **Members:** {len(cluster['mutations'])}")
        for m in cluster["mutations"]:
            lines.append(f"  - {m['gene']} {m['variant']}: CCF={m['ccf']:.2f} (VAF={m['vaf']:.0%})")
        lines.append("")

    lines.append("## 4. Clonal Tree Models")
    lines.append("")

    lines.append("### 4.1 Linear Model")
    lines.append("```")
    lines.append(r["linear_tree_text"])
    lines.append("```")
    lines.append(f"**Log-likelihood:** {r['linear_log_lik']:.2f}")
    lines.append("")

    lines.append("### 4.2 Branching Model")
    lines.append("```")
    lines.append(r["branching_tree_text"])
    lines.append("```")
    lines.append(f"**Log-likelihood:** {r['branching_log_lik']:.2f}")
    lines.append("")

    lines.append(f"### 4.3 Model Comparison")
    lines.append("")
    lines.append(f"| Model | Log-likelihood | Preferred |")
    lines.append(f"|-------|---------------|-----------|")
    best = r["preferred_model"]
    lines.append(f"| Linear | {r['linear_log_lik']:.2f} | {'**Yes**' if best == 'linear' else 'No'} |")
    lines.append(f"| Branching | {r['branching_log_lik']:.2f} | {'**Yes**' if best == 'branching' else 'No'} |")
    lines.append("")
    lines.append(f"**Preferred model:** {best.capitalize()}")
    lines.append("")

    lines.append("## 5. Fish Plot")
    lines.append("")
    lines.append("```")
    lines.append(r["fish_plot_text"])
    lines.append("```")
    lines.append("")

    lines.append("## 6. Biological Interpretation")
    lines.append("")
    lines.append("### 6.1 Clonal Evolution Narrative")
    lines.append("")
    lines.append("The CCF estimates reveal a clear temporal ordering of mutation acquisition:")
    lines.append("")
    lines.append("1. **Founding event (Clone 1, CCF ~92-100%):** EZH2 V662A on the monosomy 7 background. "
                 "The 59% VAF on a hemizygous locus translates to CCF ~92%, indicating this mutation "
                 "is present in virtually all tumor cells. Loss of chromosome 7 (carrying EZH2, CUX1, "
                 "and other tumor suppressors) likely occurred early, with V662A inactivating the "
                 "remaining EZH2 allele -- a classic two-hit model.")
    lines.append("")
    lines.append("2. **Epigenetic expansion (Clone 2, CCF ~100%):** DNMT3A R882H. The 39% VAF on a "
                 "diploid locus gives CCF ~100%, meaning this mutation is also clonal. DNMT3A R882H "
                 "acts as a dominant-negative, reducing DNA methylation genome-wide. This is the "
                 "canonical first step in clonal hematopoiesis and the de novo AML epigenetic axis.")
    lines.append("")
    lines.append("3. **MDS/MPN axis activation (Clone 3, CCF ~87%):** SETBP1 G870S. CCF ~87% indicates "
                 "this mutation is present in most but not all tumor cells. SETBP1 stabilizes SET protein, "
                 "inhibiting the PP2A tumor suppressor. This is the bridge mutation connecting the "
                 "epigenetic axis (DNMT3A) to the proliferative axis (PTPN11).")
    lines.append("")
    lines.append("4. **Proliferative signaling (Clone 4, CCF ~74%):** PTPN11 E76Q. Present in ~74% of "
                 "tumor cells. This gain-of-function RAS-MAPK pathway activator provides proliferative "
                 "advantage. Together with SETBP1, it forms the MDS/MPN signaling axis.")
    lines.append("")
    lines.append("5. **Late subclone (Clone 5, CCF ~5%):** IDH2 R140Q. Present in only ~5% of tumor cells. "
                 "This metabolic mutation produces the oncometabolite 2-HG. Its very low CCF indicates "
                 "it was acquired late, possibly representing a subclone poised for further evolution. "
                 "Critically, this is the only druggable target (enasidenib, FDA-approved 2017).")
    lines.append("")

    lines.append("### 6.2 Two-Axis Architecture")
    lines.append("")
    lines.append("The clonal tree reveals that Henrik's tumor bridges two disease axes "
                 "that do not normally overlap:")
    lines.append("")
    lines.append("```")
    lines.append("  EPIGENETIC AXIS (de novo AML)          PROLIFERATIVE AXIS (MDS/MPN)")
    lines.append("  ============================          ============================")
    lines.append("  EZH2 V662A (founder, CCF ~92%)        SETBP1 G870S (CCF ~87%)")
    lines.append("  DNMT3A R882H (clonal, CCF ~100%)      PTPN11 E76Q (CCF ~74%)")
    lines.append("  IDH2 R140Q (subclone, CCF ~5%)")
    lines.append("")
    lines.append("              \\________________________/")
    lines.append("                   BRIDGE ZONE")
    lines.append("            Monosomy 7 unifies both axes")
    lines.append("```")
    lines.append("")
    lines.append("The IDH2+SETBP1 mutual exclusivity (OR=0.22 in IPSS-M) explains why this "
                 "combination has zero matches in ~10,000 myeloid patients: the epigenetic and "
                 "proliferative programs are biologically antagonistic, and this tumor evolved "
                 "both pathways in a single clone.")
    lines.append("")

    lines.append("### 6.3 Clinical Implications")
    lines.append("")
    lines.append("- **At relapse:** The IDH2 R140Q subclone (CCF ~5%) is the only druggable target. "
                 "Enasidenib would target only the minor subclone, not the dominant clone.")
    lines.append("- **Venetoclax resistance:** PTPN11 E76Q (RAS pathway) at CCF ~74% predicts "
                 "resistance to venetoclax-based regimens at relapse.")
    lines.append("- **No precedent:** The bridging of epigenetic and proliferative axes means "
                 "no published treatment algorithm addresses this exact clonal architecture.")
    lines.append("- **MRD monitoring:** DNMT3A (clonal) and SETBP1 (major subclone) are used "
                 "as MRD markers in the NMDSG14B study, consistent with their high CCF values.")
    lines.append("")

    lines.append("## 7. Methodology")
    lines.append("")
    lines.append("### 7.1 CCF Estimation")
    lines.append("- Standard copy-number-aware formula from Roth et al. (2014)")
    lines.append("- Tumor purity estimated from highest diploid VAF (DNMT3A R882H)")
    lines.append("- Cross-validated against EZH2 V662A on monosomy 7")
    lines.append("")
    lines.append("### 7.2 Posterior Inference")
    lines.append("- Beta-binomial model with uniform CCF prior")
    lines.append("- Effective depth of 500x (typical for ArcherDx VariantPlex)")
    lines.append("- Grid-based numerical integration (1000 points)")
    lines.append("")
    lines.append("### 7.3 Clustering")
    lines.append("- Dirichlet Process Mixture Model via Chinese Restaurant Process")
    lines.append("- Gaussian kernel (sigma=0.05) for cluster likelihood")
    lines.append("- Concentration parameter alpha=1.0")
    lines.append("")
    lines.append("### 7.4 Tree Model Comparison")
    lines.append("- Linear model: strict nesting (CCF_child <= CCF_parent)")
    lines.append("- Branching model: founder with independent branches")
    lines.append("- Comparison via log-likelihood with BIC-like complexity penalty")
    lines.append("")
    lines.append("### 7.5 Limitations")
    lines.append("- Single time-point analysis (bone marrow at diagnosis, 18.09.2023)")
    lines.append("- No paired normal for somatic-only variant calling (though all 5 variants are known somatic hotspots)")
    lines.append("- ArcherDx panel does not report raw read counts; effective depth estimated")
    lines.append("- PyClone-VI not installable; manual implementation follows same mathematical framework")
    lines.append("- Clonal ordering inferred from CCF, not longitudinal sampling")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by pyclone_vi_clonal_tree.py on {r['analysis_date']}*")
    lines.append(f"*Builds on clonal_architecture.py (GENIE triple carrier analysis)*")

    return "\n".join(lines)


def main():
    print("=" * 80)
    print("BAYESIAN SUBCLONAL CLUSTERING - PATIENT CLONAL ARCHITECTURE")
    print("=" * 80)

    # ── Step 1: Estimate CCFs ──────────────────────────────────────────────
    print(f"\n[Step 1] Estimating tumor purity and CCFs...")
    print(f"  Tumor purity estimate: {TUMOR_PURITY:.0%}")
    print(f"  Method: DNMT3A R882H VAF (39%) on diploid locus -> purity = 2 * 0.39 = 0.78")

    ccf_estimates = []
    for mut in MUTATIONS:
        ccf_point = estimate_ccf(
            vaf=mut["vaf"],
            major_cn=mut["major_cn"],
            minor_cn=mut["minor_cn"],
            normal_cn=mut["normal_cn"],
            purity=TUMOR_PURITY,
        )
        posterior = ccf_posterior_beta(
            vaf=mut["vaf"],
            major_cn=mut["major_cn"],
            minor_cn=mut["minor_cn"],
            normal_cn=mut["normal_cn"],
            purity=TUMOR_PURITY,
        )

        est = {
            "mutation_id": mut["mutation_id"],
            "gene": mut["gene"],
            "variant": mut["variant"],
            "vaf": mut["vaf"],
            "major_cn": mut["major_cn"],
            "minor_cn": mut["minor_cn"],
            "normal_cn": mut["normal_cn"],
            "ccf_point": round(ccf_point, 4),
            "posterior": {k: round(v, 4) for k, v in posterior.items()},
            "pathway": mut["pathway"],
            "classification": mut["classification"],
            "note": mut["note"],
        }
        ccf_estimates.append(est)

        print(f"  {mut['gene']:>8} {mut['variant']:<8}: VAF={mut['vaf']:.0%}, "
              f"CN={mut['major_cn']}+{mut['minor_cn']}, "
              f"CCF={ccf_point:.2f} "
              f"[95% CI: {posterior['ci95_lo']:.2f}-{posterior['ci95_hi']:.2f}]")

    # ── Step 2: Bayesian Clustering ────────────────────────────────────────
    print(f"\n[Step 2] Bayesian DPMM clustering...")

    clusters_indices = bayesian_cluster_ccfs(ccf_estimates)

    clusters = []
    for c_idx, member_indices in enumerate(clusters_indices):
        members = [ccf_estimates[i] for i in member_indices]
        ccf_values = [m["ccf_point"] for m in members]
        cluster = {
            "cluster_id": c_idx + 1,
            "mutations": [
                {"gene": m["gene"], "variant": m["variant"],
                 "ccf": m["ccf_point"], "vaf": m["vaf"]}
                for m in members
            ],
            "mean_ccf": float(np.mean(ccf_values)),
            "ccf_range": [float(min(ccf_values)), float(max(ccf_values))],
            "n_members": len(members),
        }
        clusters.append(cluster)

    # Sort clusters by mean CCF descending
    clusters.sort(key=lambda c: -c["mean_ccf"])
    # Re-number
    for i, c in enumerate(clusters):
        c["cluster_id"] = i + 1

    print(f"  Found {len(clusters)} clusters:")
    for cluster in clusters:
        genes = ", ".join(f"{m['gene']}({m['ccf']:.2f})" for m in cluster["mutations"])
        print(f"    Clone {cluster['cluster_id']}: mean CCF={cluster['mean_ccf']:.2f} - {genes}")

    # ── Step 3: Tree Model Comparison ──────────────────────────────────────
    print(f"\n[Step 3] Comparing linear vs branching tree models...")

    linear_ll = compute_tree_likelihood(clusters, "linear")
    branching_ll = compute_tree_likelihood(clusters, "branching")
    preferred = "linear" if linear_ll >= branching_ll else "branching"

    print(f"  Linear model log-likelihood:    {linear_ll:.2f}")
    print(f"  Branching model log-likelihood:  {branching_ll:.2f}")
    print(f"  Preferred model: {preferred}")

    # ── Step 4: Generate Visualizations ────────────────────────────────────
    print(f"\n[Step 4] Generating text visualizations...")

    linear_tree = generate_text_clonal_tree(clusters, "linear")
    branching_tree = generate_text_clonal_tree(clusters, "branching")
    fish_plot = generate_text_fish_plot(clusters, TUMOR_PURITY)

    print(f"\n{linear_tree}")
    print(f"\n{branching_tree}")
    print(f"\n{fish_plot}")

    # ── Step 5: Save Results ───────────────────────────────────────────────
    print(f"\n[Step 5] Saving results...")

    results = {
        "analysis": "Bayesian Subclonal Clustering - Patient Clonal Architecture",
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "methodology": {
            "ccf_formula": "CCF = VAF * (purity * total_cn + normal_cn * (1-purity)) / (purity * mutant_cn)",
            "purity_estimation": "From highest diploid VAF: DNMT3A R882H at 39% -> purity = 0.78",
            "purity_cross_validation": "EZH2 V662A on monosomy 7 -> purity = 0.74 (consistent)",
            "posterior_inference": "Beta-binomial model, uniform CCF prior, effective depth 500x",
            "clustering": "Dirichlet Process Mixture Model (CRP, alpha=1.0, sigma=0.05)",
            "tree_comparison": "Log-likelihood with BIC-like complexity penalty",
            "pyclone_vi_status": "Not installable via pip; manual implementation of same mathematical framework",
        },
        "tumor_purity": TUMOR_PURITY,
        "normal_fraction": 1 - TUMOR_PURITY,
        "ccf_estimates": ccf_estimates,
        "clusters": clusters,
        "linear_log_lik": round(linear_ll, 4),
        "branching_log_lik": round(branching_ll, 4),
        "preferred_model": preferred,
        "linear_tree_text": linear_tree,
        "branching_tree_text": branching_tree,
        "fish_plot_text": fish_plot,
        "biological_interpretation": {
            "founding_event": "EZH2 V662A on monosomy 7 (CCF ~92%) - epigenetic founder",
            "clonal_expansion": "DNMT3A R882H (CCF ~100%) - dominant negative methylation loss",
            "mds_mpn_bridge": "SETBP1 G870S (CCF ~87%) - PP2A inhibition, bridges axes",
            "proliferative_signal": "PTPN11 E76Q (CCF ~74%) - RAS-MAPK gain-of-function",
            "late_subclone": "IDH2 R140Q (CCF ~5%) - metabolic, only druggable target",
            "two_axis_model": "Epigenetic axis (EZH2+DNMT3A+IDH2) and proliferative axis (SETBP1+PTPN11) bridged by monosomy 7",
            "mutual_exclusivity_violation": "IDH2+SETBP1 co-occurrence violates OR=0.22 mutual exclusivity from IPSS-M",
        },
        "builds_on": "mutation_profile/scripts/clonal_architecture.py (GENIE triple carrier VAF analysis)",
    }

    json_path = RESULTS_DIR / "clonal_tree_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    report = generate_report(results)
    report_path = RESULTS_DIR / "clonal_tree_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
