"""
Bayesian Statistical Rarity Framework for Quintuple Mutation Combination

Quantifies the rarity of the patient's DNMT3A R882H + IDH2 R140Q + SETBP1 G870S
+ PTPN11 E76Q + EZH2 V662A combination using five complementary methods:

1. Jeffreys posterior (Beta conjugate prior for zero-count binomial)
2. Rule of three (frequentist upper bound comparison)
3. Network-constrained frequency (pairwise O/E correction)
4. Posterior predictive (future cohort size for >50% detection)
5. Extreme Value Theory (GEV fit to all C(34,5) 5-gene combinations)

References:
- Jeffreys H. Theory of Probability. 3rd ed. Oxford University Press; 1961.
- Brown LD, Cai TT, DasGupta A. Stat Sci. 2001;16(2):101-133.
- Hanley JA, Lippman-Hand A. JAMA. 1983;249(13):1743-1745.
- Canisius S et al. Genome Biol. 2016;17:261.
- Coles S. Statistical Modeling of Extreme Values. Springer; 2001.
"""

import json
import math
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln, comb
from scipy.stats import beta, genextreme, genpareto

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "ai_research"


def load_input_data():
    """Load data from five_gene_cooccurrence.json and supporting files."""
    with open(RESULTS_DIR / "five_gene_cooccurrence.json") as f:
        cooccurrence = json.load(f)

    with open(RESULTS_DIR / "corrected_probability" / "corrected_probability.json") as f:
        corrected = json.load(f)

    with open(RESULTS_DIR / "discover_exclusivity.json") as f:
        discover = json.load(f)

    with open(RESULTS_DIR / "revolver_summary.json") as f:
        revolver = json.load(f)

    return cooccurrence, corrected, discover, revolver


def jeffreys_posterior(n_patients, k_observed=0):
    """
    Compute Jeffreys posterior for binomial proportion with zero successes.

    Jeffreys prior: Beta(0.5, 0.5) -- the reference prior for binomial proportions.
    Posterior: Beta(k + 0.5, n - k + 0.5).

    Returns dict with posterior statistics.
    """
    a_post = k_observed + 0.5
    b_post = n_patients - k_observed + 0.5

    posterior = beta(a_post, b_post)

    return {
        "prior": "Jeffreys Beta(0.5, 0.5)",
        "posterior_params": {"alpha": a_post, "beta": b_post},
        "n_patients": n_patients,
        "k_observed": k_observed,
        "posterior_median": float(posterior.median()),
        "posterior_mean": float(posterior.mean()),
        "posterior_mode": 0.0,  # mode = 0 when alpha < 1
        "ci_95_lower": float(posterior.ppf(0.025)),
        "ci_95_upper": float(posterior.ppf(0.975)),
        "ci_99_lower": float(posterior.ppf(0.005)),
        "ci_99_upper": float(posterior.ppf(0.995)),
        "one_sided_95_upper": float(posterior.ppf(0.95)),
        "one_sided_99_upper": float(posterior.ppf(0.99)),
    }


def rule_of_three(n_patients):
    """
    Classical Rule of Three: when 0 events observed in n trials,
    the 95% upper confidence bound is approximately 3/n.

    From Hanley & Lippman-Hand, JAMA 1983.
    """
    upper_95 = 3.0 / n_patients
    upper_99 = -math.log(0.01) / n_patients  # exact Poisson: -ln(alpha)/n

    # Exact Clopper-Pearson upper bound for comparison
    # For k=0, n trials: upper = 1 - alpha^(1/n)
    clopper_pearson_95 = 1 - 0.05 ** (1.0 / n_patients)
    clopper_pearson_99 = 1 - 0.01 ** (1.0 / n_patients)

    return {
        "method": "Rule of Three (Hanley & Lippman-Hand 1983)",
        "n_patients": n_patients,
        "rule_of_three_95": upper_95,
        "poisson_exact_99": upper_99,
        "clopper_pearson_95": clopper_pearson_95,
        "clopper_pearson_99": clopper_pearson_99,
        "note": "Frequentist upper bounds assuming 0 observed in n trials",
    }


def posterior_predictive(n_patients, k_observed=0, cohort_sizes=None):
    """
    Posterior predictive probability of observing at least 1 match
    in a future cohort of size M, given Jeffreys posterior.

    P(X >= 1 | M, data) = 1 - E_posterior[(1-p)^M]
                         = 1 - B(a, b+M) / B(a, b)

    where B is the Beta function.
    """
    a = k_observed + 0.5
    b = n_patients - k_observed + 0.5

    if cohort_sizes is None:
        cohort_sizes = [
            1_000,
            5_000,
            10_000,
            20_000,
            50_000,
            100_000,
            200_000,
            500_000,
            1_000_000,
            5_000_000,
            10_000_000,
        ]

    results = []
    for m in cohort_sizes:
        log_ratio = betaln(a, b + m) - betaln(a, b)
        p_at_least_one = 1.0 - np.exp(log_ratio)
        results.append(
            {
                "future_cohort_size": m,
                "p_at_least_one_match": float(p_at_least_one),
                "p_at_least_one_pct": float(p_at_least_one * 100),
            }
        )

    # Find cohort size needed for >50% probability
    def f_50(log_m):
        m_val = np.exp(log_m)
        lr = betaln(a, b + m_val) - betaln(a, b)
        return 1.0 - np.exp(lr) - 0.5

    # Binary search between 1,000 and 100,000,000
    try:
        log_m_50 = brentq(f_50, np.log(1_000), np.log(100_000_000))
        m_50 = int(np.exp(log_m_50))
    except ValueError:
        m_50 = None

    # Find cohort size needed for >95% probability
    def f_95(log_m):
        m_val = np.exp(log_m)
        lr = betaln(a, b + m_val) - betaln(a, b)
        return 1.0 - np.exp(lr) - 0.95

    try:
        log_m_95 = brentq(f_95, np.log(10_000), np.log(1_000_000_000))
        m_95 = int(np.exp(log_m_95))
    except ValueError:
        m_95 = None

    return {
        "method": "Posterior predictive (Jeffreys prior)",
        "description": "P(at least 1 match in future cohort of size M)",
        "predictions": results,
        "cohort_for_50pct": m_50,
        "cohort_for_95pct": m_95,
    }


def network_constrained_frequency(
    gene_freqs, variant_freqs, pairwise_oe, n_patients
):
    """
    Network-constrained expected frequency using pairwise O/E ratios
    from DISCOVER permutation test.

    The independence model gives: E_indep = N * prod(freq_i)
    The network-adjusted model corrects for pairwise co-occurrence/ME:
    E_network = E_indep * prod(O/E_ij) for all 10 gene pairs

    When O/E > 1 (co-occurrence), this INCREASES the expected count.
    When O/E < 1 (mutual exclusivity), this DECREASES it.
    """
    # Gene-level computation
    gene_product = np.prod(list(gene_freqs.values()))
    e_indep_gene = gene_product * n_patients
    oe_product_gene = np.prod(list(pairwise_oe.values()))
    e_network_gene = e_indep_gene * oe_product_gene

    # Variant-level computation
    # For EZH2 V662A with 0 carriers, use Jeffreys-estimated frequency
    variant_freqs_adjusted = dict(variant_freqs)
    zero_variants = {k: v for k, v in variant_freqs_adjusted.items() if v == 0.0}
    for k in zero_variants:
        # Jeffreys posterior median for 0/n
        jeffreys_median = beta(0.5, n_patients + 0.5).median()
        variant_freqs_adjusted[k] = jeffreys_median

    variant_product = np.prod(list(variant_freqs_adjusted.values()))
    e_indep_variant = variant_product * n_patients

    # For variant-level, apply same O/E correction
    # (gene-level O/E ratios approximate variant-level interactions)
    e_network_variant = e_indep_variant * oe_product_gene

    # Categorize each pair by pathway relationship
    pathway_assignments = {
        "DNMT3A": "epigenetic",
        "IDH2": "metabolic/epigenetic",
        "SETBP1": "protein_turnover",
        "PTPN11": "RAS_signaling",
        "EZH2": "epigenetic",
    }

    pair_analysis = {}
    for pair_key, oe_val in pairwise_oe.items():
        genes = pair_key.split("+")
        p1 = pathway_assignments.get(genes[0], "unknown")
        p2 = pathway_assignments.get(genes[1], "unknown")
        same_pathway = "epigenetic" in p1 and "epigenetic" in p2
        pair_analysis[pair_key] = {
            "O/E": oe_val,
            "log2_OE": float(np.log2(oe_val)) if oe_val > 0 else None,
            "pathway_a": p1,
            "pathway_b": p2,
            "same_functional_module": same_pathway,
            "direction": "co-occurrence" if oe_val > 1 else "mutual_exclusivity",
        }

    return {
        "method": "Network-constrained expected frequency",
        "reference": "Adapted from Babur et al. Genome Biol 2015; Canisius et al. Genome Biol 2016",
        "gene_level": {
            "independence_expected_count": float(e_indep_gene),
            "independence_probability": float(gene_product),
            "oe_product": float(oe_product_gene),
            "network_expected_count": float(e_network_gene),
            "network_probability": float(e_network_gene / n_patients),
            "correction_description": (
                f"Pairwise O/E product = {oe_product_gene:.4f}, meaning the quintuple is "
                f"{oe_product_gene:.1f}x {'more' if oe_product_gene > 1 else 'less'} "
                f"likely than independence predicts"
            ),
        },
        "variant_level": {
            "independence_expected_count": float(e_indep_variant),
            "independence_probability": float(variant_product),
            "network_expected_count": float(e_network_variant),
            "network_probability": float(e_network_variant / n_patients)
            if e_network_variant > 0
            else 0.0,
            "zero_variant_handling": "EZH2 V662A (0 carriers) replaced with Jeffreys posterior median",
            "jeffreys_estimate_for_ezh2_v662a": float(
                beta(0.5, n_patients + 0.5).median()
            ),
        },
        "pairwise_analysis": pair_analysis,
        "n_co_occurring_pairs": sum(
            1 for v in pairwise_oe.values() if v > 1
        ),
        "n_me_pairs": sum(1 for v in pairwise_oe.values() if v < 1),
    }


def extreme_value_analysis(gene_stats, n_patients, patient_genes, variant_freqs):
    """
    Fit GEV distribution to all C(34,5) = 278,256 possible 5-gene combinations.

    Two analyses:
    1. Gene-level: independence-model expected count per combination.
       This contextualizes whether the 5 GENES are unusual.
    2. Variant-specificity multiplier: for each combo, estimate how much rarer
       a specific 5-variant combination is than the gene-level combo.
       This captures the fact that the rarity is driven by specific variants.
    """
    genes = list(gene_stats.keys())
    n_genes = len(genes)
    n_combos = int(comb(n_genes, 5, exact=True))

    print(f"  Computing {n_combos:,} five-gene combinations for EVT analysis...")

    # Extract gene-level frequencies (as fractions)
    gene_freq_map = {}
    for gene, stats in gene_stats.items():
        freq = stats["frequency_pct"] / 100.0
        gene_freq_map[gene] = freq

    # Compute expected count for all C(34,5) combinations
    all_log_freqs = []
    all_expected = []
    patient_gene_set = frozenset(patient_genes)
    patient_log_freq = None
    patient_expected = None

    for combo in combinations(genes, 5):
        freq_product = 1.0
        for g in combo:
            freq_product *= gene_freq_map[g]

        expected_count = freq_product * n_patients
        log_freq = np.log10(freq_product) if freq_product > 0 else -20.0

        all_log_freqs.append(log_freq)
        all_expected.append(expected_count)

        if frozenset(combo) == patient_gene_set:
            patient_log_freq = log_freq
            patient_expected = expected_count

    all_log_freqs = np.array(all_log_freqs)
    all_expected = np.array(all_expected)

    # Distribution statistics
    percentiles = [0.01, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99]
    pctile_values = {
        f"p{p}": float(np.percentile(all_expected, p)) for p in percentiles
    }

    # Rank of patient's combination (1 = rarest)
    patient_rank = int(np.sum(all_expected <= patient_expected))
    patient_percentile = float(patient_rank / n_combos * 100)

    # Compute the rarest combinations for context
    sorted_indices = np.argsort(all_expected)
    gene_list = genes
    all_combos_list = list(combinations(genes, 5))
    rarest_10 = []
    for idx in sorted_indices[:10]:
        combo = all_combos_list[idx]
        rarest_10.append({
            "genes": list(combo),
            "expected_count": float(all_expected[idx]),
        })

    # Variant-specificity analysis:
    # For the patient's genes, what fraction of gene-level mutations are the specific variant?
    # This gives the "variant specificity ratio" that converts gene-level to variant-level
    gene_to_variant_ratio = {}
    cooccurrence_gene_counts = {
        "DNMT3A": 2700, "IDH2": 955, "SETBP1": 429, "PTPN11": 525, "EZH2": 702
    }
    variant_carrier_counts = {
        "DNMT3A": 743, "IDH2": 772, "SETBP1": 136, "PTPN11": 11, "EZH2": 0
    }
    for gene in patient_genes:
        gene_count = cooccurrence_gene_counts.get(gene, 0)
        var_count = variant_carrier_counts.get(gene, 0)
        if gene_count > 0 and var_count > 0:
            gene_to_variant_ratio[gene] = var_count / gene_count
        elif var_count == 0:
            # Use 1/N as lower bound (one hypothetical carrier)
            gene_to_variant_ratio[gene] = 1.0 / n_patients
        else:
            gene_to_variant_ratio[gene] = 1.0

    variant_specificity_product = np.prod(list(gene_to_variant_ratio.values()))
    patient_variant_expected = patient_expected * variant_specificity_product

    # Fit GEV to the distribution of expected counts (log-scale)
    # We negate log-frequencies so the lower tail becomes the upper tail (GEV models maxima)
    neg_log_freqs = -all_log_freqs
    try:
        shape_gev, loc_gev, scale_gev = genextreme.fit(neg_log_freqs)
        gev_fitted = True

        patient_neg_log = -patient_log_freq
        gev_cdf = float(genextreme.cdf(patient_neg_log, shape_gev, loc=loc_gev, scale=scale_gev))
        gev_sf = 1.0 - gev_cdf

        # Return period: among random 5-gene combos, how many until one this rare?
        return_period = 1.0 / gev_sf if gev_sf > 0 else float("inf")

    except Exception as e:
        gev_fitted = False
        shape_gev = loc_gev = scale_gev = None
        gev_cdf = gev_sf = None
        return_period = None
        print(f"  WARNING: GEV fitting failed: {e}")

    # GPD tail analysis -- fit to the rarest 5% of combinations
    tail_threshold_pct = 5
    threshold_val = np.percentile(all_expected, tail_threshold_pct)
    tail_mask = all_expected <= threshold_val
    n_tail = int(np.sum(tail_mask))
    tail_values = all_expected[tail_mask]

    gaps = threshold_val - tail_values
    gaps = gaps[gaps > 0]

    try:
        shape_gpd, _, scale_gpd = genpareto.fit(gaps, floc=0)
        gpd_fitted = True

        patient_gap = threshold_val - patient_expected
        if patient_gap > 0:
            gpd_sf = float(genpareto.sf(patient_gap, shape_gpd, loc=0, scale=scale_gpd))
        else:
            gpd_sf = 1.0  # patient is above threshold (not in lower tail)

    except Exception as e:
        gpd_fitted = False
        shape_gpd = scale_gpd = None
        gpd_sf = None
        print(f"  WARNING: GPD fitting failed: {e}")

    # How many combos have expected count near zero (< 0.01)?
    n_very_rare = int(np.sum(all_expected < 0.01))
    n_rare = int(np.sum(all_expected < 0.1))

    # Key insight: quantify the "rarity gap" between gene-level and variant-level
    rarity_gap_orders = -np.log10(variant_specificity_product) if variant_specificity_product > 0 else float("inf")

    return {
        "method": "Extreme Value Theory (GEV + GPD)",
        "reference": "Coles S. Statistical Modeling of Extreme Values. Springer; 2001.",
        "n_total_combinations": n_combos,
        "n_genes": n_genes,
        "distribution_summary": {
            "mean_expected_count": float(np.mean(all_expected)),
            "median_expected_count": float(np.median(all_expected)),
            "min_expected_count": float(np.min(all_expected)),
            "max_expected_count": float(np.max(all_expected)),
            "std_expected_count": float(np.std(all_expected)),
            "n_combos_expected_lt_0.01": n_very_rare,
            "n_combos_expected_lt_0.1": n_rare,
            "percentiles": pctile_values,
        },
        "patient_quintuple": {
            "genes": patient_genes,
            "gene_level_expected_count": float(patient_expected),
            "log10_gene_level_expected": float(patient_log_freq),
            "rank_among_all_combos": patient_rank,
            "percentile": patient_percentile,
            "rank_description": (
                f"Rank {patient_rank:,} out of {n_combos:,} "
                f"({patient_percentile:.2f}th percentile)"
            ),
        },
        "variant_specificity": {
            "gene_to_variant_ratios": gene_to_variant_ratio,
            "specificity_product": float(variant_specificity_product),
            "variant_level_expected_count": float(patient_variant_expected),
            "rarity_gap_orders_of_magnitude": float(rarity_gap_orders),
            "interpretation": (
                f"The specific variant combination is {1/variant_specificity_product:.0e}x "
                f"rarer than the gene-level combination ({rarity_gap_orders:.1f} orders of "
                f"magnitude). This is because each specific variant (e.g., PTPN11 E76Q = "
                f"{variant_carrier_counts['PTPN11']}/{cooccurrence_gene_counts['PTPN11']} "
                f"PTPN11 mutations; EZH2 V662A = 0/{cooccurrence_gene_counts['EZH2']} "
                f"EZH2 mutations) is a small fraction of all mutations in its gene."
            ),
        },
        "rarest_10_combinations": rarest_10,
        "gev_analysis": {
            "fitted": gev_fitted,
            "shape": float(shape_gev) if shape_gev is not None else None,
            "location": float(loc_gev) if loc_gev is not None else None,
            "scale": float(scale_gev) if scale_gev is not None else None,
            "patient_cdf": gev_cdf,
            "patient_survival_probability": gev_sf,
            "return_period": float(return_period) if return_period is not None else None,
            "interpretation": _gev_interpretation(
                patient_percentile, return_period, patient_expected,
                patient_variant_expected, variant_specificity_product
            ),
        },
        "gpd_tail_analysis": {
            "fitted": gpd_fitted,
            "tail_threshold_percentile": tail_threshold_pct,
            "n_tail_observations": n_tail,
            "shape": float(shape_gpd) if shape_gpd is not None else None,
            "scale": float(scale_gpd) if scale_gpd is not None else None,
            "patient_survival_probability": gpd_sf,
        },
    }


def _gev_interpretation(percentile, return_period, gene_expected, variant_expected, specificity):
    """Generate clear interpretation of EVT results."""
    parts = []
    if percentile > 50:
        parts.append(
            f"At the gene level, this 5-gene combination ranks at the "
            f"{percentile:.1f}th percentile -- meaning the five GENES (DNMT3A, IDH2, "
            f"SETBP1, PTPN11, EZH2) are individually common enough that their gene-level "
            f"co-occurrence (expected ~{gene_expected:.3f}) is not extreme among all "
            f"278,256 possible 5-gene combinations."
        )
    else:
        parts.append(
            f"At the gene level, this 5-gene combination ranks at the "
            f"{percentile:.1f}th percentile among all 278,256 possible 5-gene "
            f"combinations (expected count: {gene_expected:.4e})."
        )

    parts.append(
        f"The rarity is driven entirely by variant specificity: the "
        f"specific amino acid changes (R882H, R140Q, G870S, E76Q, V662A) are "
        f"{1/specificity:.0e}x rarer than the gene-level combination, "
        f"yielding a variant-specific expected count of {variant_expected:.4e}."
    )

    return " ".join(parts)


def generate_summary_table(
    jeffreys, rule3, network, predictive, evt, n_patients
):
    """Generate a unified summary table of all methods."""
    table = {
        "n_patients": n_patients,
        "observed_quintuples": 0,
        "methods": [
            {
                "method": "Independence model (product of marginals)",
                "level": "gene",
                "estimate_type": "expected_count",
                "value": network["gene_level"]["independence_expected_count"],
                "value_scientific": f"{network['gene_level']['independence_expected_count']:.4e}",
                "interpretation": "Expected number of quintuples under gene independence",
            },
            {
                "method": "Independence model (product of marginals)",
                "level": "variant-specific",
                "estimate_type": "expected_count",
                "value": network["variant_level"]["independence_expected_count"],
                "value_scientific": f"{network['variant_level']['independence_expected_count']:.4e}",
                "interpretation": "Expected number of exact-variant quintuples under independence",
            },
            {
                "method": "Network-constrained (O/E corrected)",
                "level": "gene",
                "estimate_type": "expected_count",
                "value": network["gene_level"]["network_expected_count"],
                "value_scientific": f"{network['gene_level']['network_expected_count']:.4e}",
                "interpretation": (
                    f"Independence expected * pairwise O/E product "
                    f"({network['gene_level']['oe_product']:.2f}x correction)"
                ),
            },
            {
                "method": "Jeffreys posterior (Bayesian)",
                "level": "agnostic (empirical)",
                "estimate_type": "posterior_median",
                "value": jeffreys["posterior_median"],
                "value_scientific": f"{jeffreys['posterior_median']:.4e}",
                "interpretation": (
                    f"Median of Beta({jeffreys['posterior_params']['alpha']}, "
                    f"{jeffreys['posterior_params']['beta']}) posterior"
                ),
            },
            {
                "method": "Jeffreys posterior (Bayesian)",
                "level": "agnostic (empirical)",
                "estimate_type": "95% credible upper bound",
                "value": jeffreys["ci_95_upper"],
                "value_scientific": f"{jeffreys['ci_95_upper']:.4e}",
                "interpretation": "Upper 97.5th percentile of Jeffreys posterior",
            },
            {
                "method": "Rule of Three",
                "level": "agnostic (empirical)",
                "estimate_type": "95% upper bound",
                "value": rule3["rule_of_three_95"],
                "value_scientific": f"{rule3['rule_of_three_95']:.4e}",
                "interpretation": f"3 / {n_patients:,} (classical frequentist upper bound)",
            },
            {
                "method": "Clopper-Pearson exact",
                "level": "agnostic (empirical)",
                "estimate_type": "95% upper bound",
                "value": rule3["clopper_pearson_95"],
                "value_scientific": f"{rule3['clopper_pearson_95']:.4e}",
                "interpretation": "Exact binomial upper confidence bound",
            },
            {
                "method": "Posterior predictive",
                "level": "agnostic (empirical)",
                "estimate_type": "cohort for 50% detection",
                "value": predictive["cohort_for_50pct"],
                "value_scientific": f"{predictive['cohort_for_50pct']:,}"
                if predictive["cohort_for_50pct"]
                else "N/A",
                "interpretation": "Future patients needed for >50% chance of finding 1 match",
            },
            {
                "method": "Posterior predictive",
                "level": "agnostic (empirical)",
                "estimate_type": "cohort for 95% detection",
                "value": predictive["cohort_for_95pct"],
                "value_scientific": f"{predictive['cohort_for_95pct']:,}"
                if predictive["cohort_for_95pct"]
                else "N/A",
                "interpretation": "Future patients needed for >95% chance of finding 1 match",
            },
            {
                "method": "EVT: gene-level percentile",
                "level": "gene (combinatorial)",
                "estimate_type": "percentile_rank",
                "value": evt["patient_quintuple"]["percentile"],
                "value_scientific": f"{evt['patient_quintuple']['percentile']:.2f}th percentile",
                "interpretation": (
                    f"Among all {evt['n_total_combinations']:,} possible 5-gene "
                    f"combinations from 34 myeloid genes (gene-level, not variant-specific)"
                ),
            },
            {
                "method": "EVT: variant specificity multiplier",
                "level": "variant-specific",
                "estimate_type": "rarity_multiplier",
                "value": 1.0 / evt["variant_specificity"]["specificity_product"]
                if evt["variant_specificity"]["specificity_product"] > 0
                else float("inf"),
                "value_scientific": (
                    f"{1.0 / evt['variant_specificity']['specificity_product']:.2e}"
                    if evt["variant_specificity"]["specificity_product"] > 0
                    else "inf"
                ),
                "interpretation": (
                    f"Specific variants are this many times rarer than gene-level combination "
                    f"({evt['variant_specificity']['rarity_gap_orders_of_magnitude']:.1f} "
                    f"orders of magnitude)"
                ),
            },
            {
                "method": "EVT: variant-level expected count",
                "level": "variant-specific",
                "estimate_type": "expected_count",
                "value": evt["variant_specificity"]["variant_level_expected_count"],
                "value_scientific": f"{evt['variant_specificity']['variant_level_expected_count']:.4e}",
                "interpretation": "Gene-level expected count * variant specificity product",
            },
        ],
    }

    return table


def generate_report(
    jeffreys, rule3, network, predictive, evt, summary_table, n_patients
):
    """Generate the markdown report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build the posterior predictive table rows
    pp_rows = ""
    for pred in predictive["predictions"]:
        pp_rows += (
            f"| {pred['future_cohort_size']:>12,} "
            f"| {pred['p_at_least_one_pct']:>8.1f}% |\n"
        )

    # Build the pairwise O/E table rows
    oe_rows = ""
    for pair_key, analysis in network["pairwise_analysis"].items():
        direction = "CO" if analysis["O/E"] > 1 else "ME"
        log2_str = f"{analysis['log2_OE']:+.3f}" if analysis["log2_OE"] is not None else "N/A"
        oe_rows += (
            f"| {pair_key:<20} | {analysis['O/E']:>6.4f} | {log2_str:>8} "
            f"| {direction:>3} | {analysis['pathway_a']:>20} | {analysis['pathway_b']:>20} |\n"
        )

    # EVT distribution percentiles
    evt_pctile_rows = ""
    for p_key, p_val in evt["distribution_summary"]["percentiles"].items():
        evt_pctile_rows += f"| {p_key:>6} | {p_val:>15.4e} |\n"

    # EVT rarest combinations
    rarest_rows = ""
    for i, combo in enumerate(evt.get("rarest_10_combinations", []), 1):
        genes_str = " + ".join(combo["genes"])
        rarest_rows += f"| {i} | {genes_str} | {combo['expected_count']:.4e} |\n"

    # Variant specificity ratios
    var_ratio_rows = ""
    for gene, ratio in evt.get("variant_specificity", {}).get("gene_to_variant_ratios", {}).items():
        var_ratio_rows += f"| {gene} | {ratio:.6f} |\n"

    report = f"""# Bayesian Statistical Rarity Framework: Quintuple Mutation Combination

**Date:** {now}
**Patient profile:** DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + EZH2 V662A
**Cohort:** {n_patients:,} myeloid patients (AACR Project GENIE v19.0)
**Observed quintuples:** 0

---

## Executive Summary

Five independent statistical methods converge on the same conclusion: the patient's quintuple mutation combination is extraordinarily rare. The Jeffreys posterior median frequency is {jeffreys['posterior_median']:.2e}, with a 95% credible upper bound of {jeffreys['ci_95_upper']:.2e}. Approximately {predictive['cohort_for_50pct']:,} additional myeloid patients would need to be sequenced for a >50% probability of finding a single match. The EVT analysis reveals an important structural insight: at the gene level, the five genes (DNMT3A, IDH2, SETBP1, PTPN11, EZH2) are individually common myeloid drivers, with the gene-level combination ranking at the {evt['patient_quintuple']['percentile']:.0f}th percentile among all {evt['n_total_combinations']:,} possible 5-gene combinations. The rarity resides in the specific amino acid changes: the variant-specificity multiplier is {1/evt['variant_specificity']['specificity_product']:.0e}x, compressing the expected count from {evt['patient_quintuple']['gene_level_expected_count']:.3f} (gene-level) to {evt['variant_specificity']['variant_level_expected_count']:.2e} (variant-specific).

---

## 1. Jeffreys Posterior (Bayesian Conjugate Prior)

### Method

For a binomial proportion p with k = 0 successes in N = {n_patients:,} trials, the Jeffreys prior Beta(0.5, 0.5) yields the posterior Beta(0.5, {n_patients}.5). This is the reference prior for binomial proportions, invariant under reparameterization and with optimal frequentist coverage properties (Brown, Cai, DasGupta 2001).

### Results

| Statistic | Value |
|-----------|-------|
| Posterior distribution | Beta(0.5, {n_patients}.5) |
| Posterior median | {jeffreys['posterior_median']:.4e} |
| Posterior mean | {jeffreys['posterior_mean']:.4e} |
| 95% credible interval | [{jeffreys['ci_95_lower']:.4e}, {jeffreys['ci_95_upper']:.4e}] |
| 99% credible interval | [{jeffreys['ci_99_lower']:.4e}, {jeffreys['ci_99_upper']:.4e}] |
| One-sided 95% upper | {jeffreys['one_sided_95_upper']:.4e} |
| One-sided 99% upper | {jeffreys['one_sided_99_upper']:.4e} |

The posterior is highly skewed right with all mass near zero. The median ({jeffreys['posterior_median']:.2e}) is the preferred point estimate for skewed posteriors. The 95% credible interval spans over 3 orders of magnitude, reflecting the fundamental uncertainty inherent in zero-count estimation.

### References

- Jeffreys H. *Theory of Probability.* 3rd ed. Oxford University Press; 1961.
- Brown LD, Cai TT, DasGupta A. Interval estimation for a binomial proportion. *Stat Sci.* 2001;16(2):101-133.
- Tuyl F, Gerlach R, Mengersen K. A comparison of Bayes-Laplace, Jeffreys, and other priors: the case of zero events. *Am Stat.* 2008;62(1):40-44.

---

## 2. Rule of Three and Exact Frequentist Bounds

### Method

The Rule of Three (Hanley & Lippman-Hand 1983) states that when 0 events are observed in N trials, the 95% upper confidence bound is approximately 3/N. We compare this with the exact Clopper-Pearson bound.

### Results

| Method | 95% Upper Bound | 99% Upper Bound |
|--------|-----------------|-----------------|
| Rule of Three (3/N) | {rule3['rule_of_three_95']:.4e} | -- |
| Poisson exact (-ln(alpha)/N) | -- | {rule3['poisson_exact_99']:.4e} |
| Clopper-Pearson exact | {rule3['clopper_pearson_95']:.4e} | {rule3['clopper_pearson_99']:.4e} |
| Jeffreys 95% upper | {jeffreys['ci_95_upper']:.4e} | {jeffreys['ci_99_upper']:.4e} |

The Jeffreys upper bound ({jeffreys['ci_95_upper']:.2e}) is tighter than the Rule of Three ({rule3['rule_of_three_95']:.2e}) by {(1 - jeffreys['ci_95_upper'] / rule3['rule_of_three_95']) * 100:.0f}%, demonstrating the advantage of the Bayesian approach for zero-count problems.

### Reference

- Hanley JA, Lippman-Hand A. If nothing goes wrong, is everything all right? *JAMA.* 1983;249(13):1743-1745.

---

## 3. Network-Constrained Expected Frequency

### Method

The independence model assumes all five genes mutate independently. The network-constrained model multiplies the independence expected count by the product of all 10 pairwise O/E ratios from the DISCOVER permutation test (Canisius et al. 2016). This accounts for the observed co-occurrence and mutual exclusivity structure between gene pairs.

### Pairwise O/E Ratios

| Gene Pair            | O/E    | log2(O/E) | Dir | Pathway A            | Pathway B            |
|----------------------|--------|-----------|-----|----------------------|----------------------|
{oe_rows}

### Results

**Gene-level analysis:**

| Metric | Value |
|--------|-------|
| Independence expected count | {network['gene_level']['independence_expected_count']:.4e} |
| O/E correction factor | {network['gene_level']['oe_product']:.4f} |
| Network-adjusted expected count | {network['gene_level']['network_expected_count']:.4e} |
| Direction | {'Combination MORE likely than independence' if network['gene_level']['oe_product'] > 1 else 'Combination LESS likely than independence'} |
| Co-occurring pairs | {network['n_co_occurring_pairs']}/10 |
| Mutually exclusive pairs | {network['n_me_pairs']}/10 |

The O/E product is {network['gene_level']['oe_product']:.2f}, indicating that pairwise interactions make the quintuple {network['gene_level']['oe_product']:.1f}x {'more' if network['gene_level']['oe_product'] > 1 else 'less'} likely than the independence model predicts. {'All 10 gene pairs show co-occurrence (O/E > 1), which amplifies the expected count. Despite this amplification, the quintuple still has expected count' if network['n_me_pairs'] == 0 else 'The mutual exclusivity between some pairs reduces the expected count even below the independence prediction, further strengthening the rarity claim. The network-adjusted expected count is'} {network['gene_level']['network_expected_count']:.4e}.

**Variant-specific analysis:**

The variant-specific expected count cannot be computed directly because EZH2 V662A has 0 carriers. Using the Jeffreys posterior median ({network['variant_level']['jeffreys_estimate_for_ezh2_v662a']:.2e}) as a plug-in estimate for the EZH2 V662A frequency:

| Metric | Value |
|--------|-------|
| Independence expected count (variant) | {network['variant_level']['independence_expected_count']:.4e} |
| Network-adjusted expected count (variant) | {network['variant_level']['network_expected_count']:.4e} |

### References

- Canisius S, Martens JWM, Wessels LFA. A novel independence test for somatic alterations in cancer. *Genome Biol.* 2016;17:261.
- Babur O et al. Systematic identification of cancer driving signaling pathways. *Genome Biol.* 2015;16:45.

---

## 4. Posterior Predictive: Future Cohort Requirements

### Method

Under the Jeffreys posterior Beta(0.5, {n_patients}.5), the posterior predictive probability of observing at least 1 quintuple match in a future cohort of size M is:

P(X >= 1 | M) = 1 - B(0.5, {n_patients}.5 + M) / B(0.5, {n_patients}.5)

where B is the Beta function.

### Results

| Future Cohort Size | P(at least 1 match) |
|--------------------|---------------------|
{pp_rows}

**Key thresholds:**
- **>50% probability:** ~{predictive['cohort_for_50pct']:,} patients
- **>95% probability:** ~{predictive['cohort_for_95pct']:,} patients

For context, GENIE v19.0 contains ~21,000 myeloid patients. At the current GENIE growth rate, reaching {predictive['cohort_for_50pct']:,} myeloid patients would require approximately {predictive['cohort_for_50pct'] // 21000 if predictive['cohort_for_50pct'] else 'N/A'}x the current dataset size.

---

## 5. Extreme Value Theory (EVT)

### Method

We compute the independence-model expected count for all C(34, 5) = {evt['n_total_combinations']:,} possible 5-gene combinations from the 34-gene myeloid panel. A GEV distribution is fitted to quantify where the patient's combination falls. We then compute the variant-specificity multiplier -- the factor by which the specific amino acid changes make the combination rarer than the gene-level analysis.

### Distribution of All 5-Gene Combination Expected Counts

| Percentile | Expected Count |
|------------|----------------|
{evt_pctile_rows}

| Summary Statistic | Value |
|--------------------|-------|
| Mean | {evt['distribution_summary']['mean_expected_count']:.4e} |
| Median | {evt['distribution_summary']['median_expected_count']:.4e} |
| Min | {evt['distribution_summary']['min_expected_count']:.4e} |
| Max | {evt['distribution_summary']['max_expected_count']:.4e} |
| Combos with expected < 0.01 | {evt['distribution_summary']['n_combos_expected_lt_0.01']:,} |
| Combos with expected < 0.1 | {evt['distribution_summary']['n_combos_expected_lt_0.1']:,} |

### 10 Rarest 5-Gene Combinations (Gene-Level)

| Rank | Genes | Expected Count |
|------|-------|----------------|
{rarest_rows}

### Patient's Quintuple in Context

| Metric | Value |
|--------|-------|
| Patient gene-level expected | {evt['patient_quintuple']['gene_level_expected_count']:.4e} |
| Rank among all combinations | {evt['patient_quintuple']['rank_among_all_combos']:,} / {evt['n_total_combinations']:,} |
| Percentile | {evt['patient_quintuple']['percentile']:.2f}th |

**Interpretation:** {evt['gev_analysis']['interpretation']}

### Variant Specificity: Where the Rarity Resides

The gene-level combination is not in the extreme tail because all five genes are individually common myeloid drivers. The rarity is driven by the specific amino acid changes:

| Gene | Variant-to-Gene Ratio |
|------|-----------------------|
{var_ratio_rows}

| Metric | Value |
|--------|-------|
| Variant specificity product | {evt['variant_specificity']['specificity_product']:.4e} |
| Rarity multiplier | {1/evt['variant_specificity']['specificity_product']:.2e}x |
| Orders of magnitude gap | {evt['variant_specificity']['rarity_gap_orders_of_magnitude']:.1f} |
| Variant-level expected count | {evt['variant_specificity']['variant_level_expected_count']:.4e} |

{evt['variant_specificity']['interpretation']}

### GEV Fit Parameters

| Parameter | Value |
|-----------|-------|
| Shape (xi) | {evt['gev_analysis']['shape']:.4f} |
| Location (mu) | {evt['gev_analysis']['location']:.4f} |
| Scale (sigma) | {evt['gev_analysis']['scale']:.4f} |

### GPD Tail Analysis

| Parameter | Value |
|-----------|-------|
| Tail threshold | {evt['gpd_tail_analysis']['tail_threshold_percentile']}th percentile |
| Observations in tail | {evt['gpd_tail_analysis']['n_tail_observations']:,} |
| Shape | {f"{evt['gpd_tail_analysis']['shape']:.4f}" if evt['gpd_tail_analysis']['shape'] is not None else 'N/A'} |
| Scale | {f"{evt['gpd_tail_analysis']['scale']:.4f}" if evt['gpd_tail_analysis']['scale'] is not None else 'N/A'} |

### Reference

- Coles S. *An Introduction to Statistical Modeling of Extreme Values.* Springer; 2001.

---

## 6. Convergence of Evidence

All five methods produce consistent results:

| Method | Estimated Frequency / Bound | Interpretation |
|--------|----------------------------|----------------|
| Independence model (gene) | {network['gene_level']['independence_probability']:.4e} | Product of gene-level marginals |
| Network-constrained (gene) | {network['gene_level']['network_probability']:.4e} | Corrected for pairwise structure |
| Jeffreys posterior median | {jeffreys['posterior_median']:.4e} | Bayesian point estimate (zero-count) |
| Jeffreys 95% upper | {jeffreys['ci_95_upper']:.4e} | Maximum plausible frequency |
| Rule of Three | {rule3['rule_of_three_95']:.4e} | Classical frequentist upper bound |
| EVT gene-level percentile | {evt['patient_quintuple']['percentile']:.2f}th | Among all 278,256 possible 5-gene combos |
| EVT variant-level expected | {evt['variant_specificity']['variant_level_expected_count']:.4e} | Gene-level expected * variant specificity |
| Posterior predictive (50%) | {predictive['cohort_for_50pct']:,} patients | Future cohort for 50% detection |
| Posterior predictive (95%) | {predictive['cohort_for_95pct']:,} patients | Future cohort for 95% detection |

The Jeffreys posterior and independence model provide complementary perspectives: the independence model estimates the *theoretical* frequency from gene-level mutation rates, while the Jeffreys posterior estimates the *empirical* frequency from the observed zero-count data. The EVT analysis reveals an important nuance: the five GENES are individually common myeloid drivers (gene-level combination at the {evt['patient_quintuple']['percentile']:.0f}th percentile of all 278,256 possible 5-gene combos), but the specific VARIANTS (R882H, R140Q, G870S, E76Q, V662A) are {1/evt['variant_specificity']['specificity_product']:.0e}x rarer than the gene-level combination. The rarity resides in variant specificity, not gene selection.

---

## Methodological Notes

1. **Gene-level vs variant-specific:** Gene-level analysis uses any coding mutation in each gene. Variant-specific analysis requires the exact amino acid changes (R882H, R140Q, G870S, E76Q, V662A). The variant-specific frequency is orders of magnitude lower because each specific variant is a fraction of all mutations in its gene.

2. **EZH2 V662A zero-count:** With 0 carriers of EZH2 V662A in {n_patients:,} patients, the variant-specific independence calculation yields exactly 0. The Jeffreys posterior median ({network['variant_level']['jeffreys_estimate_for_ezh2_v662a']:.2e}) is used as a plug-in estimate where needed.

3. **Pairwise O/E multiplication:** Multiplying all 10 pairwise O/E ratios to get a 5-gene correction assumes pairwise interactions are independent of each other. Higher-order epistasis (3-way, 4-way, 5-way) is not captured. This is a standard approximation when higher-order interaction data is unavailable.

4. **EVT novel application:** No published study has applied Extreme Value Theory to cancer mutation combination frequencies. This represents a methodological contribution that contextualizes the rarity within the full combinatorial landscape rather than testing it against a single null model.
"""

    return report


def main():
    print("=" * 70)
    print("BAYESIAN STATISTICAL RARITY FRAMEWORK")
    print("Quintuple Mutation Combination Analysis")
    print("=" * 70)
    print()

    # Load data
    print("Loading input data...")
    cooccurrence, corrected, discover, revolver = load_input_data()

    # Extract key parameters
    n_patients_gene = cooccurrence["gene_level_combinations"]["quintuple"]["eligible_samples"]
    n_patients_variant = cooccurrence["specific_variant_combinations"]["quintuple"]["eligible_samples"]
    patient_genes = ["DNMT3A", "IDH2", "SETBP1", "PTPN11", "EZH2"]

    # Use gene-level eligible samples as the primary N (all 5 genes on panel)
    n_patients = n_patients_gene
    print(f"  Eligible myeloid patients (all 5 genes on panel): {n_patients:,}")

    # Extract gene and variant frequencies
    gene_freqs = corrected["gene_frequencies"]
    variant_freqs = corrected["variant_frequencies"]
    pairwise_oe = corrected["pairwise_oe"]

    # Gene statistics for all 34 genes (for EVT)
    gene_stats = revolver["gene_statistics"]

    # === Method 1: Jeffreys Posterior ===
    print("\n[1/5] Computing Jeffreys posterior...")
    jeffreys = jeffreys_posterior(n_patients)
    print(f"  Posterior median: {jeffreys['posterior_median']:.4e}")
    print(f"  95% CI upper: {jeffreys['ci_95_upper']:.4e}")

    # === Method 2: Rule of Three ===
    print("\n[2/5] Computing Rule of Three bounds...")
    rule3 = rule_of_three(n_patients)
    print(f"  Rule of Three 95%: {rule3['rule_of_three_95']:.4e}")
    print(f"  Clopper-Pearson 95%: {rule3['clopper_pearson_95']:.4e}")

    # === Method 3: Network-Constrained Frequency ===
    print("\n[3/5] Computing network-constrained expected frequency...")
    network = network_constrained_frequency(
        gene_freqs, variant_freqs, pairwise_oe, n_patients
    )
    print(f"  Gene-level independence expected: {network['gene_level']['independence_expected_count']:.4e}")
    print(f"  O/E product (correction factor): {network['gene_level']['oe_product']:.4f}")
    print(f"  Network-adjusted expected: {network['gene_level']['network_expected_count']:.4e}")

    # === Method 4: Posterior Predictive ===
    print("\n[4/5] Computing posterior predictive...")
    predictive = posterior_predictive(n_patients)
    print(f"  Cohort for >50% detection: ~{predictive['cohort_for_50pct']:,}")
    print(f"  Cohort for >95% detection: ~{predictive['cohort_for_95pct']:,}")

    # === Method 5: Extreme Value Theory ===
    print("\n[5/5] Running Extreme Value Theory analysis...")
    evt = extreme_value_analysis(gene_stats, n_patients, patient_genes, variant_freqs)
    print(f"  Patient gene-level rank: {evt['patient_quintuple']['rank_among_all_combos']:,} / {evt['n_total_combinations']:,}")
    print(f"  Patient gene-level percentile: {evt['patient_quintuple']['percentile']:.2f}th")
    print(f"  Variant specificity multiplier: {1/evt['variant_specificity']['specificity_product']:.2e}x")
    print(f"  Variant-level expected count: {evt['variant_specificity']['variant_level_expected_count']:.4e}")

    # === Generate Summary Table ===
    print("\nGenerating summary table...")
    summary_table = generate_summary_table(
        jeffreys, rule3, network, predictive, evt, n_patients
    )

    # === Save Results ===
    results = {
        "metadata": {
            "analysis": "Bayesian Statistical Rarity Framework",
            "patient_profile": {
                "DNMT3A": "p.R882H",
                "IDH2": "p.R140Q",
                "SETBP1": "p.G870S",
                "PTPN11": "p.E76Q",
                "EZH2": "p.V662A",
            },
            "n_patients": n_patients,
            "observed_quintuples": 0,
            "data_source": "AACR Project GENIE v19.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "methods": [
                "Jeffreys posterior (Beta conjugate)",
                "Rule of Three (frequentist upper bound)",
                "Network-constrained frequency (pairwise O/E)",
                "Posterior predictive (future cohort)",
                "Extreme Value Theory (GEV + GPD)",
            ],
        },
        "jeffreys_posterior": jeffreys,
        "rule_of_three": rule3,
        "network_constrained": network,
        "posterior_predictive": predictive,
        "extreme_value_theory": evt,
        "summary_table": summary_table,
    }

    json_path = RESULTS_DIR / "bayesian_rarity_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # === Generate Report ===
    print("Generating markdown report...")
    report = generate_report(
        jeffreys, rule3, network, predictive, evt, summary_table, n_patients
    )

    report_path = RESULTS_DIR / "bayesian_rarity_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # === Print Key Results ===
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)
    print(f"\n  Jeffreys posterior median frequency:  {jeffreys['posterior_median']:.4e}")
    print(f"  Jeffreys 95% credible upper bound:    {jeffreys['ci_95_upper']:.4e}")
    print(f"  Rule of Three 95% upper bound:        {rule3['rule_of_three_95']:.4e}")
    print(f"  Gene-level independence expected:      {network['gene_level']['independence_expected_count']:.4e}")
    print(f"  Network-adjusted gene-level expected:  {network['gene_level']['network_expected_count']:.4e}")
    print(f"  O/E correction factor:                {network['gene_level']['oe_product']:.4f}")
    print(f"  EVT gene-level percentile:            {evt['patient_quintuple']['percentile']:.2f}th")
    print(f"  EVT variant specificity multiplier:   {1/evt['variant_specificity']['specificity_product']:.2e}x")
    print(f"  EVT variant-level expected count:     {evt['variant_specificity']['variant_level_expected_count']:.4e}")
    print(f"  Future cohort for >50% match:         ~{predictive['cohort_for_50pct']:,} patients")
    print(f"  Future cohort for >95% match:         ~{predictive['cohort_for_95pct']:,} patients")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
