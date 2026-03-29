#!/usr/bin/env python3
"""
Disease-stratified Fisher's exact tests for DNMT3A, IDH2, PTPN11, SETBP1 co-occurrence.

Mutation co-occurrence patterns may differ between AML, MDS, and MDS/MPN.
This script stratifies GENIE samples by ONCOTREE_CODE, then runs all 6 pairwise
Fisher's exact tests within each disease group.

Data sources:
  - GENIE v19 data_clinical_sample.txt (ONCOTREE_CODE per sample)
  - GENIE v19 data_mutations_extended.txt (somatic mutations)

Author: Automated analysis
Date: 2026-03-19
"""

import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLINICAL_FILE = DATA_DIR / "data_clinical_sample.txt"
MAF_FILE = DATA_DIR / "data_mutations_extended.txt"

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]

PROTEIN_ALTERING = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Nonstop_Mutation",
    "Translation_Start_Site",
}

# ── Disease group classification ───────────────────────────────────────────
# AML subtypes (OncoTree)
AML_CODES = {
    "AML", "AMLBCRABL1", "AMLCBFBMYH11", "AMLCEBPA", "AMLDEKNUP214",
    "AMLGATA2MECOM", "AMLMD", "AMLMLLT3KMT2A", "AMLMRC", "AMLNOS",
    "AMLNPM1", "AMLRARA", "AMLRBM15MKL1", "AMLRGA", "AMLRUNX1",
    "AMLRUNX1RUNX1T1",
    "AMML",            # Acute myelomonocytic leukemia
    "APLPMLRARA",      # APL
    "APMF",            # Acute panmyelosis with myelofibrosis
    "AUL",             # Acute undifferentiated leukemia
    "MS",              # Myeloid sarcoma
}

MDS_CODES_PREFIX = {"MDS"}  # MDS, MDSEB, MDSEB1, MDSEB2, etc. (but NOT MDS/MPN)

MDSMPN_CODES = {
    "CMML", "CMML0", "CMML1", "CMML2",
    "JMML",
    "MDS/MPN",
    "MDSMPNRST",       # MDS/MPN with ring sideroblasts and thrombocytosis
    "MDSMPNU",          # MDS/MPN, unclassifiable
}

MPN_CODES_PREFIX = {"MPN", "CML", "CMLBCRABL1", "SM"}
MPN_EXACT = {"MPNST", "MPNU", "CELI", "CELNOS", "SMAHN", "SMMCL"}
# HES codes -- check if present
HES_CODES = {"HES"}

OTHER_MYELOID = {"TMN", "MLADS"}


def classify_oncotree(code: str) -> str | None:
    """Classify an ONCOTREE_CODE into a disease group. Returns None if not myeloid."""
    uc = code.upper().strip()

    # AML exact matches
    if uc in AML_CODES:
        return "AML"

    # MDS/MPN overlap -- must check BEFORE MDS prefix
    if uc in MDSMPN_CODES:
        return "MDS/MPN"

    # MDS prefix (MDS, MDSEB, MDSEB1, MDSEB2, MDSMD, MDSRS, etc.)
    if uc.startswith("MDS"):
        return "MDS"

    # MPN exact matches
    if uc in MPN_EXACT or uc in HES_CODES:
        return "MPN"
    # MPN prefix (MPN, MPNST, MPNU, CML, CMLBCRABL1, SM, SMAHN, etc.)
    if uc.startswith("MPN") or uc.startswith("CML") or uc.startswith("SM") or uc.startswith("CEL") or uc.startswith("HES"):
        return "MPN"

    # Other myeloid
    if uc in OTHER_MYELOID:
        return "Other_myeloid"

    return None


def woolf_ci(table: list[list[int]], alpha: float = 0.05) -> tuple[float, float]:
    """
    Woolf logit method for 95% CI on odds ratio.

    table: [[a, b], [c, d]]
    Returns (lower, upper). Returns (0, inf) if any cell is zero.
    """
    a, b = table[0]
    c, d = table[1]

    if a == 0 or b == 0 or c == 0 or d == 0:
        # Haldane-Anscombe correction: add 0.5 to each cell
        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_c, b_c, c_c, d_c = a, b, c, d

    log_or = math.log(a_c * d_c / (b_c * c_c))
    se = math.sqrt(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)
    z = 1.96  # for 95% CI

    lower = math.exp(log_or - z * se)
    upper = math.exp(log_or + z * se)
    return lower, upper


def make_2x2(n_total: int, n_a: int, n_b: int, n_ab: int) -> list[list[int]]:
    """Construct 2x2 contingency table from marginals and overlap."""
    ab = n_ab
    a_only = n_a - n_ab
    b_only = n_b - n_ab
    neither = n_total - n_a - n_b + n_ab

    assert ab >= 0, f"Negative cell ab={ab}"
    assert a_only >= 0, f"Negative cell a_only={a_only} (n_a={n_a}, n_ab={n_ab})"
    assert b_only >= 0, f"Negative cell b_only={b_only} (n_b={n_b}, n_ab={n_ab})"
    assert neither >= 0, f"Negative cell neither={neither} (N={n_total}, n_a={n_a}, n_b={n_b}, n_ab={n_ab})"

    return [[ab, a_only], [b_only, neither]]


def run_pairwise_fisher(
    n_total: int,
    gene_samples: dict[str, set[str]],
) -> dict:
    """
    Run all 6 pairwise Fisher's exact tests for the 4 target genes.

    Returns dict keyed by "GENE_A+GENE_B" with OR, CI, p-values.
    """
    results = {}
    for ga, gb in combinations(TARGET_GENES, 2):
        n_a = len(gene_samples.get(ga, set()))
        n_b = len(gene_samples.get(gb, set()))
        n_ab = len(gene_samples.get(ga, set()) & gene_samples.get(gb, set()))

        if n_total == 0:
            results[f"{ga}+{gb}"] = {
                "n_a": n_a, "n_b": n_b, "n_ab": n_ab, "n_total": n_total,
                "expected": 0, "obs_exp_ratio": None,
                "odds_ratio": None, "ci_lower": None, "ci_upper": None,
                "p_two_sided": None, "p_less": None, "p_greater": None,
            }
            continue

        expected = n_a * n_b / n_total if n_total > 0 else 0
        obs_exp = n_ab / expected if expected > 0 else (float("inf") if n_ab > 0 else 0)

        table = make_2x2(n_total, n_a, n_b, n_ab)
        table_np = np.array(table)

        or_2s, p_2s = fisher_exact(table_np, alternative="two-sided")
        _, p_less = fisher_exact(table_np, alternative="less")
        _, p_greater = fisher_exact(table_np, alternative="greater")

        ci_lo, ci_hi = woolf_ci(table)

        results[f"{ga}+{gb}"] = {
            "n_a": n_a,
            "n_b": n_b,
            "n_ab": n_ab,
            "n_total": n_total,
            "expected": round(expected, 4),
            "obs_exp_ratio": round(obs_exp, 4) if obs_exp != float("inf") else "inf",
            "odds_ratio": round(or_2s, 6) if or_2s != float("inf") else "inf",
            "ci_lower": round(ci_lo, 6),
            "ci_upper": round(ci_hi, 6),
            "p_two_sided": p_2s,
            "p_less": p_less,
            "p_greater": p_greater,
            "table": table,
        }

    return results


def count_higher_order(
    gene_samples: dict[str, set[str]],
) -> dict:
    """Count triple and quadruple co-occurrences."""
    triples = {}
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        overlap = gene_samples.get(combo[0], set()).copy()
        for g in combo[1:]:
            overlap &= gene_samples.get(g, set())
        triples[key] = {
            "count": len(overlap),
            "samples": sorted(list(overlap))[:20],  # cap for JSON size
        }

    quad_key = "+".join(TARGET_GENES)
    quad_set = gene_samples.get(TARGET_GENES[0], set()).copy()
    for g in TARGET_GENES[1:]:
        quad_set &= gene_samples.get(g, set())

    quadruple = {
        quad_key: {
            "count": len(quad_set),
            "samples": sorted(list(quad_set))[:20],
        }
    }

    return {"triples": triples, "quadruple": quadruple}


def format_pval(p):
    if p is None:
        return "N/A"
    if p < 1e-10:
        return f"{p:.2e}"
    if p < 0.001:
        return f"{p:.2e}"
    if p < 0.05:
        return f"{p:.4f}"
    return f"{p:.4f}"


def main():
    print("=" * 78)
    print("DISEASE-STRATIFIED CO-OCCURRENCE ANALYSIS")
    print("Genes: DNMT3A, IDH2, PTPN11, SETBP1")
    print("=" * 78)

    # ── Step 1: Read clinical sample file, build sample -> disease group map ──
    print("\n[1/4] Reading clinical sample file...")
    sample_disease: dict[str, str] = {}
    disease_samples: dict[str, set[str]] = defaultdict(set)
    skipped_codes = defaultdict(int)

    with open(CLINICAL_FILE) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if parts[0] == "PATIENT_ID":
                continue
            sample_id = parts[1]
            oncotree = parts[3]

            group = classify_oncotree(oncotree)
            if group is not None:
                sample_disease[sample_id] = group
                disease_samples[group].add(sample_id)
            else:
                skipped_codes[oncotree] += 1

    print(f"  Myeloid samples classified: {len(sample_disease):,}")
    for grp in sorted(disease_samples):
        print(f"    {grp:15s}: {len(disease_samples[grp]):>7,} samples")
    print(f"  Non-myeloid ONCOTREE codes skipped: {len(skipped_codes):,} codes, "
          f"{sum(skipped_codes.values()):,} samples")

    # ── Step 2: Scan MAF for target gene mutations ────────────────────────────
    print("\n[2/4] Scanning MAF file for protein-altering mutations in target genes...")
    # gene -> set of sample IDs (across ALL samples, not just myeloid)
    gene_all_samples: dict[str, set[str]] = defaultdict(set)
    target_set = set(TARGET_GENES)
    n_maf_lines = 0
    n_target_hits = 0

    with open(MAF_FILE) as f:
        header = f.readline().strip().split("\t")
        hugo_idx = header.index("Hugo_Symbol")
        vc_idx = header.index("Variant_Classification")
        tsb_idx = header.index("Tumor_Sample_Barcode")

        for line in f:
            n_maf_lines += 1
            parts = line.split("\t")
            gene = parts[hugo_idx]
            if gene not in target_set:
                continue
            var_class = parts[vc_idx]
            if var_class not in PROTEIN_ALTERING:
                continue
            sample = parts[tsb_idx]
            gene_all_samples[gene].add(sample)
            n_target_hits += 1

    print(f"  MAF lines scanned: {n_maf_lines:,}")
    print(f"  Protein-altering hits in target genes: {n_target_hits:,}")
    for gene in TARGET_GENES:
        print(f"    {gene}: {len(gene_all_samples[gene]):,} samples (all cancer types)")

    # ── Step 3: Stratify by disease group ──────────────────────────────────────
    print("\n[3/4] Stratifying mutations by disease group...")

    # For each disease group, find which of its samples have mutations in each gene
    # Also create an "All_myeloid" group combining everything
    all_myeloid_samples = set()
    for grp_samples in disease_samples.values():
        all_myeloid_samples |= grp_samples

    groups_to_analyze = {
        "AML": disease_samples.get("AML", set()),
        "MDS": disease_samples.get("MDS", set()),
        "MDS/MPN": disease_samples.get("MDS/MPN", set()),
        "MPN": disease_samples.get("MPN", set()),
        "Other_myeloid": disease_samples.get("Other_myeloid", set()),
        "All_myeloid": all_myeloid_samples,
    }

    output = {
        "analysis": "Disease-stratified co-occurrence of DNMT3A, IDH2, PTPN11, SETBP1",
        "data_source": "AACR GENIE v19",
        "date": "2026-03-19",
        "protein_altering_types": sorted(PROTEIN_ALTERING),
        "disease_groups": {},
        "direction_changes": [],
        "index_patient_relevance": {},
    }

    # Store results for cross-group comparison
    all_group_results: dict[str, dict] = {}

    for grp_name, grp_samples in groups_to_analyze.items():
        n_total = len(grp_samples)
        if n_total == 0:
            print(f"\n  {grp_name}: 0 samples -- skipping")
            continue

        print(f"\n  {grp_name} (N={n_total:,}):")

        # Find which samples in this group have mutations in each gene
        grp_gene_samples: dict[str, set[str]] = {}
        for gene in TARGET_GENES:
            mutated = gene_all_samples[gene] & grp_samples
            grp_gene_samples[gene] = mutated
            pct = len(mutated) / n_total * 100
            print(f"    {gene:8s}: {len(mutated):>5,} ({pct:5.2f}%)")

        # Pairwise Fisher's exact tests
        pairwise = run_pairwise_fisher(n_total, grp_gene_samples)

        print(f"    Pairwise tests:")
        for pair, res in pairwise.items():
            direction = "depleted" if (res["odds_ratio"] is not None and
                                       res["odds_ratio"] != "inf" and
                                       res["odds_ratio"] < 1) else "enriched"
            sig = ""
            if res["p_two_sided"] is not None and res["p_two_sided"] < 0.05:
                sig = " *SIGNIFICANT*"
            print(f"      {pair:20s}: obs={res['n_ab']:>3d}, exp={res['expected']:>7.2f}, "
                  f"OR={res['odds_ratio'] if res['odds_ratio'] is not None else 'N/A':>10}, "
                  f"p(2s)={format_pval(res['p_two_sided']):>10s}, "
                  f"p(less)={format_pval(res['p_less']):>10s} [{direction}]{sig}")

        # Triple and quadruple counts
        higher = count_higher_order(grp_gene_samples)
        print(f"    Triple co-occurrences:")
        for key, val in higher["triples"].items():
            print(f"      {key}: {val['count']}")
        quad_key = "+".join(TARGET_GENES)
        print(f"    Quadruple ({quad_key}): {higher['quadruple'][quad_key]['count']}")

        # Prepare JSON-serializable pairwise (convert numpy/special types)
        pairwise_json = {}
        for pair, res in pairwise.items():
            pairwise_json[pair] = {
                "n_a": res["n_a"],
                "n_b": res["n_b"],
                "n_ab": res["n_ab"],
                "n_total": res["n_total"],
                "expected": res["expected"],
                "obs_exp_ratio": res["obs_exp_ratio"],
                "odds_ratio": float(res["odds_ratio"]) if isinstance(res["odds_ratio"], (int, float)) else res["odds_ratio"],
                "ci_95_lower": float(res["ci_lower"]) if res["ci_lower"] is not None else None,
                "ci_95_upper": float(res["ci_upper"]) if res["ci_upper"] is not None else None,
                "p_two_sided": float(res["p_two_sided"]) if res["p_two_sided"] is not None else None,
                "p_less_depletion": float(res["p_less"]) if res["p_less"] is not None else None,
                "p_greater_enrichment": float(res["p_greater"]) if res["p_greater"] is not None else None,
                "table_2x2": res.get("table"),
            }

        # Gene counts for JSON
        gene_counts = {}
        for gene in TARGET_GENES:
            n_mut = len(grp_gene_samples[gene])
            gene_counts[gene] = {
                "count": n_mut,
                "frequency_pct": round(n_mut / n_total * 100, 3),
            }

        output["disease_groups"][grp_name] = {
            "n_samples": n_total,
            "gene_counts": gene_counts,
            "pairwise_fisher": pairwise_json,
            "triples": {k: v["count"] for k, v in higher["triples"].items()},
            "triple_samples": {k: v["samples"] for k, v in higher["triples"].items()},
            "quadruple": higher["quadruple"][quad_key]["count"],
            "quadruple_samples": higher["quadruple"][quad_key]["samples"],
        }

        all_group_results[grp_name] = pairwise

    # ── Step 4: Cross-group comparison ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("[4/4] Cross-group comparison: direction changes")
    print("=" * 78)

    compare_groups = ["AML", "MDS", "MDS/MPN", "MPN"]
    direction_changes = []

    for pair in [f"{a}+{b}" for a, b in combinations(TARGET_GENES, 2)]:
        directions = {}
        for grp in compare_groups:
            if grp not in all_group_results:
                continue
            res = all_group_results[grp].get(pair)
            if res is None or res["odds_ratio"] is None:
                continue
            or_val = res["odds_ratio"]
            if or_val == "inf":
                directions[grp] = "enriched_inf"
            elif or_val > 1:
                directions[grp] = "enriched"
            elif or_val < 1:
                directions[grp] = "depleted"
            else:
                directions[grp] = "neutral"

        # Check if direction changes between any two groups
        unique_dirs = set(directions.values())
        has_enriched = any("enriched" in d for d in unique_dirs)
        has_depleted = "depleted" in unique_dirs

        if has_enriched and has_depleted:
            detail = {
                "pair": pair,
                "direction_by_group": directions,
                "change": "DIRECTION REVERSAL: depleted in some, enriched in others",
            }

            # Add specifics
            dep_groups = [g for g, d in directions.items() if d == "depleted"]
            enr_groups = [g for g, d in directions.items() if "enriched" in d]
            detail["depleted_in"] = dep_groups
            detail["enriched_in"] = enr_groups

            # Add OR and p-values for each group
            for grp in compare_groups:
                if grp in all_group_results and pair in all_group_results[grp]:
                    r = all_group_results[grp][pair]
                    detail[f"OR_{grp}"] = r["odds_ratio"]
                    detail[f"p_{grp}"] = r["p_two_sided"]

            direction_changes.append(detail)
            print(f"\n  ** {pair}: DIRECTION REVERSAL **")
            for grp, d in directions.items():
                r = all_group_results[grp][pair]
                sig = " *" if r["p_two_sided"] is not None and r["p_two_sided"] < 0.05 else ""
                print(f"      {grp:10s}: {d:12s} (OR={r['odds_ratio']}, p={format_pval(r['p_two_sided'])}){sig}")
        else:
            # No reversal -- still report
            if len(directions) >= 2:
                consistent_dir = list(unique_dirs)[0] if len(unique_dirs) == 1 else "mixed_magnitude"
                print(f"\n  {pair}: consistent direction ({consistent_dir})")
                for grp, d in directions.items():
                    r = all_group_results[grp][pair]
                    sig = " *" if r["p_two_sided"] is not None and r["p_two_sided"] < 0.05 else ""
                    print(f"      {grp:10s}: {d:12s} (OR={r['odds_ratio']}, p={format_pval(r['p_two_sided'])}){sig}")

    output["direction_changes"] = direction_changes

    # ── Henrik's case: MDS-IB2/AML overlap ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("HENRIK RELEVANCE: MDS-IB2/AML overlap")
    print("=" * 78)

    # MDS-EB2 codes: MDSEB2 is the primary code
    mdseb2_samples = set()
    with open(CLINICAL_FILE) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if parts[0] == "PATIENT_ID":
                continue
            if parts[3] in ("MDSEB2",):
                mdseb2_samples.add(parts[1])

    n_mdseb2 = len(mdseb2_samples)
    print(f"\n  MDS-EB2 samples (ONCOTREE=MDSEB2): {n_mdseb2}")

    if n_mdseb2 > 0:
        mdseb2_gene_samples = {}
        for gene in TARGET_GENES:
            mutated = gene_all_samples[gene] & mdseb2_samples
            mdseb2_gene_samples[gene] = mutated
            pct = len(mutated) / n_mdseb2 * 100
            print(f"    {gene:8s}: {len(mutated):>4d} ({pct:5.2f}%)")

        mdseb2_pairwise = run_pairwise_fisher(n_mdseb2, mdseb2_gene_samples)
        mdseb2_higher = count_higher_order(mdseb2_gene_samples)

        print(f"\n  Pairwise tests (MDS-EB2):")
        for pair, res in mdseb2_pairwise.items():
            or_val = res['odds_ratio']
            direction = "depleted" if (or_val is not None and or_val != "inf" and or_val < 1) else "enriched"
            sig = " *SIGNIFICANT*" if res["p_two_sided"] is not None and res["p_two_sided"] < 0.05 else ""
            print(f"    {pair:20s}: obs={res['n_ab']:>3d}, exp={res['expected']:>7.2f}, "
                  f"OR={or_val}, p(2s)={format_pval(res['p_two_sided'])}{sig}")

        print(f"\n  Triple co-occurrences (MDS-EB2):")
        for key, val in mdseb2_higher["triples"].items():
            print(f"    {key}: {val['count']}")
        quad_key = "+".join(TARGET_GENES)
        print(f"  Quadruple: {mdseb2_higher['quadruple'][quad_key]['count']}")

        # JSON output for MDS-EB2
        mdseb2_json = {}
        for pair, res in mdseb2_pairwise.items():
            mdseb2_json[pair] = {
                "n_a": res["n_a"], "n_b": res["n_b"], "n_ab": res["n_ab"],
                "n_total": res["n_total"],
                "expected": res["expected"],
                "obs_exp_ratio": res["obs_exp_ratio"],
                "odds_ratio": float(res["odds_ratio"]) if isinstance(res["odds_ratio"], (int, float)) else res["odds_ratio"],
                "ci_95_lower": float(res["ci_lower"]) if res["ci_lower"] is not None else None,
                "ci_95_upper": float(res["ci_upper"]) if res["ci_upper"] is not None else None,
                "p_two_sided": float(res["p_two_sided"]) if res["p_two_sided"] is not None else None,
                "p_less_depletion": float(res["p_less"]) if res["p_less"] is not None else None,
                "p_greater_enrichment": float(res["p_greater"]) if res["p_greater"] is not None else None,
            }

        mdseb2_gene_json = {}
        for gene in TARGET_GENES:
            n_mut = len(mdseb2_gene_samples[gene])
            mdseb2_gene_json[gene] = {
                "count": n_mut,
                "frequency_pct": round(n_mut / n_mdseb2 * 100, 3),
            }

        output["index_patient_relevance"] = {
            "diagnosis": "MDS-IB2 / MDS-AML overlap (WHO: MDS with increased blasts-2)",
            "oncotree_code": "MDSEB2",
            "n_mdseb2_samples": n_mdseb2,
            "gene_counts": mdseb2_gene_json,
            "pairwise_fisher": mdseb2_json,
            "triples": {k: v["count"] for k, v in mdseb2_higher["triples"].items()},
            "quadruple": mdseb2_higher["quadruple"][quad_key]["count"],
            "commentary": (
                "Henrik's diagnosis of MDS-IB2 (12-15% blasts) sits at the MDS/AML boundary. "
                "Under WHO criteria this is MDS with increased blasts-2; under ICC with "
                "myelodysplasia-related cytogenetic changes it is classified as AML. "
                "The MDS-EB2 stratum is the single most relevant comparison group. "
                "However, given the diagnostic overlap, both the MDS and AML strata are informative. "
                "If co-occurrence patterns differ between MDS-EB2 and either broad MDS or AML groups, "
                "this indicates that the MDS/AML boundary itself represents a biological transition "
                "in mutational constraint."
            ),
            "most_relevant_strata": ["MDS-EB2 (MDSEB2)", "MDS (all)", "AML"],
        }

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY TABLE: Odds Ratios by Disease Group")
    print("=" * 78)

    pairs = [f"{a}+{b}" for a, b in combinations(TARGET_GENES, 2)]
    summary_groups = ["AML", "MDS", "MDS/MPN", "MPN", "All_myeloid"]

    # Header
    header = f"{'Pair':>22s}"
    for grp in summary_groups:
        header += f" | {grp:>14s}"
    print(header)
    print("-" * len(header))

    for pair in pairs:
        row = f"{pair:>22s}"
        for grp in summary_groups:
            if grp in all_group_results and pair in all_group_results[grp]:
                r = all_group_results[grp][pair]
                or_val = r["odds_ratio"]
                p_val = r["p_two_sided"]
                if or_val is None:
                    row += f" | {'N/A':>14s}"
                elif or_val == "inf":
                    row += f" | {'inf':>14s}"
                else:
                    sig = "*" if p_val is not None and p_val < 0.05 else " "
                    row += f" | {or_val:>12.4f}{sig} "
                    # intentionally put sig marker after OR
            else:
                row += f" | {'--':>14s}"
        print(row)

    # ── Write JSON ─────────────────────────────────────────────────────────────
    output_path = RESULTS_DIR / "disease_stratified_cooccurrence.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults written to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
