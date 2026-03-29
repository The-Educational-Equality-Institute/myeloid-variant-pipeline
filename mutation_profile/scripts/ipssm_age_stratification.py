#!/usr/bin/env python3
"""
IPSS-M Age Stratification Analysis
Analyzes age distributions for patients carrying DNMT3A, IDH2, PTPN11, SETBP1
mutations and their combinations in the IPSS-M cohort (n=2,957).
"""

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("mutation_profile/data/ipssm")
OUT_DIR = Path("mutation_profile/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
clin = pd.read_csv(DATA_DIR / "df_clinical.tsv", sep="\t")
mut = pd.read_csv(DATA_DIR / "df_mut.tsv", sep="\t")

# Merge on ID
df = clin.merge(mut, on="ID", how="inner")
assert len(df) == 2957, f"Expected 2957, got {len(df)}"

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]

# ── Helper functions ───────────────────────────────────────────────────────

def age_stats(ages: pd.Series) -> dict:
    """Compute age distribution statistics."""
    ages = ages.dropna()
    if len(ages) == 0:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None,
                "q25": None, "q75": None, "std": None, "range": "N/A",
                "histogram_bins": {}}
    
    # Histogram bins: decades
    bins = list(range(0, 110, 10))
    hist_counts, _ = np.histogram(ages, bins=bins)
    hist_dict = {}
    for i in range(len(bins) - 1):
        label = f"{bins[i]}-{bins[i+1]-1}"
        hist_dict[label] = int(hist_counts[i])
    
    return {
        "n": int(len(ages)),
        "median": round(float(ages.median()), 1),
        "mean": round(float(ages.mean()), 1),
        "min": int(ages.min()),
        "max": int(ages.max()),
        "q25": round(float(ages.quantile(0.25)), 1),
        "q75": round(float(ages.quantile(0.75)), 1),
        "std": round(float(ages.std()), 1),
        "range": f"{int(ages.min())}-{int(ages.max())}",
        "histogram_bins": hist_dict,
    }


def detect_monosomy7(cyto_str: str) -> bool:
    """Detect monosomy 7 from cytogenetics string."""
    if pd.isna(cyto_str):
        return False
    s = str(cyto_str).lower()
    # Patterns: -7, del(7), del(7q), del(7p), monosomy 7
    # But NOT +7, t(...;7;...) alone without deletion
    # -7 as standalone (not part of -7q which is partial)
    if re.search(r'(?<!\d)-7(?!\d|[pq])', s):
        return True
    if re.search(r'del\(7\)', s):
        return True
    if 'monosomy' in s and '7' in s:
        return True
    return False


def detect_del7q(cyto_str: str) -> bool:
    """Detect del(7q) — partial 7q deletion — from cytogenetics or CNACS."""
    if pd.isna(cyto_str):
        return False
    s = str(cyto_str).lower()
    if re.search(r'del\(7\)\(q', s) or re.search(r'del\(7q\)', s):
        return True
    return False


# ── Monosomy 7 detection (combine cytogenetics + CNACS) ───────────────────
df["mono7_cyto"] = df["CYTOGENETICS"].apply(detect_monosomy7)

# Also check CNACS_chrarm_loss for chromosome 7
df["mono7_cnacs"] = df["CNACS_chrarm_loss"].apply(
    lambda x: bool(re.search(r'(?<!\d)7[pq]?(?!\d)', str(x))) if pd.notna(x) else False
)
# Also check full chromosome 7 loss in CNACS
df["mono7_cnacs_full"] = df["CNACS_chrarm_loss"].apply(
    lambda x: ('7p' in str(x) and '7q' in str(x)) if pd.notna(x) else False
)

# Combined monosomy 7 flag (any evidence)
df["has_monosomy7"] = df["mono7_cyto"] | df["mono7_cnacs_full"]
# Also include del(7q) separately
df["has_del7q"] = df["CYTOGENETICS"].apply(detect_del7q)
# Broader: any chr7 abnormality
df["any_chr7_abn"] = df["mono7_cyto"] | df["mono7_cnacs"] | df["has_del7q"]

print(f"Monosomy 7 (cytogenetics): {df['mono7_cyto'].sum()}")
print(f"Monosomy 7 (CNACS full 7p+7q loss): {df['mono7_cnacs_full'].sum()}")
print(f"Combined monosomy 7: {df['has_monosomy7'].sum()}")
print(f"Any chr7 abnormality: {df['any_chr7_abn'].sum()}")

# ── Count target gene mutations per patient ────────────────────────────────
df["target_gene_count"] = df[TARGET_GENES].sum(axis=1)

# Count ALL gene mutations (all 126+ gene columns in mut)
gene_cols = [c for c in mut.columns if c != "ID" and c not in ["TP53mono", "TP53multi", "TET2bi", "TET2other"]]
df["total_mutations"] = df[gene_cols].sum(axis=1)

# ── 1. Overall age distribution ───────────────────────────────────────────
overall = age_stats(df["AGE"])

# ── 2. Age by single gene ─────────────────────────────────────────────────
age_by_gene = {}
for gene in TARGET_GENES:
    mask = df[gene] == 1
    stats = age_stats(df.loc[mask, "AGE"])
    stats["n_total"] = int(mask.sum())
    stats["pct_of_cohort"] = round(100.0 * mask.sum() / len(df), 2)
    age_by_gene[gene] = stats

# ── 3. Age by pairwise combinations ──────────────────────────────────────
pairwise = {}
for g1, g2 in combinations(TARGET_GENES, 2):
    key = f"{g1}+{g2}"
    mask = (df[g1] == 1) & (df[g2] == 1)
    n = mask.sum()
    if n > 0:
        stats = age_stats(df.loc[mask, "AGE"])
        stats["n_total"] = int(n)
        # List individual ages if small group
        if n <= 20:
            ages_list = sorted(df.loc[mask, "AGE"].dropna().tolist())
            stats["individual_ages"] = [int(a) if pd.notna(a) else None for a in ages_list]
            # Also list patient IDs
            stats["patient_ids"] = df.loc[mask, "ID"].tolist()
        pairwise[key] = stats
    else:
        pairwise[key] = {"n_total": 0, "note": "No patients with this combination"}

# ── 4. Triple combinations ───────────────────────────────────────────────
triples = {}
for g1, g2, g3 in combinations(TARGET_GENES, 3):
    key = f"{g1}+{g2}+{g3}"
    mask = (df[g1] == 1) & (df[g2] == 1) & (df[g3] == 1)
    n = mask.sum()
    if n > 0:
        stats = age_stats(df.loc[mask, "AGE"])
        stats["n_total"] = int(n)
        ages_list = sorted(df.loc[mask, "AGE"].dropna().tolist())
        stats["individual_ages"] = [int(a) if pd.notna(a) else None for a in ages_list]
        stats["patient_ids"] = df.loc[mask, "ID"].tolist()
        # Get additional clinical details for these rare patients
        details = []
        for _, row in df.loc[mask].iterrows():
            details.append({
                "patient_id": row["ID"],
                "age": int(row["AGE"]) if pd.notna(row["AGE"]) else None,
                "sex": row["SEX"],
                "who_2016": row["WHO_2016"],
                "ipssm": row["IPSSM"],
                "cytogenetics": row["CYTOGENETICS"],
                "has_monosomy7": bool(row["has_monosomy7"]),
                "total_mutations": int(row["total_mutations"]),
                "genes": key,
            })
        stats["patient_details"] = details
        triples[key] = stats
    else:
        triples[key] = {"n_total": 0, "note": "No patients with this combination"}

# ── 5. Quad combination (all 4 genes) ────────────────────────────────────
quad_mask = (df["DNMT3A"] == 1) & (df["IDH2"] == 1) & (df["PTPN11"] == 1) & (df["SETBP1"] == 1)
quad_n = quad_mask.sum()
quad_result = {"n_total": int(quad_n)}
if quad_n > 0:
    quad_result.update(age_stats(df.loc[quad_mask, "AGE"]))
    quad_result["patient_ids"] = df.loc[quad_mask, "ID"].tolist()
    quad_result["individual_ages"] = sorted(df.loc[quad_mask, "AGE"].dropna().tolist())

# ── 6. DNMT3A+PTPN11+SETBP1 triple carriers (specifically asked) ────────
triple_dps_mask = (df["DNMT3A"] == 1) & (df["PTPN11"] == 1) & (df["SETBP1"] == 1)
triple_dps_n = triple_dps_mask.sum()
triple_carrier_ages = []
if triple_dps_n > 0:
    for _, row in df.loc[triple_dps_mask].iterrows():
        triple_carrier_ages.append({
            "patient_id": row["ID"],
            "age": int(row["AGE"]) if pd.notna(row["AGE"]) else None,
            "sex": row["SEX"],
            "who_2016": row["WHO_2016"],
            "ipssm": row["IPSSM"],
            "cytogenetics": row["CYTOGENETICS"],
            "has_monosomy7": bool(row["has_monosomy7"]),
            "total_mutations": int(row["total_mutations"]),
            "genes": "DNMT3A+PTPN11+SETBP1",
            "also_has_IDH2": bool(row["IDH2"] == 1),
        })

# ── 7. Patients under 40 ─────────────────────────────────────────────────
under40 = df[df["AGE"] < 40].copy()
under40_result = {
    "count": int(len(under40)),
    "pct_of_total": round(100.0 * len(under40) / len(df), 2),
    "age_distribution": age_stats(under40["AGE"]),
    "with_DNMT3A": int((under40["DNMT3A"] == 1).sum()),
    "with_IDH2": int((under40["IDH2"] == 1).sum()),
    "with_PTPN11": int((under40["PTPN11"] == 1).sum()),
    "with_SETBP1": int((under40["SETBP1"] == 1).sum()),
    "with_2plus_target_genes": int((under40["target_gene_count"] >= 2).sum()),
    "with_3plus_target_genes": int((under40["target_gene_count"] >= 3).sum()),
    "with_4_target_genes": int((under40["target_gene_count"] == 4).sum()),
    "with_monosomy7": int(under40["has_monosomy7"].sum()),
    "with_any_chr7_abn": int(under40["any_chr7_abn"].sum()),
    "with_2plus_total_mutations": int((under40["total_mutations"] >= 2).sum()),
    "with_3plus_total_mutations": int((under40["total_mutations"] >= 3).sum()),
    "with_4plus_total_mutations": int((under40["total_mutations"] >= 4).sum()),
    "with_5plus_total_mutations": int((under40["total_mutations"] >= 5).sum()),
}

# Detail patients under 40 with 2+ target genes
under40_multi = under40[under40["target_gene_count"] >= 2]
under40_multi_details = []
for _, row in under40_multi.iterrows():
    genes_carried = [g for g in TARGET_GENES if row[g] == 1]
    under40_multi_details.append({
        "patient_id": row["ID"],
        "age": int(row["AGE"]) if pd.notna(row["AGE"]) else None,
        "sex": row["SEX"],
        "who_2016": row["WHO_2016"],
        "ipssm": row["IPSSM"],
        "target_genes": "+".join(genes_carried),
        "total_mutations": int(row["total_mutations"]),
        "cytogenetics": row["CYTOGENETICS"],
        "has_monosomy7": bool(row["has_monosomy7"]),
    })
under40_result["multi_target_gene_details"] = under40_multi_details

# ── 8. Patients under 35 ─────────────────────────────────────────────────
under35 = df[df["AGE"] < 35].copy()
under35_result = {
    "count": int(len(under35)),
    "pct_of_total": round(100.0 * len(under35) / len(df), 2),
    "age_distribution": age_stats(under35["AGE"]),
    "with_DNMT3A": int((under35["DNMT3A"] == 1).sum()),
    "with_IDH2": int((under35["IDH2"] == 1).sum()),
    "with_PTPN11": int((under35["PTPN11"] == 1).sum()),
    "with_SETBP1": int((under35["SETBP1"] == 1).sum()),
    "with_2plus_target_genes": int((under35["target_gene_count"] >= 2).sum()),
    "with_3plus_target_genes": int((under35["target_gene_count"] >= 3).sum()),
    "with_4_target_genes": int((under35["target_gene_count"] == 4).sum()),
    "with_monosomy7": int(under35["has_monosomy7"].sum()),
    "with_any_chr7_abn": int(under35["any_chr7_abn"].sum()),
    "with_2plus_total_mutations": int((under35["total_mutations"] >= 2).sum()),
    "with_3plus_total_mutations": int((under35["total_mutations"] >= 3).sum()),
    "with_4plus_total_mutations": int((under35["total_mutations"] >= 4).sum()),
    "with_5plus_total_mutations": int((under35["total_mutations"] >= 5).sum()),
}

# Detail patients under 35 with any target gene
under35_with_targets = under35[under35["target_gene_count"] >= 1]
under35_target_details = []
for _, row in under35_with_targets.iterrows():
    genes_carried = [g for g in TARGET_GENES if row[g] == 1]
    under35_target_details.append({
        "patient_id": row["ID"],
        "age": int(row["AGE"]) if pd.notna(row["AGE"]) else None,
        "sex": row["SEX"],
        "who_2016": row["WHO_2016"],
        "ipssm": row["IPSSM"],
        "target_genes": "+".join(genes_carried),
        "total_mutations": int(row["total_mutations"]),
        "cytogenetics": row["CYTOGENETICS"],
        "has_monosomy7": bool(row["has_monosomy7"]),
    })
under35_result["patients_with_any_target_gene"] = under35_target_details

# ── 9. Monosomy 7 cross-reference with age <40 ───────────────────────────
mono7_under40 = df[(df["AGE"] < 40) & (df["has_monosomy7"])].copy()
mono7_under40_details = []
for _, row in mono7_under40.iterrows():
    genes_carried = [g for g in TARGET_GENES if row[g] == 1]
    mono7_under40_details.append({
        "patient_id": row["ID"],
        "age": int(row["AGE"]) if pd.notna(row["AGE"]) else None,
        "sex": row["SEX"],
        "who_2016": row["WHO_2016"],
        "ipssm": row["IPSSM"],
        "cytogenetics": row["CYTOGENETICS"],
        "target_genes": "+".join(genes_carried) if genes_carried else "none",
        "total_mutations": int(row["total_mutations"]),
    })

monosomy7_age_analysis = {
    "total_monosomy7_patients": int(df["has_monosomy7"].sum()),
    "total_any_chr7_abn": int(df["any_chr7_abn"].sum()),
    "monosomy7_age_distribution": age_stats(df.loc[df["has_monosomy7"], "AGE"]),
    "monosomy7_under_40": {
        "count": int(len(mono7_under40)),
        "details": mono7_under40_details,
    },
    "monosomy7_under_35": {
        "count": int(((df["AGE"] < 35) & (df["has_monosomy7"])).sum()),
    },
    "monosomy7_with_DNMT3A": int(((df["has_monosomy7"]) & (df["DNMT3A"] == 1)).sum()),
    "monosomy7_with_IDH2": int(((df["has_monosomy7"]) & (df["IDH2"] == 1)).sum()),
    "monosomy7_with_PTPN11": int(((df["has_monosomy7"]) & (df["PTPN11"] == 1)).sum()),
    "monosomy7_with_SETBP1": int(((df["has_monosomy7"]) & (df["SETBP1"] == 1)).sum()),
}

# ── 10. Mutation burden analysis (all genes) ─────────────────────────────
mutation_burden = {
    "mean_mutations_per_patient": round(float(df["total_mutations"].mean()), 2),
    "median_mutations_per_patient": round(float(df["total_mutations"].median()), 1),
    "patients_with_0_mutations": int((df["total_mutations"] == 0).sum()),
    "patients_with_1_mutation": int((df["total_mutations"] == 1).sum()),
    "patients_with_2_mutations": int((df["total_mutations"] == 2).sum()),
    "patients_with_3_mutations": int((df["total_mutations"] == 3).sum()),
    "patients_with_4_mutations": int((df["total_mutations"] == 4).sum()),
    "patients_with_5plus_mutations": int((df["total_mutations"] >= 5).sum()),
    "patients_with_2plus_mutations": int((df["total_mutations"] >= 2).sum()),
    "pct_with_2plus_mutations": round(100.0 * (df["total_mutations"] >= 2).sum() / len(df), 2),
}

# Under 40 mutation burden
mutation_burden["under_40"] = {
    "mean_mutations": round(float(under40["total_mutations"].mean()), 2) if len(under40) > 0 else None,
    "median_mutations": round(float(under40["total_mutations"].median()), 1) if len(under40) > 0 else None,
    "with_2plus_driver_mutations": int((under40["total_mutations"] >= 2).sum()),
    "with_4plus_driver_mutations": int((under40["total_mutations"] >= 4).sum()),
    "pct_with_2plus": round(100.0 * (under40["total_mutations"] >= 2).sum() / len(under40), 2) if len(under40) > 0 else None,
}

# ── 11. Context: how rare is a mid-30s patient with 4+ driver mutations? ──
mid30s = df[(df["AGE"] >= 33) & (df["AGE"] <= 37)]
mid30s_with_4plus = mid30s[mid30s["total_mutations"] >= 4]
mid30s_with_target = mid30s[mid30s["target_gene_count"] >= 2]

patient_profile_context = {
    "description": "Patient diagnosed MDS-IB2→AML in mid-30s with DNMT3A+IDH2+PTPN11+SETBP1 + monosomy 7",
    "patients_aged_33_37": int(len(mid30s)),
    "pct_of_cohort_aged_33_37": round(100.0 * len(mid30s) / len(df), 2),
    "aged_33_37_with_4plus_mutations": int(len(mid30s_with_4plus)),
    "aged_33_37_with_2plus_target_genes": int(len(mid30s_with_target)),
    "aged_33_37_with_4plus_mutations_details": [],
}
for _, row in mid30s_with_4plus.iterrows():
    genes = [g for g in TARGET_GENES if row[g] == 1]
    all_mutated = [c for c in gene_cols if row[c] == 1]
    patient_profile_context["aged_33_37_with_4plus_mutations_details"].append({
        "patient_id": row["ID"],
        "age": int(row["AGE"]),
        "sex": row["SEX"],
        "who_2016": row["WHO_2016"],
        "ipssm": row["IPSSM"],
        "target_genes": genes,
        "all_mutated_genes": all_mutated,
        "total_mutations": int(row["total_mutations"]),
        "cytogenetics": row["CYTOGENETICS"],
        "has_monosomy7": bool(row["has_monosomy7"]),
    })

# How many patients in entire cohort have all 4 target genes?
all4_mask = (df["DNMT3A"] == 1) & (df["IDH2"] == 1) & (df["PTPN11"] == 1) & (df["SETBP1"] == 1)
patient_profile_context["patients_with_all_4_target_genes"] = int(all4_mask.sum())

# How many have 3+ of the target genes?
patient_profile_context["patients_with_3plus_target_genes"] = int((df["target_gene_count"] >= 3).sum())

# The patient also has monosomy 7 — find anyone with 3+ target genes AND monosomy 7
target3_mono7 = (df["target_gene_count"] >= 3) & (df["has_monosomy7"])
patient_profile_context["patients_with_3plus_target_AND_monosomy7"] = int(target3_mono7.sum())
if target3_mono7.sum() > 0:
    patient_profile_context["patients_with_3plus_target_AND_monosomy7_details"] = []
    for _, row in df[target3_mono7].iterrows():
        genes = [g for g in TARGET_GENES if row[g] == 1]
        patient_profile_context["patients_with_3plus_target_AND_monosomy7_details"].append({
            "patient_id": row["ID"],
            "age": int(row["AGE"]) if pd.notna(row["AGE"]) else None,
            "target_genes": "+".join(genes),
            "total_mutations": int(row["total_mutations"]),
            "cytogenetics": row["CYTOGENETICS"],
        })

# ── Assemble final JSON ──────────────────────────────────────────────────
result = {
    "analysis": "IPSS-M Age Stratification Analysis",
    "dataset": "IPSS-M cohort (Bernard et al., NEJM Evidence 2022)",
    "date": "2026-03-14",
    "total_patients": 2957,
    "age_column_used": "AGE",
    "age_available_for_n": int(df["AGE"].notna().sum()),
    "age_missing_for_n": int(df["AGE"].isna().sum()),
    "overall_age_distribution": overall,
    "age_by_gene_group": age_by_gene,
    "pairwise_combinations": pairwise,
    "triple_combinations": triples,
    "quadruple_combination_all_4_genes": quad_result,
    "triple_carrier_ages_DNMT3A_PTPN11_SETBP1": triple_carrier_ages,
    "patients_under_40": under40_result,
    "patients_under_35": under35_result,
    "monosomy7_analysis": monosomy7_age_analysis,
    "mutation_burden_analysis": mutation_burden,
    "patient_profile_context": patient_profile_context,
}

# ── Write output ──────────────────────────────────────────────────────────
outpath = OUT_DIR / "ipssm_age_stratification.json"
with open(outpath, "w") as f:
    json.dump(result, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"Output written to: {outpath}")
print(f"{'='*70}")

# ── Print summary ─────────────────────────────────────────────────────────
print(f"\n=== SUMMARY ===")
print(f"Total patients: {result['total_patients']}")
print(f"Age available: {result['age_available_for_n']}")
print(f"Overall median age: {overall['median']}")
print(f"Overall range: {overall['range']}")
print()
print("--- Single gene carriers ---")
for g in TARGET_GENES:
    s = age_by_gene[g]
    print(f"  {g}: n={s['n_total']}, median_age={s['median']}, range={s['range']}")
print()
print("--- Pairwise combinations ---")
for key, s in pairwise.items():
    if s["n_total"] > 0:
        ages_str = ""
        if "individual_ages" in s:
            ages_str = f" ages={s['individual_ages']}"
        print(f"  {key}: n={s['n_total']}, median_age={s.get('median')}{ages_str}")
    else:
        print(f"  {key}: n=0")
print()
print("--- Triple combinations ---")
for key, s in triples.items():
    if s["n_total"] > 0:
        print(f"  {key}: n={s['n_total']}, ages={s.get('individual_ages')}")
    else:
        print(f"  {key}: n=0")
print()
print(f"--- Quadruple (all 4 genes): n={quad_result['n_total']} ---")
print()
print(f"--- DNMT3A+PTPN11+SETBP1 triple carriers: n={len(triple_carrier_ages)} ---")
for tc in triple_carrier_ages:
    print(f"  {tc['patient_id']}: age={tc['age']}, {tc['who_2016']}, IPSSM={tc['ipssm']}")
    print(f"    cyto: {tc['cytogenetics']}, mono7: {tc['has_monosomy7']}, total_mut: {tc['total_mutations']}")
print()
print(f"--- Under 40 ---")
print(f"  Count: {under40_result['count']} ({under40_result['pct_of_total']}%)")
print(f"  DNMT3A: {under40_result['with_DNMT3A']}, IDH2: {under40_result['with_IDH2']}")
print(f"  PTPN11: {under40_result['with_PTPN11']}, SETBP1: {under40_result['with_SETBP1']}")
print(f"  2+ target genes: {under40_result['with_2plus_target_genes']}")
print(f"  Monosomy 7: {under40_result['with_monosomy7']}")
print()
print(f"--- Under 35 ---")
print(f"  Count: {under35_result['count']} ({under35_result['pct_of_total']}%)")
print(f"  DNMT3A: {under35_result['with_DNMT3A']}, IDH2: {under35_result['with_IDH2']}")
print(f"  PTPN11: {under35_result['with_PTPN11']}, SETBP1: {under35_result['with_SETBP1']}")
print()
print(f"--- Mutation burden ---")
print(f"  Mean mutations/patient: {mutation_burden['mean_mutations_per_patient']}")
print(f"  2+ mutations: {mutation_burden['patients_with_2plus_mutations']} ({mutation_burden['pct_with_2plus_mutations']}%)")
print()
print(f"--- Patient profile context (mid-30s, 4+ driver mutations) ---")
print(f"  Patients aged 33-37: {patient_profile_context['patients_aged_33_37']}")
print(f"  Of those with 4+ mutations: {patient_profile_context['aged_33_37_with_4plus_mutations']}")
print(f"  All 4 target genes in entire cohort: {patient_profile_context['patients_with_all_4_target_genes']}")
print(f"  3+ target genes: {patient_profile_context['patients_with_3plus_target_genes']}")
print(f"  3+ target + mono7: {patient_profile_context['patients_with_3plus_target_AND_monosomy7']}")
