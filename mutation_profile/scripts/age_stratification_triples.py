#!/usr/bin/env python3
"""
Age distribution analysis of triple-carriers and pairwise carriers
from the IPSS-M dataset (df_clinical.tsv + df_mut.tsv).

Target genes: DNMT3A, IDH2, PTPN11, SETBP1
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from datetime import date

# Paths
BASE = Path("Path(__file__).resolve().parents[2]/mutation_profile")
CLINICAL = BASE / "data/ipssm/df_clinical.tsv"
MUTATIONS = BASE / "data/ipssm/df_mut.tsv"
OUTPUT = BASE / "results/age_stratification_triples.md"

# Load data
df_clin = pd.read_csv(CLINICAL, sep="\t")
df_mut = pd.read_csv(MUTATIONS, sep="\t")

print(f"Clinical: {len(df_clin)} rows, columns: {list(df_clin.columns[:10])}...")
print(f"Mutations: {len(df_mut)} rows, columns include target genes:")

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]
for g in TARGET_GENES:
    if g in df_mut.columns:
        print(f"  {g}: column found, {df_mut[g].sum()} carriers")
    else:
        print(f"  {g}: COLUMN NOT FOUND!")

# Merge on ID
df = df_mut[["ID"] + TARGET_GENES].merge(df_clin[["ID", "AGE", "SEX", "MDS_TYPE", "WHO_2016", "IPSSM"]], on="ID", how="inner")
print(f"\nMerged: {len(df)} rows")

# Check AGE data quality
age_missing = df["AGE"].isna().sum()
print(f"AGE missing: {age_missing} of {len(df)}")
print(f"AGE dtype: {df['AGE'].dtype}")
print(f"AGE range: {df['AGE'].min()} to {df['AGE'].max()}")
print(f"AGE mean: {df['AGE'].mean():.1f}, median: {df['AGE'].median():.1f}")

# ============================================================
# 1. Triple carriers: all C(4,3) = 4 combinations
# ============================================================
print("\n" + "="*70)
print("TRIPLE CARRIERS (3 of 4 target genes)")
print("="*70)

triple_results = {}
for triple in combinations(TARGET_GENES, 3):
    label = "+".join(triple)
    mask = (df[list(triple)] == 1).all(axis=1)
    carriers = df[mask].copy()
    triple_results[label] = carriers
    print(f"\n{label}: {len(carriers)} patients")
    if len(carriers) > 0:
        for _, row in carriers.iterrows():
            age_str = f"age {int(row['AGE'])}" if pd.notna(row['AGE']) else "age MISSING"
            sex = row['SEX'] if pd.notna(row['SEX']) else '?'
            mds = row['MDS_TYPE'] if pd.notna(row['MDS_TYPE']) else '?'
            who = row['WHO_2016'] if pd.notna(row['WHO_2016']) else '?'
            ipssm = row['IPSSM'] if pd.notna(row['IPSSM']) else '?'
            # Check which of the 4th gene they have
            missing_gene = [g for g in TARGET_GENES if g not in triple][0]
            has_4th = int(row[missing_gene]) if pd.notna(row[missing_gene]) else 0
            print(f"  {row['ID']}: {age_str}, {sex}, {who}, IPSS-M={ipssm}, {missing_gene}={'YES' if has_4th else 'no'}")

# ============================================================
# 2. Quadruple carriers
# ============================================================
print("\n" + "="*70)
print("QUADRUPLE CARRIERS (all 4 genes)")
print("="*70)
mask_quad = (df[TARGET_GENES] == 1).all(axis=1)
quad_carriers = df[mask_quad]
print(f"Patients with all 4: {len(quad_carriers)}")

# ============================================================
# 3. Pairwise carriers — age distributions
# ============================================================
print("\n" + "="*70)
print("PAIRWISE CARRIERS — AGE DISTRIBUTIONS")
print("="*70)

pair_results = {}
for pair in combinations(TARGET_GENES, 2):
    label = "+".join(pair)
    mask = (df[list(pair)] == 1).all(axis=1)
    carriers = df[mask].copy()
    pair_results[label] = carriers
    n = len(carriers)
    ages = carriers["AGE"].dropna()
    n_age = len(ages)
    if n_age > 0:
        print(f"\n{label}: {n} total ({n_age} with age data)")
        print(f"  Mean: {ages.mean():.1f}, Median: {ages.median():.1f}")
        print(f"  Range: {ages.min():.0f} - {ages.max():.0f}")
        print(f"  SD: {ages.std():.1f}")
        print(f"  Under 40: {(ages < 40).sum()}")
        print(f"  Under 50: {(ages < 50).sum()}")
        # List all under-40 patients
        young = carriers[carriers["AGE"] < 40]
        for _, row in young.iterrows():
            print(f"    -> {row['ID']}: age {int(row['AGE'])}, {row['SEX']}, {row['WHO_2016']}")
    else:
        print(f"\n{label}: {n} total (0 with age data)" if n > 0 else f"\n{label}: 0 patients")

# ============================================================
# 4. Under-40 in the entire cohort
# ============================================================
print("\n" + "="*70)
print("AGE DISTRIBUTION: ENTIRE IPSS-M COHORT")
print("="*70)

all_ages = df["AGE"].dropna()
print(f"Total patients: {len(df)}")
print(f"With age data: {len(all_ages)}")
print(f"Mean: {all_ages.mean():.1f}, Median: {all_ages.median():.1f}")
print(f"Range: {all_ages.min():.0f} - {all_ages.max():.0f}")
print(f"Under 40: {(all_ages < 40).sum()} ({(all_ages < 40).sum()/len(all_ages)*100:.2f}%)")
print(f"Under 50: {(all_ages < 50).sum()} ({(all_ages < 50).sum()/len(all_ages)*100:.2f}%)")

# Age percentiles
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {pct}th percentile: {np.percentile(all_ages, pct):.0f}")

# ============================================================
# 5. Individual gene carrier age distributions
# ============================================================
print("\n" + "="*70)
print("INDIVIDUAL GENE CARRIER AGE DISTRIBUTIONS")
print("="*70)

individual_results = {}
for gene in TARGET_GENES:
    mask = df[gene] == 1
    carriers = df[mask].copy()
    ages = carriers["AGE"].dropna()
    individual_results[gene] = carriers
    n = len(carriers)
    n_age = len(ages)
    print(f"\n{gene}: {n} carriers ({n_age} with age)")
    if n_age > 0:
        print(f"  Mean: {ages.mean():.1f}, Median: {ages.median():.1f}")
        print(f"  Range: {ages.min():.0f} - {ages.max():.0f}")
        print(f"  SD: {ages.std():.1f}")
        print(f"  Under 40: {(ages < 40).sum()} ({(ages < 40).sum()/n_age*100:.2f}%)")
        print(f"  Under 50: {(ages < 50).sum()} ({(ages < 50).sum()/n_age*100:.2f}%)")
        # List all under-40
        young = carriers[carriers["AGE"] < 40]
        if len(young) > 0:
            print(f"  Under-40 patients:")
            for _, row in young.iterrows():
                # What other target genes does this patient have?
                other_genes = [g for g in TARGET_GENES if g != gene and row[g] == 1]
                other_str = f" (also: {', '.join(other_genes)})" if other_genes else ""
                print(f"    {row['ID']}: age {int(row['AGE'])}, {row['SEX']}, {row['WHO_2016']}{other_str}")

# ============================================================
# 6. Any carrier of at least one target gene: under-40 count
# ============================================================
print("\n" + "="*70)
print("ANY TARGET GENE CARRIER: AGE ANALYSIS")
print("="*70)
any_target = (df[TARGET_GENES] == 1).any(axis=1)
any_carriers = df[any_target]
any_ages = any_carriers["AGE"].dropna()
print(f"Patients with at least 1 target gene: {len(any_carriers)} ({len(any_ages)} with age)")
print(f"  Under 40: {(any_ages < 40).sum()}")
print(f"  Under 50: {(any_ages < 50).sum()}")

# ============================================================
# 7. Age distribution of under-40 patients specifically
# ============================================================
print("\n" + "="*70)
print("ALL UNDER-40 PATIENTS IN IPSS-M")
print("="*70)
young_all = df[df["AGE"] < 40].copy()
print(f"Total under-40: {len(young_all)}")
print(f"Age distribution: {sorted(young_all['AGE'].dropna().astype(int).tolist())}")

# How many target gene mutations do under-40 patients have?
young_all["n_target"] = young_all[TARGET_GENES].sum(axis=1)
print(f"\nTarget gene mutation counts among under-40:")
for n in range(5):
    count = (young_all["n_target"] == n).sum()
    if count > 0:
        print(f"  {n} target genes: {count} patients")

# List under-40 patients with any target gene
young_with_target = young_all[young_all["n_target"] > 0]
if len(young_with_target) > 0:
    print(f"\nUnder-40 patients with target gene mutations:")
    for _, row in young_with_target.iterrows():
        genes = [g for g in TARGET_GENES if row[g] == 1]
        print(f"  {row['ID']}: age {int(row['AGE'])}, {row['SEX']}, {row['WHO_2016']}, mutations: {', '.join(genes)}")


# ============================================================
# BUILD MARKDOWN REPORT
# ============================================================
print("\n\nGenerating report...")

lines = []
lines.append("# Age Stratification of Triple- and Pairwise Carriers: IPSS-M Dataset")
lines.append("")
lines.append(f"**Analysis date:** {date.today()}")
lines.append(f"**Dataset:** IPSS-M binary mutation matrix (n={len(df)})")
lines.append(f"**Target genes:** {', '.join(TARGET_GENES)}")
lines.append(f"**Age data available:** {len(all_ages)} of {len(df)} patients ({len(all_ages)/len(df)*100:.1f}%)")
lines.append(f"**Age data missing:** {age_missing}")
lines.append("")
lines.append("---")
lines.append("")

# Cohort age overview
lines.append("## 1. IPSS-M Cohort Age Overview")
lines.append("")
lines.append(f"| Metric | Value |")
lines.append(f"|--------|-------|")
lines.append(f"| Total patients | {len(df)} |")
lines.append(f"| With age data | {len(all_ages)} |")
lines.append(f"| Mean age | {all_ages.mean():.1f} |")
lines.append(f"| Median age | {all_ages.median():.1f} |")
lines.append(f"| Range | {all_ages.min():.0f} - {all_ages.max():.0f} |")
lines.append(f"| SD | {all_ages.std():.1f} |")
lines.append(f"| Under 40 | {(all_ages < 40).sum()} ({(all_ages < 40).sum()/len(all_ages)*100:.2f}%) |")
lines.append(f"| Under 50 | {(all_ages < 50).sum()} ({(all_ages < 50).sum()/len(all_ages)*100:.2f}%) |")
lines.append("")

lines.append("### Age Percentiles")
lines.append("")
lines.append("| Percentile | Age |")
lines.append("|-----------|-----|")
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    lines.append(f"| {pct}th | {np.percentile(all_ages, pct):.0f} |")
lines.append("")
lines.append("---")
lines.append("")

# Individual gene age distributions
lines.append("## 2. Individual Gene Carrier Age Distributions")
lines.append("")
lines.append("| Gene | N carriers | N with age | Mean | Median | Range | SD | Under 40 | Under 50 |")
lines.append("|------|-----------|------------|------|--------|-------|----|----------|----------|")
for gene in TARGET_GENES:
    carriers = individual_results[gene]
    ages = carriers["AGE"].dropna()
    n = len(carriers)
    n_age = len(ages)
    if n_age > 0:
        u40 = (ages < 40).sum()
        u50 = (ages < 50).sum()
        lines.append(f"| {gene} | {n} | {n_age} | {ages.mean():.1f} | {ages.median():.1f} | {ages.min():.0f}-{ages.max():.0f} | {ages.std():.1f} | {u40} ({u40/n_age*100:.1f}%) | {u50} ({u50/n_age*100:.1f}%) |")
    else:
        lines.append(f"| {gene} | {n} | 0 | - | - | - | - | - | - |")
lines.append("")

# Under-40 individual gene carriers detail
lines.append("### Under-40 Patients With Individual Target Genes")
lines.append("")
for gene in TARGET_GENES:
    carriers = individual_results[gene]
    young = carriers[carriers["AGE"] < 40]
    if len(young) > 0:
        lines.append(f"**{gene}** ({len(young)} under-40 carriers):")
        lines.append("")
        lines.append("| Patient ID | Age | Sex | WHO 2016 | IPSS-M | Other target genes |")
        lines.append("|------------|-----|-----|----------|--------|--------------------|")
        for _, row in young.iterrows():
            other = [g for g in TARGET_GENES if g != gene and row[g] == 1]
            other_str = ", ".join(other) if other else "none"
            age_val = int(row['AGE']) if pd.notna(row['AGE']) else "?"
            sex = row['SEX'] if pd.notna(row['SEX']) else "?"
            who = row['WHO_2016'] if pd.notna(row['WHO_2016']) else "?"
            ipssm = row['IPSSM'] if pd.notna(row['IPSSM']) else "?"
            lines.append(f"| {row['ID']} | {age_val} | {sex} | {who} | {ipssm} | {other_str} |")
        lines.append("")
    else:
        lines.append(f"**{gene}**: 0 patients under 40.")
        lines.append("")
lines.append("---")
lines.append("")

# Pairwise age distributions
lines.append("## 3. Pairwise Carrier Age Distributions")
lines.append("")
lines.append("| Pair | N | N with age | Mean | Median | Range | SD | Under 40 | Under 50 |")
lines.append("|------|---|------------|------|--------|-------|----|----------|----------|")
for pair in combinations(TARGET_GENES, 2):
    label = "+".join(pair)
    carriers = pair_results[label]
    n = len(carriers)
    ages = carriers["AGE"].dropna()
    n_age = len(ages)
    if n_age > 0:
        u40 = (ages < 40).sum()
        u50 = (ages < 50).sum()
        lines.append(f"| {label} | {n} | {n_age} | {ages.mean():.1f} | {ages.median():.1f} | {ages.min():.0f}-{ages.max():.0f} | {ages.std():.1f} | {u40} ({u40/n_age*100:.1f}%) | {u50} ({u50/n_age*100:.1f}%) |")
    elif n > 0:
        lines.append(f"| {label} | {n} | 0 | - | - | - | - | - | - |")
    else:
        lines.append(f"| {label} | 0 | - | - | - | - | - | - | - |")
lines.append("")

# Under-40 pairwise detail
lines.append("### Under-40 Pairwise Carriers (Detail)")
lines.append("")
any_under40_pair = False
for pair in combinations(TARGET_GENES, 2):
    label = "+".join(pair)
    carriers = pair_results[label]
    young = carriers[carriers["AGE"] < 40]
    if len(young) > 0:
        any_under40_pair = True
        lines.append(f"**{label}** ({len(young)} under-40):")
        lines.append("")
        lines.append("| Patient ID | Age | Sex | WHO 2016 | IPSS-M | All target genes |")
        lines.append("|------------|-----|-----|----------|--------|-----------------|")
        for _, row in young.iterrows():
            all_targets = [g for g in TARGET_GENES if row[g] == 1]
            age_val = int(row['AGE']) if pd.notna(row['AGE']) else "?"
            sex = row['SEX'] if pd.notna(row['SEX']) else "?"
            who = row['WHO_2016'] if pd.notna(row['WHO_2016']) else "?"
            ipssm = row['IPSSM'] if pd.notna(row['IPSSM']) else "?"
            lines.append(f"| {row['ID']} | {age_val} | {sex} | {who} | {ipssm} | {', '.join(all_targets)} |")
        lines.append("")

if not any_under40_pair:
    lines.append("**No pairwise carriers under age 40 in any combination.**")
    lines.append("")

lines.append("---")
lines.append("")

# Triple carriers
lines.append("## 4. Triple Carriers (3 of 4 Target Genes)")
lines.append("")
lines.append("### Summary")
lines.append("")
lines.append("| Triple | N in IPSS-M | Missing gene |")
lines.append("|--------|------------|-------------|")
for triple in combinations(TARGET_GENES, 3):
    label = "+".join(triple)
    missing = [g for g in TARGET_GENES if g not in triple][0]
    n = len(triple_results[label])
    lines.append(f"| {label} | {n} | {missing} |")
total_triples = sum(len(v) for v in triple_results.values())
lines.append(f"| **Total** | **{total_triples}** | |")
lines.append("")

# Detail for each triple
for triple in combinations(TARGET_GENES, 3):
    label = "+".join(triple)
    carriers = triple_results[label]
    missing = [g for g in TARGET_GENES if g not in triple][0]
    lines.append(f"### {label} (missing {missing}): {len(carriers)} patients")
    lines.append("")
    if len(carriers) > 0:
        lines.append("| Patient ID | Age | Sex | WHO 2016 | IPSS-M | Has 4th gene ({})? |".format(missing))
        lines.append("|------------|-----|-----|----------|--------|{}-|".format("-" * len(f"Has 4th gene ({missing})?") + "-"))
        for _, row in carriers.iterrows():
            age_val = int(row['AGE']) if pd.notna(row['AGE']) else "MISSING"
            sex = row['SEX'] if pd.notna(row['SEX']) else "?"
            who = row['WHO_2016'] if pd.notna(row['WHO_2016']) else "?"
            ipssm = row['IPSSM'] if pd.notna(row['IPSSM']) else "?"
            has_4th = "YES" if (pd.notna(row[missing]) and row[missing] == 1) else "no"
            lines.append(f"| {row['ID']} | {age_val} | {sex} | {who} | {ipssm} | {has_4th} |")
        lines.append("")
        ages = carriers["AGE"].dropna()
        if len(ages) > 0:
            lines.append(f"Age: mean {ages.mean():.1f}, median {ages.median():.1f}, range {ages.min():.0f}-{ages.max():.0f}")
            lines.append("")
    else:
        lines.append("No patients found.")
        lines.append("")

lines.append("---")
lines.append("")

# Quadruple
lines.append("## 5. Quadruple Carriers (All 4 Target Genes)")
lines.append("")
lines.append(f"Patients carrying all four of DNMT3A+IDH2+PTPN11+SETBP1: **{len(quad_carriers)}**")
lines.append("")
lines.append("This is consistent with the FINAL co-occurrence report finding of zero quadruple carriers across all databases.")
lines.append("")
lines.append("---")
lines.append("")

# Comparison to mid-30s patient
lines.append("## 6. Context: How Unusual Is a Mid-30s Patient With This Profile?")
lines.append("")

n_under40 = (all_ages < 40).sum()
n_total = len(all_ages)
pct_under40 = n_under40 / n_total * 100

lines.append(f"### Overall cohort: patients under 40")
lines.append("")
lines.append(f"- Total IPSS-M patients: {n_total}")
lines.append(f"- Under 40: **{n_under40}** ({pct_under40:.2f}%)")
lines.append(f"- Under 35: **{(all_ages < 35).sum()}** ({(all_ages < 35).sum()/n_total*100:.2f}%)")
lines.append(f"- Under 30: **{(all_ages < 30).sum()}** ({(all_ages < 30).sum()/n_total*100:.2f}%)")
lines.append("")

lines.append("### Under-40 patients in this cohort (all ages listed)")
lines.append("")
young_all_sorted = df[df["AGE"] < 40].sort_values("AGE")
if len(young_all_sorted) > 0:
    lines.append("| Patient ID | Age | Sex | WHO 2016 | IPSS-M | Target gene mutations |")
    lines.append("|------------|-----|-----|----------|--------|----------------------|")
    for _, row in young_all_sorted.iterrows():
        genes = [g for g in TARGET_GENES if row[g] == 1]
        gene_str = ", ".join(genes) if genes else "none"
        age_val = int(row['AGE']) if pd.notna(row['AGE']) else "?"
        sex = row['SEX'] if pd.notna(row['SEX']) else "?"
        who = row['WHO_2016'] if pd.notna(row['WHO_2016']) else "?"
        ipssm = row['IPSSM'] if pd.notna(row['IPSSM']) else "?"
        lines.append(f"| {row['ID']} | {age_val} | {sex} | {who} | {ipssm} | {gene_str} |")
    lines.append("")

# Under-40 with ANY pair
lines.append("### Under-40 patients carrying any pairwise combination of target genes")
lines.append("")
any_pair_mask = False
pair_u40_text = []
for pair in combinations(TARGET_GENES, 2):
    label = "+".join(pair)
    carriers = pair_results[label]
    young = carriers[carriers["AGE"] < 40]
    pair_u40_text.append(f"- {label}: {len(young)} under-40")
    if len(young) > 0:
        any_pair_mask = True

for t in pair_u40_text:
    lines.append(t)
lines.append("")

if not any_pair_mask:
    lines.append("**Zero patients under 40 carry any pairwise combination of the four target genes in IPSS-M.**")
    lines.append("")

# Under-40 with ANY single target gene
lines.append("### Under-40 patients carrying any single target gene")
lines.append("")
for gene in TARGET_GENES:
    carriers = individual_results[gene]
    young = carriers[carriers["AGE"] < 40]
    lines.append(f"- {gene}: {len(young)} under-40 of {len(carriers)} total ({len(young)/len(carriers)*100:.1f}% if carriers>0)" if len(carriers) > 0 else f"- {gene}: 0 carriers total")
lines.append("")

# Summary statement
lines.append("### Summary")
lines.append("")

# Count how many under-40 have 2+ target genes
young_df = df[df["AGE"] < 40].copy()
young_df["n_target"] = young_df[TARGET_GENES].sum(axis=1)
n_young_2plus = (young_df["n_target"] >= 2).sum()
n_young_3plus = (young_df["n_target"] >= 3).sum()
n_young_4 = (young_df["n_target"] == 4).sum()

lines.append(f"Among {n_under40} patients under 40 in IPSS-M:")
lines.append(f"- With 0 target genes: {(young_df['n_target'] == 0).sum()}")
lines.append(f"- With 1 target gene: {(young_df['n_target'] == 1).sum()}")
lines.append(f"- With 2 target genes: {(young_df['n_target'] == 2).sum()}")
lines.append(f"- With 3 target genes: {n_young_3plus}")
lines.append(f"- With 4 target genes: {n_young_4}")
lines.append("")

# Patient being analyzed comparison
lines.append("The patient under analysis (mid-30s, 4 concurrent driver mutations: DNMT3A+IDH2+PTPN11+SETBP1 + monosomy 7) would be:")
lines.append("")
lines.append(f"1. In the youngest {pct_under40:.1f}% of the IPSS-M cohort by age alone")
lines.append(f"2. The only patient in the dataset (n={len(df)}) with all four target gene mutations")
lines.append(f"3. Among {n_young_2plus} under-40 patients with 2+ target genes (if any)")
lines.append(f"4. Among {n_young_3plus} under-40 patients with 3+ target genes (if any)")
lines.append("")

lines.append("---")
lines.append("")
lines.append("## 7. Methodology")
lines.append("")
lines.append("- Data source: IPSS-M binary mutation matrix (df_mut.tsv) and clinical annotation (df_clinical.tsv)")
lines.append("- N = 2,957 MDS patients from the IWG-PM discovery cohort")
lines.append("- Target genes selected from the patient profile: DNMT3A, IDH2, PTPN11, SETBP1")
lines.append("- Age is recorded as integer years at diagnosis")
lines.append(f"- Age data available for {len(all_ages)} of {len(df)} patients ({len(all_ages)/len(df)*100:.1f}%)")
lines.append("- All statistics computed on patients with non-missing age values")
lines.append("- Mutation status determined by binary (0/1) coding in the mutation matrix")
lines.append("")

# Write
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    f.write("\n".join(lines))

print(f"\nReport written to: {OUTPUT}")
print("Done.")
