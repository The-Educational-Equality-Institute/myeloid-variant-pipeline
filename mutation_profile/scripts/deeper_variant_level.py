#!/usr/bin/env python3
"""
Variant-level deep analysis: specific mutations in co-occurring patients,
Henrik-specific variant matching, VAF analysis, clonal architecture.
Single comprehensive MAF scan.
"""

import json
import re
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mrna-hematology-research"
GENIE_RAW = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]
TARGET_GENES_SET = set(TARGET_GENES)

PATHOGENIC_VAR_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    "Translation_Start_Site",
}

MYELOID_ONCOTREE_CODES = {
    "AML", "AMLMRC", "AMLRGA", "AMLRR", "AMLNOS", "APL", "AMOL", "APMF",
    "AMLMBC", "AMLCBFB", "AMLRUNX1", "AMLMLLT3", "AMLDEKNUP", "AMLBCR",
    "AMLGATA2MECOM", "AMLNPM1", "AMLCEBPA", "AMLTP53", "AMLDEK",
    "MDS", "MDS5Q", "MDSEB1", "MDSEB2", "MDSMD", "MDSSLD", "MDSRS",
    "MDSU", "MDSRSMD", "MDSSID", "MDSLB", "MDSIB1", "MDSIB2",
    "RCMD", "RARS", "RA", "RAEB", "RAEB1", "RAEB2", "RCUD",
    "CMML", "CMML0", "CMML1", "CMML2", "JMML", "MDSMPNU",
    "MDSMPNRST", "ACML", "ACML_ATYPICAL",
    "MPN", "CML", "ET", "PV", "PMF", "CMLBCRABL1", "SM", "CEL",
    "MPNU", "MPNST",
    "BPDCN", "MPAL", "TMN", "ALAL", "TMDS", "TAML",
}

# Henrik's exact variants
HENRIK_VARIANTS = {
    "DNMT3A": "R882H",
    "IDH2": "R140Q",
    "PTPN11": "E76Q",
    "SETBP1": "G870S",
}


def extract_variant_name(protein_change):
    if not protein_change:
        return ""
    return protein_change.replace("p.", "")


def main():
    print("=" * 80)
    print("VARIANT-LEVEL DEEP ANALYSIS - GENIE v19.0")
    print("=" * 80)

    # Load clinical samples
    print("\nLoading clinical samples...")
    samples = {}
    with open(GENIE_RAW / "data_clinical_sample.txt") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            if header is None:
                header = line.strip().split("\t")
                continue
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))
            sid = row.get("SAMPLE_ID", "")
            if sid:
                samples[sid] = row

    myeloid_sids = set()
    sid_to_pid = {}
    for sid, srow in samples.items():
        code = srow.get("ONCOTREE_CODE", "").upper()
        if code in MYELOID_ONCOTREE_CODES:
            myeloid_sids.add(sid)
            sid_to_pid[sid] = srow.get("PATIENT_ID", sid)

    # Load patients
    patients = {}
    with open(GENIE_RAW / "data_clinical_patient.txt") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            if header is None:
                header = line.strip().split("\t")
                continue
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))
            pid = row.get("PATIENT_ID", "")
            if pid:
                patients[pid] = row

    print(f"  Myeloid samples: {len(myeloid_sids):,}")

    # ──────────────────────────────────────────────────────────────────
    # SCAN MAF - comprehensive data extraction
    # ──────────────────────────────────────────────────────────────────
    print("\nScanning MAF file...")

    # Per-patient mutation details
    patient_mutations = defaultdict(lambda: defaultdict(list))

    # Variant frequency counters
    variant_counts = defaultdict(Counter)  # gene -> variant_name -> count of patients
    patient_variant_set = defaultdict(lambda: defaultdict(set))  # pid -> gene -> set of variants

    # Henrik-specific matching
    index_matches = defaultdict(set)  # "DNMT3A_R882H" -> set of patient_ids

    # VAF data
    vaf_data = defaultdict(list)  # gene -> list of (pid, variant, vaf)

    row_count = 0
    with open(GENIE_RAW / "data_mutations_extended.txt") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            if "Hugo_Symbol" in line and header is None:
                header = line.strip().split("\t")
                continue
            if header is None:
                header = line.strip().split("\t")
                continue

            row_count += 1
            if row_count % 1_000_000 == 0:
                print(f"  {row_count:,} rows...")

            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))

            gene = row.get("Hugo_Symbol", "")
            if gene not in TARGET_GENES_SET:
                continue

            sid = row.get("Tumor_Sample_Barcode", "")
            if sid not in myeloid_sids:
                continue

            var_class = row.get("Variant_Classification", "")
            if var_class not in PATHOGENIC_VAR_CLASSES:
                continue

            protein = row.get("HGVSp_Short", "")
            if re.match(r'^p\.\*\d+\*$', protein):
                continue

            pid = sid_to_pid.get(sid, sid)
            variant_name = extract_variant_name(protein)

            # Store mutation
            mut_info = {
                "protein_change": protein,
                "variant_name": variant_name,
                "variant_classification": var_class,
                "chromosome": row.get("Chromosome", ""),
                "start_pos": row.get("Start_Position", ""),
                "t_depth": row.get("t_depth", ""),
                "t_alt_count": row.get("t_alt_count", ""),
                "center": samples.get(sid, {}).get("CENTER", ""),
            }
            patient_mutations[pid][gene].append(mut_info)

            # Track variant frequencies
            patient_variant_set[pid][gene].add(variant_name)

            # Henrik matching
            for hgene, hvar in HENRIK_VARIANTS.items():
                if gene == hgene and variant_name == hvar:
                    index_matches[f"{hgene}_{hvar}"].add(pid)

            # VAF calculation
            try:
                depth = int(row.get("t_depth", 0))
                alt = int(row.get("t_alt_count", 0))
                if depth > 0:
                    vaf = alt / depth
                    vaf_data[gene].append((pid, variant_name, vaf, depth))
            except (ValueError, TypeError):
                pass

    print(f"  Total rows scanned: {row_count:,}")

    # Count variant frequencies per gene
    for pid, gene_vars in patient_variant_set.items():
        for gene, variants in gene_vars.items():
            for v in variants:
                variant_counts[gene][v] += 1

    # ──────────────────────────────────────────────────────────────────
    # 1. VARIANT SPECTRUM PER GENE
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("1. VARIANT SPECTRUM (top 20 per gene)")
    print("─" * 80)

    for gene in TARGET_GENES:
        total_patients = sum(variant_counts[gene].values())
        print(f"\n  {gene} ({total_patients} total patient-variant pairs):")
        for var, count in variant_counts[gene].most_common(20):
            pct = count / total_patients * 100 if total_patients > 0 else 0
            print(f"    {var:>25}: {count:>5} ({pct:>5.1f}%)")

    # ──────────────────────────────────────────────────────────────────
    # 2. HENRIK-SPECIFIC VARIANT MATCHING
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("2. HENRIK-SPECIFIC VARIANT MATCHING")
    print(f"   Target: DNMT3A R882H + IDH2 R140Q + PTPN11 E76Q + SETBP1 G870S")
    print("─" * 80)

    for key, pids in sorted(index_matches.items()):
        print(f"\n  {key}: {len(pids)} patients")

    # Find patients with exact Henrik variants
    print(f"\n  Combination matching:")

    r882h_pids = index_matches.get("DNMT3A_R882H", set())
    r140q_pids = index_matches.get("IDH2_R140Q", set())
    e76q_pids = index_matches.get("PTPN11_E76Q", set())
    g870s_pids = index_matches.get("SETBP1_G870S", set())

    # Pairwise
    for g1, g2 in combinations(HENRIK_VARIANTS.keys(), 2):
        key1 = f"{g1}_{HENRIK_VARIANTS[g1]}"
        key2 = f"{g2}_{HENRIK_VARIANTS[g2]}"
        overlap = index_matches.get(key1, set()) & index_matches.get(key2, set())
        print(f"    {HENRIK_VARIANTS[g1]}+{HENRIK_VARIANTS[g2]}: {len(overlap)} patients")
        if overlap and len(overlap) <= 10:
            for pid in sorted(overlap):
                muts = patient_mutations[pid]
                all_genes = ", ".join(
                    f"{g}:{muts[g][0]['protein_change']}" for g in TARGET_GENES if g in muts
                )
                age = patients.get(pid, {}).get("AGE_AT_SEQ_REPORT", "?")
                print(f"      {pid}: {all_genes} (age {age})")

    # Triple
    print(f"\n    R882H+R140Q+E76Q (Henrik's 3 pathogenic): "
          f"{len(r882h_pids & r140q_pids & e76q_pids)} patients")
    triple_match = r882h_pids & r140q_pids & e76q_pids
    for pid in sorted(triple_match):
        muts = patient_mutations[pid]
        all_genes = ", ".join(
            f"{g}:{[m['protein_change'] for m in muts[g]]}" for g in TARGET_GENES if g in muts
        )
        age = patients.get(pid, {}).get("AGE_AT_SEQ_REPORT", "?")
        sex = patients.get(pid, {}).get("SEX", "?")
        print(f"      {pid}: {all_genes} ({sex}, age {age})")

    # Quadruple (Henrik's exact profile)
    quad_match = r882h_pids & r140q_pids & e76q_pids & g870s_pids
    print(f"\n    R882H+R140Q+E76Q+G870S (Henrik's exact profile): {len(quad_match)} patients")

    # ──────────────────────────────────────────────────────────────────
    # 3. WHICH DNMT3A VARIANTS CO-OCCUR WITH EACH GENE?
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("3. DNMT3A VARIANT SPECTRUM IN CO-OCCURRENCES")
    print("─" * 80)

    for g2 in ["IDH2", "PTPN11", "SETBP1"]:
        # Find patients with both DNMT3A + g2
        cooccur_pids = set()
        for pid, gmuts in patient_mutations.items():
            if "DNMT3A" in gmuts and g2 in gmuts:
                cooccur_pids.add(pid)

        dnmt3a_vars_in_cooccur = Counter()
        for pid in cooccur_pids:
            for v in patient_variant_set[pid].get("DNMT3A", set()):
                dnmt3a_vars_in_cooccur[v] += 1

        print(f"\n  DNMT3A variants in DNMT3A+{g2} patients (n={len(cooccur_pids)}):")
        for var, count in dnmt3a_vars_in_cooccur.most_common(10):
            pct = count / len(cooccur_pids) * 100
            print(f"    {var:>20}: {count:>4} ({pct:>5.1f}%)")

    # ──────────────────────────────────────────────────────────────────
    # 4. IDH2 VARIANT DISTRIBUTION IN CO-OCCURRENCES
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("4. IDH2 R140 vs R172 IN CO-OCCURRENCES")
    print("─" * 80)

    for g2 in ["DNMT3A", "PTPN11", "SETBP1"]:
        cooccur_pids = set()
        for pid, gmuts in patient_mutations.items():
            if "IDH2" in gmuts and g2 in gmuts:
                cooccur_pids.add(pid)

        r140_count = 0
        r172_count = 0
        other_count = 0
        for pid in cooccur_pids:
            idh2_vars = patient_variant_set[pid].get("IDH2", set())
            has_r140 = any(v.startswith("R140") for v in idh2_vars)
            has_r172 = any(v.startswith("R172") for v in idh2_vars)
            if has_r140:
                r140_count += 1
            if has_r172:
                r172_count += 1
            if not has_r140 and not has_r172:
                other_count += 1

        n = len(cooccur_pids)
        print(f"\n  IDH2 in IDH2+{g2} (n={n}):")
        print(f"    R140x: {r140_count} ({r140_count/n*100:.1f}%)")
        print(f"    R172x: {r172_count} ({r172_count/n*100:.1f}%)")
        print(f"    Other: {other_count} ({other_count/n*100:.1f}%)")

    # ──────────────────────────────────────────────────────────────────
    # 5. SETBP1 VARIANT SPECTRUM (SKI domain vs non-SKI)
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("5. SETBP1 VARIANT SPECTRUM (SKI domain aa 858-871 vs other)")
    print("─" * 80)

    setbp1_ski = Counter()
    setbp1_nonski = Counter()
    for pid, gene_vars in patient_variant_set.items():
        for v in gene_vars.get("SETBP1", set()):
            # Extract amino acid position
            m = re.match(r'[A-Z](\d+)', v)
            if m:
                pos = int(m.group(1))
                if 858 <= pos <= 871:
                    setbp1_ski[v] += 1
                else:
                    setbp1_nonski[v] += 1
            else:
                setbp1_nonski[v] += 1

    ski_total = sum(setbp1_ski.values())
    nonski_total = sum(setbp1_nonski.values())
    print(f"\n  SKI domain (aa 858-871): {ski_total} patients")
    for v, c in setbp1_ski.most_common(15):
        print(f"    {v:>15}: {c}")
    print(f"\n  Non-SKI domain: {nonski_total} patients")
    for v, c in setbp1_nonski.most_common(15):
        print(f"    {v:>15}: {c}")

    # ──────────────────────────────────────────────────────────────────
    # 6. VAF DISTRIBUTION
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("6. VAF DISTRIBUTION PER GENE")
    print("─" * 80)

    for gene in TARGET_GENES:
        vafs = [v for _, _, v, d in vaf_data[gene] if d >= 50]  # min 50x depth
        if not vafs:
            print(f"\n  {gene}: no VAF data with sufficient depth")
            continue

        vafs.sort()
        n = len(vafs)
        mean_vaf = sum(vafs) / n
        median_vaf = vafs[n // 2]
        q25 = vafs[int(n * 0.25)]
        q75 = vafs[int(n * 0.75)]

        # VAF bins
        bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
        print(f"\n  {gene} (n={n} measurements, min depth 50x):")
        print(f"    Mean VAF: {mean_vaf:.3f}, Median: {median_vaf:.3f}, IQR: {q25:.3f}-{q75:.3f}")
        for lo, hi in bins:
            count = sum(1 for v in vafs if lo <= v < hi)
            pct = count / n * 100
            bar = "#" * int(pct / 2)
            print(f"    {lo:.1f}-{hi:.1f}: {count:>5} ({pct:>5.1f}%) {bar}")

    # ──────────────────────────────────────────────────────────────────
    # 7. TRIPLE CARRIER DETAILED PROFILES
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("7. DETAILED PROFILES: DNMT3A+IDH2+PTPN11 TRIPLE CARRIERS")
    print("─" * 80)

    # Build gene carrier sets
    gene_carriers = {g: set() for g in TARGET_GENES}
    for pid, gmuts in patient_mutations.items():
        for g in TARGET_GENES:
            if g in gmuts:
                gene_carriers[g].add(pid)

    triple_pids = gene_carriers["DNMT3A"] & gene_carriers["IDH2"] & gene_carriers["PTPN11"]
    print(f"\n  {len(triple_pids)} patients with DNMT3A + IDH2 + PTPN11:")

    for pid in sorted(triple_pids):
        muts = patient_mutations[pid]
        prow = patients.get(pid, {})
        age = prow.get("AGE_AT_SEQ_REPORT", "?")
        sex = prow.get("SEX", "?")
        center = prow.get("CENTER", "?")

        # Check if also has SETBP1
        has_setbp1 = "SETBP1" in muts
        setbp1_str = f" + SETBP1:{muts['SETBP1'][0]['protein_change']}" if has_setbp1 else ""

        print(f"\n  {pid} ({sex}, age {age}, {center}){setbp1_str}")
        for g in TARGET_GENES:
            if g in muts:
                for m in muts[g][:3]:  # max 3 variants per gene
                    vaf_str = ""
                    try:
                        d = int(m.get("t_depth", 0))
                        a = int(m.get("t_alt_count", 0))
                        if d > 0:
                            vaf_str = f" VAF={a/d:.1%}"
                    except (ValueError, TypeError):
                        pass
                    print(f"    {g:>8}: {m['protein_change']:<20} {m['variant_classification']}{vaf_str}")

    # ──────────────────────────────────────────────────────────────────
    # 8. CLOSEST MATCHES TO HENRIK
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("8. PATIENTS MOST SIMILAR TO HENRIK'S PROFILE")
    print(f"   Target: DNMT3A R882H, IDH2 R140Q, PTPN11 E76Q, SETBP1 G870S")
    print("─" * 80)

    # Score each patient by how many of Henrik's exact variants they carry
    patient_scores = []
    for pid, gene_vars in patient_variant_set.items():
        score = 0
        matches = []
        for gene, target_var in HENRIK_VARIANTS.items():
            if target_var in gene_vars.get(gene, set()):
                score += 1
                matches.append(f"{gene}:{target_var}")
        if score >= 2:  # At least 2 of Henrik's variants
            patient_scores.append((pid, score, matches))

    patient_scores.sort(key=lambda x: -x[1])

    for pid, score, matches in patient_scores[:30]:
        prow = patients.get(pid, {})
        age = prow.get("AGE_AT_SEQ_REPORT", "?")
        sex = prow.get("SEX", "?")
        # List other target gene mutations this patient has
        other_muts = []
        for g in TARGET_GENES:
            if g in patient_variant_set[pid]:
                vars_str = ",".join(patient_variant_set[pid][g])
                other_muts.append(f"{g}:{vars_str}")
        all_muts = " | ".join(other_muts)
        print(f"  {score}/4 match: {pid} ({sex}, age {age})")
        print(f"    Matches: {', '.join(matches)}")
        print(f"    All target: {all_muts}")

    # Save results
    output = {
        "analysis": "Variant-level deep analysis - GENIE v19.0",
        "index_matches": {k: len(v) for k, v in index_matches.items()},
        "triple_exact_match": len(r882h_pids & r140q_pids & e76q_pids),
        "quad_exact_match": len(quad_match),
        "setbp1_ski_domain_carriers": ski_total,
        "setbp1_non_ski_carriers": nonski_total,
        "top_variants_per_gene": {
            g: dict(variant_counts[g].most_common(20)) for g in TARGET_GENES
        },
        "patients_2plus_index_matches": len(patient_scores),
    }

    output_path = RESULTS_DIR / "genie_variant_level.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
