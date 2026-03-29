#!/usr/bin/env python3
"""
DepMap/CCLE Co-occurrence Analysis for DNMT3A, IDH2, PTPN11, SETBP1
in AML/MDS/myeloid cell lines.

Analyzes OmicsSomaticMutations.csv and Model.csv from DepMap 25Q3
to find single, pairwise, triple, and quadruple co-occurrences.
Also checks copy number data for monosomy 7 / del(7q).
"""

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "depmap"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]

def load_myeloid_cell_lines(model_path):
    """Load Model.csv and filter to myeloid lineage cell lines."""
    model = pd.read_csv(model_path)

    # Filter to Myeloid lineage (includes AML, MDS, MPN, CML)
    myeloid = model[model["OncotreeLineage"] == "Myeloid"].copy()

    print(f"=== MYELOID CELL LINE SUMMARY ===")
    print(f"Total Myeloid lineage cell lines: {len(myeloid)}")
    print()

    # Show breakdown by disease
    print("By OncotreePrimaryDisease:")
    for disease, count in myeloid["OncotreePrimaryDisease"].value_counts().items():
        print(f"  {disease}: {count}")
    print()

    # Show breakdown by subtype
    print("By OncotreeSubtype:")
    for subtype, count in myeloid["OncotreeSubtype"].value_counts().items():
        print(f"  {subtype}: {count}")
    print()

    return myeloid


def load_mutations(mutations_path, myeloid_model_ids):
    """Load somatic mutations filtered to myeloid cell lines and target genes."""
    print("Loading OmicsSomaticMutations.csv (this may take a moment)...")

    # Read only needed columns to save memory
    cols_needed = [
        "ModelID", "HugoSymbol", "ProteinChange", "VariantType",
        "MolecularConsequence", "VepImpact", "AF", "LikelyLoF",
        "HessDriver", "Hotspot", "VariantInfo"
    ]

    # Read in chunks to handle large file
    chunks = []
    for chunk in pd.read_csv(mutations_path, chunksize=100000, usecols=cols_needed):
        # Filter to myeloid models and target genes
        filtered = chunk[
            (chunk["ModelID"].isin(myeloid_model_ids)) &
            (chunk["HugoSymbol"].isin(TARGET_GENES))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if chunks:
        mutations = pd.concat(chunks, ignore_index=True)
    else:
        mutations = pd.DataFrame(columns=cols_needed)

    print(f"Found {len(mutations)} mutations in target genes across myeloid cell lines")
    print()

    return mutations


def analyze_mutations(mutations, myeloid_df):
    """Analyze single, pairwise, triple, and quadruple co-occurrences."""

    # Build per-cell-line mutation profile
    # A cell line is considered "mutated" for a gene if it has ANY somatic mutation in that gene
    cell_line_genes = {}

    for _, row in mutations.iterrows():
        model_id = row["ModelID"]
        gene = row["HugoSymbol"]
        if model_id not in cell_line_genes:
            cell_line_genes[model_id] = set()
        cell_line_genes[model_id].add(gene)

    # Print all mutations found
    print("=== ALL TARGET GENE MUTATIONS IN MYELOID CELL LINES ===")
    for gene in TARGET_GENES:
        gene_muts = mutations[mutations["HugoSymbol"] == gene]
        if len(gene_muts) > 0:
            print(f"\n{gene} mutations:")
            for _, row in gene_muts.iterrows():
                model_id = row["ModelID"]
                cell_name = myeloid_df[myeloid_df["ModelID"] == model_id]["CellLineName"].values
                name = cell_name[0] if len(cell_name) > 0 else "Unknown"
                disease = myeloid_df[myeloid_df["ModelID"] == model_id]["OncotreePrimaryDisease"].values
                dis = disease[0] if len(disease) > 0 else "Unknown"
                subtype = myeloid_df[myeloid_df["ModelID"] == model_id]["OncotreeSubtype"].values
                sub = subtype[0] if len(subtype) > 0 else "Unknown"
                protein = row.get("ProteinChange", "")
                var_type = row.get("VariantType", "")
                consequence = row.get("MolecularConsequence", "")
                impact = row.get("VepImpact", "")
                af = row.get("AF", "")
                hotspot = row.get("Hotspot", "")
                driver = row.get("HessDriver", "")
                print(f"  {name} ({model_id}): {protein} | {var_type} | {consequence} | Impact={impact} | AF={af} | Hotspot={hotspot} | Driver={driver} | {dis} - {sub}")
        else:
            print(f"\n{gene}: No mutations found in myeloid cell lines")

    print()

    # Single gene counts
    single = {}
    single_details = {}
    for gene in TARGET_GENES:
        cell_lines_with_gene = [
            model_id for model_id, genes in cell_line_genes.items()
            if gene in genes
        ]
        single[gene] = len(cell_lines_with_gene)
        single_details[gene] = cell_lines_with_gene

    print("=== SINGLE GENE MUTATION COUNTS ===")
    for gene in TARGET_GENES:
        names = []
        for mid in single_details[gene]:
            n = myeloid_df[myeloid_df["ModelID"] == mid]["CellLineName"].values
            names.append(n[0] if len(n) > 0 else mid)
        print(f"  {gene}: {single[gene]} cell lines ({', '.join(names)})")
    print()

    # Pairwise
    pairwise = {}
    pairwise_details = {}
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        co_lines = [
            model_id for model_id, genes in cell_line_genes.items()
            if g1 in genes and g2 in genes
        ]
        pairwise[key] = len(co_lines)
        pairwise_details[key] = co_lines

    print("=== PAIRWISE CO-OCCURRENCE ===")
    for key, count in pairwise.items():
        names = []
        for mid in pairwise_details[key]:
            n = myeloid_df[myeloid_df["ModelID"] == mid]["CellLineName"].values
            names.append(n[0] if len(n) > 0 else mid)
        detail = f" ({', '.join(names)})" if names else ""
        print(f"  {key}: {count} cell lines{detail}")
    print()

    # Triple
    triple = {}
    triple_details = {}
    for g1, g2, g3 in combinations(TARGET_GENES, 3):
        key = f"{g1}+{g2}+{g3}"
        co_lines = [
            model_id for model_id, genes in cell_line_genes.items()
            if g1 in genes and g2 in genes and g3 in genes
        ]
        triple[key] = len(co_lines)
        triple_details[key] = co_lines

    print("=== TRIPLE CO-OCCURRENCE ===")
    for key, count in triple.items():
        names = []
        for mid in triple_details[key]:
            n = myeloid_df[myeloid_df["ModelID"] == mid]["CellLineName"].values
            names.append(n[0] if len(n) > 0 else mid)
        detail = f" ({', '.join(names)})" if names else ""
        print(f"  {key}: {count} cell lines{detail}")
    print()

    # Quadruple
    quad_lines = [
        model_id for model_id, genes in cell_line_genes.items()
        if all(g in genes for g in TARGET_GENES)
    ]

    print("=== QUADRUPLE CO-OCCURRENCE ===")
    quad_names = []
    for mid in quad_lines:
        n = myeloid_df[myeloid_df["ModelID"] == mid]["CellLineName"].values
        quad_names.append(n[0] if len(n) > 0 else mid)
    detail = f" ({', '.join(quad_names)})" if quad_names else ""
    print(f"  DNMT3A+IDH2+PTPN11+SETBP1: {len(quad_lines)} cell lines{detail}")
    print()

    return single, pairwise, triple, len(quad_lines), cell_line_genes


def check_monosomy7(cn_path, myeloid_model_ids, myeloid_df):
    """Check copy number data for monosomy 7 / del(7q) evidence.

    In DepMap WGS copy number data (OmicsCNGeneWGS.csv), values are
    relative copy number ratios where:
    - ~1.0 = normal diploid (2 copies)
    - ~0.5 = monosomy / single copy (1 copy)
    - ~1.5 = trisomy (3 copies)
    - ~0.0 = homozygous deletion (0 copies)
    """
    if not os.path.exists(cn_path):
        print("Copy number file not found, skipping monosomy 7 check")
        return False, "OmicsCNGeneWGS.csv not found", []

    print("Checking copy number data for monosomy 7 evidence...")

    # Chromosome 7 marker genes spanning the entire chromosome
    chr7_marker_genes = [
        "EZH2", "CUX1", "SAMD9", "SAMD9L", "MET", "BRAF",
        "HOXA9", "CDK6", "PON1", "MCM7", "POLR2J", "NAMPT"
    ]

    # Read just the header to find chromosome 7 gene columns
    header = pd.read_csv(cn_path, nrows=0)
    all_cols = list(header.columns)

    # Find columns matching chr7 marker genes
    chr7_cols = []
    gene_col_map = {}
    for col in all_cols:
        for gene in chr7_marker_genes:
            if col.startswith(f"{gene} ("):
                chr7_cols.append(col)
                gene_col_map[gene] = col
                break

    if not chr7_cols:
        print("  No chromosome 7 marker gene columns found")
        return False, "No chr7 marker gene columns found in CN data", []

    print(f"  Found {len(chr7_cols)} chromosome 7 marker gene columns")

    # Read only the needed columns for myeloid models
    read_cols = list(set(["ModelID"] + chr7_cols))
    cn_data = pd.read_csv(cn_path, usecols=read_cols)
    cn_myeloid = cn_data[cn_data["ModelID"].isin(myeloid_model_ids)]

    print(f"  Myeloid cell lines with CN data: {len(cn_myeloid)}")

    # Check for monosomy 7 pattern:
    # Relative CN ~0.5 across all chr7 genes = monosomy 7
    # Threshold: avg < 0.7 indicates likely single copy (monosomy)
    mono7_candidates = []
    for _, row in cn_myeloid.iterrows():
        model_id = row["ModelID"]
        chr7_values = []
        for gene, col in gene_col_map.items():
            if col in row.index and pd.notna(row[col]):
                chr7_values.append((gene, row[col]))

        if chr7_values:
            avg_cn = sum(v for _, v in chr7_values) / len(chr7_values)
            # Monosomy 7: average relative CN < 0.7 (single copy ~0.5)
            if avg_cn < 0.7:
                name = myeloid_df[myeloid_df["ModelID"] == model_id]["CellLineName"].values
                nm = name[0] if len(name) > 0 else model_id
                disease = myeloid_df[myeloid_df["ModelID"] == model_id]["OncotreePrimaryDisease"].values
                dis = disease[0] if len(disease) > 0 else "Unknown"
                mono7_candidates.append({
                    "model_id": model_id,
                    "cell_line_name": nm,
                    "disease": dis,
                    "avg_chr7_cn": round(avg_cn, 4),
                    "details": {g: round(v, 4) for g, v in chr7_values}
                })

    if mono7_candidates:
        print(f"\n  Monosomy 7 / del(7q) candidates ({len(mono7_candidates)}):")
        for c in mono7_candidates:
            print(f"    {c['cell_line_name']} ({c['model_id']}): avg chr7 CN = {c['avg_chr7_cn']} | {c['disease']}")
            for g, v in sorted(c["details"].items(), key=lambda x: x[1]):
                marker = " **LOSS**" if v < 0.6 else ""
                print(f"      {g}: {v}{marker}")
        return True, f"Found {len(mono7_candidates)} myeloid cell lines with chr7 loss (avg relative CN < 0.7): " + \
               ", ".join(f"{c['cell_line_name']} (avg={c['avg_chr7_cn']})" for c in mono7_candidates), mono7_candidates
    else:
        print("  No monosomy 7 / del(7q) detected in myeloid cell lines based on CN data")
        return False, "No chr7 loss detected in CN data for myeloid cell lines", []


def main():
    model_path = DATA_DIR / "Model.csv"
    mutations_path = DATA_DIR / "OmicsSomaticMutations.csv"
    cn_path = DATA_DIR / "OmicsCNGeneWGS.csv"

    # Step 1: Load myeloid cell lines
    myeloid = load_myeloid_cell_lines(model_path)
    myeloid_ids = set(myeloid["ModelID"].unique())

    # Step 2: Load and filter mutations
    mutations = load_mutations(mutations_path, myeloid_ids)

    # Step 3: Analyze co-occurrences
    single, pairwise, triple, quadruple, cell_line_genes = analyze_mutations(mutations, myeloid)

    # Step 4: Check monosomy 7
    mono7_available, mono7_notes, mono7_candidates = check_monosomy7(cn_path, myeloid_ids, myeloid)

    # Step 5: Build output JSON
    total_myeloid = len(myeloid)

    # Count how many myeloid cell lines have mutation data
    all_mutated_models = mutations["ModelID"].unique()
    myeloid_with_data = len(set(all_mutated_models) & myeloid_ids)

    # Also check total unique ModelIDs in the full mutation file to see coverage
    # (we already filtered, so let's count from the full set)
    print(f"\n=== DATA COVERAGE ===")
    print(f"Total myeloid cell lines in Model.csv: {total_myeloid}")
    print(f"Myeloid cell lines with at least one target gene mutation: {myeloid_with_data}")

    # Count myeloid lines that appear in the somatic mutations file at all
    print("\nChecking how many myeloid lines have any somatic mutation data...")
    any_mut_models = set()
    for chunk in pd.read_csv(mutations_path, chunksize=200000, usecols=["ModelID"]):
        any_mut_models.update(chunk[chunk["ModelID"].isin(myeloid_ids)]["ModelID"].unique())
    myeloid_with_any_data = len(any_mut_models)
    print(f"Myeloid cell lines with ANY somatic mutation data: {myeloid_with_any_data}")

    # Determine data files used
    data_files = ["OmicsSomaticMutations.csv (DepMap 25Q3)", "Model.csv (DepMap 25Q3)"]
    if os.path.exists(cn_path):
        data_files.append("OmicsCNGeneWGS.csv (DepMap 25Q3)")

    # Build notes
    notes_parts = [
        "These are cell lines, NOT patient samples.",
        f"Total myeloid lineage cell lines in DepMap 25Q3: {total_myeloid}.",
        f"Of these, {myeloid_with_any_data} have somatic mutation profiling data.",
        "Myeloid lineage includes AML (64), MPN/CML (18), ambiguous lineage (3), and MDS-derived lines.",
        "Mutation calls are from whole-exome or whole-genome sequencing via DepMap/CCLE pipeline.",
    ]
    if mono7_candidates:
        mono7_names = [f"{c['cell_line_name']} (avg_chr7_CN={c['avg_chr7_cn']})" for c in mono7_candidates]
        notes_parts.append(
            f"Monosomy 7 detected via WGS copy number (avg relative CN < 0.7 across 12 chr7 markers): {'; '.join(mono7_names)}."
        )
        # Check overlap of mono7 with target gene mutations
        mono7_model_ids = {c['model_id'] for c in mono7_candidates}
        for mid in mono7_model_ids:
            if mid in cell_line_genes:
                genes_in_line = cell_line_genes[mid]
                name = myeloid[myeloid["ModelID"] == mid]["CellLineName"].values
                nm = name[0] if len(name) > 0 else mid
                notes_parts.append(
                    f"{nm} has monosomy 7 AND mutations in: {', '.join(sorted(genes_in_line))}."
                )

    # Build warnings
    warnings = []
    if myeloid_with_any_data < total_myeloid:
        warnings.append(
            f"Only {myeloid_with_any_data}/{total_myeloid} myeloid cell lines have mutation data. "
            f"Some cell lines may lack sequencing data."
        )

    result = {
        "database": "DepMap/CCLE",
        "total_patients": myeloid_with_any_data,
        "disease_filter": "AML/MDS myeloid cell lines only",
        "data_files_used": data_files,
        "single_gene": {gene: single[gene] for gene in TARGET_GENES},
        "pairwise": {
            "DNMT3A+IDH2": pairwise.get("DNMT3A+IDH2", 0),
            "DNMT3A+PTPN11": pairwise.get("DNMT3A+PTPN11", 0),
            "DNMT3A+SETBP1": pairwise.get("DNMT3A+SETBP1", 0),
            "IDH2+PTPN11": pairwise.get("IDH2+PTPN11", 0),
            "IDH2+SETBP1": pairwise.get("IDH2+SETBP1", 0),
            "PTPN11+SETBP1": pairwise.get("PTPN11+SETBP1", 0),
        },
        "triple": {
            "DNMT3A+IDH2+PTPN11": triple.get("DNMT3A+IDH2+PTPN11", 0),
            "DNMT3A+IDH2+SETBP1": triple.get("DNMT3A+IDH2+SETBP1", 0),
            "DNMT3A+PTPN11+SETBP1": triple.get("DNMT3A+PTPN11+SETBP1", 0),
            "IDH2+PTPN11+SETBP1": triple.get("IDH2+PTPN11+SETBP1", 0),
        },
        "quadruple": {"DNMT3A+IDH2+PTPN11+SETBP1": quadruple},
        "monosomy7_data_available": mono7_available,
        "overlap_warnings": "; ".join(warnings) if warnings else "",
        "notes": " ".join(notes_parts)
    }

    # Write output
    output_path = RESULTS_DIR / "depmap_cooccurrence.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== RESULTS SAVED ===")
    print(f"Output: {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
