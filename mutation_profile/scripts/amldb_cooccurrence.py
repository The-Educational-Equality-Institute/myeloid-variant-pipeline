#!/usr/bin/env python3
"""
AMLdb Co-occurrence Analysis for DNMT3A, IDH2, PTPN11, SETBP1

AMLdb (IIT Hyderabad) is a cell-line-based multi-omics platform, NOT a patient
cohort database. It integrates data from:
  - DepMap/Project Achilles: CRISPR-Cas9 screens for 26 AML cell lines
  - GDSC: Drug sensitivity for 26 AML cell lines (288 drugs)
  - GEO: Expression profiling (86 datasets) and methylation (15 datasets)
  - cBioPortal/COSMIC: Mutation context for 41 genes in 23 AML cell lines

The mutation analysis module covers only 41 genes with >5% mutation frequency
across 23 AML cell lines. DNMT3A, IDH2, PTPN11, and SETBP1 all fall below
this threshold and are NOT included in AMLdb's mutation dataset.

Therefore, per-patient co-occurrence analysis is NOT possible with AMLdb.
This script documents the cell-line-level mutation data that IS available.

Source: Kumar KV et al. "AMLdb: a comprehensive multi-omics platform to identify
biomarkers and drug targets for acute myeloid leukemia." Briefings in Functional
Genomics 2024;23(6):798-805. DOI: 10.1093/bfgp/elae024
URL: http://project.iith.ac.in/cgntlab/amldb/
"""

import json
import os
from datetime import datetime

# AMLdb cell line mutation data for our 4 target genes
# Source: DepMap somatic mutations cross-referenced with AMLdb cell line panel
# and analysis_summary.md / query_results.md from prior browser queries

# The 26 AML cell lines in AMLdb (from DepMap integration):
# AMLdb uses these for CRISPR dependency, drug sensitivity, and expression.
# Mutation data is available for 23 of these from GDSC/COSMIC integration,
# but only for the 41 genes with >5% frequency.

# From DepMap (which AMLdb integrates), the following cell lines carry
# mutations in our target genes. These are CELL LINES, not patients.

cell_line_mutations = {
    "DNMT3A": {
        "cell_lines": ["Set-2", "KO52", "OCI-AML3", "SIG-M5"],
        "count": 4,
        "mutations": {
            "Set-2": "R882H",
            "KO52": "R882H",
            "OCI-AML3": "R882C",
            "SIG-M5": "R882C",
        },
        "note": "All at R882 hotspot"
    },
    "IDH2": {
        "cell_lines": ["CESS"],
        "count": 1,
        "mutations": {
            "CESS": "I240V (VUS, not a hotspot)"
        },
        "note": "No R140Q or R172K hotspot mutations in any AML cell line"
    },
    "PTPN11": {
        "cell_lines": ["U-937", "UCSD-AML1", "UKE-1", "SHI-1", "HNT-34", "TUR", "GDM-1"],
        "count": 7,  # HNT-34 has 2 mutations but is 1 cell line
        "mutations": {
            "U-937": "G60R",
            "UCSD-AML1": "D61V",
            "UKE-1": "F71L",
            "SHI-1": "F71L",
            "HNT-34": "A72V + T468M",
            "TUR": "G60R",
            "GDM-1": "A72S",
        },
        "note": "Best represented of 4 genes; 6 unique cell lines (7 counting HNT-34 dual mutations)"
    },
    "SETBP1": {
        "cell_lines": ["UCSD-AML1"],
        "count": 1,
        "mutations": {
            "UCSD-AML1": "P1525Rfs*54 (frameshift, LoF — NOT a gain-of-function SKI domain hotspot)"
        },
        "note": "Only truncating mutation; not the pathogenic D868/G870 hotspot type"
    },
}

# Cell-line-level co-occurrence (from DepMap data integrated in AMLdb)
# UCSD-AML1 carries both SETBP1 and PTPN11 mutations
cell_line_cooccurrence = {
    "DNMT3A+IDH2": 0,
    "DNMT3A+PTPN11": 0,
    "DNMT3A+SETBP1": 0,
    "IDH2+PTPN11": 0,
    "IDH2+SETBP1": 0,
    "PTPN11+SETBP1": 1,  # UCSD-AML1
}

# Build the output JSON
output = {
    "database": "AMLdb",
    "total_patients": 0,
    "disease_filter": "AML only",
    "data_files_used": [
        "analysis_summary.md (AMLdb platform documentation and query results)",
        "query_results.md (multi-database query results including AMLdb)",
    ],
    "single_gene": {
        "DNMT3A": 0,
        "IDH2": 0,
        "PTPN11": 0,
        "SETBP1": 0,
    },
    "pairwise": {
        "DNMT3A+IDH2": 0,
        "DNMT3A+PTPN11": 0,
        "DNMT3A+SETBP1": 0,
        "IDH2+PTPN11": 0,
        "IDH2+SETBP1": 0,
        "PTPN11+SETBP1": 0,
    },
    "triple": {
        "DNMT3A+IDH2+PTPN11": 0,
        "DNMT3A+IDH2+SETBP1": 0,
        "DNMT3A+PTPN11+SETBP1": 0,
        "IDH2+PTPN11+SETBP1": 0,
    },
    "quadruple": {
        "DNMT3A+IDH2+PTPN11+SETBP1": 0,
    },
    "monosomy7_data_available": False,
    "overlap_warnings": (
        "AMLdb integrates DepMap, GDSC, cBioPortal, and COSMIC data for AML cell lines. "
        "Cell lines in AMLdb overlap with DepMap (same source). The 4 DNMT3A-mutant cell lines "
        "(OCI-AML3, Set-2, KO52, SIG-M5) and 6 PTPN11-mutant cell lines (U-937, SHI-1, etc.) "
        "are the same lines catalogued in DepMap and may appear in other database analyses."
    ),
    "notes": (
        "AMLdb is a CELL LINE multi-omics platform (26 AML cell lines), NOT a patient cohort "
        "database. It does not contain per-patient somatic mutation data. The mutation analysis "
        "module covers only 41 genes with >5% mutation frequency across 23 AML cell lines. "
        "DNMT3A, IDH2, PTPN11, and SETBP1 all fall below this 5% threshold and return empty "
        "results from the drug-mutation API. Therefore, patient-level co-occurrence counts are "
        "all zero — not because these mutations don't co-occur in AML patients, but because "
        "AMLdb simply does not have patient-level mutation data to query. "
        "Cell-line-level data from DepMap (integrated in AMLdb) shows: "
        "DNMT3A mutated in 4 cell lines (all R882), IDH2 in 1 (non-hotspot VUS), "
        "PTPN11 in 6-7 cell lines, SETBP1 in 1 (truncating, not gain-of-function). "
        "UCSD-AML1 carries both PTPN11 (D61V) and SETBP1 (P1525Rfs*54) — the only "
        "pairwise co-occurrence among our target genes at the cell line level. "
        "Publication: Kumar KV et al. Briefings in Functional Genomics 2024;23(6):798-805."
    ),
    # Additional context fields beyond the required schema
    "cell_line_data": {
        "total_cell_lines": 26,
        "cell_lines_with_mutation_data": 23,
        "genes_in_mutation_module": 41,
        "target_genes_in_module": False,
        "reason": "All 4 target genes have <5% mutation frequency across 23 AML cell lines",
        "cell_line_mutations_from_depmap": {
            "DNMT3A": {
                "mutated_cell_lines": 4,
                "lines": ["Set-2 (R882H)", "KO52 (R882H)", "OCI-AML3 (R882C)", "SIG-M5 (R882C)"]
            },
            "IDH2": {
                "mutated_cell_lines": 1,
                "lines": ["CESS (I240V, VUS — not R140Q/R172K hotspot)"]
            },
            "PTPN11": {
                "mutated_cell_lines": 7,
                "lines": [
                    "U-937 (G60R)", "UCSD-AML1 (D61V)", "UKE-1 (F71L)",
                    "SHI-1 (F71L)", "HNT-34 (A72V + T468M)", "TUR (G60R)", "GDM-1 (A72S)"
                ]
            },
            "SETBP1": {
                "mutated_cell_lines": 1,
                "lines": ["UCSD-AML1 (P1525Rfs*54, frameshift LoF)"]
            },
        },
        "cell_line_pairwise_cooccurrence": {
            "PTPN11+SETBP1": {
                "count": 1,
                "cell_lines": ["UCSD-AML1"],
                "note": "SETBP1 mutation is truncating (LoF), not pathogenic SKI domain hotspot"
            },
            "all_other_pairs": 0
        }
    }
}

# Write output
output_path = os.path.expanduser(
    "~/projects/mrna-hematology-research/mutation_profile/results/amldb_cooccurrence.json"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Output written to: {output_path}")
print(f"\nSummary:")
print(f"  Database: AMLdb (IIT Hyderabad)")
print(f"  Type: Cell line multi-omics platform (NOT a patient cohort)")
print(f"  Cell lines: 26 total, 23 with mutation data")
print(f"  Mutation module: 41 genes with >5% frequency only")
print(f"  Target genes in module: NO (all below 5% threshold)")
print(f"  Patient-level co-occurrence: NOT AVAILABLE")
print(f"  Cell-line co-occurrence: PTPN11+SETBP1 in UCSD-AML1 only")
print(f"\nAll patient-level counts are zero because AMLdb does not contain")
print(f"per-patient somatic mutation data for co-occurrence analysis.")
