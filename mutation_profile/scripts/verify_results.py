#!/usr/bin/env python3
"""
verify_results.py -- Verify analysis results against known reference values.

Loads result files from mutation_profile/results/ and checks each
value against the targets documented in CLAUDE.md. Prints PASS/FAIL
for every check with the actual vs expected values and tolerance used.

Inputs:
    - mutation_profile/results/cooccurrence/four_gene_cooccurrence.json
    - mutation_profile/results/cooccurrence/myeloid_pairwise_results.json
    - mutation_profile/results/setbp1_makishima/setbp1_cooccurrence_matrix.json
    - mutation_profile/results/esm2_variant_scoring/esm2_results.json
    - mutation_profile/results/cross_database/cross_database_results.json
    - mutation_profile/results/ai_research/oncokb_annotations.json
    - mutation_profile/results/ai_research/civic_annotations.json
    - mutation_profile/results/ai_research/dgidb_interactions.json
    - mutation_profile/results/ai_research/gnomad_v4_results.json
    - mutation_profile/results/ai_research/clonal_tree_results.json
    - mutation_profile/results/ai_research/string_network.json
    - mutation_profile/results/ai_research/clingen_validity.json

Outputs:
    Prints PASS/FAIL to stdout (no files written).

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/verify_results.py [--strict]

Runtime: <1 second
Dependencies: pandas
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"
COOCCURRENCE_DIR = RESULTS_DIR / "cooccurrence"
CROSS_DB_DIR = RESULTS_DIR / "cross_database"
ESM2_DIR = RESULTS_DIR / "esm2_variant_scoring"
SETBP1_DIR = RESULTS_DIR / "setbp1_makishima"
AI_RESEARCH_DIR = RESULTS_DIR / "ai_research"


# ---------------------------------------------------------------------------
# Verification targets
# ---------------------------------------------------------------------------
# Each target: (description, expected, tolerance_type, tolerance_value)
#   tolerance_type: "abs" = absolute, "pct" = percent, "exact" = must match
TARGETS: list[dict[str, Any]] = [
    # --- Sample counts ---
    {
        "name": "Total GENIE samples",
        "expected": 271_837,
        "tolerance": ("abs", 500),
        "file_pattern": "cooccurrence/*summary*",
        "json_key": "total_genie_samples",
        "alt_keys": ["total_samples"],
    },
    {
        "name": "Myeloid samples",
        "expected": 27_585,
        "tolerance": ("abs", 200),
        "file_pattern": "cooccurrence/*summary*",
        "json_key": "myeloid_samples",
        "alt_keys": ["myeloid_count"],
    },
    {
        "name": "SETBP1 patients",
        "expected": 271,
        "tolerance": ("abs", 10),
        "file_pattern": "cooccurrence/*setbp1*",
        "json_key": "setbp1_patients",
        "alt_keys": ["n_setbp1", "setbp1_count"],
    },
    {
        "name": "Hypermutated excluded",
        "expected": 253,
        "tolerance": ("abs", 50),
        "file_pattern": "cooccurrence/*summary*",
        "json_key": "hypermutated_excluded",
        "alt_keys": ["excluded_hypermutated"],
    },
    # --- Four-gene co-occurrence ---
    {
        "name": "Quadruple co-occurrence (DNMT3A+IDH2+SETBP1+PTPN11)",
        "expected": 0,
        "tolerance": ("exact", 0),
        "file_pattern": "cooccurrence/*four_gene*",
        "json_key": "observed_quadruple",
        "alt_keys": ["quad_observed", "n_quadruple"],
    },
    # --- SETBP1 co-occurrence pairs ---
    {
        "name": "CSF3R+SETBP1 O/E ratio",
        "expected": 8.60,
        "tolerance": ("abs", 0.5),
        "file_pattern": "cooccurrence/*setbp1*",
        "json_key": None,
        "lookup": ("csv", "CSF3R", "obs_exp_ratio"),
    },
    {
        "name": "IDH1+SETBP1 O/E ratio",
        "expected": 0.13,
        "tolerance": ("abs", 0.05),
        "file_pattern": "cooccurrence/*setbp1*",
        "json_key": None,
        "lookup": ("csv", "IDH1", "obs_exp_ratio"),
    },
    {
        "name": "DDX41+SETBP1 co-occurrence count",
        "expected": 3,
        "tolerance": ("exact", 0),
        "file_pattern": "cooccurrence/*setbp1*",
        "json_key": None,
        "lookup": ("csv", "DDX41", "observed"),
    },
    # --- Pairwise matrix ---
    {
        "name": "FLT3+NPM1 O/E ratio",
        "expected": 8.18,
        "tolerance": ("abs", 0.5),
        "file_pattern": "cooccurrence/*pairwise*",
        "json_key": None,
        "lookup": ("csv_pair", "FLT3", "NPM1", "obs_exp_ratio"),
    },
    {
        "name": "Significant pairs after BH correction",
        "expected": 138,
        "tolerance": ("abs", 10),
        "file_pattern": "cooccurrence/*pairwise*",
        "json_key": "significant_pairs_bh",
        "alt_keys": ["n_significant"],
    },
    {
        "name": "Total gene pairs tested",
        "expected": 190,
        "tolerance": ("exact", 0),
        "file_pattern": "cooccurrence/*pairwise*",
        "json_key": "total_pairs",
        "alt_keys": ["n_pairs"],
    },
    # --- Cross-database ---
    {
        "name": "Patient 2642 exists with IDH2+PTPN11+SETBP1",
        "expected": True,
        "tolerance": ("exact", 0),
        "file_pattern": "cross_database/*",
        "json_key": "patient_2642_idh2_ptpn11_setbp1",
        "alt_keys": ["patient_2642"],
    },
    # --- ESM-2 variant scores ---
    {
        "name": "ESM-2 DNMT3A R882H score",
        "expected": -8.79,
        "tolerance": ("abs", 0.5),
        "file_pattern": "esm2_variant_scoring/*",
        "json_key": None,
        "lookup": ("csv_variant", "DNMT3A", "R882H", "esm2_score"),
    },
    {
        "name": "ESM-2 SETBP1 G870S score",
        "expected": -10.10,
        "tolerance": ("abs", 0.5),
        "file_pattern": "esm2_variant_scoring/*",
        "json_key": None,
        "lookup": ("csv_variant", "SETBP1", "G870S", "esm2_score"),
    },
    {
        "name": "ESM-2 EZH2 V662A score",
        "expected": -3.18,
        "tolerance": ("abs", 0.5),
        "file_pattern": "esm2_variant_scoring/*",
        "json_key": None,
        "lookup": ("csv_variant", "EZH2", "V662A", "esm2_score"),
    },
]


# ---------------------------------------------------------------------------
# AI Research verification checks (added 2026-03-27)
# These are run separately via dedicated JSON-loading functions.
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None on failure."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def run_ai_research_verification() -> tuple[int, int, int]:
    """
    Run verification checks for AI research result files.

    Returns (passed, failed, skipped) counts.
    """
    passed = 0
    failed = 0
    skipped = 0

    checks: list[tuple[str, bool | None, str]] = []

    # --- OncoKB ---
    oncokb = _load_json(AI_RESEARCH_DIR / "oncokb_annotations.json")
    if oncokb is None:
        checks.append(("OncoKB: file exists", None, "file not found"))
    else:
        annotations = oncokb.get("annotations", [])
        # IDH2 R140Q should be Level 1
        idh2_entry = next(
            (a for a in annotations if a.get("gene") == "IDH2" and "R140Q" in a.get("protein_change", "")),
            None,
        )
        if idh2_entry:
            level = idh2_entry.get("highest_sensitive_level")
            ok = level == "LEVEL_1"
            checks.append((
                "OncoKB: IDH2 R140Q highest_sensitive_level = LEVEL_1",
                ok,
                f"actual={level}, expected=LEVEL_1",
            ))
        else:
            checks.append(("OncoKB: IDH2 R140Q entry exists", False, "NOT FOUND"))

        # All 5 variants should have oncogenicity classifications
        genes_with_oncogenicity = [
            a["gene"] for a in annotations
            if a.get("oncogenicity") is not None
        ]
        ok = len(genes_with_oncogenicity) == 5
        checks.append((
            "OncoKB: all 5 variants have oncogenicity",
            ok,
            f"actual={len(genes_with_oncogenicity)}, expected=5, genes={genes_with_oncogenicity}",
        ))

        # Summary check
        summary = oncokb.get("summary", {})
        total = summary.get("total_variants", 0)
        checks.append((
            "OncoKB: total_variants = 5",
            total == 5,
            f"actual={total}, expected=5",
        ))

    # --- CIViC ---
    civic = _load_json(AI_RESEARCH_DIR / "civic_annotations.json")
    if civic is None:
        checks.append(("CIViC: file exists", None, "file not found"))
    else:
        results = civic.get("results", [])
        queried_genes = {r["gene"] for r in results}
        expected_genes = {"DNMT3A", "IDH2", "SETBP1", "PTPN11", "EZH2"}
        ok = expected_genes.issubset(queried_genes)
        checks.append((
            "CIViC: query results for all 5 genes",
            ok,
            f"actual={sorted(queried_genes)}, expected={sorted(expected_genes)}",
        ))
        # Variants queried count
        n_queried = civic.get("variants_queried", 0)
        checks.append((
            "CIViC: variants_queried = 5",
            n_queried == 5,
            f"actual={n_queried}, expected=5",
        ))

    # --- DGIdb ---
    dgidb = _load_json(AI_RESEARCH_DIR / "dgidb_interactions.json")
    if dgidb is None:
        checks.append(("DGIdb: file exists", None, "file not found"))
    else:
        total_interactions = dgidb.get("total_interactions", 0)
        if total_interactions == 0:
            total_interactions = dgidb.get("summary", {}).get("total_interactions", 0)
        checks.append((
            "DGIdb: total_interactions = 76",
            total_interactions == 76,
            f"actual={total_interactions}, expected=76",
        ))
        # SETBP1 should have 0 drug interactions
        genes_data = dgidb.get("genes", {})
        setbp1_count = genes_data.get("SETBP1", {}).get("interaction_count", -1)
        checks.append((
            "DGIdb: SETBP1 interaction_count = 0",
            setbp1_count == 0,
            f"actual={setbp1_count}, expected=0",
        ))

    # --- gnomAD v4 ---
    gnomad = _load_json(AI_RESEARCH_DIR / "gnomad_v4_results.json")
    if gnomad is None:
        checks.append(("gnomAD v4: file exists", None, "file not found"))
    else:
        results = gnomad.get("results", [])
        for gene, expect_absent in [("EZH2", True), ("SETBP1", True), ("PTPN11", True)]:
            entry = next((r for r in results if r.get("gene") == gene), None)
            if entry is None:
                checks.append((f"gnomAD v4: {gene} entry exists", False, "NOT FOUND"))
            else:
                found = entry.get("gnomad_found", True)
                pm2 = entry.get("pm2_classification", {}).get("strength", "")
                ok = not found and pm2 == "PM2_Strong"
                checks.append((
                    f"gnomAD v4: {gene} absent (PM2_Strong)",
                    ok,
                    f"gnomad_found={found}, pm2={pm2}",
                ))

        # DNMT3A should be present (CHIP)
        dnmt3a = next((r for r in results if r.get("gene") == "DNMT3A"), None)
        if dnmt3a is None:
            checks.append(("gnomAD v4: DNMT3A entry exists", False, "NOT FOUND"))
        else:
            found = dnmt3a.get("gnomad_found", False)
            checks.append((
                "gnomAD v4: DNMT3A present (CHIP variant)",
                found is True,
                f"gnomad_found={found}",
            ))

    # --- Clonal tree ---
    clonal = _load_json(AI_RESEARCH_DIR / "clonal_tree_results.json")
    if clonal is None:
        checks.append(("Clonal tree: file exists", None, "file not found"))
    else:
        purity = clonal.get("tumor_purity", 0)
        ok = abs(purity - 0.78) < 0.05
        checks.append((
            "Clonal tree: tumor_purity ~0.78",
            ok,
            f"actual={purity:.4f}, expected=0.78, tol=0.05",
        ))

        ccf_estimates = clonal.get("ccf_estimates", [])
        # DNMT3A CCF ~1.0
        dnmt3a_ccf = next(
            (c for c in ccf_estimates if c.get("gene") == "DNMT3A"), None
        )
        if dnmt3a_ccf:
            ccf_val = dnmt3a_ccf.get("ccf_point", 0)
            ok = abs(ccf_val - 1.0) < 0.05
            checks.append((
                "Clonal tree: DNMT3A CCF ~1.0",
                ok,
                f"actual={ccf_val:.4f}, expected=1.0, tol=0.05",
            ))
        else:
            checks.append(("Clonal tree: DNMT3A CCF entry", False, "NOT FOUND"))

        # IDH2 CCF ~0.05
        idh2_ccf = next(
            (c for c in ccf_estimates if c.get("gene") == "IDH2"), None
        )
        if idh2_ccf:
            ccf_val = idh2_ccf.get("ccf_point", 0)
            ok = abs(ccf_val - 0.05) < 0.02
            checks.append((
                "Clonal tree: IDH2 CCF ~0.05",
                ok,
                f"actual={ccf_val:.4f}, expected=0.05, tol=0.02",
            ))
        else:
            checks.append(("Clonal tree: IDH2 CCF entry", False, "NOT FOUND"))

    # --- STRING network ---
    string_data = _load_json(AI_RESEARCH_DIR / "string_network.json")
    if string_data is None:
        checks.append(("STRING: file exists", None, "file not found"))
    else:
        # Check that all 5 patient genes appear in the pairwise_scores
        pairwise = string_data.get("pairwise_scores", {})
        genes_in_network: set[str] = set()
        for pair_key in pairwise:
            parts = pair_key.split("-")
            genes_in_network.update(parts)
        # Also check interactions list for gene names
        for interaction in string_data.get("interactions", []):
            genes_in_network.add(interaction.get("preferredName_A", ""))
            genes_in_network.add(interaction.get("preferredName_B", ""))

        expected_genes = {"DNMT3A", "IDH2", "SETBP1", "PTPN11", "EZH2"}
        ok = expected_genes.issubset(genes_in_network)
        checks.append((
            "STRING: all 5 patient genes in network",
            ok,
            f"found={sorted(expected_genes & genes_in_network)}, missing={sorted(expected_genes - genes_in_network)}",
        ))

    # --- ClinGen ---
    clingen = _load_json(AI_RESEARCH_DIR / "clingen_validity.json")
    if clingen is None:
        checks.append(("ClinGen: file exists", None, "file not found"))
    else:
        genes_data = clingen.get("genes", {})
        # Count total gene-disease assertions
        total_assertions = sum(
            len(v.get("curated_reference", []))
            for v in genes_data.values()
        )
        checks.append((
            "ClinGen: 12 gene-disease assertions",
            total_assertions == 12,
            f"actual={total_assertions}, expected=12",
        ))
        # All 5 genes represented
        represented = set(genes_data.keys())
        expected_genes = {"DNMT3A", "IDH2", "SETBP1", "PTPN11", "EZH2"}
        ok = expected_genes.issubset(represented)
        checks.append((
            "ClinGen: all 5 genes represented",
            ok,
            f"found={sorted(represented)}, expected={sorted(expected_genes)}",
        ))

    # --- File count ---
    if AI_RESEARCH_DIR.exists():
        file_count = sum(1 for _ in AI_RESEARCH_DIR.rglob("*") if _.is_file())
        ok = file_count >= 165
        checks.append((
            f"AI research file count >= 165",
            ok,
            f"actual={file_count}, threshold=165",
        ))
    else:
        checks.append(("AI research directory exists", None, "directory not found"))

    # --- Print results ---
    print()
    print("=" * 72)
    print("AI RESEARCH VERIFICATION")
    print("=" * 72)
    print()

    for name, result, detail in checks:
        if result is None:
            marker = "\033[33mSKIP\033[0m"
            skipped += 1
        elif result:
            marker = "\033[32mPASS\033[0m"
            passed += 1
        else:
            marker = "\033[31mFAIL\033[0m"
            failed += 1
        print(f"  [{marker}] {name}")
        print(f"         {detail}")

    return passed, failed, skipped


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_result_files(pattern: str) -> list[Path]:
    """Find result files matching a glob pattern under RESULTS_DIR."""
    parts = pattern.split("/")
    if len(parts) == 2:
        subdir, file_glob = parts
        search_dir = RESULTS_DIR / subdir
    else:
        search_dir = RESULTS_DIR
        file_glob = pattern

    if not search_dir.exists():
        return []

    return sorted(search_dir.glob(file_glob))


def load_json_value(files: list[Path], key: str, alt_keys: list[str] | None = None) -> Any:
    """Try to extract a value from JSON files using key or alt_keys."""
    all_keys = [key] + (alt_keys or [])

    for fpath in files:
        if fpath.suffix != ".json":
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        for k in all_keys:
            if k in data:
                return data[k]
            # Try nested: check top-level dict values that are dicts
            for v in data.values():
                if isinstance(v, dict) and k in v:
                    return v[k]

    return None


def load_csv_value(
    files: list[Path], gene: str, column: str
) -> Optional[float]:
    """Find a gene row in a CSV/TSV result file and return the column value."""
    for fpath in files:
        if fpath.suffix not in (".csv", ".tsv", ".txt"):
            continue
        try:
            sep = "\t" if fpath.suffix in (".tsv", ".txt") else ","
            df = pd.read_csv(fpath, sep=sep)
        except Exception:
            continue

        # Look for gene column (various possible names)
        gene_col = None
        for col_name in ["gene", "Gene", "partner", "Partner", "gene_b", "Gene_B",
                         "gene_partner", "Hugo_Symbol"]:
            if col_name in df.columns:
                gene_col = col_name
                break
        if gene_col is None:
            continue

        row = df[df[gene_col].str.upper() == gene.upper()]
        if row.empty:
            continue

        # Find the value column (case-insensitive)
        val_col = None
        for c in df.columns:
            if c.lower().replace(" ", "_") == column.lower().replace(" ", "_"):
                val_col = c
                break
            if c.lower() == column.lower():
                val_col = c
                break
        if val_col is None:
            # Try common alternatives
            alt_map = {
                "obs_exp_ratio": ["o_e", "oe_ratio", "obs_exp", "observed_expected"],
                "observed": ["n_observed", "count", "n_cooccur", "co_occurrence_count"],
            }
            for alt in alt_map.get(column, []):
                for c in df.columns:
                    if c.lower().replace(" ", "_") == alt:
                        val_col = c
                        break
                if val_col:
                    break

        if val_col is None:
            continue

        val = row.iloc[0][val_col]
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

    return None


def load_csv_pair_value(
    files: list[Path], gene_a: str, gene_b: str, column: str
) -> Optional[float]:
    """Find a gene-pair row in a pairwise CSV and return the column value."""
    for fpath in files:
        if fpath.suffix not in (".csv", ".tsv", ".txt"):
            continue
        try:
            sep = "\t" if fpath.suffix in (".tsv", ".txt") else ","
            df = pd.read_csv(fpath, sep=sep)
        except Exception:
            continue

        # Identify gene_a / gene_b columns
        col_a, col_b = None, None
        for ca_name in ["gene_a", "Gene_A", "gene1", "Gene1"]:
            if ca_name in df.columns:
                col_a = ca_name
                break
        for cb_name in ["gene_b", "Gene_B", "gene2", "Gene2"]:
            if cb_name in df.columns:
                col_b = cb_name
                break
        if col_a is None or col_b is None:
            continue

        # Search both orderings
        mask = (
            (df[col_a].str.upper() == gene_a.upper())
            & (df[col_b].str.upper() == gene_b.upper())
        ) | (
            (df[col_a].str.upper() == gene_b.upper())
            & (df[col_b].str.upper() == gene_a.upper())
        )
        row = df[mask]
        if row.empty:
            continue

        # Find value column
        val_col = None
        for c in df.columns:
            if c.lower().replace(" ", "_") == column.lower().replace(" ", "_"):
                val_col = c
                break
        if val_col is None:
            alt_map = {
                "obs_exp_ratio": ["o_e", "oe_ratio", "obs_exp", "observed_expected"],
            }
            for alt in alt_map.get(column, []):
                for c in df.columns:
                    if c.lower().replace(" ", "_") == alt:
                        val_col = c
                        break
                if val_col:
                    break
        if val_col is None:
            continue

        val = row.iloc[0][val_col]
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

    return None


def load_csv_variant_value(
    files: list[Path], gene: str, variant: str, column: str
) -> Optional[float]:
    """Find a gene+variant row in an ESM-2 result CSV and return the score."""
    for fpath in files:
        if fpath.suffix not in (".csv", ".tsv", ".txt"):
            continue
        try:
            sep = "\t" if fpath.suffix in (".tsv", ".txt") else ","
            df = pd.read_csv(fpath, sep=sep)
        except Exception:
            continue

        # Find gene and variant columns
        gene_col, var_col = None, None
        for name in ["gene", "Gene", "Hugo_Symbol", "gene_name"]:
            if name in df.columns:
                gene_col = name
                break
        for name in ["variant", "Variant", "mutation", "Mutation", "hgvsp_short",
                      "HGVSp_Short", "aa_change", "protein_change"]:
            if name in df.columns:
                var_col = name
                break

        if gene_col is None or var_col is None:
            continue

        # Match gene and variant
        mask = (df[gene_col].str.upper() == gene.upper()) & (
            df[var_col].str.contains(variant, case=False, na=False)
        )
        row = df[mask]
        if row.empty:
            continue

        # Find score column
        val_col = None
        for c in df.columns:
            if c.lower().replace(" ", "_") == column.lower().replace(" ", "_"):
                val_col = c
                break
        if val_col is None:
            for c in df.columns:
                if "score" in c.lower() or "llr" in c.lower() or "esm" in c.lower():
                    val_col = c
                    break
        if val_col is None:
            continue

        val = row.iloc[0][val_col]
        try:
            return float(val)
        except (ValueError, TypeError):
            return val

    return None


def check_value(actual: Any, expected: Any, tolerance: tuple) -> tuple[bool, str]:
    """Compare actual vs expected with tolerance. Returns (passed, detail_str)."""
    if actual is None:
        return False, "NOT FOUND"

    tol_type, tol_val = tolerance

    if tol_type == "exact":
        if isinstance(expected, bool):
            passed = bool(actual) == expected
        else:
            passed = actual == expected
        return passed, f"actual={actual}, expected={expected}"

    try:
        actual_f = float(actual)
        expected_f = float(expected)
    except (ValueError, TypeError):
        return False, f"actual={actual} (not numeric), expected={expected}"

    if tol_type == "abs":
        diff = abs(actual_f - expected_f)
        passed = diff <= tol_val
        return passed, f"actual={actual_f:.4f}, expected={expected_f:.4f}, diff={diff:.4f}, tol={tol_val}"

    if tol_type == "pct":
        if expected_f == 0:
            passed = actual_f == 0
        else:
            diff_pct = abs(actual_f - expected_f) / abs(expected_f) * 100
            passed = diff_pct <= tol_val
        return passed, f"actual={actual_f:.4f}, expected={expected_f:.4f}, tol={tol_val}%"

    return False, f"unknown tolerance type: {tol_type}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_verification(strict: bool = False) -> tuple[int, int, int]:
    """
    Run all verification checks.

    Returns (passed, failed, skipped) counts.
    """
    passed = 0
    failed = 0
    skipped = 0

    print("=" * 72)
    print("GENIE ANALYSIS VERIFICATION")
    print("=" * 72)
    print(f"Results dir: {RESULTS_DIR}")
    print()

    for target in TARGETS:
        name = target["name"]
        expected = target["expected"]
        tolerance = target["tolerance"]

        # Find result files
        files = find_result_files(target["file_pattern"])

        # Extract actual value
        actual = None

        if target.get("json_key"):
            actual = load_json_value(
                files, target["json_key"], target.get("alt_keys")
            )

        if actual is None and target.get("lookup"):
            lookup = target["lookup"]
            if lookup[0] == "csv":
                _, gene, column = lookup
                actual = load_csv_value(files, gene, column)
            elif lookup[0] == "csv_pair":
                _, gene_a, gene_b, column = lookup
                actual = load_csv_pair_value(files, gene_a, gene_b, column)
            elif lookup[0] == "csv_variant":
                _, gene, variant, column = lookup
                actual = load_csv_variant_value(files, gene, variant, column)

        # If still None and we have alt_keys, try JSON with alt_keys
        if actual is None and target.get("alt_keys"):
            for alt in target["alt_keys"]:
                actual = load_json_value(files, alt)
                if actual is not None:
                    break

        # Check
        if actual is None and not files:
            status = "SKIP"
            detail = "no result files found"
            skipped += 1
        elif actual is None:
            status = "SKIP"
            detail = f"value not found in {len(files)} file(s)"
            skipped += 1
        else:
            ok, detail = check_value(actual, expected, tolerance)
            if ok:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1

        # Color-coded output
        if status == "PASS":
            marker = "\033[32mPASS\033[0m"
        elif status == "FAIL":
            marker = "\033[31mFAIL\033[0m"
        else:
            marker = "\033[33mSKIP\033[0m"

        print(f"  [{marker}] {name}")
        print(f"         {detail}")

    # Run AI research verification
    ai_passed, ai_failed, ai_skipped = run_ai_research_verification()
    passed += ai_passed
    failed += ai_failed
    skipped += ai_skipped

    # Summary
    total = passed + failed + skipped
    print()
    print("=" * 72)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print("=" * 72)

    if failed > 0 and strict:
        print("\nStrict mode: exiting with error due to failures.")

    return passed, failed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Verify GENIE analysis results against known targets"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any check fails",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text",
    )
    args = parser.parse_args()

    passed, failed, skipped = run_verification(strict=args.strict)

    if args.json:
        result = {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": passed + failed + skipped,
            "all_passed": failed == 0,
        }
        print(json.dumps(result, indent=2))

    if args.strict and failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
