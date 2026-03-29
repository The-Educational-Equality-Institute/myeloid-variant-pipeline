#!/usr/bin/env python3
"""
Athena Co-occurrence Queries for GENIE v19.0 + TCGA + ClinVar
==============================================================

Leverages existing AWS Athena infrastructure (genomics_db database,
us-east-1) to run SQL-based co-occurrence queries against:

1. GENIE v19.0 (3.4M mutations, uploaded from local data)
2. TCGA pan-cancer MAFs (21,300 files, already in Athena)
3. ClinVar variant annotations (already in Athena)

This replaces the Python-based flat-file processing in analyze_genie_filtered.py
with SQL queries, enabling faster iteration and cross-dataset joins.

Prerequisites:
  - AWS profile 'default' configured (~/.aws/credentials)
  - Athena database 'genomics_db' exists in us-east-1
  - S3 bucket 'athena-results' exists

Usage:
  python mutation_profile/scripts/athena_cooccurrence.py --setup     # Upload GENIE + create table
  python mutation_profile/scripts/athena_cooccurrence.py --query     # Run co-occurrence queries
  python mutation_profile/scripts/athena_cooccurrence.py --all       # Setup + query
  python mutation_profile/scripts/athena_cooccurrence.py --cross-ref # Cross-ref GENIE with TCGA/ClinVar
  python mutation_profile/scripts/athena_cooccurrence.py --dry-run   # Print SQL without executing

Author: Henrik Roine / TEEI
Date: 2026-03-22
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from datetime import datetime
from io import BytesIO
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
except ImportError:
    print("ERROR: boto3 not installed. pip install boto3")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GENIE_MAF = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw" / "data_mutations_extended.txt"
GENIE_CLINICAL = PROJECT_ROOT / "mutation_profile" / "data" / "genie" / "raw" / "data_clinical_sample.txt"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "athena"

AWS_PROFILE = "default"
AWS_REGION = "us-east-1"
S3_BUCKET = "athena-results"
ATHENA_DATABASE = "genomics_db"
ATHENA_WORKGROUP = "primary"

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]
ALL_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1", "EZH2"]

SPECIFIC_VARIANTS = {
    "DNMT3A": "R882H",
    "SETBP1": "G870S",
    "PTPN11": "E76Q",
    "IDH2": "R140Q",
    "EZH2": "V662A",
}

# Variant classifications to keep (protein-altering)
PATHOGENIC_VAR_CLASSES = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Nonstop_Mutation", "Translation_Start_Site",
}


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------

def get_session(profile: str = AWS_PROFILE, region: str = AWS_REGION) -> boto3.Session:
    """Create AWS session with profile."""
    try:
        return boto3.Session(profile_name=profile, region_name=region)
    except ProfileNotFound:
        print(f"ERROR: AWS profile '{profile}' not found.")
        print("  Configure with: aws configure --profile default")
        sys.exit(1)


def run_athena_query(client, sql: str, database: str = ATHENA_DATABASE,
                     output_location: str | None = None) -> list[list[str]]:
    """Execute Athena query and wait for results."""
    if output_location is None:
        output_location = f"s3://{S3_BUCKET}/athena-output/"

    resp = client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
        WorkGroup=ATHENA_WORKGROUP,
    )
    query_id = resp["QueryExecutionId"]
    print(f"  Query {query_id[:12]}... ", end="", flush=True)

    # Poll for completion
    while True:
        status = client.get_query_execution(QueryExecutionId=query_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)

    if state != "SUCCEEDED":
        reason = status["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
        print(f"FAILED: {reason}")
        return []

    # Fetch results
    rows = []
    paginator = client.get_paginator("get_query_results")
    for page in paginator.paginate(QueryExecutionId=query_id):
        for row in page["ResultSet"]["Rows"]:
            rows.append([col.get("VarCharValue", "") for col in row["Data"]])

    stats = status["QueryExecution"].get("Statistics", {})
    scanned_mb = stats.get("DataScannedInBytes", 0) / 1024 / 1024
    runtime_ms = stats.get("EngineExecutionTimeInMillis", 0)
    cost = stats.get("DataScannedInBytes", 0) / (1024**4) * 5.0
    print(f"OK ({runtime_ms}ms, {scanned_mb:.1f} MB scanned, ${cost:.6f})")

    return rows


# ---------------------------------------------------------------------------
# Setup: Upload GENIE to S3 + create Athena table
# ---------------------------------------------------------------------------

def upload_genie_to_s3(session: boto3.Session) -> str:
    """Upload GENIE v19.0 MAF file to S3 (gzip compressed)."""
    s3 = session.client("s3")
    s3_key = "genie_v19/data_mutations_extended.tsv.gz"

    # Check if already uploaded
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        print(f"  GENIE MAF already in s3://{S3_BUCKET}/{s3_key}")
        return f"s3://{S3_BUCKET}/genie_v19/"
    except ClientError:
        pass

    if not GENIE_MAF.exists():
        print(f"  ERROR: GENIE MAF not found at {GENIE_MAF}")
        sys.exit(1)

    print(f"  Uploading GENIE MAF ({GENIE_MAF.stat().st_size / 1024 / 1024:.0f} MB) to S3...")

    # Gzip compress and upload
    buf = BytesIO()
    with gzip.open(buf, "wt", compresslevel=6) as gz:
        with open(GENIE_MAF) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                gz.write(line)
    buf.seek(0)

    s3.upload_fileobj(buf, S3_BUCKET, s3_key)
    size_mb = buf.tell() / 1024 / 1024
    print(f"  Uploaded {size_mb:.0f} MB (compressed) to s3://{S3_BUCKET}/{s3_key}")

    return f"s3://{S3_BUCKET}/genie_v19/"


def create_genie_table(client, s3_location: str) -> None:
    """Create Athena external table for GENIE MAF data."""
    sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DATABASE}.genie_v19_mutations (
        hugo_symbol STRING,
        entrez_gene_id STRING,
        center STRING,
        ncbi_build STRING,
        chromosome STRING,
        start_position BIGINT,
        end_position BIGINT,
        strand STRING,
        consequence STRING,
        variant_classification STRING,
        variant_type STRING,
        reference_allele STRING,
        tumor_seq_allele1 STRING,
        tumor_seq_allele2 STRING,
        dbsnp_rs STRING,
        dbsnp_val_status STRING,
        tumor_sample_barcode STRING,
        matched_norm_sample_barcode STRING,
        match_norm_seq_allele1 STRING,
        match_norm_seq_allele2 STRING,
        tumor_validation_allele1 STRING,
        tumor_validation_allele2 STRING,
        match_norm_validation_allele1 STRING,
        match_norm_validation_allele2 STRING,
        verification_status STRING,
        validation_status STRING,
        mutation_status STRING,
        sequencing_phase STRING,
        sequence_source STRING,
        validation_method STRING,
        score STRING,
        bam_file STRING,
        sequencer STRING,
        t_ref_count STRING,
        t_alt_count STRING,
        n_ref_count STRING,
        n_alt_count STRING,
        hgvsc STRING,
        hgvsp STRING,
        hgvsp_short STRING,
        transcript_id STRING,
        refseq STRING,
        protein_position STRING,
        codons STRING,
        exon_number STRING,
        gnomad_af STRING,
        gnomad_afr_af STRING,
        gnomad_amr_af STRING,
        gnomad_asj_af STRING,
        gnomad_eas_af STRING,
        gnomad_fin_af STRING,
        gnomad_nfe_af STRING,
        gnomad_oth_af STRING,
        gnomad_sas_af STRING,
        filter STRING,
        polyphen_prediction STRING,
        polyphen_score STRING,
        sift_prediction STRING,
        sift_score STRING,
        swissprot STRING,
        n_depth STRING,
        t_depth STRING,
        annotation_status STRING,
        mutationincis_flag STRING
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY '\\t'
    STORED AS TEXTFILE
    LOCATION '{s3_location}'
    TBLPROPERTIES (
        'skip.header.line.count'='1',
        'compressionType'='gzip'
    )
    """
    print("  Creating Athena table genie_v19_mutations...")
    run_athena_query(client, sql)


def run_setup(session: boto3.Session) -> None:
    """Upload GENIE data and create Athena table."""
    print("\n" + "=" * 70)
    print("SETUP: Upload GENIE v19.0 to S3 and create Athena table")
    print("=" * 70)

    s3_location = upload_genie_to_s3(session)
    client = session.client("athena", region_name=AWS_REGION)
    create_genie_table(client, s3_location)

    # Verify with a count query
    print("\n  Verifying table...")
    rows = run_athena_query(client, f"SELECT COUNT(*) FROM {ATHENA_DATABASE}.genie_v19_mutations")
    if rows and len(rows) > 1:
        count = rows[1][0]
        print(f"  Table contains {int(count):,} rows")


# ---------------------------------------------------------------------------
# SQL Queries
# ---------------------------------------------------------------------------

def sql_genie_cohort_summary() -> str:
    """Summary statistics for the GENIE v19.0 cohort."""
    return f"""
-- GENIE v19.0 Cohort Summary
SELECT
    COUNT(*) AS total_mutations,
    COUNT(DISTINCT tumor_sample_barcode) AS total_samples,
    COUNT(DISTINCT hugo_symbol) AS total_genes
FROM {ATHENA_DATABASE}.genie_v19_mutations
WHERE variant_classification IN (
    'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
    'Frame_Shift_Ins', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins',
    'Nonstop_Mutation', 'Translation_Start_Site'
)
"""


def sql_genie_target_gene_frequencies() -> str:
    """Frequency of each target gene in GENIE (protein-altering only)."""
    genes_str = ", ".join(f"'{g}'" for g in ALL_GENES)
    return f"""
-- Target Gene Frequencies in GENIE v19.0 (protein-altering only)
SELECT
    hugo_symbol,
    COUNT(DISTINCT tumor_sample_barcode) AS patients,
    COUNT(*) AS mutations
FROM {ATHENA_DATABASE}.genie_v19_mutations
WHERE hugo_symbol IN ({genes_str})
  AND variant_classification IN (
    'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
    'Frame_Shift_Ins', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins',
    'Nonstop_Mutation', 'Translation_Start_Site'
  )
GROUP BY hugo_symbol
ORDER BY patients DESC
"""


def sql_genie_pairwise_cooccurrence() -> str:
    """Pairwise co-occurrence of the 4 pathogenic genes."""
    genes_str = ", ".join(f"'{g}'" for g in TARGET_GENES)
    return f"""
-- Pairwise Co-occurrence in GENIE v19.0 (protein-altering mutations)
WITH target AS (
    SELECT DISTINCT
        tumor_sample_barcode,
        hugo_symbol
    FROM {ATHENA_DATABASE}.genie_v19_mutations
    WHERE hugo_symbol IN ({genes_str})
      AND variant_classification IN (
        'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
        'Frame_Shift_Ins', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins',
        'Nonstop_Mutation', 'Translation_Start_Site'
      )
)
SELECT
    t1.hugo_symbol AS gene1,
    t2.hugo_symbol AS gene2,
    COUNT(DISTINCT t1.tumor_sample_barcode) AS patients
FROM target t1
JOIN target t2
    ON t1.tumor_sample_barcode = t2.tumor_sample_barcode
WHERE t1.hugo_symbol < t2.hugo_symbol
GROUP BY t1.hugo_symbol, t2.hugo_symbol
ORDER BY patients DESC
"""


def sql_genie_multi_gene_carriers() -> str:
    """Count patients carrying 2, 3, 4 of the target genes."""
    genes_str = ", ".join(f"'{g}'" for g in TARGET_GENES)
    return f"""
-- Multi-Gene Carriers in GENIE v19.0
WITH patient_gene_count AS (
    SELECT
        tumor_sample_barcode,
        COUNT(DISTINCT hugo_symbol) AS gene_count,
        ARRAY_AGG(DISTINCT hugo_symbol) AS genes
    FROM {ATHENA_DATABASE}.genie_v19_mutations
    WHERE hugo_symbol IN ({genes_str})
      AND variant_classification IN (
        'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
        'Frame_Shift_Ins', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins',
        'Nonstop_Mutation', 'Translation_Start_Site'
      )
    GROUP BY tumor_sample_barcode
)
SELECT
    gene_count,
    COUNT(*) AS patients,
    ARRAY_AGG(tumor_sample_barcode) AS sample_ids
FROM patient_gene_count
WHERE gene_count >= 2
GROUP BY gene_count
ORDER BY gene_count DESC
"""


def sql_genie_exact_hotspot_search() -> str:
    """Search for exact hotspot variants in GENIE."""
    return f"""
-- Exact Hotspot Variant Search in GENIE v19.0
SELECT
    hugo_symbol,
    hgvsp_short,
    tumor_sample_barcode,
    variant_classification,
    t_alt_count,
    t_ref_count,
    center
FROM {ATHENA_DATABASE}.genie_v19_mutations
WHERE (hugo_symbol = 'DNMT3A' AND hgvsp_short LIKE '%R882H%')
   OR (hugo_symbol = 'SETBP1' AND hgvsp_short LIKE '%G870S%')
   OR (hugo_symbol = 'PTPN11' AND hgvsp_short LIKE '%E76Q%')
   OR (hugo_symbol = 'IDH2'   AND hgvsp_short LIKE '%R140Q%')
   OR (hugo_symbol = 'EZH2'   AND hgvsp_short LIKE '%V662A%')
ORDER BY hugo_symbol, tumor_sample_barcode
"""


def sql_genie_hotspot_cooccurrence() -> str:
    """Find patients carrying 2+ exact hotspot variants."""
    return f"""
-- Exact Hotspot Co-occurrence in GENIE v19.0
WITH hotspot_carriers AS (
    SELECT DISTINCT
        tumor_sample_barcode,
        hugo_symbol,
        hgvsp_short
    FROM {ATHENA_DATABASE}.genie_v19_mutations
    WHERE (hugo_symbol = 'DNMT3A' AND hgvsp_short LIKE '%R882H%')
       OR (hugo_symbol = 'SETBP1' AND hgvsp_short LIKE '%G870S%')
       OR (hugo_symbol = 'PTPN11' AND hgvsp_short LIKE '%E76Q%')
       OR (hugo_symbol = 'IDH2'   AND hgvsp_short LIKE '%R140Q%')
       OR (hugo_symbol = 'EZH2'   AND hgvsp_short LIKE '%V662A%')
)
SELECT
    tumor_sample_barcode,
    COUNT(DISTINCT hugo_symbol) AS hotspot_count,
    ARRAY_AGG(DISTINCT CONCAT(hugo_symbol, ' ', hgvsp_short)) AS mutations
FROM hotspot_carriers
GROUP BY tumor_sample_barcode
HAVING COUNT(DISTINCT hugo_symbol) >= 2
ORDER BY hotspot_count DESC
"""


def sql_genie_cross_tcga() -> str:
    """Cross-reference GENIE target gene carriers with TCGA pan-cancer data.

    Finds genes mutated in both databases for the same patient-equivalent pairs.
    """
    genes_str = ", ".join(f"'{g}'" for g in TARGET_GENES)
    return f"""
-- Cross-database gene frequencies: GENIE vs TCGA
-- GENIE target gene patients
WITH genie_genes AS (
    SELECT DISTINCT hugo_symbol, 'GENIE' AS source
    FROM {ATHENA_DATABASE}.genie_v19_mutations
    WHERE hugo_symbol IN ({genes_str})
      AND variant_classification IN (
        'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
        'Frame_Shift_Ins', 'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins'
      )
),
-- TCGA LAML target gene patients
tcga_genes AS (
    SELECT DISTINCT hugo_symbol, 'TCGA-LAML' AS source
    FROM {ATHENA_DATABASE}.tcga_laml_mutations
    WHERE hugo_symbol IN ({genes_str})
),
-- Combine
combined AS (
    SELECT * FROM genie_genes
    UNION ALL
    SELECT * FROM tcga_genes
)
SELECT
    hugo_symbol,
    source,
    COUNT(*) AS records
FROM combined
GROUP BY hugo_symbol, source
ORDER BY hugo_symbol, source
"""


# ---------------------------------------------------------------------------
# Main query runner
# ---------------------------------------------------------------------------

def run_queries(session: boto3.Session, dry_run: bool = False) -> dict[str, Any]:
    """Run all co-occurrence queries and collect results."""
    client = session.client("athena", region_name=AWS_REGION)

    queries = {
        "cohort_summary": ("GENIE Cohort Summary", sql_genie_cohort_summary()),
        "gene_frequencies": ("Target Gene Frequencies", sql_genie_target_gene_frequencies()),
        "pairwise": ("Pairwise Co-occurrence", sql_genie_pairwise_cooccurrence()),
        "multi_gene": ("Multi-Gene Carriers (2+/3+/4)", sql_genie_multi_gene_carriers()),
        "hotspot_search": ("Exact Hotspot Search", sql_genie_exact_hotspot_search()),
        "hotspot_cooccurrence": ("Hotspot Co-occurrence (2+)", sql_genie_hotspot_cooccurrence()),
    }

    print("\n" + "=" * 70)
    print("QUERIES: GENIE v19.0 Co-occurrence via Athena SQL")
    print("=" * 70)

    results = {}
    total_cost = 0.0

    for key, (name, sql) in queries.items():
        print(f"\n--- {name} ---")
        if dry_run:
            print(sql)
            continue

        rows = run_athena_query(client, sql)

        if rows:
            header = rows[0]
            data = rows[1:]
            results[key] = {"header": header, "data": data}
            print(f"  Returned {len(data)} rows")

            # Print results in tabular form
            if len(data) <= 30:
                col_widths = [max(len(str(r[i])) for r in [header] + data)
                              for i in range(len(header))]
                fmt = "  " + " | ".join(f"{{:<{w}}}" for w in col_widths)
                print(fmt.format(*header))
                print("  " + "-+-".join("-" * w for w in col_widths))
                for row in data:
                    print(fmt.format(*row))

    return results


def run_cross_ref(session: boto3.Session, dry_run: bool = False) -> dict[str, Any]:
    """Run cross-reference queries between GENIE and TCGA."""
    client = session.client("athena", region_name=AWS_REGION)

    print("\n" + "=" * 70)
    print("CROSS-REFERENCE: GENIE vs TCGA via Athena SQL")
    print("=" * 70)

    sql = sql_genie_cross_tcga()
    if dry_run:
        print(sql)
        return {}

    rows = run_athena_query(client, sql)
    if rows:
        header = rows[0]
        data = rows[1:]
        print(f"  Returned {len(data)} rows")
        for row in data:
            print(f"  {row[0]:10s} {row[1]:12s} {row[2]:>6s}")
        return {"cross_ref": {"header": header, "data": data}}
    return {}


def save_results(results: dict[str, Any]) -> Path:
    """Save results to JSON and markdown."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = RESULTS_DIR / f"athena_cooccurrence_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON saved to {json_path}")

    # Markdown summary
    md_path = RESULTS_DIR / f"athena_cooccurrence_{ts}.md"
    with open(md_path, "w") as f:
        f.write("# GENIE v19.0 Athena Co-occurrence Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Database:** {ATHENA_DATABASE}\n")
        f.write(f"**Region:** {AWS_REGION}\n\n")

        for key, result in results.items():
            if not isinstance(result, dict) or "header" not in result:
                continue
            f.write(f"## {key}\n\n")
            header = result["header"]
            data = result["data"]

            f.write("| " + " | ".join(header) + " |\n")
            f.write("| " + " | ".join("---" for _ in header) + " |\n")
            for row in data[:50]:  # Limit to 50 rows for readability
                f.write("| " + " | ".join(str(v) for v in row) + " |\n")
            if len(data) > 50:
                f.write(f"\n*({len(data)} total rows, showing first 50)*\n")
            f.write("\n")

    print(f"  Markdown saved to {md_path}")
    return json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Athena co-occurrence queries for GENIE v19.0 + TCGA + ClinVar"
    )
    parser.add_argument("--setup", action="store_true",
                        help="Upload GENIE to S3 and create Athena table")
    parser.add_argument("--query", action="store_true",
                        help="Run co-occurrence queries")
    parser.add_argument("--cross-ref", action="store_true",
                        help="Cross-reference GENIE with TCGA/ClinVar")
    parser.add_argument("--all", action="store_true",
                        help="Run setup + query + cross-ref")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print SQL without executing")
    parser.add_argument("--profile", default=AWS_PROFILE,
                        help=f"AWS profile (default: {AWS_PROFILE})")
    parser.add_argument("--region", default=AWS_REGION,
                        help=f"AWS region (default: {AWS_REGION})")
    args = parser.parse_args()

    if not any([args.setup, args.query, args.cross_ref, args.all]):
        parser.print_help()
        sys.exit(0)

    session = get_session(profile=args.profile, region=args.region)

    all_results = {}

    if args.setup or args.all:
        run_setup(session)

    if args.query or args.all:
        results = run_queries(session, dry_run=args.dry_run)
        all_results.update(results)

    if args.cross_ref or args.all:
        results = run_cross_ref(session, dry_run=args.dry_run)
        all_results.update(results)

    if all_results and not args.dry_run:
        save_results(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
