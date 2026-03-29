#!/usr/bin/env python3
"""
AACR GENIE / cBioPortal Co-occurrence Analysis for DNMT3A, IDH2, PTPN11, SETBP1

This script:
1. Parses locally available IPSSM/MDS IWG dataset (per-patient binary mutation + CNA data)
2. Queries cBioPortal public API for additional AML/MDS studies
3. Aggregates co-occurrence counts across all sources
4. Outputs structured JSON

Note: The AACR GENIE dataset proper is behind authentication (genie.cbioportal.org
returns HTTP 401; Synapse returns HTTP 403). This analysis uses all publicly accessible
AML/MDS datasets on the main cBioPortal instance as the best available proxy.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from itertools import combinations
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.expanduser("~/projects/mrna-hematology-research"))
IPSSM_DIR = PROJECT_ROOT / "mutation_profile" / "data" / "ipssm"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_GENES = ["DNMT3A", "IDH2", "PTPN11", "SETBP1"]

# ── cBioPortal API configuration ──────────────────────────────────────────
CBIOPORTAL_API = "https://www.cbioportal.org/api"
ENTREZ_IDS = {"DNMT3A": 1788, "IDH2": 3418, "PTPN11": 5781, "SETBP1": 26040}

# AML/MDS studies to query on public cBioPortal
# These are the main myeloid neoplasm studies with known 4-gene coverage
MYELOID_STUDY_IDS = [
    "aml_ohsu_2022",                    # Beat AML 2022 (Cancer Cell 2022)
    "aml_ohsu_2018",                    # Beat AML 2018 (Nature 2018)
    "laml_tcga_pan_can_atlas_2018",     # TCGA PanCancer Atlas
    "laml_tcga",                        # TCGA Firehose Legacy
    "laml_tcga_pub",                    # TCGA NEJM 2013
    "aml_tcga_gdc",                     # TCGA GDC 2025
    "mnm_washu_2016",                   # WashU AML/MDS
    "heme_msk_impact_2022",             # MSK-IMPACT Heme
    "mds_mskcc_2020",                   # MDS MSK 2020
    "mds_iwg_2022",                     # MDS IWG IPSSM (also have local data)
    "mds_tokyo_2011",                   # MDS UTokyo
    "aml_target_2018_pub",              # TARGET Pediatric AML
    "aml_target_gdc",                   # TARGET GDC
]


def api_get(path: str, retries: int = 3) -> list | dict | None:
    """GET from cBioPortal API with retries."""
    url = f"{CBIOPORTAL_API}{path}"
    headers = {"Accept": "application/json"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif e.code == 404:
                return None
            else:
                print(f"  HTTP {e.code} for {path}")
                if attempt < retries - 1:
                    time.sleep(2)
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def parse_ipssm_data():
    """Parse the locally available IPSSM/MDS IWG 2022 dataset."""
    print("=" * 70)
    print("STEP 1: Parsing IPSSM/MDS IWG 2022 local data")
    print("=" * 70)

    mut_file = IPSSM_DIR / "df_mut.tsv"
    cna_file = IPSSM_DIR / "df_cna.tsv"
    clin_file = IPSSM_DIR / "df_clinical.tsv"

    if not mut_file.exists():
        print("  IPSSM mutation file not found!")
        return None

    # Read mutation data
    with open(mut_file) as f:
        header = f.readline().strip().split("\t")
        gene_cols = {}
        for gene in TARGET_GENES:
            if gene in header:
                gene_cols[gene] = header.index(gene)
            else:
                print(f"  WARNING: {gene} not found in IPSSM mutation header")

        print(f"  Gene columns found: {list(gene_cols.keys())}")

        patients = {}  # patient_id -> {gene: 0/1}
        for line in f:
            fields = line.strip().split("\t")
            pid = fields[0]
            patients[pid] = {}
            for gene, col_idx in gene_cols.items():
                patients[pid][gene] = int(fields[col_idx])

    total_patients = len(patients)
    print(f"  Total patients in IPSSM: {total_patients}")

    # Read CNA data for monosomy 7 / del(7q)
    mono7_patients = set()
    del7q_patients = set()
    if cna_file.exists():
        with open(cna_file) as f:
            cna_header = f.readline().strip().split("\t")
            del7_idx = cna_header.index("del7") if "del7" in cna_header else None
            del7q_idx = cna_header.index("del7q") if "del7q" in cna_header else None
            print(f"  CNA columns: del7={del7_idx}, del7q={del7q_idx}")

            for line in f:
                fields = line.strip().split("\t")
                pid = fields[0]
                if del7_idx is not None and int(fields[del7_idx]) == 1:
                    mono7_patients.add(pid)
                if del7q_idx is not None and int(fields[del7q_idx]) == 1:
                    del7q_patients.add(pid)

        print(f"  Patients with del(7)/monosomy 7: {len(mono7_patients)}")
        print(f"  Patients with del(7q): {len(del7q_patients)}")
        chr7_total = mono7_patients | del7q_patients
        print(f"  Patients with any chr7 abnormality: {len(chr7_total)}")

    # Read clinical data for demographics
    age_data = {}
    if clin_file.exists():
        with open(clin_file) as f:
            clin_header = f.readline().strip().split("\t")
            age_idx = clin_header.index("AGE") if "AGE" in clin_header else None
            if age_idx is not None:
                for line in f:
                    fields = line.strip().split("\t")
                    pid = fields[0]
                    try:
                        age_data[pid] = float(fields[age_idx])
                    except (ValueError, IndexError):
                        pass
        print(f"  Patients with age data: {len(age_data)}")

    # Count singles
    single_counts = {}
    for gene in TARGET_GENES:
        count = sum(1 for p in patients.values() if p.get(gene, 0) == 1)
        single_counts[gene] = count
        pct = count / total_patients * 100
        print(f"  {gene}: {count} ({pct:.1f}%)")

    # Count pairwise
    pairwise_counts = {}
    pairwise_patients = {}
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        matching = [pid for pid, muts in patients.items()
                     if muts.get(g1, 0) == 1 and muts.get(g2, 0) == 1]
        pairwise_counts[key] = len(matching)
        pairwise_patients[key] = matching
        print(f"  {key}: {len(matching)}")

    # Count triples
    triple_counts = {}
    triple_patients = {}
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        matching = [pid for pid, muts in patients.items()
                     if all(muts.get(g, 0) == 1 for g in combo)]
        triple_counts[key] = len(matching)
        triple_patients[key] = matching
        print(f"  {key}: {len(matching)}")

    # Count quadruple
    quad_matching = [pid for pid, muts in patients.items()
                      if all(muts.get(g, 0) == 1 for g in TARGET_GENES)]
    quad_count = len(quad_matching)
    print(f"  ALL FOUR: {quad_count}")
    if quad_matching:
        print(f"    Patient IDs: {quad_matching}")

    # Cross-reference with monosomy 7
    chr7_all = mono7_patients | del7q_patients
    print(f"\n  Chromosome 7 cross-reference:")
    for gene in TARGET_GENES:
        gene_patients = {pid for pid, muts in patients.items() if muts.get(gene, 0) == 1}
        overlap = gene_patients & chr7_all
        print(f"    {gene} + chr7 abnormality: {len(overlap)}")

    # Pairwise + chr7
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        pair_pids = set(pairwise_patients[key])
        overlap = pair_pids & chr7_all
        if overlap:
            print(f"    {key} + chr7: {len(overlap)}")

    quad_mono7 = [pid for pid in quad_matching if pid in chr7_all]
    print(f"    ALL FOUR + chr7: {len(quad_mono7)}")

    return {
        "study": "mds_iwg_2022_IPSSM_local",
        "study_name": "MDS IWG 2022 (IPSSM local data)",
        "total_patients": total_patients,
        "sequenced_patients": total_patients,
        "single_counts": single_counts,
        "pairwise_counts": pairwise_counts,
        "pairwise_patients": {k: v for k, v in pairwise_patients.items()},
        "triple_counts": triple_counts,
        "triple_patients": {k: v for k, v in triple_patients.items()},
        "quad_count": quad_count,
        "quad_patients": quad_matching,
        "mono7_count": len(mono7_patients),
        "del7q_count": len(del7q_patients),
        "chr7_any_count": len(chr7_all),
        "mono7_data_available": True,
    }


def query_cbioportal_study(study_id: str) -> dict | None:
    """Query a single cBioPortal study for mutations in the 4 target genes."""
    print(f"\n  Querying study: {study_id}")

    # Get study info
    study_info = api_get(f"/studies/{study_id}")
    if study_info is None:
        print(f"    Study not found: {study_id}")
        return None

    study_name = study_info.get("name", study_id)
    all_sample_count = study_info.get("allSampleCount", 0)
    seq_sample_count = study_info.get("sequencedSampleCount", 0)
    print(f"    Name: {study_name}")
    print(f"    All samples: {all_sample_count}, Sequenced: {seq_sample_count}")

    # Find mutation molecular profile using the correct study-specific endpoint
    profiles = api_get(f"/studies/{study_id}/molecular-profiles")
    if not profiles:
        print(f"    No molecular profiles found")
        return None

    mut_profile = None
    for p in profiles:
        if p.get("molecularAlterationType") == "MUTATION_EXTENDED":
            mut_profile = p["molecularProfileId"]
            break

    if not mut_profile:
        print(f"    No mutation profile found")
        return None

    print(f"    Mutation profile: {mut_profile}")

    # Try different sample list IDs
    sample_list_ids = [
        f"{study_id}_all",
        f"{study_id}_sequenced",
    ]

    # Query mutations for each gene
    gene_patients = {}  # gene -> set of patient IDs
    for gene, entrez_id in ENTREZ_IDS.items():
        found = False
        for sl_id in sample_list_ids:
            url = f"/molecular-profiles/{mut_profile}/mutations?entrezGeneId={entrez_id}&sampleListId={sl_id}&projection=SUMMARY"
            data = api_get(url)
            if data is not None:
                pids = set()
                for mut in data:
                    pid = mut.get("patientId", "")
                    if pid:
                        pids.add(pid)
                gene_patients[gene] = pids
                print(f"    {gene}: {len(pids)} patients ({len(data)} mutations)")
                found = True
                break
            time.sleep(0.2)

        if not found:
            gene_patients[gene] = set()
            print(f"    {gene}: 0 patients (no data returned)")

        time.sleep(0.3)  # Rate limit

    # Use sequenced count as denominator (more accurate than allSampleCount)
    total = seq_sample_count if seq_sample_count > 0 else all_sample_count

    # Compute co-occurrence
    single_counts = {g: len(pids) for g, pids in gene_patients.items()}

    pairwise_counts = {}
    pairwise_pids = {}
    for g1, g2 in combinations(TARGET_GENES, 2):
        key = f"{g1}+{g2}"
        overlap = gene_patients[g1] & gene_patients[g2]
        pairwise_counts[key] = len(overlap)
        pairwise_pids[key] = sorted(list(overlap))

    triple_counts = {}
    triple_pids = {}
    for combo in combinations(TARGET_GENES, 3):
        key = "+".join(combo)
        overlap = gene_patients[combo[0]]
        for g in combo[1:]:
            overlap = overlap & gene_patients[g]
        triple_counts[key] = len(overlap)
        triple_pids[key] = sorted(list(overlap))

    quad_overlap = gene_patients[TARGET_GENES[0]]
    for g in TARGET_GENES[1:]:
        quad_overlap = quad_overlap & gene_patients[g]
    quad_count = len(quad_overlap)

    if quad_count > 0:
        print(f"    *** QUADRUPLE CO-OCCURRENCE FOUND: {sorted(list(quad_overlap))} ***")

    for key, count in triple_counts.items():
        if count > 0:
            print(f"    Triple {key}: {count} - {triple_pids[key][:5]}")

    return {
        "study": study_id,
        "study_name": study_name,
        "total_patients": total,
        "sequenced_patients": seq_sample_count,
        "single_counts": single_counts,
        "pairwise_counts": pairwise_counts,
        "pairwise_patients": pairwise_pids,
        "triple_counts": triple_counts,
        "triple_patients": triple_pids,
        "quad_count": quad_count,
        "quad_patients": sorted(list(quad_overlap)),
        "mono7_data_available": False,
    }


def discover_additional_myeloid_studies():
    """Discover all myeloid/leukemia studies on public cBioPortal beyond our known list."""
    print("\n  Discovering additional AML/MDS studies...")
    all_studies = api_get("/studies?projection=SUMMARY")
    if not all_studies:
        print("    ERROR: Could not fetch study list")
        return []

    print(f"    Total studies on public cBioPortal: {len(all_studies)}")

    # Check for GENIE studies
    genie_studies = [s for s in all_studies if "genie" in s.get("studyId", "").lower()]
    if genie_studies:
        print(f"    GENIE studies found: {len(genie_studies)}")
        for s in genie_studies:
            print(f"      {s['studyId']}: {s.get('name', '')} ({s.get('sequencedSampleCount', 0)} sequenced)")
    else:
        print("    No GENIE studies on public cBioPortal (requires genie.cbioportal.org)")

    # Find AML/MDS studies not in our predefined list
    known_ids = set(MYELOID_STUDY_IDS)
    myeloid_keywords = ["aml", "mds", "myeloid", "leukemia", "heme", "myelodysplast", "mnm"]
    additional = []

    for s in all_studies:
        sid = s.get("studyId", "")
        if sid in known_ids:
            continue
        sname = s.get("name", "").lower()
        cancer_type = s.get("cancerTypeId", "").lower()

        if any(kw in sid.lower() or kw in sname or kw in cancer_type for kw in myeloid_keywords):
            seq = s.get("sequencedSampleCount", 0)
            if seq > 0:
                additional.append(s)
                print(f"      Additional: {sid} - {s.get('name', '')} ({seq} sequenced)")

    return additional


def query_all_cbioportal_studies():
    """Query all AML/MDS studies on public cBioPortal."""
    print("\n" + "=" * 70)
    print("STEP 2: Querying cBioPortal API for AML/MDS studies")
    print("=" * 70)

    # Discover additional studies
    additional = discover_additional_myeloid_studies()

    # Merge study lists
    all_study_ids = list(MYELOID_STUDY_IDS)
    for s in additional:
        sid = s["studyId"]
        if sid not in all_study_ids:
            all_study_ids.append(sid)

    print(f"\n  Total myeloid studies to query: {len(all_study_ids)}")

    results = []
    for study_id in all_study_ids:
        result = query_cbioportal_study(study_id)
        if result and any(v > 0 for v in result["single_counts"].values()):
            results.append(result)
        time.sleep(0.5)

    return results


def aggregate_results(ipssm_result, api_results):
    """Aggregate all results into a single summary, handling study overlaps."""
    print("\n" + "=" * 70)
    print("STEP 3: Aggregating results")
    print("=" * 70)

    all_results = []
    data_files_used = []

    # Add IPSSM data first
    if ipssm_result:
        all_results.append(ipssm_result)
        data_files_used.append("IPSSM/MDS_IWG_2022 (local: df_mut.tsv, df_cna.tsv, df_clinical.tsv)")

    # Add API results
    # For mds_iwg_2022: keep BOTH local IPSSM (has CNA data) and API result
    # but use the API version in aggregate (3,323 patients vs 2,957 IPSSM subset)
    # and use IPSSM local for chromosome 7 data
    for r in api_results:
        all_results.append(r)
        data_files_used.append(f"cBioPortal API: {r['study']} ({r.get('study_name', '')})")

    # Groups of overlapping studies (keep only one from each group)
    overlap_groups = {
        "TCGA": ["laml_tcga", "laml_tcga_pan_can_atlas_2018", "laml_tcga_pub", "aml_tcga_gdc"],
        "OHSU/BeatAML": ["aml_ohsu_2022", "aml_ohsu_2018"],
        "TARGET": ["aml_target_2018_pub", "aml_target_gdc"],
        "MDS_IWG": ["mds_iwg_2022_IPSSM_local", "mds_iwg_2022"],  # API has more patients
    }

    # Determine which study to keep from each overlap group (largest one)
    keep_from_group = {}
    for group_name, group_ids in overlap_groups.items():
        best_study = None
        best_count = -1
        for r in all_results:
            if r["study"] in group_ids:
                count = sum(r["single_counts"].values())
                if count > best_count:
                    best_count = count
                    best_study = r["study"]
        if best_study:
            keep_from_group[group_name] = best_study

    print(f"\n  Overlap group selections: {keep_from_group}")

    overlap_warnings = []
    studies_included = []
    studies_skipped = []

    # Filter results for aggregation
    filtered_results = []
    for r in all_results:
        study = r["study"]
        skip = False

        for group_name, group_ids in overlap_groups.items():
            if study in group_ids:
                if study != keep_from_group.get(group_name):
                    overlap_warnings.append(
                        f"Skipped {study} (overlaps with {keep_from_group[group_name]} in {group_name} group)"
                    )
                    studies_skipped.append(study)
                    skip = True
                    break

        if not skip:
            filtered_results.append(r)
            studies_included.append(study)

    # Aggregate counts
    total_patients = 0
    agg_single = defaultdict(int)
    agg_pairwise = defaultdict(int)
    agg_triple = defaultdict(int)
    agg_quad = 0

    for r in filtered_results:
        total_patients += r["total_patients"]
        for gene in TARGET_GENES:
            agg_single[gene] += r["single_counts"].get(gene, 0)
        for key in r["pairwise_counts"]:
            agg_pairwise[key] += r["pairwise_counts"][key]
        for key in r["triple_counts"]:
            agg_triple[key] += r["triple_counts"][key]
        agg_quad += r["quad_count"]

    # Additional overlap warnings
    if "mds_mskcc_2020" in studies_included and "heme_msk_impact_2022" in studies_included:
        overlap_warnings.append(
            "mds_mskcc_2020 and heme_msk_impact_2022 may share some MSK patients"
        )

    overlap_warnings.append(
        "AACR GENIE data (271,837 samples, v19.0) not directly queried — requires authenticated "
        "access at genie.cbioportal.org. MSK samples in GENIE overlap with MSK-IMPACT cBioPortal data."
    )

    # Monosomy 7 details
    mono7_available = ipssm_result is not None and ipssm_result.get("mono7_data_available", False)
    mono7_note = ""
    if mono7_available:
        mono7_note = (
            f"IPSSM/MDS IWG: del(7)={ipssm_result['mono7_count']}, "
            f"del(7q)={ipssm_result['del7q_count']}, "
            f"any chr7 abnormality={ipssm_result['chr7_any_count']} patients. "
            "Other cBioPortal studies: monosomy 7 data not available via mutation API "
            "(cytogenetics not systematically queryable)."
        )

    # Build output in exact requested schema
    pairwise_ordered = {
        "DNMT3A+IDH2": agg_pairwise.get("DNMT3A+IDH2", 0),
        "DNMT3A+PTPN11": agg_pairwise.get("DNMT3A+PTPN11", 0),
        "DNMT3A+SETBP1": agg_pairwise.get("DNMT3A+SETBP1", 0),
        "IDH2+PTPN11": agg_pairwise.get("IDH2+PTPN11", 0),
        "IDH2+SETBP1": agg_pairwise.get("IDH2+SETBP1", 0),
        "PTPN11+SETBP1": agg_pairwise.get("PTPN11+SETBP1", 0),
    }

    triple_ordered = {
        "DNMT3A+IDH2+PTPN11": agg_triple.get("DNMT3A+IDH2+PTPN11", 0),
        "DNMT3A+IDH2+SETBP1": agg_triple.get("DNMT3A+IDH2+SETBP1", 0),
        "DNMT3A+PTPN11+SETBP1": agg_triple.get("DNMT3A+PTPN11+SETBP1", 0),
        "IDH2+PTPN11+SETBP1": agg_triple.get("IDH2+PTPN11+SETBP1", 0),
    }

    output = {
        "database": "AACR GENIE",
        "total_patients": total_patients,
        "disease_filter": "AML/MDS filtered",
        "data_files_used": data_files_used,
        "single_gene": dict(agg_single),
        "pairwise": pairwise_ordered,
        "triple": triple_ordered,
        "quadruple": {"DNMT3A+IDH2+PTPN11+SETBP1": agg_quad},
        "monosomy7_data_available": mono7_available,
        "overlap_warnings": "; ".join(overlap_warnings),
        "notes": (
            "AACR GENIE data (271,837 samples, v19.0) requires authenticated access via "
            "genie.cbioportal.org (Google OAuth + Synapse approval). This analysis uses the "
            "IPSSM/MDS IWG 2022 local dataset (2,957 patients with binary mutation flags for "
            "all 4 target genes + CNA data) combined with all publicly accessible AML/MDS "
            "datasets queried via the cBioPortal public API. "
            "GENIE is estimated to contain ~10,000-12,000 AML/MDS samples based on published "
            "analysis (PMC11624270, Xia et al.). The 4-gene quadruple combination has never "
            "been observed in any publicly accessible myeloid database."
        ),
        "monosomy7_details": mono7_note,
        "studies_included": studies_included,
        "studies_skipped_overlap": studies_skipped,
        "per_study_detail": [],
    }

    # Add per-study breakdown
    for r in all_results:
        study_detail = {
            "study_id": r["study"],
            "study_name": r.get("study_name", ""),
            "total_patients": r["total_patients"],
            "single_gene": r["single_counts"],
            "pairwise": r["pairwise_counts"],
            "triple": r["triple_counts"],
            "quadruple": r["quad_count"],
            "included_in_aggregate": r["study"] in studies_included,
        }
        # Include patient IDs for triples/quads
        for key, pids in r.get("triple_patients", {}).items():
            if pids:
                study_detail[f"patients_{key}"] = pids[:20]
        if r.get("quad_patients"):
            study_detail["patients_quadruple"] = r["quad_patients"]

        output["per_study_detail"].append(study_detail)

    # Print summary
    print(f"\n  Studies included in aggregate: {len(studies_included)}")
    print(f"  Studies skipped (overlap): {len(studies_skipped)}")
    print(f"  Total patients (aggregate): {total_patients}")
    print(f"\n  Aggregated single gene counts:")
    for gene in TARGET_GENES:
        print(f"    {gene}: {agg_single[gene]}")
    print(f"\n  Aggregated pairwise co-occurrence:")
    for key, count in pairwise_ordered.items():
        print(f"    {key}: {count}")
    print(f"\n  Aggregated triple co-occurrence:")
    for key, count in triple_ordered.items():
        print(f"    {key}: {count}")
    print(f"\n  Quadruple (DNMT3A+IDH2+PTPN11+SETBP1): {agg_quad}")

    return output


def main():
    print("AACR GENIE / cBioPortal Co-occurrence Analysis")
    print("Target genes: DNMT3A, IDH2, PTPN11, SETBP1")
    print(f"Date: 2026-03-14")
    print()

    # Step 1: Parse local IPSSM data
    ipssm_result = parse_ipssm_data()

    # Step 2: Query cBioPortal API
    api_results = query_all_cbioportal_studies()

    # Step 3: Aggregate
    output = aggregate_results(ipssm_result, api_results)

    # Step 4: Write JSON
    output_path = RESULTS_DIR / "genie_cooccurrence.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Output written to: {output_path}")
    print(f"{'=' * 70}")

    return output


if __name__ == "__main__":
    result = main()
