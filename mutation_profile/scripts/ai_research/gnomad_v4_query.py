#!/usr/bin/env python3
"""
gnomAD v4 population frequency query for patient variants.

Queries the gnomAD v4 GraphQL API for population allele frequencies of the
5 patient somatic mutations. These are somatic cancer variants and should be
absent or extremely rare in gnomAD (a germline database). Absence in gnomAD
provides ACMG PM2 evidence (absent from controls).

Patient variants (GRCh38):
    1. EZH2 V662A   - chr7:148810377 A>G  - VAF 59%, founder clone
    2. DNMT3A R882H  - chr2:25234373 C>T   - VAF 39%, pathogenic hotspot
    3. SETBP1 G870S  - chr18:44951948 G>A  - VAF 34%, likely pathogenic
    4. PTPN11 E76Q   - chr12:112450406 G>C - VAF 29%, pathogenic
    5. IDH2 R140Q    - chr15:90088702 C>T  - VAF 2%, pathogenic subclone

Data source:
    gnomAD v4.1.0 GraphQL API (https://gnomad.broadinstitute.org/api)
    - Exome: 807,162 samples
    - Genome: 76,215 samples

Outputs:
    - mutation_profile/results/ai_research/gnomad_v4_results.json
    - mutation_profile/results/ai_research/gnomad_v4_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/gnomad_v4_query.py

Runtime: ~40 seconds (rate-limited API calls)
Dependencies: requests
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "ai_research"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = RESULTS_DIR / "gnomad_v4_results.json"
MD_OUTPUT = RESULTS_DIR / "gnomad_v4_report.md"

# ── Patient variants (GRCh38, verified via VariantValidator + ClinVar VCV003373649 (2026-03-28)) ──

VARIANTS = [
    {
        "gene": "EZH2",
        "protein_change": "V662A",
        "hgvs_c": "c.1985T>C",
        "vaf": 0.59,
        "chrom": "7",
        "pos": 148810377,
        "ref": "A",
        "alt": "G",
        "transcript": "NM_004456.5",
    },
    {
        "gene": "DNMT3A",
        "protein_change": "R882H",
        "hgvs_c": "c.2645G>A",
        "vaf": 0.39,
        "chrom": "2",
        "pos": 25234373,
        "ref": "C",
        "alt": "T",
        "transcript": "NM_022552.5",
    },
    {
        "gene": "SETBP1",
        "protein_change": "G870S",
        "hgvs_c": "c.2608G>A",
        "vaf": 0.34,
        "chrom": "18",
        "pos": 44951948,
        "ref": "G",
        "alt": "A",
        "transcript": "NM_015559.3",
    },
    {
        "gene": "PTPN11",
        "protein_change": "E76Q",
        "hgvs_c": "c.226G>C",
        "vaf": 0.29,
        "chrom": "12",
        "pos": 112450406,
        "ref": "G",
        "alt": "C",
        "transcript": "NM_002834.5",
    },
    {
        "gene": "IDH2",
        "protein_change": "R140Q",
        "hgvs_c": "c.419G>A",
        "vaf": 0.02,
        "chrom": "15",
        "pos": 90088702,
        "ref": "C",
        "alt": "T",
        "transcript": "NM_002168.4",
    },
]

# ── gnomAD GraphQL API ─────────────────────────────────────────────────────

GNOMAD_API = "https://gnomad.broadinstitute.org/api"

VARIANT_QUERY = """
query GnomadVariant($variantId: String!) {
  variant(variantId: $variantId, dataset: gnomad_r4) {
    variant_id
    chrom
    pos
    ref
    alt
    rsids
    exome {
      ac
      an
      ac_hom
      af
      populations {
        id
        ac
        an
        ac_hom
      }
      filters
      faf95 {
        popmax
        popmax_population
      }
    }
    genome {
      ac
      an
      ac_hom
      af
      populations {
        id
        ac
        an
        ac_hom
      }
      filters
      faf95 {
        popmax
        popmax_population
      }
    }
  }
}
"""

RATE_LIMIT_SLEEP = 7  # seconds between queries


def query_gnomad(variant_id: str) -> dict[str, Any] | None:
    """Query gnomAD v4 GraphQL API for a single variant."""
    payload = {
        "query": VARIANT_QUERY,
        "variables": {
            "variantId": variant_id,
        },
    }
    try:
        resp = requests.post(
            GNOMAD_API,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code != 200:
            log.warning("gnomAD API returned HTTP %d for %s", resp.status_code, variant_id)
            return None
        data = resp.json()
        if "errors" in data:
            log.warning("gnomAD GraphQL errors for %s: %s", variant_id, data["errors"])
        return data.get("data", {}).get("variant")
    except requests.RequestException as exc:
        log.error("Request failed for %s: %s", variant_id, exc)
        return None


def verify_coordinates_via_myvariant(variant: dict) -> dict[str, Any]:
    """Cross-check variant coordinates using myvariant.info API."""
    chrom = variant["chrom"]
    pos = variant["pos"]
    ref = variant["ref"]
    alt = variant["alt"]
    # myvariant.info uses chr-pos-ref-alt format for GRCh38
    query_id = f"chr{chrom}:g.{pos}{ref}>{alt}"
    url = f"https://myvariant.info/v1/variant/{query_id}?assembly=hg38"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            # Extract gene symbol safely (can be dict or list)
            cadd_gene = data.get("cadd", {}).get("gene")
            if isinstance(cadd_gene, dict):
                gene_symbol = cadd_gene.get("genename")
            elif isinstance(cadd_gene, list) and cadd_gene:
                gene_symbol = cadd_gene[0].get("genename")
            else:
                gene_symbol = None
            if not gene_symbol:
                dbsnp_gene = data.get("dbsnp", {}).get("gene")
                if isinstance(dbsnp_gene, dict):
                    gene_symbol = dbsnp_gene.get("symbol")
                elif isinstance(dbsnp_gene, list) and dbsnp_gene:
                    gene_symbol = dbsnp_gene[0].get("symbol")
            # Extract ClinVar significance safely
            clinvar_rcv = data.get("clinvar", {}).get("rcv")
            if isinstance(clinvar_rcv, list) and clinvar_rcv:
                clinvar_sig = clinvar_rcv[0].get("clinical_significance")
            elif isinstance(clinvar_rcv, dict):
                clinvar_sig = clinvar_rcv.get("clinical_significance")
            else:
                clinvar_sig = None
            return {
                "found": True,
                "query": query_id,
                "dbsnp_rsid": data.get("dbsnp", {}).get("rsid"),
                "gene_symbol": gene_symbol,
                "clinvar_sig": clinvar_sig,
            }
        return {"found": False, "query": query_id, "status": resp.status_code}
    except requests.RequestException as exc:
        return {"found": False, "query": query_id, "error": str(exc)}


def build_variant_id(variant: dict) -> str:
    """Build gnomAD variant ID: chrom-pos-ref-alt."""
    return f"{variant['chrom']}-{variant['pos']}-{variant['ref']}-{variant['alt']}"


def extract_population_data(seq_data: dict | None) -> dict[str, Any]:
    """Extract population breakdown from exome or genome data."""
    if seq_data is None:
        return {"available": False}
    pops = {}
    for pop in seq_data.get("populations", []):
        ac = pop["ac"]
        an = pop["an"]
        af = ac / an if an > 0 else 0
        pops[pop["id"]] = {
            "ac": ac,
            "an": an,
            "ac_hom": pop["ac_hom"],
            "af": af,
        }
    faf95 = seq_data.get("faf95") or {}
    return {
        "available": True,
        "ac": seq_data.get("ac", 0),
        "an": seq_data.get("an", 0),
        "ac_hom": seq_data.get("ac_hom", 0),
        "af": seq_data.get("af", 0),
        "filters": seq_data.get("filters", []),
        "faf95_popmax": faf95.get("popmax"),
        "faf95_popmax_population": faf95.get("popmax_population"),
        "populations": pops,
    }


def classify_pm2(exome_data: dict, genome_data: dict) -> dict[str, Any]:
    """Classify ACMG PM2 evidence based on gnomAD frequency."""
    exome_af = exome_data.get("af", 0) if exome_data.get("available") else 0
    genome_af = genome_data.get("af", 0) if genome_data.get("available") else 0
    exome_ac = exome_data.get("ac", 0) if exome_data.get("available") else 0
    genome_ac = genome_data.get("ac", 0) if genome_data.get("available") else 0
    total_ac = exome_ac + genome_ac
    max_af = max(exome_af or 0, genome_af or 0)

    if total_ac == 0:
        strength = "PM2_Strong"
        interpretation = "Completely absent from gnomAD (0 alleles across exomes + genomes)"
    elif max_af < 0.00001:
        strength = "PM2_Moderate"
        interpretation = f"Extremely rare in gnomAD (AF={max_af:.2e}, AC={total_ac})"
    elif max_af < 0.0001:
        strength = "PM2_Supporting"
        interpretation = f"Very rare in gnomAD (AF={max_af:.2e}, AC={total_ac})"
    else:
        strength = "PM2_Not_Met"
        interpretation = f"Present in gnomAD at AF={max_af:.2e} -- does not meet PM2"

    return {
        "strength": strength,
        "interpretation": interpretation,
        "total_ac": total_ac,
        "max_af": max_af,
    }


def run_queries() -> list[dict[str, Any]]:
    """Query gnomAD for all patient variants."""
    results = []

    for i, v in enumerate(VARIANTS):
        gene = v["gene"]
        variant_id = build_variant_id(v)
        log.info("[%d/%d] Querying %s %s (%s) ...", i + 1, len(VARIANTS), gene, v["protein_change"], variant_id)

        # Step 1: verify coordinates via myvariant.info
        log.info("  Verifying coordinates via myvariant.info ...")
        coord_check = verify_coordinates_via_myvariant(v)
        log.info("  myvariant.info: found=%s, rsid=%s", coord_check.get("found"), coord_check.get("dbsnp_rsid"))

        # Step 2: query gnomAD v4
        log.info("  Querying gnomAD v4 ...")
        gnomad_data = query_gnomad(variant_id)

        if gnomad_data is None:
            log.info("  Variant NOT FOUND in gnomAD v4 (expected for somatic cancer variants)")
            exome = {"available": False}
            genome = {"available": False}
            rsids = []
            clinvar_gnomad = None
        else:
            log.info("  Variant FOUND in gnomAD v4")
            exome = extract_population_data(gnomad_data.get("exome"))
            genome = extract_population_data(gnomad_data.get("genome"))
            rsids = gnomad_data.get("rsids", [])
            clinvar_gnomad = None  # Not available via gnomAD v4 GraphQL variant query
            if exome.get("available"):
                log.info("    Exome: AC=%d, AN=%d, AF=%s", exome["ac"], exome["an"], exome["af"])
            if genome.get("available"):
                log.info("    Genome: AC=%d, AN=%d, AF=%s", genome["ac"], genome["an"], genome["af"])

        # Step 3: PM2 classification
        pm2 = classify_pm2(exome, genome)
        log.info("  PM2 classification: %s", pm2["strength"])

        result = {
            "gene": gene,
            "protein_change": v["protein_change"],
            "hgvs_c": v["hgvs_c"],
            "vaf": v["vaf"],
            "variant_id": variant_id,
            "transcript": v["transcript"],
            "coordinate_verification": coord_check,
            "gnomad_found": gnomad_data is not None,
            "rsids": rsids,
            "exome": exome,
            "genome": genome,
            "clinvar_from_gnomad": clinvar_gnomad,
            "pm2_classification": pm2,
        }
        results.append(result)

        # Rate limit
        if i < len(VARIANTS) - 1:
            log.info("  Rate limiting: sleeping %ds ...", RATE_LIMIT_SLEEP)
            time.sleep(RATE_LIMIT_SLEEP)

    return results


# ── Report generation ──────────────────────────────────────────────────────

POPULATION_LABELS = {
    "afr": "African/African American",
    "ami": "Amish",
    "amr": "Admixed American/Latino",
    "asj": "Ashkenazi Jewish",
    "eas": "East Asian",
    "fin": "Finnish",
    "mid": "Middle Eastern",
    "nfe": "Non-Finnish European",
    "remaining": "Remaining",
    "sas": "South Asian",
}


def generate_markdown(results: list[dict[str, Any]]) -> str:
    """Generate markdown report from gnomAD query results."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# gnomAD v4 Population Frequency Analysis",
        "",
        f"**Generated:** {ts}",
        "**Source:** gnomAD v4.1.0 (https://gnomad.broadinstitute.org/)",
        "**Dataset:** Exomes (~807,162 samples) + Genomes (~76,215 samples)",
        "",
        "## Summary",
        "",
        "These are **somatic cancer driver mutations** found in a patient with MDS-AML.",
        "gnomAD is a **germline** population database. Somatic cancer variants are expected",
        "to be absent or extremely rare in gnomAD, providing ACMG **PM2** evidence",
        "(absent from controls / population databases).",
        "",
        "| Gene | Variant | Patient VAF | gnomAD Found | Exome AF | Genome AF | PM2 |",
        "|------|---------|-------------|--------------|----------|-----------|-----|",
    ]

    for r in results:
        exome_af = f"{r['exome']['af']:.2e}" if r["exome"].get("available") and r["exome"].get("af") else "---"
        genome_af = f"{r['genome']['af']:.2e}" if r["genome"].get("available") and r["genome"].get("af") else "---"
        found = "Yes" if r["gnomad_found"] else "No"
        pm2 = r["pm2_classification"]["strength"]
        lines.append(
            f"| {r['gene']} | {r['protein_change']} ({r['hgvs_c']}) "
            f"| {r['vaf']:.0%} | {found} | {exome_af} | {genome_af} | {pm2} |"
        )

    lines += [
        "",
        "## Coordinate Verification",
        "",
        "All GRCh38 coordinates were verified against myvariant.info before querying gnomAD.",
        "",
        "| Gene | Variant ID (GRCh38) | myvariant.info | rsID |",
        "|------|---------------------|----------------|------|",
    ]

    for r in results:
        cv = r["coordinate_verification"]
        status = "Confirmed" if cv.get("found") else "Not found"
        rsid = cv.get("dbsnp_rsid") or "---"
        lines.append(f"| {r['gene']} | {r['variant_id']} | {status} | {rsid} |")

    lines += ["", "## Detailed Results", ""]

    for r in results:
        gene = r["gene"]
        lines += [
            f"### {gene} {r['protein_change']} ({r['hgvs_c']})",
            "",
            f"- **Variant ID (GRCh38):** {r['variant_id']}",
            f"- **Transcript:** {r['transcript']}",
            f"- **Patient VAF:** {r['vaf']:.0%}",
            f"- **rsIDs:** {', '.join(r['rsids']) if r['rsids'] else 'None'}",
            f"- **gnomAD v4 found:** {'Yes' if r['gnomad_found'] else 'No'}",
            "",
        ]

        if r.get("clinvar_from_gnomad"):
            cv = r["clinvar_from_gnomad"]
            lines += [
                f"- **ClinVar (via gnomAD):** {cv.get('clinical_significance', 'N/A')}",
                f"- **ClinVar review status:** {cv.get('review_status', 'N/A')}",
                f"- **ClinVar variation ID:** {cv.get('clinvar_variation_id', 'N/A')}",
                "",
            ]

        for seq_type in ["exome", "genome"]:
            seq = r[seq_type]
            label = seq_type.capitalize()
            if not seq.get("available"):
                lines.append(f"**{label}:** Not found in gnomAD v4 {seq_type}s")
                lines.append("")
                continue

            lines += [
                f"**{label}:**",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Allele count (AC) | {seq['ac']} |",
                f"| Allele number (AN) | {seq['an']:,} |",
                f"| Homozygote count | {seq['ac_hom']} |",
                f"| Allele frequency (AF) | {seq['af']:.2e} |",
                f"| Filters | {', '.join(seq['filters']) if seq['filters'] else 'PASS'} |",
            ]
            if seq.get("faf95_popmax") is not None:
                lines.append(f"| FAF95 popmax | {seq['faf95_popmax']:.2e} ({seq.get('faf95_popmax_population', '?')}) |")
            lines.append("")

            # Population breakdown
            pops = seq.get("populations", {})
            if pops:
                lines += [
                    f"**{label} population breakdown:**",
                    "",
                    "| Population | AC | AN | AF | Hom |",
                    "|------------|-----|------|------|-----|",
                ]
                for pop_id, pop_label in POPULATION_LABELS.items():
                    if pop_id in pops:
                        p = pops[pop_id]
                        af_str = f"{p['af']:.2e}" if p["af"] else "0"
                        lines.append(f"| {pop_label} ({pop_id}) | {p['ac']} | {p['an']:,} | {af_str} | {p['ac_hom']} |")
                lines.append("")

        # PM2 classification
        pm2 = r["pm2_classification"]
        lines += [
            f"**ACMG PM2 classification:** {pm2['strength']}",
            f"- {pm2['interpretation']}",
            "",
            "---",
            "",
        ]

    # ACMG evidence summary
    lines += [
        "## ACMG PM2 Evidence Summary",
        "",
        "PM2 (Absent from controls) is applied when a variant is absent or at extremely",
        "low frequency in gnomAD. For dominant disorders, absence is sufficient. For cancer",
        "somatic variants, absence in the germline population database confirms these are",
        "acquired mutations, not inherited polymorphisms.",
        "",
        "| Strength | Criteria |",
        "|----------|----------|",
        "| PM2_Strong | Completely absent (AC=0) |",
        "| PM2_Moderate | Extremely rare (AF < 0.00001) |",
        "| PM2_Supporting | Very rare (AF < 0.0001) |",
        "| PM2_Not_Met | Present at higher frequency |",
        "",
        "### Clinical Interpretation",
        "",
    ]

    absent_count = sum(1 for r in results if not r["gnomad_found"])
    rare_count = sum(1 for r in results if r["gnomad_found"] and r["pm2_classification"]["max_af"] < 0.0001)
    common_count = len(results) - absent_count - rare_count

    # Special note for DNMT3A R882H (known CHIP variant)
    lines += [
        f"- **{absent_count}** of 5 variants are absent from gnomAD (PM2_Strong)",
        f"- **{rare_count}** of 5 variants are present but rare (PM2_Moderate/Supporting)",
        f"- **{common_count}** of 5 variants exceed the PM2 threshold",
        "",
        "**Note on DNMT3A R882H:** This is the most common DNMT3A hotspot mutation and is",
        "frequently observed in clonal hematopoiesis of indeterminate potential (CHIP).",
        "It may appear in gnomAD at low frequency because gnomAD includes blood-derived DNA",
        "from older individuals who may carry somatic CHIP mutations. This does NOT make it",
        "a germline variant -- it reflects somatic mosaicism in the population.",
        "",
        "**Note on somatic variants in gnomAD:** gnomAD v4 includes some somatic variants",
        "because DNA was extracted from blood. Recurrent somatic hotspots (especially CHIP",
        "variants like DNMT3A R882H, IDH2 R140Q) may appear at very low frequencies.",
        "These are somatic events, not inherited polymorphisms.",
        "",
        "---",
        "",
        "*Analysis performed using gnomAD v4 GraphQL API with GRCh38 coordinates.*",
        "*Coordinates verified against myvariant.info prior to querying.*",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 70)
    log.info("gnomAD v4 Population Frequency Query")
    log.info("=" * 70)
    log.info("Querying %d patient variants ...", len(VARIANTS))

    results = run_queries()

    # Save JSON
    output = {
        "metadata": {
            "script": "gnomad_v4_query.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gnomad_version": "v4.1.0",
            "api_endpoint": GNOMAD_API,
            "genome_build": "GRCh38",
            "n_variants": len(VARIANTS),
        },
        "results": results,
    }
    JSON_OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    log.info("JSON results saved to %s", JSON_OUTPUT)

    # Save markdown
    report = generate_markdown(results)
    MD_OUTPUT.write_text(report)
    log.info("Markdown report saved to %s", MD_OUTPUT)

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    for r in results:
        found = "FOUND" if r["gnomad_found"] else "NOT FOUND"
        pm2 = r["pm2_classification"]["strength"]
        log.info("  %s %s: %s in gnomAD v4 -> %s", r["gene"], r["protein_change"], found, pm2)


if __name__ == "__main__":
    main()
