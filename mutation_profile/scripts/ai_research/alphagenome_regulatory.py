#!/usr/bin/env python3
"""
AlphaGenome regulatory impact prediction for patient variants.

Predicts regulatory effects of the 5 patient somatic mutations using Google
DeepMind's AlphaGenome model, which predicts thousands of molecular properties
at single base-pair resolution from DNA sequence: gene expression, chromatin
accessibility, histone modifications, TF binding, splice site usage.

Strategy (cascading fallback):
    1. Try the alphagenome Python package (pip install alphagenome)
    2. Try the AlphaGenome REST API endpoint (if publicly available)
    3. Fall back to:
       a) Generate properly formatted AlphaGenome input files for manual submission
       b) Query Ensembl VEP REST API with regulatory=1 and hematopoietic cell types
          to retrieve regulatory region overlaps, TF binding site overlaps, and
          chromatin state annotations for each variant

Patient variants (GRCh38):
    1. DNMT3A R882H  - chr2:25234373 G>A  - VAF 39%, pathogenic hotspot
    2. IDH2 R140Q    - chr15:90088702 C>T - VAF 2%, pathogenic subclone
    3. SETBP1 G870S  - chr18:44951948 G>A - VAF 34%, likely pathogenic
    4. PTPN11 E76Q   - chr12:112450406 G>C - VAF 29%, pathogenic
    5. EZH2 V662A    - chr7:148810377 A>G - VAF 59%, founder clone

Data sources:
    - AlphaGenome (DeepMind, announced Google I/O May 2025)
    - Ensembl VEP REST API (GRCh38, regulatory annotations with cell type context)
    - Ensembl Regulatory Build (ENCODE + Roadmap Epigenomics)

Outputs:
    - mutation_profile/results/ai_research/alphagenome_regulatory/regulatory_predictions.json
    - mutation_profile/results/ai_research/alphagenome_regulatory/regulatory_impact_report.md
    - mutation_profile/results/ai_research/alphagenome_regulatory/alphagenome_inputs/ (if fallback)

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/alphagenome_regulatory.py

Runtime: ~30 seconds (VEP API calls with rate limiting)
Dependencies: requests
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# -- Paths -------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "ai_research" / "alphagenome_regulatory"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUTS_DIR = RESULTS_DIR / "alphagenome_inputs"
JSON_OUTPUT = RESULTS_DIR / "regulatory_predictions.json"
MD_OUTPUT = RESULTS_DIR / "regulatory_impact_report.md"

# -- Constants ----------------------------------------------------------------

ENSEMBL_VEP_HGVS = "https://rest.ensembl.org/vep/human/hgvs"
ENSEMBL_VEP_REGION = "https://rest.ensembl.org/vep/human/region"
ENSEMBL_REGULATORY = "https://rest.ensembl.org/overlap/region/human"

# Hematopoietic cell types available in the Ensembl Regulatory Build
# (from ENCODE and Roadmap Epigenomics projects)
HEMATOPOIETIC_CELL_TYPES = [
    "CD14+CD16-_monocyte",
    "CD34+_mobilized",
    "HL-60",
    "K562",
    "CD4+_ab_T_cell",
    "CD8+_ab_T_cell",
    "natural_killer_cell",
    "neutrophil",
    "B_cell_(CD19+)",
]

# Rate limiting for Ensembl REST API (max 15 requests/second for POST)
RATE_LIMIT_DELAY = 0.5  # seconds between requests
REQUEST_TIMEOUT = 30

# AlphaGenome sequence context: the model uses a large genomic window
ALPHAGENOME_CONTEXT_BP = 196_608  # ~196 kb context window (typical for Enformer-class models)


# -- Patient variants --------------------------------------------------------

@dataclass
class PatientVariant:
    """Patient somatic variant with genomic coordinates and regulatory annotations."""

    gene: str
    protein_change: str
    hgvs_c: str
    hgvs_genomic: str  # HGVS genomic notation for VEP queries
    vaf: float
    transcript: str
    chrom: str
    pos_grch38: int
    ref: str
    alt: str
    # Populated by queries
    method_used: str = "pending"
    regulatory_regions: list[dict] = field(default_factory=list)
    motif_features: list[dict] = field(default_factory=list)
    transcript_consequences: list[dict] = field(default_factory=list)
    regulatory_consequences: list[dict] = field(default_factory=list)
    nearby_regulatory_elements: list[dict] = field(default_factory=list)
    chromatin_state: str | None = None
    hematopoietic_relevance: str | None = None
    regulatory_summary: str | None = None
    raw_vep_response: dict | None = None
    alphagenome_result: dict | None = None
    error: str | None = None


PATIENT_VARIANTS = [
    PatientVariant(
        gene="DNMT3A",
        protein_change="R882H",
        hgvs_c="c.2645G>A",
        hgvs_genomic="2:g.25234373G>A",
        vaf=0.39,
        transcript="NM_022552.5",
        chrom="2",
        pos_grch38=25_234_373,
        ref="G",
        alt="A",
    ),
    PatientVariant(
        gene="IDH2",
        protein_change="R140Q",
        hgvs_c="c.419G>A",
        hgvs_genomic="15:g.90088702C>T",
        vaf=0.02,
        transcript="NM_002168.4",
        chrom="15",
        pos_grch38=90_088_702,
        ref="C",
        alt="T",
    ),
    PatientVariant(
        gene="SETBP1",
        protein_change="G870S",
        hgvs_c="c.2608G>A",
        hgvs_genomic="18:g.44951948G>A",
        vaf=0.34,
        transcript="NM_015559.3",
        chrom="18",
        pos_grch38=44_951_948,
        ref="G",
        alt="A",
    ),
    PatientVariant(
        gene="PTPN11",
        protein_change="E76Q",
        hgvs_c="c.226G>C",
        hgvs_genomic="12:g.112450406G>C",
        vaf=0.29,
        transcript="NM_002834.5",
        chrom="12",
        pos_grch38=112_450_406,
        ref="G",
        alt="C",
    ),
    PatientVariant(
        gene="EZH2",
        protein_change="V662A",
        hgvs_c="c.1985T>C",
        hgvs_genomic="7:g.148810377A>G",
        vaf=0.59,
        transcript="NM_004456.5",
        chrom="7",
        pos_grch38=148_810_377,
        ref="A",
        alt="G",
    ),
]


# -- Strategy 1: AlphaGenome Python package -----------------------------------


def try_alphagenome_package(variants: list[PatientVariant]) -> bool:
    """Attempt to use the alphagenome Python package for variant effect prediction.

    AlphaGenome (DeepMind, announced Google I/O May 2025) predicts regulatory
    effects from DNA sequence at single base-pair resolution. If the package is
    available, we use it to predict chromatin accessibility delta, gene expression
    changes, and histone modification shifts for each variant.

    Returns True if successful for at least one variant, False otherwise.
    """
    try:
        import alphagenome  # type: ignore[import-not-found]
    except ImportError:
        log.warning("alphagenome package not installed (pip install alphagenome)")
        return False

    log.info("AlphaGenome package found. Attempting native predictions...")

    success = False
    for v in variants:
        try:
            model = alphagenome.AlphaGenome()

            # Define the genomic interval centered on the variant
            half_ctx = ALPHAGENOME_CONTEXT_BP // 2
            interval = alphagenome.Interval(
                chromosome=f"chr{v.chrom}",
                start=v.pos_grch38 - half_ctx,
                end=v.pos_grch38 + half_ctx,
            )

            # Define the variant
            variant = alphagenome.Variant(
                chromosome=f"chr{v.chrom}",
                position=v.pos_grch38,
                ref=v.ref,
                alt=v.alt,
            )

            # Predict variant effects
            result = model.predict_variant(interval, variant)

            v.alphagenome_result = {
                "chromatin_accessibility_delta": getattr(result, "chromatin_accessibility_delta", None),
                "gene_expression_changes": getattr(result, "gene_expression_changes", None),
                "histone_modification_shifts": getattr(result, "histone_modification_shifts", None),
                "tf_binding_changes": getattr(result, "tf_binding_changes", None),
                "splice_site_effects": getattr(result, "splice_site_effects", None),
            }
            v.method_used = "alphagenome_package"
            log.info("  %s %s: AlphaGenome prediction successful", v.gene, v.protein_change)
            success = True

        except Exception as exc:
            log.warning("  %s %s: AlphaGenome prediction failed: %s", v.gene, v.protein_change, exc)
            v.error = f"AlphaGenome package error: {exc}"

    return success


# -- Strategy 2: AlphaGenome API endpoint ------------------------------------


def try_alphagenome_api(variants: list[PatientVariant]) -> bool:
    """Attempt to use the AlphaGenome REST API for variant effect prediction.

    Tries known potential API endpoints for AlphaGenome. Since the API may not
    be publicly available yet, this is a best-effort attempt.

    Returns True if successful for at least one variant, False otherwise.
    """
    api_candidates = [
        "https://alphagenome.deepmind.com/api/v1/predict",
        "https://api.deepmind.com/alphagenome/v1/predict",
        "https://alphagenome.googleapis.com/v1/predict",
    ]

    # Probe endpoints for availability
    available_url = None
    for url in api_candidates:
        try:
            resp = requests.head(url, timeout=5)
            if resp.status_code < 500:
                available_url = url
                log.info("AlphaGenome API endpoint responding at %s (HTTP %d)", url, resp.status_code)
                break
        except requests.RequestException:
            continue

    if available_url is None:
        log.warning("No AlphaGenome API endpoint found (tried %d candidates)", len(api_candidates))
        return False

    log.info("Attempting AlphaGenome API predictions via %s ...", available_url)

    success = False
    for v in variants:
        try:
            payload = {
                "chromosome": f"chr{v.chrom}",
                "position": v.pos_grch38,
                "ref": v.ref,
                "alt": v.alt,
                "assembly": "GRCh38",
                "context_bp": ALPHAGENOME_CONTEXT_BP,
            }
            resp = requests.post(available_url, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                v.alphagenome_result = data
                v.method_used = "alphagenome_api"
                log.info("  %s %s: AlphaGenome API prediction successful", v.gene, v.protein_change)
                success = True
            else:
                log.warning(
                    "  %s %s: AlphaGenome API returned HTTP %d",
                    v.gene, v.protein_change, resp.status_code,
                )
        except requests.RequestException as exc:
            log.warning("  %s %s: AlphaGenome API request failed: %s", v.gene, v.protein_change, exc)

        time.sleep(RATE_LIMIT_DELAY)

    return success


# -- Strategy 3a: Generate AlphaGenome input files ----------------------------


def generate_alphagenome_inputs(variants: list[PatientVariant]) -> None:
    """Generate AlphaGenome-compatible input files for manual submission.

    Creates VCF and BED files that can be submitted to the AlphaGenome web
    interface or used with the alphagenome CLI when it becomes available.
    """
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate VCF file
    vcf_path = INPUTS_DIR / "patient_variants.vcf"
    vcf_lines = [
        "##fileformat=VCFv4.3",
        "##reference=GRCh38",
        f"##fileDate={datetime.now(timezone.utc).strftime('%Y%m%d')}",
        '##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">',
        '##INFO=<ID=AACHANGE,Number=1,Type=String,Description="Amino acid change">',
        '##INFO=<ID=VAF,Number=1,Type=Float,Description="Variant allele frequency">',
        '##INFO=<ID=HGVSC,Number=1,Type=String,Description="HGVS coding notation">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]
    for v in variants:
        info = f"GENE={v.gene};AACHANGE={v.protein_change};VAF={v.vaf};HGVSC={v.hgvs_c}"
        variant_id = f"{v.gene}_{v.protein_change}"
        vcf_lines.append(
            f"chr{v.chrom}\t{v.pos_grch38}\t{variant_id}\t{v.ref}\t{v.alt}\t.\tPASS\t{info}"
        )
    vcf_path.write_text("\n".join(vcf_lines) + "\n")
    log.info("Generated VCF input: %s", vcf_path)

    # Generate BED file with context windows for AlphaGenome
    bed_path = INPUTS_DIR / "patient_variants_context.bed"
    half_ctx = ALPHAGENOME_CONTEXT_BP // 2
    bed_lines = [
        f"# AlphaGenome context windows ({ALPHAGENOME_CONTEXT_BP:,} bp) for patient variants",
        "# chrom\tstart\tend\tname\tstrand",
    ]
    for v in variants:
        start = max(0, v.pos_grch38 - half_ctx)
        end = v.pos_grch38 + half_ctx
        name = f"{v.gene}_{v.protein_change}"
        bed_lines.append(f"chr{v.chrom}\t{start}\t{end}\t{name}\t.")
    bed_path.write_text("\n".join(bed_lines) + "\n")
    log.info("Generated BED input: %s", bed_path)

    # Generate JSON input (alternative format)
    json_input_path = INPUTS_DIR / "patient_variants_input.json"
    json_input = {
        "assembly": "GRCh38",
        "context_bp": ALPHAGENOME_CONTEXT_BP,
        "variants": [
            {
                "gene": v.gene,
                "protein_change": v.protein_change,
                "chromosome": f"chr{v.chrom}",
                "position": v.pos_grch38,
                "ref": v.ref,
                "alt": v.alt,
                "hgvs_c": v.hgvs_c,
                "hgvs_genomic": v.hgvs_genomic,
                "transcript": v.transcript,
            }
            for v in variants
        ],
    }
    json_input_path.write_text(json.dumps(json_input, indent=2) + "\n")
    log.info("Generated JSON input: %s", json_input_path)

    # Generate instructions file
    readme_path = INPUTS_DIR / "SUBMISSION_INSTRUCTIONS.txt"
    readme_lines = [
        "AlphaGenome Manual Submission Instructions",
        "=" * 50,
        "",
        "The alphagenome Python package and API are not yet publicly available.",
        "Use these input files when AlphaGenome access becomes available.",
        "",
        "Files generated:",
        "  - patient_variants.vcf          : VCF format (standard genomic variant format)",
        "  - patient_variants_context.bed   : BED format with 196kb context windows",
        "  - patient_variants_input.json    : JSON format with full variant metadata",
        "",
        "When AlphaGenome is available, try:",
        "",
        "  Option A: Python package",
        "    pip install alphagenome",
        "    python alphagenome_regulatory.py   # will auto-detect package",
        "",
        "  Option B: Web interface",
        "    Upload patient_variants.vcf to the AlphaGenome web portal",
        "",
        "  Option C: API",
        "    POST each variant to the AlphaGenome API endpoint",
        "",
        "Expected output predictions:",
        "  - Chromatin accessibility delta (ATAC-seq equivalent)",
        "  - Gene expression changes across cell types",
        "  - Histone modification shifts (H3K4me3, H3K27ac, H3K27me3, etc.)",
        "  - Transcription factor binding site disruption/creation",
        "  - Splice site usage changes",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n")
    log.info("Generated submission instructions: %s", readme_path)


# -- Strategy 3b: Ensembl VEP regulatory fallback ----------------------------


def _ensembl_headers() -> dict[str, str]:
    """Standard headers for Ensembl REST API requests."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def query_vep_regulatory(variant: PatientVariant) -> dict[str, Any] | None:
    """Query Ensembl VEP with regulatory annotations for a single variant.

    Uses the VEP HGVS endpoint with regulatory=1 to get:
    - Regulatory feature overlaps (promoters, enhancers, CTCF sites, open chromatin)
    - Transcription factor binding site motif overlaps
    - Cell-type-specific activity annotations for hematopoietic cells
    """
    log.info("  Querying VEP for %s %s (%s) ...", variant.gene, variant.protein_change, variant.hgvs_genomic)

    payload = {
        "hgvs_notations": [variant.hgvs_genomic],
        "regulatory": True,
        "cell_type": HEMATOPOIETIC_CELL_TYPES,
        "variant_class": True,
        "numbers": True,
        "protein": True,
        "domains": True,
    }

    try:
        resp = requests.post(
            ENSEMBL_VEP_HGVS,
            json=payload,
            headers=_ensembl_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 5))
            log.warning("  VEP rate limited, waiting %d seconds...", retry_after)
            time.sleep(retry_after)
            resp = requests.post(
                ENSEMBL_VEP_HGVS,
                json=payload,
                headers=_ensembl_headers(),
                timeout=REQUEST_TIMEOUT,
            )

        if resp.status_code != 200:
            log.error("  VEP returned HTTP %d for %s: %s", resp.status_code, variant.hgvs_genomic, resp.text[:200])
            return None

        data = resp.json()
        if not data or not isinstance(data, list):
            log.warning("  VEP returned empty response for %s", variant.hgvs_genomic)
            return None

        return data[0]

    except requests.RequestException as exc:
        log.error("  VEP request failed for %s: %s", variant.hgvs_genomic, exc)
        return None


def query_nearby_regulatory_elements(variant: PatientVariant, window_bp: int = 5000) -> list[dict[str, Any]]:
    """Query Ensembl Overlap API for regulatory features near the variant.

    Searches a window around the variant position for regulatory features
    (promoters, enhancers, CTCF binding sites, TF binding sites, open chromatin).
    """
    start = max(1, variant.pos_grch38 - window_bp)
    end = variant.pos_grch38 + window_bp
    region = f"{variant.chrom}:{start}-{end}"

    log.info("  Querying regulatory elements near %s (%s +/- %d bp) ...", variant.gene, region, window_bp)

    elements: list[dict[str, Any]] = []

    # Query regulatory features
    for feature_type in ["regulatory", "motif"]:
        try:
            url = f"{ENSEMBL_REGULATORY}/{region}"
            params = {
                "content-type": "application/json",
                "feature": feature_type,
            }
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 200:
                features = resp.json()
                if features and isinstance(features, list):
                    for feat in features:
                        feat["query_feature_type"] = feature_type
                        feat["distance_to_variant"] = abs(
                            (feat.get("start", 0) + feat.get("end", 0)) // 2 - variant.pos_grch38
                        )
                    elements.extend(features)
                    log.info("    Found %d %s features near %s", len(features), feature_type, variant.gene)
            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                log.warning("    Rate limited, waiting %d seconds...", retry_after)
                time.sleep(retry_after)
            else:
                log.warning("    Overlap API returned HTTP %d for %s features", resp.status_code, feature_type)

        except requests.RequestException as exc:
            log.error("    Overlap API request failed for %s: %s", feature_type, exc)

        time.sleep(RATE_LIMIT_DELAY)

    return elements


def parse_vep_regulatory(variant: PatientVariant, vep_data: dict[str, Any]) -> None:
    """Extract regulatory annotations from VEP response and populate the variant."""
    # Regulatory feature consequences
    for rc in vep_data.get("regulatory_feature_consequences", []):
        parsed = {
            "regulatory_feature_id": rc.get("regulatory_feature_id"),
            "biotype": rc.get("biotype"),
            "consequence_terms": rc.get("consequence_terms", []),
            "impact": rc.get("impact"),
            "variant_allele": rc.get("variant_allele"),
        }

        # Extract cell-type-specific activity
        cell_type_activity = {}
        for ct in HEMATOPOIETIC_CELL_TYPES:
            activity = rc.get(f"{ct}_activity")
            if activity:
                cell_type_activity[ct] = activity
        if cell_type_activity:
            parsed["cell_type_activity"] = cell_type_activity

        variant.regulatory_consequences.append(parsed)

    # Motif feature consequences (TF binding site overlaps)
    for mc in vep_data.get("motif_feature_consequences", []):
        parsed = {
            "motif_feature_id": mc.get("motif_feature_id"),
            "motif_name": mc.get("motif_name"),
            "motif_pos": mc.get("motif_pos"),
            "motif_score_change": mc.get("motif_score_change"),
            "high_inf_pos": mc.get("high_inf_pos"),
            "consequence_terms": mc.get("consequence_terms", []),
            "impact": mc.get("impact"),
            "strand": mc.get("strand"),
            "transcription_factors": mc.get("transcription_factors", []),
        }
        variant.motif_features.append(parsed)

    # Transcript consequences (for completeness -- regulatory context of coding variant)
    for tc in vep_data.get("transcript_consequences", []):
        if tc.get("transcript_id", "").startswith(variant.transcript.split(".")[0]):
            parsed = {
                "transcript_id": tc.get("transcript_id"),
                "consequence_terms": tc.get("consequence_terms", []),
                "impact": tc.get("impact"),
                "biotype": tc.get("biotype"),
                "gene_symbol": tc.get("gene_symbol"),
                "protein_start": tc.get("protein_start"),
                "protein_end": tc.get("protein_end"),
                "amino_acids": tc.get("amino_acids"),
                "codons": tc.get("codons"),
                "domains": tc.get("domains", []),
            }
            variant.transcript_consequences.append(parsed)


def classify_chromatin_state(variant: PatientVariant) -> None:
    """Classify the chromatin state at the variant position based on regulatory annotations."""
    # Check regulatory feature overlaps
    biotypes = {rc.get("biotype") for rc in variant.regulatory_consequences if rc.get("biotype")}

    if "promoter" in biotypes:
        variant.chromatin_state = "Promoter"
    elif "enhancer" in biotypes:
        variant.chromatin_state = "Enhancer"
    elif "CTCF_binding_site" in biotypes:
        variant.chromatin_state = "CTCF binding site"
    elif "open_chromatin_region" in biotypes:
        variant.chromatin_state = "Open chromatin"
    elif "TF_binding_site" in biotypes:
        variant.chromatin_state = "TF binding site"
    elif "promoter_flanking_region" in biotypes:
        variant.chromatin_state = "Promoter flanking region"
    else:
        # Check nearby elements for context
        nearby_types = {
            elem.get("feature_type") or elem.get("description", "")
            for elem in variant.nearby_regulatory_elements
        }
        if nearby_types:
            variant.chromatin_state = f"Coding (nearest regulatory: {', '.join(sorted(nearby_types)[:3])})"
        else:
            variant.chromatin_state = "Coding region (no regulatory feature overlap)"


def assess_hematopoietic_relevance(variant: PatientVariant) -> None:
    """Assess whether the variant falls in a regulatory element relevant to hematopoietic cells."""
    hsc_cell_types = {"CD34+_mobilized", "HL-60", "K562"}  # HSC/myeloid relevant
    relevance_notes = []

    # Check cell-type-specific activity in regulatory consequences
    for rc in variant.regulatory_consequences:
        cell_activity = rc.get("cell_type_activity", {})
        for ct in hsc_cell_types:
            activity = cell_activity.get(ct)
            if activity and activity.upper() == "ACTIVE":
                relevance_notes.append(
                    f"Overlaps {rc.get('biotype', 'regulatory feature')} "
                    f"active in {ct}"
                )
            elif activity and activity.upper() == "POISED":
                relevance_notes.append(
                    f"Overlaps {rc.get('biotype', 'regulatory feature')} "
                    f"poised in {ct}"
                )

    # Check motif features for hematopoietic TFs
    hematopoietic_tfs = {
        "RUNX1", "GATA1", "GATA2", "TAL1", "SCL", "PU.1", "SPI1",
        "CEBPA", "C/EBPa", "ETV6", "FLI1", "ERG", "MYB",
        "KLF1", "GFI1", "GFI1B", "IKZF1", "IKAROS",
    }
    for mf in variant.motif_features:
        motif_name = mf.get("motif_name", "")
        tfs = mf.get("transcription_factors", [])
        matched_tfs = [tf for tf in tfs if any(ht in tf.upper() for ht in {t.upper() for t in hematopoietic_tfs})]
        if matched_tfs:
            relevance_notes.append(
                f"Overlaps TF binding motif ({motif_name}) for hematopoietic TFs: {', '.join(matched_tfs)}"
            )

    # Check nearby elements for hematopoietic relevance
    for elem in variant.nearby_regulatory_elements:
        cell_activity = elem.get("activity_evidence", {})
        if isinstance(cell_activity, dict):
            for ct in hsc_cell_types:
                if cell_activity.get(ct, "").upper() == "ACTIVE":
                    dist = elem.get("distance_to_variant", "?")
                    relevance_notes.append(
                        f"Near {elem.get('description', 'regulatory element')} "
                        f"active in {ct} ({dist} bp away)"
                    )

    if relevance_notes:
        variant.hematopoietic_relevance = "; ".join(relevance_notes)
    else:
        # For coding variants, note the gene's role in hematopoiesis
        gene_hema_roles = {
            "DNMT3A": "DNMT3A is a master epigenetic regulator in HSC self-renewal; "
                      "R882H causes dominant-negative hypomethylation at hematopoietic enhancers",
            "IDH2": "IDH2 R140Q produces 2-HG which inhibits TET2-mediated demethylation "
                    "in hematopoietic progenitors, blocking differentiation",
            "SETBP1": "SETBP1 mutations stabilize SET protein, activating HOXA9/HOXA10 "
                      "in myeloid progenitors and driving MDS/MPN",
            "PTPN11": "PTPN11/SHP2 is a key RAS-MAPK pathway regulator in hematopoietic "
                      "cytokine signaling; E76Q constitutively activates myeloid proliferation",
            "EZH2": "EZH2 is the catalytic subunit of PRC2; V662A (SET domain, LoF) "
                    "causes loss of H3K27me3 at hematopoietic differentiation genes. "
                    "On monosomy 7: hemizygous = effective biallelic inactivation",
        }
        variant.hematopoietic_relevance = (
            f"No direct regulatory element overlap, but coding mutation in gene "
            f"with critical hematopoietic function. {gene_hema_roles.get(variant.gene, '')}"
        )


def build_regulatory_summary(variant: PatientVariant) -> None:
    """Build a concise regulatory impact summary for the variant."""
    parts = []

    # Coding consequence
    if variant.transcript_consequences:
        tc = variant.transcript_consequences[0]
        consequences = ", ".join(tc.get("consequence_terms", []))
        impact = tc.get("impact", "unknown")
        parts.append(f"Coding: {consequences} ({impact} impact)")

        domains = tc.get("domains", [])
        if domains:
            domain_names = [d.get("db", "") + ":" + d.get("name", "") for d in domains[:3]]
            parts.append(f"Domains: {', '.join(domain_names)}")

    # Regulatory overlaps
    n_reg = len(variant.regulatory_consequences)
    if n_reg > 0:
        biotypes = [rc.get("biotype", "unknown") for rc in variant.regulatory_consequences]
        parts.append(f"Regulatory overlaps: {n_reg} ({', '.join(set(biotypes))})")

    # Motif features
    n_motif = len(variant.motif_features)
    if n_motif > 0:
        motif_names = [mf.get("motif_name", "unknown") for mf in variant.motif_features]
        parts.append(f"TF motifs affected: {n_motif} ({', '.join(set(motif_names)[:5])})")

    # Nearby regulatory elements
    n_nearby = len(variant.nearby_regulatory_elements)
    if n_nearby > 0:
        parts.append(f"Nearby regulatory elements: {n_nearby} within 5 kb")

    # Chromatin state
    if variant.chromatin_state:
        parts.append(f"Chromatin context: {variant.chromatin_state}")

    variant.regulatory_summary = " | ".join(parts) if parts else "No regulatory annotations retrieved"


def run_vep_fallback(variants: list[PatientVariant]) -> int:
    """Run the Ensembl VEP regulatory fallback for all variants.

    Returns the number of variants successfully annotated.
    """
    log.info("Running Ensembl VEP regulatory annotation fallback...")
    log.info("Querying VEP with regulatory=True and %d hematopoietic cell types", len(HEMATOPOIETIC_CELL_TYPES))

    success_count = 0

    for v in variants:
        if v.method_used not in ("pending", "vep_fallback"):
            log.info("  Skipping %s %s (already annotated via %s)", v.gene, v.protein_change, v.method_used)
            success_count += 1
            continue

        # Query VEP regulatory annotations
        vep_data = query_vep_regulatory(v)
        if vep_data:
            v.raw_vep_response = vep_data
            parse_vep_regulatory(v, vep_data)
            v.method_used = "vep_fallback"
            success_count += 1
        else:
            v.method_used = "failed"
            v.error = "VEP query returned no data"

        time.sleep(RATE_LIMIT_DELAY)

        # Query nearby regulatory elements
        nearby = query_nearby_regulatory_elements(v)
        v.nearby_regulatory_elements = nearby

        time.sleep(RATE_LIMIT_DELAY)

        # Classify chromatin state and hematopoietic relevance
        classify_chromatin_state(v)
        assess_hematopoietic_relevance(v)
        build_regulatory_summary(v)

        log.info(
            "  %s %s: %d regulatory overlaps, %d motifs, %d nearby elements, chromatin=%s",
            v.gene,
            v.protein_change,
            len(v.regulatory_consequences),
            len(v.motif_features),
            len(v.nearby_regulatory_elements),
            v.chromatin_state,
        )

    return success_count


# -- Output generation --------------------------------------------------------


def build_json_output(variants: list[PatientVariant], method: str, timestamp: str) -> dict[str, Any]:
    """Build the JSON output structure."""
    variant_results = []
    for v in variants:
        result: dict[str, Any] = {
            "gene": v.gene,
            "protein_change": v.protein_change,
            "hgvs_c": v.hgvs_c,
            "hgvs_genomic": v.hgvs_genomic,
            "vaf": v.vaf,
            "transcript": v.transcript,
            "chrom": v.chrom,
            "pos_grch38": v.pos_grch38,
            "ref": v.ref,
            "alt": v.alt,
            "method_used": v.method_used,
            "chromatin_state": v.chromatin_state,
            "hematopoietic_relevance": v.hematopoietic_relevance,
            "regulatory_summary": v.regulatory_summary,
            "regulatory_consequences": v.regulatory_consequences,
            "motif_features": v.motif_features,
            "transcript_consequences": v.transcript_consequences,
            "nearby_regulatory_elements_count": len(v.nearby_regulatory_elements),
            "nearby_regulatory_elements": v.nearby_regulatory_elements[:20],  # cap to avoid huge JSON
        }
        if v.alphagenome_result:
            result["alphagenome_result"] = v.alphagenome_result
        if v.error:
            result["error"] = v.error
        variant_results.append(result)

    return {
        "timestamp": timestamp,
        "method": method,
        "assembly": "GRCh38",
        "sources": _method_sources(method),
        "variants_queried": len(variants),
        "variants_annotated": sum(1 for v in variants if v.method_used not in ("pending", "failed")),
        "hematopoietic_cell_types_queried": HEMATOPOIETIC_CELL_TYPES,
        "results": variant_results,
    }


def _method_sources(method: str) -> list[str]:
    """Return data source descriptions for the method used."""
    if method == "alphagenome_package":
        return ["AlphaGenome (DeepMind) Python package"]
    if method == "alphagenome_api":
        return ["AlphaGenome (DeepMind) REST API"]
    return [
        "Ensembl VEP REST API (GRCh38) with regulatory annotations",
        "Ensembl Regulatory Build (ENCODE + Roadmap Epigenomics)",
        "AlphaGenome input files generated for future submission",
    ]


def generate_report(variants: list[PatientVariant], method: str, timestamp: str) -> str:
    """Generate the markdown regulatory impact report."""
    lines = [
        "# AlphaGenome Regulatory Impact Predictions",
        "",
        f"**Generated:** {timestamp}",
        f"**Assembly:** GRCh38",
        f"**Method:** {_method_label(method)}",
        "",
    ]

    # Method explanation
    if method == "vep_fallback":
        lines.extend([
            "## Method Note",
            "",
            "The AlphaGenome Python package and API are not yet publicly available",
            "(announced at Google I/O May 2025). This analysis uses the **Ensembl VEP**",
            "**REST API** with regulatory annotations as a practical fallback:",
            "",
            "- **Regulatory feature overlaps**: promoters, enhancers, CTCF sites, open chromatin",
            "- **TF binding motif overlaps**: transcription factor binding site disruption",
            "- **Cell-type-specific activity**: annotations from ENCODE and Roadmap Epigenomics",
            "  for hematopoietic cell types (CD34+, K562, HL-60, monocytes, etc.)",
            "",
            "AlphaGenome input files have been generated in the `alphagenome_inputs/`",
            "directory for submission when the model becomes accessible.",
            "",
        ])
    elif method in ("alphagenome_package", "alphagenome_api"):
        lines.extend([
            "## Method Note",
            "",
            f"Predictions generated using AlphaGenome ({'Python package' if method == 'alphagenome_package' else 'REST API'}).",
            "AlphaGenome predicts thousands of molecular properties at single base-pair",
            "resolution from DNA sequence: gene expression, chromatin accessibility,",
            "histone modifications, TF binding, and splice site usage.",
            "",
        ])

    # Summary table
    lines.extend([
        "## Summary Table",
        "",
        "| Variant | Gene | VAF | Chromatin Context | Regulatory Overlaps | TF Motifs | Nearby Elements | Hematopoietic Relevance |",
        "|---------|------|-----|-------------------|--------------------:|----------:|----------------:|------------------------|",
    ])

    for v in variants:
        hema_short = _truncate(v.hematopoietic_relevance or "N/A", 60)
        lines.append(
            f"| {v.protein_change} | {v.gene} | {v.vaf:.0%} "
            f"| {v.chromatin_state or 'N/A'} "
            f"| {len(v.regulatory_consequences)} "
            f"| {len(v.motif_features)} "
            f"| {len(v.nearby_regulatory_elements)} "
            f"| {hema_short} |"
        )
    lines.append("")

    # Per-variant detailed sections
    for v in variants:
        lines.extend([
            f"## {v.gene} {v.protein_change}",
            "",
            f"- **HGVS:** {v.hgvs_c} ({v.hgvs_genomic})",
            f"- **Position:** chr{v.chrom}:{v.pos_grch38:,} ({v.ref}>{v.alt})",
            f"- **VAF:** {v.vaf:.0%}",
            f"- **Transcript:** {v.transcript}",
            f"- **Method:** {v.method_used}",
            f"- **Chromatin state:** {v.chromatin_state or 'N/A'}",
            "",
        ])

        # Regulatory feature consequences
        if v.regulatory_consequences:
            lines.extend([
                "### Regulatory Feature Overlaps",
                "",
                "| Feature ID | Biotype | Consequence | Impact |",
                "|------------|---------|-------------|--------|",
            ])
            for rc in v.regulatory_consequences:
                consequences = ", ".join(rc.get("consequence_terms", []))
                lines.append(
                    f"| {rc.get('regulatory_feature_id', 'N/A')} "
                    f"| {rc.get('biotype', 'N/A')} "
                    f"| {consequences} "
                    f"| {rc.get('impact', 'N/A')} |"
                )

                # Cell-type activity
                cell_activity = rc.get("cell_type_activity", {})
                if cell_activity:
                    lines.append("")
                    lines.append(f"**Cell-type activity for {rc.get('regulatory_feature_id', 'feature')}:**")
                    lines.append("")
                    for ct, activity in sorted(cell_activity.items()):
                        marker = "**ACTIVE**" if activity.upper() == "ACTIVE" else activity
                        lines.append(f"  - {ct}: {marker}")
            lines.append("")
        else:
            lines.extend([
                "### Regulatory Feature Overlaps",
                "",
                "No direct regulatory feature overlaps detected. This variant is in a coding region.",
                "",
            ])

        # Motif features
        if v.motif_features:
            lines.extend([
                "### Transcription Factor Motif Overlaps",
                "",
                "| Motif | Position in Motif | Score Change | High Info Position | TFs |",
                "|-------|------------------:|-------------:|-------------------:|-----|",
            ])
            for mf in v.motif_features:
                tfs = ", ".join(mf.get("transcription_factors", [])[:5])
                score_change = mf.get("motif_score_change")
                score_str = f"{score_change:.4f}" if score_change is not None else "N/A"
                high_inf = "Yes" if mf.get("high_inf_pos") == "Y" else "No"
                lines.append(
                    f"| {mf.get('motif_name', 'N/A')} "
                    f"| {mf.get('motif_pos', 'N/A')} "
                    f"| {score_str} "
                    f"| {high_inf} "
                    f"| {tfs or 'N/A'} |"
                )
            lines.append("")
        else:
            lines.extend([
                "### Transcription Factor Motif Overlaps",
                "",
                "No TF binding motif overlaps detected at this position.",
                "",
            ])

        # Transcript consequences
        if v.transcript_consequences:
            tc = v.transcript_consequences[0]
            consequences = ", ".join(tc.get("consequence_terms", []))
            lines.extend([
                "### Coding Consequence",
                "",
                f"- **Consequence:** {consequences}",
                f"- **Impact:** {tc.get('impact', 'N/A')}",
                f"- **Amino acids:** {tc.get('amino_acids', 'N/A')}",
                f"- **Codons:** {tc.get('codons', 'N/A')}",
            ])
            domains = tc.get("domains", [])
            if domains:
                lines.append(f"- **Protein domains:**")
                for d in domains:
                    lines.append(f"  - {d.get('db', '')}: {d.get('name', '')}")
            lines.append("")

        # Nearby regulatory elements summary
        if v.nearby_regulatory_elements:
            lines.extend([
                "### Nearby Regulatory Elements (within 5 kb)",
                "",
                f"Found **{len(v.nearby_regulatory_elements)}** regulatory elements within 5 kb.",
                "",
            ])

            # Group by type
            by_type: dict[str, int] = {}
            for elem in v.nearby_regulatory_elements:
                etype = elem.get("feature_type") or elem.get("description") or elem.get("query_feature_type", "unknown")
                by_type[etype] = by_type.get(etype, 0) + 1

            if by_type:
                lines.append("| Element Type | Count |")
                lines.append("|-------------|------:|")
                for etype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                    lines.append(f"| {etype} | {count} |")
                lines.append("")

            # Show closest elements
            sorted_elements = sorted(v.nearby_regulatory_elements, key=lambda e: e.get("distance_to_variant", 999999))
            closest = sorted_elements[:5]
            if closest:
                lines.append("**Closest elements:**")
                lines.append("")
                for elem in closest:
                    dist = elem.get("distance_to_variant", "?")
                    desc = elem.get("description") or elem.get("feature_type") or elem.get("query_feature_type", "unknown")
                    eid = elem.get("id", "")
                    lines.append(f"- {desc} ({eid}): {dist:,} bp away" if isinstance(dist, int) else f"- {desc} ({eid}): {dist} bp away")
                lines.append("")
        else:
            lines.extend([
                "### Nearby Regulatory Elements (within 5 kb)",
                "",
                "No regulatory elements found within 5 kb.",
                "",
            ])

        # Hematopoietic relevance
        lines.extend([
            "### Hematopoietic Relevance",
            "",
            v.hematopoietic_relevance or "Not assessed.",
            "",
        ])

        # AlphaGenome results (if available)
        if v.alphagenome_result:
            lines.extend([
                "### AlphaGenome Predictions",
                "",
            ])
            for key, value in v.alphagenome_result.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

    # Integrated analysis
    lines.extend([
        "## Integrated Regulatory Analysis",
        "",
        "### Key Findings",
        "",
    ])

    # Summarize findings per variant
    for v in variants:
        n_reg = len(v.regulatory_consequences)
        n_motif = len(v.motif_features)
        n_nearby = len(v.nearby_regulatory_elements)

        finding = f"- **{v.gene} {v.protein_change}** (VAF {v.vaf:.0%}): "
        if n_reg > 0 or n_motif > 0:
            finding += f"{n_reg} regulatory feature overlaps, {n_motif} TF motif overlaps. "
        else:
            finding += "Coding variant without direct regulatory element overlap. "
        finding += f"{n_nearby} regulatory elements within 5 kb. "
        finding += f"Chromatin: {v.chromatin_state or 'N/A'}."
        lines.append(finding)
    lines.append("")

    # Epigenetic cascade interpretation
    lines.extend([
        "### Epigenetic Cascade in This Patient",
        "",
        "The 5 patient mutations converge on epigenetic and signaling pathways that",
        "fundamentally alter the regulatory landscape of hematopoietic stem cells:",
        "",
        "1. **DNMT3A R882H** (VAF 39%): Dominant-negative loss of DNA methyltransferase",
        "   activity causes genome-wide hypomethylation, particularly at hematopoietic",
        "   enhancers and CpG islands. This derepresses self-renewal programs (HOXA cluster)",
        "   and creates a permissive epigenetic state for secondary mutations.",
        "",
        "2. **EZH2 V662A** (VAF 59%, LoF): Loss of PRC2 catalytic activity removes",
        "   H3K27me3 repressive marks at differentiation genes. Combined with monosomy 7",
        "   (hemizygous loss), this creates effective biallelic EZH2 inactivation.",
        "   The dual loss of DNMT3A + EZH2 creates a profoundly disrupted epigenome",
        "   where both DNA methylation and histone-mediated repression are impaired.",
        "",
        "3. **IDH2 R140Q** (VAF 2%): Produces the oncometabolite 2-hydroxyglutarate (2-HG),",
        "   which inhibits TET2, KDM histone demethylases, and other alpha-ketoglutarate-dependent",
        "   enzymes. This creates focal DNA hypermethylation and altered histone marks,",
        "   paradoxically opposing and complementing the DNMT3A hypomethylation.",
        "",
        "4. **SETBP1 G870S** (VAF 34%): Stabilizes SET protein, which inhibits PP2A",
        "   phosphatase. This activates HOXA9/HOXA10 transcription programs and cooperates",
        "   with the epigenetic deregulation to drive MDS/MPN transformation.",
        "",
        "5. **PTPN11 E76Q** (VAF 29%): Constitutive SHP2 activation amplifies RAS-MAPK",
        "   signaling downstream of cytokine receptors. This provides the proliferative",
        "   signal that, combined with the epigenetic permissiveness above, drives",
        "   clonal expansion.",
        "",
        "This combination of 3 epigenetic regulators (DNMT3A, EZH2, IDH2) + 1 signaling",
        "modifier (PTPN11) + 1 chromatin/SET pathway effector (SETBP1) represents a",
        "comprehensive attack on the hematopoietic epigenome from multiple axes.",
        "",
    ])

    # Methodology
    lines.extend([
        "## Methodology",
        "",
        "### Data Sources",
        "",
    ])
    for source in _method_sources(method):
        lines.append(f"- {source}")
    lines.append("")

    lines.extend([
        "### Query Parameters",
        "",
        f"- **Assembly:** GRCh38",
        f"- **VEP regulatory:** enabled",
        f"- **Hematopoietic cell types queried:** {len(HEMATOPOIETIC_CELL_TYPES)}",
    ])
    for ct in HEMATOPOIETIC_CELL_TYPES:
        lines.append(f"  - {ct}")
    lines.append(f"- **Nearby regulatory element window:** +/- 5,000 bp")
    lines.append("")

    lines.extend([
        "### Limitations",
        "",
        "- All 5 patient variants are coding missense mutations; regulatory impact",
        "  predictions are most informative for non-coding variants. For coding variants,",
        "  the primary effect is through protein function, not cis-regulatory disruption.",
        "- The Ensembl Regulatory Build cell-type annotations are derived from bulk",
        "  epigenomic data (ENCODE/Roadmap) and may not capture the precise chromatin",
        "  state in the patient's malignant clone.",
        "- AlphaGenome (when available) would provide higher-resolution predictions of",
        "  variant effects on chromatin accessibility, gene expression, and TF binding",
        "  across multiple cell types, including context-dependent effects.",
        "",
        "---",
        f"*Generated by alphagenome_regulatory.py on {timestamp}*",
    ])

    return "\n".join(lines)


def _method_label(method: str) -> str:
    """Human-readable method label."""
    labels = {
        "alphagenome_package": "AlphaGenome Python package (native)",
        "alphagenome_api": "AlphaGenome REST API",
        "vep_fallback": "Ensembl VEP regulatory annotations (AlphaGenome not yet available)",
    }
    return labels.get(method, method)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# -- Main --------------------------------------------------------------------


def main() -> None:
    log.info("AlphaGenome Regulatory Impact Predictions")
    log.info("=" * 60)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    method = "vep_fallback"  # default, updated if AlphaGenome works

    # Strategy 1: Try AlphaGenome Python package
    log.info("Strategy 1: Attempting AlphaGenome Python package...")
    if try_alphagenome_package(PATIENT_VARIANTS):
        method = "alphagenome_package"
        log.info("AlphaGenome package predictions successful.")
    else:
        log.info("AlphaGenome package not available.")

        # Strategy 2: Try AlphaGenome API
        log.info("Strategy 2: Attempting AlphaGenome API endpoint...")
        if try_alphagenome_api(PATIENT_VARIANTS):
            method = "alphagenome_api"
            log.info("AlphaGenome API predictions successful.")
        else:
            log.info("AlphaGenome API not available.")

            # Strategy 3a: Generate AlphaGenome input files
            log.info("Strategy 3a: Generating AlphaGenome input files for manual submission...")
            generate_alphagenome_inputs(PATIENT_VARIANTS)

            # Strategy 3b: Run VEP regulatory fallback
            log.info("Strategy 3b: Running Ensembl VEP regulatory annotation fallback...")
            n_annotated = run_vep_fallback(PATIENT_VARIANTS)
            method = "vep_fallback"
            log.info("VEP fallback complete: %d/%d variants annotated", n_annotated, len(PATIENT_VARIANTS))

    # Save JSON results
    output = build_json_output(PATIENT_VARIANTS, method, timestamp)
    JSON_OUTPUT.write_text(json.dumps(output, indent=2, default=str) + "\n")
    log.info("Saved JSON results to %s", JSON_OUTPUT)

    # Generate markdown report
    report = generate_report(PATIENT_VARIANTS, method, timestamp)
    MD_OUTPUT.write_text(report + "\n")
    log.info("Saved markdown report to %s", MD_OUTPUT)

    # Print summary
    log.info("")
    log.info("=" * 60)
    log.info("REGULATORY IMPACT SUMMARY")
    log.info("=" * 60)
    log.info("Method: %s", _method_label(method))
    log.info("")
    for v in PATIENT_VARIANTS:
        log.info(
            "  %s %s (VAF %s): %d reg overlaps, %d motifs, %d nearby | %s",
            v.gene,
            v.protein_change,
            f"{v.vaf:.0%}",
            len(v.regulatory_consequences),
            len(v.motif_features),
            len(v.nearby_regulatory_elements),
            v.chromatin_state or "N/A",
        )
    log.info("")
    log.info("Results: %s", JSON_OUTPUT)
    log.info("Report:  %s", MD_OUTPUT)
    if method == "vep_fallback":
        log.info("Inputs:  %s", INPUTS_DIR)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sys.exit(1)
    except Exception as exc:
        log.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
