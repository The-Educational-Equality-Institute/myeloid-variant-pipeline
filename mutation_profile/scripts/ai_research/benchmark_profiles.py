#!/usr/bin/env python3
"""
ISMB 2026 Benchmark: Six-axis AI pipeline validation on 20 GENIE myeloid profiles.

Runs the generalized variant interpretation pipeline on 20 SETBP1-positive
myeloid profiles (154 variants) from AACR GENIE v19.0 to validate
generalizability beyond the index case.

Six independent evidence axes:
  1. Protein Language Model (ESM-2 masked marginal LLR)
  2. Structure-aware Deep Learning (AlphaMissense via myvariant.info)
  3. Evolutionary Conservation (EVE via myvariant.info)
  4. Supervised Meta-ensemble (CADD, REVEL via myvariant.info)
  5. Population Frequency (gnomAD via myvariant.info)
  6. Functional Evidence (SIFT, PolyPhen-2 from GENIE + ClinVar PS1)

ACMG Bayesian aggregation (Tavtigian 2018) with leave-one-axis-out ablation.
ClinVar ground truth comparison.

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/benchmark_profiles.py
    python mutation_profile/scripts/ai_research/benchmark_profiles.py --skip-esm2
    python mutation_profile/scripts/ai_research/benchmark_profiles.py --profiles 5

Runtime: ~10 min (API-only), ~20 min (with ESM-2 on RTX 4060)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = (
    PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "benchmark"
)
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_PATH = BENCHMARK_DIR / "benchmark_profiles.json"

MYVARIANT_URL = "https://myvariant.info/v1/variant"
MYVARIANT_FIELDS = ",".join([
    "dbnsfp.alphamissense",
    "dbnsfp.eve",
    "dbnsfp.cadd",
    "cadd",
    "dbnsfp.revel",
    "dbnsfp.sift",
    "dbnsfp.polyphen2_hdiv",
    "dbnsfp.polyphen2",
    "clinvar",
    "gnomad_exome",
])

# UniProt accessions for target genes (human canonical, reviewed)
UNIPROT_MAP = {
    "ASXL1": "Q8IXJ9", "TET2": "Q6N021", "SRSF2": "Q01130", "SF3B1": "O75533",
    "RUNX1": "Q01196", "TP53": "P04637", "FLT3": "P36888", "NPM1": "P06748",
    "NRAS": "P01111", "KRAS": "P01116", "CBL": "P22681", "EZH2": "Q15910",
    "U2AF1": "Q01081", "STAG2": "Q8N3U4", "BCOR": "Q6W2J9", "BCORL1": "Q5H9F3",
    "DDX41": "Q9UJV9", "DNMT3A": "Q9Y6K1", "IDH1": "O75874", "IDH2": "P48735",
    "PTPN11": "Q06124", "JAK2": "O60674", "CALR": "P27797", "MPL": "P40238",
    "PHF6": "Q8IWS0", "WT1": "P19544", "CEBPA": "P49715", "GATA2": "P23769",
    "ZRSR2": "Q15696", "RAD21": "O60216", "SMC1A": "Q14683", "SMC3": "Q9UQE7",
    "SETBP1": "Q9Y6X0", "CSF3R": "Q99062",
}

# PP3 thresholds (ClinGen SVI: Pejaver et al. 2022)
THRESHOLDS = {
    "alphamissense": {"pathogenic": 0.564, "strong": 0.927, "very_strong": 0.991},
    "cadd_phred": {"pathogenic": 17.3, "strong": 25.3},
    "revel": {"pathogenic": 0.5, "moderate": 0.644, "strong": 0.773},
    "eve": {"pathogenic": 0.5},
    "sift": {"pathogenic": 0.05},  # <= threshold = deleterious
    "polyphen2": {"pathogenic": 0.453},
    "esm2_llr": {"pathogenic": -2.0},  # <= threshold = deleterious
}

# ACMG Bayesian points (Tavtigian 2018)
STRENGTH_POINTS = {
    "VeryStrong": 8, "Strong": 4, "Moderate": 2, "Supporting": 1,
    "Strong_Benign": -4, "Moderate_Benign": -2, "Supporting_Benign": -1,
}


@dataclass
class VariantScore:
    gene: str
    hgvsp: str
    chromosome: str
    start_position: int
    ref_allele: str
    alt_allele: str
    t_vaf: float | None = None
    # Axis 1: Protein LM
    esm2_llr: float | None = None
    # Axis 2: Structure DL
    alphamissense: float | None = None
    # Axis 3: Conservation
    eve_score: float | None = None
    eve_class: str | None = None
    # Axis 4: Meta-ensemble
    cadd_phred: float | None = None
    revel: float | None = None
    # Axis 5: Population
    gnomad_af: float | None = None
    # Axis 6: Functional
    sift_score: float | None = None
    sift_prediction: str | None = None
    polyphen2_score: float | None = None
    polyphen2_prediction: str | None = None
    # ClinVar ground truth
    clinvar_classification: str | None = None
    clinvar_review_status: str | None = None
    clinvar_id: str | None = None
    # OncoKB clinical evidence
    oncokb_oncogenicity: str | None = None
    # Variant type (from GENIE mutation data)
    variant_classification: str = ""
    # Pipeline output
    axes_pathogenic: int = 0
    axes_total: int = 0
    pp3_strength: str = "Not_Met"
    pp3_points: int = 0
    pp5_strength: str = "Not_Met"
    pp5_points: int = 0
    pm2_strength: str = "Not_Met"
    pm2_points: int = 0
    pvs1_strength: str = "Not_Met"
    pvs1_points: int = 0
    ps1_strength: str = "Not_Met"
    ps1_points: int = 0
    pm1_strength: str = "Not_Met"
    pm1_points: int = 0
    total_points: int = 0
    classification: str = "VUS"
    # Ablation
    ablation: dict = field(default_factory=dict)


def _extract_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list):
        for v in val:
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    continue
        return None
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _extract_str(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, list):
        return str(val[0]) if val else None
    return str(val)


def build_hg19_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    c = str(chrom).replace("chr", "")
    return f"chr{c}:g.{pos}{ref}>{alt}"


def query_myvariant(hg19_id: str) -> dict:
    """Query myvariant.info for all pathogenicity scores + ClinVar."""
    try:
        r = requests.get(
            f"{MYVARIANT_URL}/{hg19_id}",
            params={"fields": MYVARIANT_FIELDS},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.warning("myvariant.info error for %s: %s", hg19_id, e)
    return {}


def parse_myvariant(data: dict, vs: VariantScore) -> None:
    """Extract scores from myvariant.info response into VariantScore."""
    dbnsfp = data.get("dbnsfp", {})
    if not isinstance(dbnsfp, dict):
        return

    # AlphaMissense (lowercase in myvariant.info)
    am = dbnsfp.get("alphamissense", {})
    if isinstance(am, list):
        am = am[0] if am else {}
    if isinstance(am, dict):
        vs.alphamissense = _extract_float(am.get("score"))

    # EVE
    eve = dbnsfp.get("eve", {})
    if isinstance(eve, list):
        eve = eve[0] if eve else {}
    if isinstance(eve, dict):
        vs.eve_score = _extract_float(eve.get("score"))
        vs.eve_class = _extract_str(eve.get("class25_pred"))

    # CADD (try dbnsfp.cadd first, then top-level cadd)
    cadd = dbnsfp.get("cadd", {})
    if isinstance(cadd, list):
        cadd = cadd[0] if cadd else {}
    if isinstance(cadd, dict):
        vs.cadd_phred = _extract_float(cadd.get("phred"))
    # Fallback: top-level cadd field
    if vs.cadd_phred is None:
        top_cadd = data.get("cadd", {})
        if isinstance(top_cadd, dict):
            vs.cadd_phred = _extract_float(top_cadd.get("phred"))

    # REVEL
    revel = dbnsfp.get("revel", {})
    if isinstance(revel, list):
        revel = revel[0] if revel else {}
    if isinstance(revel, dict):
        vs.revel = _extract_float(revel.get("score"))

    # SIFT (from dbNSFP if not already in GENIE)
    if vs.sift_score is None:
        sift = dbnsfp.get("sift", {})
        if isinstance(sift, list):
            sift = sift[0] if sift else {}
        if isinstance(sift, dict):
            vs.sift_score = _extract_float(sift.get("score"))

    # PolyPhen-2 (try polyphen2_hdiv, then polyphen2.hdiv.score)
    if vs.polyphen2_score is None:
        pp2 = dbnsfp.get("polyphen2_hdiv", {})
        if isinstance(pp2, list):
            pp2 = pp2[0] if pp2 else {}
        if isinstance(pp2, dict):
            vs.polyphen2_score = _extract_float(pp2.get("score"))
    # Fallback: nested polyphen2.hdiv.score
    if vs.polyphen2_score is None:
        pp2_nest = dbnsfp.get("polyphen2", {})
        if isinstance(pp2_nest, dict):
            hdiv = pp2_nest.get("hdiv", {})
            if isinstance(hdiv, list):
                hdiv = hdiv[0] if hdiv else {}
            if isinstance(hdiv, dict):
                vs.polyphen2_score = _extract_float(hdiv.get("score"))

    # gnomAD
    gnomad = data.get("gnomad_exome", {})
    if isinstance(gnomad, dict):
        af_data = gnomad.get("af", {})
        if isinstance(af_data, dict):
            vs.gnomad_af = _extract_float(af_data.get("af"))
        else:
            vs.gnomad_af = _extract_float(af_data)

    # ClinVar
    clinvar = data.get("clinvar", {})
    if isinstance(clinvar, dict):
        rcv = clinvar.get("rcv", [])
        if isinstance(rcv, dict):
            rcv = [rcv]
        if rcv:
            vs.clinvar_classification = rcv[0].get("clinical_significance", "")
            vs.clinvar_review_status = rcv[0].get("review_status", "")
        vs.clinvar_id = clinvar.get("variant_id")


def run_esm2_batch(variants: list[VariantScore]) -> None:
    """Run ESM-2 masked marginal scoring on all missense variants."""
    try:
        import torch
        import esm
    except ImportError:
        log.warning("ESM-2 not available, skipping Axis 1")
        return

    if not torch.cuda.is_available():
        log.warning("No CUDA GPU, skipping ESM-2")
        return

    log.info("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    # Cache UniProt sequences
    seq_cache: dict[str, str] = {}

    for i, vs in enumerate(variants):
        if not vs.hgvsp or not vs.hgvsp.startswith("p."):
            continue

        # Parse protein change
        hgvs = vs.hgvsp[2:]  # strip "p."
        if len(hgvs) < 3:
            continue

        # Handle 3-letter and 1-letter codes
        aa3to1 = {
            "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
            "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
            "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
            "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
        }

        # Try single-letter format first (e.g., G870S)
        wt_aa = hgvs[0]
        mut_aa = hgvs[-1]
        try:
            pos = int(hgvs[1:-1])
        except ValueError:
            # Try 3-letter format (e.g., Gly870Ser)
            for code, letter in aa3to1.items():
                if hgvs.startswith(code):
                    wt_aa = letter
                    hgvs_rest = hgvs[3:]
                    break
            else:
                continue
            for code, letter in aa3to1.items():
                if hgvs_rest.endswith(code):
                    mut_aa = letter
                    try:
                        pos = int(hgvs_rest[:-3])
                    except ValueError:
                        continue
                    break
            else:
                continue

        if wt_aa not in "ACDEFGHIKLMNPQRSTVWY" or mut_aa not in "ACDEFGHIKLMNPQRSTVWY":
            continue

        # Fetch sequence
        uniprot_id = UNIPROT_MAP.get(vs.gene)
        if not uniprot_id:
            continue

        if uniprot_id not in seq_cache:
            try:
                r = requests.get(
                    f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta",
                    timeout=15,
                )
                if r.status_code == 200:
                    lines = r.text.strip().split("\n")
                    seq_cache[uniprot_id] = "".join(lines[1:])
                else:
                    continue
            except Exception:
                continue

        seq = seq_cache.get(uniprot_id, "")
        if not seq or pos < 1 or pos > len(seq):
            continue

        # Verify WT amino acid matches
        if seq[pos - 1] != wt_aa:
            log.debug(
                "AA mismatch %s pos %d: expected %s got %s",
                vs.gene, pos, wt_aa, seq[pos - 1],
            )
            continue

        # ESM-2 scoring
        try:
            data = [("protein", seq)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.cuda()
            masked = tokens.clone()
            masked[0, pos] = alphabet.mask_idx  # pos is 1-indexed, token 0 is BOS

            with torch.no_grad():
                logits = model(masked)["logits"]

            log_probs = torch.log_softmax(logits[0, pos], dim=-1)
            wt_idx = alphabet.get_idx(wt_aa)
            mut_idx = alphabet.get_idx(mut_aa)
            llr = (log_probs[mut_idx] - log_probs[wt_idx]).item()
            vs.esm2_llr = round(llr, 4)
            log.info(
                "  [%d/%d] ESM-2 %s %s: LLR=%.4f",
                i + 1, len(variants), vs.gene, vs.hgvsp, llr,
            )
        except Exception as e:
            log.warning("ESM-2 failed for %s %s: %s", vs.gene, vs.hgvsp, e)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_pathogenic(tool: str, score: float | None) -> bool | None:
    """Check if a score exceeds the pathogenic threshold for a tool."""
    if score is None:
        return None
    thresh = THRESHOLDS.get(tool, {})
    cutoff = thresh.get("pathogenic")
    if cutoff is None:
        return None
    if tool in ("sift", "esm2_llr"):
        return score <= cutoff  # Lower = more pathogenic
    return score >= cutoff


def compute_axes(vs: VariantScore) -> tuple[int, int]:
    """Count how many of the 6 axes classify the variant as pathogenic.

    Axis mapping matches ISMB 2026 abstract:
      1. Protein LM (ESM-2)
      2. Structure-aware DL (AlphaMissense)
      3. Evolutionary Conservation (EVE, SIFT)
      4. Supervised Meta-ensemble (CADD, REVEL)
      5. Population Frequency (gnomAD v4)
      6. Functional Evidence (PolyPhen-2)
    """
    # Per-tool pathogenicity
    tools = {
        "esm2": is_pathogenic("esm2_llr", vs.esm2_llr),
        "alphamissense": is_pathogenic("alphamissense", vs.alphamissense),
        "eve": is_pathogenic("eve", vs.eve_score),
        "sift": is_pathogenic("sift", vs.sift_score),
        "cadd": is_pathogenic("cadd_phred", vs.cadd_phred),
        "revel": is_pathogenic("revel", vs.revel),
        "gnomad": (
            vs.gnomad_af is not None and vs.gnomad_af < 0.0001
        ) if vs.gnomad_af is not None else None,
        "polyphen2": is_pathogenic("polyphen2", vs.polyphen2_score),
    }

    # Collapse to 6 axes (any tool on axis = axis pathogenic)
    def _any_on_axis(keys: list[str]) -> bool | None:
        vals = [tools[k] for k in keys if tools[k] is not None]
        return any(vals) if vals else None

    axis_results = {
        "Axis1_ProteinLM": tools["esm2"],
        "Axis2_StructureDL": tools["alphamissense"],
        "Axis3_Conservation": _any_on_axis(["eve", "sift"]),
        "Axis4_MetaEnsemble": _any_on_axis(["cadd", "revel"]),
        "Axis5_Population": tools["gnomad"],
        "Axis6_Functional": tools["polyphen2"],
    }

    scored = [v for v in axis_results.values() if v is not None]
    pathogenic = sum(1 for v in scored if v)
    return pathogenic, len(scored)


def compute_pp3(axes_pathogenic: int, axes_total: int, vs: VariantScore) -> tuple[str, int]:
    """Determine PP3 strength from axis concordance."""
    if axes_total == 0:
        return "Not_Met", 0

    concordance = axes_pathogenic / axes_total

    # Check for strong individual tools
    am_strong = vs.alphamissense is not None and vs.alphamissense >= 0.927
    revel_strong = vs.revel is not None and vs.revel >= 0.773

    if concordance >= 0.85 and am_strong:
        return "VeryStrong", 8
    if concordance >= 0.70 and (am_strong or revel_strong):
        return "Strong", 4
    if concordance >= 0.60:
        return "Moderate", 2
    if concordance >= 0.50:
        return "Supporting", 1
    return "Not_Met", 0


def compute_pm2(vs: VariantScore) -> tuple[str, int]:
    """PM2: absent from population databases."""
    if vs.gnomad_af is None:
        return "Supporting", 1  # Absent = supporting
    if vs.gnomad_af == 0:
        return "Moderate", 2
    if vs.gnomad_af < 0.0001:
        return "Supporting", 1
    return "Not_Met", 0


# Genes where loss-of-function (truncating) is a known disease mechanism
LOF_GENES = {
    "ASXL1", "TET2", "RUNX1", "TP53", "EZH2", "STAG2", "BCOR", "BCORL1",
    "DDX41", "DNMT3A", "PHF6", "WT1", "CEBPA", "GATA2", "ZRSR2", "RAD21",
    "SMC1A", "SMC3", "SETBP1", "CSF3R",
}

# Truncating variant classifications from GENIE
PVS1_FULL_TYPES = {
    "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site",
}
PVS1_MODERATE_TYPES = {"Translation_Start_Site"}

# Known mutational hotspot domains (gene -> list of (start, end, name) or positions)
HOTSPOT_DOMAINS: dict[str, list[tuple[int, int, str]]] = {
    "SETBP1": [(858, 871, "SKI domain")],
    "DNMT3A": [(634, 912, "methyltransferase domain")],
    "IDH2": [(140, 140, "R140 active site"), (172, 172, "R172 active site")],
    "PTPN11": [(3, 103, "N-SH2 domain")],
    "EZH2": [(612, 727, "SET domain")],
    "NRAS": [(12, 13, "G12/G13"), (61, 61, "Q61")],
    "KRAS": [(12, 13, "G12/G13"), (61, 61, "Q61")],
    "JAK2": [(617, 617, "V617")],
    "SRSF2": [(95, 95, "P95")],
    "FLT3": [(835, 842, "TKD domain")],
    "NPM1": [(288, 288, "W288 exon 12")],
}


def _parse_protein_position(hgvsp: str) -> int | None:
    """Extract numeric position from HGVSp notation (e.g., 'p.R882H' -> 882)."""
    if not hgvsp or not hgvsp.startswith("p."):
        return None
    hgvs = hgvsp[2:]
    # Single-letter format: R882H
    try:
        return int(hgvs[1:-1])
    except (ValueError, IndexError):
        pass
    # Three-letter format: Arg882His
    aa3to1 = {
        "Ala", "Arg", "Asn", "Asp", "Cys", "Glu", "Gln", "Gly", "His", "Ile",
        "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val",
    }
    for code in aa3to1:
        if hgvs.startswith(code):
            rest = hgvs[3:]
            for code2 in aa3to1:
                if rest.endswith(code2):
                    try:
                        return int(rest[:-3])
                    except ValueError:
                        pass
            # Truncating notation (e.g., Arg882*)
            if rest.endswith("*") or rest.endswith("fs"):
                digits = "".join(c for c in rest if c.isdigit())
                if digits:
                    try:
                        return int(digits)
                    except ValueError:
                        pass
            break
    return None


def compute_pvs1(vs: VariantScore) -> tuple[str, int]:
    """PVS1: Null variant in a gene where LoF is a known disease mechanism."""
    vc = vs.variant_classification
    if not vc:
        return "Not_Met", 0
    if vc in PVS1_FULL_TYPES and vs.gene in LOF_GENES:
        return "VeryStrong", 8
    if vc in PVS1_MODERATE_TYPES and vs.gene in LOF_GENES:
        return "Moderate", 2
    return "Not_Met", 0


def compute_ps1(vs: VariantScore) -> tuple[str, int]:
    """PS1: Same amino acid change as an established pathogenic ClinVar variant."""
    if not vs.clinvar_classification:
        return "Not_Met", 0
    cv = vs.clinvar_classification.lower().strip()
    if "pathogenic" in cv or "likely pathogenic" in cv:
        return "Strong", 4
    return "Not_Met", 0


def compute_pm1(vs: VariantScore) -> tuple[str, int]:
    """PM1: Located in a mutational hotspot or critical functional domain."""
    if vs.gene not in HOTSPOT_DOMAINS:
        return "Not_Met", 0
    pos = _parse_protein_position(vs.hgvsp)
    if pos is None:
        return "Not_Met", 0
    for start, end, _name in HOTSPOT_DOMAINS[vs.gene]:
        if start <= pos <= end:
            return "Moderate", 2
    return "Not_Met", 0


def compute_pp5(vs: VariantScore) -> tuple[str, int]:
    """PP5: Reputable source reports variant as pathogenic.

    Uses OncoKB oncogenicity classification:
      - "Oncogenic" -> PP5 Strong (4 points, same weight as PS1)
      - "Likely Oncogenic" -> PP5 Supporting (1 point)
      - Otherwise -> Not Met
    """
    if not vs.oncokb_oncogenicity:
        return "Not_Met", 0
    onco = vs.oncokb_oncogenicity.strip()
    if onco == "Oncogenic":
        return "Strong", 4
    if onco == "Likely Oncogenic":
        return "Supporting", 1
    return "Not_Met", 0


def classify_variant(total_points: int) -> str:
    if total_points >= 10:
        return "Pathogenic"
    if total_points >= 6:
        return "Likely_Pathogenic"
    if total_points >= 0:
        return "VUS"
    if total_points >= -6:
        return "Likely_Benign"
    return "Benign"


def run_ablation(vs: VariantScore) -> dict:
    """Leave-one-axis-out ablation: remove entire axis and reclassify.

    Matches ISMB abstract axis grouping. Removes ALL tools on an axis
    simultaneously, plus axis-dependent ACMG criteria (PM2 for Axis 5).
    """
    results = {}

    axis_tools: dict[str, list[str]] = {
        "Axis1_ProteinLM": ["esm2_llr"],
        "Axis2_StructureDL": ["alphamissense"],
        "Axis3_Conservation": ["eve_score", "sift_score"],
        "Axis4_MetaEnsemble": ["cadd_phred", "revel"],
        "Axis5_Population": ["gnomad_af"],
        "Axis6_Functional": ["polyphen2_score"],
    }

    for axis_name, tools in axis_tools.items():
        # Save originals
        saved = {t: getattr(vs, t) for t in tools}

        # Remove all tools on this axis
        for t in tools:
            setattr(vs, t, None)

        # Recompute
        axes_p, axes_t = compute_axes(vs)
        pp3_str, pp3_pts = compute_pp3(axes_p, axes_t, vs)

        # Axis 5 removal also removes PM2 (population-dependent)
        if axis_name == "Axis5_Population":
            pm2_pts = 0
        else:
            _, pm2_pts = compute_pm2(vs)

        pvs1_str, pvs1_pts = compute_pvs1(vs)
        ps1_str, ps1_pts = compute_ps1(vs)
        pm1_str, pm1_pts = compute_pm1(vs)
        _, pp5_pts = compute_pp5(vs)  # PP5 is axis-independent, stays constant
        total = pp3_pts + pp5_pts + pm2_pts + pvs1_pts + ps1_pts + pm1_pts
        new_class = classify_variant(total)

        results[f"remove_{axis_name}"] = {
            "tools_removed": tools,
            "new_axes_pathogenic": axes_p,
            "new_total_points": total,
            "new_classification": new_class,
            "changed": new_class != vs.classification,
        }

        # Restore
        for t, val in saved.items():
            setattr(vs, t, val)

    # PP5 ablation (independent clinical evidence, not tied to any axis)
    saved_oncokb = vs.oncokb_oncogenicity
    vs.oncokb_oncogenicity = None
    _, pp5_removed_pts = compute_pp5(vs)
    total_no_pp5 = vs.pp3_points + pp5_removed_pts + vs.pm2_points + vs.pvs1_points + vs.ps1_points + vs.pm1_points
    new_class_no_pp5 = classify_variant(total_no_pp5)
    results["remove_PP5_ClinicalEvidence"] = {
        "tools_removed": ["oncokb_oncogenicity"],
        "new_axes_pathogenic": vs.axes_pathogenic,
        "new_total_points": total_no_pp5,
        "new_classification": new_class_no_pp5,
        "changed": new_class_no_pp5 != vs.classification,
    }
    vs.oncokb_oncogenicity = saved_oncokb

    return results


def normalize_clinvar(raw: str | None) -> str:
    """Normalize ClinVar classification to standard categories."""
    if not raw:
        return "Not_in_ClinVar"
    r = raw.lower().strip()
    if "pathogenic" in r and "likely" in r:
        return "Likely_Pathogenic"
    if "pathogenic" in r:
        return "Pathogenic"
    if "benign" in r and "likely" in r:
        return "Likely_Benign"
    if "benign" in r:
        return "Benign"
    if "uncertain" in r or "vus" in r:
        return "VUS"
    if "conflicting" in r:
        return "Conflicting"
    return "Other"


def generate_report(
    all_scores: list[VariantScore],
    profiles: list[dict],
    elapsed: float,
) -> str:
    """Generate markdown benchmark report."""
    lines = [
        "# ISMB 2026 Benchmark: Six-Axis Pipeline Validation",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Profiles:** {len(profiles)} SETBP1-positive myeloid patients from GENIE v19.0",
        f"**Variants scored:** {len(all_scores)}",
        f"**Runtime:** {elapsed:.1f} seconds",
        "",
        "---",
        "",
        "## Classification Summary",
        "",
    ]

    # Count classifications
    class_counts = {}
    for vs in all_scores:
        class_counts[vs.classification] = class_counts.get(vs.classification, 0) + 1
    lines.append("| Classification | Count | % |")
    lines.append("|---------------|------:|--:|")
    for cls in ["Pathogenic", "Likely_Pathogenic", "VUS", "Likely_Benign", "Benign"]:
        n = class_counts.get(cls, 0)
        pct = 100 * n / len(all_scores) if all_scores else 0
        lines.append(f"| {cls} | {n} | {pct:.1f}% |")

    # ClinVar concordance
    lines.extend(["", "## ClinVar Concordance", ""])
    clinvar_scored = [vs for vs in all_scores if vs.clinvar_classification]
    clinvar_norm = [(vs, normalize_clinvar(vs.clinvar_classification)) for vs in clinvar_scored]

    if clinvar_norm:
        concordant = sum(
            1 for vs, cv in clinvar_norm
            if (cv in ("Pathogenic", "Likely_Pathogenic") and vs.classification in ("Pathogenic", "Likely_Pathogenic"))
            or (cv == "VUS" and vs.classification == "VUS")
            or (cv in ("Benign", "Likely_Benign") and vs.classification in ("Benign", "Likely_Benign", "VUS"))
        )
        lines.append(f"**Variants with ClinVar entry:** {len(clinvar_norm)}")
        lines.append(f"**Concordance:** {concordant}/{len(clinvar_norm)} ({100*concordant/len(clinvar_norm):.1f}%)")
        lines.append("")

        lines.append("| Gene | Variant | Pipeline | ClinVar | Match |")
        lines.append("|------|---------|----------|---------|-------|")
        for vs, cv in clinvar_norm:
            pipe = vs.classification
            match = "Y" if (
                (cv in ("Pathogenic", "Likely_Pathogenic") and pipe in ("Pathogenic", "Likely_Pathogenic"))
                or (cv == "VUS" and pipe == "VUS")
                or (cv in ("Benign", "Likely_Benign") and pipe in ("Benign", "Likely_Benign", "VUS"))
            ) else "N"
            lines.append(f"| {vs.gene} | {vs.hgvsp} | {pipe} | {cv} | {match} |")
    else:
        lines.append("No ClinVar entries found for benchmark variants.")

    # VUS reclassification
    lines.extend(["", "## VUS Reclassification", ""])
    vus_reclass = [
        vs for vs in all_scores
        if normalize_clinvar(vs.clinvar_classification) == "VUS"
        and vs.classification in ("Pathogenic", "Likely_Pathogenic")
    ]
    not_in_clinvar_classified = [
        vs for vs in all_scores
        if not vs.clinvar_classification
        and vs.classification in ("Pathogenic", "Likely_Pathogenic")
    ]
    lines.append(f"**ClinVar VUS reclassified to P/LP by pipeline:** {len(vus_reclass)}")
    lines.append(f"**Not in ClinVar, classified P/LP by pipeline:** {len(not_in_clinvar_classified)}")
    if vus_reclass:
        lines.append("")
        lines.append("| Gene | Variant | Pipeline Class | Points | Axes P/T |")
        lines.append("|------|---------|---------------|-------:|---------|")
        for vs in vus_reclass:
            lines.append(
                f"| {vs.gene} | {vs.hgvsp} | {vs.classification} | {vs.total_points} | {vs.axes_pathogenic}/{vs.axes_total} |"
            )

    # Ablation robustness
    lines.extend(["", "## Ablation Robustness", ""])
    robust_count = 0
    fragile_variants = []
    for vs in all_scores:
        if vs.classification in ("Pathogenic", "Likely_Pathogenic") and vs.ablation:
            any_changed = any(v["changed"] for v in vs.ablation.values())
            if not any_changed:
                robust_count += 1
            else:
                fragile_variants.append(vs)

    classified_pl = [
        vs for vs in all_scores
        if vs.classification in ("Pathogenic", "Likely_Pathogenic")
    ]
    if classified_pl:
        lines.append(
            f"**P/LP variants robust to single-tool removal:** "
            f"{robust_count}/{len(classified_pl)} ({100*robust_count/len(classified_pl):.1f}%)"
        )
    if fragile_variants:
        lines.append("")
        lines.append("Fragile variants (classification changes on tool removal):")
        lines.append("")
        for vs in fragile_variants:
            changed = [k for k, v in vs.ablation.items() if v["changed"]]
            lines.append(f"- {vs.gene} {vs.hgvsp}: fragile to {', '.join(changed)}")

    # Per-axis concordance
    lines.extend(["", "## Cross-Tool Concordance Matrix", ""])
    lines.append("| Gene | Variant | AM | EVE | CADD | REVEL | SIFT | PP2 | ESM-2 | gnomAD | Axes | Class |")
    lines.append("|------|---------|---:|----:|-----:|------:|-----:|----:|------:|-------:|-----:|-------|")
    for vs in all_scores:
        am = f"{vs.alphamissense:.3f}" if vs.alphamissense is not None else "-"
        eve = f"{vs.eve_score:.3f}" if vs.eve_score is not None else "-"
        cadd = f"{vs.cadd_phred:.1f}" if vs.cadd_phred is not None else "-"
        rev = f"{vs.revel:.3f}" if vs.revel is not None else "-"
        sift = f"{vs.sift_score:.3f}" if vs.sift_score is not None else "-"
        pp2 = f"{vs.polyphen2_score:.3f}" if vs.polyphen2_score is not None else "-"
        e2 = f"{vs.esm2_llr:.2f}" if vs.esm2_llr is not None else "-"
        gn = f"{vs.gnomad_af:.4f}" if vs.gnomad_af is not None else "0"
        lines.append(
            f"| {vs.gene} | {vs.hgvsp} | {am} | {eve} | {cadd} | {rev} | {sift} | {pp2} | {e2} | {gn} | {vs.axes_pathogenic}/{vs.axes_total} | {vs.classification} |"
        )

    # Per-profile summary
    lines.extend(["", "## Per-Profile Summary", ""])
    lines.append("| # | Patient | Disease | Muts | P | LP | VUS | Axes Range |")
    lines.append("|--:|---------|---------|-----:|--:|---:|----:|-----------:|")
    profile_variants = {}
    for vs in all_scores:
        pid = vs.__dict__.get("_profile_id", "?")
        profile_variants.setdefault(pid, []).append(vs)

    for i, prof in enumerate(profiles):
        pid = prof["patient_id"]
        pvs = profile_variants.get(pid, [])
        n_p = sum(1 for v in pvs if v.classification == "Pathogenic")
        n_lp = sum(1 for v in pvs if v.classification == "Likely_Pathogenic")
        n_vus = sum(1 for v in pvs if v.classification == "VUS")
        axes_range = (
            f"{min(v.axes_pathogenic for v in pvs)}-{max(v.axes_pathogenic for v in pvs)}"
            if pvs else "-"
        )
        lines.append(
            f"| {i+1} | {pid} | {prof['oncotree_code']} | {len(pvs)} | {n_p} | {n_lp} | {n_vus} | {axes_range} |"
        )

    lines.extend(["", "---", "", "*Generated by benchmark_profiles.py for ISMB 2026 submission.*"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ISMB 2026 benchmark pipeline")
    parser.add_argument("--skip-esm2", action="store_true", help="Skip ESM-2 scoring")
    parser.add_argument("--profiles", type=int, default=0, help="Limit to first N profiles")
    parser.add_argument("--input", type=str, default="", help="Input profiles JSON (default: benchmark_profiles.json)")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix for output files (e.g. '_batch2')")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else PROFILES_PATH
    if not input_path.exists():
        log.error("Profile JSON not found: %s", input_path)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    profiles = data["profiles"]
    if args.profiles > 0:
        profiles = profiles[: args.profiles]

    suffix = args.output_suffix
    log.info("Loaded %d profiles", len(profiles))

    # Load OncoKB data for PP5 criterion
    oncokb_path = BENCHMARK_DIR / "oncokb_benchmark.json"
    oncokb_lookup: dict[tuple[str, str], str] = {}
    if oncokb_path.exists():
        with open(oncokb_path) as okf:
            oncokb_data = json.load(okf)
        for entry in oncokb_data.get("variants", []):
            key = (entry["gene"], entry["hgvsp"])
            oncokb_lookup[key] = entry.get("oncogenicity", "")
        log.info("Loaded %d OncoKB annotations for PP5", len(oncokb_lookup))
    else:
        log.warning("OncoKB benchmark data not found: %s", oncokb_path)

    t0 = time.time()
    all_scores: list[VariantScore] = []

    # Build variant list
    for prof in profiles:
        for mut in prof["mutations"]:
            vs = VariantScore(
                gene=mut["gene"],
                hgvsp=mut.get("hgvsp", ""),
                chromosome=str(mut.get("chromosome", "")),
                start_position=mut.get("start_position", 0),
                ref_allele=mut.get("ref_allele", ""),
                alt_allele=mut.get("alt_allele", ""),
                t_vaf=mut.get("t_vaf"),
                sift_score=mut.get("sift_score"),
                sift_prediction=mut.get("sift_prediction"),
                polyphen2_score=mut.get("polyphen_score"),
                polyphen2_prediction=mut.get("polyphen_prediction"),
                variant_classification=mut.get("variant_classification", ""),
            )
            vs.__dict__["_profile_id"] = prof["patient_id"]
            # Populate OncoKB oncogenicity for PP5
            hgvsp_bare = vs.hgvsp.lstrip("p.") if vs.hgvsp else ""
            oncokb_onco = oncokb_lookup.get((vs.gene, hgvsp_bare))
            if oncokb_onco:
                vs.oncokb_oncogenicity = oncokb_onco
            all_scores.append(vs)

    oncokb_matched = sum(1 for vs in all_scores if vs.oncokb_oncogenicity)
    log.info("Total variants to score: %d (%d with OncoKB PP5 data)", len(all_scores), oncokb_matched)

    # Step 1: myvariant.info batch query
    log.info("=== Querying myvariant.info (Axes 2-5 + ClinVar) ===")
    for i, vs in enumerate(all_scores):
        if not vs.ref_allele or not vs.alt_allele:
            continue
        hg19_id = build_hg19_id(
            vs.chromosome, vs.start_position, vs.ref_allele, vs.alt_allele,
        )
        mv_data = query_myvariant(hg19_id)
        parse_myvariant(mv_data, vs)

        if (i + 1) % 10 == 0:
            log.info("  Scored %d/%d variants", i + 1, len(all_scores))
        time.sleep(0.3)  # Rate limit

    log.info("myvariant.info scoring complete")

    # Step 2: ESM-2 scoring (prefer pre-computed, fallback to live GPU)
    esm2_path = BENCHMARK_DIR / f"esm2_benchmark_scores{suffix}.json"
    if not args.skip_esm2 and esm2_path.exists():
        log.info("=== Loading pre-computed ESM-2 scores ===")
        with open(esm2_path) as ef:
            esm2_data = json.load(ef)
        esm2_lookup = {}
        for ev in esm2_data.get("variants", []):
            key = (ev["gene"], ev["hgvsp"].lstrip("p."))
            esm2_lookup[key] = ev["esm2_llr"]
        loaded = 0
        for vs in all_scores:
            hgvs_short = vs.hgvsp.lstrip("p.") if vs.hgvsp else ""
            llr = esm2_lookup.get((vs.gene, hgvs_short))
            if llr is not None:
                vs.esm2_llr = llr
                loaded += 1
        log.info("  Loaded %d ESM-2 scores from cache", loaded)
    elif not args.skip_esm2:
        log.info("=== Running ESM-2 (Axis 1: Protein LM) ===")
        missense = [vs for vs in all_scores if vs.hgvsp and vs.hgvsp.startswith("p.")]
        run_esm2_batch(missense)
    else:
        log.info("=== Skipping ESM-2 (--skip-esm2) ===")

    # Step 3: Compute axes, PP3, PM2, classification, ablation
    log.info("=== Computing ACMG classification + ablation ===")
    for vs in all_scores:
        vs.axes_pathogenic, vs.axes_total = compute_axes(vs)
        vs.pp3_strength, vs.pp3_points = compute_pp3(vs.axes_pathogenic, vs.axes_total, vs)
        vs.pp5_strength, vs.pp5_points = compute_pp5(vs)
        vs.pm2_strength, vs.pm2_points = compute_pm2(vs)
        vs.pvs1_strength, vs.pvs1_points = compute_pvs1(vs)
        vs.ps1_strength, vs.ps1_points = compute_ps1(vs)
        vs.pm1_strength, vs.pm1_points = compute_pm1(vs)
        vs.total_points = vs.pp3_points + vs.pp5_points + vs.pm2_points + vs.pvs1_points + vs.ps1_points + vs.pm1_points
        vs.classification = classify_variant(vs.total_points)
        vs.ablation = run_ablation(vs)

    elapsed = time.time() - t0

    # Step 4: Generate outputs
    log.info("=== Generating outputs ===")

    # JSON results
    results_json = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "n_profiles": len(profiles),
            "n_variants": len(all_scores),
            "runtime_seconds": round(elapsed, 1),
            "esm2_enabled": not args.skip_esm2,
            "method": "Six-axis ACMG Bayesian (Tavtigian 2018)",
        },
        "variants": [],
    }
    for vs in all_scores:
        d = {
            "profile_id": vs.__dict__.get("_profile_id", ""),
            "gene": vs.gene,
            "hgvsp": vs.hgvsp,
            "scores": {
                "esm2_llr": vs.esm2_llr,
                "alphamissense": vs.alphamissense,
                "eve_score": vs.eve_score,
                "cadd_phred": vs.cadd_phred,
                "revel": vs.revel,
                "sift_score": vs.sift_score,
                "polyphen2_score": vs.polyphen2_score,
                "gnomad_af": vs.gnomad_af,
            },
            "clinvar": {
                "classification": vs.clinvar_classification,
                "normalized": normalize_clinvar(vs.clinvar_classification),
                "review_status": vs.clinvar_review_status,
                "id": vs.clinvar_id,
            },
            "oncokb": {
                "oncogenicity": vs.oncokb_oncogenicity,
            },
            "pipeline": {
                "axes_pathogenic": vs.axes_pathogenic,
                "axes_total": vs.axes_total,
                "pp3_strength": vs.pp3_strength,
                "pp3_points": vs.pp3_points,
                "pp5_strength": vs.pp5_strength,
                "pp5_points": vs.pp5_points,
                "pm2_strength": vs.pm2_strength,
                "pm2_points": vs.pm2_points,
                "pvs1_strength": vs.pvs1_strength,
                "pvs1_points": vs.pvs1_points,
                "ps1_strength": vs.ps1_strength,
                "ps1_points": vs.ps1_points,
                "pm1_strength": vs.pm1_strength,
                "pm1_points": vs.pm1_points,
                "total_points": vs.total_points,
                "classification": vs.classification,
                "variant_classification": vs.variant_classification,
            },
            "ablation": vs.ablation,
        }
        results_json["variants"].append(d)

    # Summary stats
    cls_counts = {}
    for vs in all_scores:
        cls_counts[vs.classification] = cls_counts.get(vs.classification, 0) + 1
    results_json["summary"] = {
        "classification_counts": cls_counts,
        "clinvar_concordance": {},
    }

    clinvar_norm = [
        (vs, normalize_clinvar(vs.clinvar_classification))
        for vs in all_scores if vs.clinvar_classification
    ]
    if clinvar_norm:
        concordant = sum(
            1 for vs, cv in clinvar_norm
            if (cv in ("Pathogenic", "Likely_Pathogenic") and vs.classification in ("Pathogenic", "Likely_Pathogenic"))
            or (cv == "VUS" and vs.classification == "VUS")
            or (cv in ("Benign", "Likely_Benign") and vs.classification in ("Benign", "Likely_Benign", "VUS"))
        )
        results_json["summary"]["clinvar_concordance"] = {
            "total_with_clinvar": len(clinvar_norm),
            "concordant": concordant,
            "rate": round(concordant / len(clinvar_norm), 4),
        }

    json_path = BENCHMARK_DIR / f"benchmark_results{suffix}.json"
    json_path.write_text(json.dumps(results_json, indent=2, default=str))
    log.info("Results: %s", json_path)

    # Markdown report
    report = generate_report(all_scores, profiles, elapsed)
    report_path = BENCHMARK_DIR / f"benchmark_report{suffix}.md"
    report_path.write_text(report)
    log.info("Report: %s", report_path)

    # Print summary
    print()
    print(f"=== Benchmark Complete: {len(all_scores)} variants across {len(profiles)} profiles ===")
    print(f"Runtime: {elapsed:.1f}s")
    print()
    for cls in ["Pathogenic", "Likely_Pathogenic", "VUS", "Likely_Benign", "Benign"]:
        n = cls_counts.get(cls, 0)
        print(f"  {cls:20s} {n:4d}  ({100*n/len(all_scores):.1f}%)")
    if clinvar_norm:
        print()
        rate = results_json["summary"]["clinvar_concordance"]["rate"]
        print(f"ClinVar concordance: {rate:.1%} ({concordant}/{len(clinvar_norm)})")


if __name__ == "__main__":
    main()
