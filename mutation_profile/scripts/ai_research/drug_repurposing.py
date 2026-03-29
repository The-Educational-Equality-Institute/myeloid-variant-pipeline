#!/usr/bin/env python3
"""
drug_repurposing.py -- Drug repurposing analysis for patient mutation profile.

Compiles targeted therapies, combination strategies, novel candidates, clinical trials,
and a ranked treatment strategy using database queries and Gemini AI analysis.

Patient mutation profile (MDS-AML):
    1. DNMT3A R882H -- epigenetic regulator (DNA methyltransferase)
    2. IDH2 R140Q   -- metabolic enzyme (isocitrate dehydrogenase)
    3. SETBP1 G870S -- oncogene (SET binding protein, stabilises SET/PP2A inhibition)
    4. PTPN11 E76Q  -- signalling (SHP2 phosphatase, RAS/MAPK pathway)
    5. Monosomy 7   -- cytogenetic (loss of chromosome 7)

Inputs:
    - .env (GEMINI_API_KEY)
    - ClinicalTrials.gov API (remote)
    - Google Gemini API (remote)

Outputs:
    - mutation_profile/results/ai_research/drug_repurposing/drug_targets.json
    - mutation_profile/results/ai_research/drug_repurposing/combination_strategies.md
    - mutation_profile/results/ai_research/drug_repurposing/clinical_trials.md
    - mutation_profile/results/ai_research/drug_repurposing/treatment_ranking.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/drug_repurposing.py

Runtime: ~2-5 minutes (Gemini API calls)
Dependencies: requests, python-dotenv
"""

import json
import os
import sys
import time
import re
import textwrap
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # mutation_profile/
RESULTS_DIR = PROJECT_ROOT / "results" / "ai_research" / "drug_repurposing"
ENV_PATH = PROJECT_ROOT.parent / ".env"

load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set in .env")
    sys.exit(1)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Patient mutation profile
# ---------------------------------------------------------------------------
MUTATION_PROFILE = {
    "patient_context": "MDS-AML with complex mutation profile, zero matches across ~14,600 myeloid patients in GENIE v19.0",
    "mutations": [
        {
            "gene": "DNMT3A",
            "variant": "R882H",
            "protein_change": "p.Arg882His",
            "function": "DNA methyltransferase 3A — dominant-negative loss of methyltransferase activity",
            "pathway": "Epigenetic regulation / DNA methylation",
            "frequency_aml": "~20-25% of AML",
            "esm2_score": -8.79,
            "esm2_interpretation": "PP3_Strong — high pathogenicity",
        },
        {
            "gene": "IDH2",
            "variant": "R140Q",
            "protein_change": "p.Arg140Gln",
            "function": "Isocitrate dehydrogenase 2 — neomorphic gain-of-function producing 2-hydroxyglutarate (2-HG)",
            "pathway": "Metabolism / TCA cycle / epigenetic dysregulation via 2-HG",
            "frequency_aml": "~8-12% of AML",
            "esm2_score": -1.20,
            "esm2_interpretation": "Benign/uncertain — but known oncogenic hotspot",
        },
        {
            "gene": "SETBP1",
            "variant": "G870S",
            "protein_change": "p.Gly870Ser",
            "function": "SET binding protein 1 — gain-of-function, stabilises SET protein, inhibits PP2A tumour suppressor",
            "pathway": "PP2A inhibition / proliferative signalling",
            "frequency_aml": "~2-5% of MDS/MPN, rare in de novo AML",
            "esm2_score": -10.10,
            "esm2_interpretation": "PP3_Strong — highest pathogenicity score in profile",
        },
        {
            "gene": "PTPN11",
            "variant": "E76Q",
            "protein_change": "p.Glu76Gln",
            "function": "SHP2 phosphatase — gain-of-function, constitutively active RAS/MAPK signalling",
            "pathway": "RAS/MAPK/ERK signalling",
            "frequency_aml": "~5-10% of AML",
            "esm2_score": -1.76,
            "esm2_interpretation": "PP3_Supporting",
            "diffdock_results": {
                "RMC-4550": -9.26,
                "SHP099": -7.83,
                "TNO155": -7.49,
            },
        },
    ],
    "cytogenetics": {
        "abnormality": "Monosomy 7 (-7)",
        "significance": "Adverse risk per ELN 2022, associated with poor prognosis, MDS-related",
    },
}


# ---------------------------------------------------------------------------
# 1. Known Targeted Therapies — compile from curated knowledge
# ---------------------------------------------------------------------------
def compile_drug_targets() -> dict:
    """Build a structured drug-target database for the patient's mutations."""
    print("\n[1/5] Compiling known targeted therapies...")

    drug_targets = {
        "generated": datetime.now().isoformat(),
        "patient_profile": "DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + monosomy 7",
        "targets": [
            {
                "gene": "IDH2",
                "variant": "R140Q",
                "drugs": [
                    {
                        "name": "Enasidenib (AG-221 / Idhifa)",
                        "mechanism": "Selective IDH2 inhibitor — blocks 2-HG production by mutant IDH2, restores normal alpha-ketoglutarate levels and myeloid differentiation",
                        "status": "FDA-approved (2017)",
                        "indication": "Relapsed/refractory AML with IDH2 mutation",
                        "evidence_level": "Approved",
                        "key_trials": [
                            "AG221-C-001 (NCT01915498): ORR 40.3%, CR 19.3%, median OS 9.3 months in R/R AML",
                            "AGILE (NCT03173248): enasidenib + azacitidine vs azacitidine alone in newly diagnosed AML",
                        ],
                        "response_rate": "ORR ~40%, CR ~20% monotherapy in R/R setting",
                        "notable": "2-HG reduction is a pharmacodynamic biomarker; differentiation syndrome risk (IDH-DS) ~10-14%",
                    },
                    {
                        "name": "Olutasidenib (Rezlidhia, FT-2102)",
                        "mechanism": "Selective IDH1 inhibitor — Note: targets IDH1, not IDH2. Listed for completeness as some protocols test IDH1/2 together.",
                        "status": "FDA-approved for IDH1 (2022)",
                        "indication": "R/R AML with IDH1 mutation — NOT directly applicable to IDH2 R140Q",
                        "evidence_level": "Not applicable (wrong target)",
                    },
                    {
                        "name": "Vorasidenib (AG-881)",
                        "mechanism": "Dual IDH1/IDH2 inhibitor — brain-penetrant, primarily studied in glioma",
                        "status": "FDA-approved for glioma (2024). Not yet in AML trials.",
                        "indication": "Low-grade glioma with IDH1/2 mutation",
                        "evidence_level": "Phase 1 (for AML context — off-label consideration)",
                        "notable": "Could theoretically target IDH2 R140Q but no AML efficacy data",
                    },
                ],
            },
            {
                "gene": "PTPN11 (SHP2)",
                "variant": "E76Q",
                "drugs": [
                    {
                        "name": "RMC-4550 (Revolution Medicines)",
                        "mechanism": "Allosteric SHP2 inhibitor — locks SHP2 in autoinhibited conformation, blocks RAS/MAPK signalling",
                        "status": "Phase 1/2",
                        "indication": "Advanced solid tumours and haematologic malignancies with SHP2/RAS pathway mutations",
                        "evidence_level": "Phase 1/2",
                        "diffdock_score": -9.26,
                        "key_trials": ["NCT03634982: Phase 1 dose-escalation in advanced cancers"],
                        "notable": "Best DiffDock binding score among tested SHP2 inhibitors for PTPN11 E76Q",
                    },
                    {
                        "name": "TNO155 (Novartis)",
                        "mechanism": "Allosteric SHP2 inhibitor",
                        "status": "Phase 1/2",
                        "indication": "Advanced solid tumours, often combined with other targeted agents",
                        "evidence_level": "Phase 1/2",
                        "diffdock_score": -7.49,
                        "key_trials": [
                            "NCT03114319: Phase 1 dose-escalation",
                            "NCT04330664: TNO155 + spartalizumab (anti-PD1)",
                        ],
                    },
                    {
                        "name": "JAB-3312 (Jacobio Pharmaceuticals)",
                        "mechanism": "Allosteric SHP2 inhibitor",
                        "status": "Phase 1/2",
                        "indication": "Advanced solid tumours with NF1/KRAS/BRAF mutations",
                        "evidence_level": "Phase 1/2",
                        "key_trials": ["NCT04045496: Phase 1/2 dose-escalation"],
                    },
                    {
                        "name": "RMC-4630 (Revolution Medicines)",
                        "mechanism": "Allosteric SHP2 inhibitor — clinical-grade compound from same programme as RMC-4550",
                        "status": "Phase 1/2",
                        "indication": "Advanced solid tumours",
                        "evidence_level": "Phase 1/2",
                        "key_trials": [
                            "NCT03634982: Phase 1 monotherapy and combinations",
                        ],
                    },
                    {
                        "name": "SHP099 (tool compound)",
                        "mechanism": "Allosteric SHP2 inhibitor — first-in-class tool compound, not clinical",
                        "status": "Preclinical",
                        "evidence_level": "Preclinical",
                        "diffdock_score": -7.83,
                        "notable": "Reference compound for SHP2 inhibitor class",
                    },
                ],
            },
            {
                "gene": "DNMT3A",
                "variant": "R882H",
                "drugs": [
                    {
                        "name": "Azacitidine (Vidaza)",
                        "mechanism": "Hypomethylating agent (HMA) — DNA methyltransferase inhibitor, induces hypomethylation and re-expression of silenced genes",
                        "status": "FDA-approved",
                        "indication": "MDS, AML (especially unfit for intensive chemo), CMML",
                        "evidence_level": "Approved",
                        "rationale_for_dnmt3a": "DNMT3A R882H causes hypomethylation through dominant-negative mechanism; azacitidine further depletes DNMT3A/DNMT3B/DNMT1, which may paradoxically help by inducing viral mimicry / immune activation rather than further hypomethylation. Clinical data shows mixed benefit — some studies suggest DNMT3A-mutated AML responds to HMAs, others show no clear advantage.",
                    },
                    {
                        "name": "Decitabine (Dacogen)",
                        "mechanism": "Hypomethylating agent — similar mechanism to azacitidine but pure deoxynucleoside analogue",
                        "status": "FDA-approved",
                        "indication": "MDS, AML",
                        "evidence_level": "Approved",
                    },
                    {
                        "name": "Decitabine/Cedazuridine (Inqovi)",
                        "mechanism": "Oral HMA — decitabine + CDA inhibitor for oral bioavailability",
                        "status": "FDA-approved (2020)",
                        "indication": "MDS, CMML",
                        "evidence_level": "Approved",
                        "notable": "Oral administration advantage for extended dosing schedules",
                    },
                ],
            },
            {
                "gene": "SETBP1",
                "variant": "G870S",
                "drugs": [
                    {
                        "name": "No direct SETBP1 inhibitors exist",
                        "mechanism": "N/A — SETBP1 is currently undruggable as a direct target",
                        "status": "No compounds in development",
                        "evidence_level": "No data",
                        "notable": "SETBP1 G870S stabilises SET protein, which inhibits PP2A. Therapeutic strategies target downstream effects.",
                    },
                    {
                        "name": "FTY720 (Fingolimod) — PP2A activator",
                        "mechanism": "PP2A activating drug (PAD) — disrupts SET-PP2A interaction, reactivates PP2A tumour suppressor",
                        "status": "FDA-approved for MS, preclinical for AML",
                        "indication": "Repurposing candidate — PP2A reactivation in SETBP1-mutant leukaemia",
                        "evidence_level": "Preclinical (for AML)",
                        "rationale": "SETBP1 G870S → SET stabilisation → PP2A inhibition. FTY720 directly reactivates PP2A by disrupting SET-PP2A complex.",
                        "key_references": [
                            "Neviani et al., J Clin Invest 2005 — FTY720 activates PP2A in CML blast crisis",
                            "Cristobal et al., Blood 2011 — SET overexpression in AML",
                        ],
                    },
                    {
                        "name": "OP449 (formerly SET antagonist peptide)",
                        "mechanism": "Cell-penetrating peptide that binds SET and releases PP2A from SET-mediated inhibition",
                        "status": "Preclinical",
                        "evidence_level": "Preclinical",
                    },
                    {
                        "name": "DT-061 (SMAP class)",
                        "mechanism": "Small molecule activator of PP2A — stabilises active B56alpha-containing PP2A holoenzyme",
                        "status": "Preclinical",
                        "evidence_level": "Preclinical",
                        "key_references": [
                            "Sangodkar et al., Cell 2017 — SMAPs as direct PP2A activators in cancer",
                            "Leonard et al., Nat Chem Biol 2020 — SMAP mechanism",
                        ],
                    },
                ],
            },
            {
                "gene": "General AML",
                "variant": "N/A",
                "drugs": [
                    {
                        "name": "Venetoclax (Venclexta)",
                        "mechanism": "BCL-2 inhibitor — induces apoptosis in AML cells dependent on BCL-2 for survival",
                        "status": "FDA-approved (2018, with HMA or LDAC)",
                        "indication": "Newly diagnosed AML in adults 75+ or unfit for intensive chemo, in combination with azacitidine, decitabine, or low-dose cytarabine",
                        "evidence_level": "Approved",
                        "key_trials": [
                            "VIALE-A (NCT02993523): ven+aza vs aza alone — CR/CRi 66.4% vs 28.3%, median OS 14.7 vs 9.6 months",
                            "VIALE-C (NCT02993523): ven+LDAC",
                        ],
                        "mutation_specific_notes": {
                            "IDH2": "IDH2 mutant AML shows high sensitivity to venetoclax — 2-HG accumulation creates BCL-2 dependency via mitochondrial metabolic vulnerability. CR/CRi rates ~75% with ven+aza in IDH-mutant AML.",
                            "DNMT3A": "DNMT3A mutant AML generally shows good response to ven+aza",
                            "PTPN11": "RAS pathway mutations (including PTPN11) may confer relative resistance through MCL-1 upregulation via MAPK signalling",
                        },
                    },
                    {
                        "name": "Midostaurin (Rydapt)",
                        "mechanism": "Multi-kinase inhibitor (FLT3, KIT, PDGFR, VEGFR, PKC)",
                        "status": "FDA-approved (2017)",
                        "indication": "FLT3-mutated AML — NOT directly applicable unless FLT3 mutation present",
                        "evidence_level": "Not applicable unless FLT3-mutated",
                    },
                    {
                        "name": "Gilteritinib (Xospata)",
                        "mechanism": "FLT3 inhibitor (FLT3-ITD and FLT3-TKD)",
                        "status": "FDA-approved (2018)",
                        "indication": "R/R FLT3-mutated AML — NOT applicable without FLT3 mutation",
                        "evidence_level": "Not applicable unless FLT3-mutated",
                    },
                ],
            },
            {
                "gene": "Monosomy 7",
                "variant": "-7",
                "drugs": [
                    {
                        "name": "No monosomy 7-specific therapies approved",
                        "mechanism": "Chromosome 7 loss removes multiple tumour suppressors (EZH2, CUX1, MLL3/KMT2C). Synthetic lethality approaches under investigation.",
                        "status": "Preclinical / early phase",
                        "evidence_level": "Preclinical",
                    },
                    {
                        "name": "EZH2 inhibitors (tazemetostat) — synthetic lethality concept",
                        "mechanism": "If monosomy 7 removes one EZH2 allele, remaining allele may be haploinsufficient. However, EZH2 inhibition may not be appropriate if residual EZH2 is needed. Complex biology.",
                        "status": "Conceptual / preclinical",
                        "evidence_level": "Preclinical",
                        "notable": "Paradoxical — EZH2 loss-of-function is pathogenic in MDS, so EZH2 inhibition could worsen disease. Use with extreme caution.",
                    },
                    {
                        "name": "Allogeneic stem cell transplant",
                        "mechanism": "Only curative option for monosomy 7 — replaces malignant clone with donor haematopoiesis",
                        "status": "Standard of care",
                        "evidence_level": "Approved (transplant)",
                        "notable": "Monosomy 7 is an indication for transplant given adverse risk category",
                    },
                ],
            },
        ],
    }

    # Save drug targets
    out_path = RESULTS_DIR / "drug_targets.json"
    with open(out_path, "w") as f:
        json.dump(drug_targets, f, indent=2)
    print(f"  Saved: {out_path}")
    print(f"  Total target categories: {len(drug_targets['targets'])}")
    total_drugs = sum(len(t["drugs"]) for t in drug_targets["targets"])
    print(f"  Total drug entries: {total_drugs}")

    return drug_targets


# ---------------------------------------------------------------------------
# 2. Combination Strategy Analysis via Gemini
# ---------------------------------------------------------------------------
def call_gemini(prompt: str, model: str = "gemini-2.5-flash", max_tokens: int = 16384) -> str:
    """Call Gemini API and return text response."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_tokens,
        },
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    # Extract text from response
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        print(f"  WARNING: Gemini response parse error: {e}")
        print(f"  Raw response keys: {list(data.keys())}")
        return f"ERROR: Could not parse Gemini response. Raw: {json.dumps(data, indent=2)[:500]}"


def analyse_combination_strategies() -> str:
    """Use Gemini to analyse drug combination strategies for the mutation profile."""
    print("\n[2/5] Analysing combination strategies via Gemini...")

    prompt = textwrap.dedent("""\
    You are a haematology-oncology clinical pharmacologist specialising in AML drug combinations.

    Analyse the following drug combination strategies for a patient with MDS-AML harbouring:
    - DNMT3A R882H (dominant-negative loss of methyltransferase activity)
    - IDH2 R140Q (neomorphic, produces 2-hydroxyglutarate)
    - SETBP1 G870S (gain-of-function, stabilises SET, inhibits PP2A)
    - PTPN11 E76Q (gain-of-function SHP2, constitutive RAS/MAPK activation)
    - Monosomy 7 (adverse cytogenetics)

    ESM-2 pathogenicity scores: DNMT3A=-8.79 (strong), SETBP1=-10.10 (strongest), PTPN11=-1.76, IDH2=-1.20
    DiffDock binding scores for SHP2 inhibitors on PTPN11 E76Q: RMC-4550=-9.26, SHP099=-7.83, TNO155=-7.49

    For EACH of these combination strategies, provide:
    1. Mechanistic rationale for synergy
    2. Published preclinical or clinical evidence (cite specific studies if they exist)
    3. Potential antagonisms or toxicity concerns
    4. A synergy score (1-5, where 5 = strong evidence of synergy)
    5. Clinical feasibility assessment

    COMBINATION STRATEGIES TO ANALYSE:

    **Strategy A: Enasidenib + Azacitidine**
    - Enasidenib targets IDH2 R140Q directly
    - Azacitidine as HMA backbone for DNMT3A context
    - AGILE trial data applicability?

    **Strategy B: SHP2 inhibitor (RMC-4550 or TNO155) + Enasidenib**
    - Dual pathway targeting: RAS/MAPK (PTPN11) + metabolism (IDH2)
    - Any preclinical data on SHP2i + IDH2i combinations?

    **Strategy C: Venetoclax + Azacitidine (standard backbone) + Enasidenib**
    - Ven+Aza as standard of care
    - IDH2 mutations create BCL-2 dependency
    - Triple combination feasibility?

    **Strategy D: PP2A activator (FTY720 or DT-061) + Venetoclax + Azacitidine**
    - Targeting SETBP1 G870S effect (PP2A reactivation)
    - Combined with ven+aza backbone
    - Any preclinical rationale?

    **Strategy E: SHP2 inhibitor + Venetoclax + Azacitidine**
    - RAS pathway mutations may cause venetoclax resistance via MCL-1
    - SHP2 inhibition could resensitise to venetoclax
    - Evidence for this approach?

    **Strategy F: Comprehensive quadruple approach**
    - Venetoclax + Azacitidine (backbone) + Enasidenib (IDH2) + SHP2 inhibitor (PTPN11)
    - Would this be tolerable? Any precedent for 4-drug targeted combos in AML?

    Format your response in Markdown with clear headers for each strategy.
    Be specific about evidence — distinguish between "published data shows" vs "mechanistic rationale suggests".
    End with a SUMMARY TABLE ranking all strategies by synergy score and clinical feasibility.

    CRITICAL FORMATTING RULES:
    - Use proper multi-line Markdown. Never put entire sections on a single line.
    - Tables must have each row on its own line.
    - Cover ALL SIX strategies completely — do not truncate or skip any.
    - Use line breaks between sections.
    """)

    response = call_gemini(prompt, max_tokens=32768)

    # Build the full document
    doc = f"""# Drug Combination Strategy Analysis
## Patient: DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + Monosomy 7

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
*Model: Gemini 2.5 Flash*
*Analysis type: Mechanistic synergy assessment with literature grounding*

---

## Mutation Profile Context

| Gene | Variant | Pathway | ESM-2 Score | Therapeutic Target |
|------|---------|---------|-------------|-------------------|
| DNMT3A | R882H | DNA methylation | -8.79 (Strong) | HMAs (azacitidine) |
| IDH2 | R140Q | TCA cycle / 2-HG | -1.20 (Uncertain) | Enasidenib |
| SETBP1 | G870S | PP2A inhibition | -10.10 (Strong) | PP2A activators |
| PTPN11 | E76Q | RAS/MAPK | -1.76 (Supporting) | SHP2 inhibitors |
| -7 | Monosomy | Multiple TSGs | N/A | Transplant |

---

{response}

---

*Disclaimer: This analysis is for research purposes only and does not constitute medical advice.
Drug combinations should only be administered under clinical trial protocols or specialist supervision.
Evidence levels and synergy scores are approximations based on available literature.*
"""

    out_path = RESULTS_DIR / "combination_strategies.md"
    with open(out_path, "w") as f:
        f.write(doc)
    print(f"  Saved: {out_path}")
    print(f"  Response length: {len(response)} characters")

    return doc


# ---------------------------------------------------------------------------
# 3. Novel Drug Candidates via Gemini
# ---------------------------------------------------------------------------
def analyse_novel_candidates() -> str:
    """Query Gemini for novel drug candidates targeting undruggable mutations."""
    print("\n[3/5] Querying novel drug candidates via Gemini...")

    prompt = textwrap.dedent("""\
    You are a drug discovery scientist specialising in haematologic malignancies.

    For a patient with MDS-AML harbouring DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + monosomy 7:

    Address these SPECIFIC questions with evidence:

    **1. SETBP1 G870S-specific inhibitors:**
    - Are there any direct SETBP1 inhibitors in development as of early 2025?
    - What about indirect approaches (PROTAC degraders, antisense oligonucleotides)?
    - Status of any screening campaigns against SETBP1?

    **2. PP2A activators (since SETBP1 G870S stabilises SET → inhibits PP2A):**
    - FTY720/fingolimod: preclinical data in myeloid malignancies?
    - DT-061 and SMAP class compounds: latest development status?
    - OP449 (SET antagonist peptide): still in development?
    - Any other PP2A-targeting approaches?
    - What is the theoretical therapeutic window?

    **3. Chromosome 7 synthetic lethality approaches:**
    - Key tumour suppressors on chromosome 7 (EZH2 at 7q36, CUX1 at 7q22, MLL3/KMT2C at 7q36)?
    - Synthetic lethal partners for loss of these genes?
    - Any compounds exploiting monosomy 7 specifically?
    - CRISPR screen data identifying dependencies in -7 cells?

    **4. Emerging approaches (2024-2025):**
    - Menin inhibitors (revumenib, ziftomenib) — any relevance to this mutation profile?
    - CD47 antibodies (magrolimab) — status in MDS/AML?
    - Bispecific antibodies or ADCs targeting AML surface markers?
    - Any novel targets identified through functional genomics?

    For each candidate:
    - Name the compound and developer
    - Current development phase
    - Specific relevance to this mutation profile
    - Key reference/trial if available

    Be rigorous — clearly state when no data exists rather than speculating.
    Format as Markdown.
    """)

    response = call_gemini(prompt, max_tokens=32768)
    return response


# ---------------------------------------------------------------------------
# 4. Clinical Trial Search — ClinicalTrials.gov API
# ---------------------------------------------------------------------------
def search_clinical_trials() -> str:
    """Search ClinicalTrials.gov API for relevant active trials."""
    print("\n[4/5] Searching ClinicalTrials.gov for active trials...")

    # ClinicalTrials.gov v2 API
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    search_queries = [
        ("IDH2 inhibitor AML", "IDH2 AND inhibitor AND (AML OR acute myeloid leukemia)"),
        ("SHP2 inhibitor myeloid", "(SHP2 OR PTPN11) AND inhibitor AND (myeloid OR leukemia OR AML OR MDS)"),
        ("SETBP1 myeloid", "SETBP1 AND (myeloid OR leukemia OR MDS OR MPN)"),
        ("Venetoclax AML combination", "venetoclax AND (AML OR acute myeloid leukemia) AND (combination OR azacitidine)"),
        ("PP2A activator cancer", "(PP2A OR protein phosphatase 2A) AND (activator OR reactivator) AND (leukemia OR cancer)"),
        ("Enasidenib combination AML", "enasidenib AND (combination OR azacitidine) AND AML"),
        ("Monosomy 7 MDS", "(monosomy 7 OR del(7q)) AND (MDS OR myelodysplastic)"),
        ("Hypomethylating agent IDH2 AML", "(azacitidine OR decitabine) AND IDH2 AND (AML OR myeloid)"),
    ]

    all_trials = []

    for label, query in search_queries:
        print(f"  Searching: {label}...")
        params = {
            "query.term": query,
            "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING,ENROLLING_BY_INVITATION",
            "pageSize": 10,
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,Condition,InterventionName,LeadSponsorName,StartDate,StudyType",
            "format": "json",
        }
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            studies = data.get("studies", [])
            print(f"    Found {len(studies)} trials")
            for study in studies:
                proto = study.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                design = proto.get("designModule", {})
                cond_mod = proto.get("conditionsModule", {})
                arms_mod = proto.get("armsInterventionsModule", {})
                sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

                nct_id = ident.get("nctId", "N/A")
                title = ident.get("briefTitle", "N/A")
                status = status_mod.get("overallStatus", "N/A")

                phases_list = design.get("phases", [])
                phase = ", ".join(phases_list) if phases_list else "N/A"

                conditions = cond_mod.get("conditions", [])
                conditions_str = "; ".join(conditions[:3]) if conditions else "N/A"

                interventions = arms_mod.get("interventions", [])
                intervention_names = [i.get("name", "") for i in interventions[:5]] if interventions else []
                intervention_str = "; ".join(intervention_names) if intervention_names else "N/A"

                lead_sponsor = sponsor_mod.get("leadSponsor", {})
                sponsor = lead_sponsor.get("name", "N/A")

                trial_info = {
                    "nct_id": nct_id,
                    "title": title,
                    "status": status,
                    "phase": phase,
                    "conditions": conditions_str,
                    "interventions": intervention_str,
                    "sponsor": sponsor,
                    "search_query": label,
                }

                # Deduplicate
                if not any(t["nct_id"] == nct_id for t in all_trials):
                    all_trials.append(trial_info)

        except requests.exceptions.RequestException as e:
            print(f"    WARNING: Search failed for '{label}': {e}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    WARNING: Parse error for '{label}': {e}")

        time.sleep(0.5)  # Rate limiting

    print(f"\n  Total unique trials found: {len(all_trials)}")

    # Build Markdown document
    doc_lines = [
        "# Clinical Trials Relevant to Patient Mutation Profile",
        f"## DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + Monosomy 7",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*Source: ClinicalTrials.gov API v2*",
        f"*Filter: Active / Recruiting / Enrolling*",
        f"*Total unique trials: {len(all_trials)}*",
        "",
        "---",
        "",
    ]

    # Group by search query
    queries_seen = []
    for trial in all_trials:
        q = trial["search_query"]
        if q not in queries_seen:
            queries_seen.append(q)

    for query in queries_seen:
        matching = [t for t in all_trials if t["search_query"] == query]
        doc_lines.append(f"## {query}")
        doc_lines.append("")
        doc_lines.append("| NCT ID | Title | Phase | Status | Interventions | Sponsor |")
        doc_lines.append("|--------|-------|-------|--------|---------------|---------|")
        for t in matching:
            title_short = t["title"][:80] + "..." if len(t["title"]) > 80 else t["title"]
            interv_short = t["interventions"][:60] + "..." if len(t["interventions"]) > 60 else t["interventions"]
            sponsor_short = t["sponsor"][:30] + "..." if len(t["sponsor"]) > 30 else t["sponsor"]
            doc_lines.append(
                f"| [{t['nct_id']}](https://clinicaltrials.gov/study/{t['nct_id']}) "
                f"| {title_short} | {t['phase']} | {t['status']} "
                f"| {interv_short} | {sponsor_short} |"
            )
        doc_lines.append("")

    # Interpretation section
    doc_lines.extend([
        "---",
        "",
        "## Interpretation for This Patient",
        "",
        "### Highest Relevance Trials",
        "Trials combining IDH2 inhibitors with HMAs or venetoclax are most directly applicable,",
        "as they target two of the patient's mutations simultaneously (IDH2 R140Q + DNMT3A R882H context).",
        "",
        "### Emerging Opportunities",
        "- SHP2 inhibitor trials expanding into myeloid malignancies could address PTPN11 E76Q",
        "- PP2A-targeting approaches remain preclinical but could address SETBP1 G870S",
        "",
        "### Limitations",
        "- No trials specifically enrol for the quadruple mutation combination",
        "- SETBP1-specific trials are essentially absent",
        "- Monosomy 7 trials focus on transplant rather than targeted therapy",
        "",
        "*Note: Trial availability and eligibility criteria change frequently.",
        "Verify current status at clinicaltrials.gov before any clinical decisions.*",
    ])

    doc = "\n".join(doc_lines)
    out_path = RESULTS_DIR / "clinical_trials.md"
    with open(out_path, "w") as f:
        f.write(doc)
    print(f"  Saved: {out_path}")

    return doc, all_trials


# ---------------------------------------------------------------------------
# 5. Treatment Ranking — compiled via Gemini with all data
# ---------------------------------------------------------------------------
def compile_treatment_ranking(drug_targets: dict, trial_count: int, novel_text: str) -> str:
    """Use Gemini to compile a final treatment strategy ranking."""
    print("\n[5/5] Compiling treatment ranking via Gemini...")

    prompt = textwrap.dedent(f"""\
    You are a haematology-oncology treatment strategist. Based on ALL of the following data,
    compile a DEFINITIVE treatment strategy ranking for this patient.

    **PATIENT:**
    MDS-AML with DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + Monosomy 7
    - Unique profile: zero matches in ~14,600 myeloid patients in GENIE v19.0
    - ESM-2 pathogenicity: SETBP1 G870S = -10.10 (highest), DNMT3A R882H = -8.79, PTPN11 E76Q = -1.76, IDH2 R140Q = -1.20
    - Adverse risk per ELN 2022 (monosomy 7)

    **AVAILABLE DRUG TARGETS:**
    - IDH2 R140Q: Enasidenib (FDA-approved), vorasidenib (Phase 1 for AML)
    - PTPN11 E76Q/SHP2: RMC-4550 (Phase 1/2), TNO155 (Phase 1/2), JAB-3312 (Phase 1/2)
      DiffDock scores: RMC-4550=-9.26, TNO155=-7.49 (RMC-4550 best binder)
    - DNMT3A R882H: Azacitidine, decitabine (HMAs, approved)
    - SETBP1 G870S: No direct inhibitors. PP2A activators (FTY720, DT-061) are preclinical.
    - General AML: Venetoclax + azacitidine (approved standard of care)
    - Monosomy 7: Allogeneic SCT (only curative option)

    **CLINICAL TRIAL LANDSCAPE:**
    {trial_count} active trials identified across IDH2 inhibitor, SHP2 inhibitor, venetoclax combination, and HMA categories.

    **NOVEL CANDIDATES ANALYSIS:**
    {novel_text[:3000]}

    Create a TREATMENT RANKING document with:

    1. **TIER 1 — Recommended (Approved therapies with strong evidence):**
       Rank by evidence level and mutation-specificity.

    2. **TIER 2 — Clinical Trial Candidates (Phase 1-3):**
       Rank by relevance to specific mutations.

    3. **TIER 3 — Emerging/Investigational (Preclinical with strong rationale):**
       Rank by scientific rationale.

    4. **TIER 4 — Speculative/Future (No current compounds):**
       Note research directions.

    For EACH therapy/combination, provide:
    - Drug name(s)
    - Target mutation(s)
    - Evidence level (Approved / Phase 3 / Phase 2 / Phase 1 / Preclinical)
    - Expected benefit for this specific patient
    - Key risk or limitation
    - Priority score (1-10, where 10 = highest priority)

    5. **RECOMMENDED TREATMENT SEQUENCE:**
       If you were advising a haematologist, what would be the optimal treatment sequence
       considering all mutations? Account for:
       - Induction vs consolidation vs maintenance
       - Transplant timing (given monosomy 7 adverse risk)
       - Sequential vs concurrent targeted therapy

    6. **UNMET NEEDS:**
       What is missing from the current therapeutic landscape for this specific patient?

    Format as Markdown. Include a summary table at the end.
    Be precise about evidence — no over-promising on preclinical data.

    CRITICAL FORMATTING RULES:
    - Use proper multi-line Markdown with line breaks between sections.
    - Tables must have each row on its own line (never concatenate all rows into one line).
    - Keep table cell contents concise (under 80 characters per cell).
    - Use bullet points and numbered lists for detailed explanations.
    - Cover ALL tiers and the recommended treatment sequence completely.
    """)

    response = call_gemini(prompt, max_tokens=32768)

    doc = f"""# Treatment Strategy Ranking
## Patient: DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + Monosomy 7

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
*Model: Gemini 2.5 Flash*
*Data sources: FDA labels, ClinicalTrials.gov, ESM-2 variant scoring, DiffDock molecular docking, GENIE v19.0 co-occurrence*

---

## Key Patient Characteristics

- **Diagnosis:** MDS-AML
- **Unique profile:** 0 matches in ~14,600 myeloid patients (GENIE v19.0)
- **Risk category:** Adverse (ELN 2022) — monosomy 7
- **Highest pathogenicity:** SETBP1 G870S (ESM-2: -10.10) — yet least druggable
- **Most druggable:** IDH2 R140Q — FDA-approved inhibitor available
- **Best docking result:** RMC-4550 on PTPN11 E76Q (DiffDock: -9.26)

---

{response}

---

## Methodology Notes

This ranking integrates:
1. **FDA approval status** — from drug labels and published trial data
2. **ESM-2 pathogenicity scores** — protein language model variant effect prediction
3. **DiffDock molecular docking** — computational binding affinity for SHP2 inhibitors on PTPN11 E76Q
4. **GENIE v19.0 co-occurrence data** — population frequency and co-mutation patterns
5. **ClinicalTrials.gov** — active trial landscape ({trial_count} trials identified)
6. **Gemini AI analysis** — mechanistic synergy assessment and literature synthesis

*Disclaimer: This document is for research purposes only and does not constitute medical advice.
Treatment decisions should be made by qualified haematologist-oncologists with access to the
patient's complete clinical picture, performance status, comorbidities, and preferences.*
"""

    out_path = RESULTS_DIR / "treatment_ranking.md"
    with open(out_path, "w") as f:
        f.write(doc)
    print(f"  Saved: {out_path}")

    return doc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start = time.time()
    print("=" * 80)
    print("DRUG REPURPOSING ANALYSIS")
    print(f"Patient: DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + Monosomy 7")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Step 1: Compile drug targets
    drug_targets = compile_drug_targets()

    # Step 2: Combination strategy analysis
    combination_doc = analyse_combination_strategies()

    # Step 3: Novel drug candidates
    novel_text = analyse_novel_candidates()

    # Save novel candidates as part of combination_strategies or separately
    novel_doc = f"""# Novel Drug Candidates Analysis
## For SETBP1 G870S, PP2A Activation, Chromosome 7 Synthetic Lethality

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
*Model: Gemini 2.5 Flash*

---

{novel_text}

---

*This analysis supplements the main drug repurposing report.
Preclinical candidates require extensive validation before clinical consideration.*
"""
    novel_path = RESULTS_DIR / "novel_candidates.md"
    with open(novel_path, "w") as f:
        f.write(novel_doc)
    print(f"  Saved: {novel_path}")

    # Step 4: Clinical trial search
    trials_doc, all_trials = search_clinical_trials()

    # Step 5: Treatment ranking
    ranking_doc = compile_treatment_ranking(drug_targets, len(all_trials), novel_text)

    # Summary
    elapsed = time.time() - start
    print("\n" + "=" * 80)
    print("DRUG REPURPOSING ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"\nOutput files:")
    for f in sorted(RESULTS_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s} {size_kb:6.1f} KB")
    print(f"\nResults directory: {RESULTS_DIR}")
    print()


if __name__ == "__main__":
    main()
