#!/usr/bin/env python3
"""
gemini_clinical_analysis.py -- Gemini AI clinical interpretation of patient mutation profile.

Uses Google Gemini (gemini-2.5-pro or gemini-2.0-flash fallback) for clinical
interpretation of a complex MDS-AML mutation profile.

Mutation profile:
  - DNMT3A R882H (VAF 39%)
  - IDH2 R140Q (VAF 2%)
  - SETBP1 G870S (VAF 34%)
  - PTPN11 E76Q (VAF 29%)
  - Monosomy 7
  - Diagnosis: MDS-AML (myelodysplastic syndrome / acute myeloid leukemia)

Analyses:
  1. Clinical interpretation (clonal architecture, prognosis, treatment)
  2. Drug interaction analysis (targeted therapies per mutation)
  3. Literature synthesis (2023-2026 publications)
  4. Prognosis modeling (ELN 2022, IPSS-M)

Inputs:
    - .env (GEMINI_API_KEY)
    - Google Gemini API (remote)

Outputs:
    - mutation_profile/results/ai_research/medgemma_clinical/clinical_interpretation.md
    - mutation_profile/results/ai_research/medgemma_clinical/drug_interaction_analysis.md
    - mutation_profile/results/ai_research/medgemma_clinical/literature_synthesis.md
    - mutation_profile/results/ai_research/medgemma_clinical/prognosis_modeling.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/gemini_clinical_analysis.py

Runtime: ~2-4 minutes (Gemini API calls)
Dependencies: google-genai, python-dotenv
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

from google import genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# ---------- Setup ----------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "medgemma_clinical"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

# Try gemini-2.5-pro first, fall back to gemini-2.0-flash
MODEL_CONFIG = {"primary": "gemini-2.5-pro", "fallback": "gemini-2.0-flash", "active": "gemini-2.5-pro"}

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")

# ---------- Shared patient context ----------

PATIENT_CONTEXT = """
PATIENT MUTATION PROFILE (MDS-AML):

Somatic mutations detected by next-generation sequencing panel:
1. DNMT3A R882H — VAF 39%
2. IDH2 R140Q — VAF 2%
3. SETBP1 G870S — VAF 34%
4. PTPN11 E76Q — VAF 29%

Cytogenetics:
- Monosomy 7 (-7)

Diagnosis: Myelodysplastic syndrome / Acute myeloid leukemia (MDS-AML)

Additional clinical context:
- Post-allogeneic hematopoietic stem cell transplant (HSCT) setting
- The combination of DNMT3A + IDH2 + SETBP1 + PTPN11 with monosomy 7 is rare
"""

# ---------- Prompts ----------

PROMPTS = {
    "clinical_interpretation": {
        "filename": "clinical_interpretation.md",
        "title": "Clinical Interpretation of Mutation Profile",
        "prompt": f"""You are a molecular hematologist and clinical genomics expert. Analyze this patient's mutation profile in detail.

{PATIENT_CONTEXT}

Provide a comprehensive clinical interpretation covering:

## 1. Clonal Architecture Interpretation
- Based on the variant allele frequencies (VAFs), reconstruct the likely clonal hierarchy
- Which mutation(s) are likely founder/truncal? Which are subclonal?
- What does the VAF of 39% for DNMT3A R882H suggest about zygosity or clonal fraction?
- How do you interpret IDH2 R140Q at only 2% VAF?
- Discuss whether SETBP1 (34%) and PTPN11 (29%) are in the same clone or separate subclones

## 2. Prognostic Implications
- For each gene individually, what is the prognostic impact?
  - DNMT3A R882H: impact on epigenetic regulation, clonal hematopoiesis
  - IDH2 R140Q: metabolic reprogramming, 2-HG production
  - SETBP1 G870S: role in MDS/MDS-MPN overlap, protein stabilization
  - PTPN11 E76Q: RAS pathway activation, SHP2 gain-of-function
- Combined prognostic impact of this specific constellation
- Impact of monosomy 7 on prognosis

## 3. Treatment Implications
- What targeted therapies are available or relevant?
- How does each mutation affect treatment response?
- Are there any known drug sensitivities or resistances conferred by these mutations?
- How does the post-HSCT setting affect treatment options?

## 4. Molecular Pathogenesis
- How do these mutations cooperate mechanistically?
- What signaling pathways are affected (epigenetic, metabolic, signaling)?
- Is there a known oncogenic synergy between any of these mutations?

Be specific with citations to key studies where possible. Use evidence from 2020-2026 literature."""
    },

    "drug_analysis": {
        "filename": "drug_analysis.md",
        "title": "Targeted Drug Analysis per Mutation",
        "prompt": f"""You are a clinical pharmacologist specializing in precision oncology for myeloid malignancies. Analyze targeted therapy options for this mutation profile.

{PATIENT_CONTEXT}

For EACH mutation, provide a detailed drug analysis:

## 1. DNMT3A R882H (VAF 39%)
- Hypomethylating agents (HMAs): azacitidine, decitabine, oral decitabine/cedazuridine (ASTX727)
- Does R882H specifically affect HMA sensitivity vs wild-type DNMT3A?
- Any DNMT3A-specific therapeutic strategies in development?
- Impact on venetoclax combination therapy

## 2. IDH2 R140Q (VAF 2%)
- Enasidenib (AG-221/IDHIFA): FDA-approved for IDH2-mutant R/R AML
  - Expected response rates, duration of response
  - Differentiation syndrome risk
  - Does the low VAF (2%) affect expected benefit?
- Vorasidenib (dual IDH1/2 inhibitor)
- Olutasidenib (IDH1 — not relevant but discuss cross-reactivity)

## 3. PTPN11 E76Q (VAF 29%)
- SHP2 inhibitors:
  - RMC-4550 (Revolution Medicines)
  - TNO155 (Novartis)
  - RMC-4630 (REACTIVATION trial)
  - BBP-398
  - JAB-3312
- Clinical trial status and availability
- Expected efficacy for E76Q specifically
- Combination strategies with MEK inhibitors

## 4. SETBP1 G870S (VAF 34%)
- Is SETBP1 currently druggable?
- PP2A activators (reactivation of the SET-PP2A axis):
  - FTY720 (fingolimod) — preclinical evidence
  - Perphenazine
  - OP449/TCPA-1
- Any emerging therapeutic strategies?
- Indirect targeting through downstream pathways

## 5. Monosomy 7
- Does -7 affect response to any of the above therapies?
- TP53-independent mechanisms of treatment resistance with -7

## 6. Combination Treatment Strategies
- Optimal combination regimens considering all mutations
- Venetoclax + azacitidine backbone: how do these mutations affect response?
- Sequential vs concurrent targeted therapy approaches
- Clinical trial recommendations for this specific profile

Include drug names, mechanisms, clinical trial identifiers (NCT numbers) where possible, and approval status as of 2026."""
    },

    "literature_synthesis": {
        "filename": "literature_synthesis.md",
        "title": "Literature Synthesis: Co-occurring Mutations in MDS-AML",
        "prompt": f"""You are a hematology researcher conducting a literature review. Synthesize what is known about this specific mutation combination from recent publications.

{PATIENT_CONTEXT}

Provide a comprehensive literature synthesis covering:

## 1. DNMT3A + IDH2 Co-occurrence
- Frequency of co-occurrence in AML/MDS cohorts
- Impact on clonal evolution and disease progression
- Studies on epigenetic-metabolic cooperativity
- Key publications (authors, journals, years)

## 2. SETBP1 in Myeloid Malignancies
- SETBP1 G870S — is this the most common hotspot? (vs D868N, I871T)
- SETBP1 in MDS vs MDS/MPN overlap syndromes
- Co-occurrence with monosomy 7 — is this a known association?
- Makishima et al. and subsequent key studies
- Prognostic impact of SETBP1 mutations in various contexts

## 3. PTPN11 in Adult Myeloid Malignancies
- PTPN11 E76Q — somatic vs germline (Noonan syndrome distinction)
- Frequency and prognostic impact in AML/MDS
- RAS pathway activation and therapeutic targeting
- Recent clinical data with SHP2 inhibitors in myeloid malignancies

## 4. The Four-Gene Combination
- Has this specific combination (DNMT3A + IDH2 + SETBP1 + PTPN11) been reported?
- Studies on multi-hit myeloid malignancies with >3 driver mutations
- AACR GENIE, TCGA AML, or other large cohort data on this combination
- What does the rarity of this combination suggest about clonal evolution?

## 5. Monosomy 7 with These Mutations
- Association between -7 and each of these genes individually
- -7 in post-HSCT relapse or therapy-related MDS/AML
- Impact on treatment response and survival

## 6. Post-HSCT Relapse Genomics
- Mutation landscape in post-HSCT relapse
- Donor-derived vs recipient-derived clonal hematopoiesis
- Implications for salvage therapy selection

## 7. Emerging Research (2024-2026)
- Single-cell genomics studies of clonal architecture in similar profiles
- Novel therapeutic approaches for multi-hit myeloid malignancies
- Precision medicine frameworks (e.g., Beat AML, MDS Foundation trials)

For each section, cite specific studies with authors, journal, year, and key findings. Distinguish between well-established evidence and emerging/preliminary data."""
    },

    "prognosis": {
        "filename": "prognosis.md",
        "title": "Prognosis Modeling: ELN 2022 & IPSS-M Classification",
        "prompt": f"""You are a hematologist specializing in risk stratification of myeloid malignancies. Provide a detailed prognostic assessment.

{PATIENT_CONTEXT}

## 1. ELN 2022 Risk Classification
- Apply the ELN 2022 risk stratification criteria step by step
- Which risk category does this patient fall into?
- How does each mutation contribute to the classification?
- Specifically address:
  - DNMT3A R882H: ELN 2022 considers DNMT3A mutations — what category?
  - IDH2 R140Q: ELN 2022 categorizes IDH2 mutations as favorable IF no adverse features
  - SETBP1: How is SETBP1 handled in ELN 2022?
  - PTPN11: Classification of RAS pathway mutations
  - Monosomy 7: Cytogenetic risk category
- What is the FINAL ELN 2022 risk when adverse features override favorable mutations?

## 2. IPSS-M (Molecular International Prognostic Scoring System)
- Walk through IPSS-M scoring for this profile
- Which mutation categories are scored?
- Impact of each gene on the IPSS-M score:
  - DNMT3A as a gene with established prognostic weight
  - IDH2 and its IPSS-M contribution
  - SETBP1 and its category in IPSS-M
  - PTPN11 / RAS pathway gene scoring
- Cytogenetic score for monosomy 7
- Expected IPSS-M risk category (Very Low / Low / Moderate Low / Moderate High / High / Very High)

## 3. Survival Estimates
- Median overall survival based on ELN 2022 adverse risk
- IPSS-M-based survival projections
- Impact of post-HSCT setting on these estimates
- Does the specific mutation combination modify standard survival curves?

## 4. Comparison of Risk Models
- How does the risk classification compare between:
  - Original IPSS (cytogenetics only)
  - IPSS-R (revised, still cytogenetics-heavy)
  - IPSS-M (molecular-integrated)
  - ELN 2022
- Which model best captures the biology of this case?

## 5. Dynamic Risk Assessment
- How should risk be reassessed over time?
- VAF monitoring: which mutations to track?
- Measurable residual disease (MRD) considerations
- When would re-biopsy and re-staging be indicated?

## 6. Clinical Decision Points
- Based on this risk stratification:
  - Is re-transplantation an option?
  - Donor lymphocyte infusion (DLI) timing
  - When to initiate salvage therapy vs watchful waiting
  - Clinical trial eligibility based on risk category

Provide specific numbers (percentages, months) where evidence supports them. Clearly label estimated vs evidence-based figures."""
    }
}


def query_gemini(prompt: str, model: str = MODEL_CONFIG["primary"]) -> str:
    """Send a prompt to Gemini and return the response text."""
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 16384,
            }
        )
        return response.text
    except Exception as e:
        if model == MODEL_CONFIG["primary"]:
            print(f"  [!] {MODEL_CONFIG["primary"]} failed ({e}), falling back to {MODEL_CONFIG["fallback"]}...")
            return query_gemini(prompt, model=MODEL_CONFIG["fallback"])
        else:
            raise


def save_result(filename: str, title: str, content: str, model_used: str):
    """Save analysis result as markdown."""
    output_path = RESULTS_DIR / filename
    header = f"""# {title}

**Generated:** {TIMESTAMP}
**Model:** Google Gemini ({model_used})
**Patient:** MDS-AML with DNMT3A R882H, IDH2 R140Q, SETBP1 G870S, PTPN11 E76Q, monosomy 7

---

"""
    output_path.write_text(header + content, encoding="utf-8")
    print(f"  Saved: {output_path}")
    return output_path


def run_analysis(key: str, config: dict) -> tuple[str, str]:
    """Run a single analysis and return (content, model_used)."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS: {config['title']}")
    print(f"{'='*70}")

    start = time.time()

    # Try primary model
    model_used = MODEL_CONFIG["primary"]
    try:
        content = query_gemini(config["prompt"], model=MODEL_CONFIG["primary"])
    except Exception as e:
        print(f"  [!] Both models failed for {key}: {e}")
        content = f"**ERROR**: Analysis failed. Error: {e}"
        model_used = "FAILED"

    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s (model: {model_used})")

    # If primary failed and fallback was used, update model_used
    # (handled inside query_gemini, but we track it here for reporting)
    if "falling back" in content if isinstance(content, str) else False:
        model_used = MODEL_CONFIG["fallback"]

    save_result(config["filename"], config["title"], content, model_used)
    return content, model_used


def print_key_findings(results: dict):
    """Print a summary of key findings from all analyses."""
    print(f"\n{'='*70}")
    print("  KEY FINDINGS SUMMARY")
    print(f"{'='*70}")

    for key, (content, model) in results.items():
        title = PROMPTS[key]["title"]
        # Extract first meaningful paragraph (skip headers)
        lines = content.split("\n")
        summary_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("*") and len(stripped) > 40:
                summary_lines.append(stripped)
                if len(summary_lines) >= 3:
                    break
        summary = " ".join(summary_lines)[:300]
        print(f"\n  [{title}]")
        print(f"  Model: {model}")
        print(f"  Preview: {summary}...")


def main():
    primary = MODEL_CONFIG["primary"]
    fallback = MODEL_CONFIG["fallback"]

    print(f"\nGemini Clinical Analysis — MDS-AML Mutation Profile")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Output: {RESULTS_DIR}")
    print(f"Primary model: {primary}")
    print(f"Fallback model: {fallback}")

    # Verify API key works with a quick test
    print("\nVerifying Gemini API connection...")
    try:
        test = client.models.generate_content(
            model=primary,
            contents="Respond with exactly: OK",
            config={"max_output_tokens": 10}
        )
        print(f"  API connection verified (model: {primary})")
    except Exception as e:
        print(f"  {primary} unavailable ({e}), trying {fallback}...")
        try:
            test = client.models.generate_content(
                model=fallback,
                contents="Respond with exactly: OK",
                config={"max_output_tokens": 10}
            )
            # Switch active model to fallback
            MODEL_CONFIG["active"] = fallback
            MODEL_CONFIG["primary"] = fallback
            print(f"  Using {fallback} for all analyses")
        except Exception as e2:
            print(f"  ERROR: Both models failed. Check API key. Error: {e2}")
            sys.exit(1)

    # Run all four analyses sequentially
    results = {}
    analysis_order = [
        "clinical_interpretation",
        "drug_analysis",
        "literature_synthesis",
        "prognosis"
    ]

    for key in analysis_order:
        config = PROMPTS[key]
        content, model = run_analysis(key, config)
        results[key] = (content, model)

    # Print summary
    print_key_findings(results)

    # Final output
    print(f"\n{'='*70}")
    print(f"  ALL ANALYSES COMPLETE")
    print(f"{'='*70}")
    print(f"  Results saved to: {RESULTS_DIR}")
    for key in analysis_order:
        filepath = RESULTS_DIR / PROMPTS[key]["filename"]
        size_kb = filepath.stat().st_size / 1024
        print(f"    - {PROMPTS[key]['filename']} ({size_kb:.1f} KB)")
    print()


if __name__ == "__main__":
    main()
