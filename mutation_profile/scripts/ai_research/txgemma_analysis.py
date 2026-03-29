#!/usr/bin/env python3
"""
txgemma_analysis.py -- TxGemma therapeutic prediction analysis for patient mutation profile.

Runs Google's TxGemma therapeutic AI model for drug-target interaction predictions,
synthetic lethality analysis, clinical trial matching, and resistance prediction
for a patient with DNMT3A R882H + IDH2 R140Q + SETBP1 G870S + PTPN11 E76Q + monosomy 7.

Approach A: TxGemma-2B-predict via HuggingFace (local GPU inference)
Approach B: Gemini API with specialized therapeutic prompting (fallback)

Inputs:
    - .env (GEMINI_API_KEY, HF_TOKEN)
    - HuggingFace model: google/txgemma-2b-predict (optional, requires GPU)
    - Google Gemini API (remote, fallback)

Outputs:
    - mutation_profile/results/ai_research/txgemma/drug_target_interactions.json
    - mutation_profile/results/ai_research/txgemma/synthetic_lethality.json
    - mutation_profile/results/ai_research/txgemma/clinical_trial_matches.json
    - mutation_profile/results/ai_research/txgemma/resistance_predictions.json
    - mutation_profile/results/ai_research/txgemma/txgemma_report.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/txgemma_analysis.py

Runtime: ~3-8 minutes (GPU with TxGemma), ~2-4 minutes (Gemini fallback)
Dependencies: torch, transformers, huggingface_hub, google-genai
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "ai_research" / "txgemma_therapeutic"

# Patient mutation profile
PATIENT_MUTATIONS = {
    "DNMT3A": {
        "variant": "R882H",
        "vaf": 0.39,
        "role": "founder clone",
        "mechanism": "Disrupts catalytic domain, reduces DNA methyltransferase activity by ~80%",
        "esm2_score": -8.383,
        "esm2_acmg": "PP3_Strong",
    },
    "IDH2": {
        "variant": "R140Q",
        "vaf": 0.02,
        "role": "subclonal",
        "mechanism": "Gain-of-function neomorphic, produces 2-hydroxyglutarate oncometabolite",
        "esm2_score": -1.478,
        "esm2_acmg": "Benign/VUS (gain-of-function, not captured by conservation)",
    },
    "SETBP1": {
        "variant": "G870S",
        "vaf": 0.37,
        "role": "co-dominant clone",
        "mechanism": "SKI homology domain, disrupts degron motif, stabilizes protein",
        "esm2_score": -9.804,
        "esm2_acmg": "PP3_Strong",
    },
    "PTPN11": {
        "variant": "E76Q",
        "vaf": 0.05,
        "role": "subclonal",
        "mechanism": "N-SH2 domain, disrupts autoinhibitory interface, constitutive RAS activation",
        "esm2_score": -1.865,
        "esm2_acmg": "PP3_Supporting",
    },
}

CYTOGENETICS = {
    "karyotype": "monosomy 7 / del(7q)",
    "significance": "Adverse-risk cytogenetic abnormality in MDS/AML, associated with poor prognosis",
}

# Drug-target pairs for binding affinity prediction
DRUG_TARGET_PAIRS = [
    # PTPN11/SHP2 inhibitors
    {
        "target": "PTPN11 (SHP2)",
        "mutation": "E76Q",
        "drug": "RMC-4550",
        "drug_class": "SHP2 allosteric inhibitor",
        "mechanism": "Binds tunnel between N-SH2 and PTP domains, stabilizes autoinhibited conformation",
        "clinical_status": "Preclinical (Revolution Medicines)",
        "smiles": "CC1=CC(=CC(=C1)C2=CC3=C(C=C2)N(C(=O)C4=C3C=CC(=C4)Cl)CC5=CC=C(C=C5)C(=O)O)C",
    },
    {
        "target": "PTPN11 (SHP2)",
        "mutation": "E76Q",
        "drug": "SHP099",
        "drug_class": "SHP2 allosteric inhibitor",
        "mechanism": "Locks SHP2 in closed autoinhibited state",
        "clinical_status": "Tool compound (Novartis)",
        "smiles": "CC1=CC(=C(C=C1Cl)Cl)NC2=CC(=NC=C2C#N)NC3CCCCC3",
    },
    {
        "target": "PTPN11 (SHP2)",
        "mutation": "E76Q",
        "drug": "TNO155",
        "drug_class": "SHP2 allosteric inhibitor",
        "mechanism": "Clinical-grade SHP2 inhibitor, binds allosteric tunnel",
        "clinical_status": "Phase I/II (Novartis NCT04330664)",
        "smiles": "CC(C)C1=CC2=C(C=C1F)C(=O)N(C3=C2C=C(C=C3)OC4CCNCC4)C",
    },
    # IDH2 inhibitor
    {
        "target": "IDH2",
        "mutation": "R140Q",
        "drug": "Enasidenib (AG-221/Idhifa)",
        "drug_class": "IDH2 inhibitor",
        "mechanism": "Allosteric inhibitor of mutant IDH2, blocks 2-HG production",
        "clinical_status": "FDA-approved for relapsed/refractory AML with IDH2 mutation",
        "smiles": "CC(C)(O)C1=CC(=NC(=N1)NC2=CC=C(C=C2)C(F)(F)F)NC3=CC=C(C=C3)C(F)(F)F",
    },
    # DNMT3A-targeting agents (hypomethylating)
    {
        "target": "DNMT3A",
        "mutation": "R882H",
        "drug": "Azacitidine (Vidaza)",
        "drug_class": "Hypomethylating agent (HMA)",
        "mechanism": "Cytidine analog, incorporates into DNA/RNA, traps DNMTs causing degradation",
        "clinical_status": "FDA-approved for MDS, standard of care",
        "smiles": "C1=CN(C(=O)N=C1N)[C@H]2[C@@H]([C@@H](CO)O2)O",
    },
    {
        "target": "DNMT3A",
        "mutation": "R882H",
        "drug": "Decitabine (Dacogen)",
        "drug_class": "Hypomethylating agent (HMA)",
        "mechanism": "Deoxycytidine analog, incorporates into DNA, traps DNMTs",
        "clinical_status": "FDA-approved for MDS",
        "smiles": "C1=CN(C(=O)N=C1N)[C@H]2C[C@@H]([C@H](O2)CO)O",
    },
]

# Synthetic lethality queries
SYNTHETIC_LETHALITY_QUERIES = [
    {
        "gene_pair": ("DNMT3A", "IDH2"),
        "context": "Both affect DNA methylation: DNMT3A R882H reduces de novo methylation, "
                   "IDH2 R140Q produces 2-HG that inhibits TET2/demethylation. "
                   "Combined epigenetic disruption.",
        "question": "What synergistic therapeutic targets emerge from combined DNMT3A loss-of-function "
                    "and IDH2 gain-of-function in myeloid malignancy?",
    },
    {
        "gene_pair": ("SETBP1", "monosomy_7"),
        "context": "SETBP1 G870S stabilizes SET protein which inhibits PP2A tumor suppressor. "
                   "Monosomy 7 causes haploinsufficiency of multiple tumor suppressors on chr7. "
                   "Both co-occur in aggressive MDS/AML.",
        "question": "What therapeutic vulnerabilities arise from combined SETBP1 activation and "
                    "chromosome 7 loss in MDS/AML? Are there specific chromosome instability "
                    "or PP2A pathway targets?",
    },
    {
        "gene_pair": ("PTPN11", "SETBP1"),
        "context": "PTPN11 E76Q activates RAS/MAPK signaling. SETBP1 G870S inhibits PP2A "
                   "which normally counteracts RAS signaling. Double activation of proliferative pathways.",
        "question": "Does co-occurring PTPN11 E76Q and SETBP1 G870S create enhanced vulnerability "
                    "to MEK inhibitors or combined MAPK/PP2A-targeted therapy?",
    },
    {
        "gene_pair": ("DNMT3A", "SETBP1"),
        "context": "DNMT3A R882H (VAF 39%) and SETBP1 G870S (VAF 37%) appear co-dominant. "
                   "DNMT3A affects epigenetic regulation, SETBP1 affects PP2A/SET axis. "
                   "Zero co-occurrence in GENIE 14,600 myeloid patients.",
        "question": "Given the extreme rarity of DNMT3A R882H + SETBP1 G870S co-occurrence "
                    "(0 in ~14,600 patients), what unique synthetic lethal vulnerabilities "
                    "might exist in this combination?",
    },
]


# ---------------------------------------------------------------------------
# TxGemma Predict-mode prompts
# ---------------------------------------------------------------------------

def build_drug_interaction_prompt(pair: dict) -> str:
    """Build TxGemma-style drug-target interaction prompt."""
    return (
        f"Predict the binding affinity and therapeutic efficacy of {pair['drug']} "
        f"against {pair['target']} with the {pair['mutation']} mutation.\n\n"
        f"Drug class: {pair['drug_class']}\n"
        f"Drug mechanism: {pair['mechanism']}\n"
        f"SMILES: {pair['smiles']}\n"
        f"Mutation effect: The {pair['mutation']} mutation in {pair['target']} "
        f"{'disrupts the autoinhibitory interface causing constitutive activation' if 'SHP2' in pair['target'] else 'alters protein function'}.\n\n"
        f"For the mutant protein, predict:\n"
        f"1. Whether the drug retains binding to the mutant form (yes/no/reduced)\n"
        f"2. Estimated change in binding affinity vs wildtype (increase/decrease/similar)\n"
        f"3. Expected clinical efficacy for this specific mutation\n"
        f"4. Known resistance mechanisms\n"
    )


def build_synthetic_lethality_prompt(query: dict) -> str:
    """Build synthetic lethality prediction prompt."""
    g1, g2 = query["gene_pair"]
    return (
        f"Predict synthetic lethal interactions for a myeloid malignancy patient "
        f"with co-occurring {g1} and {g2} alterations.\n\n"
        f"Context: {query['context']}\n\n"
        f"Question: {query['question']}\n\n"
        f"Provide:\n"
        f"1. Top 3 predicted synthetic lethal targets with rationale\n"
        f"2. Existing drugs that could exploit each vulnerability\n"
        f"3. Confidence level (high/medium/low) based on preclinical evidence\n"
        f"4. Any active clinical trials targeting this vulnerability\n"
    )


def build_clinical_trial_prompt() -> str:
    """Build clinical trial matching prompt."""
    mutations_desc = "\n".join(
        f"  - {gene} {info['variant']} (VAF {info['vaf']:.0%}, {info['role']})"
        for gene, info in PATIENT_MUTATIONS.items()
    )
    return (
        f"Match the following myeloid malignancy patient to relevant clinical trials.\n\n"
        f"Patient profile:\n"
        f"  Diagnosis: MDS/AML (myelodysplastic syndrome with excess blasts / acute myeloid leukemia)\n"
        f"  Cytogenetics: {CYTOGENETICS['karyotype']}\n"
        f"  Molecular mutations:\n{mutations_desc}\n"
        f"  Prior therapy: Post allogeneic HSCT (hematopoietic stem cell transplant)\n"
        f"  Status: Post-transplant, monitoring for relapse\n\n"
        f"Based on this molecular profile, identify:\n"
        f"1. Clinical trials specifically targeting IDH2 R140Q (enasidenib-based)\n"
        f"2. Clinical trials targeting SHP2/PTPN11 (allosteric inhibitors)\n"
        f"3. Clinical trials for SETBP1-mutant myeloid malignancies\n"
        f"4. Combination therapy trials relevant to this mutation constellation\n"
        f"5. Post-transplant maintenance trials that accept this molecular profile\n"
        f"6. Any precision medicine basket trials that would accept multiple mutations\n\n"
        f"For each trial, provide: trial ID (NCT number if known), phase, drug(s), "
        f"eligibility criteria relevant to this patient, and current enrollment status.\n"
    )


def build_resistance_prompt() -> str:
    """Build resistance prediction prompt."""
    return (
        f"Predict resistance mechanisms for the following clonal architecture in MDS/AML.\n\n"
        f"Clonal hierarchy (inferred from VAF):\n"
        f"  Founder clone (VAF ~39%): DNMT3A R882H\n"
        f"  Co-dominant (VAF ~37%): SETBP1 G870S\n"
        f"  Subclonal (VAF ~5%): PTPN11 E76Q\n"
        f"  Subclonal (VAF ~2%): IDH2 R140Q\n"
        f"  Cytogenetics: Monosomy 7\n\n"
        f"Clinical context: Post-HSCT MDS/AML\n\n"
        f"Predict:\n"
        f"1. If treated with enasidenib (IDH2 inhibitor): Will the IDH2 R140Q subclone "
        f"be eliminated while DNMT3A/SETBP1 clones persist? Expected resistance mechanisms?\n\n"
        f"2. If treated with azacitidine (HMA): How does the DNMT3A R882H founder clone "
        f"respond? Does the dominant-negative effect of R882H affect HMA efficacy?\n\n"
        f"3. If treated with SHP2 inhibitor (TNO155): Will the E76Q mutation affect "
        f"drug binding? Can the PTPN11 subclone evolve secondary resistance?\n\n"
        f"4. Clonal evolution prediction: Which clone is most likely to drive relapse?\n"
        f"   Consider: DNMT3A R882H clones are known to persist post-HSCT as "
        f"   pre-leukemic clones with competitive advantage in hematopoiesis.\n\n"
        f"5. What combination strategy would address all four mutations simultaneously?\n"
    )


# ---------------------------------------------------------------------------
# Approach A: TxGemma via HuggingFace (local)
# ---------------------------------------------------------------------------

def try_txgemma_local():
    """Attempt to load and run TxGemma-2B-predict locally via HuggingFace."""
    print("\n--- Approach A: TxGemma-2B-predict via HuggingFace ---")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  transformers not available")
        return None

    # Check HF token
    try:
        from huggingface_hub import get_token
        token = get_token()
        if not token:
            print("  No HuggingFace token found. TxGemma is gated and requires:")
            print("    1. Accept license at https://huggingface.co/google/txgemma-2b-predict")
            print("    2. Run: huggingface-cli login")
            return None
    except Exception as e:
        print(f"  HF token check failed: {e}")
        return None

    model_name = "google/txgemma-2b-predict"
    print(f"  Attempting to load {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"  TxGemma-2B loaded successfully on {model.device}")
        return {"model": model, "tokenizer": tokenizer, "backend": "txgemma-2b-predict"}
    except Exception as e:
        print(f"  Failed to load TxGemma: {e}")
        return None


def run_txgemma_local(model_info: dict, prompt: str, max_new_tokens: int = 1024) -> str:
    """Run inference with local TxGemma model."""
    import torch

    model = model_info["model"]
    tokenizer = model_info["tokenizer"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
        )
    # Decode only the generated part
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ---------------------------------------------------------------------------
# Approach B: Gemini API with therapeutic prompting
# ---------------------------------------------------------------------------

def try_gemini_api():
    """Attempt to connect to Gemini API."""
    print("\n--- Approach B: Gemini API ---")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  No GEMINI_API_KEY found in environment.")
        print("  Set it with: export GEMINI_API_KEY='your-key-here'")
        print("  Get a key at: https://aistudio.google.com/apikey")
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Verify connection
        models = [m.name for m in client.models.list()]
        print(f"  Gemini API connected. Available models: {len(models)}")

        # Check for TxGemma in API
        txgemma_models = [m for m in models if "txgemma" in m.lower()]
        if txgemma_models:
            print(f"  TxGemma available via API: {txgemma_models}")
            return {"client": client, "model": txgemma_models[0], "backend": "txgemma-api"}

        # Fall back to gemini-2.5-flash or gemini-2.0-flash
        preferred = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro", "gemini-1.5-pro"]
        for pref in preferred:
            matches = [m for m in models if pref in m]
            if matches:
                model_name = matches[0]
                print(f"  Using {model_name} with therapeutic prompting")
                return {"client": client, "model": model_name, "backend": f"gemini-therapeutic ({model_name})"}

        # Use whatever is available
        gemini = [m for m in models if "gemini" in m.lower()]
        if gemini:
            model_name = gemini[0]
            print(f"  Using {model_name}")
            return {"client": client, "model": model_name, "backend": f"gemini ({model_name})"}

        print("  No suitable model found in API")
        return None
    except Exception as e:
        print(f"  Gemini API connection failed: {e}")
        return None


THERAPEUTIC_SYSTEM_PROMPT = """You are a computational oncology expert specializing in therapeutic prediction for myeloid malignancies. You have deep knowledge of:

1. Drug-target interactions in leukemia, including binding affinity predictions for mutant proteins
2. Synthetic lethality in cancer genetics, especially epigenetic and signaling pathway interactions
3. Clinical trial landscapes for AML/MDS, including precision medicine and basket trials
4. Clonal evolution and resistance mechanisms in post-transplant settings

Base your predictions on published literature, mechanistic understanding of protein structure-function relationships, and clinical pharmacology data. When making predictions:
- Cite specific mechanisms and evidence
- Distinguish between established evidence and extrapolation
- Provide confidence levels for each prediction
- Note any caveats or limitations

You are analyzing data for a specific patient with MDS/AML who has undergone allogeneic HSCT."""


def run_gemini_api(api_info: dict, prompt: str) -> str:
    """Run inference via Gemini API with therapeutic system context."""
    client = api_info["client"]
    model_name = api_info["model"]

    full_prompt = THERAPEUTIC_SYSTEM_PROMPT + "\n\n" + prompt

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"ERROR: API call failed: {e}"


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def run_analysis(inference_fn, backend_name: str) -> dict:
    """Run all four analysis modules and collect results."""
    results = {
        "metadata": {
            "backend": backend_name,
            "date": datetime.now().isoformat(),
            "patient_mutations": {
                gene: {
                    "variant": info["variant"],
                    "vaf": info["vaf"],
                    "role": info["role"],
                    "esm2_score": info["esm2_score"],
                }
                for gene, info in PATIENT_MUTATIONS.items()
            },
            "cytogenetics": CYTOGENETICS,
        },
        "analyses": {},
    }

    # -----------------------------------------------------------------------
    # 1. Drug-target interaction predictions
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Drug-Target Interaction Predictions")
    print("=" * 70)

    drug_results = []
    for i, pair in enumerate(DRUG_TARGET_PAIRS, 1):
        print(f"\n  [{i}/{len(DRUG_TARGET_PAIRS)}] {pair['drug']} -> {pair['target']} {pair['mutation']}")
        prompt = build_drug_interaction_prompt(pair)
        t0 = time.time()
        response = inference_fn(prompt)
        elapsed = time.time() - t0
        print(f"    Response received ({elapsed:.1f}s, {len(response)} chars)")

        drug_results.append({
            "target": pair["target"],
            "mutation": pair["mutation"],
            "drug": pair["drug"],
            "drug_class": pair["drug_class"],
            "clinical_status": pair["clinical_status"],
            "prediction": response,
            "inference_time_s": round(elapsed, 1),
        })

    results["analyses"]["drug_target_interactions"] = drug_results

    # -----------------------------------------------------------------------
    # 2. Synthetic lethality predictions
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Synthetic Lethality Predictions")
    print("=" * 70)

    synth_results = []
    for i, query in enumerate(SYNTHETIC_LETHALITY_QUERIES, 1):
        g1, g2 = query["gene_pair"]
        print(f"\n  [{i}/{len(SYNTHETIC_LETHALITY_QUERIES)}] {g1} + {g2}")
        prompt = build_synthetic_lethality_prompt(query)
        t0 = time.time()
        response = inference_fn(prompt)
        elapsed = time.time() - t0
        print(f"    Response received ({elapsed:.1f}s, {len(response)} chars)")

        synth_results.append({
            "gene_pair": list(query["gene_pair"]),
            "context": query["context"],
            "prediction": response,
            "inference_time_s": round(elapsed, 1),
        })

    results["analyses"]["synthetic_lethality"] = synth_results

    # -----------------------------------------------------------------------
    # 3. Clinical trial matching
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Clinical Trial Matching")
    print("=" * 70)

    prompt = build_clinical_trial_prompt()
    t0 = time.time()
    response = inference_fn(prompt)
    elapsed = time.time() - t0
    print(f"  Response received ({elapsed:.1f}s, {len(response)} chars)")

    results["analyses"]["clinical_trial_matching"] = {
        "prediction": response,
        "inference_time_s": round(elapsed, 1),
    }

    # -----------------------------------------------------------------------
    # 4. Resistance prediction
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Resistance & Clonal Evolution Prediction")
    print("=" * 70)

    prompt = build_resistance_prompt()
    t0 = time.time()
    response = inference_fn(prompt)
    elapsed = time.time() - t0
    print(f"  Response received ({elapsed:.1f}s, {len(response)} chars)")

    results["analyses"]["resistance_prediction"] = {
        "prediction": response,
        "inference_time_s": round(elapsed, 1),
    }

    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_results(results: dict):
    """Save results to JSON and markdown."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. JSON
    json_path = RESULTS_DIR / "txgemma_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON: {json_path}")

    # 2. Markdown summary
    md_path = RESULTS_DIR / "txgemma_summary.md"
    with open(md_path, "w") as f:
        meta = results["metadata"]
        f.write("# TxGemma Therapeutic Prediction Analysis\n\n")
        f.write(f"**Backend:** {meta['backend']}  \n")
        f.write(f"**Date:** {meta['date']}  \n")
        f.write(f"**Patient:** MDS/AML, post-HSCT  \n")
        f.write(f"**Cytogenetics:** {meta['cytogenetics']['karyotype']}  \n\n")

        f.write("## Patient Mutation Profile\n\n")
        f.write("| Gene | Variant | VAF | Role | ESM-2 Score |\n")
        f.write("|------|---------|-----|------|-------------|\n")
        for gene, info in meta["patient_mutations"].items():
            f.write(f"| {gene} | {info['variant']} | {info['vaf']:.0%} | "
                    f"{info['role']} | {info['esm2_score']} |\n")

        analyses = results["analyses"]

        # Drug-target interactions
        f.write("\n---\n\n## 1. Drug-Target Interaction Predictions\n\n")
        for r in analyses["drug_target_interactions"]:
            f.write(f"### {r['drug']} vs {r['target']} {r['mutation']}\n\n")
            f.write(f"**Drug class:** {r['drug_class']}  \n")
            f.write(f"**Clinical status:** {r['clinical_status']}  \n")
            f.write(f"**Inference time:** {r['inference_time_s']}s  \n\n")
            f.write(f"{r['prediction']}\n\n")

        # Synthetic lethality
        f.write("\n---\n\n## 2. Synthetic Lethality Predictions\n\n")
        for r in analyses["synthetic_lethality"]:
            g1, g2 = r["gene_pair"]
            f.write(f"### {g1} + {g2}\n\n")
            f.write(f"**Context:** {r['context']}  \n\n")
            f.write(f"{r['prediction']}\n\n")

        # Clinical trials
        f.write("\n---\n\n## 3. Clinical Trial Matching\n\n")
        ct = analyses["clinical_trial_matching"]
        f.write(f"{ct['prediction']}\n\n")

        # Resistance
        f.write("\n---\n\n## 4. Resistance & Clonal Evolution Prediction\n\n")
        rp = analyses["resistance_prediction"]
        f.write(f"{rp['prediction']}\n\n")

        # Disclaimer
        f.write("\n---\n\n## Disclaimer\n\n")
        f.write("These predictions are generated by AI models and are intended for "
                "research purposes only. They do not constitute medical advice. "
                "All therapeutic decisions should be made in consultation with "
                "qualified oncology/hematology specialists and should incorporate "
                "clinical judgment, patient-specific factors, and current treatment "
                "guidelines.\n")

    print(f"  MD:   {md_path}")

    # 3. Individual analysis files for easier reading
    for analysis_name, data in analyses.items():
        txt_path = RESULTS_DIR / f"{analysis_name}.txt"
        with open(txt_path, "w") as f:
            f.write(f"TxGemma Analysis: {analysis_name.replace('_', ' ').title()}\n")
            f.write(f"Backend: {meta['backend']}\n")
            f.write(f"Date: {meta['date']}\n")
            f.write("=" * 70 + "\n\n")

            if isinstance(data, list):
                for item in data:
                    if "drug" in item:
                        f.write(f"--- {item['drug']} vs {item['target']} {item['mutation']} ---\n\n")
                    elif "gene_pair" in item:
                        f.write(f"--- {item['gene_pair'][0]} + {item['gene_pair'][1]} ---\n\n")
                    f.write(item["prediction"] + "\n\n")
            else:
                f.write(data["prediction"] + "\n")

        print(f"  TXT:  {txt_path}")


def print_summary(results: dict):
    """Print a concise summary to stdout."""
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE -- SUMMARY")
    print("=" * 70)

    analyses = results["analyses"]

    # Drug interactions summary
    print("\nDrug-Target Interactions:")
    print("-" * 50)
    for r in analyses["drug_target_interactions"]:
        preview = r["prediction"][:120].replace("\n", " ")
        print(f"  {r['drug']:<30} -> {r['target']} {r['mutation']}")
        print(f"    {preview}...")

    # Synthetic lethality
    print("\nSynthetic Lethality Predictions:")
    print("-" * 50)
    for r in analyses["synthetic_lethality"]:
        preview = r["prediction"][:120].replace("\n", " ")
        print(f"  {r['gene_pair'][0]} + {r['gene_pair'][1]}")
        print(f"    {preview}...")

    # Clinical trials
    print("\nClinical Trial Matching:")
    print("-" * 50)
    preview = analyses["clinical_trial_matching"]["prediction"][:200].replace("\n", " ")
    print(f"  {preview}...")

    # Resistance
    print("\nResistance Prediction:")
    print("-" * 50)
    preview = analyses["resistance_prediction"]["prediction"][:200].replace("\n", " ")
    print(f"  {preview}...")

    total_time = sum(
        r.get("inference_time_s", 0)
        for section in analyses.values()
        for r in (section if isinstance(section, list) else [section])
    )
    print(f"\nTotal inference time: {total_time:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    print("=" * 70)
    print("TxGemma Therapeutic Prediction Analysis")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\nPatient profile:")
    for gene, info in PATIENT_MUTATIONS.items():
        print(f"  {gene} {info['variant']} (VAF {info['vaf']:.0%}, {info['role']}, ESM-2: {info['esm2_score']})")
    print(f"  Cytogenetics: {CYTOGENETICS['karyotype']}")

    # Try Approach A: Local TxGemma
    model_info = try_txgemma_local()

    if model_info:
        print(f"\n  Using: {model_info['backend']}")
        inference_fn = lambda prompt: run_txgemma_local(model_info, prompt)
        backend_name = model_info["backend"]
    else:
        # Try Approach B: Gemini API
        api_info = try_gemini_api()

        if api_info:
            print(f"\n  Using: {api_info['backend']}")
            inference_fn = lambda prompt: run_gemini_api(api_info, prompt)
            backend_name = api_info["backend"]
        else:
            print("\n" + "!" * 70)
            print("NEITHER TXGEMMA NOR GEMINI API AVAILABLE")
            print("!" * 70)
            print("\nTo enable this analysis, do ONE of the following:")
            print("\n  Option 1 - TxGemma via HuggingFace (local GPU, preferred):")
            print("    1. Accept license: https://huggingface.co/google/txgemma-2b-predict")
            print("    2. Login: huggingface-cli login")
            print("    3. Re-run this script")
            print("\n  Option 2 - Gemini API (cloud):")
            print("    1. Get API key: https://aistudio.google.com/apikey")
            print("    2. export GEMINI_API_KEY='your-key-here'")
            print("    3. Re-run this script")
            sys.exit(1)

    # Run all analyses
    results = run_analysis(inference_fn, backend_name)

    # Save outputs
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    save_results(results)

    # Print summary
    print_summary(results)

    elapsed = time.time() - start_time
    print(f"\nTotal wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
