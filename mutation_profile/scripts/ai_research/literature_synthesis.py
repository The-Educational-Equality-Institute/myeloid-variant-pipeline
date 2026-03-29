#!/usr/bin/env python3
"""
literature_synthesis.py -- AI-powered literature synthesis for mutation profile analysis.

Uses Google Gemini (gemini-2.5-pro) with Google Search grounding
to perform deep literature analysis on the patient's mutation profile:

  - SETBP1 G870S
  - DNMT3A R882H + IDH2 R140Q
  - PTPN11 E76Q
  - Monosomy 7
  - Full 4-gene combination synthesis
  - Treatment landscape 2026

Inputs:
    - .env (GEMINI_API_KEY)
    - Google Gemini API with Google Search grounding (remote)

Outputs:
    - mutation_profile/results/ai_research/literature_synthesis/setbp1_g870s.md
    - mutation_profile/results/ai_research/literature_synthesis/dnmt3a_idh2.md
    - mutation_profile/results/ai_research/literature_synthesis/ptpn11_e76q.md
    - mutation_profile/results/ai_research/literature_synthesis/monosomy7.md
    - mutation_profile/results/ai_research/literature_synthesis/four_gene_synthesis.md
    - mutation_profile/results/ai_research/literature_synthesis/treatment_landscape_2026.md

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/literature_synthesis.py

Runtime: ~3-6 minutes (Gemini API calls with search grounding)
Dependencies: google-genai, python-dotenv
"""

import os
import sys
import time
import datetime
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent          # mrna-hematology-research/
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "literature_synthesis"

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
load_dotenv(ENV_PATH)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    sys.exit("ERROR: GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use gemini-2.5-pro for deep analysis with Google Search grounding
MODEL = "gemini-2.5-pro"
GOOGLE_SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())

# ---------------------------------------------------------------------------
# Shared system instruction
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are a senior hematology researcher and molecular oncologist with deep expertise "
    "in myeloid malignancies (MDS, AML, aCML, MDS/MPN). You are analyzing the mutation "
    "profile of an adult patient with: SETBP1 G870S, DNMT3A R882H, IDH2 R140Q, "
    "PTPN11 E76Q, and monosomy 7 (del(7)/−7). Provide evidence-based analysis with "
    "specific citations (author, year, journal) when possible. Structure your response "
    "with clear markdown headings. Be thorough, precise, and clinically relevant."
)

# ---------------------------------------------------------------------------
# Analysis prompts
# ---------------------------------------------------------------------------
ANALYSES = [
    {
        "id": "setbp1_review",
        "title": "SETBP1 G870S in Myeloid Malignancies",
        "prompt": """Provide a comprehensive literature review of SETBP1 G870S in myeloid malignancies.

Cover the following in detail:

## 1. Foundational Discovery
- Makishima et al. 2013 (Nature Genetics) — the landmark paper identifying recurrent SETBP1 mutations
  - How were mutations discovered? What was the cohort?
  - What was the reported frequency across myeloid neoplasms?
  - What was the functional consequence demonstrated?

## 2. SKI Domain Hotspot Significance
- Where does G870S sit within the SKI homology domain?
- What is the degradation signal (degron) and how do hotspot mutations disrupt it?
- How does stabilized SETBP1 protein activate SET → PP2A inhibition?
- Compare G870S with other hotspot mutations (D868N, I871T, etc.)

## 3. SETBP1 in Atypical CML and MDS/MPN
- Piazza et al. 2018 — SETBP1 in atypical CML
- What is the prevalence of SETBP1 mutations in aCML vs other MDS/MPN overlap syndromes?
- How does SETBP1 status affect diagnostic classification?

## 4. Recent Advances (2024-2026)
- Any new publications on SETBP1 in myeloid disease from 2024 onward
- New mechanistic insights, structural studies, or therapeutic approaches
- SETBP1 as a potential therapeutic target

## 5. Prognostic Impact
- What is the prognostic significance of SETBP1 mutations?
- Does variant allele frequency (VAF) matter?
- How does SETBP1 interact with co-occurring mutations (ASXL1, CBL, etc.)?

Cite specific papers with author, year, and journal where possible.""",
    },
    {
        "id": "dnmt3a_idh2_review",
        "title": "DNMT3A R882H + IDH2 R140Q Co-occurrence",
        "prompt": """Provide a detailed literature review on the co-occurrence of DNMT3A R882H and IDH2 R140Q in myeloid malignancies.

## 1. Individual Mutation Biology
- DNMT3A R882H: dominant-negative effect on DNA methylation, Ley et al. 2010 (NEJM)
  - What fraction of DNMT3A mutations are R882H?
  - How does R882H disrupt the methyltransferase tetramer?
  - Effect on global DNA hypomethylation
- IDH2 R140Q: neomorphic enzyme producing 2-hydroxyglutarate (2-HG)
  - How does 2-HG inhibit TET2 and other alpha-KG-dependent dioxygenases?
  - Effect on histone demethylation and differentiation block

## 2. Epigenetic Synergy
- How do DNMT3A loss-of-function and IDH2 gain-of-function cooperate?
- Loss of DNA methylation (DNMT3A) + block of DNA demethylation (IDH2→TET2 inhibition)
- Does this create a specific epigenetic state that promotes leukemogenesis?
- Relevant mouse model studies

## 3. Key Publications
- Papaemmanuil et al. 2016 (NEJM) — genomic classification of AML
  - How common is DNMT3A+IDH2 co-occurrence?
  - Which prognostic group does this fall into?
- Ley et al. 2013 — DNMT3A in AML
- Any studies specifically examining DNMT3A+IDH2 co-mutation outcomes

## 4. Treatment Response Implications
- Does DNMT3A R882H affect response to IDH2 inhibitors (enasidenib)?
- Impact on hypomethylating agent (HMA) response
- Impact on venetoclax-based regimens
- Any clinical trial data on DNMT3A+IDH2 combination

## 5. Clonal Architecture
- Is DNMT3A typically an early/founding mutation?
- What does co-occurrence with IDH2 tell us about clonal evolution?
- CHIP implications of DNMT3A R882H

Cite specific papers with author, year, and journal.""",
    },
    {
        "id": "ptpn11_review",
        "title": "PTPN11 E76Q in Myeloid Disease",
        "prompt": """Provide a comprehensive review of PTPN11 E76Q (SHP2) mutations in myeloid malignancies.

## 1. SHP2 Biology
- What is the normal function of SHP2 (encoded by PTPN11)?
- How does SHP2 regulate RAS/MAPK signaling?
- Crystal structure: how does the autoinhibited vs active conformation work?
- Where does E76Q sit in the N-SH2 domain and how does it disrupt autoinhibition?

## 2. PTPN11 in Myeloid Malignancies
- Frequency of PTPN11 mutations in AML, MDS, JMML, MDS/MPN
- E76Q specifically — is it one of the most common PTPN11 hotspot mutations?
- How does gain-of-function SHP2 drive myeloid proliferation?
- RAS pathway activation without RAS mutation itself

## 3. PTPN11 + Co-occurring Mutations
- How does PTPN11 cooperate with SETBP1, DNMT3A, IDH2?
- Is PTPN11 typically early or late in clonal evolution?
- Mutual exclusivity with RAS mutations (NRAS, KRAS)?

## 4. SHP2 as a Therapeutic Target
- SHP2 inhibitor drug candidates: TNO155, RMC-4630, JAB-3312, ERAS-601, BBP-398
- Current clinical trial results (2024-2026) for SHP2 inhibitors
- Combination strategies: SHP2i + MEKi, SHP2i + targeted agents
- Any trials specifically in myeloid malignancies (vs solid tumors)?

## 5. Prognostic Significance
- What is the prognostic impact of PTPN11 mutations in AML/MDS?
- Does PTPN11 mutation status affect treatment decisions?
- Any data on PTPN11 as a measurable residual disease (MRD) marker?

Cite specific papers with author, year, and journal.""",
    },
    {
        "id": "monosomy7_review",
        "title": "Monosomy 7 in MDS and AML",
        "prompt": """Provide a thorough review of monosomy 7 (−7) and del(7q) in myeloid malignancies.

## 1. Cytogenetic Significance
- How common is monosomy 7 in MDS, AML, and MDS/MPN?
- Difference between monosomy 7 (complete loss) vs del(7q) (partial deletion)
- Is monosomy 7 a primary or secondary cytogenetic abnormality?
- Association with therapy-related MDS/AML

## 2. Critical Tumor Suppressor Genes on Chromosome 7
- EZH2 (7q36.1) — PRC2 component, role in myeloid disease
- CUX1 (7q22.1) — transcription factor, haploinsufficiency effects
- MLL3/KMT2C (7q36.1) — histone methyltransferase
- SAMD9/SAMD9L (7q21.2) — role in inherited predisposition syndromes
- Other candidate genes: are there additional TSGs being investigated?

## 3. Monosomy 7 + Specific Mutations
- Association with SETBP1 mutations
- Association with ASXL1 mutations
- How does monosomy 7 cooperate with signaling mutations (RAS pathway, PTPN11)?
- Is there a specific mutational signature associated with −7?

## 4. Prognostic Impact
- IPSS-R and IPSS-M scoring: where does −7 fall?
- ELN 2022 risk classification for AML with −7
- Complex karyotype vs isolated monosomy 7
- Impact on overall survival and treatment response

## 5. Treatment Considerations
- Does monosomy 7 affect response to HMAs (azacitidine, decitabine)?
- Impact on venetoclax-based regimen efficacy
- Is allogeneic stem cell transplant recommended?
- Any emerging therapies specifically targeting −7 biology?

Cite specific papers with author, year, and journal.""",
    },
    {
        "id": "combination_analysis",
        "title": "Complete 4-Gene Combination Analysis",
        "prompt": """Perform an in-depth synthesis analyzing the complete mutation profile: SETBP1 G870S + DNMT3A R882H + IDH2 R140Q + PTPN11 E76Q with monosomy 7.

## 1. Has This Exact Combination Been Reported?
- Search the literature thoroughly: has ANY publication reported a patient with all four of these mutations together?
- If not the exact combination, what is the closest reported case?
- Check large genomic studies: Papaemmanuil et al. 2016, Tyner et al. 2018, AACR Project GENIE data

## 2. Pairwise Co-occurrence Patterns
- Which pairs of these mutations are commonly co-occurring?
  - SETBP1 + monosomy 7 (known association)
  - DNMT3A + IDH2 (known association)
  - PTPN11 + SETBP1 (both activate proliferative pathways)
- Which pairs are unusual or rarely reported together?
- What does the pairwise pattern tell us?

## 3. Pathway Convergence Analysis
- Map each mutation to its pathway:
  - SETBP1 G870S → PP2A inhibition → proliferation
  - DNMT3A R882H → DNA methylation loss → epigenetic deregulation
  - IDH2 R140Q → 2-HG → differentiation block
  - PTPN11 E76Q → RAS/MAPK activation → proliferation
  - Monosomy 7 → loss of multiple tumor suppressors
- How do these pathways converge to drive disease?
- Is there a unifying disease model?

## 4. Disease Classification Implications
- What diagnosis does this profile most likely represent?
  - aCML (atypical CML) — due to SETBP1?
  - MDS/MPN overlap — due to −7 + proliferative mutations?
  - AML with myelodysplasia-related changes?
- WHO 2022 and ICC 2022 classification considerations

## 5. Clonal Architecture Hypothesis
- What is the likely order of mutation acquisition?
- Which mutation is likely the founder? (DNMT3A often earliest)
- How might clonal evolution have proceeded?

## 6. Literature Gap
- What aspects of this combination are NOT well-studied?
- What research questions does this case raise?

Be as specific as possible about publications and evidence.""",
    },
    {
        "id": "treatment_landscape_2026",
        "title": "Treatment Landscape 2026",
        "prompt": """Provide a comprehensive review of the current (2025-2026) treatment landscape relevant to a patient with SETBP1 G870S, DNMT3A R882H, IDH2 R140Q, PTPN11 E76Q, and monosomy 7.

## 1. IDH2-Targeted Therapy
- Enasidenib (AG-221/Idhifa): current approval status, efficacy data
- Olutasidenib (FT-2102): approval status, data in IDH2-mutant disease
- Next-generation IDH inhibitors in development
- Does DNMT3A co-mutation affect IDH2 inhibitor response?
- Resistance mechanisms to IDH2 inhibitors

## 2. SHP2 Inhibitor Clinical Trials
- TNO155 (Novartis): latest trial results
- RMC-4630 (Revolution Medicines): latest data
- JAB-3312 (Jacobio): current status
- ERAS-601 (Erasca): current status
- BBP-398 (BridgeBio): current status
- Any of these being tested in myeloid malignancies specifically?

## 3. Combination Strategies for Complex AML
- Venetoclax + azacitidine: baseline data, any data with this mutation profile?
- Venetoclax + IDH2 inhibitor combinations
- Triple combinations under investigation
- Does monosomy 7 predict poor response to ven/aza?

## 4. Emerging Targeted Approaches
- PP2A activators (relevant to SETBP1 → PP2A inhibition)
- MEK inhibitors (relevant to PTPN11 → RAS activation)
- Menin inhibitors: any relevance to this profile?
- Immunotherapy approaches: BiTEs, ADCs, checkpoint inhibitors in AML

## 5. Transplant Considerations
- Is allo-HSCT recommended for this risk profile?
- Optimal timing of transplant
- Pre-transplant bridging therapies
- Post-transplant maintenance options (IDH2 inhibitor maintenance?)

## 6. Clinical Trials to Consider
- Search for currently recruiting trials (2025-2026) that would accept a patient with this profile
- Trials combining IDH2 inhibitors with other targeted agents
- Trials for high-risk MDS/MPN or AML with adverse cytogenetics
- Geographic availability (Europe/US)

Be specific about drug names, trial identifiers (NCT numbers), and recent data presentations (ASH, EHA, ASCO).""",
    },
]


# ---------------------------------------------------------------------------
# Run analyses
# ---------------------------------------------------------------------------
def run_analysis(analysis: dict) -> str:
    """Send a single analysis prompt to Gemini with Google Search grounding."""
    print(f"\n{'='*70}")
    print(f"  Running: {analysis['title']}")
    print(f"{'='*70}")

    response = client.models.generate_content(
        model=MODEL,
        contents=analysis["prompt"],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[GOOGLE_SEARCH_TOOL],
            temperature=0.3,        # lower temp for factual accuracy
            max_output_tokens=16384,
        ),
    )

    text = response.text
    print(f"  -> Received {len(text):,} characters")
    return text


def format_output(analysis: dict, content: str) -> str:
    """Wrap the Gemini response in a structured markdown document."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"# {analysis['title']}\n\n"
        f"> **AI Literature Synthesis** | Model: {MODEL} | Google Search Grounding: enabled\n"
        f"> Generated: {now}\n"
        f"> Patient profile: SETBP1 G870S, DNMT3A R882H, IDH2 R140Q, PTPN11 E76Q, monosomy 7\n\n"
        f"---\n\n"
    )
    return header + content + "\n"


def main():
    print(f"\nLiterature Synthesis — Gemini {MODEL}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    start = time.time()

    results = {}
    for analysis in ANALYSES:
        try:
            content = run_analysis(analysis)
            md = format_output(analysis, content)
            out_path = OUTPUT_DIR / f"{analysis['id']}.md"
            out_path.write_text(md, encoding="utf-8")
            results[analysis["id"]] = "OK"
            print(f"  -> Saved: {out_path.name}")
        except Exception as exc:
            print(f"  !! ERROR on {analysis['id']}: {exc}")
            results[analysis["id"]] = f"FAILED: {exc}"
            # Save error info
            err_path = OUTPUT_DIR / f"{analysis['id']}_ERROR.txt"
            err_path.write_text(f"Error: {exc}\n", encoding="utf-8")

    elapsed = time.time() - start

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SYNTHESIS COMPLETE — {elapsed:.1f}s")
    print(f"{'='*70}")
    for aid, status in results.items():
        icon = "OK" if status == "OK" else "FAIL"
        print(f"  [{icon}] {aid}")
    print(f"\nResults saved to: {OUTPUT_DIR}")

    # Save run log
    log = OUTPUT_DIR / "_run_log.txt"
    log.write_text(
        f"Literature Synthesis Run Log\n"
        f"Date: {datetime.datetime.now().isoformat()}\n"
        f"Model: {MODEL}\n"
        f"Grounding: Google Search\n"
        f"Duration: {elapsed:.1f}s\n\n"
        + "\n".join(f"  {aid}: {status}" for aid, status in results.items())
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
