# Six-Axis AI Pipeline for Myeloid Variant Interpretation

An open-source computational pipeline integrating six independent evidence axes for somatic variant classification in myeloid malignancies.

Presented at **ISMB 2026** (VarI Track, Washington DC, July 2026).

## Pipeline Architecture

The pipeline classifies somatic variants through six analytically independent evidence axes:

1. **Protein Language Models** (ESM-2, 650M parameters)
2. **Structure-Aware Deep Learning** (AlphaMissense, PrimateAI-3D)
3. **Evolutionary Conservation** (EVE, SIFT, CADD, REVEL)
4. **Structural Prediction & Molecular Docking** (ESMFold, Chai-1, Boltz-1, AutoDock Vina, DiffDock)
5. **Population Frequency** (gnomAD v4)
6. **Functional Experimental Evidence** (deep mutational scanning, biochemical assays)

Evidence is aggregated using the ACMG/AMP Bayesian point system (Tavtigian et al. 2018). Classification robustness is verified by systematic axis-level ablation analysis.

## Key Results

- **92.8% ClinVar concordance** across 40 independent GENIE myeloid profiles (284 variants)
- Reclassified a novel EZH2 V662A variant from VUS to Pathogenic (14 ACMG points)
- Identified an ESM-2 gain-of-function blind spot for myeloid driver mutations
- Loss-of-function classification of EZH2 V662A concordant with the global withdrawal of tazemetostat (March 2026)

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. GPU recommended for ESM-2 scoring but not required.

## Usage

```bash
# Full pipeline (steps 1-13)
python mutation_profile/scripts/run_all.py

# Skip GPU-dependent steps
python mutation_profile/scripts/run_all.py --skip-esm2

# Dry run
python mutation_profile/scripts/run_all.py --dry-run
```

## Data Requirements

- AACR GENIE v19.0 (requires Data Use Agreement from Synapse: syn7222066)
- gnomAD v4 (accessed via API)
- OncoKB, CIViC, ClinGen (accessed via public APIs)

Raw GENIE data is not included due to DUA restrictions.

## Citation

> Roine H. From sequence to structure to drug binding: a six-axis AI pipeline for myeloid variant interpretation. ISMB 2026, VarI Track. The Educational Equality Institute.

## License

MIT License. See [LICENSE](LICENSE) for details.
