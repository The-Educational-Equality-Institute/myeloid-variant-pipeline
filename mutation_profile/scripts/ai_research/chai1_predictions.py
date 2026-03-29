#!/usr/bin/env python3
"""
chai1_predictions.py -- Prepare and run Chai-1 multi-chain protein complex
predictions for the patient's 5 driver mutations (6 complexes total).

Generates Chai-1 input files (FASTA + ligand configs) for local inference
via the chai-lab Python API and/or the Chai Discovery web platform at
lab.chaidiscovery.com. Adds the EZH2 V662A / PRC2 complex not present
in the original alphafold3_inputs.py.

6 Complexes:
  1. DNMT3A R882H catalytic domain + CpG DNA + SAM cofactor
  2. SETBP1 G870S SKI domain + SET/I2PP2A protein
  3. IDH2 R140Q homodimer + NADP+ + alpha-ketoglutarate
  4. PTPN11 E76Q full-length (autoinhibition disruption)
  5. PTPN11 E76Q + TNO155 allosteric inhibitor
  6. EZH2 V662A in PRC2 complex (EZH2 + EED + SUZ12 VEFS domain)

For each complex, both wildtype and mutant versions are generated (12 jobs).

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/chai1_predictions.py [--run] [--analyze]

Flags:
    --run      Attempt local Chai-1 inference (requires chai-lab + GPU)
    --analyze  Run PLIP contact comparison on existing output PDBs
    (default)  Generate input files only

Outputs:
    mutation_profile/results/ai_research/chai1_predictions/
      - fasta/          FASTA files for each complex (WT + mutant)
      - configs/        JSON configs with ligand SMILES and run parameters
      - outputs/        Prediction results (when --run is used)
      - analysis/       PLIP contact comparison reports (when --analyze is used)
      - REPORT.md       Summary of inputs and submission instructions
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests

# -- Paths -------------------------------------------------------------------
PROJECT = Path("Path(__file__).resolve().parents[2]")
OUTPUT_DIR = PROJECT / "mutation_profile" / "results" / "ai_research" / "chai1_predictions"
FASTA_DIR = OUTPUT_DIR / "fasta"
CONFIG_DIR = OUTPUT_DIR / "configs"
RESULTS_DIR = OUTPUT_DIR / "outputs"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

for d in [OUTPUT_DIR, FASTA_DIR, CONFIG_DIR, RESULTS_DIR, ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "chai1_predictions.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# PROTEIN SEQUENCES
# =============================================================================

def fetch_uniprot_sequence(uniprot_id: str) -> str:
    """Fetch canonical protein sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    log.info(f"Fetching {uniprot_id} from UniProt...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    log.info(f"  {uniprot_id}: {len(seq)} residues")
    return seq


# Pre-verified full-length sequences (from alphafold3_inputs.py pipeline)
PTPN11_WT_SEQ = (
    "MTSRRWFHPNITGVEAENLLLTRGVDGSFLARPSKSNPGDFTLSVRRNGAVTHIKIQNTGD"
    "YYDLYGGEKFATLAELVQYYMEHHGQLKEKNGDVIELKYPLNCADPTSERWFHGHLSGKEA"
    "EKLLTEKGKHGSFLVRESQSHPGDFVLSVRTGDDKGESNDGKSKVTHVMIRCQELKYDVGG"
    "GERFDSLTDLVEHYKKNPMVETLGTVLQLKQPLNTTRINAAEIESRVRELSKLAETTDKVK"
    "QGFWEEFETLQQQECKLLYSRKEGQRQENKNKNRYKNILPFDHTRVVLHDGDPNEPVSDYI"
    "NANIIMPEFETKCNNSKPKKSYIATQGCLQNTVNDFWRMVFQENSRVIVMTTKEVERGKSK"
    "CVKYWPDEYALKEYGVMRVRNVKESAAHDYTLRELKLSKVGQGNTERTVWQYHFRTWPDHG"
    "VPSDPGGVLDFLEEVHHKQESIMDAGPVVVHCSAGIGRTGTFIVIDILIDIIREKGVDCDID"
    "VPKTIQMVRSQRSGMVQTEAQYRFIYMAVQHYIETLQRRIEEEQKSKRKGHEYTNIKYSLAD"
    "QTSGDQSPLPPCTPTPPCAEMREDSARVYENVGLMQQQKSFR"
)

IDH2_WT_SEQ = (
    "MAGYLRVVRSLCRASGSRPAWAPAALTAPTSQEQPRRHYADKRIKVAKPVVEMDGDEMTRII"
    "WQFIKEKLILPHVDIQLKYFDLGLPNRDQTDDQVTIDSALATQKYSVAVKCATITPDEARVEE"
    "FKLKKMWKSPNGTIRNILGGTVFREPIICKNIPRLVPGWTKPITIGRHAHGDQYKATDFVAD"
    "RAGTFKMVFTPKDGSGVKEWEVYNFPAGGVGMGMYNTDESISGFAHSCFQYAIQKKWPLYMS"
    "TKNTILKAYDGRFKDIFQEIFDKHYKTDFDKNKIWYEHRLIDDMVAQVLKSSGGFVWACKNY"
    "DGDVQSDILAQGFGSLGLMTSVLVCPDGKTIEAEAAHGTVTRHYREHQKGRPTSTNPIASI"
    "FAWTRGLEHRGKLDGNQDLIRFAQMLEKVCVETVESGAMTKDLAGCIHGLSNVKLNEHFLNT"
    "TDFLDTIKSNLDRALGRQ"
)

# Verify known positions
assert PTPN11_WT_SEQ[75] == "E", f"PTPN11 pos 76: expected E, got {PTPN11_WT_SEQ[75]}"
assert len(PTPN11_WT_SEQ) == 593, f"PTPN11: expected 593 aa, got {len(PTPN11_WT_SEQ)}"
assert IDH2_WT_SEQ[139] == "R", f"IDH2 pos 140: expected R, got {IDH2_WT_SEQ[139]}"
assert len(IDH2_WT_SEQ) == 452, f"IDH2: expected 452 aa, got {len(IDH2_WT_SEQ)}"

# Apply mutations
PTPN11_E76Q_SEQ = PTPN11_WT_SEQ[:75] + "Q" + PTPN11_WT_SEQ[76:]
IDH2_R140Q_SEQ = IDH2_WT_SEQ[:139] + "Q" + IDH2_WT_SEQ[140:]


# -- Small molecule SMILES (PubChem-verified) --------------------------------
SMILES = {
    # TNO155 - SHP2 allosteric inhibitor (Novartis, Phase 1/2)
    "TNO155": "CC1=CC2=C(C=C1)N(C(=O)C3=CC(=CC=C3F)C(F)(F)F)CC(C2)NS(=O)(=O)C4=CC=CC=C4",
    # S-adenosylmethionine (SAM) - methyl donor cofactor for DNMT3A
    "SAM": "C[S+](CC[C@@H](N)C(=O)[O-])C[C@@H]1OC(n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
    # NADP+ - cofactor for IDH2
    "NADP+": (
        "NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]2OC("
        "[C@H](OP(=O)([O-])[O-])[C@@H]2O)n2cnc3c(N)ncnc32)[C@@H](O)[C@H]1O"
    ),
    # Alpha-ketoglutarate (2-oxoglutarate) - IDH2 substrate
    "AKG": "OC(=O)CCC(=O)C(=O)O",
    # S-adenosyl-L-homocysteine (SAH) - product/cofactor analog for EZH2
    "SAH": "C(CC(C(=O)O)N)SCC1C(C(C(O1)N2C=NC3=C2N=CN=C3N)O)O",
}

# -- DNA sequences for Complex 1 --------------------------------------------
DNA_SENSE = "AACGCGAACGCG"
DNA_ANTISENSE = "CGCGTTCGCGTT"


# =============================================================================
# UNIPROT IDs
# =============================================================================
UNIPROT_IDS = {
    "DNMT3A": "Q9Y6K1",   # 912 aa
    "SETBP1": "Q9Y6X0",   # 1596 aa
    "SET": "Q01105",       # 290 aa (I2PP2A / PP2A inhibitor)
    "EZH2": "Q15910",     # 746 aa (PRC2 catalytic subunit)
    "EED": "O75530",      # 441 aa (PRC2 WD40 scaffold)
    "SUZ12": "Q15022",    # 739 aa (PRC2 zinc finger / VEFS domain)
}


# =============================================================================
# SEQUENCE FETCHING
# =============================================================================

def get_all_sequences() -> dict:
    """Fetch and prepare all protein sequences for 6 complexes."""
    sequences = {}

    log.info("=" * 60)
    log.info("Fetching protein sequences from UniProt")
    log.info("=" * 60)

    # DNMT3A (Q9Y6K1) - catalytic domain residues 600-912
    dnmt3a_full = fetch_uniprot_sequence(UNIPROT_IDS["DNMT3A"])
    assert dnmt3a_full[881] == "R", f"DNMT3A pos 882: expected R, got {dnmt3a_full[881]}"
    sequences["DNMT3A_full"] = dnmt3a_full
    sequences["DNMT3A_cat_WT"] = dnmt3a_full[599:]
    sequences["DNMT3A_cat_R882H"] = dnmt3a_full[599:881] + "H" + dnmt3a_full[882:]
    log.info(f"  DNMT3A catalytic domain: {len(sequences['DNMT3A_cat_WT'])} residues (600-{len(dnmt3a_full)})")

    # SETBP1 (Q9Y6X0) - SKI homology domain residues 850-930
    setbp1_full = fetch_uniprot_sequence(UNIPROT_IDS["SETBP1"])
    assert setbp1_full[869] == "G", f"SETBP1 pos 870: expected G, got {setbp1_full[869]}"
    sequences["SETBP1_full"] = setbp1_full
    sequences["SETBP1_SKI_WT"] = setbp1_full[849:930]
    sequences["SETBP1_SKI_G870S"] = setbp1_full[849:869] + "S" + setbp1_full[870:930]
    log.info(f"  SETBP1 SKI domain: {len(sequences['SETBP1_SKI_WT'])} residues (850-930)")

    # SET/I2PP2A (Q01105) - full-length PP2A inhibitor
    set_full = fetch_uniprot_sequence(UNIPROT_IDS["SET"])
    sequences["SET_full"] = set_full
    log.info(f"  SET protein: {len(set_full)} residues (full length)")

    # EZH2 (Q15910) - PRC2 catalytic subunit, 746 aa (UniProt canonical)
    # Clinical annotation V662A uses the 751-aa ENST00000320356 protein (5 extra N-terminal residues).
    # UniProt Q15910 (746 aa) position = clinical position - 5, so V662 -> V657 (0-indexed: 656).
    ezh2_full = fetch_uniprot_sequence(UNIPROT_IDS["EZH2"])
    assert ezh2_full[656] == "V", f"EZH2 UniProt pos 657 (clinical V662): expected V, got {ezh2_full[656]}"
    sequences["EZH2_full_WT"] = ezh2_full
    sequences["EZH2_full_V662A"] = ezh2_full[:656] + "A" + ezh2_full[657:]
    log.info(f"  EZH2: {len(ezh2_full)} residues (clinical V662A = UniProt V657A)")

    # EED (O75530) - PRC2 WD40 scaffold
    eed_full = fetch_uniprot_sequence(UNIPROT_IDS["EED"])
    sequences["EED_full"] = eed_full
    log.info(f"  EED: {len(eed_full)} residues (full length)")

    # SUZ12 (Q15022) - PRC2 zinc finger, VEFS domain ~530-695
    suz12_full = fetch_uniprot_sequence(UNIPROT_IDS["SUZ12"])
    sequences["SUZ12_full"] = suz12_full
    # VEFS domain (residues 530-695) - the minimal region needed for PRC2 core
    sequences["SUZ12_VEFS"] = suz12_full[529:695]
    log.info(f"  SUZ12: {len(suz12_full)} residues full, VEFS domain: {len(sequences['SUZ12_VEFS'])} residues (530-695)")

    # Pre-verified PTPN11 and IDH2
    sequences["PTPN11_WT"] = PTPN11_WT_SEQ
    sequences["PTPN11_E76Q"] = PTPN11_E76Q_SEQ
    sequences["IDH2_WT"] = IDH2_WT_SEQ
    sequences["IDH2_R140Q"] = IDH2_R140Q_SEQ
    log.info(f"  PTPN11: {len(PTPN11_WT_SEQ)} residues (pre-verified)")
    log.info(f"  IDH2: {len(IDH2_WT_SEQ)} residues (pre-verified)")

    return sequences


# =============================================================================
# CHAI-1 INPUT GENERATION
# =============================================================================

def write_fasta(name: str, chains: list[dict]) -> Path:
    """Write a Chai-1 compatible FASTA file.

    chains: list of dicts with keys: id, type (protein/dna/rna), sequence
    """
    path = FASTA_DIR / f"{name}.fasta"
    lines = []
    for chain in chains:
        entity_type = chain["type"]
        chain_id = chain["id"]
        seq = chain["sequence"]
        lines.append(f">chain_{chain_id}|entity_type={entity_type}")
        for i in range(0, len(seq), 80):
            lines.append(seq[i : i + 80])
    path.write_text("\n".join(lines) + "\n")
    log.info(f"  FASTA: {path.name}")
    return path


def write_config(name: str, description: str, ligands: list[dict] | None = None,
                 num_trunk_recycles: int = 3, num_diffn_timesteps: int = 200) -> Path:
    """Write a Chai-1 config JSON with ligand SMILES and run parameters."""
    config = {
        "name": name,
        "description": description,
        "ligands": ligands or [],
        "num_trunk_recycles": num_trunk_recycles,
        "num_diffn_timesteps": num_diffn_timesteps,
        "seed": 42,
    }
    path = CONFIG_DIR / f"{name}_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    log.info(f"  Config: {path.name}")
    return path


# =============================================================================
# COMPLEX DEFINITIONS (6 complexes x 2 variants = 12 jobs)
# =============================================================================

def generate_complex_1(seqs: dict) -> list[dict]:
    """Complex 1: DNMT3A R882H catalytic domain + CpG DNA + SAM cofactor.

    R882 directly contacts the DNA backbone. R882H reduces CpG methylation
    by ~80% and acts dominant-negative on WT DNMT3A. The protein-DNA-ligand
    complex reveals binding geometry changes at the mutation site.
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 1: DNMT3A R882H + CpG DNA + SAM")
    log.info("=" * 60)

    jobs = []
    for variant, label, seq in [
        ("R882H", "mutant", seqs["DNMT3A_cat_R882H"]),
        ("WT", "wildtype", seqs["DNMT3A_cat_WT"]),
    ]:
        name = f"complex1_DNMT3A_{variant}_DNA_SAM"
        write_fasta(name, [
            {"id": "A", "type": "protein", "sequence": seq},
            {"id": "B", "type": "dna", "sequence": DNA_SENSE},
            {"id": "C", "type": "dna", "sequence": DNA_ANTISENSE},
        ])
        write_config(
            name,
            f"DNMT3A catalytic domain {variant} + CpG DNA duplex + SAM cofactor",
            ligands=[{"name": "SAM", "smiles": SMILES["SAM"]}],
        )
        jobs.append({"name": name, "complex": 1, "variant": label})

    return jobs


def generate_complex_2(seqs: dict) -> list[dict]:
    """Complex 2: SETBP1 G870S SKI domain + SET/I2PP2A protein.

    G870S disrupts the degron motif recognized by beta-TrCP, stabilizing
    SETBP1, which in turn stabilizes SET (PP2A inhibitor). Models the
    protein-protein binding interface at the SKI domain.
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 2: SETBP1 G870S SKI domain + SET protein")
    log.info("=" * 60)

    jobs = []
    for variant, label, seq in [
        ("G870S", "mutant", seqs["SETBP1_SKI_G870S"]),
        ("WT", "wildtype", seqs["SETBP1_SKI_WT"]),
    ]:
        name = f"complex2_SETBP1_{variant}_SET"
        write_fasta(name, [
            {"id": "A", "type": "protein", "sequence": seq},
            {"id": "B", "type": "protein", "sequence": seqs["SET_full"]},
        ])
        write_config(name, f"SETBP1 SKI domain {variant} + SET/I2PP2A protein")
        jobs.append({"name": name, "complex": 2, "variant": label})

    return jobs


def generate_complex_3(seqs: dict) -> list[dict]:
    """Complex 3: IDH2 R140Q homodimer + NADP+ + alpha-ketoglutarate.

    IDH2 functions as a homodimer. R140Q is neomorphic: converts AKG to
    the oncometabolite 2-HG. The homodimer context is critical because
    enasidenib binds the dimer interface.
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 3: IDH2 R140Q homodimer + NADP+ + AKG")
    log.info("=" * 60)

    jobs = []
    for variant, label, seq in [
        ("R140Q", "mutant", seqs["IDH2_R140Q"]),
        ("WT", "wildtype", seqs["IDH2_WT"]),
    ]:
        name = f"complex3_IDH2_{variant}_dimer_NADP_AKG"
        write_fasta(name, [
            {"id": "A", "type": "protein", "sequence": seq},
            {"id": "B", "type": "protein", "sequence": seq},
        ])
        write_config(
            name,
            f"IDH2 {variant} homodimer + NADP+ + alpha-ketoglutarate",
            ligands=[
                {"name": "NADP+", "smiles": SMILES["NADP+"]},
                {"name": "AKG", "smiles": SMILES["AKG"]},
            ],
        )
        jobs.append({"name": name, "complex": 3, "variant": label})

    return jobs


def generate_complex_4(seqs: dict) -> list[dict]:
    """Complex 4: PTPN11 E76Q full-length (autoinhibition disruption).

    SHP2 domains: N-SH2 (1-105), C-SH2 (112-216), PTP (246-527).
    E76Q in N-SH2 disrupts the D61/E76 hydrogen bond network at the
    N-SH2/PTP interface, shifting equilibrium toward the open (active)
    conformation. Single-chain conformational prediction.
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 4: PTPN11 E76Q full-length (autoinhibition)")
    log.info("=" * 60)

    jobs = []
    for variant, label, seq in [
        ("E76Q", "mutant", seqs["PTPN11_E76Q"]),
        ("WT", "wildtype", seqs["PTPN11_WT"]),
    ]:
        name = f"complex4_PTPN11_{variant}_fulllength"
        write_fasta(name, [
            {"id": "A", "type": "protein", "sequence": seq},
        ])
        write_config(name, f"PTPN11/SHP2 full-length {variant} (autoinhibition)")
        jobs.append({"name": name, "complex": 4, "variant": label})

    return jobs


def generate_complex_5(seqs: dict) -> list[dict]:
    """Complex 5: PTPN11 E76Q + TNO155 allosteric SHP2 inhibitor.

    TNO155 stabilizes the autoinhibited (closed) conformation by binding
    at the N-SH2/PTP interface. E76Q disrupts this interface. The key
    question: can the inhibitor overcome constitutive activation?

    Prior docking: AutoDock Vina = -9.34 kcal/mol, DiffDock = -7.49
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 5: PTPN11 E76Q + TNO155")
    log.info("=" * 60)

    jobs = []
    for variant, label, seq in [
        ("E76Q", "mutant", seqs["PTPN11_E76Q"]),
        ("WT", "wildtype", seqs["PTPN11_WT"]),
    ]:
        name = f"complex5_PTPN11_{variant}_TNO155"
        write_fasta(name, [
            {"id": "A", "type": "protein", "sequence": seq},
        ])
        write_config(
            name,
            f"PTPN11/SHP2 {variant} + TNO155 allosteric SHP2 inhibitor",
            ligands=[{"name": "TNO155", "smiles": SMILES["TNO155"]}],
        )
        jobs.append({"name": name, "complex": 5, "variant": label})

    return jobs


def generate_complex_6(seqs: dict) -> list[dict]:
    """Complex 6: EZH2 V662A in PRC2 core complex (EZH2 + EED + SUZ12 VEFS).

    EZH2 is the catalytic subunit of Polycomb Repressive Complex 2 (PRC2).
    V662 is in the SET domain (catalytic, residues ~613-726) responsible for
    H3K27 trimethylation. V662A disrupts SET domain packing, impairing
    methyltransferase activity.

    With monosomy 7 deleting the other EZH2 allele, V662A achieves effective
    biallelic PRC2 inactivation - a recognized tumor suppressor mechanism
    in myeloid malignancies.

    The minimal PRC2 core (EZH2 + EED + SUZ12-VEFS) is required because:
    - EZH2 alone has NO methyltransferase activity
    - EED allosterically activates EZH2 via H3K27me3 binding
    - SUZ12-VEFS stabilizes the EZH2-EED interface

    SAH (S-adenosyl-L-homocysteine) is included as the product analog of
    the SAM methyl donor, marking the active site.

    Total residues: ~746 + 441 + 166 = ~1353 (well within Chai-1 limits).
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 6: EZH2 V662A in PRC2 (EZH2 + EED + SUZ12-VEFS)")
    log.info("=" * 60)

    jobs = []
    for variant, label, seq in [
        ("V662A", "mutant", seqs["EZH2_full_V662A"]),
        ("WT", "wildtype", seqs["EZH2_full_WT"]),
    ]:
        name = f"complex6_EZH2_{variant}_PRC2"
        write_fasta(name, [
            {"id": "A", "type": "protein", "sequence": seq},
            {"id": "B", "type": "protein", "sequence": seqs["EED_full"]},
            {"id": "C", "type": "protein", "sequence": seqs["SUZ12_VEFS"]},
        ])
        write_config(
            name,
            f"PRC2 core: EZH2 {variant} + EED + SUZ12-VEFS + SAH",
            ligands=[{"name": "SAH", "smiles": SMILES["SAH"]}],
        )
        jobs.append({"name": name, "complex": 6, "variant": label})

    return jobs


# =============================================================================
# LOCAL CHAI-1 INFERENCE
# =============================================================================

def check_chai_lab_installed() -> bool:
    """Check if chai-lab is installed and importable."""
    try:
        import chai_lab  # noqa: F401
        return True
    except ImportError:
        return False


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def run_local_inference(jobs: list[dict]) -> list[dict]:
    """Run Chai-1 inference locally using the chai-lab Python API.

    Requires: pip install chai-lab, CUDA GPU with >= 16 GB VRAM.
    """
    if not check_chai_lab_installed():
        log.error("chai-lab not installed. Install with: pip install chai-lab")
        log.error("Alternatively, upload FASTA files to lab.chaidiscovery.com")
        return []

    if not check_gpu_available():
        log.error("No CUDA GPU detected. Chai-1 requires GPU for inference.")
        log.error("Options: (1) Vast.ai A100, (2) lab.chaidiscovery.com web API")
        return []

    from chai_lab.chai1 import run_inference

    results = []
    for job in jobs:
        name = job["name"]
        fasta_path = FASTA_DIR / f"{name}.fasta"
        config_path = CONFIG_DIR / f"{name}_config.json"
        output_dir = RESULTS_DIR / name

        if not fasta_path.exists():
            log.warning(f"  Skipping {name}: FASTA file not found")
            continue

        config = json.loads(config_path.read_text())
        ligand_smiles = [lig["smiles"] for lig in config.get("ligands", [])]

        log.info(f"  Running inference: {name}")
        log.info(f"    FASTA: {fasta_path}")
        log.info(f"    Ligands: {len(ligand_smiles)}")
        log.info(f"    Output: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            candidates = run_inference(
                fasta_file=fasta_path,
                output_dir=output_dir,
                num_trunk_recycles=config.get("num_trunk_recycles", 3),
                num_diffn_timesteps=config.get("num_diffn_timesteps", 200),
                seed=config.get("seed", 42),
                device="cuda:0",
                use_esm_embeddings=True,
            )
            log.info(f"    Generated {len(candidates.cif_paths)} structure(s)")
            results.append({
                "name": name,
                "status": "success",
                "n_structures": len(candidates.cif_paths),
                "output_dir": str(output_dir),
            })
        except Exception as e:
            log.error(f"    FAILED: {e}")
            results.append({
                "name": name,
                "status": "error",
                "error": str(e),
            })

    return results


# =============================================================================
# PLIP CONTACT ANALYSIS
# =============================================================================

def run_plip_analysis(jobs: list[dict]) -> list[dict]:
    """Run PLIP (Protein-Ligand Interaction Profiler) on output structures.

    Compares contacts between WT and mutant for each complex.
    Requires: pip install plip (or plip command-line tool).
    """
    plip_available = shutil.which("plip") is not None
    if not plip_available:
        try:
            from plip.structure.preparation import PDBComplex  # noqa: F401
            plip_available = True
        except ImportError:
            pass

    if not plip_available:
        log.warning("PLIP not installed. Install with: pip install plip")
        log.warning("Skipping contact analysis.")
        return []

    results = []
    # Group jobs by complex number
    complexes: dict[int, dict[str, dict]] = {}
    for job in jobs:
        cn = job["complex"]
        if cn not in complexes:
            complexes[cn] = {}
        complexes[cn][job["variant"]] = job

    for cn, variants in sorted(complexes.items()):
        if "mutant" not in variants or "wildtype" not in variants:
            continue

        mut_name = variants["mutant"]["name"]
        wt_name = variants["wildtype"]["name"]

        # Look for output CIF/PDB files
        mut_dir = RESULTS_DIR / mut_name
        wt_dir = RESULTS_DIR / wt_name

        if not mut_dir.exists() or not wt_dir.exists():
            log.info(f"  Complex {cn}: output directories not found, skipping PLIP")
            continue

        mut_structures = list(mut_dir.glob("*.cif")) + list(mut_dir.glob("*.pdb"))
        wt_structures = list(wt_dir.glob("*.cif")) + list(wt_dir.glob("*.pdb"))

        if not mut_structures or not wt_structures:
            log.info(f"  Complex {cn}: no structure files found")
            continue

        log.info(f"  Complex {cn}: running PLIP comparison")
        log.info(f"    Mutant: {mut_structures[0].name}")
        log.info(f"    Wildtype: {wt_structures[0].name}")

        # Run PLIP via command line for each
        for label, struct_file in [("mutant", mut_structures[0]), ("wildtype", wt_structures[0])]:
            out_prefix = ANALYSIS_DIR / f"complex{cn}_{label}"
            cmd = [
                "plip", "-f", str(struct_file),
                "-o", str(out_prefix),
                "-x",  # XML output
                "-t",  # Text output
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
                log.info(f"    PLIP {label}: done")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                log.warning(f"    PLIP {label} failed: {e}")

        results.append({
            "complex": cn,
            "mutant": mut_name,
            "wildtype": wt_name,
            "status": "analyzed",
        })

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(seqs: dict, all_jobs: list[dict], inference_results: list[dict] | None = None) -> Path:
    """Generate REPORT.md summarizing all inputs and submission instructions."""

    # Compute residue totals per complex
    complex_info = [
        {
            "num": 1,
            "name": "DNMT3A R882H + CpG DNA + SAM",
            "gene": "DNMT3A",
            "mutation": "R882H",
            "uniprot": "Q9Y6K1",
            "chains": f"Protein ({len(seqs['DNMT3A_cat_WT'])} aa) + DNA (12+12 nt) + SAM ligand",
            "total_residues": len(seqs["DNMT3A_cat_WT"]) + 24,
            "question": "How does R882H alter DNA-binding geometry and SAM positioning?",
        },
        {
            "num": 2,
            "name": "SETBP1 G870S SKI + SET",
            "gene": "SETBP1",
            "mutation": "G870S",
            "uniprot": "Q9Y6X0",
            "chains": f"SETBP1 SKI ({len(seqs['SETBP1_SKI_WT'])} aa) + SET ({len(seqs['SET_full'])} aa)",
            "total_residues": len(seqs["SETBP1_SKI_WT"]) + len(seqs["SET_full"]),
            "question": "Does G870S alter the SETBP1-SET binding interface?",
        },
        {
            "num": 3,
            "name": "IDH2 R140Q homodimer + NADP+ + AKG",
            "gene": "IDH2",
            "mutation": "R140Q",
            "uniprot": "P48735",
            "chains": f"IDH2 dimer (2x {len(seqs['IDH2_WT'])} aa) + NADP+ + AKG",
            "total_residues": 2 * len(seqs["IDH2_WT"]),
            "question": "How does R140Q alter active site geometry for 2-HG production?",
        },
        {
            "num": 4,
            "name": "PTPN11 E76Q full-length",
            "gene": "PTPN11",
            "mutation": "E76Q",
            "uniprot": "Q06124",
            "chains": f"PTPN11 full-length ({len(seqs['PTPN11_WT'])} aa)",
            "total_residues": len(seqs["PTPN11_WT"]),
            "question": "Does E76Q shift N-SH2/PTP toward the open conformation?",
        },
        {
            "num": 5,
            "name": "PTPN11 E76Q + TNO155",
            "gene": "PTPN11",
            "mutation": "E76Q",
            "uniprot": "Q06124",
            "chains": f"PTPN11 full-length ({len(seqs['PTPN11_WT'])} aa) + TNO155",
            "total_residues": len(seqs["PTPN11_WT"]),
            "question": "Can TNO155 overcome E76Q-driven constitutive activation?",
        },
        {
            "num": 6,
            "name": "EZH2 V662A in PRC2 (EZH2 + EED + SUZ12-VEFS)",
            "gene": "EZH2",
            "mutation": "V662A",
            "uniprot": "Q15910",
            "chains": (
                f"EZH2 ({len(seqs['EZH2_full_WT'])} aa) + EED ({len(seqs['EED_full'])} aa) "
                f"+ SUZ12-VEFS ({len(seqs['SUZ12_VEFS'])} aa) + SAH"
            ),
            "total_residues": len(seqs["EZH2_full_WT"]) + len(seqs["EED_full"]) + len(seqs["SUZ12_VEFS"]),
            "question": "How does V662A alter PRC2 methyltransferase activity and complex stability?",
        },
    ]

    report_lines = [
        f"# Chai-1 Multi-Chain Complex Predictions",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Script: `mutation_profile/scripts/ai_research/chai1_predictions.py`",
        f"",
        f"## Patient Driver Mutations (5 genes, 6 complexes)",
        f"",
        f"| Gene | Mutation | VAF | UniProt | Classification |",
        f"|------|----------|-----|---------|----------------|",
        f"| DNMT3A | R882H | 39% | Q9Y6K1 | Pathogenic (hotspot) |",
        f"| SETBP1 | G870S | 34% | Q9Y6X0 | Likely pathogenic |",
        f"| IDH2 | R140Q | 2% | P48735 | Pathogenic (subclone) |",
        f"| PTPN11 | E76Q | 29% | Q06124 | Pathogenic (gain-of-function) |",
        f"| EZH2 | V662A | 59% | Q15910 | Pathogenic (computational consensus) |",
        f"",
        f"## 6 Complexes (12 total: WT + mutant each)",
        f"",
    ]

    for ci in complex_info:
        report_lines.extend([
            f"### Complex {ci['num']}: {ci['name']}",
            f"",
            f"**Biological question:** {ci['question']}",
            f"",
            f"- Gene: {ci['gene']} ({ci['uniprot']}), Mutation: {ci['mutation']}",
            f"- Chains: {ci['chains']}",
            f"- Total residues: ~{ci['total_residues']}",
            f"- Files: `complex{ci['num']}_*` (WT + mutant FASTA + config)",
            f"",
        ])

    report_lines.extend([
        f"## How to Run",
        f"",
        f"### Option A: Chai Discovery Web Platform (recommended, free)",
        f"",
        f"1. Go to https://lab.chaidiscovery.com",
        f"2. Create account or sign in",
        f"3. Click 'New prediction'",
        f"4. For each job:",
        f"   a. Upload the `.fasta` file from `fasta/`",
        f"   b. For complexes with ligands, paste the SMILES from the `_config.json` file",
        f"   c. Set number of seeds to 5 for statistical confidence",
        f"   d. Submit",
        f"5. Download results when complete (typically 5-30 min per job)",
        f"",
        f"### Option B: Local Chai-1 (requires GPU >= 16 GB VRAM)",
        f"",
        f"```bash",
        f"# Install",
        f"pip install chai-lab",
        f"",
        f"# Run all predictions",
        f"source ~/projects/helse/.venv/bin/activate",
        f"python mutation_profile/scripts/ai_research/chai1_predictions.py --run",
        f"```",
        f"",
        f"### Option C: Vast.ai GPU (A100 80GB, ~$0.80/hr)",
        f"",
        f"```bash",
        f"# Upload FASTA files to Vast.ai instance",
        f"# Install chai-lab, run inference, download results",
        f"pip install chai-lab",
        f"python mutation_profile/scripts/ai_research/chai1_predictions.py --run",
        f"```",
        f"",
        f"## Analysis Plan",
        f"",
        f"For each complex, compare mutant vs wildtype:",
        f"",
        f"1. **pLDDT** - per-residue confidence at the mutation site",
        f"2. **PAE** - predicted aligned error (interface confidence)",
        f"3. **RMSD** - structural deviation at key regions",
        f"4. **Contact maps** - gain/loss of inter-chain contacts near mutation",
        f"5. **PLIP** - protein-ligand interaction profiling (hydrogen bonds, salt bridges, hydrophobic contacts)",
        f"",
        f"Key comparisons per complex:",
        f"",
        f"| Complex | Key metric |",
        f"|---------|------------|",
        f"| 1 (DNMT3A) | R882-DNA backbone distance (arginine vs histidine) |",
        f"| 2 (SETBP1) | SETBP1-SET interface area and confidence |",
        f"| 3 (IDH2) | Active site geometry for 2-HG vs isocitrate |",
        f"| 4 (PTPN11) | N-SH2/PTP domain separation (open vs closed) |",
        f"| 5 (PTPN11+drug) | TNO155 binding pose and allosteric pocket access |",
        f"| 6 (EZH2/PRC2) | SET domain catalytic geometry, EZH2-EED interface stability |",
        f"",
        f"Run PLIP analysis on completed structures:",
        f"```bash",
        f"python mutation_profile/scripts/ai_research/chai1_predictions.py --analyze",
        f"```",
        f"",
        f"## File Inventory",
        f"",
        f"### FASTA files (`fasta/`)",
        f"",
        f"| File | Complex | Variant |",
        f"|------|---------|---------|",
    ])

    for job in all_jobs:
        report_lines.append(f"| {job['name']}.fasta | {job['complex']} | {job['variant'].title()} |")

    report_lines.extend([
        f"",
        f"### Config files (`configs/`)",
        f"",
        f"| File | Ligands |",
        f"|------|---------|",
    ])

    for job in all_jobs:
        config_path = CONFIG_DIR / f"{job['name']}_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            ligs = ", ".join(l["name"] for l in config.get("ligands", [])) or "None"
        else:
            ligs = "?"
        report_lines.append(f"| {job['name']}_config.json | {ligs} |")

    if inference_results:
        report_lines.extend([
            f"",
            f"## Inference Results",
            f"",
            f"| Job | Status | Structures |",
            f"|-----|--------|------------|",
        ])
        for r in inference_results:
            n_struct = r.get("n_structures", "-")
            report_lines.append(f"| {r['name']} | {r['status']} | {n_struct} |")

    report_lines.append("")

    report_path = OUTPUT_DIR / "REPORT.md"
    report_path.write_text("\n".join(report_lines))
    log.info(f"\nReport: {report_path}")
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chai-1 multi-chain complex predictions for patient driver mutations"
    )
    parser.add_argument("--run", action="store_true", help="Run local Chai-1 inference (requires GPU)")
    parser.add_argument("--analyze", action="store_true", help="Run PLIP contact analysis on output structures")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Chai-1 Multi-Chain Complex Predictions")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("=" * 60)

    # Fetch all sequences
    seqs = get_all_sequences()

    # Generate input files for all 6 complexes
    all_jobs = []
    all_jobs.extend(generate_complex_1(seqs))
    all_jobs.extend(generate_complex_2(seqs))
    all_jobs.extend(generate_complex_3(seqs))
    all_jobs.extend(generate_complex_4(seqs))
    all_jobs.extend(generate_complex_5(seqs))
    all_jobs.extend(generate_complex_6(seqs))

    log.info("")
    log.info("=" * 60)
    log.info(f"Generated {len(all_jobs)} job inputs ({len(all_jobs)} FASTA + {len(all_jobs)} configs)")
    log.info("=" * 60)

    # Run local inference if requested
    inference_results = None
    if args.run:
        log.info("")
        log.info("=" * 60)
        log.info("Running local Chai-1 inference")
        log.info("=" * 60)
        inference_results = run_local_inference(all_jobs)

    # Run PLIP analysis if requested
    if args.analyze:
        log.info("")
        log.info("=" * 60)
        log.info("Running PLIP contact analysis")
        log.info("=" * 60)
        run_plip_analysis(all_jobs)

    # Generate report
    generate_report(seqs, all_jobs, inference_results)

    log.info("")
    log.info("Done. Next steps:")
    if not args.run:
        log.info("  1. Upload FASTA files to lab.chaidiscovery.com")
        log.info("     (paste ligand SMILES from config JSONs for complexes with ligands)")
        log.info("  2. OR run locally: python chai1_predictions.py --run")
    log.info("  3. After getting results: python chai1_predictions.py --analyze")


if __name__ == "__main__":
    main()
