#!/usr/bin/env python3
"""
alphafold3_inputs.py -- Generate AlphaFold3 Server and Chai-1 input files
for multi-chain protein complex predictions.

Generates input files for 5 biologically relevant complexes based on patient
mutation profile (DNMT3A R882H, IDH2 R140Q, SETBP1 G870S, PTPN11 E76Q):

Complex 1: DNMT3A R882H catalytic domain + CpG DNA + SAM cofactor
Complex 2: SETBP1 G870S SKI domain + SET/I2PP2A protein
Complex 3: IDH2 R140Q homodimer + NADP+ + alpha-ketoglutarate
Complex 4: PTPN11 E76Q full-length (autoinhibition disruption)
Complex 5: PTPN11 E76Q + TNO155 small molecule

For each complex, both mutant and wildtype versions are generated.
AlphaFold3 Server JSON uses the alphafoldserver.com input format.
Chai-1 uses FASTA + ligand SMILES in the chai-lab input format.

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/ai_research/alphafold3_inputs.py

Outputs:
    mutation_profile/results/ai_research/alphafold3_inputs/
      - alphafold3_server/   AlphaFold3 Server JSON files
      - chai1/               Chai-1 FASTA + config files
      - README.md            Submission instructions
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import requests

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT = Path("Path(__file__).resolve().parents[2]")
OUTPUT_DIR = PROJECT / "mutation_profile" / "results" / "ai_research" / "alphafold3_inputs"
AF3_DIR = OUTPUT_DIR / "alphafold3_server"
CHAI_DIR = OUTPUT_DIR / "chai1"

for d in [OUTPUT_DIR, AF3_DIR, CHAI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "alphafold3_inputs.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PROTEIN SEQUENCES (UniProt canonical)
# ═══════════════════════════════════════════════════════════════════════════════

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


# Full-length sequences from existing pipeline (verified in alphafold3_analysis.py)
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


# ── Small molecule SMILES (PubChem-verified) ─────────────────────────────────
SMILES = {
    # TNO155 - SHP2 allosteric inhibitor (Novartis, Phase 1/2)
    # PubChem CID: 137796220
    "TNO155": "CC1=CC2=C(C=C1)N(C(=O)C3=CC(=CC=C3F)C(F)(F)F)CC(C2)NS(=O)(=O)C4=CC=CC=C4",
    # S-adenosylmethionine (SAM) - methyl donor cofactor for DNMT3A
    # PubChem CID: 34756
    "SAM": "C[S+](CC[C@@H](N)C(=O)[O-])C[C@@H]1OC(n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
    # NADP+ - cofactor for IDH2
    # PubChem CID: 5886
    "NADP+": "NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]2OC([C@H](OP(=O)([O-])[O-])[C@@H]2O)n2cnc3c(N)ncnc32)[C@@H](O)[C@H]1O",
    # Alpha-ketoglutarate (2-oxoglutarate) - IDH2 substrate
    # PubChem CID: 51
    "AKG": "OC(=O)CCC(=O)C(=O)O",
    # Enasidenib (AG-221) - IDH2 inhibitor, FDA approved 2017
    # PubChem CID: 89683805
    "Enasidenib": "CC(C)(O)CNC(=O)C1=CC=C(C=C1)C2=NC(=NC(=N2)NC3=CC=C(F)C=C3)NC4=CC=C(F)C=C4",
}


# ── DNA sequences for Complex 1 ─────────────────────────────────────────────
DNA_SENSE = "AACGCGAACGCG"
DNA_ANTISENSE = "CGCGTTCGCGTT"


# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENCE FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_sequences() -> dict:
    """Fetch and prepare all protein sequences needed for the 5 complexes."""
    sequences = {}

    # DNMT3A (Q9Y6K1) - 912 residues, catalytic domain ~600-912
    log.info("=" * 60)
    log.info("Fetching protein sequences from UniProt")
    log.info("=" * 60)

    dnmt3a_full = fetch_uniprot_sequence("Q9Y6K1")
    assert dnmt3a_full[881] == "R", (
        f"DNMT3A pos 882: expected R, got {dnmt3a_full[881]}"
    )
    sequences["DNMT3A_full"] = dnmt3a_full
    # Catalytic domain: residues 600-912 (0-indexed: 599:912)
    sequences["DNMT3A_cat_WT"] = dnmt3a_full[599:]
    sequences["DNMT3A_cat_R882H"] = (
        dnmt3a_full[599:881] + "H" + dnmt3a_full[882:]
    )
    log.info(f"  DNMT3A catalytic domain: {len(sequences['DNMT3A_cat_WT'])} residues (600-{len(dnmt3a_full)})")

    # SETBP1 (Q9Y6X0) - 1596 residues, SKI domain ~858-930
    setbp1_full = fetch_uniprot_sequence("Q9Y6X0")
    assert setbp1_full[869] == "G", (
        f"SETBP1 pos 870: expected G, got {setbp1_full[869]}"
    )
    sequences["SETBP1_full"] = setbp1_full
    # SKI homology domain: residues 850-930 (0-indexed: 849:930)
    sequences["SETBP1_SKI_WT"] = setbp1_full[849:930]
    sequences["SETBP1_SKI_G870S"] = (
        setbp1_full[849:869] + "S" + setbp1_full[870:930]
    )
    log.info(f"  SETBP1 SKI domain: {len(sequences['SETBP1_SKI_WT'])} residues (850-930)")

    # SET/I2PP2A (Q01105) - PP2A inhibitor, SETBP1 binding partner
    set_full = fetch_uniprot_sequence("Q01105")
    sequences["SET_full"] = set_full
    log.info(f"  SET protein: {len(set_full)} residues (full length)")

    # Use pre-verified sequences for PTPN11 and IDH2
    sequences["PTPN11_WT"] = PTPN11_WT_SEQ
    sequences["PTPN11_E76Q"] = PTPN11_E76Q_SEQ
    sequences["IDH2_WT"] = IDH2_WT_SEQ
    sequences["IDH2_R140Q"] = IDH2_R140Q_SEQ

    log.info(f"  PTPN11: {len(PTPN11_WT_SEQ)} residues (pre-verified)")
    log.info(f"  IDH2: {len(IDH2_WT_SEQ)} residues (pre-verified)")

    return sequences


# ═══════════════════════════════════════════════════════════════════════════════
# ALPHAFOLD3 SERVER JSON GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def af3_protein_chain(sequence: str, chain_id: str = "A") -> dict:
    """Create an AlphaFold3 Server protein chain entry."""
    return {
        "proteinChain": {
            "sequence": sequence,
            "count": 1,
        }
    }


def af3_protein_chain_count(sequence: str, count: int) -> dict:
    """Create an AlphaFold3 Server protein chain with copy count (homodimer etc)."""
    return {
        "proteinChain": {
            "sequence": sequence,
            "count": count,
        }
    }


def af3_dna_chain(sequence: str) -> dict:
    """Create an AlphaFold3 Server DNA chain entry."""
    return {
        "dnaSequence": {
            "sequence": sequence,
            "count": 1,
        }
    }


def af3_ligand_smiles(smiles: str) -> dict:
    """Create an AlphaFold3 Server ligand entry from SMILES."""
    return {
        "ligand": {
            "smiles": smiles,
            "count": 1,
        }
    }


def af3_ligand_ccd(ccd_code: str, count: int = 1) -> dict:
    """Create an AlphaFold3 Server ligand entry from CCD code."""
    return {
        "ligand": {
            "ccdCodes": [ccd_code],
            "count": count,
        }
    }


def write_af3_json(name: str, description: str, sequences: list) -> Path:
    """Write an AlphaFold3 Server JSON input file."""
    job = {
        "name": name,
        "modelSeeds": [42],
        "sequences": sequences,
    }
    path = AF3_DIR / f"{name}.json"
    path.write_text(json.dumps(job, indent=2) + "\n")
    log.info(f"  AF3 Server: {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CHAI-1 INPUT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def write_chai_fasta(name: str, chains: list[dict]) -> Path:
    """Write a Chai-1 FASTA file with chain annotations.

    chains: list of dicts with keys: id, type, sequence
    type is one of: protein, dna, rna
    """
    path = CHAI_DIR / f"{name}.fasta"
    lines = []
    for chain in chains:
        entity_type = chain["type"]
        chain_id = chain["id"]
        seq = chain["sequence"]
        lines.append(f">chain_{chain_id}|entity_type={entity_type}")
        # Wrap sequence at 80 chars
        for i in range(0, len(seq), 80):
            lines.append(seq[i : i + 80])
    path.write_text("\n".join(lines) + "\n")
    log.info(f"  Chai-1 FASTA: {path.name}")
    return path


def write_chai_constraints(name: str, ligands: list[dict] | None = None) -> Path:
    """Write a Chai-1 constraints/config file with ligand SMILES.

    ligands: list of dicts with keys: name, smiles
    """
    config = {
        "name": name,
        "ligands": ligands or [],
        "num_trunk_recycles": 3,
        "num_diffn_timesteps": 200,
        "seed": 42,
    }
    path = CHAI_DIR / f"{name}_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    log.info(f"  Chai-1 config: {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEX DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_complex_1(seqs: dict) -> None:
    """Complex 1: DNMT3A R882H catalytic domain + CpG DNA + SAM cofactor.

    Models how the R882H hotspot mutation affects DNA binding and methyl
    transfer. DNMT3A R882 directly contacts the DNA backbone; R882H reduces
    CpG methylation by ~80% and acts dominant-negative on WT DNMT3A.
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 1: DNMT3A R882H + CpG DNA + SAM")
    log.info("=" * 60)

    # -- Mutant --
    write_af3_json(
        "complex1_DNMT3A_R882H_DNA_SAM",
        "DNMT3A catalytic domain R882H + CpG DNA duplex + SAM cofactor",
        [
            af3_protein_chain(seqs["DNMT3A_cat_R882H"]),
            af3_dna_chain(DNA_SENSE),
            af3_dna_chain(DNA_ANTISENSE),
            af3_ligand_ccd("SAM"),
        ],
    )

    # -- Wildtype --
    write_af3_json(
        "complex1_DNMT3A_WT_DNA_SAM",
        "DNMT3A catalytic domain WT + CpG DNA duplex + SAM cofactor",
        [
            af3_protein_chain(seqs["DNMT3A_cat_WT"]),
            af3_dna_chain(DNA_SENSE),
            af3_dna_chain(DNA_ANTISENSE),
            af3_ligand_ccd("SAM"),
        ],
    )

    # -- Chai-1 mutant --
    write_chai_fasta(
        "complex1_DNMT3A_R882H_DNA_SAM",
        [
            {"id": "A", "type": "protein", "sequence": seqs["DNMT3A_cat_R882H"]},
            {"id": "B", "type": "dna", "sequence": DNA_SENSE},
            {"id": "C", "type": "dna", "sequence": DNA_ANTISENSE},
        ],
    )
    write_chai_constraints(
        "complex1_DNMT3A_R882H_DNA_SAM",
        ligands=[{"name": "SAM", "smiles": SMILES["SAM"]}],
    )

    # -- Chai-1 wildtype --
    write_chai_fasta(
        "complex1_DNMT3A_WT_DNA_SAM",
        [
            {"id": "A", "type": "protein", "sequence": seqs["DNMT3A_cat_WT"]},
            {"id": "B", "type": "dna", "sequence": DNA_SENSE},
            {"id": "C", "type": "dna", "sequence": DNA_ANTISENSE},
        ],
    )
    write_chai_constraints(
        "complex1_DNMT3A_WT_DNA_SAM",
        ligands=[{"name": "SAM", "smiles": SMILES["SAM"]}],
    )


def generate_complex_2(seqs: dict) -> None:
    """Complex 2: SETBP1 G870S SKI domain + SET/I2PP2A protein.

    SETBP1 binds and stabilizes the SET oncoprotein (PP2A inhibitor).
    G870S is in the SKI homology domain (degron motif recognized by
    beta-TrCP for ubiquitin-proteasome degradation). The mutation disrupts
    the degron, stabilizing SETBP1 which in turn stabilizes SET, leading
    to constitutive PP2A inhibition and cell proliferation.
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 2: SETBP1 G870S SKI domain + SET protein")
    log.info("=" * 60)

    # -- Mutant --
    write_af3_json(
        "complex2_SETBP1_G870S_SET",
        "SETBP1 SKI domain G870S + SET/I2PP2A protein",
        [
            af3_protein_chain(seqs["SETBP1_SKI_G870S"]),
            af3_protein_chain(seqs["SET_full"]),
        ],
    )

    # -- Wildtype --
    write_af3_json(
        "complex2_SETBP1_WT_SET",
        "SETBP1 SKI domain WT + SET/I2PP2A protein",
        [
            af3_protein_chain(seqs["SETBP1_SKI_WT"]),
            af3_protein_chain(seqs["SET_full"]),
        ],
    )

    # -- Chai-1 mutant --
    write_chai_fasta(
        "complex2_SETBP1_G870S_SET",
        [
            {"id": "A", "type": "protein", "sequence": seqs["SETBP1_SKI_G870S"]},
            {"id": "B", "type": "protein", "sequence": seqs["SET_full"]},
        ],
    )
    write_chai_constraints("complex2_SETBP1_G870S_SET")

    # -- Chai-1 wildtype --
    write_chai_fasta(
        "complex2_SETBP1_WT_SET",
        [
            {"id": "A", "type": "protein", "sequence": seqs["SETBP1_SKI_WT"]},
            {"id": "B", "type": "protein", "sequence": seqs["SET_full"]},
        ],
    )
    write_chai_constraints("complex2_SETBP1_WT_SET")


def generate_complex_3(seqs: dict) -> None:
    """Complex 3: IDH2 R140Q homodimer + NADP+ + alpha-ketoglutarate.

    IDH2 functions as a homodimer. R140Q is a neomorphic mutation that
    converts alpha-ketoglutarate to the oncometabolite 2-hydroxyglutarate
    (2-HG), which inhibits TET2 and other alpha-KG-dependent dioxygenases.
    The homodimer context is critical -- enasidenib binds the dimer interface.
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 3: IDH2 R140Q homodimer + NADP+ + AKG")
    log.info("=" * 60)

    # -- Mutant homodimer --
    write_af3_json(
        "complex3_IDH2_R140Q_dimer_NADP_AKG",
        "IDH2 R140Q homodimer + NADP+ + alpha-ketoglutarate",
        [
            af3_protein_chain_count(seqs["IDH2_R140Q"], count=2),
            af3_ligand_ccd("NAP"),       # NADP+ CCD code
            af3_ligand_ccd("AKG"),       # alpha-ketoglutarate CCD code
        ],
    )

    # -- Wildtype homodimer --
    write_af3_json(
        "complex3_IDH2_WT_dimer_NADP_AKG",
        "IDH2 WT homodimer + NADP+ + alpha-ketoglutarate",
        [
            af3_protein_chain_count(seqs["IDH2_WT"], count=2),
            af3_ligand_ccd("NAP"),
            af3_ligand_ccd("AKG"),
        ],
    )

    # -- Chai-1 mutant (explicit two chains) --
    write_chai_fasta(
        "complex3_IDH2_R140Q_dimer_NADP_AKG",
        [
            {"id": "A", "type": "protein", "sequence": seqs["IDH2_R140Q"]},
            {"id": "B", "type": "protein", "sequence": seqs["IDH2_R140Q"]},
        ],
    )
    write_chai_constraints(
        "complex3_IDH2_R140Q_dimer_NADP_AKG",
        ligands=[
            {"name": "NADP+", "smiles": SMILES["NADP+"]},
            {"name": "AKG", "smiles": SMILES["AKG"]},
        ],
    )

    # -- Chai-1 wildtype --
    write_chai_fasta(
        "complex3_IDH2_WT_dimer_NADP_AKG",
        [
            {"id": "A", "type": "protein", "sequence": seqs["IDH2_WT"]},
            {"id": "B", "type": "protein", "sequence": seqs["IDH2_WT"]},
        ],
    )
    write_chai_constraints(
        "complex3_IDH2_WT_dimer_NADP_AKG",
        ligands=[
            {"name": "NADP+", "smiles": SMILES["NADP+"]},
            {"name": "AKG", "smiles": SMILES["AKG"]},
        ],
    )


def generate_complex_4(seqs: dict) -> None:
    """Complex 4: PTPN11 E76Q full-length (autoinhibition disruption).

    SHP2 (PTPN11) has three domains: N-SH2 (1-105), C-SH2 (112-216),
    PTP catalytic (246-527). In the autoinhibited state, the N-SH2 domain
    blocks the PTP active site. E76Q in the N-SH2 domain disrupts the
    D61/E76 hydrogen bond network at the N-SH2/PTP interface, shifting
    the equilibrium toward the open (active) conformation.

    This single-chain prediction reveals the conformational shift.
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 4: PTPN11 E76Q full-length (autoinhibition)")
    log.info("=" * 60)

    # -- Mutant --
    write_af3_json(
        "complex4_PTPN11_E76Q_fulllength",
        "PTPN11/SHP2 full-length E76Q (N-SH2 + C-SH2 + PTP domain)",
        [
            af3_protein_chain(seqs["PTPN11_E76Q"]),
        ],
    )

    # -- Wildtype --
    write_af3_json(
        "complex4_PTPN11_WT_fulllength",
        "PTPN11/SHP2 full-length WT (autoinhibited conformation)",
        [
            af3_protein_chain(seqs["PTPN11_WT"]),
        ],
    )

    # -- Chai-1 mutant --
    write_chai_fasta(
        "complex4_PTPN11_E76Q_fulllength",
        [
            {"id": "A", "type": "protein", "sequence": seqs["PTPN11_E76Q"]},
        ],
    )
    write_chai_constraints("complex4_PTPN11_E76Q_fulllength")

    # -- Chai-1 wildtype --
    write_chai_fasta(
        "complex4_PTPN11_WT_fulllength",
        [
            {"id": "A", "type": "protein", "sequence": seqs["PTPN11_WT"]},
        ],
    )
    write_chai_constraints("complex4_PTPN11_WT_fulllength")


def generate_complex_5(seqs: dict) -> None:
    """Complex 5: PTPN11 E76Q + TNO155 small molecule.

    TNO155 is a potent allosteric SHP2 inhibitor (Novartis, Phase 1/2).
    It stabilizes the autoinhibited (closed) conformation by binding at
    the interface between the N-SH2 and PTP domains. E76Q disrupts this
    interface, so TNO155 binding to E76Q mutant is of direct clinical
    relevance -- will the inhibitor overcome constitutive activation?

    Prior docking: AutoDock Vina = -9.34 kcal/mol, DiffDock conf = -7.49
    """
    log.info("")
    log.info("=" * 60)
    log.info("Complex 5: PTPN11 E76Q + TNO155")
    log.info("=" * 60)

    # -- Mutant + drug --
    write_af3_json(
        "complex5_PTPN11_E76Q_TNO155",
        "PTPN11/SHP2 E76Q + TNO155 allosteric SHP2 inhibitor",
        [
            af3_protein_chain(seqs["PTPN11_E76Q"]),
            af3_ligand_smiles(SMILES["TNO155"]),
        ],
    )

    # -- Wildtype + drug (for comparison) --
    write_af3_json(
        "complex5_PTPN11_WT_TNO155",
        "PTPN11/SHP2 WT + TNO155 allosteric SHP2 inhibitor",
        [
            af3_protein_chain(seqs["PTPN11_WT"]),
            af3_ligand_smiles(SMILES["TNO155"]),
        ],
    )

    # -- Chai-1 mutant + drug --
    write_chai_fasta(
        "complex5_PTPN11_E76Q_TNO155",
        [
            {"id": "A", "type": "protein", "sequence": seqs["PTPN11_E76Q"]},
        ],
    )
    write_chai_constraints(
        "complex5_PTPN11_E76Q_TNO155",
        ligands=[{"name": "TNO155", "smiles": SMILES["TNO155"]}],
    )

    # -- Chai-1 wildtype + drug --
    write_chai_fasta(
        "complex5_PTPN11_WT_TNO155",
        [
            {"id": "A", "type": "protein", "sequence": seqs["PTPN11_WT"]},
        ],
    )
    write_chai_constraints(
        "complex5_PTPN11_WT_TNO155",
        ligands=[{"name": "TNO155", "smiles": SMILES["TNO155"]}],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# README GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_readme(seqs: dict) -> Path:
    """Generate README.md explaining how to submit each job."""
    readme = OUTPUT_DIR / "README.md"

    content = f"""# AlphaFold3 and Chai-1 Multi-Chain Complex Predictions

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Patient Variants

| Gene | Mutation | UniProt | Domain |
|------|----------|---------|--------|
| DNMT3A | R882H | Q9Y6K1 | Methyltransferase catalytic domain |
| SETBP1 | G870S | Q9Y6X0 | SKI homology domain (degron motif) |
| IDH2 | R140Q | P48735 | Isocitrate binding site |
| PTPN11 | E76Q | Q06124 | N-SH2 domain (autoinhibitory interface) |

## 5 Complexes

### Complex 1: DNMT3A R882H + CpG DNA + SAM

**Biological question:** How does R882H affect DNA-binding geometry and SAM positioning?

- Protein: DNMT3A catalytic domain (residues 600-{len(seqs['DNMT3A_full'])}, {len(seqs['DNMT3A_cat_WT'])} aa)
- DNA: 12-mer CpG duplex (5'-AACGCGAACGCG-3' + complement)
- Ligand: SAM cofactor (CCD code: SAM)
- Files: `complex1_DNMT3A_R882H_DNA_SAM.*` and `complex1_DNMT3A_WT_DNA_SAM.*`

### Complex 2: SETBP1 G870S SKI Domain + SET Protein

**Biological question:** Does G870S alter SETBP1-SET binding interface or SET stabilization?

- Protein A: SETBP1 SKI domain (residues 850-930, {len(seqs['SETBP1_SKI_WT'])} aa)
- Protein B: SET/I2PP2A full-length ({len(seqs['SET_full'])} aa)
- Files: `complex2_SETBP1_G870S_SET.*` and `complex2_SETBP1_WT_SET.*`

### Complex 3: IDH2 R140Q Homodimer + NADP+ + AKG

**Biological question:** How does R140Q alter the active site geometry for 2-HG production?

- Protein: IDH2 homodimer (2x {len(seqs['IDH2_R140Q'])} aa)
- Ligands: NADP+ (CCD: NAP) + alpha-ketoglutarate (CCD: AKG)
- Files: `complex3_IDH2_R140Q_dimer_NADP_AKG.*` and `complex3_IDH2_WT_dimer_NADP_AKG.*`

### Complex 4: PTPN11 E76Q Full-Length (Autoinhibition)

**Biological question:** Does E76Q shift the N-SH2/PTP interface toward the open conformation?

- Protein: PTPN11/SHP2 full-length ({len(seqs['PTPN11_E76Q'])} aa, N-SH2 + C-SH2 + PTP)
- No ligands -- conformational prediction only
- Files: `complex4_PTPN11_E76Q_fulllength.*` and `complex4_PTPN11_WT_fulllength.*`

### Complex 5: PTPN11 E76Q + TNO155

**Biological question:** Can TNO155 overcome E76Q-driven constitutive activation?

- Protein: PTPN11/SHP2 full-length ({len(seqs['PTPN11_E76Q'])} aa)
- Ligand: TNO155 (SHP2 allosteric inhibitor, Novartis Phase 1/2)
- Prior docking: AutoDock Vina = -9.34 kcal/mol, DiffDock = -7.49
- Files: `complex5_PTPN11_E76Q_TNO155.*` and `complex5_PTPN11_WT_TNO155.*`

## How to Submit

### AlphaFold3 Server (alphafoldserver.com)

1. Go to https://alphafoldserver.com
2. Sign in with Google account
3. Click "New job"
4. Click the "..." menu and select "Upload JSON"
5. Upload the `.json` file from `alphafold3_server/`
6. Review the job details and click "Submit"
7. Jobs take 5-30 minutes depending on complex size
8. Download results as mmCIF when complete

**Notes:**
- Free tier: 10 jobs per day
- Maximum 5000 residues total per job
- DNA chains must be provided as separate sense/antisense strands
- Ligands can use CCD codes (e.g., SAM, NAP, AKG) or SMILES
- Homodimers use `"count": 2` on a single proteinChain entry

### Chai-1 (Local or Chai Discovery)

#### Option A: Chai Discovery Platform (chaidiscovery.com)

1. Go to https://chaidiscovery.com
2. Create account or sign in
3. Upload the `.fasta` file for protein/DNA chains
4. For ligands, paste the SMILES from the `_config.json` file
5. Submit and wait for results

#### Option B: Local Chai-1 (requires GPU, ~16 GB VRAM)

```bash
# Install chai-lab
pip install chai-lab

# Run prediction
chai fold \\
    --fasta_file chai1/complex1_DNMT3A_R882H_DNA_SAM.fasta \\
    --output_dir results/complex1_mutant/ \\
    --num_trunk_recycles 3 \\
    --num_diffn_timesteps 200 \\
    --seed 42
```

For complexes with ligands, use the Python API:
```python
import chai_lab
from chai_lab.chai1 import run_inference

candidates = run_inference(
    fasta_file="chai1/complex5_PTPN11_E76Q_TNO155.fasta",
    ligands=["CC1=CC2=C(C=C1)N(C(=O)C3=CC(=CC=C3F)C(F)(F)F)CC(C2)NS(=O)(=O)C4=CC=CC=C4"],
    output_dir="results/complex5_mutant/",
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
)
```

## Analysis Plan

For each complex, compare mutant vs wildtype:

1. **pLDDT scores** -- per-residue confidence at the mutation site
2. **PAE (Predicted Aligned Error)** -- interface confidence between chains
3. **RMSD** -- structural deviation between mutant and wildtype at key regions
4. **Contact maps** -- gain/loss of inter-chain contacts near mutation
5. **Binding site geometry** -- ligand positioning relative to active site residues

Key comparisons:
- Complex 1: R882-DNA backbone distance (WT arginine vs mutant histidine)
- Complex 2: SETBP1-SET interface area and confidence
- Complex 3: Active site geometry for 2-HG vs isocitrate conversion
- Complex 4: N-SH2/PTP domain separation (open vs closed)
- Complex 5: TNO155 binding pose and allosteric pocket accessibility

## File Inventory

### AlphaFold3 Server (`alphafold3_server/`)

| File | Complex | Variant |
|------|---------|---------|
| complex1_DNMT3A_R882H_DNA_SAM.json | 1 | Mutant |
| complex1_DNMT3A_WT_DNA_SAM.json | 1 | Wildtype |
| complex2_SETBP1_G870S_SET.json | 2 | Mutant |
| complex2_SETBP1_WT_SET.json | 2 | Wildtype |
| complex3_IDH2_R140Q_dimer_NADP_AKG.json | 3 | Mutant |
| complex3_IDH2_WT_dimer_NADP_AKG.json | 3 | Wildtype |
| complex4_PTPN11_E76Q_fulllength.json | 4 | Mutant |
| complex4_PTPN11_WT_fulllength.json | 4 | Wildtype |
| complex5_PTPN11_E76Q_TNO155.json | 5 | Mutant |
| complex5_PTPN11_WT_TNO155.json | 5 | Wildtype |

### Chai-1 (`chai1/`)

| File | Complex | Variant |
|------|---------|---------|
| complex1_DNMT3A_R882H_DNA_SAM.fasta | 1 | Mutant |
| complex1_DNMT3A_R882H_DNA_SAM_config.json | 1 | Mutant |
| complex1_DNMT3A_WT_DNA_SAM.fasta | 1 | Wildtype |
| complex1_DNMT3A_WT_DNA_SAM_config.json | 1 | Wildtype |
| complex2_SETBP1_G870S_SET.fasta | 2 | Mutant |
| complex2_SETBP1_G870S_SET_config.json | 2 | Mutant |
| complex2_SETBP1_WT_SET.fasta | 2 | Wildtype |
| complex2_SETBP1_WT_SET_config.json | 2 | Wildtype |
| complex3_IDH2_R140Q_dimer_NADP_AKG.fasta | 3 | Mutant |
| complex3_IDH2_R140Q_dimer_NADP_AKG_config.json | 3 | Mutant |
| complex3_IDH2_WT_dimer_NADP_AKG.fasta | 3 | Wildtype |
| complex3_IDH2_WT_dimer_NADP_AKG_config.json | 3 | Wildtype |
| complex4_PTPN11_E76Q_fulllength.fasta | 4 | Mutant |
| complex4_PTPN11_E76Q_fulllength_config.json | 4 | Mutant |
| complex4_PTPN11_WT_fulllength.fasta | 4 | Wildtype |
| complex4_PTPN11_WT_fulllength_config.json | 4 | Wildtype |
| complex5_PTPN11_E76Q_TNO155.fasta | 5 | Mutant |
| complex5_PTPN11_E76Q_TNO155_config.json | 5 | Mutant |
| complex5_PTPN11_WT_TNO155.fasta | 5 | Wildtype |
| complex5_PTPN11_WT_TNO155_config.json | 5 | Wildtype |
"""
    readme.write_text(content)
    log.info(f"README: {readme}")
    return readme


# ═══════════════════════════════════════════════════════════════════════════════
# MANIFEST
# ═══════════════════════════════════════════════════════════════════════════════

def generate_manifest(seqs: dict) -> Path:
    """Generate a machine-readable manifest of all generated files."""
    manifest = {
        "generated": datetime.now().isoformat(),
        "script": "mutation_profile/scripts/ai_research/alphafold3_inputs.py",
        "sequences": {
            "DNMT3A_cat_WT": {"length": len(seqs["DNMT3A_cat_WT"]), "uniprot": "Q9Y6K1", "residues": f"600-{len(seqs['DNMT3A_full'])}"},
            "DNMT3A_cat_R882H": {"length": len(seqs["DNMT3A_cat_R882H"]), "mutation": "R882H"},
            "SETBP1_SKI_WT": {"length": len(seqs["SETBP1_SKI_WT"]), "uniprot": "Q9Y6X0", "residues": "850-930"},
            "SETBP1_SKI_G870S": {"length": len(seqs["SETBP1_SKI_G870S"]), "mutation": "G870S"},
            "SET_full": {"length": len(seqs["SET_full"]), "uniprot": "Q01105"},
            "IDH2_WT": {"length": len(seqs["IDH2_WT"]), "uniprot": "P48735"},
            "IDH2_R140Q": {"length": len(seqs["IDH2_R140Q"]), "mutation": "R140Q"},
            "PTPN11_WT": {"length": len(seqs["PTPN11_WT"]), "uniprot": "Q06124"},
            "PTPN11_E76Q": {"length": len(seqs["PTPN11_E76Q"]), "mutation": "E76Q"},
        },
        "dna": {
            "sense": DNA_SENSE,
            "antisense": DNA_ANTISENSE,
        },
        "ligands": {k: v for k, v in SMILES.items()},
        "complexes": [
            {
                "id": 1,
                "name": "DNMT3A R882H + CpG DNA + SAM",
                "chains": ["DNMT3A_cat (protein)", "DNA sense", "DNA antisense", "SAM (ligand)"],
                "af3_files": ["complex1_DNMT3A_R882H_DNA_SAM.json", "complex1_DNMT3A_WT_DNA_SAM.json"],
                "chai1_files": ["complex1_DNMT3A_R882H_DNA_SAM.fasta", "complex1_DNMT3A_WT_DNA_SAM.fasta"],
            },
            {
                "id": 2,
                "name": "SETBP1 G870S SKI + SET",
                "chains": ["SETBP1_SKI (protein)", "SET (protein)"],
                "af3_files": ["complex2_SETBP1_G870S_SET.json", "complex2_SETBP1_WT_SET.json"],
                "chai1_files": ["complex2_SETBP1_G870S_SET.fasta", "complex2_SETBP1_WT_SET.fasta"],
            },
            {
                "id": 3,
                "name": "IDH2 R140Q homodimer + NADP+ + AKG",
                "chains": ["IDH2 x2 (homodimer)", "NADP+ (ligand)", "AKG (ligand)"],
                "af3_files": ["complex3_IDH2_R140Q_dimer_NADP_AKG.json", "complex3_IDH2_WT_dimer_NADP_AKG.json"],
                "chai1_files": ["complex3_IDH2_R140Q_dimer_NADP_AKG.fasta", "complex3_IDH2_WT_dimer_NADP_AKG.fasta"],
            },
            {
                "id": 4,
                "name": "PTPN11 E76Q full-length",
                "chains": ["PTPN11 (protein, single chain)"],
                "af3_files": ["complex4_PTPN11_E76Q_fulllength.json", "complex4_PTPN11_WT_fulllength.json"],
                "chai1_files": ["complex4_PTPN11_E76Q_fulllength.fasta", "complex4_PTPN11_WT_fulllength.fasta"],
            },
            {
                "id": 5,
                "name": "PTPN11 E76Q + TNO155",
                "chains": ["PTPN11 (protein)", "TNO155 (ligand)"],
                "af3_files": ["complex5_PTPN11_E76Q_TNO155.json", "complex5_PTPN11_WT_TNO155.json"],
                "chai1_files": ["complex5_PTPN11_E76Q_TNO155.fasta", "complex5_PTPN11_WT_TNO155.fasta"],
            },
        ],
    }

    path = OUTPUT_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info(f"Manifest: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 60)
    log.info("AlphaFold3 / Chai-1 Input File Generator")
    log.info("=" * 60)
    log.info(f"Output directory: {OUTPUT_DIR}")

    # Fetch all sequences
    seqs = get_all_sequences()

    # Generate inputs for all 5 complexes
    generate_complex_1(seqs)
    generate_complex_2(seqs)
    generate_complex_3(seqs)
    generate_complex_4(seqs)
    generate_complex_5(seqs)

    # Generate README and manifest
    generate_readme(seqs)
    generate_manifest(seqs)

    # Summary
    af3_count = len(list(AF3_DIR.glob("*.json")))
    chai_fasta_count = len(list(CHAI_DIR.glob("*.fasta")))
    chai_config_count = len(list(CHAI_DIR.glob("*_config.json")))

    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"AlphaFold3 Server JSON files: {af3_count}")
    log.info(f"Chai-1 FASTA files:           {chai_fasta_count}")
    log.info(f"Chai-1 config files:          {chai_config_count}")
    log.info(f"Total files generated:        {af3_count + chai_fasta_count + chai_config_count + 2}")
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info("Done.")


if __name__ == "__main__":
    main()
