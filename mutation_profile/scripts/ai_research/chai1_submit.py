#!/usr/bin/env python3
"""
Chai-1 protein structure prediction: local runner + Vast.ai remote submission.

Converts prepared FASTA/config files to Chai-1 format and runs predictions
locally (if GPU has sufficient VRAM) or generates Vast.ai commands for remote execution.

Priority complex: PTPN11 E76Q + TNO155 (SHP2 allosteric inhibitor docking).

Usage:
    # Run locally (requires >= 16 GB VRAM for 593-residue complexes)
    python mutation_profile/scripts/ai_research/chai1_submit.py --local --complex 5

    # Run all complexes locally
    python mutation_profile/scripts/ai_research/chai1_submit.py --local --all

    # Generate Vast.ai remote commands
    python mutation_profile/scripts/ai_research/chai1_submit.py --vastai --gpu a100

    # Convert existing FASTA files to Chai-1 format only
    python mutation_profile/scripts/ai_research/chai1_submit.py --convert-only

    # Dry run: show what would be done
    python mutation_profile/scripts/ai_research/chai1_submit.py --dry-run

Requires:
    pip install chai-lab  (v0.6.1+)
    CHAI_DOWNLOADS_DIR can be set to cache model weights (~6 GB total)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHAI1_INPUT_DIR = (
    PROJECT_ROOT
    / "mutation_profile"
    / "results"
    / "ai_research"
    / "alphafold3_inputs"
    / "chai1"
)
CHAI1_OUTPUT_DIR = (
    PROJECT_ROOT / "mutation_profile" / "results" / "ai_research" / "chai1_local"
)

# All 10 complexes: 5 mutant + 5 wildtype pairs
COMPLEXES = {
    1: {
        "name": "DNMT3A R882H + CpG DNA + SAM",
        "mutant": "complex1_DNMT3A_R882H_DNA_SAM",
        "wildtype": "complex1_DNMT3A_WT_DNA_SAM",
        "estimated_vram_gb": 8,
        "description": "DNA methyltransferase with CpG substrate and SAM cofactor",
    },
    2: {
        "name": "SETBP1 G870S + SET protein",
        "mutant": "complex2_SETBP1_G870S_SET",
        "wildtype": "complex2_SETBP1_WT_SET",
        "estimated_vram_gb": 6,
        "description": "SKI domain degron motif with SET/I2PP2A binding partner",
    },
    3: {
        "name": "IDH2 R140Q homodimer + NADP+ + AKG",
        "mutant": "complex3_IDH2_R140Q_dimer_NADP_AKG",
        "wildtype": "complex3_IDH2_WT_dimer_NADP_AKG",
        "estimated_vram_gb": 16,
        "description": "Isocitrate dehydrogenase dimer with cofactor and substrate",
    },
    4: {
        "name": "PTPN11 E76Q full-length (autoinhibition)",
        "mutant": "complex4_PTPN11_E76Q_fulllength",
        "wildtype": "complex4_PTPN11_WT_fulllength",
        "estimated_vram_gb": 12,
        "description": "SHP2 phosphatase N-SH2/PTP autoinhibition conformation",
    },
    5: {
        "name": "PTPN11 E76Q + TNO155",
        "mutant": "complex5_PTPN11_E76Q_TNO155",
        "wildtype": "complex5_PTPN11_WT_TNO155",
        "estimated_vram_gb": 16,
        "description": "SHP2 with allosteric inhibitor TNO155 (Novartis Phase I/II)",
    },
}


def convert_fasta_to_chai1_format(
    fasta_path: Path,
    config_path: Path,
    output_path: Path,
) -> Path:
    """Convert existing FASTA + config to Chai-1 native FASTA format.

    Chai-1 expects headers like:
        >protein|name=PTPN11_E76Q
        MTSRRWFH...
        >ligand|name=TNO155
        CC1=CC2=...  (SMILES)
        >dna|name=sense_strand
        AACGCGAACGCG

    Existing files use:
        >chain_A|entity_type=protein
        MTSRRWFH...
    """
    # Parse existing FASTA
    chains: list[tuple[str, str, str]] = []  # (entity_type, name, sequence)
    with open(fasta_path) as f:
        current_header = None
        current_seq_parts: list[str] = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    seq = "".join(current_seq_parts)
                    chains.append((*_parse_header(current_header), seq))
                current_header = line[1:]
                current_seq_parts = []
            elif line:
                current_seq_parts.append(line)
        if current_header is not None:
            seq = "".join(current_seq_parts)
            chains.append((*_parse_header(current_header), seq))

    # Parse config for ligands
    config = json.loads(config_path.read_text())
    ligands = config.get("ligands", [])

    # Build Chai-1 format FASTA
    lines: list[str] = []
    for entity_type, name, sequence in chains:
        lines.append(f">{entity_type}|name={name}")
        lines.append(sequence)

    for lig in ligands:
        lig_name = lig["name"].replace(" ", "_").replace("+", "plus")
        lines.append(f">ligand|name={lig_name}")
        lines.append(lig["smiles"])

    lines.append("")  # trailing newline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    log.info(
        "Converted %s -> %s (%d chains, %d ligands)",
        fasta_path.name,
        output_path.name,
        len(chains),
        len(ligands),
    )
    return output_path


def _parse_header(header: str) -> tuple[str, str]:
    """Parse FASTA header into (entity_type, chain_name).

    Handles both formats:
        chain_A|entity_type=protein  -> (protein, chain_A)
        protein|name=PTPN11          -> (protein, PTPN11)
    """
    parts = header.split("|")
    entity_type = "protein"
    name = parts[0]

    for part in parts:
        if part.startswith("entity_type="):
            entity_type = part.split("=", 1)[1]
        elif part.startswith("name="):
            name = part.split("=", 1)[1]

    # If first part looks like chain_X and second is entity_type=..., use chain as name
    if parts[0].startswith("chain_") and len(parts) > 1:
        name = parts[0]
        for p in parts[1:]:
            if p.startswith("entity_type="):
                entity_type = p.split("=", 1)[1]

    return entity_type, name


def run_local_inference(
    fasta_path: Path,
    output_dir: Path,
    config: dict,
    num_samples: int = 1,
) -> bool:
    """Run Chai-1 inference locally. Returns True on success."""
    try:
        import torch
        from chai_lab.chai1 import run_inference
    except ImportError:
        log.error("chai-lab not installed. Run: pip install chai-lab")
        return False

    if not torch.cuda.is_available():
        log.error("No CUDA GPU available. Use --vastai for remote execution.")
        return False

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log.info(
        "GPU: %s (%.1f GB VRAM)", torch.cuda.get_device_name(0), vram_gb
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=output_dir,
            num_trunk_recycles=config.get("num_trunk_recycles", 3),
            num_diffn_timesteps=config.get("num_diffn_timesteps", 200),
            seed=config.get("seed", 42),
            low_memory=True,
            use_esm_embeddings=True,
            num_diffn_samples=num_samples,
        )
        log.info("Generated %d structure(s):", len(candidates.cif_paths))
        for p in candidates.cif_paths:
            log.info("  %s", p)
        return True
    except torch.cuda.OutOfMemoryError:
        log.error(
            "OOM: %.1f GB VRAM insufficient. Need >= 16 GB. Use --vastai.", vram_gb
        )
        return False
    except Exception:
        log.exception("Chai-1 inference failed")
        return False


def generate_vastai_script(complexes_to_run: list[int]) -> str:
    """Generate a self-contained bash script for Vast.ai execution."""
    # Build list of FASTA files to process
    fasta_list = []
    for cid in complexes_to_run:
        info = COMPLEXES[cid]
        for variant_key in ("mutant", "wildtype"):
            fasta_list.append(info[variant_key])

    script = f"""#!/bin/bash
set -e

# Chai-1 Remote Execution Script for Vast.ai
# Generated for complexes: {', '.join(str(c) for c in complexes_to_run)}
#
# Provision an instance:
#   python scripts/gpu_runner.py custom --gpu a100 --max-cost 1.50 \\
#       --cmd "bash /workspace/run_chai1.sh"
#
# Or manually:
#   vastai search offers 'gpu_ram>=40 disk_space>=50 cuda_vers>=12.0 reliability>=0.95'
#   vastai create instance <ID> --image pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel \\
#       --disk 50 --onstart-cmd "bash /workspace/run_chai1.sh"

echo "=== Installing Chai-1 ==="
pip install -q chai-lab>=0.6.1

echo "=== GPU Info ==="
nvidia-smi
python3 -c "import torch; print(f'CUDA: {{torch.cuda.is_available()}}, GPU: {{torch.cuda.get_device_name(0)}}, VRAM: {{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}} GB')"

echo "=== Running Chai-1 predictions ==="
cd /workspace

"""
    for cid in complexes_to_run:
        info = COMPLEXES[cid]
        for variant_key in ("mutant", "wildtype"):
            name = info[variant_key]
            script += f"""
echo "--- {name} ---"
python3 -c "
from pathlib import Path
from chai_lab.chai1 import run_inference

candidates = run_inference(
    fasta_file=Path('/workspace/inputs/{name}.fasta'),
    output_dir=Path('/workspace/output/{name}/'),
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    seed=42,
    low_memory=True,
    use_esm_embeddings=True,
    num_diffn_samples=5,
)
print(f'Generated {{len(candidates.cif_paths)}} structures for {name}')
for p in candidates.cif_paths:
    print(f'  {{p}}')
"
"""

    script += """
echo "=== All predictions complete ==="
ls -la /workspace/output/
echo "=== Download results with: ==="
echo "  vastai copy <instance_id>:/workspace/output/ ./chai1_results/"
"""
    return script


def convert_all_fastas(complexes_to_run: list[int]) -> list[Path]:
    """Convert all specified complexes to Chai-1 format."""
    converted: list[Path] = []
    for cid in complexes_to_run:
        info = COMPLEXES[cid]
        for variant_key in ("mutant", "wildtype"):
            name = info[variant_key]
            fasta_src = CHAI1_INPUT_DIR / f"{name}.fasta"
            config_src = CHAI1_INPUT_DIR / f"{name}_config.json"
            fasta_dst = CHAI1_OUTPUT_DIR / f"{name}.fasta"

            if not fasta_src.exists():
                log.warning("Missing FASTA: %s", fasta_src)
                continue
            if not config_src.exists():
                log.warning("Missing config: %s", config_src)
                continue

            convert_fasta_to_chai1_format(fasta_src, config_src, fasta_dst)
            converted.append(fasta_dst)
    return converted


def check_web_submission_info() -> None:
    """Print web submission instructions for Chai Discovery platform."""
    print(
        """
=== Chai Discovery Web Platform ===
URL: https://lab.chaidiscovery.com

Free tier: unlimited predictions (account required)
Supports: protein, DNA/RNA, ligands (SMILES), glycans, covalent modifications

Steps for PTPN11 E76Q + TNO155:
1. Go to https://lab.chaidiscovery.com and sign in / create account
2. Click "New Fold" or "New Structure Prediction"
3. Add protein entity:
   - Type: Protein
   - Paste the PTPN11 E76Q sequence (593 aa):
     MTSRRWFHPNITGVEAENLLLTRGVDGSFLARPSKSNPGDFTLSVRRNGAVTHIKIQNTGDYYDLYGGEK...
     (full sequence in chai1/complex5_PTPN11_E76Q_TNO155.fasta)
4. Add ligand entity:
   - Type: Small molecule / Ligand
   - SMILES: CC1=CC2=C(C=C1)N(C(=O)C3=CC(=CC=C3F)C(F)(F)F)CC(C2)NS(=O)(=O)C4=CC=CC=C4
   - Name: TNO155
5. Submit prediction
6. Results typically available in 5-30 minutes
7. Download mmCIF/PDB output for analysis

Repeat for wildtype (E76 instead of Q76) for comparison.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chai-1 structure prediction: local or Vast.ai remote",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run Chai-1 locally (requires CUDA GPU with sufficient VRAM)",
    )
    parser.add_argument(
        "--vastai",
        action="store_true",
        help="Generate Vast.ai remote execution script",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Print Chai Discovery web submission instructions",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert FASTA files to Chai-1 format",
    )
    parser.add_argument(
        "--complex",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific complex (1-5). Default: 5 (PTPN11 E76Q + TNO155)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 5 complexes (both mutant and wildtype)",
    )
    parser.add_argument(
        "--gpu",
        default="a100",
        choices=["a100", "a100-80", "a6000", "4090", "h100"],
        help="GPU type for Vast.ai (default: a100)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of diffusion samples (default: 1 for local, 5 for Vast.ai)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    args = parser.parse_args()

    # Determine which complexes to process
    if args.all:
        complexes_to_run = [1, 2, 3, 4, 5]
    elif args.complex:
        complexes_to_run = [args.complex]
    else:
        complexes_to_run = [5]  # Default: priority complex

    log.info(
        "Complexes to process: %s",
        ", ".join(f"{c} ({COMPLEXES[c]['name']})" for c in complexes_to_run),
    )

    # Convert FASTA files
    converted = convert_all_fastas(complexes_to_run)
    log.info("Converted %d FASTA files to Chai-1 format", len(converted))

    if args.convert_only:
        for p in converted:
            print(f"  {p}")
        return

    if args.web:
        check_web_submission_info()
        return

    if args.vastai:
        script = generate_vastai_script(complexes_to_run)
        script_path = CHAI1_OUTPUT_DIR / "run_chai1_vastai.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        os.chmod(script_path, 0o755)
        log.info("Vast.ai script written to: %s", script_path)
        print(f"\nTo run on Vast.ai:")
        print(f"  1. Upload FASTA files from {CHAI1_OUTPUT_DIR}/ to /workspace/inputs/")
        print(f"  2. Upload {script_path} to /workspace/run_chai1.sh")
        print(f"  3. Run: python scripts/gpu_runner.py custom --gpu {args.gpu} \\")
        print(f'       --cmd "bash /workspace/run_chai1.sh"')
        return

    if args.local:
        if args.dry_run:
            log.info("DRY RUN: would run %d predictions locally", len(converted))
            for p in converted:
                print(f"  {p}")
            return

        success_count = 0
        for cid in complexes_to_run:
            info = COMPLEXES[cid]
            for variant_key in ("mutant", "wildtype"):
                name = info[variant_key]
                fasta = CHAI1_OUTPUT_DIR / f"{name}.fasta"
                config_path = CHAI1_INPUT_DIR / f"{name}_config.json"
                output = CHAI1_OUTPUT_DIR / f"{name}_output"

                if not fasta.exists():
                    log.warning("Skipping %s: FASTA not found", name)
                    continue

                config = json.loads(config_path.read_text())
                log.info("Running: %s", name)

                if run_local_inference(fasta, output, config, args.num_samples):
                    success_count += 1

        log.info(
            "Completed %d/%d predictions",
            success_count,
            len(complexes_to_run) * 2,
        )
        return

    # Default: show help
    parser.print_help()
    print("\nQuick start:")
    print("  # Try locally (priority complex):")
    print(
        "  python mutation_profile/scripts/ai_research/chai1_submit.py --local --complex 5"
    )
    print("  # Generate Vast.ai script (all complexes):")
    print(
        "  python mutation_profile/scripts/ai_research/chai1_submit.py --vastai --all"
    )
    print("  # Web submission guide:")
    print("  python mutation_profile/scripts/ai_research/chai1_submit.py --web")


if __name__ == "__main__":
    main()
