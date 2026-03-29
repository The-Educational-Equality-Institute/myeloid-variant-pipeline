#!/usr/bin/env python3
"""
run_all.py -- Master pipeline script for GENIE myeloid co-occurrence analysis.

Runs the full analysis pipeline in order:
  1. Validate GENIE data exists and is readable
  2. four_gene_cooccurrence.py       -- DNMT3A+IDH2+SETBP1+PTPN11 quadruple search
  3. setbp1_cooccurrence.py          -- All SETBP1 pairwise co-occurrences
  4. pairwise_matrix.py              -- Full 34-gene pairwise matrix
  5. cross_database.py               -- Cross-database validation
  6. esm2_variant_scoring.py         -- ESM-2 protein language model scoring
  7. verify_results.py               -- Automated result verification
  8. generate_piazza_package.py      -- Package results for Piazza review
  9. five_gene_cooccurrence.py       -- Five-gene (quintuple) co-occurrence search
  10. ablation_analysis.py           -- Ablation analysis of mutation combinations
  11. visualize_ismb_funnel.py       -- ISMB funnel chart visualization
  12. visualize_ismb_heatmap.py      -- ISMB concordance heatmap visualization
  13. visualize_pipeline_diagram.py  -- Pipeline diagram visualization
  14. Print summary

Inputs:
    - mutation_profile/data/genie/raw/ (all GENIE data files)
    - UniProt REST API (for ESM-2 scoring step)

Outputs:
    All outputs from the individual pipeline steps (see each script).

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/run_all.py [--skip-esm2] [--skip-package] [--dry-run]
           [--skip-ai-research] [--skip-viz]

Runtime: ~2-3 minutes (full pipeline, longer with AI research and visualization steps)
Dependencies: (standard library only -- invokes sub-scripts via subprocess)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "mutation_profile" / "scripts"
DATA_DIR = PROJECT_ROOT / "mutation_profile" / "data"
RESULTS_DIR = PROJECT_ROOT / "mutation_profile" / "results"
LOG_DIR = PROJECT_ROOT / "mutation_profile" / "logs"
GENIE_RAW = DATA_DIR / "genie" / "raw"

# Python interpreter — use the active venv
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

class PipelineStep:
    """A single step in the analysis pipeline."""

    def __init__(self, name: str, script: str, args: list[str] | None = None,
                 skip_flag: str | None = None, required: bool = True):
        self.name = name
        self.script = script
        self.args = args or []
        self.skip_flag = skip_flag
        self.required = required
        self.status = "pending"
        self.duration = 0.0
        self.error = ""

    @property
    def script_path(self) -> Path:
        return SCRIPTS_DIR / self.script

    def exists(self) -> bool:
        return self.script_path.exists()


PIPELINE: list[PipelineStep] = [
    PipelineStep(
        name="Four-gene co-occurrence (DNMT3A+IDH2+SETBP1+PTPN11)",
        script="four_gene_cooccurrence.py",
    ),
    PipelineStep(
        name="SETBP1 pairwise co-occurrence",
        script="setbp1_cooccurrence.py",
    ),
    PipelineStep(
        name="Full 34-gene pairwise matrix",
        script="pairwise_matrix.py",
    ),
    PipelineStep(
        name="Cross-database validation",
        script="cross_database.py",
    ),
    PipelineStep(
        name="ESM-2 variant scoring",
        script="esm2_variant_scoring.py",
        skip_flag="skip_esm2",
    ),
    PipelineStep(
        name="Result verification",
        script="verify_results.py",
        required=False,  # verification failure should not stop packaging
    ),
    PipelineStep(
        name="Generate Piazza package",
        script="generate_piazza_package.py",
        skip_flag="skip_package",
    ),
    # --- AI research steps (8-9) ---
    PipelineStep(
        name="Five-gene co-occurrence (quintuple search)",
        script="ai_research/five_gene_cooccurrence.py",
        skip_flag="skip_ai_research",
        required=False,
    ),
    PipelineStep(
        name="Ablation analysis",
        script="ai_research/ablation_analysis.py",
        skip_flag="skip_ai_research",
        required=False,
    ),
    # --- Visualization steps (10-12) ---
    PipelineStep(
        name="Visualization: ISMB funnel chart",
        script="visualize_ismb_funnel.py",
        skip_flag="skip_viz",
        required=False,
    ),
    PipelineStep(
        name="Visualization: ISMB concordance heatmap",
        script="visualize_ismb_heatmap.py",
        skip_flag="skip_viz",
        required=False,
    ),
    PipelineStep(
        name="Visualization: pipeline diagram",
        script="visualize_pipeline_diagram.py",
        skip_flag="skip_viz",
        required=False,
    ),
    PipelineStep(
        name="Benchmark: 40-profile validation",
        script="ai_research/benchmark_profiles.py",
        skip_flag="skip_benchmark",
        required=False,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str, level: str = "INFO"):
    """Print a timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def validate_genie_data() -> bool:
    """Check that critical GENIE files exist before running the pipeline."""
    critical_files = [
        "data_mutations_extended.txt",
        "data_clinical_sample.txt",
        "data_gene_matrix.txt",
    ]

    # Check in raw/ and also in the nested release directory
    release_dir = GENIE_RAW / "Data Releases" / "Release 19.0-public"
    search_dirs = [d for d in [release_dir, GENIE_RAW] if d.is_dir()]

    missing = []
    for fname in critical_files:
        found = any((d / fname).exists() for d in search_dirs)
        if not found:
            missing.append(fname)

    if missing:
        log(f"Missing GENIE files: {missing}", "ERROR")
        log("Run download_genie.py first, or check data paths.", "ERROR")
        return False

    # Quick row count on mutations file to confirm it's not truncated
    for d in search_dirs:
        mut_file = d / "data_mutations_extended.txt"
        if mut_file.exists():
            size_mb = mut_file.stat().st_size / (1024 * 1024)
            log(f"Mutations file: {size_mb:.0f} MB ({mut_file.relative_to(PROJECT_ROOT)})")
            if size_mb < 100:
                log("WARNING: Mutations file seems small (<100 MB). May be truncated.", "WARN")
            break

    return True


def run_step(step: PipelineStep, dry_run: bool = False) -> bool:
    """
    Run a single pipeline step. Returns True if succeeded or was skipped.
    """
    if not step.exists():
        log(f"Script not found: {step.script_path.relative_to(PROJECT_ROOT)}", "WARN")
        step.status = "missing"
        step.error = "script not found"
        return not step.required

    if dry_run:
        log(f"[DRY RUN] Would run: {PYTHON} {step.script_path}")
        step.status = "dry_run"
        return True

    cmd = [PYTHON, str(step.script_path)] + step.args
    log(f"Running: {step.script} {' '.join(step.args)}".strip())

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,   # let output stream to terminal
            text=True,
            timeout=3600,           # 1 hour max per step
        )
        step.duration = time.time() - start

        if result.returncode != 0:
            step.status = "failed"
            step.error = f"exit code {result.returncode}"
            log(f"FAILED ({step.duration:.1f}s): {step.name}", "ERROR")
            return False
        else:
            step.status = "passed"
            log(f"DONE ({step.duration:.1f}s): {step.name}")
            return True

    except subprocess.TimeoutExpired:
        step.duration = time.time() - start
        step.status = "timeout"
        step.error = "exceeded 1 hour timeout"
        log(f"TIMEOUT: {step.name}", "ERROR")
        return False

    except Exception as e:
        step.duration = time.time() - start
        step.status = "error"
        step.error = str(e)
        log(f"ERROR: {step.name}: {e}", "ERROR")
        return False


def print_summary(steps: list[PipelineStep], total_time: float):
    """Print a formatted summary of all pipeline steps."""
    print()
    print("=" * 72)
    print("PIPELINE SUMMARY")
    print("=" * 72)

    status_colors = {
        "passed": "\033[32m",
        "failed": "\033[31m",
        "timeout": "\033[31m",
        "error": "\033[31m",
        "skipped": "\033[33m",
        "missing": "\033[33m",
        "dry_run": "\033[36m",
        "pending": "\033[90m",
    }
    reset = "\033[0m"

    for i, step in enumerate(steps, 1):
        color = status_colors.get(step.status, "")
        status_str = step.status.upper()
        time_str = f"{step.duration:.1f}s" if step.duration > 0 else "--"
        print(f"  {i}. [{color}{status_str:>8}{reset}] {step.name} ({time_str})")
        if step.error:
            print(f"     Error: {step.error}")

    passed = sum(1 for s in steps if s.status == "passed")
    failed = sum(1 for s in steps if s.status in ("failed", "timeout", "error"))
    skipped = sum(1 for s in steps if s.status in ("skipped", "missing"))

    print()
    print(f"Total time: {total_time:.1f}s")
    print(f"Results:    {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 72)

    # Write summary to log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, "w") as f:
        f.write(f"Pipeline run: {datetime.now().isoformat()}\n")
        f.write(f"Total time: {total_time:.1f}s\n\n")
        for i, step in enumerate(steps, 1):
            f.write(f"{i}. [{step.status}] {step.name} ({step.duration:.1f}s)\n")
            if step.error:
                f.write(f"   Error: {step.error}\n")
    log(f"Log written to: {log_file.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full GENIE myeloid co-occurrence analysis pipeline"
    )
    parser.add_argument(
        "--skip-esm2",
        action="store_true",
        help="Skip ESM-2 variant scoring (requires GPU or is slow on CPU)",
    )
    parser.add_argument(
        "--skip-package",
        action="store_true",
        help="Skip Piazza package generation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing anything",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from step N (1-indexed), skipping earlier steps",
    )
    parser.add_argument(
        "--skip-ai-research",
        action="store_true",
        help="Skip AI research steps (five-gene co-occurrence, ablation analysis)",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization steps (funnel chart, heatmap, pipeline diagram)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip 40-profile benchmark validation",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop the pipeline on the first required step failure",
    )
    args = parser.parse_args()

    skip_flags = {
        "skip_esm2": args.skip_esm2,
        "skip_package": args.skip_package,
        "skip_ai_research": args.skip_ai_research,
        "skip_viz": args.skip_viz,
        "skip_benchmark": args.skip_benchmark,
    }

    print("=" * 72)
    print("GENIE MYELOID CO-OCCURRENCE ANALYSIS PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Python:  {PYTHON}")
    print("=" * 72)
    print()

    # Step 0: Validate data
    log("Validating GENIE data...")
    if not validate_genie_data():
        if not args.dry_run:
            log("Data validation failed. Aborting.", "ERROR")
            sys.exit(1)
        else:
            log("[DRY RUN] Data validation would be required.", "WARN")
    else:
        log("GENIE data validated.")
    print()

    # Run pipeline steps
    pipeline_start = time.time()

    for i, step in enumerate(PIPELINE, 1):
        # Skip if before start-from
        if i < args.start_from:
            step.status = "skipped"
            continue

        # Skip if flag is set
        if step.skip_flag and skip_flags.get(step.skip_flag, False):
            log(f"Skipping (--{step.skip_flag.replace('_', '-')}): {step.name}")
            step.status = "skipped"
            continue

        print(f"\n--- Step {i}/{len(PIPELINE)}: {step.name} ---")
        success = run_step(step, dry_run=args.dry_run)

        if not success and step.required and args.stop_on_failure:
            log("Stopping pipeline due to --stop-on-failure.", "ERROR")
            break

    total_time = time.time() - pipeline_start

    # Summary
    print_summary(PIPELINE, total_time)

    # Exit code
    any_required_failed = any(
        s.status in ("failed", "timeout", "error")
        for s in PIPELINE
        if s.required
    )
    sys.exit(1 if any_required_failed else 0)


if __name__ == "__main__":
    main()
