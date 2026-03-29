#!/usr/bin/env python3
"""
download_genie.py -- Download/update AACR GENIE data from Synapse.

Uses synapseclient to download GENIE v19.0-public.
Reads SYNAPSE_AUTH_TOKEN from the project .env file.
Downloads to mutation_profile/data/genie/raw/ and creates
convenience symlinks for the key analysis files.

Inputs:
    - .env (SYNAPSE_AUTH_TOKEN)
    - Synapse GENIE v19.0-public dataset (remote)

Outputs:
    - mutation_profile/data/genie/raw/data_mutations_extended.txt
    - mutation_profile/data/genie/raw/data_clinical_sample.txt
    - mutation_profile/data/genie/raw/genomic_information_*.txt

Usage:
    source ~/projects/helse/.venv/bin/activate
    python mutation_profile/scripts/download_genie.py [--force]

Runtime: ~5-15 minutes (network-dependent)
Dependencies: synapseclient, python-dotenv
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # mrna-hematology-research/
DATA_DIR = PROJECT_ROOT / "mutation_profile" / "data" / "genie"
RAW_DIR = DATA_DIR / "raw"
LOG_DIR = PROJECT_ROOT / "mutation_profile" / "logs"

# Synapse entity for the full GENIE dataset (syn3380222 = top-level project)
GENIE_PROJECT_SYN = "syn3380222"
# The 19.0-public release folder
RELEASE_VERSION = "19.0-public"
RELEASE_FOLDER_PATH = f"Data Releases/Release {RELEASE_VERSION}/"

# Key files we create symlinks for (raw name -> symlink name in DATA_DIR)
SYMLINK_TARGETS = {
    "data_mutations_extended.txt": "mutations.txt",
    "data_clinical_sample.txt": "clinical_sample.txt",
    "data_clinical_patient.txt": "clinical_patient.txt",
    "data_gene_matrix.txt": "gene_matrix.txt",
    "data_CNA.txt": "cna.txt",
    "genomic_information.txt": "genomic_info.txt",
    "assay_information.txt": "assay_info.txt",
}


def load_env() -> str:
    """Load SYNAPSE_AUTH_TOKEN from .env, return the token string."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        sys.exit("ERROR: python-dotenv not installed. Run: pip install python-dotenv")

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        sys.exit(f"ERROR: .env not found at {env_path}")
    load_dotenv(env_path)

    token = os.environ.get("SYNAPSE_AUTH_TOKEN", "").strip()
    if not token:
        sys.exit("ERROR: SYNAPSE_AUTH_TOKEN is empty or missing in .env")
    return token


def authenticate(token: str):
    """Return an authenticated Synapse client."""
    try:
        import synapseclient
    except ImportError:
        sys.exit("ERROR: synapseclient not installed. Run: pip install synapseclient")

    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)
    print(f"Authenticated as: {syn.getUserProfile().get('userName', 'unknown')}")
    return syn


def find_release_folder(syn, project_syn: str, release_path: str):
    """Walk the Synapse folder hierarchy to find the release folder entity."""
    import synapseclient

    parts = [p for p in release_path.strip("/").split("/") if p]
    parent_id = project_syn

    for part in parts:
        children = list(syn.getChildren(parent_id, includeTypes=["folder"]))
        match = [c for c in children if c["name"] == part]
        if not match:
            available = [c["name"] for c in children]
            sys.exit(
                f"ERROR: folder '{part}' not found under {parent_id}.\n"
                f"Available: {available}"
            )
        parent_id = match[0]["id"]
        print(f"  Found: {part} -> {parent_id}")

    return parent_id


def download_release(syn, folder_id: str, dest: Path, force: bool = False):
    """Download all files from the release folder (and sub-folders) to dest."""
    dest.mkdir(parents=True, exist_ok=True)
    children = list(syn.getChildren(folder_id, includeTypes=["file", "folder"]))
    file_count = 0
    skip_count = 0

    for child in children:
        if child["type"] == "org.sagebionetworks.repo.model.Folder":
            sub_dest = dest / child["name"]
            print(f"  Entering sub-folder: {child['name']}/")
            sub_files, sub_skips = _download_folder(syn, child["id"], sub_dest, force)
            file_count += sub_files
            skip_count += sub_skips
        else:
            local_path = dest / child["name"]
            if local_path.exists() and not force:
                skip_count += 1
                continue
            print(f"  Downloading: {child['name']} ...", end="", flush=True)
            entity = syn.get(child["id"], downloadLocation=str(dest))
            # synapseclient may place file in a cache path; move if needed
            downloaded = Path(entity.path)
            if downloaded != local_path:
                downloaded.rename(local_path)
            file_count += 1
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f" {size_mb:.1f} MB")

    return file_count, skip_count


def _download_folder(syn, folder_id: str, dest: Path, force: bool):
    """Recursively download a sub-folder."""
    dest.mkdir(parents=True, exist_ok=True)
    children = list(syn.getChildren(folder_id, includeTypes=["file", "folder"]))
    file_count = 0
    skip_count = 0

    for child in children:
        if child["type"] == "org.sagebionetworks.repo.model.Folder":
            sub_files, sub_skips = _download_folder(
                syn, child["id"], dest / child["name"], force
            )
            file_count += sub_files
            skip_count += sub_skips
        else:
            local_path = dest / child["name"]
            if local_path.exists() and not force:
                skip_count += 1
                continue
            print(f"  Downloading: {child['name']} ...", end="", flush=True)
            entity = syn.get(child["id"], downloadLocation=str(dest))
            downloaded = Path(entity.path)
            if downloaded != local_path:
                downloaded.rename(local_path)
            file_count += 1
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f" {size_mb:.1f} MB")

    return file_count, skip_count


def create_symlinks(raw_dir: Path, data_dir: Path):
    """Create convenience symlinks in data/genie/ pointing to raw files."""
    release_dir = raw_dir / "Data Releases" / f"Release {RELEASE_VERSION}"

    # Try both the release sub-folder and the raw dir itself (flat download)
    for source_dir in [release_dir, raw_dir]:
        if not source_dir.is_dir():
            continue
        for raw_name, link_name in SYMLINK_TARGETS.items():
            source = source_dir / raw_name
            link = data_dir / link_name
            if source.exists():
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(source)
                print(f"  Symlink: {link_name} -> {source.relative_to(PROJECT_ROOT)}")


def verify_download(raw_dir: Path):
    """Quick sanity check that critical files exist and have reasonable sizes."""
    release_dir = raw_dir / "Data Releases" / f"Release {RELEASE_VERSION}"

    critical_files = {
        "data_mutations_extended.txt": 500,   # expect >500 MB
        "data_clinical_sample.txt": 10,       # expect >10 MB
        "data_gene_matrix.txt": 5,            # expect >5 MB
    }

    # Check in both possible locations
    all_ok = True
    for fname, min_mb in critical_files.items():
        found = False
        for search_dir in [release_dir, raw_dir]:
            fpath = search_dir / fname
            if fpath.exists():
                size_mb = fpath.stat().st_size / (1024 * 1024)
                if size_mb < min_mb:
                    print(f"  WARNING: {fname} is only {size_mb:.1f} MB (expected >{min_mb} MB)")
                    all_ok = False
                else:
                    print(f"  OK: {fname} = {size_mb:.1f} MB")
                found = True
                break
        if not found:
            print(f"  MISSING: {fname}")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download GENIE v19.0 from Synapse")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing download, do not download anything",
    )
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir:     {DATA_DIR}")
    print(f"Raw dir:      {RAW_DIR}")
    print()

    if args.verify_only:
        print("=== Verify existing download ===")
        ok = verify_download(RAW_DIR)
        print()
        print("=== Create symlinks ===")
        create_symlinks(RAW_DIR, DATA_DIR)
        sys.exit(0 if ok else 1)

    # Full download flow
    token = load_env()
    syn = authenticate(token)

    print(f"\n=== Locating release: {RELEASE_FOLDER_PATH} ===")
    folder_id = find_release_folder(syn, GENIE_PROJECT_SYN, RELEASE_FOLDER_PATH)

    print(f"\n=== Downloading to {RAW_DIR} ===")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded, skipped = download_release(syn, folder_id, RAW_DIR, force=args.force)
    print(f"\nDownloaded: {downloaded} files, Skipped: {skipped} (already exist)")

    print("\n=== Creating symlinks ===")
    create_symlinks(RAW_DIR, DATA_DIR)

    print("\n=== Verifying download ===")
    ok = verify_download(RAW_DIR)

    if ok:
        print("\nDone. All critical files present.")
    else:
        print("\nWARNING: Some files missing or undersized. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
