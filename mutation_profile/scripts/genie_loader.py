"""
genie_loader.py -- Shared data loading module for GENIE v19.0 myeloid analysis.

Loads mutation, clinical, and gene panel data once. Provides filtering,
co-occurrence computation, and panel-aware sample selection so that all
analysis scripts operate on identically prepared data.

Inputs:
    - mutation_profile/data/genie/raw/data_mutations_extended.txt
    - mutation_profile/data/genie/raw/data_clinical_sample.txt
    - mutation_profile/data/genie/raw/genomic_information_*.txt  (gene panel files)

Outputs:
    None (library module, imported by other scripts).

Usage:
    from genie_loader import GenieLoader
    loader = GenieLoader()

Runtime: ~8 seconds (initial data load)
Dependencies: pandas, numpy, scipy
"""

from __future__ import annotations

import glob
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MYELOID_ONCOTREE_CODES: set[str] = {
    # Broad categories
    "AML", "MDS", "MPN", "CMML", "JMML", "MDS/MPN",
    # MPN subtypes
    "ET", "PMF", "PV", "SM", "CEL", "CNL", "MPN-U",
    # MDS/MPN overlap
    "MDSMPNU", "aCML", "CMML-1", "CMML-2",
    # MDS subtypes (WHO 2016 / earlier)
    "RARS", "RCMD", "RAEB", "RAEB-T",
    "MDS-RS", "MDS-SLD", "MDS-MLD", "MDS-EB1", "MDS-EB2", "MDS-U",
    # AML subtypes
    "AML-MRC", "APL", "AML-NOS",
}

NONCODING_CLASSIFICATIONS: set[str] = {
    "Intron", "Silent",
    "3'UTR", "5'UTR", "3'Flank", "5'Flank",
    "IGR", "RNA", "Splice_Region",
}

TARGET_GENES: list[str] = [
    "ASXL1", "TET2", "SRSF2", "SF3B1", "RUNX1", "TP53", "FLT3", "NPM1",
    "NRAS", "KRAS", "CBL", "EZH2", "U2AF1", "STAG2", "BCOR", "BCORL1",
    "DDX41", "DNMT3A", "IDH1", "IDH2", "PTPN11", "JAK2", "CALR", "MPL",
    "PHF6", "WT1", "CEBPA", "GATA2", "ZRSR2", "RAD21", "SMC1A", "SMC3",
    "SETBP1", "CSF3R",
]


class GENIEData:
    """Load, filter, and provide access to GENIE v19.0 data."""

    def __init__(self, data_dir: str | Path | None = None):
        """Load all GENIE data.

        Parameters
        ----------
        data_dir : str or Path, optional
            Path to ``mutation_profile/data/genie/raw/``.  Auto-detected
            relative to this source file when omitted.
        """
        if data_dir is None:
            data_dir = (
                Path(__file__).resolve().parent.parent / "data" / "genie" / "raw"
            )
        self._data_dir = Path(data_dir)
        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self._data_dir}")

        print(f"[GENIEData] Loading from {self._data_dir} ...")

        self._mutations = self._load_mutations()
        self._clinical_sample = self._load_clinical_sample()
        self._clinical_patient = self._load_clinical_patient()
        self._clinical = self._merge_clinical()
        self._gene_panels = self._load_gene_panels()
        self._gene_matrix = self._load_gene_matrix()

        print(
            f"[GENIEData] Ready. "
            f"{len(self._mutations):,} mutations | "
            f"{self._clinical['SAMPLE_ID'].nunique():,} clinical samples | "
            f"{len(self._gene_panels)} gene panels"
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def mutations(self) -> pd.DataFrame:
        """Raw mutations DataFrame (all rows from data_mutations_extended)."""
        return self._mutations

    @property
    def clinical(self) -> pd.DataFrame:
        """Merged clinical DataFrame (sample + patient)."""
        return self._clinical

    @property
    def gene_panels(self) -> dict[str, set[str]]:
        """Mapping panel_id -> set of gene symbols."""
        return self._gene_panels

    @property
    def gene_matrix(self) -> pd.DataFrame:
        """Gene matrix DataFrame (SAMPLE_ID, mutations, cna, sv columns)."""
        return self._gene_matrix

    # ------------------------------------------------------------------
    # Filtering helpers
    # ------------------------------------------------------------------

    def get_myeloid_samples(self) -> set[str]:
        """Return set of SAMPLE_IDs with a myeloid ONCOTREE_CODE."""
        mask = self._clinical["ONCOTREE_CODE"].isin(MYELOID_ONCOTREE_CODES)
        return set(self._clinical.loc[mask, "SAMPLE_ID"])

    def get_coding_mutations(
        self, samples: set[str] | None = None
    ) -> pd.DataFrame:
        """Filter to coding variants only, optionally restricted to a sample set.

        Removes rows whose Variant_Classification is in NONCODING_CLASSIFICATIONS.
        """
        mask = ~self._mutations["Variant_Classification"].isin(
            NONCODING_CLASSIFICATIONS
        )
        df = self._mutations.loc[mask]
        if samples is not None:
            df = df.loc[df["Tumor_Sample_Barcode"].isin(samples)]
        return df

    def get_hypermutated_samples(
        self, genes: list[str], threshold: int = 20
    ) -> set[str]:
        """Return SAMPLE_IDs with >threshold coding mutations across *genes*.

        Useful for excluding technical outliers / hypermutated samples.
        """
        coding = self.get_coding_mutations()
        subset = coding.loc[coding["Hugo_Symbol"].isin(genes)]
        counts = subset.groupby("Tumor_Sample_Barcode").size()
        return set(counts[counts > threshold].index)

    def samples_covering_genes(
        self, gene_list: list[str], myeloid_only: bool = True
    ) -> set[str]:
        """Return SAMPLE_IDs where ALL genes in *gene_list* are on the panel.

        Uses the gene matrix (data_gene_matrix.txt) to map each sample to its
        sequencing panel, then checks panel gene content.
        """
        # Build set of panels that cover every requested gene
        gene_set = set(gene_list)
        valid_panels: set[str] = set()
        for panel_id, panel_genes in self._gene_panels.items():
            if gene_set.issubset(panel_genes):
                valid_panels.add(panel_id)

        # Samples whose mutations-assay panel is in the valid set
        mask = self._gene_matrix["mutations"].isin(valid_panels)
        covered = set(self._gene_matrix.loc[mask, "SAMPLE_ID"])

        if myeloid_only:
            covered &= self.get_myeloid_samples()

        return covered

    # ------------------------------------------------------------------
    # Mutation matrices
    # ------------------------------------------------------------------

    def get_gene_mutation_matrix(
        self, genes: list[str], samples: set[str]
    ) -> dict[str, set[str]]:
        """Return ``{gene: set(sample_ids)}`` for coding mutations in *genes*."""
        coding = self.get_coding_mutations(samples=samples)
        subset = coding.loc[coding["Hugo_Symbol"].isin(genes)]
        result: dict[str, set[str]] = {}
        for gene, grp in subset.groupby("Hugo_Symbol"):
            result[gene] = set(grp["Tumor_Sample_Barcode"])
        # Ensure every requested gene appears even if zero hits
        for g in genes:
            result.setdefault(g, set())
        return result

    def get_variant_mutation_matrix(
        self,
        variants_dict: dict[str, str],
        samples: set[str],
    ) -> dict[str, set[str]]:
        """Return ``{gene_variant: set(sample_ids)}`` for specific AA changes.

        Parameters
        ----------
        variants_dict : dict
            ``{'DNMT3A': 'R882H', 'IDH2': 'R140Q', ...}``
            Values are the short-form amino-acid change WITHOUT the ``p.`` prefix.
        samples : set
            Restrict to these sample IDs.
        """
        coding = self.get_coding_mutations(samples=samples)
        result: dict[str, set[str]] = {}
        for gene, aa_change in variants_dict.items():
            target_hgvs = f"p.{aa_change}"
            mask = (
                (coding["Hugo_Symbol"] == gene)
                & (coding["HGVSp_Short"] == target_hgvs)
            )
            key = f"{gene}_{aa_change}"
            result[key] = set(coding.loc[mask, "Tumor_Sample_Barcode"])
        return result

    # ------------------------------------------------------------------
    # Co-occurrence statistics
    # ------------------------------------------------------------------

    def compute_cooccurrence(
        self,
        gene1: str,
        gene2: str,
        samples: set[str] | None = None,
    ) -> dict:
        """Compute co-occurrence statistics for a gene pair.

        Parameters
        ----------
        gene1, gene2 : str
            Hugo gene symbols.
        samples : set, optional
            Denominator sample set.  Defaults to myeloid samples that cover
            both genes.

        Returns
        -------
        dict with keys: observed, expected, OE_ratio, odds_ratio, p_value,
        n_gene1, n_gene2, n_denominator, contingency_table
        """
        if samples is None:
            samples = self.samples_covering_genes(
                [gene1, gene2], myeloid_only=True
            )

        mat = self.get_gene_mutation_matrix([gene1, gene2], samples)
        s1 = mat[gene1]
        s2 = mat[gene2]

        n = len(samples)
        n1 = len(s1)
        n2 = len(s2)
        both = len(s1 & s2)
        only1 = n1 - both
        only2 = n2 - both
        neither = n - n1 - n2 + both

        # Expected co-occurrence under independence
        expected = (n1 * n2) / n if n > 0 else 0.0
        oe_ratio = both / expected if expected > 0 else float("inf")

        # 2x2 contingency table for Fisher's exact test
        #              gene2+     gene2-
        # gene1+     [ both,      only1  ]
        # gene1-     [ only2,     neither ]
        table = np.array([[both, only1], [only2, neither]])
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

        return {
            "gene1": gene1,
            "gene2": gene2,
            "observed": both,
            "expected": round(expected, 2),
            "OE_ratio": round(oe_ratio, 3),
            "odds_ratio": round(odds_ratio, 4) if np.isfinite(odds_ratio) else odds_ratio,
            "p_value": p_value,
            "n_gene1": n1,
            "n_gene2": n2,
            "n_denominator": n,
            "contingency_table": table,
        }

    def compute_all_pairwise(
        self,
        genes: list[str],
        samples: set[str] | None = None,
    ) -> list[dict]:
        """Compute co-occurrence for all gene pairs with BH correction.

        Parameters
        ----------
        genes : list of str
            Gene symbols.  All unique pairs are tested.
        samples : set, optional
            If given, this fixed set is used for every pair.  Otherwise each
            pair gets its own panel-aware denominator.

        Returns
        -------
        list of dict, sorted by p_value, each dict has an additional
        ``p_value_bh`` key for Benjamini-Hochberg adjusted p-values.
        """
        results: list[dict] = []
        for g1, g2 in combinations(genes, 2):
            pair_samples = samples
            if pair_samples is None:
                pair_samples = self.samples_covering_genes(
                    [g1, g2], myeloid_only=True
                )
            if len(pair_samples) == 0:
                continue
            res = self.compute_cooccurrence(g1, g2, samples=pair_samples)
            results.append(res)

        # Benjamini-Hochberg correction
        if results:
            results.sort(key=lambda r: r["p_value"])
            m = len(results)
            for rank_i, res in enumerate(results, start=1):
                res["p_value_bh"] = min(
                    res["p_value"] * m / rank_i, 1.0
                )
            # Enforce monotonicity (BH adjusted values must be non-decreasing
            # when walking backwards from largest rank)
            for i in range(m - 2, -1, -1):
                results[i]["p_value_bh"] = min(
                    results[i]["p_value_bh"], results[i + 1]["p_value_bh"]
                )

        return results

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_mutations(self) -> pd.DataFrame:
        """Load data_mutations_extended.txt."""
        path = self._data_dir / "data_mutations_extended.txt"
        print(f"  Loading mutations from {path.name} ...")
        df = pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
            comment="#",
            dtype={
                "Chromosome": str,
                "NCBI_Build": str,
                "dbSNP_RS": str,
                "Tumor_Sample_Barcode": str,
                "Hugo_Symbol": str,
                "Variant_Classification": str,
                "HGVSp_Short": str,
                "t_ref_count": "Int64",
                "t_alt_count": "Int64",
                "t_depth": "Int64",
                "n_ref_count": "Int64",
                "n_alt_count": "Int64",
                "n_depth": "Int64",
            },
        )
        print(f"  -> {len(df):,} mutation rows loaded")
        return df

    def _load_clinical_sample(self) -> pd.DataFrame:
        """Load data_clinical_sample.txt (skip 4 comment/header lines)."""
        path = self._data_dir / "data_clinical_sample.txt"
        print(f"  Loading {path.name} ...")
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
        print(f"  -> {len(df):,} sample rows")
        return df

    def _load_clinical_patient(self) -> pd.DataFrame:
        """Load data_clinical_patient.txt (skip 4 comment/header lines)."""
        path = self._data_dir / "data_clinical_patient.txt"
        print(f"  Loading {path.name} ...")
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
        print(f"  -> {len(df):,} patient rows")
        return df

    def _merge_clinical(self) -> pd.DataFrame:
        """Merge sample and patient clinical data on PATIENT_ID."""
        merged = self._clinical_sample.merge(
            self._clinical_patient, on="PATIENT_ID", how="left"
        )
        print(f"  -> {len(merged):,} merged clinical rows")
        return merged

    def _load_gene_panels(self) -> dict[str, set[str]]:
        """Load all data_gene_panel_*.txt files into a dict."""
        pattern = str(self._data_dir / "data_gene_panel_*.txt")
        files = sorted(glob.glob(pattern))
        panels: dict[str, set[str]] = {}
        for fp in files:
            panel_id, genes = self._parse_gene_panel(fp)
            panels[panel_id] = genes
        print(f"  -> {len(panels)} gene panels loaded")
        return panels

    @staticmethod
    def _parse_gene_panel(filepath: str) -> tuple[str, set[str]]:
        """Parse a single gene panel file.

        Format:
            stable_id: PANEL-NAME
            description: ...
            gene_list:\\tGENE1\\tGENE2\\t...
        """
        panel_id = ""
        genes: set[str] = set()
        with open(filepath, "r") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("stable_id:"):
                    panel_id = line.split(":", 1)[1].strip()
                elif line.startswith("gene_list:"):
                    parts = line.split("\t")
                    genes = {g.strip() for g in parts[1:] if g.strip()}
        return panel_id, genes

    def _load_gene_matrix(self) -> pd.DataFrame:
        """Load data_gene_matrix.txt (sample -> panel mapping)."""
        path = self._data_dir / "data_gene_matrix.txt"
        print(f"  Loading {path.name} ...")
        df = pd.read_csv(path, sep="\t", low_memory=False)
        print(f"  -> {len(df):,} gene matrix rows")
        return df


# ---------------------------------------------------------------------------
# Convenience: run as script to verify data loading
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gd = GENIEData()

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    # Total samples
    total_samples = gd.clinical["SAMPLE_ID"].nunique()
    print(f"\nTotal unique clinical samples: {total_samples:,}")

    # Myeloid samples
    myeloid = gd.get_myeloid_samples()
    print(f"Myeloid samples (OncoTree): {len(myeloid):,}")

    # Coding mutations in myeloid
    coding_myeloid = gd.get_coding_mutations(samples=myeloid)
    print(f"Coding mutations in myeloid samples: {len(coding_myeloid):,}")

    # Hypermutated
    hyper = gd.get_hypermutated_samples(TARGET_GENES, threshold=20)
    print(f"Hypermutated samples (>{20} coding muts in target genes): {len(hyper):,}")

    # Panel coverage for target genes
    covered = gd.samples_covering_genes(TARGET_GENES, myeloid_only=True)
    print(f"Myeloid samples covering all 34 target genes: {len(covered):,}")

    # Sample co-occurrence: DNMT3A vs TET2
    print("\n--- Co-occurrence: DNMT3A vs TET2 ---")
    result = gd.compute_cooccurrence("DNMT3A", "TET2")
    for k, v in result.items():
        if k == "contingency_table":
            print(f"  {k}:")
            print(f"    {v}")
        else:
            print(f"  {k}: {v}")
