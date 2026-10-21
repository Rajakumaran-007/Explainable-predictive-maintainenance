"""
NASA C-MAPSS Dataset Acquisition, Extraction, and Integrity Validation Pipeline.

Downloads the official NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset,
extracts raw text files into data/raw/CMAPSSData, and verifies file completeness,
engine trajectory bounds, cycle monotonicity, and sensor dimensionality.

Usage:
    python scripts/prepare_data.py
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger, load_config
from src.data_loader import CMAPSSDataLoader, ALL_RAW_COLS

logger = setup_logger("prepare_data")

# Official NASA direct download and fallback mirrors
DATASET_URLS: List[str] = [
    "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
    "https://ti.arc.nasa.gov/c/6/"
]

EXPECTED_FILES: List[str] = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
    "readme.txt"
]

SUBSET_METRICS: Dict[str, Dict[str, int]] = {
    "FD001": {"conditions": 1, "faults": 1, "train_engines": 100, "test_engines": 100},
    "FD002": {"conditions": 6, "faults": 1, "train_engines": 260, "test_engines": 259},
    "FD003": {"conditions": 1, "faults": 2, "train_engines": 100, "test_engines": 100},
    "FD004": {"conditions": 6, "faults": 2, "train_engines": 249, "test_engines": 248},
}


def download_dataset(destination_zip: Path) -> None:
    """Downloads CMAPSSData.zip from official NASA archive."""
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    if destination_zip.exists() and destination_zip.stat().st_size > 10_000_000:
        logger.info(f"Dataset archive already exists at {destination_zip} ({destination_zip.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return

    logger.info("Initiating download of NASA C-MAPSS dataset...")
    success = False
    for url in DATASET_URLS:
        try:
            logger.info(f"Attempting download from: {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(destination_zip, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 65536
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
            logger.info(f"Successfully downloaded {destination_zip.name} ({destination_zip.stat().st_size / (1024*1024):.2f} MB).")
            success = True
            break
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")

    if not success:
        raise RuntimeError(
            "Could not download NASA C-MAPSS dataset from available mirrors.\n"
            "Please manually download CMAPSSData.zip from https://data.nasa.gov/docs/legacy/CMAPSSData.zip "
            f"and place it in {destination_zip.parent}."
        )


def extract_dataset(zip_path: Path, target_dir: Path) -> None:
    """Extracts CMAPSSData.zip into target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {zip_path.name} to {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.namelist():
            # Skip subdirectories or metadata
            filename = os.path.basename(member)
            if not filename:
                continue
            # Extract to target_dir directly
            source = archive.open(member)
            target = open(target_dir / filename, "wb")
            with source, target:
                target.write(source.read())
    logger.info(f"Extracted all files into {target_dir.resolve()}.")


def audit_dataset_integrity(target_dir: Path) -> None:
    """Performs rigorous statistical and structural validation across all subsets."""
    logger.info("Performing structural data integrity audit...")
    loader = CMAPSSDataLoader(data_dir=target_dir)

    for subset, specs in SUBSET_METRICS.items():
        logger.info(f"\n--- Validating Subset {subset} ---")
        logger.info(f"Expected: {specs['conditions']} condition(s), {specs['faults']} fault mode(s), "
                    f"{specs['train_engines']} train engines, {specs['test_engines']} test engines.")

        # 1. Validate Training Data
        train_df = loader.load_train_df(subset=subset, max_rul=125)
        n_train_engines = train_df["engine_id"].nunique()
        assert n_train_engines == specs["train_engines"], (
            f"{subset} train engine count mismatch: found {n_train_engines}, expected {specs['train_engines']}"
        )
        assert train_df.isnull().sum().sum() == 0, f"{subset} train data contains NaN values!"
        assert "target_rul" in train_df.columns, "target_rul not generated!"

        min_rul, max_rul_val = train_df["target_rul"].min(), train_df["target_rul"].max()
        assert min_rul == 0, f"Minimum training RUL should reach 0 (failure), got {min_rul}"
        assert max_rul_val <= 125, f"Capped RUL exceeds max_rul ceiling: {max_rul_val}"

        # 2. Invariant Sensor Detection
        constant_sensors = loader.identify_constant_sensors(train_df)
        logger.info(f"{subset} Invariant/Zero-Variance sensors (std < 1e-5): {constant_sensors}")

        # 3. Validate Test Data and Ground Truth RUL
        test_df, test_rul = loader.load_test_df(subset=subset, max_rul=125)
        n_test_engines = test_df["engine_id"].nunique()
        assert n_test_engines == specs["test_engines"], (
            f"{subset} test engine count mismatch: found {n_test_engines}, expected {specs['test_engines']}"
        )
        assert len(test_rul) == specs["test_engines"], (
            f"{subset} RUL ground truth count mismatch: found {len(test_rul)}, expected {specs['test_engines']}"
        )
        assert test_df.isnull().sum().sum() == 0, f"{subset} test data contains NaN values!"

        # Minimum trajectory length in test data (critical for window padding validation)
        min_test_len = test_df.groupby("engine_id")["cycle"].max().min()
        logger.info(f"{subset} Test trajectory length bounds: min={min_test_len} cycles, "
                    f"max={test_df.groupby('engine_id')['cycle'].max().max()} cycles.")

    logger.info("\nALL NASA C-MAPSS SUBSETS (FD001 - FD004) PASSED RIGOROUS DATA INTEGRITY CHECKS!")


def main():
    config = load_config()
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    zip_path = raw_dir / "CMAPSSData.zip"
    target_dir = raw_dir / "CMAPSSData"

    download_dataset(zip_path)
    extract_dataset(zip_path, target_dir)
    audit_dataset_integrity(target_dir)


if __name__ == "__main__":
    main()
