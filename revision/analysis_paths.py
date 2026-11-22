"""Locate inputs in a standalone checkout or the analysis workspace."""
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE if (WORKSPACE / "New_Dataset").is_dir() else WORKSPACE / "EcoIrrigate Codes"
SOURCE_CSV = DATA_ROOT / "New_Dataset" / "kolkata_unified_dataset.csv"
