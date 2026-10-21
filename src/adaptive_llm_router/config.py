
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "raw" / "prompt_classifier_dataset_merged.jsonl"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

