
from pathlib import Path
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).parent.parent.parent
DATASET_PATH = ROOT_DIR / "data" / "raw" / "prompt_classifier_dataset_merged.jsonl"

PROCESSED_DIR = ROOT_DIR / "data" / "processed"


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


RANDOM_STATE = 42

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

EMBEDDING_FILE = PROCESSED_DIR / "prompt_classifier_dataset_merged_embeddings.npy"
METADATA_FILE = PROCESSED_DIR / "prompt_classifier_dataset_merged_metadata.csv"

EVALUATION_DIR = ROOT_DIR / "evaluation"


CAPABILITY_TABLE_PATH = ROOT_DIR / "data" / "raw" / "model_capability_table_seed.csv"