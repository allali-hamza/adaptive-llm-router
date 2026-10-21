from src.adaptive_llm_router.config import (
    EMBEDDING_MODEL,
    DATASET_PATH,
    PROCESSED_DIR,
)
from src.adaptive_llm_router.data.loader import data_load
from sentence_transformers import SentenceTransformer
import numpy as np

dataset = data_load(dataset_file=DATASET_PATH)

embedding_model = SentenceTransformer(
    model_name_or_path=EMBEDDING_MODEL
)

prompts = dataset["prompt"].tolist()

embeddings = embedding_model.encode(
    prompts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

np.save(
    file=PROCESSED_DIR / "prompt_embeddings.npy",
    arr=embeddings
)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings dtype: {embeddings.dtype}")