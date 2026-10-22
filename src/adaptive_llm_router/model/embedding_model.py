from sentence_transformers import SentenceTransformer

from adaptive_llm_router.config import EMBEDDING_MODEL_NAME


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)