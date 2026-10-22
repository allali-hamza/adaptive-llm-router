import numpy as np

from adaptive_llm_router.config import (
    DATASET_PATH,
    PROCESSED_DIR,
)
from adaptive_llm_router.data.loader import data_load
from adaptive_llm_router.model.embedding_model import embedding_model


def prompt_to_embedding(prompts, embedding_model):

    return embedding_model.encode(
        prompts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def generate_dataset_embeddings(dataset_path):

    dataset = data_load(dataset_file=dataset_path)

    prompts = dataset["prompt"].tolist()

    embeddings = prompt_to_embedding(
        prompts=prompts,
        embedding_model=embedding_model,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / f"{dataset_path.stem}_embeddings.npy"

    np.save(output_path, embeddings)
    dataset.to_csv(
        path_or_buf=PROCESSED_DIR / f"{dataset_path.stem}_metadata.csv",
    )


    print(f"Embeddings saved to: {output_path}")
    print(f"Embeddings shape: {embeddings.shape}")

    return output_path

if __name__ == "__main__":
    generate_dataset_embeddings(
        dataset_path=DATASET_PATH,
    )