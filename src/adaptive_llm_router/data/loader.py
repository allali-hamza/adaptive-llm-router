import pandas as pd
from pathlib import Path
from src.adaptive_llm_router.config import DATASET_PATH


def data_load(dataset_file):
    return pd.read_json(
        path_or_buf=dataset_file,
        lines=True
    )


if __name__ == "__main__":
    df = data_load(
        dataset_file=DATASET_PATH
    )

    print(df.shape)

    print(df.columns)
    print(df.iloc[0])

    print(df.duplicated().sum())
    print(df.isnull().sum())




