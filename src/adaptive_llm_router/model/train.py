from adaptive_llm_router.config import PROCESSED_DIR, RANDOM_STATE, ARTIFACTS_DIR, EMBEDDING_FILE, METADATA_FILE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import joblib


def train_models():

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(filepath_or_buffer=METADATA_FILE)
    X = np.load(file=EMBEDDING_FILE)

    # Validate data
    if len(X) != len(df):
        raise ValueError("Embeddings and metadata have different lengths")

    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or infinite values")


    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    X_train = X[train_idx]

    np.save(
        PROCESSED_DIR / "train_idx.npy",
        train_idx,
    )

    np.save(
        PROCESSED_DIR / "test_idx.npy",
        test_idx,
    )


    # task_type
    y_task_type = df["task_type"].to_numpy()
    y_train_task_type = y_task_type[train_idx]

    model_task_type = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    print("Training task_type...")
    model_task_type.fit(
        X_train,
        y_train_task_type,
    )

    joblib.dump(
        model_task_type,
        ARTIFACTS_DIR / "task_type.joblib",
    )


    # difficulty score
    y_difficulty = df["difficulty"].to_numpy()
    y_train_difficulty = y_difficulty[train_idx]

    model_difficulty = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        max_depth=3
    )
    print("Training difficulty...")
    model_difficulty.fit(
        X_train,
        y_train_difficulty
    )

    joblib.dump(
        model_difficulty,
        ARTIFACTS_DIR / "difficulty.joblib",
    )


    # required_quality
    y_required_quality = df["required_quality"].to_numpy()
    y_train_required_quality = y_required_quality[train_idx]

    model_required_quality = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        max_depth=3,
    )
    print("Training required_quality...")
    model_required_quality.fit(
        X_train,
        y_train_required_quality,
    )

    joblib.dump(
        model_required_quality,
        ARTIFACTS_DIR / "required_quality.joblib",
    )


    # expected_output_tokens
    y_expected_output_tokens = df["expected_output_tokens"].to_numpy()
    y_train_expected_output_tokens = y_expected_output_tokens[train_idx]

    model_expected_output_tokens = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("Training expected_output_tokens...")
    model_expected_output_tokens.fit(
        X_train,
        y_train_expected_output_tokens,
    )

    joblib.dump(
        model_expected_output_tokens,
        ARTIFACTS_DIR / "expected_output_tokens.joblib",
    )

    # latency_sensitivity
    y_latency_sensitivity = df["latency_sensitivity"].to_numpy()
    y_train_latency_sensitivity = y_latency_sensitivity[train_idx]

    latency_scaler = StandardScaler()
    X_train_latency = latency_scaler.fit_transform(X_train)

    joblib.dump(
        latency_scaler,
        ARTIFACTS_DIR / "latency_sensitivity_scaler.joblib",
    )

    model_latency_sensitivity = LinearSVC(
        C=0.1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    print("Training latency_sensitivity...")
    model_latency_sensitivity.fit(
        X_train_latency,
        y_train_latency_sensitivity,
    )

    joblib.dump(
        model_latency_sensitivity,
        ARTIFACTS_DIR / "latency_sensitivity.joblib",
    )
    print("Training complete")


if __name__ == "__main__":
    train_models()
