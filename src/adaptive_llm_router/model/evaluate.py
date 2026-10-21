import joblib
from src.adaptive_llm_router.config import (
    PROCESSED_DIR, EMBEDDING_FILE, METADATA_FILE, ARTIFACTS_DIR
)
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

X = np.load(file=EMBEDDING_FILE)
metadata = pd.read_csv(METADATA_FILE)

test_idx = np.load(PROCESSED_DIR / "test_idx.npy")

X_test = X[test_idx]


# Load models
model_task_type = joblib.load(
    ARTIFACTS_DIR / "task_type.joblib"
)

model_difficulty = joblib.load(
    ARTIFACTS_DIR / "difficulty.joblib"
)

model_required_quality = joblib.load(
    ARTIFACTS_DIR / "required_quality.joblib"
)

model_risk_level = joblib.load(
    ARTIFACTS_DIR / "risk_level.joblib"
)

model_expected_output_tokens = joblib.load(
    ARTIFACTS_DIR / "expected_output_tokens.joblib"
)

model_latency_sensitivity = joblib.load(
    ARTIFACTS_DIR / "latency_sensitivity.joblib"
)


# Test targets
y_task_type = (
    metadata["task_type"]
    .to_numpy()[test_idx]
)

y_difficulty = (
    metadata["difficulty"]
    .to_numpy()[test_idx]
)

y_required_quality = (
    metadata["required_quality"]
    .to_numpy()[test_idx]
)

y_risk_level = (
    metadata["risk_level"]
    .to_numpy()[test_idx]
)

y_expected_output_tokens = (
    metadata["expected_output_tokens"]
    .to_numpy()[test_idx]
)

y_latency_sensitivity = (
    metadata["latency_sensitivity"]
    .to_numpy()[test_idx]
)




def evaluate_classification(model, X_test, y_test, model_name):
    y_predicted = model.predict(X_test)

    accuracy = accuracy_score(
        y_true=y_test,
        y_pred=y_predicted
    )

    precision = precision_score(
        y_true=y_test,
        y_pred=y_predicted,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true=y_test,
        y_pred=y_predicted,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
            y_true=y_test,
            y_pred=y_predicted,
            average="macro",
            zero_division=0
        )

    print(model_name)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_true=y_test,
        y_pred=y_predicted,
        zero_division=0
    ))

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def evaluate_regression(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(model_name)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }




results = {}

results["task_type"] = evaluate_classification(
    model=model_task_type,
    X_test=X_test,
    y_test=y_task_type,
    model_name="task_type",
)

results["difficulty"] = evaluate_regression(
    model=model_difficulty,
    X_test=X_test,
    y_test=y_difficulty,
    model_name="difficulty",
)

results["required_quality"] = evaluate_regression(
    model=model_required_quality,
    X_test=X_test,
    y_test=y_required_quality,
    model_name="required_quality",
)

results["risk_level"] = evaluate_classification(
    model=model_risk_level,
    X_test=X_test,
    y_test=y_risk_level,
    model_name="risk_level",
)

results["expected_output_tokens"] = evaluate_regression(
    model=model_expected_output_tokens,
    X_test=X_test,
    y_test=y_expected_output_tokens,
    model_name="expected_output_tokens",
)

results["latency_sensitivity"] = evaluate_classification(
    model=model_latency_sensitivity,
    X_test=X_test,
    y_test=y_latency_sensitivity,
    model_name="latency_sensitivity",
)