from src.adaptive_llm_router.config import PROCESSED_DIR, RANDOM_STATE, ARTIFACTS_DIR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import joblib


embedding_file = PROCESSED_DIR / "prompt_classifier_dataset_merged_embeddings.npy"
metadata_file = PROCESSED_DIR / "prompt_classifier_dataset_merged_metadata.csv"

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_csv(filepath_or_buffer=metadata_file)
X = np.load(file=embedding_file)
y_task_type= df["task_type"]


indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=RANDOM_STATE
)

X_train = X[train_idx]
X_test = X[test_idx]



# classify the task_type
y_task_type = df["task_type"].to_numpy()
y_train_task_type = y_task_type[train_idx]
y_test_task_type = y_task_type[test_idx]

model_task_type = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE
)

model_task_type.fit(
    X_train,
    y_train_task_type,
)
joblib.dump(
    model_task_type,
    ARTIFACTS_DIR / "task_type.joblib",
)




# the model that computes the difficulty score
y_difficulty = df["difficulty"].to_numpy()
y_train_difficulty = y_difficulty[train_idx]
y_test_difficulty = y_difficulty[test_idx]

model_difficulty = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    max_depth=3
)
model_difficulty.fit(X_train, y_train_difficulty)
joblib.dump(
    model_difficulty,
    ARTIFACTS_DIR / "difficulty.joblib",
)


# required_quality
y_required_quality = df["required_quality"].to_numpy()
y_train_required_quality = y_required_quality[train_idx]
y_test_required_quality = y_required_quality[test_idx]

model_required_quality = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    max_depth=3,
)

model_required_quality.fit(
    X_train,
    y_train_required_quality,
)

joblib.dump(
    model_required_quality,
    ARTIFACTS_DIR / "required_quality.joblib",
)


# risk_level
y_risk_level = df["risk_level"].to_numpy()
y_train_risk_level = y_risk_level[train_idx]
y_test_risk_level = y_risk_level[test_idx]

model_risk_level = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
)

model_risk_level.fit(
    X_train,
    y_train_risk_level,
)

joblib.dump(
    model_risk_level,
    ARTIFACTS_DIR / "risk_level.joblib",
)


# expected_output_tokens
y_expected_output_tokens = df["expected_output_tokens"].to_numpy()
y_train_expected_output_tokens = y_expected_output_tokens[train_idx]
y_test_expected_output_tokens = y_expected_output_tokens[test_idx]

model_expected_output_tokens = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

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
y_test_latency_sensitivity = y_latency_sensitivity[test_idx]

model_latency_sensitivity = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
)

model_latency_sensitivity.fit(
    X_train,
    y_train_latency_sensitivity,
)

joblib.dump(
    model_latency_sensitivity,
    ARTIFACTS_DIR / "latency_sensitivity.joblib",
)

np.save(
    PROCESSED_DIR / "train_idx.npy",
    train_idx,
)

np.save(
    PROCESSED_DIR / "test_idx.npy",
    test_idx,
)



"""

Index(['Unnamed: 0', 'id', 'prompt', 'task_type', 'difficulty',
       'difficulty_bucket', 'required_quality', 'risk_level',
       'expected_output_tokens', 'latency_sensitivity'],
      dtype='str')


"""
