from adaptive_llm_router.config import CAPABILITY_TABLE_PATH
import pandas as pd

capability_table = pd.read_csv(CAPABILITY_TABLE_PATH)

def cost_first_policy(prediction, capability_table):

    task_type = prediction["task_type"]
    difficulty_bucket = prediction["difficulty_bucket"]
    required_quality = prediction["required_quality"]

    estimated_input_tokens = prediction["estimated_input_tokens"]
    expected_output_tokens = prediction["expected_output_tokens"]

    task_candidates = capability_table[
        capability_table["task_type"] == task_type
    ]

    eligible_models = task_candidates[
        task_candidates[f"quality_{difficulty_bucket}"] >= required_quality
    ].copy()

    if eligible_models.empty:
        return None

    eligible_models["estimated_cost"] = (
        (estimated_input_tokens * eligible_models["input_usd_per_1m"] ) / 1_000_000 
        + (expected_output_tokens * eligible_models["output_usd_per_1m"]) / 1_000_000
    )

    sorted_models = eligible_models.sort_values(
        by='estimated_cost',
        ascending=True
    )

    return sorted_models["model"].iloc[0]


