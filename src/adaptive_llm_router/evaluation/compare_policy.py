from adaptive_llm_router.config import (
    CAPABILITY_TABLE_PATH,
    EVALUATION_DIR,
    METADATA_FILE,
    PROCESSED_DIR,
)
from adaptive_llm_router.router.policy import cost_first_policy
import pandas as pd
import numpy as np
import json
from math import ceil


BASELINE_MODEL = "gpt-5.5"


def estimate_input_tokens(prompt):
    return max(1, ceil(len(prompt) / 4))


def get_model_row(capability_table, model, task_type):
    model_row = capability_table[
        (capability_table["model"] == model)
        & (capability_table["task_type"] == task_type)
    ]

    if model_row.empty:
        raise ValueError(
            f"No capability entry for {model} and {task_type}"
        )

    return model_row.iloc[0]


def estimate_cost(model_row, input_tokens, output_tokens):
    return (
        (input_tokens * model_row["input_usd_per_1m"]) / 1_000_000
        + (output_tokens * model_row["output_usd_per_1m"]) / 1_000_000
    )


def get_summary(results_df, strategy):
    strategy_results = results_df[
        results_df["strategy"] == strategy
    ]

    return {
        "requests": int(len(strategy_results)),
        "quality_threshold_met_rate": float(
            strategy_results["quality_threshold_met"].mean()
        ),
        "mean_capability_quality": float(
            strategy_results["capability_quality"].mean()
        ),
        "mean_estimated_cost_usd": float(
            strategy_results["estimated_cost_usd"].mean()
        ),
        "total_estimated_cost_usd": float(
            strategy_results["estimated_cost_usd"].sum()
        ),
    }


def compare_policies():
    metadata = pd.read_csv(filepath_or_buffer=METADATA_FILE)
    capability_table = pd.read_csv(filepath_or_buffer=CAPABILITY_TABLE_PATH)
    test_idx = np.load(PROCESSED_DIR / "test_idx.npy")
    test_data = metadata.iloc[test_idx]

    results = []
    selected_models = []

    for _, request in test_data.iterrows():
        input_tokens = estimate_input_tokens(request["prompt"])

        prediction = {
            "task_type": request["task_type"],
            "difficulty_bucket": request["difficulty_bucket"],
            "required_quality": request["required_quality"],
            "estimated_input_tokens": input_tokens,
            "expected_output_tokens": request["expected_output_tokens"],
        }

        selected_model = cost_first_policy(
            prediction=prediction,
            capability_table=capability_table,
        )

        if selected_model is None:
            selected_model = BASELINE_MODEL
        else:
            selected_models.append(selected_model)

        for strategy, model in [
            ("always_premium", BASELINE_MODEL),
            ("cost_first", selected_model),
        ]:
            model_row = get_model_row(
                capability_table=capability_table,
                model=model,
                task_type=request["task_type"],
            )

            capability_quality = model_row[
                f"quality_{request['difficulty_bucket']}"
            ]

            results.append({
                "strategy": strategy,
                "model": model,
                "capability_quality": capability_quality,
                "quality_threshold_met": bool(
                    capability_quality >= request["required_quality"]
                ),
                "estimated_cost_usd": estimate_cost(
                    model_row=model_row,
                    input_tokens=input_tokens,
                    output_tokens=request["expected_output_tokens"],
                ),
            })

    results_df = pd.DataFrame(results)
    premium_results = get_summary(results_df, "always_premium")
    policy_results = get_summary(results_df, "cost_first")
    selected_models_series = pd.Series(selected_models)

    comparison = {
        "evaluation": {
            "dataset": "held-out test split",
            "requests": int(len(test_data)),
            "baseline_model": BASELINE_MODEL,
            "input_token_estimate": "max(1, ceil(prompt character count / 4))",
            "policy_fallback": BASELINE_MODEL,
            "note": (
                "Quality and cost are estimated from the seed capability table. "
                "This evaluates routing decisions, not live model outputs."
            ),
        },
        "always_premium": premium_results,
        "cost_first_policy": policy_results,
        "cost_savings": {
            "usd": (
                premium_results["total_estimated_cost_usd"]
                - policy_results["total_estimated_cost_usd"]
            ),
            "percent": (
                1
                - policy_results["total_estimated_cost_usd"]
                / premium_results["total_estimated_cost_usd"]
            ) * 100,
        },
        "policy_selection": {
            "eligible_request_rate": len(selected_models) / len(test_data),
            "models": selected_models_series.value_counts().to_dict(),
        },
    }

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = EVALUATION_DIR / "policy_comparison.json"
    with open(output_path, "w") as file:
        json.dump(
            comparison,
            file,
            indent=4,
        )

    print(f"Requests evaluated: {comparison['evaluation']['requests']}")
    print(
        f"Always-premium cost: "
        f"${premium_results['total_estimated_cost_usd']:.6f}"
    )
    print(
        f"Cost-first cost: "
        f"${policy_results['total_estimated_cost_usd']:.6f}"
    )
    print(
        f"Estimated savings: "
        f"${comparison['cost_savings']['usd']:.6f} "
        f"({comparison['cost_savings']['percent']:.1f}%)"
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    compare_policies()
