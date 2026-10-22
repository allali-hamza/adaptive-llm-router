from adaptive_llm_router.config import ARTIFACTS_DIR
from adaptive_llm_router.data.embedding import prompt_to_embedding
from adaptive_llm_router.model.embedding_model import embedding_model
import numpy as np
import joblib


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


def difficulty_to_bucket(difficulty):
    if difficulty < 0.4:
        return "easy"
    elif difficulty < 0.7:
        return "medium"
    return "hard"

def predict(prompt, embedding_model):

    estimated_input_tokens = len(embedding_model.tokenizer.encode(
        prompt,
        add_special_tokens=True,
    ))

    prompt = prompt.strip()

    if not prompt:
        return {}
    prompt_embedding = prompt_to_embedding([prompt], embedding_model)
    
    task_type = model_task_type.predict(prompt_embedding)[0]

    difficulty = np.clip(
        model_difficulty.predict(prompt_embedding)[0],
        0.0,
        1.0,
    )

    required_quality = np.clip(
        model_required_quality.predict(prompt_embedding)[0],
        0.0,
        1.0,
    )

    risk_level = model_risk_level.predict(prompt_embedding)[0]
    
    expected_output_tokens = max(
        0, 
        round(model_expected_output_tokens.predict(prompt_embedding)[0])
    )

    latency_sensitivity = model_latency_sensitivity.predict(prompt_embedding)[0]

    result = {
        "estimated_input_tokens": estimated_input_tokens,
        "task_type": str(task_type),
        "difficulty": float(difficulty),
        "difficulty_bucket": difficulty_to_bucket(float(difficulty)),
        "required_quality": float(required_quality),
        "risk_level": str(risk_level),
        "expected_output_tokens": int(expected_output_tokens),
        "latency_sensitivity": str(latency_sensitivity),
    }

    return result


if __name__ == "__main__":

    result = predict(
        prompt="Why is my Python API returning a 500 error",
        embedding_model=embedding_model
    )
    print(result)