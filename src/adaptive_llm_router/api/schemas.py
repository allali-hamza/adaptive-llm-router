from pydantic import BaseModel, Field, ValidationError



class RouteRequest(BaseModel):
    task_type: str
    difficulty_bucket: str
    required_quality: float = Field(
        ge=0,
        le=1
    )
    estimated_input_tokens: int = Field(
        ge=0
    )
    expected_output_tokens: int = Field(
        ge=0
    )


if __name__ == "__main__":

    case1 = {
        "task_type": "reasoning",
        "difficulty_bucket": "medium",
        "required_quality": 0.8,
        "estimated_input_tokens": 500,
        "expected_output_tokens": 200
    }


    case2 = {
        **case1,
        "required_quality": 1.5
    }



    case3 = {
        **case1,
        "estimated_input_tokens": -100
    }

    case4 = {
        "difficulty_bucket": "medium",
        "required_quality": 0.8,
        "estimated_input_tokens": 500,
        "expected_output_tokens": 200
    }


    for i, data in enumerate([case1, case2, case3, case4], start=1):
        print(f"\n===== CAS {i} =====")

        try:
            request = RouteRequest(**data)
            print("Objet créé :", request)

        except ValidationError as e:
            print("ValidationError")
            print(e.errors())
            for error in e.errors():
                print(f"Champ : {error['loc']}")
                print(f"Type  : {error['type']}")
                print(f"Erreur : {error['msg']}")




