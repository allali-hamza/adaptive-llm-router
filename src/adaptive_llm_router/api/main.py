from fastapi import FastAPI
from adaptive_llm_router.api.schemas import RouteRequest, RouteResponse

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/route")
def routing_function(route_request:RouteRequest) -> RouteResponse:

    result = {
        "status": 'success',
        "task_type": route_request.task_type,
        "difficulty_bucket": route_request.difficulty_bucket,
        "required_quality": 'hellooooooooooo',
    }

    return result
