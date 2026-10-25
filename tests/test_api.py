import unittest

from fastapi.testclient import TestClient

from adaptive_llm_router.api.main import app


client = TestClient(app)


class ApiTest(unittest.TestCase):
    def test_health(self):
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_route(self):
        payload = {
            "task_type": "reasoning",
            "difficulty_bucket": "medium",
            "required_quality": 0.8,
            "estimated_input_tokens": 500,
            "expected_output_tokens": 200,
        }

        response = client.post("/route", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "success",
            "task_type": "reasoning",
            "difficulty_bucket": "medium",
            "required_quality": 0.8,
        })

        response = client.post("/route", json={**payload, "required_quality": 1.5})
        self.assertEqual(response.status_code, 422)
