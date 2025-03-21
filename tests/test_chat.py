from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch

client = TestClient(app)


@patch("app.api.endpoints.chat.generate_response", return_value="AI is artificial intelligence.")
def test_chat_endpoint(mock_generate_response):
    response = client.post("/chat/", json={"query": "What is AI?"})
    assert response.status_code == 200
    json_data = response.json()
    assert "answer" in json_data
    assert json_data["answer"] == mock_generate_response.return_value


def test_empty_query():
    response = client.post("/chat/", json={"query": ""})
    assert response.status_code == 422


def test_invalid_input():
    response = client.post("/chat/", json={})
    assert response.status_code == 422


@patch("app.api.endpoints.chat.generate_response", return_value="AI is artificial intelligence.")
def test_rate_limit(mock_generate_response):
    """
    Test rate limiting (5 requests per minute).
    First 4 requests should succeed, 5th should fail with 429.
    """
    # ✅ First 4 requests should succeed
    for i in range(4):
        response = client.post("/chat/", json={"query": "What is AI?"})
        assert response.status_code == 200, f"Request {i + 1} failed with status {response.status_code}"

    # ✅ 5th request should fail with 429
    response = client.post("/chat/", json={"query": "What is AI?"})
    assert response.status_code == 429
    assert response.json()["detail"] == "Rate limit exceeded. Please try again later."
