from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_shape():
    response = client.post(
        "/predict",
        json={"comments": ["this video is amazing", "worst video ever", "ok"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["labels"]) == 3
    assert set(body["labels"]) <= {"negative", "neutral", "positive"}
    assert sum(body["counts"].values()) == 3


def test_predict_obvious_sentiment():
    response = client.post(
        "/predict",
        json={"comments": ["i love this, it is great and amazing", "this is terrible and awful, i hate it"]},
    )
    labels = response.json()["labels"]
    assert labels[0] == "positive"
    assert labels[1] == "negative"


def test_predict_empty_list():
    response = client.post("/predict", json={"comments": []})
    assert response.status_code == 200
    assert response.json()["total"] == 0


def test_predict_rejects_bad_payload():
    response = client.post("/predict", json={"wrong": "field"})
    assert response.status_code == 422