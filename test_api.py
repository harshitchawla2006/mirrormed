import pytest
from fastapi.testclient import TestClient
from api import app
import io
from PIL import Image

client = TestClient(app)

def make_test_image():
    img = Image.new('RGB', (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "efficientnet_b3"
    assert data["num_classes"] == 7

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "class_distribution" in data
    assert "avg_confidence" in data

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "MirrorMed" in response.text

def test_predict_valid_image():
    img_bytes = make_test_image()
    response  = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "class" in data
    assert "confidence" in data
    assert "uncertainty" in data
    assert "inference_time_s" in data
    assert "model_version" in data
    assert 0 <= data["confidence"] <= 1

def test_predict_invalid_file_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400

def test_predict_large_file():
    large_bytes = b"x" * (11 * 1024 * 1024)
    response = client.post(
        "/predict",
        files={"file": ("large.jpg", large_bytes, "image/jpeg")}
    )
    assert response.status_code == 400

def test_metrics_update_after_predict():
    before = client.get("/metrics").json()["total_predictions"]
    img_bytes = make_test_image()
    client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    after = client.get("/metrics").json()["total_predictions"]
    assert after == before + 1
