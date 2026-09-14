"""
Integration Tests for MedAI Flask API
Tests all API endpoints without requiring external services (Pinecone, LLM
providers, or trained PyTorch weights).

Mocking strategy:
    app.py imports RAGPipeline and XRayAnalyzer with
    `from ... import X`, which binds the name `X` directly into app.py's
    own module namespace. So the object that actually needs patching is
    `app.RAGPipeline` / `app.XRayAnalyzer` (where they are USED), not
    `src.chatbot.rag_pipeline.RAGPipeline` (where they are merely DEFINED).
    Patching the definition-site name after app.py has already imported it
    would have no effect on app.py's copy of the reference.
"""

import base64
import io
import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
from app import app  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret"
    with app.test_client() as client:
        yield client
    # Reset lazily-initialized singletons so tests don't leak state into each other.
    app_module.rag_pipeline = None
    app_module.xray_analyzer = None


@pytest.fixture
def mock_rag(client):
    """Install a mocked RAGPipeline instance directly on the app module."""
    mock_instance = MagicMock()
    mock_instance.query.return_value = {
        "answer": "Diabetes is characterized by high blood sugar levels.",
        "sources": [{"page": 1, "source": "medical_book.pdf", "snippet": "test..."}]
    }
    mock_instance.ingest_documents.return_value = {
        "status": "success", "pages_loaded": 1, "chunks_created": 3, "index": "medai-knowledge"
    }
    with patch("app.RAGPipeline", return_value=mock_instance):
        yield mock_instance


@pytest.fixture
def mock_rag_unconfigured(client):
    """Simulate a RAGPipeline that fails to init due to missing API keys."""
    with patch("app.RAGPipeline", side_effect=ValueError("PINECONE_API_KEY environment variable not set.")):
        yield


@pytest.fixture
def mock_xray_trained(client):
    """A mocked, 'trained' X-ray analyzer returning a real-looking prediction."""
    mock_instance = MagicMock()
    mock_instance.analyze_image.return_value = {
        "available": True,
        "trained": True,
        "prediction": "Normal",
        "confidence": 91.2,
        "class_probabilities": {"Normal": 91.2, "Pneumonia": 8.8},
        "disclaimer": "AI-assisted screening aid only — NOT a medical diagnosis."
    }
    with patch("app.XRayAnalyzer", return_value=mock_instance):
        yield mock_instance


@pytest.fixture
def mock_xray_untrained(client):
    """The realistic default: no trained weights bundled with the project."""
    mock_instance = MagicMock()
    mock_instance.analyze_image.return_value = {
        "available": False,
        "trained": False,
        "code": "not_trained",
        "error": "The X-ray classification model has not been trained yet."
    }
    with patch("app.XRayAnalyzer", return_value=mock_instance):
        yield mock_instance


def _sample_image_b64(fmt="PNG", size=(64, 64)):
    img = Image.new("RGB", size, color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


class TestHealthCheck:
    def test_health_endpoint(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["status"] == "healthy"
        assert data["service"] == "MedAI"

    def test_index_loads(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert b"MedAI" in res.data


class TestChatAPI:
    def test_chat_requires_message(self, client):
        res = client.post("/api/chat", data=json.dumps({}), content_type="application/json")
        assert res.status_code == 400

    def test_chat_empty_message(self, client):
        res = client.post("/api/chat", data=json.dumps({"message": ""}), content_type="application/json")
        assert res.status_code == 400

    def test_chat_malformed_json(self, client):
        res = client.post("/api/chat", data="{not valid json", content_type="application/json")
        assert res.status_code == 400

    def test_chat_clear(self, client):
        res = client.post("/api/chat/clear")
        assert res.status_code == 200

    def test_chat_success_with_mocked_rag(self, client, mock_rag):
        res = client.post(
            "/api/chat",
            data=json.dumps({"message": "What is diabetes?"}),
            content_type="application/json"
        )
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "answer" in data
        assert "sources" in data
        assert "disclaimer" in data
        mock_rag.query.assert_called_once()

    def test_chat_returns_503_when_rag_unconfigured(self, client, mock_rag_unconfigured):
        res = client.post(
            "/api/chat",
            data=json.dumps({"message": "What is diabetes?"}),
            content_type="application/json"
        )
        assert res.status_code == 503
        data = json.loads(res.data)
        assert "error" in data


class TestPredictionAPI:
    def test_prediction_requires_data(self, client):
        res = client.post("/api/predict/disease", data=json.dumps({}), content_type="application/json")
        assert res.status_code in [200, 400, 500]

    def test_prediction_malformed_json(self, client):
        res = client.post("/api/predict/disease", data="not json at all", content_type="application/json")
        assert res.status_code in [400, 500]

    def test_prediction_with_diabetes_data(self, client):
        data = {
            "Glucose": 150, "BMI": 32.0, "Age": 52,
            "Pregnancies": 4, "BloodPressure": 85,
            "SkinThickness": 28, "Insulin": 130,
            "DiabetesPedigreeFunction": 0.7
        }
        res = client.post("/api/predict/disease", data=json.dumps(data), content_type="application/json")
        assert res.status_code == 200
        result = json.loads(res.data)
        assert "predictions" in result
        assert "disclaimer" in result

    def test_prediction_response_structure(self, client):
        data = {"age": 55, "Glucose": 140, "BMI": 30, "trestbps": 140, "chol": 260}
        res = client.post("/api/predict/disease", data=json.dumps(data), content_type="application/json")
        assert res.status_code == 200
        result = json.loads(res.data)
        predictions = result.get("predictions", [])
        for pred in predictions:
            assert "condition" in pred
            assert "prediction" in pred
            assert "probability" in pred

    def test_prediction_insufficient_data_returns_400(self, client):
        # No recognizable clinical fields at all -> DiseasePredictor.predict_all raises ValueError
        data = {"favorite_color": "blue"}
        res = client.post("/api/predict/disease", data=json.dumps(data), content_type="application/json")
        assert res.status_code == 400

    def test_risk_score_endpoint(self, client):
        data = {
            "age": 50, "trestbps": 135, "chol": 230,
            "thalach": 155, "oldpeak": 1.0, "ca": 1, "exang": 0
        }
        res = client.post("/api/predict/risk-score", data=json.dumps(data), content_type="application/json")
        assert res.status_code == 200
        result = json.loads(res.data)
        assert "risk_score" in result
        assert "risk_level" in result


class TestXrayAPI:
    def test_xray_requires_image(self, client):
        res = client.post("/api/predict/xray", data=json.dumps({}), content_type="application/json")
        assert res.status_code == 400
        data = json.loads(res.data)
        assert data["available"] is False

    def test_xray_invalid_base64(self, client):
        # Mock the analyzer to exercise the app-level status-code mapping for
        # code="invalid_input" without needing torch/trained weights.
        with patch("app.XRayAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze_image.return_value = {
                "available": False,
                "code": "invalid_input",
                "error": "Invalid image encoding (expected base64)."
            }
            res = client.post(
                "/api/predict/xray",
                data=json.dumps({"image": "not-valid-base64!!!"}),
                content_type="application/json"
            )
        assert res.status_code == 400
        data = json.loads(res.data)
        assert data["available"] is False

    def test_xray_untrained_model_returns_200_with_available_false(self, client, mock_xray_untrained):
        payload = {"image": f"data:image/png;base64,{_sample_image_b64()}"}
        res = client.post("/api/predict/xray", data=json.dumps(payload), content_type="application/json")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["available"] is False
        assert data["trained"] is False

    def test_xray_success_with_mocked_trained_model(self, client, mock_xray_trained):
        payload = {"image": f"data:image/png;base64,{_sample_image_b64()}"}
        res = client.post("/api/predict/xray", data=json.dumps(payload), content_type="application/json")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["available"] is True
        assert data["prediction"] in ["Normal", "Pneumonia"]
        assert "disclaimer" in data
        mock_xray_trained.analyze_image.assert_called_once()

    def test_xray_missing_model_does_not_crash_flask(self, client):
        # Simulate XRayAnalyzer() constructor itself raising (e.g. bad checkpoint).
        # app.py's XRayAnalyzer already guards __init__ internally, but this test
        # confirms the /api/predict/xray route also never 500s into a dead app.
        with patch("app.XRayAnalyzer", side_effect=RuntimeError("simulated load failure")):
            res = client.post(
                "/api/predict/xray",
                data=json.dumps({"image": f"data:image/png;base64,{_sample_image_b64()}"}),
                content_type="application/json"
            )
        assert res.status_code == 500
        data = json.loads(res.data)
        assert data["available"] is False


class TestIngestAPI:
    def test_ingest_missing_pdf_returns_400(self, client, mock_rag):
        res = client.post(
            "/api/ingest",
            data=json.dumps({"pdf_path": "data/does_not_exist.pdf"}),
            content_type="application/json"
        )
        assert res.status_code == 400

    def test_ingest_default_pdf_path_is_medical_book(self, client, mock_rag):
        # No pdf_path given -> should default to data/medical_book.pdf, which
        # exists in this repo, and succeed via the mocked pipeline.
        res = client.post("/api/ingest", data=json.dumps({}), content_type="application/json")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert data["status"] == "success"
        mock_rag.ingest_documents.assert_called_once()
        called_path = mock_rag.ingest_documents.call_args[0][0]
        assert called_path == "data/medical_book.pdf"

    def test_ingest_returns_503_when_rag_unconfigured(self, client, mock_rag_unconfigured):
        res = client.post("/api/ingest", data=json.dumps({}), content_type="application/json")
        assert res.status_code == 503


class TestErrorHandlers:
    def test_404_returns_json(self, client):
        res = client.get("/nonexistent-endpoint")
        assert res.status_code == 404
        data = json.loads(res.data)
        assert "error" in data

    def test_method_not_allowed(self, client):
        res = client.get("/api/chat")  # Should be POST
        assert res.status_code == 405
