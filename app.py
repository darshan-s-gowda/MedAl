"""
MedAI - AI-Powered Healthcare System
Main Flask Application Entry Point
"""

import os
import logging

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv

from src.chatbot.rag_pipeline import RAGPipeline
from src.ml.predictor import DiseasePredictor
from src.ml.cnn_model import XRayAnalyzer
from src.pipelines.patient_report import generate_patient_report
from src.utils.logger import setup_logger

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "medai-secret-key-change-in-prod")

# ─── CORS ───
# In production, restrict to explicit origins via ALLOWED_ORIGINS (comma-separated).
# Falls back to a permissive "*" only when nothing is configured, which keeps local
# development frictionless without silently shipping an open policy to prod.
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
_frontend_url = os.getenv("FRONTEND_URL", "").strip()
origins = [o.strip() for o in _allowed_origins.split(",") if o.strip()]
if _frontend_url and _frontend_url not in origins:
    origins.append(_frontend_url)

if origins:
    CORS(app, origins=origins, supports_credentials=True)
else:
    # No origin restriction configured — fine for local development, but set
    # ALLOWED_ORIGINS (and/or FRONTEND_URL) in production.
    CORS(app)

logger = setup_logger(__name__)

# Default PDF path used for the medical knowledge base (RAG ingestion).
# Kept as a single constant so the API, README, Docker, and scripts all agree.
DEFAULT_PDF_PATH = os.getenv("MEDICAL_PDF_PATH", "data/medical_book.pdf")

# Lazy-loaded components
rag_pipeline = None
disease_predictor = None
xray_analyzer = None


def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline()
    return rag_pipeline


def get_disease_predictor():
    global disease_predictor
    if disease_predictor is None:
        logger.info("Initializing Disease Predictor...")
        disease_predictor = DiseasePredictor()
    return disease_predictor


def get_xray_analyzer():
    global xray_analyzer
    if xray_analyzer is None:
        logger.info("Initializing X-Ray Analyzer...")
        xray_analyzer = XRayAnalyzer()
    return xray_analyzer


# ─────────────────────────────────────────────
# Frontend Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "MedAI",
        "version": "1.0.0"
    })


# ─────────────────────────────────────────────
# Chatbot API (RAG)
# ─────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True)

        if not data or "message" not in data:
            return jsonify({"error": "Message is required"}), 400

        user_message = data["message"].strip()

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Session history
        conversation_history = session.get("chat_history", [])

        # Query RAG
        try:
            pipeline = get_rag_pipeline()
        except Exception as e:
            # Any RAGPipeline() init failure (missing API keys, no network
            # access to download the embedding model, bad Pinecone config,
            # etc.) is a "service not configured/unavailable" situation, not
            # a bug — report it as such instead of a generic 500.
            logger.error(f"RAG pipeline not configured/unavailable: {e}")
            return jsonify({
                "error": "Chatbot is not configured or its dependencies are unavailable "
                         "(check API keys and network access).",
                "detail": str(e)
            }), 503

        response = pipeline.query(
            question=user_message,
            chat_history=conversation_history
        )

        # Update history
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": response["answer"]})
        session["chat_history"] = conversation_history[-20:]

        logger.info(f"Chat success. Sources: {len(response.get('sources', []))}")

        return jsonify({
            "answer": response["answer"],
            "sources": response.get("sources", []),
            "disclaimer": "⚠️ This is NOT a substitute for professional medical advice."
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({"error": "Chat service failed"}), 500


@app.route("/api/chat/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return jsonify({"message": "Conversation cleared"})


# ─────────────────────────────────────────────
# Disease Prediction API
# ─────────────────────────────────────────────

@app.route("/api/predict/disease", methods=["POST"])
def predict_disease():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "Patient data required"}), 400

        predictor = get_disease_predictor()
        results = predictor.predict_all(data)

        logger.info("Disease prediction completed.")

        return jsonify({
            "predictions": results,
            "disclaimer": "⚠️ This is NOT a medical diagnosis."
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.error(f"Disease prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Prediction failed"}), 500


# ─────────────────────────────────────────────
# Risk Score API
# ─────────────────────────────────────────────

@app.route("/api/predict/risk-score", methods=["POST"])
def risk_score():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "Patient data required"}), 400

        predictor = get_disease_predictor()
        score = predictor.calculate_risk_score(data)

        return jsonify({
            "risk_score": score["risk_percentage"],
            "risk_level": score["risk_level"],
            "factors": score["factors"],
            "disclaimer": "⚠️ Statistical estimate only."
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.error(f"Risk score error: {str(e)}", exc_info=True)
        return jsonify({"error": "Risk calculation failed"}), 500


# ─────────────────────────────────────────────
# X-ray Prediction API
#
# Uses the single canonical XRayAnalyzer (src/ml/cnn_model.py). All base64
# decoding, image validation, and error handling for corrupted/unsupported
# images happens inside analyzer.analyze_image(); see that module for the
# current model-training status and disclaimer.
# ─────────────────────────────────────────────

@app.route("/api/predict/xray", methods=["POST"])
def predict_xray():
    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data or not data["image"]:
            return jsonify({"available": False, "error": "No image provided"}), 400

        analyzer = get_xray_analyzer()
        result = analyzer.analyze_image(data["image"])

        if not result.get("available"):
            logger.info(f"X-ray request could not be fulfilled: {result.get('error')}")
            # Bad client input (corrupt/invalid image, bad encoding) -> 400.
            # Model unavailable/untrained is a server-side state, not a
            # client error, so it's still reported with a 200 + available:false
            # so the frontend can render a friendly message.
            status = 400 if result.get("code") == "invalid_input" else 200
            return jsonify(result), status

        logger.info("X-ray prediction completed.")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"X-ray error: {str(e)}", exc_info=True)
        return jsonify({
            "available": False,
            "error": "X-ray processing failed"
        }), 500

# ─────────────────────────────────────────────
# Patient Medical Report API
# ─────────────────────────────────────────────

@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "Report data required"}), 400

        patient_data = data.get("patient", {})
        results = data.get("results", {})

        pdf_buffer = generate_patient_report(
            patient_data=patient_data,
            results=results
        )

        logger.info("Patient medical report generated successfully.")

        from flask import send_file

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="MedAI_Patient_Report.pdf"
        )

    except Exception as e:
        logger.error(
            f"Patient report generation error: {str(e)}",
            exc_info=True
        )

        return jsonify({
            "error": "Failed to generate patient report"
        }), 500


# ─────────────────────────────────────────────
# Document Ingestion
# ─────────────────────────────────────────────

@app.route("/api/ingest", methods=["POST"])
def ingest_documents():
    try:
        body = request.get_json(silent=True) or {}
        pdf_path = body.get("pdf_path", DEFAULT_PDF_PATH)

        if not os.path.exists(pdf_path):
            return jsonify({"error": f"PDF not found at path: {pdf_path}"}), 400

        try:
            pipeline = get_rag_pipeline()
        except Exception as e:
            logger.error(f"RAG pipeline not configured/unavailable: {e}")
            return jsonify({
                "error": "Ingestion is not configured or its dependencies are unavailable "
                         "(check API keys and network access).",
                "detail": str(e)
            }), 503

        result = pipeline.ingest_documents(pdf_path)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}", exc_info=True)
        return jsonify({"error": "Document ingestion failed"}), 500


# ─────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────
# Run App
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"

    logger.info(f"Starting MedAI on port {port} (debug={debug})")

    app.run(host="0.0.0.0", port=port, debug=debug)