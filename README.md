# 🏥 MedAI — AI-Powered Healthcare System

> A production-ready hackathon project combining RAG-based Medical Chatbot + ML Disease Prediction

[![CI/CD](https://github.com/yourusername/medai/actions/workflows/deploy.yml/badge.svg)](https://github.com/yourusername/medai/actions)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Flask 3.0](https://img.shields.io/badge/flask-3.0-green)
![License MIT](https://img.shields.io/badge/license-MIT-yellow)

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│            HTML/CSS/JS (Chat + Prediction + X-Ray)          │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/REST API
┌───────────────────────────▼─────────────────────────────────┐
│                      FLASK BACKEND (app.py)                  │
│   /api/chat    /api/predict/disease    /api/predict/xray     │
└────────┬──────────────────┬──────────────────────┬──────────┘
         │                  │                      │
┌────────▼────────┐ ┌───────▼────────┐  ┌─────────▼─────────┐
│   RAG PIPELINE  │ │ ML PREDICTOR   │  │   CNN X-RAY        │
│                 │ │                │  │   ANALYZER         │
│ PDF → Chunks    │ │ Diabetes:      │  │                    │
│ Embeddings      │ │ Random Forest  │  │ MedicalCNN         │
│ Pinecone DB     │ │                │  │ (PyTorch)          │
│ LLM (Groq/GPT4) │ │ Heart Disease: │  │                    │
│                 │ │ Grad. Boosting │  │ Normal vs          │
│ Conversation    │ │                │  │ Pneumonia          │
│ Memory          │ │ Risk Score:    │  │ (DEMO — see        │
│                 │ │ Ridge Reg.     │  │  X-Ray Model below)│
└────────┬────────┘ └───────┬────────┘  └─────────┬─────────┘
         │                  │                      │
┌────────▼────────┐ ┌───────▼────────┐  ┌─────────▼─────────┐
│ PINECONE        │ │ scikit-learn   │  │ PyTorch Models     │
│ Vector DB       │ │ Model Files    │  │ (.pth files)       │
│ (Cloud)         │ │ (.pkl files)   │  │                    │
└─────────────────┘ └────────────────┘  └────────────────────┘
```

---

## 📁 Project Structure

```
medai_system/
├── app.py                          # Flask application entry point
├── requirements.txt                # Pinned Python dependencies
├── Dockerfile                      # Multi-stage production Docker build
├── docker-compose.yml              # Local development orchestration
├── .env.example                    # Environment variables template
│
├── src/
│   ├── chatbot/
│   │   └── rag_pipeline.py         # RAG: PDF → Chunks → Embeddings → Pinecone → LLM
│   ├── ml/
│   │   ├── predictor.py            # Disease Prediction ML models
│   │   ├── cnn_model.py            # MedicalCNN architecture + XRayAnalyzer (single X-ray inference path)
│   │   └── train_xray.py           # Real training pipeline for the X-ray model (run against your own dataset)
│   ├── pipelines/
│   │   ├── preprocessing.py        # Data validation & feature engineering
│   │   └── generate_medical_pdf.py # Creates a sample medical knowledge PDF (skips if data/medical_book.pdf already exists)
│   └── utils/
│       └── logger.py               # Centralized logging with rotation
│
├── templates/
│   └── index.html                  # Full SPA frontend (Chat + Prediction + X-Ray)
├── static/
│   ├── css/style.css               # Light healthcare UI styles
│   └── js/app.js                   # Frontend JavaScript
│
├── tests/
│   ├── test_predictor.py           # ML model unit tests
│   └── test_api.py                 # Flask API integration tests (chat, prediction, x-ray, ingestion, errors)
│
├── models/                         # ML model files (.pkl for diabetes/heart/risk; xray_cnn.pth only if you've trained one)
├── data/                           # Medical PDF knowledge base — data/medical_book.pdf
├── logs/                           # Application logs (auto-created)
│
└── .github/
    └── workflows/
        └── deploy.yml              # CI/CD: GitHub Actions → ECR → EC2
```

---

## ⚡ Quick Start (Local Development)

### 1. Prerequisites
- Python 3.11+
- Git
- Docker (optional, for containerized run)

### 2. Clone and Setup

```bash
git clone https://github.com/yourusername/medai.git
cd medai_system

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys:
```

**Required for the chatbot (RAG):**
| Service | Cost | Get Key At |
|---------|------|-----------|
| Groq (LLM) | **FREE** | [console.groq.com](https://console.groq.com) |
| Pinecone (Vector DB) | Free tier | [pinecone.io](https://pinecone.io) |
| OpenAI (optional alternative LLM) | Paid | [platform.openai.com](https://platform.openai.com) |

**Other environment variables** (see `.env.example` for the full list with placeholders):
| Variable | Purpose | Default |
|---|---|---|
| `FLASK_SECRET_KEY` | Flask session signing key | dev fallback — **set a real random value in production** |
| `ALLOWED_ORIGINS` | Comma-separated list of origins allowed to call the API (CORS) | unset → permissive (local dev only) |
| `FRONTEND_URL` | Convenience single-origin alternative to `ALLOWED_ORIGINS` | unset |
| `MEDICAL_PDF_PATH` | Path to the RAG knowledge-base PDF | `data/medical_book.pdf` |
| `LLM_PROVIDER` | `groq` or `openai` | `groq` |

Disease/risk prediction and the health check work with **no API keys at all**; the chatbot and ingestion endpoints require Groq/OpenAI + Pinecone keys and will return a `503` with a clear message if they're missing.

The X-ray endpoint also works with no keys, but see [X-Ray Model](#-x-ray-model-status--how-to-train-it) below — no trained weights ship with this repo.

### 4. Initialize Knowledge Base

The real knowledge-base PDF (`data/medical_book.pdf`) is already included in this repo. If you ever need to regenerate a small sample PDF instead (e.g. a fresh checkout without the real book), run:

```bash
python src/pipelines/generate_medical_pdf.py   # no-ops if data/medical_book.pdf already exists
```

```bash
# Start the app
python app.py

# In another terminal, trigger ingestion (one-time setup)
curl -X POST http://localhost:5000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "data/medical_book.pdf"}'
```

### 5. Open Browser

```
http://localhost:5000
```

---

## 🐳 Docker Deployment

### Local Docker

```bash
# Build and run
docker-compose up --build

# App available at: http://localhost:5000
```

### Production (Manual Docker)

```bash
docker build -t medai:latest .
docker run -d \
  --name medai \
  -p 5000:5000 \
  --env-file .env \
  -v medai_models:/app/models \
  medai:latest
```

---

## ☁️ AWS Deployment Guide

### Step 1: Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name medai-app \
  --region us-east-1
```

### Step 2: Push to ECR

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build & push
docker build -t medai-app .
docker tag medai-app:latest <ECR_URI>/medai-app:latest
docker push <ECR_URI>/medai-app:latest
```

### Step 3: Launch EC2 Instance

```bash
# Recommended: t3.medium (2 vCPU, 4GB RAM)
# AMI: Amazon Linux 2023 or Ubuntu 22.04
# Security Group: Allow ports 22 (SSH), 5000 (App), 80 (HTTP)
```

### Step 4: Configure EC2

```bash
# SSH into EC2
ssh -i your-key.pem ec2-user@your-ec2-ip

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker
sudo usermod -aG docker $USER

# Install AWS CLI
sudo yum install aws-cli -y

# Pull and run container
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <ECR_URI>

docker run -d \
  --name medai \
  --restart unless-stopped \
  -p 5000:5000 \
  --env-file /home/ec2-user/medai.env \
  <ECR_URI>/medai-app:latest
```

### Step 5: GitHub Actions CI/CD

Add these secrets to your GitHub repository (`Settings → Secrets`):
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `ECR_REPOSITORY`
- `EC2_HOST`
- `EC2_SSH_KEY`
- `EC2_USER`
- `ENV_FILE_CONTENT` (base64 of your .env: `base64 .env`)

Now every push to `main` automatically tests, builds, and deploys! 🚀

---

## 🧪 Running Tests

```bash
# Run all tests
pytest -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

**Current status:** `43 passed, 0 failed, 0 errors` — covering the health check, chatbot (including the "not configured" 503 path), disease/risk prediction (including malformed input), X-ray prediction (untrained-model path, invalid-image path, and a mocked-trained-model path), document ingestion, and the 404/405 error handlers.

Tests never call real external services (Pinecone, Groq/OpenAI, or a live Hugging Face download) — `RAGPipeline` and `XRayAnalyzer` are mocked at `app.RAGPipeline` / `app.XRayAnalyzer` (the names as **used** inside `app.py`, not where they're defined) so the suite runs offline and deterministically.

---

## 🩻 X-Ray Model Status & How to Train It

**This repo ships a single, working X-ray inference pipeline (`XRayAnalyzer` in `src/ml/cnn_model.py`) — but it does NOT ship trained weights.**

Why: a real chest X-ray classifier needs a real, labeled dataset (e.g. NIH ChestX-ray14 or the Kaggle "Chest X-Ray Images (Pneumonia)" dataset), which is several gigabytes and has its own license/download terms — outside what can be bundled with this project. We deliberately did **not** fake this by training on random noise or shipping an untrained ImageNet backbone as if it were a working detector (an earlier version of this project made that mistake — see `src/ml/train_xray.py`'s docstring for the reasoning).

**What actually happens today**, with no `models/xray_cnn.pth` present:
- `POST /api/predict/xray` returns HTTP 200 with `{"available": false, "trained": false, "code": "not_trained", ...}` and a clear message — it never returns a confident-looking but meaningless prediction.
- Invalid images (corrupted files, bad base64, unsupported formats) are still caught early and return HTTP 400.

**To make it real:**
```bash
python src/ml/train_xray.py --data-dir path/to/chest_xray --epochs 15
```
expecting `path/to/chest_xray/{train,val}/{NORMAL,PNEUMONIA}/*.jpg`. On success this writes `models/xray_cnn.pth`, which `XRayAnalyzer` will automatically load on the next request — no other code changes needed. Even once trained, results must be presented as an AI-assisted screening aid, not a diagnosis (see Disclaimer).

---

## 🎤 Interview Explanation

### "Explain the architecture"
> "MedAI is a three-tier system. The frontend is a vanilla JS SPA that communicates with a Flask REST API. The backend has two main modules: a RAG pipeline for the medical chatbot, and a scikit-learn ML module for disease prediction. The RAG pipeline uses LangChain to load medical PDFs, chunk them, embed them with a sentence-transformer model, and store them in Pinecone. At query time, the user question is embedded, the top-4 most relevant chunks are retrieved, and fed as context to a Groq LLM to generate the answer. The ML module trains three models — Random Forest for diabetes, Gradient Boosting for heart disease, and Ridge Regression for a continuous risk score — on synthetic datasets derived from real clinical distributions."

### "Why RAG over fine-tuning?"
> "RAG is more cost-effective, interpretable (you can cite sources), and updatable without retraining. For a hackathon, RAG lets us connect any medical PDF without the compute cost of fine-tuning."

### "Why Groq instead of OpenAI?"
> "Groq offers a free API for Llama3-70B which is production-grade. It's a pragmatic choice — GPT-4 is better but costs money. The system supports both via a single env variable."

### "How does the CNN work?"
> "It's a custom 4-block VGG-style CNN with BatchNorm and Dropout, using Global Average Pooling instead of large fully-connected layers for parameter efficiency. Input X-rays are resized to 224x224, converted to 3-channel, and normalized. Output is a 2-class softmax (Normal vs Pneumonia). **Important**: this repo doesn't ship trained weights for it — training requires a real labeled chest X-ray dataset that isn't bundled here. The inference pipeline, error handling, and training script are all real and ready; only the weights are missing until someone runs `train_xray.py` against a real dataset."

---

## ⚠️ Disclaimer

This software is an academic/demonstration project. It is **NOT** a certified medical device, has **NOT** been clinically validated, and must **NOT** be used to make real medical decisions. Always consult a qualified healthcare professional.

- Disease/risk predictions are statistical estimates from models trained on synthetic data resembling real clinical distributions — not diagnoses.
- The chatbot answers are generated by a general-purpose LLM grounded on a reference PDF via RAG; it can still be wrong, and it does not provide medication dosage recommendations.
- The X-ray classifier ships **without trained weights** (see [X-Ray Model Status](#-x-ray-model-status--how-to-train-it)) — until trained on a real dataset, `/api/predict/xray` explicitly reports itself as unavailable rather than guessing.

---

## 📄 License

MIT License — See LICENSE file.
