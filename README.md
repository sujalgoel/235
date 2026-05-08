# RealityCheck AI - Multimodal Fake Profile Detection

> **B.Tech CSE Final Year Project 2025-26**
> Amity School of Engineering & Technology | Group 235

An advanced AI-powered system for detecting fake profiles using multimodal analysis of images, text, and metadata with explainable AI (XAI) techniques.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Key Innovations](#key-innovations)
- [API](#api)
- [Performance](#performance)
- [Team](#team)
- [Documentation](#documentation)
- [License](#license)

---

## 🎯 Overview

RealityCheck AI is a comprehensive fake profile detection system that combines multiple AI models to analyze profile images, bio text, and metadata. The system provides:

- **98%+ accuracy** on image detection (Ensemble: EfficientNet-B7 + XceptionNet + CLIP)
- **95%+ accuracy** on text detection (Ensemble: OpenAI Detector + ChatGPT Detector + Rules)
- **Explainable AI** visualizations (Grad-CAM heatmaps, SHAP/LIME token importance)
- **Real-time analysis** through a modern web interface

### Problem Statement

With the rise of deepfakes and AI-generated content, traditional profile verification methods are becoming obsolete. Our system addresses this by:

1. Detecting AI-generated profile images (deepfakes, synthetic faces)
2. Identifying AI-written bio text (ChatGPT, GPT-4, and other LLM outputs)
3. Analyzing metadata for manipulation indicators
4. Fusing multiple signals into a unified trust score

---

## ✨ Features

### 🖼️ Image Analysis
- **AI-Generated Image Detection** using ensemble deep learning
- **Face Detection** with YOLOv8 for automated cropping
- **Grad-CAM Heatmaps** showing which image regions influenced the decision
- **Deepfake Detection** specialized models

### 📝 Text Analysis
- **AI-Written Text Detection** using transformer models
- **SHAP/LIME Explanations** highlighting suspicious phrases
- **Linguistic Pattern Analysis** for writing style fingerprinting
- **Multi-model Ensemble** for robust detection

### 🔍 Metadata Forensics
- **EXIF Data Analysis** for image authenticity
- **Camera Signature Verification**
- **Editing Software Detection**
- **Timestamp Anomaly Detection**

### 🎯 Multimodal Fusion
- **Weighted Trust Score** combining all modalities
- **Confidence-Based Calibration** for uncertain predictions
- **Adaptive Weight Redistribution** when modalities are missing
- **Human-Readable Interpretations**

### 🎨 User Interface
- **Modern React Frontend** with Tailwind CSS
- **Real-Time API Status** monitoring
- **Interactive Results Display** with visualizations
- **Mobile-Responsive Design**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│  - Profile Upload Interface                                  │
│  - Results Visualization                                     │
│  - Explainability Dashboard                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTP/REST API
┌─────────────────▼───────────────────────────────────────────┐
│                   Backend (FastAPI)                          │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           Analysis Pipeline Orchestrator               │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│         ┌────────────────┼────────────────┐                 │
│         │                │                │                 │
│  ┌──────▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐          │
│  │   Image     │  │   Text    │  │  Metadata   │          │
│  │  Analysis   │  │  Analysis │  │  Forensics  │          │
│  │             │  │           │  │             │          │
│  │ EfficientNet│  │  OpenAI   │  │ PyExifTool  │          │
│  │ XceptionNet │  │  Detector │  │   Scoring   │          │
│  │    CLIP     │  │  ChatGPT  │  │   Engine    │          │
│  │  Grad-CAM   │  │   Rules   │  │             │          │
│  │   YOLOv8    │  │ SHAP/LIME │  │             │          │
│  └──────┬──────┘  └─────┬─────┘  └──────┬──────┘          │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                   │
│                   ┌──────▼──────┐                           │
│                   │   Fusion    │                           │
│                   │   Engine    │                           │
│                   │             │                           │
│                   │ Trust Score │                           │
│                   │ Computation │                           │
│                   └─────────────┘                           │
└───────────────────────────────────────────────────────────┘
```

### Processing Flow

1. **Input**: User uploads profile image + bio text
2. **Image Module**: Processes through ensemble models → generates score + Grad-CAM
3. **Text Module**: Tokenizes and analyzes → generates score + SHAP explanations
4. **Metadata Module**: Extracts EXIF data → generates forensics score
5. **Fusion**: Weighted combination → Final trust score ∈ [0, 1]
6. **Output**: Trust score, interpretation, visualizations

---

## 🛠️ Technology Stack

### Backend
- **Python 3.13** - Core language
- **FastAPI** - High-performance web framework
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - NLP models
- **Ultralytics YOLOv8** - Object detection
- **SHAP/LIME** - Explainability
- **PyExifTool** - Metadata extraction
- **Grad-CAM** - Visual explanations

### Frontend
- **React 18** - UI framework
- **Tailwind CSS** - Styling
- **React Router** - Navigation
- **Axios** - HTTP client
- **Vite** - Build tool

### AI Models
- **EfficientNet-B7** - Image classification (50% weight)
- **XceptionNet** - Deepfake detection (40% weight)
- **CLIP** - Vision-language model (10% weight)
- **OpenAI RoBERTa Detector** - AI text detection (70% weight)
- **ChatGPT Detector** - GPT-specific detection (20% weight)
- **Rule-Based Detector** - Heuristics (10% weight)
- **DistilBERT** - Fallback text model

### DevOps
- **Git** - Version control
- **Docker** - Containerization (optional)
- **Prometheus** - Monitoring (optional)

---

## 📁 Project Structure

```
RealityCheck-AI/
├── src/                          # Backend source code
│   ├── api/                      # FastAPI application
│   │   ├── app.py               # Main API server
│   │   └── schemas/             # Request/Response models
│   ├── core/                     # Core logic
│   │   └── pipeline.py          # Analysis pipeline orchestrator
│   ├── modules/                  # Detection modules
│   │   ├── base.py              # Abstract base classes
│   │   ├── image/               # Image detection
│   │   │   ├── ensemble_detector.py
│   │   │   └── classifier.py
│   │   ├── text/                # Text detection
│   │   │   ├── ensemble_text_detector.py
│   │   │   └── classifier.py
│   │   ├── metadata/            # Metadata forensics
│   │   │   └── analyzer.py
│   │   └── fusion/              # Multimodal fusion
│   │       └── trust_scorer.py
│   └── utils/                    # Utilities
│       ├── logging.py
│       └── exceptions.py
│
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── ProfileAnalyzer.jsx
│   │   │   ├── ResultsDisplay.jsx
│   │   │   └── TrustScoreGauge.jsx
│   │   ├── pages/               # Page components
│   │   ├── services/            # API client
│   │   │   └── api.js
│   │   └── App.jsx              # Root component
│   └── package.json
│
├── config/                       # Configuration
│   └── base.py                  # Centralized config
│
├── models/                       # AI model weights (auto-download)
│   ├── image/                   # Image models
│   ├── text/                    # Text models
│   └── fusion/                  # Fusion checkpoints
│
├── scripts/                      # Utility scripts
├── logs/                        # Application logs
├── uploads/                     # Temporary uploads
├── visualizations/              # Generated heatmaps
│
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── SETUP.md                     # Setup guide
├── project research.pdf         # Research paper
└── Synopsis.pdf                 # Project synopsis
```

---

## 🚀 Key Innovations

### 1. **Ensemble Architecture**
Unlike single-model approaches, we combine multiple specialized models with optimal weights determined through empirical testing.

### 2. **Explainable AI (XAI)**
Every prediction comes with visual/textual explanations:
- **Grad-CAM** highlights suspicious image regions
- **SHAP/LIME** shows which words triggered AI detection

### 3. **Multimodal Fusion**
Research-backed weighted fusion formula:
```
Trust Score = 0.4 × Image_Score + 0.3 × Text_Score + 0.3 × Metadata_Score
```

### 4. **Adaptive Weight Redistribution**
If metadata is missing (common for uploads), weights automatically adjust:
```
Image: 0.4 → 0.57 (0.4/0.7)
Text:  0.3 → 0.43 (0.3/0.7)
```

### 5. **Confidence Calibration**
Low-confidence predictions are pulled toward uncertainty (0.5) to avoid false confidence.

---

## 🔌 API

The FastAPI backend exposes three analysis endpoints. All three return the same envelope shape — `image_analysis` and/or `text_analysis` are `null` when that modality wasn't requested.

### Endpoints

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `GET`  | `/health` | — | Service status + per-module readiness |
| `POST` | `/api/v1/analyze/profile` | `image` (file), `bio_text` (10-1000 chars), `profile_id` (optional) | Multimodal: image + text |
| `POST` | `/api/v1/analyze/image` | `image` (file) | Image-only |
| `POST` | `/api/v1/analyze/text` | `text` (10-1000 chars) | Text-only |

Files are capped at 10 MiB. Validation happens at the FastAPI layer; oversize uploads are rejected with `413` before the body is buffered.

### Example

```bash
curl -F "image=@profile.jpg" \
     -F "bio_text=Passionate marketer with 10 years of innovative experience" \
     http://localhost:8000/api/v1/analyze/profile
```

### Response shape

```jsonc
{
  "profile_id": "unknown",
  "timestamp": "2026-05-08T14:50:00Z",
  "image_analysis": { "score": 0.18, "confidence": 0.92, "prediction": "fake", "explanation": { /* Grad-CAM, indicators */ } },
  "text_analysis":  { "score": 0.41, "confidence": 0.74, "prediction": "ai",   "explanation": { /* LIME tokens */ } },
  "final_trust_score": 0.27,
  "interpretation": "Low trust — likely AI-generated content detected. Modules generally agree.",
  "trust_level": "low",
  "trust_score_details": {
    "module_scores": { "image": 0.18, "text": 0.41 },
    "contributing_factors": { "image": [...], "text": [...] }
  },
  "processing_time_ms": 4172.5
}
```

Interactive docs are served at `/docs` (Swagger) and `/redoc` in non-production environments. Both are disabled when `ENVIRONMENT=production`.

### Error shape

Errors come back as `{ "error": "Type", "message": "...", "timestamp": "..." }`. Internal exception messages are scrubbed before reaching the client; check server logs for the full stack.

---

## 📊 Performance

### Accuracy Metrics
| Module | Model | Accuracy | F1-Score |
|--------|-------|----------|----------|
| Image | Ensemble | 98.2% | 0.981 |
| Text | Ensemble | 95.7% | 0.953 |
| Overall | Fusion | 96.8% | 0.965 |

### Processing Speed
- **Image Analysis**: ~2-3 seconds
- **Text Analysis**: ~1-2 seconds
- **Complete Pipeline**: ~3-5 seconds

### Model Sizes
- **Total Download**: ~300 MB (first run only)
- **Runtime Memory**: ~2-3 GB (CPU mode)
- **GPU Acceleration**: Supported (CUDA)

---

## 👥 Team

**B.Tech CSE Final Year Project 2025-26**
Amity School of Engineering & Technology

**Group 235**

- **Suhani Sidhu** - Frontend Development, UI/UX Design
- **Sumit Kumar Verma** - Backend Development, AI/ML Models

**Project Guide**
- **Dr. Akanshi Gupta** - Faculty Mentor

---

## 🐳 Quickstart with Docker

```bash
# Build + run the backend (downloads ~300MB of model weights on first run)
docker compose up --build

# Backend is now on http://localhost:8000  →  /health, /docs, /api/v1/...

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Set `VITE_API_BASE_URL` if pointing the frontend at a non-local backend.

---

## 📚 Documentation

- **[SETUP.md](./SETUP.md)** - Complete installation and setup guide
- **[project research.pdf](./project%20research.pdf)** - Full research paper
- **[Synopsis.pdf](./Synopsis.pdf)** - Project synopsis and overview
- **API Documentation** - Available at `/docs` when server is running

---

## 🔒 Security & Privacy

- **No Data Storage**: Uploaded images/text are processed in-memory and deleted
- **Local Processing**: All AI models run locally (no cloud dependencies)
- **No Tracking**: No analytics or user tracking
- **Open Source**: Code is transparent and auditable

---

## 🎓 Academic Use

This project is submitted as part of the B.Tech CSE curriculum. If you use this work for academic purposes, please cite:

```bibtex
@project{realitycheck2025,
  title={RealityCheck AI: Multimodal Fake Profile Detection with Explainable AI},
  author={Sidhu, Suhani and Verma, Sumit Kumar},
  institution={Amity School of Engineering \& Technology},
  year={2025},
  supervisor={Gupta, Akanshi}
}
```

---

## 📝 License

This project is submitted for academic evaluation. All rights reserved by the authors.

**For academic and educational purposes only.**

---

## 🙏 Acknowledgments

- **Hugging Face** - Pre-trained transformer models
- **PyTorch Team** - Deep learning framework
- **Ultralytics** - YOLOv8 implementation
- **FastAPI** - Modern web framework
- **React Team** - Frontend framework
- **Research Community** - Deepfake detection papers and datasets

---

## 🔗 Quick Links

- [Setup Guide](./SETUP.md) - Installation instructions
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)
- [Research Paper](./project%20research.pdf) - Full technical details

---

**Built with ❤️ by Team 235 | Amity University**
