# 🚀 RealityCheck AI - Complete Setup Guide

This guide will walk you through setting up and running RealityCheck AI on your local machine.

---

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [1. Clone/Download Project](#1-clonedownload-project)
  - [2. Backend Setup](#2-backend-setup)
  - [3. Frontend Setup](#3-frontend-setup)
- [Running the Application](#running-the-application)
  - [Method 1: Separate Terminals (Recommended)](#method-1-separate-terminals-recommended)
  - [Method 2: Background Processes](#method-2-background-processes)
- [Verification](#verification)
- [Testing the System](#testing-the-system)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)
- [Production Deployment](#production-deployment)

---

## ✅ Prerequisites

Before starting, ensure you have the following installed:

### Required Software

1. **Python 3.13** or higher
   ```bash
   # Check Python version
   python3 --version
   # Should output: Python 3.13.x
   ```

2. **Node.js 18+** and **npm**
   ```bash
   # Check Node.js version
   node --version
   # Should output: v18.x.x or higher

   # Check npm version
   npm --version
   # Should output: 9.x.x or higher
   ```

3. **Git** (for cloning the repository)
   ```bash
   git --version
   ```

### macOS Users
Install using Homebrew:
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.13
brew install python@3.13

# Install Node.js
brew install node
```

### Windows Users
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Download Node.js from [nodejs.org](https://nodejs.org/)
3. Ensure "Add to PATH" is checked during installation

### Linux Users
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.13 python3.13-venv nodejs npm

# Fedora
sudo dnf install python3.13 nodejs npm
```

---

## 💻 System Requirements

### Minimum Requirements
- **CPU**: Dual-core processor (2.0 GHz+)
- **RAM**: 4 GB
- **Storage**: 5 GB free space
- **OS**: macOS 10.15+, Windows 10+, or Linux

### Recommended Requirements
- **CPU**: Quad-core processor (3.0 GHz+)
- **RAM**: 8 GB or more
- **Storage**: 10 GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)

### Network
- **Internet connection** required for initial setup (downloading AI models ~300 MB)
- **No internet required** after setup (all processing is local)

---

## 📥 Installation

### 1. Clone/Download Project

If you have the project folder already, skip to step 2.

```bash
# Navigate to where you want the project
cd ~/Downloads

# The project folder should be: Suhani/
cd Suhani
```

Verify project structure:
```bash
ls -la
# You should see: src/, frontend/, config/, models/, requirements.txt, etc.
```

---

### 2. Backend Setup

#### Step 2.1: Create Python Virtual Environment

Navigate to project root:
```bash
cd /Users/apple/Downloads/Suhani
```

Create virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Your terminal prompt should now show (venv)
```

**Windows users:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Step 2.2: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# This will install:
# - PyTorch (deep learning framework)
# - FastAPI (web framework)
# - Transformers (NLP models)
# - YOLOv8 (object detection)
# - SHAP/LIME (explainability)
# - And ~40 other dependencies
#
# ⏱️ This may take 5-10 minutes
```

**Expected output:**
```
Successfully installed torch-2.0.0 fastapi-0.100.0 transformers-4.30.0 ...
```

#### Step 2.3: Verify Backend Installation

```bash
# Check if Python packages installed correctly
python -c "import torch; import fastapi; import transformers; print('✅ All packages installed!')"
```

#### Step 2.4: Configure Environment Variables

The `.env` file is already configured. Verify it exists:
```bash
cat .env
```

You should see 12 configuration variables:
- **ENVIRONMENT** - development/production/testing
- **IMAGE_MODEL_PATH, TEXT_MODEL_PATH** - AI model paths
- **DEVICE** - cpu/cuda (GPU support)
- **API_HOST, API_PORT** - Server settings
- **IMAGE_WEIGHT, TEXT_WEIGHT, METADATA_WEIGHT** - Fusion weights (0.4, 0.3, 0.3)
- **LOG_LEVEL** - Logging verbosity
- **SECRET_KEY** - Security key
- **USE_CLOUD_APIS** - Ensemble mode toggle

**No changes needed** unless you want to customize settings.

---

### 3. Frontend Setup

#### Step 3.1: Navigate to Frontend Directory

```bash
cd frontend
```

#### Step 3.2: Install Node Dependencies

```bash
# Install all npm packages
npm install

# This will install:
# - React
# - React Router
# - Tailwind CSS
# - Axios
# - Development tools
#
# ⏱️ This may take 3-5 minutes
```

**Expected output:**
```
added 1234 packages in 3m
```

#### Step 3.3: Verify Frontend Installation

```bash
# Check if packages installed
npm list react react-dom
```

---

## ▶️ Running the Application

### Method 1: Separate Terminals (Recommended)

This method lets you see logs from both backend and frontend.

#### Terminal 1: Start Backend

```bash
# Navigate to project root
cd /Users/apple/Downloads/Suhani

# Activate virtual environment
source venv/bin/activate

# Start FastAPI server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

**First Run:** Backend will auto-download AI models (~300 MB, 5-10 minutes):
```
Downloading ResNet-18...
Downloading DistilBERT...
Downloading YOLOv8...
✅ Models ready!
```

Keep this terminal open!

#### Terminal 2: Start Frontend

Open a **new terminal window**:

```bash
# Navigate to frontend directory
cd /Users/apple/Downloads/Suhani/frontend

# Start React development server
npm run dev

# You should see:
# VITE v4.x.x ready in 1234 ms
# ➜ Local: http://localhost:5173/
```

Keep this terminal open!

---

### Method 2: Background Processes

For experienced users who prefer background processes:

```bash
# Start backend in background
cd /Users/apple/Downloads/Suhani
source venv/bin/activate
nohup python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
echo "Backend PID: $!"

# Start frontend in background
cd frontend
nohup npm run dev > frontend.log 2>&1 &
echo "Frontend PID: $!"

# View logs
tail -f ../backend.log    # Backend logs
tail -f frontend.log      # Frontend logs

# Stop processes later
kill <PID>
```

---

## ✅ Verification

### 1. Check Backend Health

Open browser or use curl:
```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "modules": {
    "image": true,
    "text": true,
    "metadata": true
  }
}
```

### 2. Check Frontend

Open browser:
```
http://localhost:5173
```

You should see:
- ✅ Green banner: "API Server Connected"
- 🔍 RealityCheck AI header
- Upload form with image and text inputs

### 3. Check API Documentation

```
http://localhost:8000/docs
```

Interactive API documentation (Swagger UI) should load.

---

## 🧪 Testing the System

### Test 1: Basic Profile Analysis

1. Open frontend: `http://localhost:5173`
2. Upload a profile image (any JPEG/PNG)
3. Enter bio text:
   ```
   I'm passionate about innovation and technology. I love creating AI solutions that make a difference in people's lives.
   ```
4. Click "Analyze for AI-Generated Content"
5. Wait 5-10 seconds
6. You should see:
   - Trust score (0-100%)
   - Image analysis with score
   - Text analysis with score
   - Detailed explanations

### Test 2: API Direct Testing

Test the API directly:

```bash
# Create test image (1x1 pixel PNG)
echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" | base64 -d > test.png

# Test API endpoint
curl -X POST http://localhost:8000/api/v1/analyze/profile \
  -F "image=@test.png" \
  -F "bio_text=I am a passionate developer"

# Should return JSON with trust_score, interpretation, etc.
```

### Test 3: Check Logs

Backend logs should show:
```
INFO: analyzing_profile profile_id=None has_image=True has_text=True
INFO: analyzing_image
INFO: analyzing_text
INFO: analysis_complete trust_score=0.65 processing_time_ms=3542
```

---

## 🔧 Troubleshooting

### Problem: Backend won't start

**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

### Problem: Frontend won't start

**Error:** `Cannot find module 'react'`

**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

---

### Problem: "API Server Disconnected" banner

**Causes:**
1. Backend is not running
2. Backend crashed
3. Port 8000 is blocked

**Solution:**
```bash
# Check if backend is running
curl http://localhost:8000/health

# If not, restart backend
cd /Users/apple/Downloads/Suhani
source venv/bin/activate
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Check backend logs for errors
```

---

### Problem: Models not downloading

**Error:** `FileNotFoundError: models/image/resnet18_v1.0.pth`

**Solution:**
```bash
# Ensure internet connection
ping google.com

# Check models directory exists
mkdir -p models/image models/text models/fusion

# Restart backend (will retry download)
```

---

### Problem: Out of memory

**Error:** `RuntimeError: CUDA out of memory` or system freeze

**Solution:**
```bash
# Edit .env to use CPU mode
echo "DEVICE=cpu" >> .env

# Restart backend
```

---

### Problem: Port already in use

**Error:** `Address already in use: 8000`

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001
```

---

### Problem: CORS errors in browser

**Error:** `Access to XMLHttpRequest blocked by CORS policy`

**Solution:**
1. Ensure backend is running on port 8000
2. Frontend should be on port 5173
3. Check `.env` has `API_HOST=0.0.0.0`

---

### Problem: Slow processing

**Symptoms:** Analysis takes >30 seconds

**Solutions:**
1. **First run is always slower** (models loading into memory)
2. **Use GPU if available:**
   ```bash
   # Edit .env
   DEVICE=cuda
   ```
3. **Close other applications** to free RAM
4. **Check CPU usage:**
   ```bash
   top
   # Python process should be using CPU
   ```

---

## ⚙️ Configuration

### Environment Variables (.env)

Key settings you can modify:

```bash
# Device: 'cpu' or 'cuda'
DEVICE=cpu

# API Port (default: 8000)
API_PORT=8000

# Fusion Weights (must sum to 1.0)
IMAGE_WEIGHT=0.4
TEXT_WEIGHT=0.3
METADATA_WEIGHT=0.3

# Logging Level
LOG_LEVEL=INFO  # DEBUG for verbose logs

# Ensemble Mode (use advanced models)
USE_CLOUD_APIS=true
```

### Frontend Configuration

Edit `frontend/src/services/api.js`:

```javascript
// Change API base URL if backend is on different port
const API_BASE_URL = 'http://localhost:8000/api/v1';
```

---

## 🌐 Production Deployment

### For Demo/Presentation

1. **Start both services** (see Running section)
2. **Access frontend** at `http://localhost:5173`
3. **Share screen** or connect to projector
4. **Prepare test cases** beforehand

### For Server Deployment

1. **Build frontend:**
   ```bash
   cd frontend
   npm run build
   # Creates optimized build in dist/
   ```

2. **Serve with production server:**
   ```bash
   # Backend
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4

   # Frontend (using nginx or serve)
   npm install -g serve
   serve -s dist -p 3000
   ```

3. **Use reverse proxy** (nginx) for production

---

## 📊 Performance Tips

### Speed Up Processing

1. **GPU Acceleration** (if available):
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"

   # If True, set DEVICE=cuda in .env
   ```

2. **Reduce Model Size** (lower accuracy):
   ```bash
   # In .env, disable ensemble mode
   USE_CLOUD_APIS=false
   ```

---

## 🔐 Security Notes

- **Local Processing:** All data processed locally, not sent to cloud
- **Temporary Files:** Uploads auto-deleted after processing
- **No Tracking:** No analytics or data collection
- **Change SECRET_KEY** in production:
  ```bash
  # Generate new key
  python -c "import secrets; print(secrets.token_hex(32))"
  # Copy to .env
  ```

---

## 📝 Log Files

Logs are saved to:
- `logs/realitycheck.log` - Application logs
- Backend console - Real-time request logs

View logs:
```bash
tail -f logs/realitycheck.log
```

---

## 🛑 Stopping the Application

### Terminal Method
Press `Ctrl + C` in each terminal running backend/frontend

### Background Process Method
```bash
# Find PIDs
ps aux | grep uvicorn
ps aux | grep vite

# Kill processes
kill <PID>
```

### Complete Cleanup
```bash
# Deactivate virtual environment
deactivate

# Kill all related processes
pkill -f uvicorn
pkill -f vite
```

---

## 📞 Getting Help

### Common Issues
1. Check [Troubleshooting](#troubleshooting) section above
2. Review logs: `logs/realitycheck.log`
3. Check API health: `curl http://localhost:8000/health`

### Debug Mode
```bash
# Enable debug logging
# Edit .env:
LOG_LEVEL=DEBUG

# Restart backend
```

### Report Issues
If you encounter bugs, note:
1. Error message (full traceback)
2. Steps to reproduce
3. System info (OS, Python version)
4. Screenshot (if UI issue)

---

## ✅ Quick Start Checklist

- [ ] Python 3.13+ installed
- [ ] Node.js 18+ installed
- [ ] Project downloaded/cloned
- [ ] Backend dependencies installed (`pip install -r requirements.txt`)
- [ ] Frontend dependencies installed (`npm install`)
- [ ] Backend running (port 8000)
- [ ] Frontend running (port 5173)
- [ ] Health check passes
- [ ] Test analysis completed successfully

---

## 🎯 Next Steps

Once setup is complete:

1. **Read the documentation** - Understand system architecture
2. **Try different inputs** - Test with various images/text
3. **Explore explanations** - See Grad-CAM and SHAP outputs
4. **Customize settings** - Adjust fusion weights, thresholds
5. **Prepare demo** - Create test cases for presentation

---

## 📚 Additional Resources

- **API Documentation:** `http://localhost:8000/docs`
- **Research Paper:** `project research.pdf`
- **Project Synopsis:** `Synopsis.pdf`
- **README:** `README.md`

---

**Setup Questions?** Review logs and troubleshooting section first.

**Ready to analyze profiles!** 🚀
