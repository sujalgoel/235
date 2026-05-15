# RealityCheck AI on Windows

A step-by-step install guide for Windows 10 and Windows 11. The project runs on CPU by default. GPU acceleration is optional and covered at the end.

Tested on Windows 11 23H2 with Python 3.13.0 and Node.js 20 LTS.

---

## 1. Prerequisites

You will install three things, in this order:

| Tool | Version | Why |
|---|---|---|
| Python | 3.13 or newer | Backend, AI models |
| Node.js | 20 LTS or newer | Frontend build and dev server |
| Git for Windows | any recent | Clone the repository |

### 1a. Install Python

1. Download the Python 3.13 Windows installer from <https://www.python.org/downloads/windows/>. Pick the "Windows installer (64-bit)" build.
2. Run the installer. **Tick the box that says "Add python.exe to PATH" on the very first screen.** This is the most common cause of `python is not recognized` later.
3. Click "Install Now".
4. Open a fresh PowerShell window and verify:

```powershell
python --version
# Python 3.13.0
pip --version
```

If `python --version` reports something older or shows a Microsoft Store stub, type `where python` and remove any older entries from PATH via System Properties, Environment Variables.

### 1b. Install Node.js

1. Download the LTS installer from <https://nodejs.org/>.
2. Run it with default options. The installer also adds npm to PATH.
3. Verify in a fresh PowerShell window:

```powershell
node --version
# v20.11.x
npm --version
# 10.x.x
```

### 1c. Install Git

1. Download from <https://git-scm.com/download/win>.
2. Use defaults except this one screen: when asked about the line-ending policy, choose **"Checkout as-is, commit Unix-style line endings"**. The repo expects LF endings in shell scripts.
3. Verify:

```powershell
git --version
```

### 1d. Enable long path support (recommended)

Some PyTorch and Hugging Face cache paths exceed the 260-character Windows limit. Run PowerShell **as Administrator** once and execute:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then close and reopen all PowerShell windows.

---

## 2. Clone the repository

Pick a folder without spaces in the path. Spaces sometimes confuse virtualenvs and PyTorch model loaders on Windows. `C:\dev\` works well.

```powershell
mkdir C:\dev
cd C:\dev
git clone https://github.com/sujalgoel/235.git realitycheck
cd realitycheck
```

You should now be sitting in `C:\dev\realitycheck` with `src/`, `frontend/`, `requirements.txt`, and the rest of the project.

---

## 3. Backend setup

### 3a. Create the virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script with a message about execution policy, run this once as Administrator:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then re-run the activation. Your prompt should now show `(venv)` at the start.

If you prefer Command Prompt, use `venv\Scripts\activate.bat` instead.

### 3b. Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This pulls PyTorch, Transformers, FastAPI, OpenCV, Grad-CAM, LIME, SHAP, plus the OpenAI CLIP package from GitHub. Expect 5 to 15 minutes on a typical home connection. The download is around 2 GB once everything is unpacked.

If pip fails on `torch` with a message about wheels, your Python is probably 32-bit or older than 3.13. Reinstall the 64-bit Python 3.13 from step 1a.

If pip fails on `clip` with a Git error, it means the OpenAI CLIP repo could not be cloned. Make sure Git is on PATH and re-run the same `pip install` command.

### 3c. Smoke-check the install

```powershell
python -c "import torch, fastapi, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Output should look like `torch 2.6.x cuda False`. The `False` is expected on a CPU-only machine and is fine.

### 3d. Configure environment variables

Copy the example file:

```powershell
Copy-Item .env.example .env
```

Open `.env` in Notepad and confirm `DEVICE=cpu`. If you have an NVIDIA GPU and want to use it, see section 8.

---

## 4. Frontend setup

Open a **second** PowerShell window so you can run backend and frontend side by side. In the second window:

```powershell
cd C:\dev\realitycheck\frontend
npm install
```

This pulls React, Vite, Tailwind, Axios, Recharts. Expect 1 to 3 minutes. Roughly 300 MB of `node_modules` will appear.

If you see warnings about deprecated packages, ignore them. Errors are different and need attention.

---

## 5. Run the application

You need both servers running at the same time. Two PowerShell windows is the easiest pattern.

### 5a. Backend (window one)

```powershell
cd C:\dev\realitycheck
.\venv\Scripts\Activate.ps1
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The first request triggers model downloads and weight loading. Cold start is roughly 18 to 20 seconds. After that each request completes in 1 to 2 seconds on CPU.

Wait until the log shows `Uvicorn running on http://0.0.0.0:8000` and `Application startup complete` before sending traffic.

### 5b. Frontend (window two)

```powershell
cd C:\dev\realitycheck\frontend
npm run dev
```

Vite will print a URL, usually `http://localhost:5173`. Open it in your browser.

### 5c. Use the app

1. Click "Choose File" and upload a profile photo.
2. Paste a bio in the text area.
3. Click "Analyze".
4. Within a few seconds you should see the trust score, the per-module breakdown, the Grad-CAM heatmap, and the LIME token highlights.

---

## 6. Verification

In a third PowerShell window:

```powershell
curl.exe http://localhost:8000/health
```

Expected response, formatted for readability:

```json
{
  "status": "ok",
  "modules": {
    "image": "ready",
    "text": "ready",
    "metadata": "ready"
  }
}
```

If you see `503 Service Unavailable` the models are still warming up. Wait 20 seconds and try again.

---

## 7. Common Windows issues

### "python is not recognized as an internal or external command"
PATH is missing the Python install directory. Reinstall Python 3.13 from step 1a, this time with **"Add python.exe to PATH"** ticked. Or add it manually via System Properties, Environment Variables.

### Activation script blocked by execution policy
Run this once as Administrator and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Antivirus blocks PyTorch DLLs
Some antivirus products quarantine `torch_cpu.dll` because it imports many small native modules at runtime. If pip succeeds but `import torch` fails with `OSError: [WinError 126]`, add an exclusion for your `venv\Lib\site-packages\torch\` folder in your antivirus settings, then reinstall:
```powershell
pip install --force-reinstall torch
```

### `clip` install fails with `'git' is not recognized`
Git is not on PATH. Reinstall Git for Windows from step 1c, restart PowerShell, then re-run `pip install -r requirements.txt`.

### Port 8000 or 5173 is already in use
Another process holds the port. Find and stop it:
```powershell
netstat -ano | findstr ":8000"
# note the PID in the rightmost column
taskkill /F /PID <PID>
```

### CORS error in the browser
The backend rejects requests from an unexpected origin. Open `.env` and add a line:
```
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```
Then restart the backend.

### npm install hangs at "fetchMetadata"
Corporate proxy or flaky DNS. Set npm to use a public registry:
```powershell
npm config set registry https://registry.npmjs.org/
npm cache clean --force
npm install
```

### The browser shows the page but Analyze does nothing
Open DevTools, look at the Network tab. If the `/api/v1/analyze/profile` request returns red, the backend is not reachable. Confirm the backend window still says "Application startup complete" and that it is on port 8000.

---

## 8. Optional: GPU acceleration with NVIDIA CUDA

If you have a recent NVIDIA card with at least 6 GB VRAM, you can roughly halve inference latency.

1. Install the latest NVIDIA Studio or Game Ready driver from <https://www.nvidia.com/Download/index.aspx>.
2. Confirm CUDA 12 is available:
   ```powershell
   nvidia-smi
   ```
3. Reinstall PyTorch with the CUDA 12 wheels (this replaces the CPU build):
   ```powershell
   pip uninstall -y torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
4. Edit `.env` and change:
   ```
   DEVICE=cuda
   ```
5. Restart the backend. The startup log should now say something like `loaded EfficientNet on cuda:0`.

If `torch.cuda.is_available()` returns `False` after this, your driver is older than the CUDA runtime PyTorch needs. Update the driver and try again.

---

## 9. Stopping the application

In each PowerShell window, press `Ctrl+C`. Wait for the prompt to return before closing the window so file handles release cleanly.

To deactivate the Python virtual environment in the backend window:
```powershell
deactivate
```

---

## 10. Where things live

| Path | What it is |
|---|---|
| `src/api/app.py` | FastAPI entry point (`uvicorn src.api.app:app`) |
| `src/modules/fusion/trust_scorer.py` | Weighted late fusion, the central algorithm |
| `src/modules/image/` | EfficientNet-B7, XceptionNet, CLIP image lane |
| `src/modules/text/` | RoBERTa, ChatGPT-detector, rule-based text lane |
| `frontend/src/` | React 18 UI, Tailwind, SVG charts |
| `models/` | Downloaded model weights (created on first run) |
| `.env` | Local config. Never commit a populated copy. |
| `requirements.txt` | Python dependencies, pinned |
| `frontend/package.json` | Node dependencies |

---

## Need help?

The macOS and Linux walkthrough is in `SETUP.md`. The deployed viva site, including the deck and benchmark results, is at the URL in `vercel.json`.
