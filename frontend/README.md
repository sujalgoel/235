# RealityCheck AI - Frontend

Beautiful React + Tailwind CSS dashboard for RealityCheck AI fake profile detector.

## Features

- 🎨 Modern UI with Tailwind CSS
- 📊 Interactive Trust Score Gauge
- 📸 Image upload with preview
- 📝 Text analysis interface
- 🔍 Detailed results breakdown
- ⚡ Real-time API status indicator
- 📱 Responsive design

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool (super fast!)
- **Tailwind CSS** - Utility-first styling
- **Axios** - API requests

## Installation

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The app will open at: http://localhost:3000

## Usage

### Make sure the backend API is running first!

**Terminal 1: Start Backend**
```bash
cd /Users/apple/Downloads/Suhani
python3.13 -m uvicorn src.api.app:app --reload --port 8000
```

**Terminal 2: Start Frontend**
```bash
cd /Users/apple/Downloads/Suhani/frontend
npm run dev
```

### Analysis Types

1. **Complete Profile** - Analyzes both image and text
2. **Text Only** - Analyzes bio/description only
3. **Image Only** - Analyzes profile picture only

## Components

### TrustScoreGauge
Circular progress indicator showing trust score (0-100%)
- 🚨 0-30%: Likely Fake (Red)
- ⚠️ 30-50%: Suspicious (Orange)
- ⚡ 50-70%: Moderate (Yellow)
- ✅ 70-100%: Likely Real (Green)

### ProfileAnalyzer
Main upload form with:
- Image upload with drag & drop
- Bio text input
- Analysis type selector

### ResultsDisplay
Shows detailed analysis:
- Overall trust score
- Individual module scores
- Explanations and insights
- Key features detected

## Build for Production

```bash
npm run build
```

Output will be in `dist/` folder.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ProfileAnalyzer.jsx
│   │   ├── ResultsDisplay.jsx
│   │   └── TrustScoreGauge.jsx
│   ├── services/
│   │   └── api.js
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── index.html
├── package.json
├── tailwind.config.js
└── vite.config.js
```

## Troubleshooting

### "API Server Disconnected" error

Make sure the backend is running:
```bash
python3.13 -m uvicorn src.api.app:app --reload --port 8000
```

### Port 3000 already in use

Change the port in `vite.config.js`:
```js
server: {
  port: 3001,  // Use different port
}
```

### Module not found errors

Reinstall dependencies:
```bash
rm -rf node_modules package-lock.json
npm install
```

## Team

**Group 235 - B.Tech CSE 2025-26**
- Sumit Kumar Verma
- Suhani Sidhu
- Guide: Dr. Akanshi Gupta

Amity School of Engineering & Technology
