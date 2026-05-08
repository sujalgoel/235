import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ProfileAnalyzer from './components/ProfileAnalyzer';
import ProfileResult from './pages/ProfileResult';
import { healthCheck } from './services/api';

function App() {
  // ========================================
  // STATE MANAGEMENT
  // ========================================
  // Track API backend connection status: 'checking', 'connected', or 'disconnected'
  // This determines the color and message of the status banner at the top
  const [apiStatus, setApiStatus] = useState('checking');

  // Poll the backend on mount and every 30s so the banner reflects the
  // backend coming back up after a restart instead of staying red forever.
  useEffect(() => {
    let cancelled = false;

    const ping = async () => {
      try {
        await healthCheck();
        if (!cancelled) setApiStatus('connected');
      } catch (error) {
        if (!cancelled) setApiStatus('disconnected');
      }
    };

    ping();
    const interval = setInterval(ping, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
        {/* ========================================
            API STATUS BANNER
            ========================================
            Visual indicator showing backend connection status
            - Green: API is reachable (ready for analysis)
            - Red: API is down (user needs to start backend server)
            - Yellow: Currently checking connection
        */}
        <div className={`py-2 text-center text-sm font-medium ${
          apiStatus === 'connected'
            ? 'bg-green-500 text-white'
            : apiStatus === 'disconnected'
            ? 'bg-red-500 text-white'
            : 'bg-yellow-500 text-white'
        }`}>
          {apiStatus === 'connected' && '✅ API Server Connected'}
          {apiStatus === 'disconnected' && '❌ API Server Disconnected - Please start the backend server'}
          {apiStatus === 'checking' && '⏳ Checking API Status...'}
        </div>

        {/* ========================================
            HEADER
            ========================================
            Application header with branding and navigation
            Logo links back to homepage (ProfileAnalyzer)
        */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <Link to="/" className="flex items-center hover:opacity-80 transition">
                <div className="text-3xl mr-3">🔍</div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">
                    RealityCheck AI
                  </h1>
                  <p className="text-sm text-gray-600">
                    AI-Generated Content Detector
                  </p>
                </div>
              </Link>
              <div className="hidden md:flex items-center space-x-6 text-sm">
                <div className="text-gray-600">
                  <span className="font-semibold">Image</span>
                  <span className="text-gray-400 mx-1">+</span>
                  <span className="font-semibold">Text</span>
                  <span className="text-gray-400 mx-1">=</span>
                  <span className="font-semibold">AI Detection</span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* ========================================
            ROUTES
            ========================================
            React Router handles navigation between pages:
            - "/" → ProfileAnalyzer: Main upload page (image + text)
            - "/profile" → ProfileResult: Analysis results display
        */}
        <main className="py-8">
          <Routes>
            <Route path="/" element={<ProfileAnalyzer />} />
            <Route path="/profile" element={<ProfileResult />} />
          </Routes>
        </main>

        {/* ========================================
            FOOTER
            ========================================
            Project information and credits
            Shows academic institution, group number, and team members
        */}
        <footer className="bg-white border-t border-gray-200 mt-12">
          <div className="max-w-7xl mx-auto px-4 py-6">
            <div className="text-center text-sm text-gray-600">
              <p className="mb-2">
                <span className="font-semibold">RealityCheck AI</span> - B.Tech CSE Project 2025-26
              </p>
              <p className="text-xs text-gray-500">
                Amity School of Engineering & Technology | Group 235
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Suhani Sidhu & Sumit Kumar Verma | Guide: Dr. Akanshi Gupta
              </p>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
