import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyzeProfile } from '../services/api';

const ProfileAnalyzer = () => {
  // ========================================
  // NAVIGATION HOOK
  // ========================================
  // React Router hook for programmatic navigation
  // Used to redirect to /profile page after successful analysis
  const navigate = useNavigate();

  // ========================================
  // STATE MANAGEMENT
  // ========================================
  // image: File object from file input (sent to backend)
  const [image, setImage] = useState(null);

  // imagePreview: Base64 data URL for displaying image preview
  const [imagePreview, setImagePreview] = useState(null);

  // bioText: User-entered profile bio/description text
  const [bioText, setBioText] = useState('');

  // loading: True when analysis is in progress (shows spinner)
  const [loading, setLoading] = useState(false);

  // error: Error message to display if analysis fails
  const [error, setError] = useState(null);

  // ========================================
  // IMAGE UPLOAD HANDLER
  // ========================================
  // Handles file selection from file input
  // Uses FileReader API to create preview (base64 data URL)
  // FileReader is async, so we use onloadend callback
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);

      // FileReader converts file to base64 data URL for preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);  // Set preview after conversion completes
      };
      reader.readAsDataURL(file);  // Start async conversion
    }
  };

  // ========================================
  // ANALYSIS HANDLER
  // ========================================
  // Main function that sends image + text to backend for AI detection
  // Flow:
  // 1. Validate inputs (must have both image and text)
  // 2. Call API with FormData (multipart/form-data for file upload)
  // 3. On success: Navigate to results page with data in route state
  // 4. On error: Display error message and stop loading spinner
  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);

    try {
      // Validation: Both image and text are required for multimodal analysis
      if (!image || !bioText.trim()) {
        setError('Please provide both an image and bio text');
        setLoading(false);
        return;
      }

      // Call API: analyzeProfile() creates FormData and sends POST request
      // Backend processes through ensemble models and returns trust score
      const response = await analyzeProfile(image, bioText);

      // Navigate to results page, passing analysis data via React Router state
      // This avoids storing results in global state or localStorage
      navigate('/profile', { state: { results: response } });
    } catch (err) {
      // Error handling: Extract error message from API response or use fallback
      setError(
        err.response?.data?.detail ||
        'Analysis failed. Please check if the API server is running and try again.'
      );
      console.error('Analysis error:', err);
      setLoading(false);  // Re-enable analyze button
    }
  };

  // ========================================
  // RESET HANDLER
  // ========================================
  // Clears all form inputs and errors
  // Allows user to start fresh analysis
  const handleReset = () => {
    setImage(null);
    setImagePreview(null);
    setBioText('');
    setError(null);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* ========================================
          PAGE HEADER
          ========================================
          Title and description explaining the tool's purpose
      */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          RealityCheck AI
        </h1>
        <p className="text-lg text-gray-600">
          Detect AI-generated content using multimodal analysis
        </p>
      </div>

      {/* ========================================
          UPLOAD FORM CARD
          ========================================
          Main form for uploading image and entering bio text
      */}
      <div className="card max-w-4xl mx-auto">
          {/* Upload Form */}
          <>
              {/* ========================================
                  IMAGE UPLOAD SECTION
                  ========================================
                  File input with drag-and-drop styling
                  Shows preview after selection
              */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Profile Image
                </label>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-500 transition-colors">
                  {imagePreview ? (
                    <div className="relative">
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="max-h-64 mx-auto rounded-lg"
                      />
                      <button
                        onClick={() => {
                          setImage(null);
                          setImagePreview(null);
                        }}
                        className="absolute top-2 right-2 bg-red-500 text-white px-3 py-1 rounded-lg text-sm hover:bg-red-600"
                      >
                        Remove
                      </button>
                    </div>
                  ) : (
                    <div>
                      <svg
                        className="mx-auto h-12 w-12 text-gray-400"
                        stroke="currentColor"
                        fill="none"
                        viewBox="0 0 48 48"
                      >
                        <path
                          d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                          strokeWidth={2}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <div className="mt-4">
                        <label
                          htmlFor="file-upload"
                          className="cursor-pointer bg-primary-600 text-white px-4 py-2 rounded-lg hover:bg-primary-700 transition-colors"
                        >
                          Upload Image
                        </label>
                        <input
                          id="file-upload"
                          type="file"
                          className="hidden"
                          accept="image/*"
                          onChange={handleImageChange}
                        />
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        PNG, JPG, JPEG up to 10MB
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* ========================================
                  BIO TEXT INPUT SECTION
                  ========================================
                  Textarea for profile bio/description
                  Shows character count for user feedback
              */}
              <div className="mb-6">
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Profile Bio / Description
                </label>
                <textarea
                  value={bioText}
                  onChange={(e) => setBioText(e.target.value)}
                  className="input-field h-32 resize-none"
                  placeholder="Enter the profile bio or description here..."
                />
                <div className="text-xs text-gray-500 mt-1">
                  {bioText.length} characters
                </div>
              </div>
            </>

          {/* ========================================
              ERROR DISPLAY
              ========================================
              Shows validation errors or API errors
              Displayed with red styling and error icon
          */}
          {error && (
            <div className="mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              <div className="flex items-start">
                <span className="text-xl mr-2">❌</span>
                <div>
                  <p className="font-semibold">Error</p>
                  <p className="text-sm">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* ========================================
              ACTION BUTTONS
              ========================================
              - Analyze Button: Triggers AI analysis (disabled during loading)
                Shows spinner during processing
              - Reset Button: Clears all inputs
          */}
          <div className="flex gap-4">
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="btn-primary flex-1 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  {/* Loading spinner SVG */}
                  <svg
                    className="animate-spin h-5 w-5 mr-2"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Analyzing...
                </span>
              ) : (
                'Analyze for AI-Generated Content'
              )}
            </button>
            <button onClick={handleReset} className="btn-secondary">
              Reset
            </button>
          </div>
      </div>
    </div>
  );
};

export default ProfileAnalyzer;
