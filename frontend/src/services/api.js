/**
 * ========================================
 * API SERVICE LAYER
 * ========================================
 * Handles all HTTP requests to the FastAPI backend
 * Uses axios for HTTP client and FormData for file uploads
 */

import axios from 'axios';

// ========================================
// CONFIGURATION
// ========================================
// Base URL for all API endpoints
// Backend must be running on localhost:8000 for this to work
const API_BASE_URL = 'http://localhost:8000/api/v1';

// ========================================
// AXIOS INSTANCE
// ========================================
// Pre-configured axios instance with base URL and headers
// Uses multipart/form-data for file upload support
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

/**
 * ========================================
 * ANALYZE PROFILE
 * ========================================
 * Main API call for multimodal analysis (image + text)
 *
 * Sends to: POST /api/v1/analyze/profile
 *
 * Flow:
 * 1. Create FormData object (required for file upload)
 * 2. Append image file, bio text, and optional profile ID
 * 3. Send POST request to backend
 * 4. Backend processes through ensemble models
 * 5. Returns trust score and detailed analysis
 *
 * @param {File} image - Image file from file input
 * @param {string} bioText - Profile bio/description text
 * @param {string} profileId - Optional identifier for logging
 * @returns {Promise<Object>} Analysis results with trust score
 */
export const analyzeProfile = async (image, bioText, profileId = null) => {
  // Create FormData for multipart/form-data encoding
  // Required for sending files over HTTP
  const formData = new FormData();

  if (image) {
    formData.append('image', image);
  }

  if (bioText) {
    formData.append('bio_text', bioText);
  }

  if (profileId) {
    formData.append('profile_id', profileId);
  }

  try {
    const response = await api.post('/analyze/profile', formData);
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;  // Re-throw to let caller handle the error
  }
};

/**
 * ========================================
 * ANALYZE TEXT ONLY
 * ========================================
 * Analyze text without image
 *
 * Sends to: POST /api/v1/analyze/text
 *
 * @param {string} text - Text to analyze
 * @returns {Promise<Object>} Text analysis results
 */
export const analyzeText = async (text) => {
  const formData = new FormData();
  formData.append('text', text);

  try {
    const response = await api.post('/analyze/text', formData);
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

/**
 * ========================================
 * ANALYZE IMAGE ONLY
 * ========================================
 * Analyze image without text
 *
 * Sends to: POST /api/v1/analyze/image
 *
 * @param {File} image - Image file to analyze
 * @returns {Promise<Object>} Image analysis results
 */
export const analyzeImage = async (image) => {
  const formData = new FormData();
  formData.append('image', image);

  try {
    const response = await api.post('/analyze/image', formData);
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

/**
 * ========================================
 * HEALTH CHECK
 * ========================================
 * Check if backend API is running and responsive
 *
 * Sends to: GET /health
 *
 * Used by App.jsx to display connection status banner
 * Called on component mount to verify backend availability
 *
 * @returns {Promise<Object>} Health status object
 * @throws {Error} If backend is not reachable
 */
export const healthCheck = async () => {
  try {
    // Direct axios call (not using api instance) to hit /health endpoint
    const response = await axios.get('http://localhost:8000/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

export default api;
