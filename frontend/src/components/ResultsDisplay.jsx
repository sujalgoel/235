import React from 'react';
import TrustScoreGauge from './TrustScoreGauge';

const ResultsDisplay = ({ results, onAnalyzeAnother }) => {
  // ========================================
  // DESTRUCTURE RESULTS
  // ========================================
  // Extract main fields from API response:
  // - final_trust_score: Overall trust score (0-1)
  // - interpretation: Human-readable explanation
  // - image_analysis: Image module results (score, prediction, explanation)
  // - text_analysis: Text module results
  // - metadata_analysis: Metadata module results (usually null)
  const { final_trust_score, interpretation, image_analysis, text_analysis, metadata_analysis } = results;

  // ========================================
  // HELPER: Extract Score from Module Result
  // ========================================
  // Safely get score from module result object
  // Returns null if module didn't run or has no score
  const getModuleScore = (module) => {
    if (!module) return null;
    return module.score !== undefined ? module.score : null;
  };

  // ========================================
  // HELPER: Determine Status Category from Score
  // ========================================
  // Converts numerical score (0-1) to categorical label with color
  // Thresholds:
  // - < 0.3: Likely Fake (red)
  // - < 0.5: Suspicious (orange)
  // - < 0.7: Moderate (yellow)
  // - >= 0.7: Likely Real (green)
  const getModuleStatus = (score) => {
    if (score === null) return { text: 'Not Analyzed', color: 'gray' };
    if (score < 0.3) return { text: 'Likely Fake', color: 'red' };
    if (score < 0.5) return { text: 'Suspicious', color: 'orange' };
    if (score < 0.7) return { text: 'Moderate', color: 'yellow' };
    return { text: 'Likely Real', color: 'green' };
  };

  // ========================================
  // COMPONENT: ModuleCard
  // ========================================
  // Reusable card component for displaying individual module results
  // Used for Image, Text, and Metadata analysis cards
  //
  // Props:
  // - title: Module name (e.g., "Image Analysis")
  // - score: Authenticity score (0-1)
  // - prediction: Category label (e.g., "REAL", "FAKE")
  // - explanation: Detailed analysis data (varies by module)
  // - icon: Emoji icon for visual identification
  const ModuleCard = ({ title, score, prediction, explanation, icon }) => {
    const status = getModuleStatus(score);

    // Check if this is image analysis with Grad-CAM visualizations
    // Grad-CAM heatmaps show which parts of the image influenced the decision
    const hasGradCAM = explanation?.grad_cam_heatmap || explanation?.grad_cam_overlay;

    return (
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <span className="text-2xl mr-2">{icon}</span>
            <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          </div>
          {score !== null && (
            <div className={`text-${status.color}-600 font-bold text-xl`}>
              {Math.round(score * 100)}%
            </div>
          )}
        </div>

        {score !== null ? (
          <>
            <div className={`inline-block px-3 py-1 rounded-full text-sm font-semibold mb-3 bg-${status.color}-100 text-${status.color}-700`}>
              {status.text}
            </div>

            {prediction && (
              <div className="mb-3">
                <span className="text-sm text-gray-600">Prediction: </span>
                <span className="text-sm font-semibold text-gray-900 uppercase">
                  {prediction}
                </span>
              </div>
            )}

            {explanation && Object.keys(explanation).length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-200">
                <h4 className="text-xs font-semibold text-gray-700 mb-2 uppercase">
                  Analysis Details
                </h4>
                <div className="text-sm text-gray-600 space-y-1">
                  {Object.entries(explanation).map(([key, value]) => {
                    // Skip image fields - they'll be shown separately
                    if (key === 'grad_cam_heatmap' || key === 'grad_cam_overlay') {
                      return null;
                    }

                    // Handle arrays (like artifacts_detected, top_features)
                    if (Array.isArray(value)) {
                      return (
                        <div key={key} className="mb-2">
                          <span className="capitalize font-semibold block mb-1">
                            {key.replace(/_/g, ' ')}:
                          </span>
                          <ul className="list-disc list-inside ml-2">
                            {value.map((item, idx) => {
                              // Handle objects within arrays (e.g., {token, importance})
                              if (typeof item === 'object' && item !== null) {
                                if (item.token && item.importance !== undefined) {
                                  return (
                                    <li key={idx}>
                                      {item.token}: {(item.importance * 100).toFixed(1)}%
                                    </li>
                                  );
                                }
                                return <li key={idx}>{JSON.stringify(item)}</li>;
                              }
                              return <li key={idx}>{String(item)}</li>;
                            })}
                          </ul>
                        </div>
                      );
                    }

                    // Handle objects
                    if (typeof value === 'object' && value !== null) {
                      return (
                        <div key={key} className="mb-2">
                          <span className="capitalize font-semibold block mb-1">
                            {key.replace(/_/g, ' ')}:
                          </span>
                          <pre className="text-xs bg-gray-50 p-2 rounded overflow-auto">
                            {JSON.stringify(value, null, 2)}
                          </pre>
                        </div>
                      );
                    }

                    return (
                      <div key={key} className="flex justify-between">
                        <span className="capitalize">{key.replace(/_/g, ' ')}:</span>
                        <span className="font-medium">
                          {typeof value === 'number' ? value.toFixed(3) : String(value)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-gray-500 text-sm">
            This module was not included in the analysis
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* ========================================
          PAGE HEADER
          ========================================
          Title and description for results page
      */}
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Analysis Results
        </h2>
        <p className="text-gray-600">
          Multimodal fake profile detection report
        </p>
      </div>


      {/* ========================================
          TRUST SCORE SECTION
          ========================================
          Main results card showing:
          - Overall trust score (gauge visualization)
          - Human-readable interpretation
          - Individual module scores breakdown
      */}
      <div className="card mb-8 bg-gradient-to-br from-blue-50 to-indigo-50">
        <div className="text-center mb-6">
          <h3 className="text-2xl font-bold text-gray-900 mb-2">
            Overall Trust Score
          </h3>
          <p className="text-gray-600">{interpretation}</p>
        </div>
        <TrustScoreGauge score={final_trust_score} size="large" />

        {/* Score Breakdown */}
        <div className="mt-8 grid grid-cols-3 gap-4">
          {image_analysis && (
            <div className="text-center p-4 bg-white rounded-lg">
              <div className="text-2xl mb-1">📸</div>
              <div className="text-sm text-gray-600 mb-1">Image</div>
              <div className="text-xl font-bold text-primary-600">
                {Math.round(getModuleScore(image_analysis) * 100)}%
              </div>
            </div>
          )}
          {text_analysis && (
            <div className="text-center p-4 bg-white rounded-lg">
              <div className="text-2xl mb-1">📝</div>
              <div className="text-sm text-gray-600 mb-1">Text</div>
              <div className="text-xl font-bold text-primary-600">
                {Math.round(getModuleScore(text_analysis) * 100)}%
              </div>
            </div>
          )}
          {metadata_analysis && (
            <div className="text-center p-4 bg-white rounded-lg">
              <div className="text-2xl mb-1">🔍</div>
              <div className="text-sm text-gray-600 mb-1">Metadata</div>
              <div className="text-xl font-bold text-primary-600">
                {Math.round(getModuleScore(metadata_analysis) * 100)}%
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ========================================
          DETAILED ANALYSIS CARDS
          ========================================
          Individual module result cards showing:
          - Score percentage
          - Prediction label (REAL/FAKE)
          - Detailed explanation data
          Only displays cards for modules that ran
      */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {image_analysis && (
          <ModuleCard
            title="Image Analysis"
            score={getModuleScore(image_analysis)}
            prediction={image_analysis.prediction}
            explanation={image_analysis.explanation}
            icon="📸"
          />
        )}

        {text_analysis && (
          <ModuleCard
            title="Text Analysis"
            score={getModuleScore(text_analysis)}
            prediction={text_analysis.prediction}
            explanation={text_analysis.explanation}
            icon="📝"
          />
        )}

        {metadata_analysis && (
          <ModuleCard
            title="Metadata Analysis"
            score={getModuleScore(metadata_analysis)}
            prediction={metadata_analysis.prediction}
            explanation={metadata_analysis.explanation}
            icon="🔍"
          />
        )}
      </div>

      {/* ========================================
          GRAD-CAM VISUALIZATIONS
          ========================================
          Visual explanations from image analysis
          Shows heatmap and overlay highlighting important image regions
          Red/yellow areas = high importance for model's decision
          Only displayed if image analysis generated Grad-CAM outputs
      */}
      {image_analysis && image_analysis.explanation &&
       (image_analysis.explanation.grad_cam_heatmap || image_analysis.explanation.grad_cam_overlay) && (
        <div className="card mb-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <span className="mr-2">🔥</span>
            Grad-CAM Heatmap Visualization
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Visual explanation showing which regions of the image influenced the model's decision.
            Red/yellow areas indicate high importance.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {image_analysis.explanation.grad_cam_heatmap && (
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2">
                  Heatmap
                </h4>
                <img
                  src={image_analysis.explanation.grad_cam_heatmap}
                  alt="Grad-CAM Heatmap"
                  className="w-full rounded-lg border border-gray-200"
                />
              </div>
            )}

            {image_analysis.explanation.grad_cam_overlay && (
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2">
                  Overlay on Original Image
                </h4>
                <img
                  src={image_analysis.explanation.grad_cam_overlay}
                  alt="Grad-CAM Overlay"
                  className="w-full rounded-lg border border-gray-200"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================
          TEXT ANALYSIS INSIGHTS
          ========================================
          Detailed linguistic analysis showing:
          - Key Features: Notable patterns in the text
          - AI Indicators: Specific signals suggesting AI generation
            (e.g., repetitive phrasing, unnatural vocabulary)
          Only displayed if text analysis was performed
      */}
      {text_analysis && text_analysis.explanation && (
        <div className="card mb-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <span className="mr-2">💭</span>
            Text Analysis Insights
          </h3>

          {/* Key linguistic features detected by ensemble models */}
          {text_analysis.explanation.key_features && (
            <div className="mb-4">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">
                Key Features Detected:
              </h4>
              <div className="flex flex-wrap gap-2">
                {text_analysis.explanation.key_features.map((feature, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Specific indicators that suggest AI-generated text */}
          {text_analysis.explanation.ai_indicators && (
            <div>
              <h4 className="text-sm font-semibold text-red-700 mb-2">
                ⚠️ AI-Generated Indicators:
              </h4>
              <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                {text_analysis.explanation.ai_indicators.map((indicator, idx) => (
                  <li key={idx}>{indicator}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* ========================================
          ACTION BUTTON
          ========================================
          Button to return to upload page for new analysis
          Calls onAnalyzeAnother callback (passed from parent)
      */}
      <div className="text-center">
        <button onClick={onAnalyzeAnother} className="btn-primary">
          Analyze Another Profile
        </button>
      </div>
    </div>
  );
};

export default ResultsDisplay;
