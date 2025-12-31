import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ResultsDisplay from '../components/ResultsDisplay';

const ProfileResult = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const results = location.state?.results;

  const handleAnalyzeAnother = () => {
    navigate('/');
  };

  if (!results) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 flex items-center justify-center">
        <div className="card max-w-md text-center">
          <div className="text-gray-600 text-5xl mb-4">🔍</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">No Results Found</h2>
          <p className="text-gray-600 mb-6">
            Please analyze a profile first
          </p>
          <button onClick={handleAnalyzeAnother} className="btn-primary">
            Analyze a Profile
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 py-8 px-4">
      <div className="container mx-auto">
        <ResultsDisplay results={results} onAnalyzeAnother={handleAnalyzeAnother} />
      </div>
    </div>
  );
};

export default ProfileResult;
