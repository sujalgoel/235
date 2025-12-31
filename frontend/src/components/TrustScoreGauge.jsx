import React from 'react';

const TrustScoreGauge = ({ score, size = 'large' }) => {
  const percentage = Math.round(score * 100);

  const getScoreColor = (score) => {
    if (score < 0.3) return 'text-red-600';
    if (score < 0.5) return 'text-orange-600';
    if (score < 0.7) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getScoreBackground = (score) => {
    if (score < 0.3) return 'from-red-500 to-red-600';
    if (score < 0.5) return 'from-orange-500 to-orange-600';
    if (score < 0.7) return 'from-yellow-500 to-yellow-600';
    return 'from-green-500 to-green-600';
  };

  const getScoreLabel = (score) => {
    if (score < 0.3) return { text: 'LIKELY FAKE', icon: '🚨' };
    if (score < 0.5) return { text: 'SUSPICIOUS', icon: '⚠️' };
    if (score < 0.7) return { text: 'MODERATE', icon: '⚡' };
    return { text: 'LIKELY REAL', icon: '✅' };
  };

  const circumference = 2 * Math.PI * 90;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const sizeClasses = {
    small: 'w-32 h-32',
    medium: 'w-48 h-48',
    large: 'w-64 h-64',
  };

  const label = getScoreLabel(score);

  return (
    <div className={`relative ${sizeClasses[size]} mx-auto`}>
      {/* SVG Circle Gauge */}
      <svg className="transform -rotate-90 w-full h-full">
        {/* Background circle */}
        <circle
          cx="50%"
          cy="50%"
          r="90"
          stroke="currentColor"
          strokeWidth="12"
          fill="none"
          className="text-gray-200"
        />
        {/* Progress circle */}
        <circle
          cx="50%"
          cy="50%"
          r="90"
          stroke="url(#gradient)"
          strokeWidth="12"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
        />
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop
              offset="0%"
              className={getScoreBackground(score).split(' ')[0].replace('from-', '')}
              stopColor={`var(--tw-gradient-stops)`}
            />
          </linearGradient>
        </defs>
      </svg>

      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className={`text-5xl font-bold ${getScoreColor(score)}`}>
          {percentage}%
        </div>
        <div className="text-sm text-gray-500 mt-1">Trust Score</div>
        <div className="mt-3 text-center">
          <div className="text-2xl">{label.icon}</div>
          <div className={`text-xs font-semibold mt-1 ${getScoreColor(score)}`}>
            {label.text}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrustScoreGauge;
