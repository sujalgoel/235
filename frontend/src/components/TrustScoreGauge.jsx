import React from 'react';
import { bandForScore } from '../utils/trustScore';

// JIT-safe class map keyed by the trust-band color.
const COLOR_CLASS = {
  red:    'text-red-600',
  orange: 'text-orange-600',
  yellow: 'text-yellow-600',
  green:  'text-green-600',
  gray:   'text-gray-600',
};

const TrustScoreGauge = ({ score, size = 'large' }) => {
  const percentage = Math.round(score * 100);
  const band = bandForScore(score);
  const textClass = COLOR_CLASS[band.color] ?? COLOR_CLASS.gray;

  const circumference = 2 * Math.PI * 90;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const sizeClasses = {
    small: 'w-32 h-32',
    medium: 'w-48 h-48',
    large: 'w-64 h-64',
  };

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
            {/* SVG ignores Tailwind classes — set the stop color directly
                so the gauge actually renders the correct band color. */}
            <stop offset="0%" stopColor={band.stop} />
            <stop offset="100%" stopColor={band.stop} />
          </linearGradient>
        </defs>
      </svg>

      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className={`text-5xl font-bold ${textClass}`}>
          {percentage}%
        </div>
        <div className="text-sm text-gray-500 mt-1">Trust Score</div>
        <div className="mt-3 text-center">
          <div className="text-2xl">{band.icon}</div>
          <div className={`text-xs font-semibold mt-1 ${textClass}`}>
            {band.label}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrustScoreGauge;
