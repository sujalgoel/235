// Single source of truth for trust-score bands. Mirrors the backend
// thresholds in src/modules/fusion/trust_scorer.py and config FusionConfig.
// Update both sides together if these thresholds change.

export const TRUST_BANDS = [
  { max: 0.3, key: 'very_low', label: 'LIKELY FAKE',  short: 'Likely Fake',  icon: '🚨', color: 'red',    stop: '#dc2626' },
  { max: 0.5, key: 'low',      label: 'SUSPICIOUS',   short: 'Suspicious',   icon: '⚠️', color: 'orange', stop: '#ea580c' },
  { max: 0.7, key: 'moderate', label: 'MODERATE',     short: 'Moderate',     icon: '⚡', color: 'yellow', stop: '#ca8a04' },
  { max: 1.01, key: 'high',    label: 'LIKELY REAL',  short: 'Likely Real',  icon: '✅', color: 'green',  stop: '#16a34a' },
];

export const NEUTRAL_BAND = { key: 'unknown', label: 'NOT ANALYZED', short: 'Not Analyzed', icon: '—', color: 'gray', stop: '#6b7280' };

export const bandForScore = (score) => {
  if (score === null || score === undefined || Number.isNaN(score)) return NEUTRAL_BAND;
  return TRUST_BANDS.find((b) => score < b.max) ?? TRUST_BANDS[TRUST_BANDS.length - 1];
};
