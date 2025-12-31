"""
Metadata forensics analyzer with scoring algorithm.

Implements the scoring algorithm from the paper (Equation 4):
M = α·C + β·V + γ·A

Where:
- C: Completeness score
- V: Validity score
- A: Anomaly score
- α=0.5, β=0.3, γ=0.2
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime
import re

from src.modules.base import BaseModule, ModuleResult
from src.modules.metadata.extractor import MetadataExtractor
from src.utils.logging import get_logger
from src.utils.exceptions import MetadataExtractionError, PredictionError

logger = get_logger(__name__)


class MetadataForensicsModule(BaseModule):
    """
    Metadata forensics module for authenticity detection.

    Analyzes EXIF metadata to detect:
    - Missing or incomplete metadata
    - Inconsistent values
    - Suspicious software signatures
    - Anomalous patterns
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, name="MetadataForensicsModule")

        # Scoring weights (from paper)
        self.completeness_weight = config.get("completeness_weight", 0.5)  # α
        self.validity_weight = config.get("validity_weight", 0.3)  # β
        self.anomaly_weight = config.get("anomaly_weight", 0.2)  # γ

        # Expected fields for authentic images
        self.expected_fields = config.get("expected_fields", [
            "Make", "Model", "DateTime Original", "Software"
        ])

        # Suspicious software signatures
        self.suspicious_software = config.get("suspicious_software", [
            "Unknown", "GIMP", "Paint.NET", "Adobe Photoshop"
        ])

        # Components
        self.extractor = None

    def load_model(self) -> None:
        """Initialize metadata extractor"""
        try:
            logger.info("initializing_metadata_extractor")

            exiftool_path = self.config.get("exiftool_path")
            self.extractor = MetadataExtractor(exiftool_path=exiftool_path)

            self._is_loaded = True
            logger.info("metadata_extractor_ready")

        except Exception as e:
            logger.warning("metadata_initialization_warning", error=str(e))
            # Don't fail, metadata is optional
            self._is_loaded = True

    def preprocess(self, input_data: Any) -> Tuple[Dict[str, Any], str]:
        """
        Extract metadata from image.

        Args:
            input_data: Image path

        Returns:
            Tuple of (extracted_metadata, image_path)
        """
        try:
            if isinstance(input_data, str):
                image_path = input_data
            else:
                raise ValueError(f"Expected image path, got {type(input_data)}")

            # Extract metadata
            metadata = self.extractor.extract(image_path)

            return metadata, image_path

        except Exception as e:
            # Return empty metadata if extraction fails
            logger.warning("metadata_extraction_failed", error=str(e))
            return {}, str(input_data)

    def predict(self, preprocessed_data: Any) -> ModuleResult:
        """
        Analyze metadata and compute authenticity score.

        Args:
            preprocessed_data: Tuple from preprocess()

        Returns:
            ModuleResult with metadata score
        """
        try:
            metadata, image_path = preprocessed_data

            # Compute component scores
            completeness_score = self.compute_completeness_score(metadata)
            validity_score = self.compute_validity_score(metadata)
            anomaly_score = self.detect_anomalies(metadata)

            # Compute final score (Equation 4 from paper)
            score = (
                self.completeness_weight * completeness_score +
                self.validity_weight * validity_score +
                self.anomaly_weight * anomaly_score
            )

            # Determine prediction
            confidence = abs(score - 0.5) * 2  # Distance from uncertain (0.5)
            prediction = "real" if score > 0.5 else "fake"

            # Collect findings
            missing_fields = self._identify_missing_fields(metadata)
            anomalies = self._identify_anomalies(metadata)

            metadata_info = {
                "completeness_score": completeness_score,
                "validity_score": validity_score,
                "anomaly_score": anomaly_score,
                "missing_fields": missing_fields,
                "anomalies": anomalies,
                "total_fields": len(metadata.get("raw", {}))
            }

            return ModuleResult(
                score=score,
                confidence=confidence,
                prediction=prediction,
                explanation={},  # Will be filled by explain()
                metadata=metadata_info,
                raw_output={"metadata": metadata}
            )

        except Exception as e:
            raise PredictionError(f"Metadata analysis failed: {e}")

    def compute_completeness_score(self, metadata: Dict[str, Any]) -> float:
        """
        Compute completeness score based on present fields.

        C = (number of present expected fields) / (total expected fields)

        Args:
            metadata: Extracted metadata

        Returns:
            Completeness score [0, 1]
        """
        if not metadata:
            return 0.0

        present_count = 0

        for field in self.expected_fields:
            # Check all categories
            found = False
            for category in ["camera", "settings", "temporal", "software"]:
                if field in metadata.get(category, {}):
                    found = True
                    break

            if found:
                present_count += 1

        score = present_count / len(self.expected_fields) if self.expected_fields else 0.0
        return score

    def compute_validity_score(self, metadata: Dict[str, Any]) -> float:
        """
        Compute validity score based on logical consistency.

        Checks:
        - Reasonable timestamp values
        - Consistent camera settings
        - Valid GPS coordinates

        Args:
            metadata: Extracted metadata

        Returns:
            Validity score [0, 1]
        """
        if not metadata:
            return 0.0

        checks = []

        # Check temporal consistency
        temporal = metadata.get("temporal", {})
        if temporal:
            checks.append(self._check_temporal_validity(temporal))

        # Check camera settings consistency
        settings = metadata.get("settings", {})
        if settings:
            checks.append(self._check_settings_validity(settings))

        # Check GPS validity
        geo = metadata.get("geolocation", {})
        if geo:
            checks.append(self._check_geo_validity(geo))

        # Average of all checks
        if not checks:
            return 0.5  # Neutral if no data to validate

        return sum(checks) / len(checks)

    def _check_temporal_validity(self, temporal: Dict[str, Any]) -> float:
        """Check if timestamps are reasonable"""
        score = 1.0

        # Check if dates are in the past (not future)
        for field in ["DateTimeOriginal", "CreateDate"]:
            if field in temporal:
                try:
                    # Parse datetime
                    date_str = str(temporal[field])
                    # Try common formats
                    for fmt in ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            dt = datetime.strptime(date_str.split('+')[0].strip(), fmt)
                            if dt > datetime.now():
                                score *= 0.5  # Future date is suspicious
                            break
                        except:
                            continue
                except:
                    pass

        return score

    def _check_settings_validity(self, settings: Dict[str, Any]) -> float:
        """Check if camera settings are reasonable"""
        score = 1.0

        # Check ISO range
        if "ISO" in settings:
            try:
                iso = int(settings["ISO"])
                if iso < 50 or iso > 102400:
                    score *= 0.7  # Unusual ISO
            except:
                pass

        # Check aperture
        if "FNumber" in settings:
            try:
                fnum = float(settings["FNumber"])
                if fnum < 0.5 or fnum > 64:
                    score *= 0.7  # Unusual aperture
            except:
                pass

        return score

    def _check_geo_validity(self, geo: Dict[str, Any]) -> float:
        """Check if GPS coordinates are valid"""
        score = 1.0

        # Check latitude range [-90, 90]
        if "GPSLatitude" in geo:
            try:
                lat = float(str(geo["GPSLatitude"]).split()[0])
                if lat < -90 or lat > 90:
                    score = 0.0
            except:
                pass

        # Check longitude range [-180, 180]
        if "GPSLongitude" in geo:
            try:
                lon = float(str(geo["GPSLongitude"]).split()[0])
                if lon < -180 or lon > 180:
                    score = 0.0
            except:
                pass

        return score

    def detect_anomalies(self, metadata: Dict[str, Any]) -> float:
        """
        Detect anomalous patterns.

        Higher score = more anomalies = more suspicious

        Args:
            metadata: Extracted metadata

        Returns:
            Anomaly score [0, 1] (inverted for final score)
        """
        anomalies = []

        # Check for missing EXIF data
        if not metadata or len(metadata.get("raw", {})) < 10:
            anomalies.append("Minimal or missing EXIF data")

        # Check for suspicious software
        software = metadata.get("software", {})
        if software:
            software_str = str(software).lower()
            for suspicious in self.suspicious_software:
                if suspicious.lower() in software_str:
                    anomalies.append(f"Suspicious software: {suspicious}")

        # Check for missing GPS (common in synthetic images)
        if not metadata.get("geolocation", {}):
            anomalies.append("Missing GPS data")

        # Anomaly ratio
        anomaly_ratio = len(anomalies) / 5.0  # Normalize by expected max anomalies

        # Return inverted score (fewer anomalies = higher authenticity)
        return 1.0 - min(anomaly_ratio, 1.0)

    def _identify_missing_fields(self, metadata: Dict[str, Any]) -> List[str]:
        """Identify which expected fields are missing"""
        missing = []

        for field in self.expected_fields:
            found = False
            for category in ["camera", "settings", "temporal", "software"]:
                if field in metadata.get(category, {}):
                    found = True
                    break

            if not found:
                missing.append(field)

        return missing

    def _identify_anomalies(self, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify specific anomalies with descriptions"""
        anomalies = []

        # Missing data anomaly
        if not metadata or len(metadata.get("raw", {})) < 10:
            anomalies.append({
                "type": "missing_data",
                "description": "Minimal or no EXIF data present (typical of synthetic images)",
                "severity": "high"
            })

        # Software anomaly
        software = metadata.get("software", {}).get("Software", "")
        if software:
            for suspicious in self.suspicious_software:
                if suspicious.lower() in software.lower():
                    anomalies.append({
                        "type": "suspicious_software",
                        "description": f"Image edited with {suspicious}",
                        "severity": "medium"
                    })

        # GPS anomaly
        if not metadata.get("geolocation", {}):
            anomalies.append({
                "type": "missing_gps",
                "description": "No GPS data (common in generated/downloaded images)",
                "severity": "low"
            })

        return anomalies

    def explain(self, input_data: Any, prediction: Any) -> Dict[str, Any]:
        """
        Generate metadata explanation.

        Args:
            input_data: Original input
            prediction: Raw output

        Returns:
            Dictionary with metadata findings
        """
        metadata = prediction.get("metadata", {})

        explanation = {
            "completeness": {
                "score": metadata.get("completeness_score", 0),
                "missing_fields": metadata.get("missing_fields", [])
            },
            "validity": {
                "score": metadata.get("validity_score", 0)
            },
            "anomalies": metadata.get("anomalies", []),
            "summary": self._generate_summary(metadata)
        }

        return explanation

    def _generate_summary(self, metadata: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        score = (
            self.completeness_weight * metadata.get("completeness_score", 0) +
            self.validity_weight * metadata.get("validity_score", 0) +
            self.anomaly_weight * metadata.get("anomaly_score", 0)
        )

        if score < 0.3:
            return "Highly suspicious metadata - likely synthetic or heavily edited image"
        elif score < 0.5:
            return "Incomplete or suspicious metadata detected"
        elif score < 0.7:
            return "Metadata appears mostly authentic with minor concerns"
        else:
            return "Metadata appears authentic and complete"
