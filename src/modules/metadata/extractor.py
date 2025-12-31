"""
EXIF metadata extraction using PyExifTool.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json

from src.utils.logging import get_logger
from src.utils.exceptions import MetadataExtractionError

logger = get_logger(__name__)


class MetadataExtractor:
    """
    EXIF metadata extraction wrapper for PyExifTool.

    Extracts comprehensive metadata from images including:
    - Camera information (make, model, lens)
    - Capture settings (ISO, aperture, shutter speed)
    - Temporal data (creation time, modification time)
    - Geolocation (GPS coordinates)
    - Software information (editing tools)
    """

    def __init__(self, exiftool_path: Optional[str] = None):
        """
        Initialize metadata extractor.

        Args:
            exiftool_path: Path to exiftool binary (auto-detect if None)
        """
        self.exiftool_path = exiftool_path
        self._exiftool = None

    def extract(self, image_path: str) -> Dict[str, Any]:
        """
        Extract all available metadata from image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary of metadata fields

        Raises:
            MetadataExtractionError: If extraction fails
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Try to use exiftool
            try:
                import exiftool
                with exiftool.ExifTool() as et:
                    metadata = et.get_metadata(str(path))
            except ImportError:
                logger.warning("exiftool_not_available_using_pillow")
                metadata = self._extract_with_pillow(path)

            # Organize metadata
            organized = self._organize_metadata(metadata)

            logger.info("metadata_extracted",
                       image=path.name,
                       fields_found=len(organized))

            return organized

        except Exception as e:
            raise MetadataExtractionError(f"Failed to extract metadata: {e}")

    def _extract_with_pillow(self, image_path: Path) -> Dict[str, Any]:
        """
        Fallback extraction using PIL (limited metadata).

        Args:
            image_path: Path to image

        Returns:
            Basic EXIF data
        """
        from PIL import Image
        from PIL.ExifTags import TAGS

        metadata = {}

        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()

                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        metadata[f"EXIF:{tag}"] = value

        except Exception as e:
            logger.warning("pillow_extraction_failed", error=str(e))

        return metadata

    def _organize_metadata(self, raw_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Organize metadata into structured format.

        Args:
            raw_metadata: Raw metadata dictionary

        Returns:
            Organized metadata
        """
        organized = {
            "camera": {},
            "settings": {},
            "temporal": {},
            "geolocation": {},
            "software": {},
            "other": {},
            "raw": raw_metadata
        }

        # Camera information
        camera_fields = {
            "Make": ["EXIF:Make", "Make"],
            "Model": ["EXIF:Model", "Model"],
            "LensModel": ["EXIF:LensModel", "LensModel"],
            "SerialNumber": ["EXIF:SerialNumber", "SerialNumber"]
        }

        for key, possible_keys in camera_fields.items():
            for pk in possible_keys:
                if pk in raw_metadata:
                    organized["camera"][key] = raw_metadata[pk]
                    break

        # Capture settings
        settings_fields = {
            "ISO": ["EXIF:ISO", "ISO", "ISOSpeedRatings"],
            "FNumber": ["EXIF:FNumber", "FNumber"],
            "ExposureTime": ["EXIF:ExposureTime", "ExposureTime"],
            "FocalLength": ["EXIF:FocalLength", "FocalLength"],
            "Flash": ["EXIF:Flash", "Flash"]
        }

        for key, possible_keys in settings_fields.items():
            for pk in possible_keys:
                if pk in raw_metadata:
                    organized["settings"][key] = raw_metadata[pk]
                    break

        # Temporal data
        temporal_fields = {
            "DateTimeOriginal": ["EXIF:DateTimeOriginal", "DateTimeOriginal"],
            "CreateDate": ["EXIF:CreateDate", "CreateDate"],
            "ModifyDate": ["EXIF:ModifyDate", "ModifyDate", "File:FileModifyDate"]
        }

        for key, possible_keys in temporal_fields.items():
            for pk in possible_keys:
                if pk in raw_metadata:
                    organized["temporal"][key] = raw_metadata[pk]
                    break

        # Geolocation
        geo_fields = {
            "GPSLatitude": ["GPS:GPSLatitude", "GPSInfo:GPSLatitude"],
            "GPSLongitude": ["GPS:GPSLongitude", "GPSInfo:GPSLongitude"],
            "GPSAltitude": ["GPS:GPSAltitude", "GPSInfo:GPSAltitude"]
        }

        for key, possible_keys in geo_fields.items():
            for pk in possible_keys:
                if pk in raw_metadata:
                    organized["geolocation"][key] = raw_metadata[pk]
                    break

        # Software
        software_fields = {
            "Software": ["EXIF:Software", "Software"],
            "ProcessingSoftware": ["EXIF:ProcessingSoftware", "ProcessingSoftware"],
            "Creator": ["XMP:Creator", "Creator"]
        }

        for key, possible_keys in software_fields.items():
            for pk in possible_keys:
                if pk in raw_metadata:
                    organized["software"][key] = raw_metadata[pk]
                    break

        return organized

    def get_field(self, metadata: Dict[str, Any], field_name: str) -> Optional[Any]:
        """
        Get specific field from metadata.

        Args:
            metadata: Metadata dictionary
            field_name: Field name to retrieve

        Returns:
            Field value or None if not found
        """
        for category in ["camera", "settings", "temporal", "geolocation", "software"]:
            if field_name in metadata.get(category, {}):
                return metadata[category][field_name]

        return None
