"""
Image data collection script.

Downloads and organizes real and fake face datasets.

Usage:
    python scripts/data/collect_images.py \
        --source celeba \
        --output data/raw/images/real \
        --num_samples 30000
"""

import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import time

from src.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def download_thispersondoesnotexist(output_dir: Path, num_samples: int):
    """
    Download fake faces from thispersondoesnotexist.com

    Args:
        output_dir: Output directory
        num_samples: Number of images to download
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("downloading_fake_faces", source="thispersondoesnotexist", count=num_samples)

    url = "https://thispersondoesnotexist.com"

    for i in tqdm(range(num_samples), desc="Downloading fake faces"):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                output_path = output_dir / f"fake_{i:06d}.jpg"
                with open(output_path, 'wb') as f:
                    f.write(response.content)

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.warning(f"download_failed", index=i, error=str(e))
            continue

    logger.info("download_complete", saved=num_samples)


def download_celeba_kaggle(output_dir: Path, num_samples: int):
    """
    Download CelebA dataset using Kaggle API.

    Requires: kaggle API credentials configured
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("downloading_celeba", count=num_samples)

    try:
        import kaggle

        # Download dataset
        kaggle.api.dataset_download_files(
            'jessicali9530/celeba-dataset',
            path=str(output_dir),
            unzip=True
        )

        logger.info("celeba_downloaded")

    except ImportError:
        logger.error("kaggle_not_installed")
        print("Install kaggle: pip install kaggle")
        print("Configure credentials: https://github.com/Kaggle/kaggle-api")


def main():
    parser = argparse.ArgumentParser(description="Collect image data")
    parser.add_argument("--source", type=str, required=True,
                       choices=["celeba", "thispersondoesnotexist"],
                       help="Data source")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of samples to collect")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.source == "thispersondoesnotexist":
        download_thispersondoesnotexist(output_dir, args.num_samples)
    elif args.source == "celeba":
        download_celeba_kaggle(output_dir, args.num_samples)

    logger.info("data_collection_complete", source=args.source, output=args.output)


if __name__ == "__main__":
    main()
