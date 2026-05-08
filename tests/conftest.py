"""Test fixtures shared across the suite."""

import os
import sys
from pathlib import Path

# Make `src.*` and `config.*` importable when pytest is run from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force development env so importing `config.production` (which requires
# CORS_ORIGINS) never trips during collection.
os.environ.setdefault("ENVIRONMENT", "development")
