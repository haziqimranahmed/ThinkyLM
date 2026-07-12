"""
Pytest configuration — adds the project root to sys.path so that
all packages (thinkylm, data_pipeline, training, etc.) can be imported
without requiring a pip install.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
