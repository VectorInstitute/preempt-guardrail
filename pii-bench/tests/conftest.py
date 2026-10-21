"""Put ``pii-bench/`` on ``sys.path`` so ``models`` and ``sanitizers`` import."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
