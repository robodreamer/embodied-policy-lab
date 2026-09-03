#!/usr/bin/env python3
"""Repository entry point for the headless WAM benchmark."""

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from showcase.wam_benchmark import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
