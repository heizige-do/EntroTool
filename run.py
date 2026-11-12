"""Run EntroTool from the source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from entrotool.method import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
