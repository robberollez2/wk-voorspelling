#!/usr/bin/env python
"""Root-level entry point for the interactive predictor.

Thin wrapper so the project can be driven with ``python predict.py`` exactly as
described in the README. The real implementation lives in :mod:`src.predict`.
"""

from __future__ import annotations

from src.predict import main

if __name__ == "__main__":
    main()
