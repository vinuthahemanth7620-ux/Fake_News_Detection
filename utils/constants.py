"""
utils/constants.py
==================
Responsibility:
- Holds project-wide constants (label names, error messages, default confidence thresholds).
"""

LABEL_REAL = "REAL"
LABEL_FAKE = "FAKE"

MIN_INPUT_LENGTH = 10
MAX_INPUT_LENGTH = 10000

ERR_EMPTY_INPUT = "Article text cannot be empty."
ERR_SHORT_INPUT = f"Article text must be at least {MIN_INPUT_LENGTH} characters."
