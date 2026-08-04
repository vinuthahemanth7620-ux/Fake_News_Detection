"""
utils/helper.py
===============
Responsibility:
- General utility functions (file handling, string truncation, formatting helpers).

TODO:
- Implement format_confidence(score: float) -> str.
- Implement truncate_text(text: str, max_length: int) -> str.
"""

def truncate_text(text: str, max_len: int = 150) -> str:
    """Truncates text to maximum length with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + '...'
