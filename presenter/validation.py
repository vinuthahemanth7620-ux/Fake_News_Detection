"""
presenter/validation.py
=======================
Responsibility:
- Validates user input submitted from View forms or API requests.
- Checks minimum/maximum character length, non-empty text, and forbidden payloads.

TODO:
- Implement validate_news_input(text: str) -> tuple[bool, str]
"""

def validate_news_input(text: str) -> tuple[bool, str]:
    """
    Validates article text input.
    Returns (is_valid, error_message).
    """
    # TODO: Implement validation rules (e.g. check min length, non-empty)
    if not text or not text.strip():
        return False, "Article text cannot be empty."
    return True, ""
