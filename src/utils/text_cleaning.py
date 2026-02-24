import re


def normalize_text(text: str) -> str:
    """
    Fix common PDF extraction issues like spaced characters.
    """
    # Remove excessive spaces between single characters
    text = re.sub(r'(?<=\w)\s(?=\w)', '', text)

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()