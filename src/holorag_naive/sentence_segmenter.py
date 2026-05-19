import re
from typing import List


class SentenceSegmenter:
    def split(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", str(text or "").strip())
        if not normalized:
            return []
        parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\"'])", normalized)
        return [part.strip() for part in parts if part.strip()]
