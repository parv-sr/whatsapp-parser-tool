import re
import hashlib
import unicodedata
from simhash import Simhash


class PreLLMDedupe:
    """
    Pre-LLM dedupe engine reducing token cost.
    Works per-file: new instance per parse.
    """
    def __init__(self):
        self.exact_hashes = set()
        self.simhashes = []

    # -------------------------------
    # Clean message to stable form
    # -------------------------------
    def _normalize(self, s: str) -> str:
        if not s:
            return ""
        s = unicodedata.normalize("NFKC", s)
        s = s.lower()

        # remove emojis & symbols
        s = re.sub(r"[^\w\s\-.,/]", " ", s)

        # remove phone numbers
        s = re.sub(r"\+?\d[\d\s\-]{6,}", " ", s)

        # collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()

        return s

    # -------------------------------
    # Exact hash
    # -------------------------------
    def _exact_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # -------------------------------
    # Fuzzy duplicate check
    # -------------------------------
    def _is_similar(self, sh: Simhash) -> bool:
        for existing in self.simhashes:
            if sh.distance(existing) <= 3:  # near-duplicate
                return True
        return False

    # -------------------------------
    # External interface
    # -------------------------------
    def should_keep(self, text: str) -> bool:
        """
        Returns True if this message should be processed by LLM.
        False → drop early.
        """
        cleaned = self._normalize(text)

        if not cleaned or len(cleaned) < 15:
            return False

        h_exact = self._exact_hash(cleaned)
        if h_exact in self.exact_hashes:
            return False
        
        tokens = re.findall(r"[a-zA-Z0-9]+", cleaned.lower())

        h_fuzzy = Simhash(
        tokens,
        hashfunc=lambda x: int(
            hashlib.md5(
                (x.decode("utf-8") if isinstance(x, bytes) else x).encode("utf-8")
            ).hexdigest(),
            16
        )
    )

        if self._is_similar(h_fuzzy):
            return False

        # New unique message
        self.exact_hashes.add(h_exact)
        self.simhashes.append(h_fuzzy)
        return True
