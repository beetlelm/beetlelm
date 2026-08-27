# beetlelm/src/bilingual/extensions/hooks.py

from typing import Optional, Dict, Any


class CurriculumPolicy:
    """
    Optional override for curriculum sampling.
    """

    def sample_next(self, default_source: str, pipeline_state) -> str:
        return default_source

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state: Dict):
        pass


# ------------------------------------------------------------

class TokenizerSelector:
    """
    Allows language-specific tokenizers.
    """

    def get_tokenizer(self, lang: str):
        raise NotImplementedError


# ------------------------------------------------------------

class TextPreprocessor:
    """
    Optional preprocessing before tokenization.
    """

    def process(self, text: str, lang: Optional[str] = None) -> str:
        return text


# ------------------------------------------------------------

class MetricsTracker:
    """
    Tracks performance metrics per language.
    Used by adaptive curriculum.
    """

    def get_language_scores(self) -> Dict[str, float]:
        raise NotImplementedError
