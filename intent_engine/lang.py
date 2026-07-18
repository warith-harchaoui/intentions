"""Detect the language of a query so the LLM prompt can match it.

Module summary
--------------
The demo is bilingual (French / English). When a customer types a sentence,
we detect its language with :mod:`langdetect` and route it to the matching
LLM system prompt (see :mod:`intent_engine.llm_engine`). Detection is
deliberately conservative: it only ever returns a language we actually
support, and falls back to the configured default for empty or too-short
input where detection is unreliable.

Why langdetect
--------------
It is a small, offline, pure-Python port of Google's language-detection
library — no network, no heavy model, which fits the project's 100 %-local,
health-data-friendly constraint.

Author
------
Project maintainers.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The languages the product actually supports. Anything langdetect reports
# outside this set is coerced to the default (we have no prompt for it).
SUPPORTED: tuple[str, ...] = ("fr", "en")

# Below this many non-space characters, langdetect is a coin-flip ("ok",
# "merci", a bare plate number), so we skip it and use the default instead.
_MIN_CHARS = 8


def detect_language(text: str, default: str = "fr") -> str:
    """Return the query's language code, restricted to :data:`SUPPORTED`.

    Parameters
    ----------
    text : str
        The user's utterance.
    default : str, optional
        Language to fall back to when detection is unreliable or unsupported,
        by default ``"fr"`` (the knowledge base is French-first).

    Returns
    -------
    str
        ``"fr"`` or ``"en"`` — always a supported code.

    Examples
    --------
    >>> detect_language("j'ai eu un accident de voiture ce matin")
    'fr'
    >>> detect_language("I want to cancel my home insurance policy")
    'en'
    >>> detect_language("ok")  # too short → default
    'fr'
    """
    stripped = text.strip()
    # Too little signal to detect reliably: don't guess, use the default.
    if len(stripped.replace(" ", "")) < _MIN_CHARS:
        return default if default in SUPPORTED else "fr"

    try:
        # Imported lazily so importing this module never hard-fails if the
        # optional dependency is missing; callers still get the default.
        from langdetect import DetectorFactory, detect

        # Deterministic detection (langdetect is randomised by default): a
        # fixed seed makes the same text always resolve to the same language.
        DetectorFactory.seed = 0
        code = detect(stripped)
    except Exception as exc:  # noqa: BLE001 - detection must never crash routing
        logger.debug("Language detection failed (%s); using default %r", exc, default)
        return default if default in SUPPORTED else "fr"

    # Keep only supported languages; everything else maps to the default.
    return code if code in SUPPORTED else (default if default in SUPPORTED else "fr")
