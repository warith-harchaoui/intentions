"""Déraison Assurances — a teaching intent engine, five ways.

Module summary
--------------
Top-level package for a small but complete demonstration of how a phone /
chat insurance assistant detects a caller's *intention* from a Markdown
knowledge base, using five progressively heavier approaches:

* ``tfidf`` — scikit-learn TF-IDF + logistic regression (the classic).
* ``fasttext_custom`` — fastText subword embeddings learned on our examples.
* ``fasttext_pretrained`` — pretrained fastText cc.fr.300 French vectors.
* ``bert``  — BERT-family sentence embeddings + a learned classifier.
* ``llm``   — a local LLM (Gemma via Ollama) with a strict JSON contract.

This ``__init__`` re-exports the handful of names most callers need so
they can write ``from intent_engine import IntentRouter`` without knowing
the internal module layout.

Usage example
-------------
>>> from intent_engine import IntentRouter, KnowledgeBase
>>> callable(IntentRouter.from_directory)
True

Author
------
Project maintainers.
"""

from __future__ import annotations

# Re-export the public surface. Engines are exposed for advanced callers who
# want to drive one directly; most code should go through ``IntentRouter``.
from .base import IntentEngine, IntentPrediction, IntentResult
from .bert_engine import BertIntentEngine
from .config import Settings, get_settings
from .kb import Intent, KnowledgeBase
from .llm_engine import LlmIntentEngine
from .router import Execution, IntentRouter
from .tfidf_engine import TfidfIntentEngine

# Explicit ``__all__`` documents the supported public API and keeps
# ``from intent_engine import *`` tidy.
__all__ = [
    "BertIntentEngine",
    "Execution",
    "Intent",
    "IntentEngine",
    "IntentPrediction",
    "IntentResult",
    "IntentRouter",
    "KnowledgeBase",
    "LlmIntentEngine",
    "Settings",
    "TfidfIntentEngine",
    "get_settings",
]

# Single source of truth for the package version, read by packaging tools.
__version__ = "0.1.0"
