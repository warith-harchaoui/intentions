"""Shared contracts for every intent engine.

Module summary
--------------
Defines the small, engine-agnostic data types and the abstract base class
that the TF-IDF, BERT and LLM engines all implement. Keeping the contract
here means the router, the API and the front end can treat the three very
different engines through one identical interface: give me a French
sentence, hand me back a ranked list of intents plus a scripted answer.

This is deliberately the only place that knows the *shape* of a
prediction. The engines know *how* to predict; this module fixes *what* a
prediction is.

Usage example
-------------
>>> from intent_engine.base import IntentPrediction, IntentResult
>>> pred = IntentPrediction(intent="declarer_sinistre_auto", score=0.91)
>>> round(pred.score, 2)
0.91

Author
------
Project maintainers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Imported only for type hints on ``fit``/construction; the KB module owns
# the parsing. Placed here so every engine speaks the same KB vocabulary.
from .kb import KnowledgeBase


@dataclass(frozen=True)
class IntentPrediction:
    """One (intent, confidence) pair produced by an engine.

    Parameters
    ----------
    intent : str
        The intent identifier, matching an ``# h1`` slug in the knowledge
        base (e.g. ``"declarer_sinistre_auto"``).
    score : float
        Confidence in ``[0, 1]``. Semantics differ per engine (calibrated
        probability for BERT, decision-function-derived for TF-IDF, model
        self-report for the LLM) but the *ordering* is always comparable
        within a single engine's output.

    Examples
    --------
    >>> IntentPrediction(intent="resilier_contrat", score=0.5).intent
    'resilier_contrat'
    """

    # The intent slug this prediction points at.
    intent: str
    # Confidence, clamped by convention to the unit interval by callers.
    score: float


@dataclass
class IntentResult:
    """The full outcome of classifying one user utterance.

    This is what every engine returns and what the API serializes to the
    front end. It bundles the ranked predictions, the scripted answer the
    chatbot should read back, and any structured slots the engine managed
    to extract (only the LLM engine fills ``slots`` today).

    Parameters
    ----------
    engine : str
        Name of the engine that produced this result (``"tfidf"``,
        ``"bert"`` or ``"llm"``).
    query : str
        The raw user utterance that was classified.
    ranked : list[IntentPrediction]
        Predictions sorted by descending score. May be empty if the engine
        abstained (e.g. below its confidence floor, or Ollama unreachable).
    response : str
        The knowledge-base answer attached to the top intent, ready to be
        displayed or spoken. Empty when the engine abstained.
    slots : dict[str, Any]
        Structured entities pulled from the utterance (policy number,
        urgency, ...). Empty for the non-LLM engines.
    latency_ms : float
        Wall-clock time the engine took, in milliseconds — the whole point
        of the demo is to compare these across engines.
    confident : bool
        Whether the engine believes the top prediction is trustworthy
        enough to act on (vs. hand off to a human).
    meta : dict[str, Any]
        Free-form, engine-specific diagnostics (raw model output, chosen
        embedding backend, ...) surfaced for teaching/debugging.

    Examples
    --------
    >>> r = IntentResult(engine="tfidf", query="bonjour")
    >>> r.top() is None
    True
    """

    # Which engine produced this — echoed back so the UI can label it.
    engine: str
    # The exact user text that was classified.
    query: str
    # Ranked predictions, best first. Empty == abstention.
    ranked: list[IntentPrediction] = field(default_factory=list)
    # Scripted answer for the winning intent (from the KB).
    response: str = ""
    # Extracted structured entities (LLM engine only, otherwise empty).
    slots: dict[str, Any] = field(default_factory=dict)
    # How long the engine took, for the head-to-head comparison.
    latency_ms: float = 0.0
    # Whether we trust the top intent enough to route automatically.
    confident: bool = False
    # Engine-specific diagnostics for the teaching UI.
    meta: dict[str, Any] = field(default_factory=dict)

    def top(self) -> IntentPrediction | None:
        """Return the highest-scoring prediction, or ``None`` if abstained.

        Returns
        -------
        IntentPrediction | None
            The first element of :attr:`ranked`, or ``None`` when the
            engine produced no predictions.

        Examples
        --------
        >>> r = IntentResult(engine="llm", query="x",
        ...     ranked=[IntentPrediction("a", 0.9)])
        >>> r.top().intent
        'a'
        """
        # ``ranked`` is kept sorted by the engines, so the first item is the
        # winner. Guard the empty case so callers can use a simple ``if``.
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary for the API layer.

        Returns
        -------
        dict[str, Any]
            A plain ``dict`` with the dataclasses flattened to primitives,
            safe to hand to ``json``/``orjson``.

        Examples
        --------
        >>> r = IntentResult(engine="tfidf", query="x")
        >>> r.to_dict()["engine"]
        'tfidf'
        """
        # Flatten the ranked predictions into primitive dicts so the whole
        # payload is trivially serializable without a custom JSON encoder.
        return {
            "engine": self.engine,
            "query": self.query,
            "ranked": [{"intent": p.intent, "score": p.score} for p in self.ranked],
            "response": self.response,
            "slots": self.slots,
            "latency_ms": self.latency_ms,
            "confident": self.confident,
            "meta": self.meta,
        }


class IntentEngine(ABC):
    """Abstract base every concrete intent engine implements.

    The contract is intentionally tiny: an engine is fitted once against a
    :class:`~intent_engine.kb.KnowledgeBase`, then answers ``classify``
    calls. Trainable engines (TF-IDF, BERT) learn from the KB examples;
    zero-shot engines (LLM) just remember the intent catalogue. Either way
    the caller sees the same two methods.

    Attributes
    ----------
    name : str
        Short engine identifier used in results and the UI. Subclasses set
        this as a class attribute.
    """

    # Subclasses override this with "tfidf" / "bert" / "llm". Declared here
    # so type checkers know every engine exposes a ``name``.
    name: str = "base"

    @abstractmethod
    def fit(self, kb: KnowledgeBase) -> IntentEngine:
        """Prepare the engine from the knowledge base.

        For trainable engines this trains the classifier on the KB's
        example utterances; for the LLM engine it simply captures the
        intent catalogue used to build the prompt.

        Parameters
        ----------
        kb : KnowledgeBase
            The parsed knowledge base (intents, examples, answers).

        Returns
        -------
        IntentEngine
            ``self``, to allow ``engine = Cls().fit(kb)`` chaining.
        """
        raise NotImplementedError

    @abstractmethod
    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        """Classify one user utterance into ranked intents.

        Parameters
        ----------
        text : str
            The user's sentence, in natural language (French by default).
        top_k : int, optional
            Maximum number of ranked intents to return, by default 3.

        Returns
        -------
        IntentResult
            Ranked intents, the scripted answer for the top one, timing and
            engine-specific diagnostics.
        """
        raise NotImplementedError
