"""Approach 1 — the classic TF-IDF + linear classifier intent engine.

Module summary
--------------
This is the "old school" baseline every NLP practitioner should know
before reaching for anything heavier. It turns each utterance into a
sparse bag-of-character-and-word n-grams weighted by TF-IDF, then fits a
plain linear classifier (logistic regression) on top. No neural network,
no GPU, no network — a few kilobytes of model that trains in milliseconds
and predicts in microseconds.

What it teaches
---------------
* Intent classification is, at heart, supervised text classification.
* Character n-grams give surprising robustness to typos and French
  morphology ("assurer", "assurance", "assurée") without any lemmatiser.
* The ``decision_function`` margins, squashed through a softmax, give a
  usable confidence and a natural abstention threshold.

Strengths: instant, tiny, fully offline, perfectly explainable.
Weakness: it matches *surface forms*; a paraphrase with no shared words
("mon pare-brise est fissuré" vs. a training phrase about "vitre cassée")
can slip past it — which is exactly the gap Approach 2 (BERT) closes.

Usage example
-------------
>>> from intent_engine.kb import KnowledgeBase
>>> from intent_engine.tfidf_engine import TfidfIntentEngine
>>> kb = KnowledgeBase.from_directory("knowledge_base")
>>> engine = TfidfIntentEngine().fit(kb)
>>> result = engine.classify("je veux assurer ma voiture")
>>> result.engine
'tfidf'

Author
------
Project maintainers.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .base import IntentEngine, IntentPrediction, IntentResult
from .kb import KnowledgeBase

# Below this softmax confidence we treat the top intent as untrustworthy and
# abstain (hand off to a human). Tuned to be forgiving on a tiny KB where a
# single strong keyword should already win; raise it in production once you
# have real traffic to calibrate against.
_CONFIDENCE_FLOOR = 0.30


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Convert raw class scores into a probability distribution.

    Parameters
    ----------
    scores : np.ndarray
        1-D array of real-valued class scores (decision-function margins).

    Returns
    -------
    np.ndarray
        Same-shape array of non-negative values summing to 1.

    Examples
    --------
    >>> import numpy as np
    >>> p = _softmax(np.array([2.0, 1.0, 0.0]))
    >>> bool(np.isclose(p.sum(), 1.0))
    True
    """
    # Subtract the max before exponentiating: mathematically a no-op, but it
    # prevents ``exp`` from overflowing on large margins (numerical safety).
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / exp.sum()


class TfidfIntentEngine(IntentEngine):
    """TF-IDF vectoriser + logistic-regression intent classifier.

    The whole model is a two-step scikit-learn :class:`~sklearn.pipeline.Pipeline`,
    so serialisation, cloning and inspection all work with the standard
    scikit-learn tooling.

    Attributes
    ----------
    name : str
        Always ``"tfidf"``.
    """

    # Engine tag surfaced in results and the comparison UI.
    name: str = "tfidf"

    def __init__(self) -> None:
        """Initialise an unfitted engine.

        The scikit-learn pipeline and the knowledge base are attached later
        by :meth:`fit`; before that, :meth:`classify` must not be called.
        """
        # The fitted pipeline (vectoriser + classifier). ``None`` until fit.
        self._pipeline: Pipeline | None = None
        # The knowledge base is retained so we can look up the scripted
        # answer and routing metadata for whichever intent wins.
        self._kb: KnowledgeBase | None = None
        # Cached, ordered class labels (intent ids) as learned by the
        # classifier, so we can map score columns back to intent ids.
        self._labels: list[str] = []

    def _build_pipeline(self) -> Pipeline:
        """Construct the (unfitted) TF-IDF + logistic-regression pipeline.

        Returns
        -------
        Pipeline
            A fresh scikit-learn pipeline ready to be ``fit``.

        Notes
        -----
        Two design choices worth the comment:

        * ``analyzer='char_wb'`` with 3–5-grams: character n-grams bounded
          by word edges make the model robust to typos and to French
          inflection without any stemming/lemmatisation step.
        * ``LogisticRegression`` over ``LinearSVC``: we want calibrated-ish
          probabilities (via ``predict_proba``/``decision_function`` +
          softmax) to drive a confidence threshold, which SVC does not give
          natively.
        """
        # Word-level unigrams+bigrams capture obvious keyword signals; the
        # character analyser below captures sub-word/typo signals. We combine
        # them by simply using char n-grams as the primary analyzer, which in
        # practice already carries the word signal for short utterances.
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            lowercase=True,
            strip_accents="unicode",
        )
        # ``C`` is deliberately high (weak regularisation): the KB is small
        # and clean, so we let the model fit the examples closely. ``max_iter``
        # is raised so the solver always converges on this tiny problem.
        classifier = LogisticRegression(C=10.0, max_iter=1000)
        # A pipeline keeps vectoriser and classifier as one fitted object.
        return Pipeline([("tfidf", vectorizer), ("clf", classifier)])

    def fit(self, kb: KnowledgeBase) -> TfidfIntentEngine:
        """Train the pipeline on the knowledge base's example utterances.

        Parameters
        ----------
        kb : KnowledgeBase
            The parsed knowledge base providing ``(texts, labels)``.

        Returns
        -------
        TfidfIntentEngine
            ``self``, fitted and ready to :meth:`classify`.

        Raises
        ------
        ValueError
            If the knowledge base has fewer than two distinct intents (a
            classifier needs at least two classes to be meaningful).
        """
        # Pull the flattened supervised dataset out of the KB.
        texts, labels = kb.training_pairs()
        # A single-class problem is degenerate for a classifier; fail early
        # with a clear message rather than producing a model that always
        # predicts the one label it ever saw.
        if len(set(labels)) < 2:
            raise ValueError(
                "TF-IDF engine needs at least two distinct intents to train."
            )
        # Build and fit the pipeline; on this data size this is milliseconds.
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(texts, labels)
        # Cache the label ordering and the KB for prediction-time lookups.
        self._labels = list(self._pipeline.named_steps["clf"].classes_)
        self._kb = kb
        return self

    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        """Classify one utterance into ranked intents.

        Parameters
        ----------
        text : str
            The user's sentence.
        top_k : int, optional
            Maximum number of ranked intents to return, by default 3.

        Returns
        -------
        IntentResult
            Ranked intents, the scripted answer for the winner, and timing.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        # Guard against use-before-train: a clear error beats an obscure
        # ``NoneType`` attribute error deep inside scikit-learn.
        if self._pipeline is None or self._kb is None:
            raise RuntimeError("TfidfIntentEngine.classify called before fit().")

        # Time only the inference so the comparison UI reflects prediction
        # cost, not the (one-off) training cost.
        started = time.perf_counter()
        # ``decision_function`` returns the signed margin per class; softmax
        # turns those margins into a comparable confidence distribution.
        margins = self._pipeline.decision_function([text])[0]
        probabilities = _softmax(np.asarray(margins, dtype=float))
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # Rank classes by probability, highest first, and keep the top-k.
        order = np.argsort(probabilities)[::-1][:top_k]
        ranked = [
            IntentPrediction(intent=self._labels[i], score=float(probabilities[i]))
            for i in order
        ]

        # Decide whether we trust the winner enough to route automatically.
        top = ranked[0]
        confident = top.score >= _CONFIDENCE_FLOOR
        # Only surface the scripted answer when confident; otherwise the UI
        # shows an abstention and would route the caller to a human.
        intent = self._kb.get(top.intent)
        response = intent.response if (confident and intent) else ""

        # Package everything, including a little diagnostic block so the
        # teaching UI can show *why* this engine decided what it did.
        return IntentResult(
            engine=self.name,
            query=text,
            ranked=ranked,
            response=response,
            slots={},
            latency_ms=elapsed_ms,
            confident=confident,
            meta={
                "backend": "sklearn TfidfVectorizer + LogisticRegression",
                "vectorizer": "char_wb 3-5grams",
                "confidence_floor": _CONFIDENCE_FLOOR,
            },
        )
