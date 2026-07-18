"""Approach 1 — the classic TF-IDF + Random Forest intent engine.

Module summary
--------------
The "old school" baseline every NLP practitioner should know before
reaching for anything heavier. It turns each utterance into a sparse
bag-of-character-and-word n-grams weighted by TF-IDF, then fits a good old
**Random Forest** on top — the workhorse ensemble classifier of the
pre-deep-learning era. No neural network, no GPU, no network: a model that
trains in a second and predicts in milliseconds.

What it teaches
---------------
* Intent classification is, at heart, supervised text classification.
* Character n-grams give surprising robustness to typos and French
  morphology ("assurer", "assurance", "assurée") without any lemmatiser.
* A Random Forest gives calibrated-ish class probabilities out of the box
  (``predict_proba``), which drive the confidence threshold and abstention.

Strengths: fast, tiny, fully offline, robust, no tuning.
Weakness: it matches *surface forms*; a paraphrase with no shared words
("mon pare-brise est fissuré" vs. a training phrase about "vitre cassée")
can slip past it — the gap the embedding-based approaches close. Under a
proper train/test split this shows up as a much lower cross-validated
accuracy than the held-out-with-vocabulary-overlap number suggests.

Usage example
-------------
>>> from intent_engine.kb import KnowledgeBase
>>> from intent_engine.tfidf_engine import TfidfIntentEngine
>>> kb = KnowledgeBase.from_directory("knowledge_base")
>>> engine = TfidfIntentEngine().fit(kb)
>>> engine.classify("je veux assurer ma voiture").engine
'tfidf'

Author
------
Project maintainers.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .base import IntentEngine, IntentPrediction, IntentResult
from .kb import KnowledgeBase

# Below this class-probability we treat the top intent as untrustworthy and
# abstain (hand off to a human). A Random Forest's ``predict_proba`` is the
# fraction of trees voting for a class, so 0.30 means "≥30 % of the forest
# agreed" — a sensible floor on this many classes.
_CONFIDENCE_FLOOR = 0.30

# Number of trees. 300 is plenty for a few hundred short texts and keeps the
# forest fast to train; more trees would only marginally smooth the votes.
_N_ESTIMATORS = 300


class TfidfIntentEngine(IntentEngine):
    """TF-IDF vectoriser + Random Forest intent classifier.

    The whole model is a two-step scikit-learn
    :class:`~sklearn.pipeline.Pipeline`, so serialisation, cloning and
    inspection all work with the standard scikit-learn tooling.

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
        # Cached, ordered class labels (intent ids) as learned by the forest,
        # so we can map probability columns back to intent ids.
        self._labels: list[str] = []

    def _build_pipeline(self) -> Pipeline:
        """Construct the (unfitted) TF-IDF + Random Forest pipeline.

        Returns
        -------
        Pipeline
            A fresh scikit-learn pipeline ready to be ``fit``.

        Notes
        -----
        * ``analyzer='char_wb'`` with 3–5-grams: character n-grams bounded
          by word edges make the model robust to typos and to French
          inflection without any stemming/lemmatisation step.
        * ``RandomForestClassifier``: the classic, tuning-free ensemble.
          It gives ``predict_proba`` directly, so no softmax-over-margins
          trick is needed to obtain a confidence.
        """
        # Character n-grams carry both keyword and sub-word/typo signal for
        # short utterances; accents are folded so "réglé"/"regle" collide.
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            lowercase=True,
            strip_accents="unicode",
        )
        # ``class_weight='balanced'`` compensates for the small per-intent
        # count differences; the fixed ``random_state`` keeps runs reproducible.
        classifier = RandomForestClassifier(
            n_estimators=_N_ESTIMATORS,
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
        )
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
        # A single-class problem is degenerate; fail early with a clear message.
        if len(set(labels)) < 2:
            raise ValueError(
                "TF-IDF engine needs at least two distinct intents to train."
            )
        # Build and fit the pipeline; on this data size this is ~1 second.
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
        # ``predict_proba`` returns the fraction of trees voting per class —
        # already a probability distribution, no softmax needed.
        probabilities = self._pipeline.predict_proba([text])[0]
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
                "backend": "sklearn TfidfVectorizer + RandomForest",
                "vectorizer": "char_wb 3-5grams",
                "n_estimators": _N_ESTIMATORS,
                "confidence_floor": _CONFIDENCE_FLOOR,
            },
        )
