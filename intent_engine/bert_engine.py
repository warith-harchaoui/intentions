"""Approach 4 — BERT sentence embeddings + a PyTorch MLP classifier.

Module summary
--------------
The strongest of the non-generative approaches: contextual sentence
embeddings feeding a small neural net. Each utterance is projected into a
dense semantic vector by a BERT-family sentence encoder (SBERT via
sentence-transformers, or a BERT embedding model served by Ollama — see
:mod:`intent_engine.embeddings`), then a compact **PyTorch MLP** (see
:mod:`intent_engine.mlp`) is trained on those vectors.

Why this beats TF-IDF
---------------------
The embedding places paraphrases *near each other in vector space* even
when they share no words. "Mon pare-brise est fissuré" and "j'ai une vitre
cassée" land in the same neighbourhood, so the classifier generalises to
phrasings it never saw during training — the classic weakness of bag-of-
words. The cost is a heavier model (hundreds of MB) and slower inference
(a forward pass through a transformer instead of a sparse dot product).

Contextual embeddings + a neural head is where the progression peaks
before the LLM: the representation is semantic (unlike TF-IDF) and richer
than static fastText vectors, and the MLP can carve non-linear boundaries a
logistic regression cannot. That is the pedagogical payoff of the last
non-generative step.

Usage example
-------------
>>> from intent_engine.kb import KnowledgeBase
>>> from intent_engine.bert_engine import BertIntentEngine
>>> kb = KnowledgeBase.from_directory("knowledge_base")   # doctest: +SKIP
>>> engine = BertIntentEngine().fit(kb)                    # doctest: +SKIP
>>> engine.classify("ma vitre est cassée").engine          # doctest: +SKIP
'bert'

Author
------
Project maintainers.
"""

from __future__ import annotations

import time

import numpy as np

from .base import IntentEngine, IntentPrediction, IntentResult
from .embeddings import Embedder, build_embedder
from .kb import KnowledgeBase
from .mlp import TorchMLPClassifier

# The MLP head is expressive but, like most neural nets, poorly calibrated:
# it stays fairly confident even on out-of-scope input. We therefore set a
# higher abstention floor than the lexical engines so off-topic queries are
# rejected instead of confidently mis-routed. Empirically (see the
# calibration analysis in the eval), 0.60 keeps ~92 % of in-scope hits while
# abstaining on ~73 % of out-of-scope inputs — a sensible trade-off.
_CONFIDENCE_FLOOR = 0.60


class BertIntentEngine(IntentEngine):
    """Sentence-embedding representation + PyTorch MLP classifier.

    Parameters
    ----------
    embedder : Embedder | None, optional
        The embedding backend to use. When ``None`` (default), one is built
        from configuration via :func:`~intent_engine.embeddings.build_embedder`
        at :meth:`fit` time, so constructing the engine stays cheap and
        import-safe.

    Attributes
    ----------
    name : str
        Always ``"bert"``.
    """

    # Engine tag surfaced in results and the comparison UI.
    name: str = "bert"

    def __init__(self, embedder: Embedder | None = None) -> None:
        """Initialise an unfitted engine, deferring embedder construction."""
        # Injected embedder (handy for tests with a fake) or ``None`` to build
        # lazily from settings in ``fit`` — avoids loading a heavy model just
        # to import this class.
        self._embedder: Embedder | None = embedder
        # The MLP trained on top of the embeddings. ``None`` until fit.
        self._classifier: TorchMLPClassifier | None = None
        # Retained for scripted-answer/routing lookups at predict time.
        self._kb: KnowledgeBase | None = None
        # Ordered class labels learned by the classifier.
        self._labels: list[str] = []

    def fit(self, kb: KnowledgeBase) -> BertIntentEngine:
        """Embed the KB examples and train the classifier on the vectors.

        Parameters
        ----------
        kb : KnowledgeBase
            The parsed knowledge base providing ``(texts, labels)``.

        Returns
        -------
        BertIntentEngine
            ``self``, fitted and ready to :meth:`classify`.

        Raises
        ------
        ValueError
            If the knowledge base has fewer than two distinct intents.
        """
        texts, labels = kb.training_pairs()
        # Same guard as the TF-IDF engine: a classifier needs ≥2 classes.
        if len(set(labels)) < 2:
            raise ValueError(
                "BERT engine needs at least two distinct intents to train."
            )
        # Build the embedder now if one was not injected. Doing it here (not in
        # __init__) means the heavy model loads only when we actually train.
        if self._embedder is None:
            self._embedder = build_embedder()
        # Turn every example utterance into a dense vector — this is the whole
        # "BERT representation" step. On a small KB this is a few seconds.
        features = self._embedder.encode(texts)
        # A small PyTorch MLP head on top of good embeddings: LayerNorm →
        # Linear → ReLU → Dropout → Linear, trained with Adam + cross-entropy.
        # More expressive than a linear classifier while still tiny/fast.
        self._classifier = TorchMLPClassifier()
        self._classifier.fit(features, np.asarray(labels))
        # Cache labels and KB for prediction-time mapping and answer lookup.
        self._labels = list(self._classifier.classes_)
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
        # Guard against use-before-train for a clear error message: all three
        # collaborators (classifier, KB, embedder) must be present.
        if self._classifier is None or self._kb is None or self._embedder is None:
            raise RuntimeError("BertIntentEngine.classify called before fit().")

        # Time the full inference: embedding + classifier. The embedding pass
        # dominates and is exactly the cost we want the comparison UI to show.
        started = time.perf_counter()
        vector = self._embedder.encode([text])
        # ``predict_proba`` gives calibrated class probabilities directly —
        # no softmax-over-margins needed as with the linear-SVM style path.
        probabilities = self._classifier.predict_proba(vector)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # Rank by probability, highest first, keep the top-k.
        order = np.argsort(probabilities)[::-1][:top_k]
        ranked = [
            IntentPrediction(intent=self._labels[i], score=float(probabilities[i]))
            for i in order
        ]

        # Trust decision + scripted answer lookup, as in the TF-IDF engine.
        top = ranked[0]
        confident = top.score >= _CONFIDENCE_FLOOR
        intent = self._kb.get(top.intent)
        response = intent.response if (confident and intent) else ""

        # Report which embedding backend actually ran, so the teaching UI can
        # show whether SBERT or the Ollama fallback produced this result.
        return IntentResult(
            engine=self.name,
            query=text,
            ranked=ranked,
            response=response,
            slots={},
            latency_ms=elapsed_ms,
            confident=confident,
            meta={
                "backend": self._embedder.name,
                "classifier": "PyTorch MLP (LayerNorm-Linear-ReLU-Dropout-Linear)",
                "confidence_floor": _CONFIDENCE_FLOOR,
            },
        )
