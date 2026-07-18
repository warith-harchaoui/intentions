"""Approaches 2 & 3 — fastText, learned-on-our-data and pretrained.

Module summary
--------------
Two engines that sit, pedagogically, between the sparse bag-of-words
(TF-IDF, Approach 1) and the contextual transformer (BERT, Approach 4):

* :class:`FastTextSupervisedEngine` — fastText's **own** supervised
  classifier (``fasttext.train_supervised``), *learned on our example
  utterances*. It represents words as bags of character n-grams and learns
  their embeddings jointly with a linear softmax classifier — the classic
  fastText recipe (Joulin et al., 2016). No pretrained vectors: everything
  is learned from the KB, so it shows what a subword model can squeeze out
  of a few hundred in-domain sentences.

* :class:`FastTextPretrainedEngine` — the **pretrained** French vectors
  ``cc.fr.300`` (Grave et al., 2018), trained on Common Crawl + Wikipedia.
  Each utterance is embedded by averaging its word vectors
  (``get_sentence_vector``) and a classic logistic regression is fitted on
  top. This is "transfer learning, static-embedding style": the model
  already knows that *voiture* and *véhicule* are close, so it generalises
  to paraphrases the supervised model never saw.

The progression TF-IDF → fastText-supervised → fastText-pretrained → BERT
→ LLM is the whole point: each step trades more prior knowledge / compute
for better semantic generalisation.

Usage example
-------------
>>> from intent_engine.kb import KnowledgeBase
>>> from intent_engine.fasttext_engine import FastTextSupervisedEngine
>>> kb = KnowledgeBase.from_directory("knowledge_base")   # doctest: +SKIP
>>> FastTextSupervisedEngine().fit(kb).classify("assurer ma voiture").engine
'fasttext_custom'

Author
------
Project maintainers.
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import IntentEngine, IntentPrediction, IntentResult
from .config import get_settings
from .kb import KnowledgeBase

logger = logging.getLogger(__name__)

# fastText's label prefix convention in the supervised training format.
_LABEL_PREFIX = "__label__"

# Confidence floors, mirroring the other engines so the comparison is fair.
_CONFIDENCE_FLOOR_SUP = 0.30
_CONFIDENCE_FLOOR_PRE = 0.35


def _normalise(text: str) -> str:
    """Lower-case and flatten a sentence to one clean line for fastText.

    fastText reads one example per line, so any embedded newline would split
    an utterance in two. We also lower-case and squeeze whitespace so the
    subword model sees consistent tokens.

    Parameters
    ----------
    text : str
        Raw utterance.

    Returns
    -------
    str
        A single lower-cased line.

    Examples
    --------
    >>> _normalise("  J'ai   eu   un accident ")
    "j'ai eu un accident"
    """
    # Collapse all whitespace (including newlines/tabs) to single spaces.
    return re.sub(r"\s+", " ", text.strip().lower())


class FastTextSupervisedEngine(IntentEngine):
    """fastText's native supervised classifier, trained on the KB examples.

    Attributes
    ----------
    name : str
        Always ``"fasttext_custom"``.
    """

    # Engine tag surfaced in results and the comparison UI.
    name: str = "fasttext_custom"

    def __init__(self) -> None:
        """Initialise an unfitted engine."""
        # The trained fastText model handle (a ``_FastText`` object) or None.
        self._model = None
        # KB retained for scripted-answer/routing lookups at predict time.
        self._kb: KnowledgeBase | None = None

    def fit(self, kb: KnowledgeBase) -> FastTextSupervisedEngine:
        """Train a fastText supervised model on the KB's example utterances.

        Parameters
        ----------
        kb : KnowledgeBase
            The parsed knowledge base providing ``(texts, labels)``.

        Returns
        -------
        FastTextSupervisedEngine
            ``self``, fitted and ready to :meth:`classify`.

        Raises
        ------
        ValueError
            If the knowledge base has fewer than two distinct intents.
        """
        # Import lazily so the package imports without fastText installed.
        import fasttext

        texts, labels = kb.training_pairs()
        if len(set(labels)) < 2:
            raise ValueError(
                "fastText engine needs at least two distinct intents to train."
            )
        # fastText trains from a file in ``__label__<id> <text>`` format; write
        # a temporary one, train, then discard it.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as handle:
            for text, label in zip(texts, labels):
                handle.write(f"{_LABEL_PREFIX}{label} {_normalise(text)}\n")
            train_path = handle.name
        # Classic fastText hyper-parameters for small, in-domain text:
        # word bigrams capture short collocations, character n-grams (minn/maxn)
        # give robustness to morphology/typos, softmax loss for a clean
        # probability, and enough epochs to converge on a few hundred lines.
        self._model = fasttext.train_supervised(
            input=train_path,
            epoch=40,
            lr=0.5,
            wordNgrams=2,
            dim=100,
            minn=3,
            maxn=5,
            loss="softmax",
            verbose=0,
        )
        # Clean up the temporary training file.
        Path(train_path).unlink(missing_ok=True)
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
        if self._model is None or self._kb is None:
            raise RuntimeError("FastTextSupervisedEngine.classify called before fit().")
        started = time.perf_counter()
        # Call the low-level C++ predictor directly instead of the Python
        # wrapper's ``predict``: the wrapper does ``np.array(probs, copy=False)``
        # which raises under NumPy 2.x. ``f.predict`` returns a list of
        # ``(probability, label)`` tuples, already sorted best-first.
        predictions = self._model.f.predict(_normalise(text), top_k, 0.0, "strict")
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # Strip the ``__label__`` prefix to recover the intent ids.
        ranked = [
            IntentPrediction(intent=label[len(_LABEL_PREFIX) :], score=float(prob))
            for prob, label in predictions
        ]
        top = ranked[0]
        confident = top.score >= _CONFIDENCE_FLOOR_SUP
        intent = self._kb.get(top.intent)
        response = intent.response if (confident and intent) else ""
        return IntentResult(
            engine=self.name,
            query=text,
            ranked=ranked,
            response=response,
            slots={},
            latency_ms=elapsed_ms,
            confident=confident,
            meta={
                "backend": "fastText train_supervised (appris sur nos exemples)",
                "confidence_floor": _CONFIDENCE_FLOOR_SUP,
            },
        )


class FastTextPretrainedEngine(IntentEngine):
    """Pretrained French fastText vectors (cc.fr.300) + logistic regression.

    Parameters
    ----------
    model_path : str | Path | None, optional
        Path to the ``cc.fr.300.bin`` model; defaults to the configured
        location. The engine is only usable once that file is present.

    Attributes
    ----------
    name : str
        Always ``"fasttext_pretrained"``.
    """

    # Engine tag surfaced in results and the comparison UI.
    name: str = "fasttext_pretrained"

    def __init__(self, model_path: str | Path | None = None) -> None:
        """Store the model path and prepare empty state."""
        settings = get_settings()
        # Where the big pretrained model lives on disk.
        self._model_path = Path(model_path or settings.fasttext_model_path)
        # The loaded fastText vector model (shared, read-only after load).
        self._vectors = None
        # The classic classifier fitted on the sentence vectors.
        self._classifier: LogisticRegression | None = None
        self._kb: KnowledgeBase | None = None
        self._labels: list[str] = []

    @classmethod
    def is_model_available(cls, model_path: str | Path | None = None) -> bool:
        """Return whether the pretrained model file exists on disk.

        Parameters
        ----------
        model_path : str | Path | None, optional
            Override the configured path.

        Returns
        -------
        bool
            ``True`` if the ``cc.fr.300.bin`` file is present.
        """
        # Used by the router to hide this engine until the model is downloaded.
        path = Path(model_path or get_settings().fasttext_model_path)
        return path.is_file()

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed sentences by averaging their pretrained word vectors.

        Parameters
        ----------
        texts : list[str]
            Input sentences.

        Returns
        -------
        np.ndarray
            An ``(n, 300)`` array of L2-normalised sentence vectors.
        """
        # ``get_sentence_vector`` already averages the word vectors; we just
        # normalise so the downstream logistic regression sees cosine geometry.
        rows = [self._vectors.get_sentence_vector(_normalise(t)) for t in texts]
        matrix = np.asarray(rows, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def fit(self, kb: KnowledgeBase) -> FastTextPretrainedEngine:
        """Load the pretrained vectors and fit a classifier on the KB.

        Parameters
        ----------
        kb : KnowledgeBase
            The parsed knowledge base.

        Returns
        -------
        FastTextPretrainedEngine
            ``self``, fitted and ready to :meth:`classify`.

        Raises
        ------
        FileNotFoundError
            If the pretrained model file is not present.
        ValueError
            If the KB has fewer than two distinct intents.
        """
        # Fail clearly if the (large) model has not been downloaded yet.
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"Modèle fastText pré-entraîné introuvable : {self._model_path}. "
                "Lancez scripts/download_fasttext.py."
            )
        import fasttext

        texts, labels = kb.training_pairs()
        if len(set(labels)) < 2:
            raise ValueError(
                "fastText engine needs at least two distinct intents to train."
            )
        # Load the pretrained vectors once (a few seconds + a few GB of RAM).
        logger.info("Chargement du modèle fastText FR (%s) ...", self._model_path)
        self._vectors = fasttext.load_model(str(self._model_path))
        # Embed the training utterances, then fit a classic logistic regression.
        features = self._embed(texts)
        self._classifier = LogisticRegression(C=10.0, max_iter=1000)
        self._classifier.fit(features, labels)
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
        if self._classifier is None or self._kb is None or self._vectors is None:
            raise RuntimeError("FastTextPretrainedEngine.classify called before fit().")
        started = time.perf_counter()
        # Embed the query with the pretrained vectors, then predict_proba.
        probabilities = self._classifier.predict_proba(self._embed([text]))[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        order = np.argsort(probabilities)[::-1][:top_k]
        ranked = [
            IntentPrediction(intent=self._labels[i], score=float(probabilities[i]))
            for i in order
        ]
        top = ranked[0]
        confident = top.score >= _CONFIDENCE_FLOOR_PRE
        intent = self._kb.get(top.intent)
        response = intent.response if (confident and intent) else ""
        return IntentResult(
            engine=self.name,
            query=text,
            ranked=ranked,
            response=response,
            slots={},
            latency_ms=elapsed_ms,
            confident=confident,
            meta={
                "backend": "fastText cc.fr.300 (pré-entraîné) + LogisticRegression",
                "confidence_floor": _CONFIDENCE_FLOOR_PRE,
            },
        )
