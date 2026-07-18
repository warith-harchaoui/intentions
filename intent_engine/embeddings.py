"""Pluggable sentence-embedding backends for the BERT intent engine.

Module summary
--------------
The "type BERT" representation the second approach relies on is a dense
sentence embedding. This module hides *where* that embedding comes from
behind one tiny :class:`Embedder` protocol, with two interchangeable
implementations:

* :class:`SentenceTransformerEmbedder` — the canonical choice, a
  multilingual SBERT model (Reimers & Gurevych, 2019) run locally through
  PyTorch. Best quality, but pulls a heavy dependency.
* :class:`OllamaEmbedder` — the same idea served by a BERT-family model
  (``nomic-embed-text``) through the already-installed Ollama server, so
  the demo runs on a machine without PyTorch.

:func:`build_embedder` picks one according to configuration, with an
``"auto"`` mode that prefers sentence-transformers and silently falls back
to Ollama when the import is unavailable. This keeps the teaching demo
runnable everywhere while still showcasing the "proper" path when it is
installed.

Usage example
-------------
>>> from intent_engine.embeddings import build_embedder
>>> emb = build_embedder(backend="ollama")  # doctest: +SKIP
>>> vec = emb.encode(["bonjour"])           # doctest: +SKIP

Author
------
Project maintainers.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from .config import get_settings
from .ollama_client import OllamaClient

# Module logger — per the coding standard, no bare ``print`` in library code.
logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Structural type for anything that maps texts to dense vectors.

    A backend just needs an ``encode`` method and a human-readable
    ``name``; the BERT engine depends on this protocol, not on any concrete
    class, so new backends can be dropped in without touching the engine.
    """

    # Human-readable backend label, surfaced in the result diagnostics.
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into a 2-D array of shape (n, dim)."""
        ...


class SentenceTransformerEmbedder:
    """SBERT multilingual sentence embeddings via ``sentence-transformers``.

    Parameters
    ----------
    model_name : str
        Hugging Face model id to load.

    Notes
    -----
    The heavy ``SentenceTransformer`` import happens lazily in
    ``__init__`` so merely importing this module never drags in PyTorch.
    """

    def __init__(self, model_name: str) -> None:
        """Load the SBERT model, importing the library lazily."""
        # Import inside the constructor: the dependency is optional, so a
        # user on the Ollama backend never pays the (large) import cost.
        from sentence_transformers import SentenceTransformer

        # Human-readable name embeds the model id for the diagnostics panel.
        self.name: str = f"sentence-transformers:{model_name}"
        # The actual model. First construction downloads weights (~470 MB
        # for the default multilingual MiniLM) and caches them on disk.
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` into an ``(n, dim)`` float array.

        Parameters
        ----------
        texts : list[str]
            Input sentences.

        Returns
        -------
        np.ndarray
            Row-wise sentence embeddings.
        """
        # ``convert_to_numpy`` keeps us in the numpy world scikit-learn wants;
        # ``normalize_embeddings`` makes cosine == dot product, which helps a
        # linear classifier separate the classes cleanly.
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class OllamaEmbedder:
    """Sentence embeddings served by a BERT-family model through Ollama.

    Parameters
    ----------
    model_name : str
        Ollama embedding model tag (e.g. ``nomic-embed-text``).
    base_url : str
        Base URL of the Ollama server.

    Notes
    -----
    ``nomic-embed-text`` is a BERT-architecture embedding model, so this
    backend still demonstrates the "type BERT representation" idea — just
    without a local PyTorch install.
    """

    def __init__(self, model_name: str, base_url: str) -> None:
        """Wire up the Ollama client and remember the embedding model."""
        # Backend label for the diagnostics panel.
        self.name: str = f"ollama:{model_name}"
        # Reuse the shared synchronous client; embeddings are a single-text
        # POST per call, which is fine for our small KB and short queries.
        self._client = OllamaClient(base_url)
        self._model = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` one call at a time into an ``(n, dim)`` array.

        Parameters
        ----------
        texts : list[str]
            Input sentences.

        Returns
        -------
        np.ndarray
            Row-wise embeddings, L2-normalised for cosine geometry.

        Raises
        ------
        OllamaError
            If the server is unreachable or the model is not an embedder.
        """
        vectors: list[list[float]] = []
        # The classic embeddings endpoint takes one text at a time; loop over
        # the (small) batch. For a big corpus you would switch to the batched
        # ``/api/embed`` endpoint, but per-text keeps version-compatibility.
        for text in texts:
            vectors.append(self._client.embed(self._model, text))
        matrix = np.asarray(vectors, dtype=float)
        # L2-normalise so downstream cosine similarity == dot product, matching
        # the SBERT backend's geometry and keeping the classifier comparable.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero on the (pathological) all-zero vector.
        norms[norms == 0] = 1.0
        return matrix / norms


def build_embedder(backend: str | None = None) -> Embedder:
    """Build the configured embedder, honouring the ``auto`` fallback.

    Parameters
    ----------
    backend : str | None, optional
        One of ``"auto"``, ``"st"`` (sentence-transformers) or
        ``"ollama"``. Defaults to the value in settings.

    Returns
    -------
    Embedder
        A ready-to-use embedding backend.

    Raises
    ------
    OllamaError
        If ``"ollama"`` is forced but the server/model is unavailable.
    ImportError
        If ``"st"`` is forced but ``sentence-transformers`` is missing.

    Examples
    --------
    >>> callable(build_embedder)
    True
    """
    settings = get_settings()
    # Fall back to the configured default when the caller passes nothing.
    choice = (backend or settings.embedding_backend).lower()

    # Forced sentence-transformers: let the ImportError propagate so the
    # caller knows the dependency is genuinely missing.
    if choice == "st":
        return SentenceTransformerEmbedder(settings.sentence_transformer_model)

    # Forced Ollama: construct directly; embedding errors surface on encode.
    if choice == "ollama":
        return OllamaEmbedder(settings.ollama_embedding_model, settings.ollama_base_url)

    # ``auto`` (the default): prefer the higher-quality SBERT path, but if it
    # is not installed, degrade to Ollama so the demo still runs. We log the
    # decision so the choice is visible rather than silent magic.
    try:
        embedder = SentenceTransformerEmbedder(settings.sentence_transformer_model)
        logger.info("BERT engine using sentence-transformers backend.")
        return embedder
    except ImportError:
        # sentence-transformers (and PyTorch) not installed — fall back.
        logger.info(
            "sentence-transformers unavailable; BERT engine falling back "
            "to the Ollama embedding backend (%s).",
            settings.ollama_embedding_model,
        )
        return OllamaEmbedder(settings.ollama_embedding_model, settings.ollama_base_url)
