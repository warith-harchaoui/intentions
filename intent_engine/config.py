"""Runtime configuration for the Déraison Assurances intent engine.

Module summary
--------------
Centralizes every tunable knob of the project — where the Markdown
knowledge base lives, which Ollama server and model back the LLM engine,
which sentence-embedding model backs the BERT engine, and the network
timeouts. Values are read once from the environment (and an optional
``.env`` file) and cached, so the rest of the code never touches
``os.environ`` directly.

Why a single module: the intent engines, the FastAPI server and the
CLI all need the same handful of settings. Funnelling them through one
``Settings`` object keeps defaults in one place and makes the whole thing
overridable for tests without monkeypatching scattered constants.

Usage example
-------------
>>> from intent_engine.config import get_settings
>>> settings = get_settings()
>>> settings.ollama_base_url.startswith("http")
True

Author
------
Project maintainers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path to the repository root (the folder that contains this
# package). Derived from ``__file__`` so the defaults work no matter what
# the current working directory is when the server or CLI is launched.
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent


def _default_llm_model() -> str:
    """Pick the default local Gemma tag for the LLM engine.

    We default to ``gemma3:4b``: on this project's hardware it answers in
    ~5 s warm, roughly **8× faster** than the larger ``gemma4:e4b`` (~40 s
    warm) for a small accuracy trade-off — the right pick for an assistant
    where latency matters. The value stays fully overridable through the
    ``INTENT_LLM_MODEL`` environment variable (e.g. set ``gemma4:e4b-mlx``
    back if you want the bigger model).

    Returns
    -------
    str
        ``gemma3:4b`` (a compact, fast, multilingual Gemma).

    Examples
    --------
    >>> _default_llm_model()
    'gemma3:4b'
    """
    # A single fast default across platforms: gemma3:4b is a GGUF model Ollama
    # serves everywhere, and its small size keeps latency low. (There is no
    # widely-pulled MLX build of gemma3:4b, so we skip the host-aware -mlx
    # dance the previous gemma4 default used.)
    return "gemma3:4b"


class Settings(BaseSettings):
    """Environment-backed settings shared by every part of the project.

    Attributes
    ----------
    knowledge_base_dir : Path
        Folder holding the Markdown knowledge base (``# h1`` == intention).
    ollama_base_url : str
        Base URL of the local Ollama server used by the LLM engine.
    llm_model : str
        Ollama tag for the generative model that does zero-shot intent
        classification with a JSON contract.
    embedding_backend : str
        Which embedder the BERT engine uses: ``"auto"`` tries
        sentence-transformers first and falls back to Ollama, ``"st"``
        forces sentence-transformers, ``"ollama"`` forces Ollama.
    sentence_transformer_model : str
        Hugging Face id of the multilingual sentence-transformers model.
    ollama_embedding_model : str
        Ollama tag used when embeddings are served by Ollama instead of
        sentence-transformers (a BERT-family model, hence "type BERT").
    request_timeout_s : float
        Per-request timeout, in seconds, for calls to Ollama.
    default_engine : str
        Engine used by the API/CLI when the caller does not pick one.

    Notes
    -----
    Field values come from (in order of priority) explicit constructor
    arguments, environment variables prefixed with ``INTENT_``, the
    ``.env`` file at the repository root, then the defaults below.
    """

    # ``env_prefix`` namespaces our variables (``INTENT_LLM_MODEL`` etc.)
    # so they never collide with unrelated environment variables, and
    # ``.env`` is loaded from the repo root for a friendly local setup.
    model_config = SettingsConfigDict(
        env_prefix="INTENT_",
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Knowledge base -------------------------------------------------
    # Where the intention catalogue lives. Everything downstream (training
    # data for TF-IDF/BERT, the intent list for the LLM prompt, the
    # scripted answers) is derived from these Markdown files.
    knowledge_base_dir: Path = Field(default=_REPO_ROOT / "knowledge_base")

    # --- LLM engine (Ollama) -------------------------------------------
    # Local Ollama server. Default matches Ollama's out-of-the-box port.
    ollama_base_url: str = Field(default="http://localhost:11434")
    # Generative model tag; host-aware default (MLX on Apple Silicon).
    llm_model: str = Field(default_factory=_default_llm_model)

    # --- BERT engine (embeddings) --------------------------------------
    # ``auto`` keeps the demo running on a machine without PyTorch: it
    # tries sentence-transformers, and if that import fails it transparently
    # uses Ollama's embedding endpoint instead.
    embedding_backend: str = Field(default="auto")
    # Small multilingual model: good French coverage, ~470 MB, CPU-friendly.
    sentence_transformer_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    # nomic-embed-text is a BERT-architecture embedding model already
    # pulled on the target machine — the "type BERT" representation served
    # without a heavyweight PyTorch install.
    ollama_embedding_model: str = Field(default="nomic-embed-text")

    # --- fastText engines ----------------------------------------------
    # Path to the pretrained French fastText model (cc.fr.300.bin, ~4.5 GB).
    # Downloaded on demand (see scripts/download_fasttext.py); the pretrained
    # engine is simply unavailable until this file exists, degrading
    # gracefully like the LLM engine does when Ollama is down.
    fasttext_model_path: Path = Field(default=_REPO_ROOT / "models" / "cc.fr.300.bin")

    # --- Networking / defaults -----------------------------------------
    # Local models on CPU can be slow; a generous timeout avoids counting a
    # slow-but-valid completion as an error.
    request_timeout_s: float = Field(default=120.0)
    # Engine chosen when the caller does not specify one. TF-IDF is the
    # cheapest and always available, so it is the safe default.
    default_engine: str = Field(default="tfidf")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, cached :class:`Settings` instance.

    The cache guarantees the ``.env`` file and environment are parsed only
    once per process. Tests that need a different configuration can call
    ``get_settings.cache_clear()`` after patching the environment.

    Returns
    -------
    Settings
        The singleton settings object for this process.

    Examples
    --------
    >>> a = get_settings()
    >>> b = get_settings()
    >>> a is b
    True
    """
    # Instantiating with no arguments triggers pydantic-settings to read
    # the environment and the ``.env`` file per ``model_config`` above.
    return Settings()
