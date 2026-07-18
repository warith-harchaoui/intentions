"""Minimal synchronous Ollama client (chat + embeddings).

Module summary
--------------
A small, dependency-light wrapper over the local Ollama REST API, adapted
from the async client pattern used across the author's projects but kept
**synchronous** here: the intent engines and the CLI call it in a plain,
blocking style, and the FastAPI layer runs it in a threadpool. Two
capabilities are exposed:

* :meth:`OllamaClient.chat` — one buffered chat completion, optionally
  constrained to strict JSON via Ollama's ``format`` parameter. This backs
  the LLM intent engine.
* :meth:`OllamaClient.embed` — a single text embedding, used by the BERT
  engine when it runs on the Ollama embedding backend instead of
  sentence-transformers.

Every network failure is turned into a typed :class:`OllamaError` so the
callers can degrade gracefully (abstain, fall back) instead of crashing.

Usage example
-------------
>>> from intent_engine.ollama_client import OllamaClient
>>> client = OllamaClient(base_url="http://localhost:11434")
>>> client.base_url
'http://localhost:11434'

Author
------
Project maintainers.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Raised when an Ollama request fails or returns an unusable body.

    Using a dedicated exception type lets callers catch *only* Ollama
    trouble (server down, timeout, malformed JSON) and react by abstaining,
    without swallowing unrelated programming errors.
    """


class OllamaClient:
    """Blocking client for a local Ollama server.

    Parameters
    ----------
    base_url : str
        Root URL of the Ollama server, e.g. ``http://localhost:11434``.
    timeout_s : float, optional
        Per-request timeout in seconds, by default 120. Local models on CPU
        can be slow, so this is generous on purpose.

    Examples
    --------
    >>> OllamaClient("http://localhost:11434/").base_url
    'http://localhost:11434'
    """

    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        """Store the normalised base URL and the request timeout."""
        # Strip a trailing slash so endpoint concatenation never doubles it.
        self.base_url: str = base_url.rstrip("/")
        # Kept as an attribute so each call reuses the same budget.
        self.timeout_s: float = timeout_s

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Run one chat completion and return the assistant's text.

        Parameters
        ----------
        model : str
            Ollama model tag (e.g. ``gemma4:e4b-mlx``). A leading
            ``ollama/`` prefix, if present, is stripped.
        messages : list[dict[str, str]]
            Chat thread as ``{"role": ..., "content": ...}`` dicts.
        json_mode : bool, optional
            When ``True``, ask Ollama to constrain the output to valid JSON
            via its ``format="json"`` grammar, by default ``False``.
        temperature : float, optional
            Sampling temperature; ``0.0`` (default) makes intent
            classification as deterministic as the model allows.

        Returns
        -------
        str
            The assistant message content (a JSON string when
            ``json_mode`` is set).

        Raises
        ------
        OllamaError
            On any connection, timeout, HTTP or decoding failure.

        Examples
        --------
        >>> # Requires a running Ollama server, so only the shape is shown:
        >>> isinstance(OllamaClient("http://x").chat, object)
        True
        """
        endpoint = f"{self.base_url}/api/chat"
        # Build the request. ``stream: False`` asks for a single buffered
        # body instead of the token-by-token NDJSON stream, which is simpler
        # to parse and fine for short classification answers.
        payload: dict[str, Any] = {
            "model": model.split("/", 1)[-1],
            "messages": messages,
            "stream": False,
            # ``options`` is Ollama's per-request sampling override; pinning
            # temperature to 0 keeps the intent decision reproducible.
            "options": {"temperature": temperature},
        }
        # JSON mode: Ollama enforces a JSON grammar on the decoder, so the
        # model literally cannot emit a stray prose preamble around the JSON.
        if json_mode:
            payload["format"] = "json"

        try:
            # One-shot client per call keeps the class stateless and avoids
            # leaking sockets across the long-lived server process.
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            # Collapse every low-level failure into our typed error so the
            # LLM engine can catch a single exception and abstain cleanly.
            raise OllamaError(f"chat request failed: {exc}") from exc

        # Chat responses nest the text under ``message.content``. Default to
        # an empty string so a malformed body degrades to blank, not a crash.
        return data.get("message", {}).get("content", "")

    def embed(self, model: str, text: str) -> list[float]:
        """Return the embedding vector for one piece of text.

        Parameters
        ----------
        model : str
            Ollama embedding model tag (e.g. ``nomic-embed-text``).
        text : str
            The text to embed.

        Returns
        -------
        list[float]
            The dense embedding vector.

        Raises
        ------
        OllamaError
            On any network/decoding failure or an empty embedding.

        Notes
        -----
        Uses the ``/api/embeddings`` endpoint (single input) rather than the
        batched ``/api/embed`` so the client works across Ollama versions.
        """
        endpoint = f"{self.base_url}/api/embeddings"
        # ``prompt`` is the single-text field the classic embeddings endpoint
        # expects; the model tag is passed bare (no ``ollama/`` prefix).
        payload = {"model": model.split("/", 1)[-1], "prompt": text}
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise OllamaError(f"embed request failed: {exc}") from exc

        # The vector lives under ``embedding``. An empty/missing vector means
        # the model tag is wrong or not an embedding model — treat as error
        # so the BERT engine can fall back rather than train on zeros.
        vector = data.get("embedding", [])
        if not vector:
            raise OllamaError(f"empty embedding from model {model!r}")
        return [float(x) for x in vector]

    def is_available(self) -> bool:
        """Cheaply probe whether the Ollama server is reachable.

        Returns
        -------
        bool
            ``True`` if ``/api/tags`` answers, ``False`` on any failure.

        Notes
        -----
        Used by the API/UI to show an honest "LLM engine offline" state
        instead of making the user wait for a doomed request to time out.
        """
        try:
            # Short timeout: this is a liveness probe, not an inference call,
            # so we do not want to hang the UI for the full request budget.
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
            return True
        except httpx.HTTPError:
            # Any failure (connection refused, timeout) means "not available".
            return False
