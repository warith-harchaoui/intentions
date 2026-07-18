"""FastAPI application exposing the intent engines to the web front end.

Module summary
--------------
A thin HTTP layer over :class:`~intent_engine.router.IntentRouter`. It
builds the router once at startup (parsing the knowledge base), then serves
a handful of JSON endpoints the single-page front end calls, plus the static
files of that front end itself. The endpoints are deliberately synchronous:
the engines are blocking (scikit-learn, a blocking Ollama client), and
FastAPI runs sync path operations in a threadpool, so we get concurrency
without an async rewrite.

Endpoints
---------
* ``GET  /api/health``  — liveness + which engines are usable right now.
* ``GET  /api/kb``      — the intent catalogue for the UI's KB browser.
* ``POST /api/classify``— one engine, one utterance.
* ``POST /api/compare`` — every available engine, one utterance.
* ``POST /api/execute`` — classify then return the routing action + slots.
* ``GET  /``            — the single-page app (static ``web/`` folder).

Usage example
-------------
>>> from intent_engine.api import create_app
>>> app = create_app()
>>> app.title
'Déraison Assurances — Intent Engine'

Author
------
Project maintainers.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .i18n import DEFAULT_LANG as i18n_default_lang_value
from .i18n import all_ui_strings
from .i18n import available_languages as i18n_languages
from .router import IntentRouter


def i18n_default_lang() -> str:
    """Return the default UI language code (module-level shim for the route)."""
    return i18n_default_lang_value


# Absolute path to the bundled single-page front end, resolved relative to
# this file so it works regardless of the process working directory.
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class ClassifyRequest(BaseModel):
    """Request body for ``/api/classify`` and ``/api/execute``.

    Parameters
    ----------
    text : str
        The user's natural-language utterance.
    engine : str | None
        Engine name (``"tfidf"``, ``"bert"``, ``"llm"``); ``None`` uses the
        configured default.
    """

    # The sentence to classify/execute. ``min_length=1`` rejects empty posts.
    text: str = Field(min_length=1, max_length=2000)
    # Optional engine override; validated against the router at call time.
    engine: str | None = None


class CompareRequest(BaseModel):
    """Request body for ``/api/compare``.

    Parameters
    ----------
    text : str
        The user's natural-language utterance to fan out to every engine.
    """

    # Only the text is needed; ``compare`` always runs all usable engines.
    text: str = Field(min_length=1, max_length=2000)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    The knowledge base is parsed and the router constructed exactly once,
    at import/startup, and closed over by the endpoint handlers. Engines
    inside the router are still built lazily on first use.

    Returns
    -------
    FastAPI
        The configured application, ready for uvicorn.

    Examples
    --------
    >>> app = create_app()
    >>> any(r.path == "/api/health" for r in app.routes)
    True
    """
    settings = get_settings()
    # Parse the KB and build the router now: a parse error should fail loudly
    # at startup, not on the first request.
    router = IntentRouter.from_directory(settings.knowledge_base_dir)

    app = FastAPI(
        title="Déraison Assurances — Intent Engine",
        description=(
            "Comparateur pédagogique de trois moteurs d'intentions "
            "(TF-IDF, BERT, LLM) pour un chatbot d'assurance."
        ),
        version="0.1.0",
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """Report liveness and the engines usable right now.

        Returns
        -------
        dict[str, object]
            ``status``, the list of usable ``engines``, and the ``llm_model``
            tag so the UI can display which local model backs the LLM engine.
        """
        # ``available_engines`` probes Ollama, so this doubles as the LLM
        # availability signal the UI uses to enable/disable that column.
        return {
            "status": "ok",
            "engines": router.available_engines(),
            "llm_model": settings.llm_model,
        }

    @app.get("/api/kb")
    def knowledge_base() -> dict[str, object]:
        """Return the intent catalogue for the UI's knowledge-base browser.

        Returns
        -------
        dict[str, object]
            The number of intents and, per intent, its id/title/service and
            example utterances (so the UI can suggest test phrases).
        """
        # Shape a compact, UI-friendly view of each intent. We include the
        # examples so the front end can offer them as clickable test phrases.
        intents = [
            {
                "id": intent.intent_id,
                "title": intent.title,
                "service": intent.service,
                "action": intent.action,
                "examples": intent.examples,
            }
            for intent in router.kb.intents
        ]
        return {"count": len(intents), "intents": intents}

    @app.get("/api/i18n")
    def i18n() -> dict[str, object]:
        """Return the bilingual GUI string table (from ``locales/i18n.yaml``).

        The front fetches this once and applies the chosen language to every
        ``data-i18n`` node, so no user-facing copy is hard-coded in the
        JavaScript. Only the ``ui`` sections are exposed — the LLM prompts stay
        server-side.

        Returns
        -------
        dict[str, object]
            ``{"default": <lang>, "languages": [...], "strings": {lang: {...}}}``.
        """
        return {
            "default": i18n_default_lang(),
            "languages": i18n_languages(),
            "strings": all_ui_strings(),
        }

    @app.post("/api/classify")
    def classify(request: ClassifyRequest) -> dict[str, object]:
        """Classify one utterance with a single engine.

        Parameters
        ----------
        request : ClassifyRequest
            The utterance and optional engine override.

        Returns
        -------
        dict[str, object]
            The serialized :class:`~intent_engine.base.IntentResult`.

        Raises
        ------
        HTTPException
            400 if the engine name is unknown.
        """
        # Translate the router's ``KeyError`` (bad engine name) into a clean
        # 400 so the client gets a helpful message instead of a 500.
        try:
            result = router.classify(request.text, request.engine)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    @app.post("/api/compare")
    def compare(request: CompareRequest) -> dict[str, object]:
        """Run every usable engine on one utterance for a side-by-side view.

        Parameters
        ----------
        request : CompareRequest
            The utterance to classify with all engines.

        Returns
        -------
        dict[str, object]
            Mapping of engine name to its serialized result.
        """
        # Serialize each engine's result; the UI renders them as columns.
        results = router.compare(request.text)
        return {name: result.to_dict() for name, result in results.items()}

    @app.post("/api/execute")
    def execute(request: ClassifyRequest) -> dict[str, object]:
        """Classify then return the concrete routing action + slots.

        Parameters
        ----------
        request : ClassifyRequest
            The natural-language request and optional engine override.

        Returns
        -------
        dict[str, object]
            The serialized :class:`~intent_engine.router.Execution`.

        Raises
        ------
        HTTPException
            400 if the engine name is unknown.
        """
        # Same error translation as ``classify`` for a bad engine name.
        try:
            execution = router.execute(request.text, request.engine)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return execution.to_dict()

    # Mount the single-page app at the root. ``html=True`` makes ``/`` serve
    # ``index.html``. Mounted last so it never shadows the ``/api/*`` routes.
    # Guarded so importing the app in a test without the web folder present
    # (unlikely, but safe) does not crash at startup.
    if _WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")

    return app


# Module-level ASGI app so ``uvicorn intent_engine.api:app`` just works.
app = create_app()
