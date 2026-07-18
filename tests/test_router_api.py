"""Tests for the router orchestration and the FastAPI layer.

These are end-to-end-ish scenario tests: they walk a realistic workflow
(parse KB → build router → classify/execute, or spin up the API and hit its
endpoints) so they exercise many functions at once, catching integration
bugs unit tests miss.

Author
------
Project maintainers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from intent_engine.config import get_settings
from intent_engine.router import IntentRouter


def test_router_classify_and_execute(kb_dir: Path) -> None:
    """The router classifies with TF-IDF and executes into a routing action."""
    router = IntentRouter.from_directory(kb_dir)
    # Classify with the always-available TF-IDF engine.
    result = router.classify("je veux assurer ma voiture", "tfidf")
    assert result.top().intent == "assurer_voiture"
    # Execute turns the confident intent into a concrete action payload.
    execution = router.execute("je veux assurer ma voiture", "tfidf")
    assert execution.handoff is False
    assert execution.action == "form:souscription_auto"
    assert execution.service == "Souscription auto"


def test_router_execute_handoff_on_abstention(real_kb_dir: Path) -> None:
    """An out-of-scope request escalates to a human (handoff=True).

    Uses the real 20-intent KB: abstention needs enough classes for the
    top softmax score to fall below the confidence floor (the toy KB has
    too few classes to trigger it — see ``test_real_kb_abstains``).
    """
    router = IntentRouter.from_directory(real_kb_dir)
    # Nothing in the KB matches; the router must not invent a wrong action.
    execution = router.execute("donne moi la recette de la ratatouille", "tfidf")
    # Explicit "Je ne sais pas → human" safety net: flagged as a handoff, but
    # still carrying a real escalation action so a downstream IVR can route it.
    assert execution.handoff is True
    assert execution.action == "route:conseiller_humain"
    assert "conseiller" in execution.message


def test_router_unknown_engine_raises(kb_dir: Path) -> None:
    """Requesting an unknown engine name is a clear KeyError."""
    router = IntentRouter.from_directory(kb_dir)
    # Guards against typos in engine names propagating silently.
    with pytest.raises(KeyError):
        router.engine("does_not_exist")


@pytest.fixture
def api_client(kb_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a TestClient whose app points at the sample KB.

    Parameters
    ----------
    kb_dir : Path
        The temp KB directory fixture.
    monkeypatch : pytest.MonkeyPatch
        Used to redirect the app's KB directory to the sample.

    Returns
    -------
    TestClient
        A client wrapping a freshly-created app.
    """
    # Point the settings at the temp KB, then clear the cache so the app
    # picks up the override rather than the real knowledge_base/ folder.
    monkeypatch.setenv("INTENT_KNOWLEDGE_BASE_DIR", str(kb_dir))
    get_settings.cache_clear()
    # Import lazily and build a fresh app so the override takes effect.
    from intent_engine.api import create_app

    client = TestClient(create_app())
    yield client
    # Restore the settings cache for any following tests.
    get_settings.cache_clear()


def test_api_health_and_kb(api_client: TestClient) -> None:
    """The health and KB endpoints return the expected shape."""
    health = api_client.get("/api/health").json()
    # TF-IDF and BERT are always usable; status is ok.
    assert health["status"] == "ok"
    assert "tfidf" in health["engines"]
    # The KB endpoint reports the three sample intents.
    kb = api_client.get("/api/kb").json()
    assert kb["count"] == 3
    assert kb["intents"][0]["id"] == "assurer_voiture"


def test_api_classify_and_execute(api_client: TestClient) -> None:
    """The classify and execute endpoints route a clear request."""
    # Classify with TF-IDF via the HTTP layer.
    resp = api_client.post(
        "/api/classify",
        json={"text": "je veux assurer ma voiture", "engine": "tfidf"},
    )
    data = resp.json()
    assert data["ranked"][0]["intent"] == "assurer_voiture"
    # Execute returns the routing action over HTTP.
    resp = api_client.post(
        "/api/execute",
        json={"text": "je veux résilier mon contrat", "engine": "tfidf"},
    )
    execution = resp.json()
    assert execution["action"] == "route:gestion_contrats"


def test_api_rejects_unknown_engine(api_client: TestClient) -> None:
    """A bad engine name yields a 400, not a 500."""
    resp = api_client.post("/api/classify", json={"text": "bonjour", "engine": "nope"})
    # The router's KeyError is translated to a clean client error.
    assert resp.status_code == 400
