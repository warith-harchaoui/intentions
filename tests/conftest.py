"""Shared pytest fixtures for the intent-engine test suite.

Module summary
--------------
Provides a small, self-contained knowledge base written to a temporary
directory so the fast tests never depend on the real ``knowledge_base/``
folder (which may grow or change) and never touch the network. A fake
Ollama client is also provided so the LLM engine can be exercised
deterministically without a running server.

Author
------
Project maintainers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from intent_engine.kb import KnowledgeBase

# A tiny, deterministic KB with three well-separated intents. Kept minimal so
# the classifiers train in milliseconds and the expected top intent is
# unambiguous, which makes the assertions stable.
_SAMPLE_KB = """\
# assurer_voiture

> **Titre** : Assurer une voiture
> **Service** : Souscription auto
> **Action** : form:souscription_auto

## Exemples
- je veux assurer ma voiture
- souscrire une assurance auto
- devis pour mon véhicule
- comment assurer mon automobile

## Réponse
Nous ouvrons un devis auto pour vous.


# declarer_sinistre

> **Titre** : Déclarer un sinistre
> **Service** : Sinistres
> **Action** : route:sinistres

## Exemples
- j'ai eu un accident
- déclarer un sinistre
- ma voiture est abîmée
- quelqu'un a embouti ma portière

## Réponse
Je transmets votre sinistre au service dédié.


# resilier

> **Titre** : Résilier un contrat
> **Service** : Gestion des contrats
> **Action** : route:gestion_contrats

## Exemples
- je veux résilier mon contrat
- annuler mon assurance
- mettre fin à mon contrat
- comment résilier

## Réponse
Nous préparons votre résiliation.
"""


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """Write the sample KB to a temp folder and return its path.

    Parameters
    ----------
    tmp_path : Path
        Pytest's per-test temporary directory.

    Returns
    -------
    Path
        Directory containing a single ``sample.md`` knowledge-base file.
    """
    # One file is enough; the parser globs ``*.md`` regardless of count.
    md = tmp_path / "sample.md"
    md.write_text(_SAMPLE_KB, encoding="utf-8")
    return tmp_path


@pytest.fixture
def real_kb_dir() -> Path:
    """Return the repo's real ``knowledge_base/`` directory.

    Used by the few integration tests that need the full 20-intent
    catalogue (e.g. absolute-confidence abstention), which the tiny toy KB
    cannot reproduce. The folder is committed to the repo, so this stays
    deterministic and offline.

    Returns
    -------
    Path
        Absolute path to ``knowledge_base/`` at the repository root.
    """
    # Resolve relative to this test file so it works from any CWD.
    return Path(__file__).resolve().parent.parent / "knowledge_base"


@pytest.fixture
def kb(kb_dir: Path) -> KnowledgeBase:
    """Parse the sample KB into a :class:`KnowledgeBase`.

    Parameters
    ----------
    kb_dir : Path
        The temp directory fixture holding the sample Markdown.

    Returns
    -------
    KnowledgeBase
        The parsed knowledge base (three intents).
    """
    # Reuse the public parsing entry point so the fixture exercises the same
    # code path production uses.
    return KnowledgeBase.from_directory(kb_dir)


class FakeOllamaClient:
    """A stand-in for :class:`~intent_engine.ollama_client.OllamaClient`.

    It returns a scripted JSON answer for ``chat`` and reports itself as
    available, so the LLM engine's parsing/validation logic can be tested
    without a live Ollama server.

    Parameters
    ----------
    reply : str
        The exact string ``chat`` should return (usually a JSON blob).
    available : bool, optional
        What ``is_available`` reports, by default ``True``.
    """

    def __init__(self, reply: str, available: bool = True) -> None:
        """Store the canned reply and availability flag."""
        # The canned response the fake ``chat`` will echo back verbatim.
        self._reply = reply
        # Lets tests simulate a downed server via ``is_available`` == False.
        self._available = available
        # Records the last messages sent, so tests can assert on the prompt.
        self.last_messages: list[dict[str, str]] | None = None

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Return the canned reply, recording the messages for assertions."""
        # Capture the prompt so a test can verify the catalogue was injected.
        self.last_messages = messages
        return self._reply

    def is_available(self) -> bool:
        """Report the configured availability."""
        return self._available
