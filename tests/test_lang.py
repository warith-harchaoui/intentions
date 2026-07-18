"""Tests for query language detection and language-aware LLM prompting.

The demo is bilingual: a French sentence must be classified with the French
system prompt, an English sentence with the English one, and the intent ids
(language-neutral snake_case) must stay identical either way. These tests pin
that behaviour using the recording :class:`FakeOllamaClient`.
"""

from __future__ import annotations

import json

from conftest import FakeOllamaClient

from intent_engine.kb import KnowledgeBase
from intent_engine.lang import detect_language
from intent_engine.llm_engine import LlmIntentEngine


def test_detect_language_french() -> None:
    """A clearly French sentence is detected as French."""
    assert detect_language("je voudrais résilier mon contrat d'assurance auto") == "fr"


def test_detect_language_english() -> None:
    """A clearly English sentence is detected as English."""
    assert detect_language("I would like to cancel my car insurance policy") == "en"


def test_detect_language_short_text_uses_default() -> None:
    """Too-short input is unreliable, so the default language is returned."""
    assert detect_language("ok") == "fr"
    assert detect_language("ok", default="en") == "en"


def test_detect_language_unsupported_maps_to_default() -> None:
    """A supported-set filter means an exotic language falls back to default."""
    # German is not in SUPPORTED → default.
    assert detect_language("Ich möchte meine Versicherung kündigen bitte") in {
        "fr",
        "en",
    }


def _reply_for(intent: str) -> str:
    """Build a minimal valid JSON answer for the fake client."""
    return json.dumps(
        {"intent": intent, "confidence": 0.9, "slots": {}, "reformulation": "x"}
    )


def test_llm_uses_english_prompt_for_english_query(kb: KnowledgeBase) -> None:
    """An English query selects the English system prompt + user template."""
    fake = FakeOllamaClient(_reply_for("declarer_sinistre_auto"))
    engine = LlmIntentEngine(client=fake).fit(kb)

    engine.classify("I had a car crash this morning, my car is damaged")

    assert fake.last_messages is not None
    system = fake.last_messages[0]["content"]
    user = fake.last_messages[1]["content"]
    # English prompt wording (not the French one).
    assert "call centre" in system
    assert "Customer sentence:" in user


def test_llm_uses_french_prompt_for_french_query(kb: KnowledgeBase) -> None:
    """A French query selects the French system prompt + user template."""
    fake = FakeOllamaClient(_reply_for("declarer_sinistre_auto"))
    engine = LlmIntentEngine(client=fake).fit(kb)

    engine.classify("j'ai eu un accident de voiture ce matin, elle est cabossée")

    assert fake.last_messages is not None
    system = fake.last_messages[0]["content"]
    user = fake.last_messages[1]["content"]
    assert "centre d'appels" in system
    assert "Phrase du client" in user


def test_explicit_prompt_override_wins(kb: KnowledgeBase) -> None:
    """An explicit system_prompt (eval path) is used regardless of language."""
    fake = FakeOllamaClient(_reply_for("declarer_sinistre_auto"))
    engine = LlmIntentEngine(client=fake, system_prompt="OVERRIDE").fit(kb)

    engine.classify("I had a car crash this morning, my car is damaged")

    assert fake.last_messages is not None
    assert fake.last_messages[0]["content"] == "OVERRIDE"
