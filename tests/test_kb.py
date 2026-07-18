"""Tests for the Markdown knowledge-base parser.

These are functional tests: they parse a realistic sample KB and assert on
the whole extracted structure at once, rather than one micro-test per
private function (per the coding standard's 100-test guidance).

Author
------
Project maintainers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from intent_engine.kb import KnowledgeBase, slugify


def test_slugify_strips_accents_and_punctuation() -> None:
    """Accents and punctuation collapse to ASCII snake_case ids."""
    # Accents removed, spaces/punctuation collapsed to single underscores.
    assert slugify("Déclarer un sinistre !") == "declarer_un_sinistre"
    # Already-clean ids are returned unchanged.
    assert slugify("resilier_contrat") == "resilier_contrat"


def test_parses_all_intents(kb: KnowledgeBase) -> None:
    """The sample KB yields exactly its three intents, in file order."""
    # File order must be preserved for reproducible classifier labels.
    assert kb.intent_ids() == ["assurer_voiture", "declarer_sinistre", "resilier"]
    # ``len`` reflects the intent count.
    assert len(kb) == 3


def test_intent_fields_are_extracted(kb: KnowledgeBase) -> None:
    """Metadata, examples and response are correctly attached to an intent."""
    intent = kb.get("assurer_voiture")
    assert intent is not None
    # Title/service/action come from the blockquote metadata block.
    assert intent.title == "Assurer une voiture"
    assert intent.service == "Souscription auto"
    assert intent.action == "form:souscription_auto"
    # Every bullet under ## Exemples becomes a training example.
    assert "je veux assurer ma voiture" in intent.examples
    assert len(intent.examples) == 4
    # The ## Réponse section is captured and trimmed.
    assert intent.response == "Nous ouvrons un devis auto pour vous."


def test_training_pairs_are_aligned(kb: KnowledgeBase) -> None:
    """Training pairs come back as index-aligned texts and labels."""
    texts, labels = kb.training_pairs()
    # Same length is the core invariant scikit-learn depends on.
    assert len(texts) == len(labels)
    # Three intents * 4 examples each = 12 rows.
    assert len(texts) == 12
    # Spot-check alignment: the first example belongs to the first intent.
    assert labels[0] == "assurer_voiture"


def test_catalogue_caps_examples(kb: KnowledgeBase) -> None:
    """The LLM catalogue exposes at most two examples per intent."""
    catalogue = kb.catalogue()
    # One entry per intent, each with id/title/examples keys.
    assert len(catalogue) == 3
    # Examples are capped at two to keep the prompt small.
    assert all(len(entry["examples"]) <= 2 for entry in catalogue)


def test_missing_directory_raises(tmp_path: Path) -> None:
    """Parsing a non-existent folder fails loudly rather than silently."""
    # A missing KB is a configuration error we want surfaced immediately.
    with pytest.raises(FileNotFoundError):
        KnowledgeBase.from_directory(tmp_path / "does_not_exist")


def test_underscore_files_are_ignored(tmp_path: Path) -> None:
    """Files starting with ``_`` are treated as docs and skipped."""
    # Documentation file that must NOT contribute an intent.
    (tmp_path / "_FORMAT.md").write_text("# not_an_intent\n", encoding="utf-8")
    # A real intent file alongside it.
    (tmp_path / "real.md").write_text(
        "# real_intent\n\n## Exemples\n- bonjour\n", encoding="utf-8"
    )
    parsed = KnowledgeBase.from_directory(tmp_path)
    # Only the real intent survives; the underscore file is ignored.
    assert parsed.intent_ids() == ["real_intent"]
