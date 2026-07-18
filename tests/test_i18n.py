"""Tests for the i18n string-table loader (``locales/i18n.yaml``).

The bilingual GUI and the LLM prompts both read from a single YAML file. These
tests pin that the loader exposes both languages, that the two locales are kept
in lock-step (same UI keys), and that the prompt pieces the engine needs are
present and language-appropriate.
"""

from __future__ import annotations

from intent_engine.i18n import (
    STRINGS_PATH,
    all_ui_strings,
    available_languages,
    system_prompt,
    ui_strings,
    user_template,
)


def _flatten_keys(d: dict, prefix: str = "") -> set[str]:
    """Return the set of dotted leaf-key paths of a nested dict."""
    keys: set[str] = set()
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys |= _flatten_keys(value, path)
        else:
            keys.add(path)
    return keys


def test_strings_file_exists() -> None:
    """The YAML source of truth lives at the expected path."""
    assert STRINGS_PATH.name == "i18n.yaml"
    assert STRINGS_PATH.parent.name == "locales"
    assert STRINGS_PATH.is_file()


def test_available_languages() -> None:
    """Both French and English are defined."""
    assert set(available_languages()) == {"fr", "en"}


def test_locales_are_in_lockstep() -> None:
    """FR and EN expose exactly the same UI keys (no missing translation)."""
    fr_keys = _flatten_keys(ui_strings("fr"))
    en_keys = _flatten_keys(ui_strings("en"))
    assert fr_keys == en_keys, f"UI key mismatch: {fr_keys ^ en_keys}"


def test_all_ui_strings_shape() -> None:
    """The API payload maps each language to its UI dictionary."""
    payload = all_ui_strings()
    assert set(payload) == {"fr", "en"}
    assert payload["fr"]["form"]["submit_idle"] == "Analyser l'intention"
    assert payload["en"]["form"]["submit_idle"] == "Analyse the intent"


def test_prompts_present_and_localised() -> None:
    """Each language yields a non-empty, language-appropriate prompt + template."""
    assert "centre d'appels" in system_prompt("fr")
    assert "call centre" in system_prompt("en")
    # The user template carries the placeholders the engine fills in.
    for lang in ("fr", "en"):
        template = user_template(lang)
        assert "{catalogue}" in template
        assert "{text}" in template


def test_unknown_language_falls_back_to_default() -> None:
    """An unsupported language code falls back to the default (French)."""
    # 'de' is not defined; the loader returns the French section.
    assert system_prompt("de") == system_prompt("fr")
