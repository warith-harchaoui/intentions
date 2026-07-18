"""Load the bilingual string table (UI + prompts) from ``locales/i18n.yaml``.

Module summary
--------------
All localized text — the web GUI chrome *and* the LLM system prompt / user
template — lives in a single YAML file, :data:`STRINGS_PATH`. This module
parses it once (cached) and exposes typed accessors:

* :func:`ui_strings` — the GUI dictionary for one language (served to the
  browser by ``/api/i18n``);
* :func:`system_prompt` / :func:`user_template` — the LLM prompt pieces used
  by :mod:`intent_engine.llm_engine`.

Keeping GUI copy and prompts in the same file means a translator touches one
place, and the product speaks the query's language end-to-end.

Author
------
Project maintainers.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# The YAML lives under locales/ at the repo root (shared by the front and the
# backend), one dir up from this file: intent_engine/ → <repo>/locales/i18n.yaml.
STRINGS_PATH = Path(__file__).resolve().parent.parent / "locales" / "i18n.yaml"

# Languages the table is guaranteed to define; also the fallback order.
DEFAULT_LANG = "fr"


@functools.lru_cache(maxsize=1)
def load_strings() -> dict[str, Any]:
    """Parse and cache the whole string table.

    Returns
    -------
    dict[str, Any]
        The parsed YAML: ``{lang: {"ui": {...}, "prompt": {...}}}``.
    """
    with STRINGS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def available_languages() -> list[str]:
    """Return the language codes defined in the table, in file order.

    Returns
    -------
    list[str]
        e.g. ``["fr", "en"]``.
    """
    return list(load_strings().keys())


def _section(lang: str, section: str) -> dict[str, Any]:
    """Return ``strings[lang][section]``, falling back to the default language.

    Parameters
    ----------
    lang : str
        Requested language code.
    section : str
        Top-level section (``"ui"`` or ``"prompt"``).

    Returns
    -------
    dict[str, Any]
        The requested sub-table (never ``None``).
    """
    table = load_strings()
    entry = table.get(lang) or table.get(DEFAULT_LANG) or {}
    return entry.get(section, {}) or {}


def ui_strings(lang: str) -> dict[str, Any]:
    """Return the GUI string dictionary for ``lang``.

    Parameters
    ----------
    lang : str
        Language code (``"fr"`` / ``"en"``).

    Returns
    -------
    dict[str, Any]
        Nested dictionary of UI strings.
    """
    return _section(lang, "ui")


def all_ui_strings() -> dict[str, dict[str, Any]]:
    """Return every language's UI dictionary, for the ``/api/i18n`` payload.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``{lang: ui_dict}`` for each available language.
    """
    return {lang: ui_strings(lang) for lang in available_languages()}


def system_prompt(lang: str) -> str:
    """Return the LLM system prompt for ``lang``.

    Parameters
    ----------
    lang : str
        Language code.

    Returns
    -------
    str
        The system prompt text.
    """
    return _section(lang, "prompt").get("system", "")


def user_template(lang: str) -> str:
    """Return the LLM user-message template for ``lang``.

    The template contains ``{catalogue}`` and ``{text}`` placeholders filled
    in by the engine at classify time.

    Parameters
    ----------
    lang : str
        Language code.

    Returns
    -------
    str
        The user-message template.
    """
    return _section(lang, "prompt").get("user_template", "")
