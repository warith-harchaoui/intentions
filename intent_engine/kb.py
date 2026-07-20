"""Parse the Markdown knowledge base into structured intents.

Module summary
--------------
The single source of truth for the whole project is a folder of Markdown
files where **every ``# h1`` heading is one intention**. Under each
heading a small, human-writable convention captures everything the
engines need:

* a ``>`` block-quote of ``**Key** : value`` metadata (display title,
  routed service, machine action, urgency);
* a ``## Exemples`` bullet list — the training utterances for the TF-IDF
  and BERT engines, and few-shot fuel for the LLM prompt;
* a ``## Réponse`` section — the scripted answer the chatbot reads back.

This module turns that prose into :class:`Intent` records and a
:class:`KnowledgeBase` container with the little query helpers the engines
and the API need (training pairs, the catalogue for the LLM prompt, lookup
by id). Business experts edit Markdown; nobody touches Python to add an
intention.

Usage example
-------------
>>> from intent_engine.kb import KnowledgeBase
>>> kb = KnowledgeBase.from_directory("knowledge_base")
>>> len(kb) > 0
True

Author
------
Project maintainers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Line-level patterns --------------------------------------------------
# A top-level heading ``# Something`` opens a new intention. We require
# exactly one ``#`` (not ``##``) so sub-sections never start a new intent.
_H1_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$")
# A second-level heading ``## Exemples`` / ``## Réponse`` opens a section
# *inside* the current intention.
_H2_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$")
# A block-quote metadata line: ``> **Service** : Gestion des sinistres``.
# Both ``:`` and ``：`` (full-width) are tolerated for forgiving authoring.
_META_RE = re.compile(r"^>\s*\*\*(?P<key>[^*]+)\*\*\s*[:：]\s*(?P<value>.*?)\s*$")
# A bullet-list item ``- example`` or ``* example`` — one training phrase.
_BULLET_RE = re.compile(r"^\s*[-*]\s+(?P<item>.+?)\s*$")


def slugify(text: str) -> str:
    """Turn a heading into a stable, ASCII, snake_case intent id.

    Accents are stripped, spaces and punctuation collapse to single
    underscores, and the result is lower-cased. This keeps intent ids
    filename- and JSON-safe while staying readable.

    Parameters
    ----------
    text : str
        Raw heading text, possibly with accents, spaces and punctuation.

    Returns
    -------
    str
        A ``snake_case`` ASCII identifier.

    Examples
    --------
    >>> slugify("Déclarer un sinistre automobile")
    'declarer_un_sinistre_automobile'
    >>> slugify("resilier_contrat")
    'resilier_contrat'
    """
    # Decompose accented characters (é -> e + combining accent) then drop
    # the combining marks, yielding plain ASCII letters.
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    # Lower-case, then replace any run of non-alphanumeric characters with a
    # single underscore so "sinistre auto !" and "sinistre_auto" collapse.
    lowered = ascii_text.lower()
    underscored = re.sub(r"[^a-z0-9]+", "_", lowered)
    # Trim leading/trailing underscores left by punctuation at the edges.
    return underscored.strip("_")


@dataclass
class Intent:
    """One intention parsed from a knowledge-base Markdown file.

    Parameters
    ----------
    intent_id : str
        Stable machine id (``slugify`` of the heading, unless a
        ``**Id**`` metadata line overrides it).
    title : str
        Human-readable label shown in the UI (the raw ``# h1`` text, or a
        ``**Titre**`` override).
    examples : list[str]
        Training/few-shot utterances from the ``## Exemples`` section.
    response : str
        Scripted answer from the ``## Réponse`` section.
    service : str
        Department the call should be routed to (metadata ``**Service**``).
    action : str
        Machine-readable routing action (metadata ``**Action**``), e.g.
        ``"route:sinistres_auto"`` or ``"form:souscription_auto"``.
    metadata : dict[str, str]
        Any other ``**Key** : value`` pairs, verbatim, for extensibility.
    source_file : str
        Basename of the Markdown file this intent came from (for tracing).

    Examples
    --------
    >>> Intent(intent_id="a", title="A").intent_id
    'a'
    """

    # Machine id and human label.
    intent_id: str
    title: str
    # Learning material and the canned answer.
    examples: list[str] = field(default_factory=list)
    response: str = ""
    # Routing metadata used when we "execute" the detected intent.
    service: str = ""
    action: str = ""
    # Escape hatch for extra metadata keys we do not model explicitly.
    metadata: dict[str, str] = field(default_factory=dict)
    # Provenance, handy when several files define related intents.
    source_file: str = ""


class KnowledgeBase:
    """An in-memory collection of :class:`Intent` records with helpers.

    The container hides how many files the intents came from: the engines
    only ever ask for "all training pairs" or "the catalogue" or "the
    intent with this id".

    Parameters
    ----------
    intents : list[Intent]
        The parsed intents, typically produced by :meth:`from_directory`.

    Examples
    --------
    >>> kb = KnowledgeBase([Intent(intent_id="x", title="X",
    ...     examples=["salut"], response="Bonjour")])
    >>> kb.get("x").title
    'X'
    """

    def __init__(self, intents: list[Intent]) -> None:
        """Store the intents and build the id lookup index."""
        # Keep the ordered list (stable label ordering for the classifiers)
        # and a dict for O(1) id lookup used by the router and API.
        self.intents: list[Intent] = intents
        # Build the id -> Intent index once; ids are unique by construction
        # (see ``from_directory`` which de-duplicates).
        self._by_id: dict[str, Intent] = {i.intent_id: i for i in intents}

    def __len__(self) -> int:
        """Return the number of intents in the knowledge base."""
        # Enables ``len(kb)`` and truthiness checks (``if kb:``).
        return len(self.intents)

    def get(self, intent_id: str) -> Intent | None:
        """Look up an intent by its id.

        Parameters
        ----------
        intent_id : str
            The machine id to resolve.

        Returns
        -------
        Intent | None
            The matching intent, or ``None`` if unknown.
        """
        # Direct dict hit; ``None`` lets callers detect an unknown/hallucinated
        # id (relevant for the LLM engine, which could invent one).
        return self._by_id.get(intent_id)

    def intent_ids(self) -> list[str]:
        """Return all intent ids, in file order.

        Returns
        -------
        list[str]
            The ordered list of intent identifiers.
        """
        # Order matters: scikit-learn label ordering stays reproducible.
        return [intent.intent_id for intent in self.intents]

    def training_pairs(self) -> tuple[list[str], list[str]]:
        """Flatten every example into aligned ``(texts, labels)`` lists.

        This is the supervised dataset the TF-IDF and BERT engines learn
        from: each example utterance paired with its intent id.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(texts, labels)`` of equal length.

        Examples
        --------
        >>> kb = KnowledgeBase([Intent(intent_id="x", title="X",
        ...     examples=["a", "b"])])
        >>> kb.training_pairs()
        (['a', 'b'], ['x', 'x'])
        """
        texts: list[str] = []
        labels: list[str] = []
        # Walk intents in order and emit one (text, label) row per example
        # so the two parallel lists stay index-aligned for scikit-learn.
        for intent in self.intents:
            for example in intent.examples:
                texts.append(example)
                labels.append(intent.intent_id)
        return texts, labels

    def catalogue(self) -> list[dict[str, Any]]:
        """Return a compact catalogue for the LLM prompt.

        The LLM engine is zero-shot: instead of training data it needs a
        short description of each intent to reason over. We hand it the id,
        the human title and one or two example utterances.

        Returns
        -------
        list[dict[str, Any]]
            One dict per intent with ``id``, ``title`` and ``examples``.
        """
        catalogue: list[dict[str, Any]] = []
        # Cap examples per intent so the prompt stays small: two utterances
        # are enough to disambiguate an intent for a capable LLM, and a huge
        # prompt would slow the local model down for no accuracy gain.
        for intent in self.intents:
            catalogue.append(
                {
                    "id": intent.intent_id,
                    "title": intent.title,
                    "examples": intent.examples[:2],
                }
            )
        return catalogue

    @classmethod
    def from_directory(cls, directory: str | Path) -> KnowledgeBase:
        """Parse every ``*.md`` file in a directory into a knowledge base.

        Files are read in sorted order for reproducibility. Any file whose
        name starts with an underscore is treated as documentation and
        skipped, so a ``_FORMAT.md`` note can live alongside the data.

        Parameters
        ----------
        directory : str | Path
            Folder containing the Markdown knowledge base.

        Returns
        -------
        KnowledgeBase
            The parsed, de-duplicated knowledge base.

        Raises
        ------
        FileNotFoundError
            If ``directory`` does not exist.
        """
        base = Path(directory)
        # Fail loudly on a missing folder: a silent empty KB would make every
        # engine abstain with a confusing "no intents" symptom downstream.
        if not base.is_dir():
            raise FileNotFoundError(f"Knowledge base folder not found: {base}")

        collected: list[Intent] = []
        seen_ids: set[str] = set()
        # Sorted iteration keeps intent/label ordering stable across runs and
        # machines, which matters for reproducible model training.
        for path in sorted(base.glob("*.md")):
            # Underscore-prefixed files are human docs (format spec, notes),
            # never intent data — skip them.
            if path.name.startswith("_"):
                continue
            # Parse one file into zero-or-more intents and merge, dropping
            # duplicate ids (first definition wins) to keep the index sane.
            for intent in _parse_file(path):
                if intent.intent_id in seen_ids:
                    continue
                seen_ids.add(intent.intent_id)
                collected.append(intent)
        return cls(collected)


def _parse_file(path: Path) -> list[Intent]:
    """Parse a single Markdown file into a list of intents.

    Parameters
    ----------
    path : Path
        The Markdown file to parse.

    Returns
    -------
    list[Intent]
        Every intention defined under an ``# h1`` heading in the file.
    """
    # Read as UTF-8 text; the KB is French and full of accents.
    text = path.read_text(encoding="utf-8")
    intents: list[Intent] = []

    # Accumulator for the intent currently being built. ``None`` until the
    # first ``# h1`` line is seen, so any preamble before the first heading
    # is ignored rather than misattributed.
    current: dict[str, Any] | None = None
    # Which ``## section`` we are inside right now ("exemples"/"reponse"/...);
    # empty string means "directly under the h1, before any sub-section".
    section = ""

    # Walk the file line by line — a tiny state machine is clearer here than
    # pulling in a full Markdown AST for such a constrained format.
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")

        # A new ``# h1`` closes the previous intent and starts a fresh one.
        h1 = _H1_RE.match(line)
        if h1:
            # Flush the intent we were building before switching context.
            if current is not None:
                intents.append(_finalize(current, path.name))
            title = h1.group("title").strip()
            # Seed a new accumulator. ``id`` starts from the slug of the
            # title but a ``**Id**`` metadata line can override it later.
            current = {
                "title": title,
                "id": slugify(title),
                "examples": [],
                "response_lines": [],
                "service": "",
                "action": "",
                "metadata": {},
            }
            section = ""
            continue

        # Ignore everything until the first heading appears.
        if current is None:
            continue

        # A ``## section`` header switches which bucket following lines fill.
        h2 = _H2_RE.match(line)
        if h2:
            # Normalise the section name (strip accents/case) so "Réponse",
            # "reponse" and "Response" all route to the same handler.
            section = slugify(h2.group("name"))
            continue

        # Metadata block-quote lines feed the routing fields regardless of
        # the current section (they usually sit right under the h1).
        meta = _META_RE.match(line)
        if meta:
            _apply_metadata(current, meta.group("key"), meta.group("value"))
            continue

        # In the examples section, each bullet is one training utterance.
        if section.startswith("exemple") or section.startswith("example"):
            bullet = _BULLET_RE.match(line)
            if bullet:
                current["examples"].append(bullet.group("item").strip())
            continue

        # In the response section, keep raw lines (blank lines included) so
        # paragraph breaks survive into the spoken/displayed answer.
        if section.startswith("repons") or section.startswith("respons"):
            current["response_lines"].append(line)
            continue

    # Flush the final intent once the file ends.
    if current is not None:
        intents.append(_finalize(current, path.name))
    return intents


def _apply_metadata(acc: dict[str, Any], key: str, value: str) -> None:
    """Route one ``**Key** : value`` metadata pair into the accumulator.

    Parameters
    ----------
    acc : dict[str, Any]
        The mutable intent accumulator built in :func:`_parse_file`.
    key : str
        The metadata key as written between the ``**`` markers.
    value : str
        The metadata value.

    Notes
    -----
    Recognised keys (accent/case-insensitive): ``id`` overrides the slug,
    ``titre``/``title`` overrides the display label, ``service`` and
    ``action`` fill the routing fields. Anything else is preserved in the
    ``metadata`` dict so authors can add their own keys.
    """
    # Normalise the key for matching without losing the original in metadata.
    norm = slugify(key)
    value = value.strip()
    # Explicit id override lets authors decouple the machine id from the
    # display title (e.g. keep a short id under a long French heading).
    if norm == "id":
        acc["id"] = slugify(value)
    elif norm in {"titre", "title"}:
        acc["title"] = value
    elif norm == "service":
        acc["service"] = value
    elif norm == "action":
        acc["action"] = value
    else:
        # Unknown-but-valid metadata: keep it verbatim for forward-compat.
        acc["metadata"][key.strip()] = value


def _finalize(acc: dict[str, Any], source_file: str) -> Intent:
    """Convert a finished accumulator into an immutable-ish :class:`Intent`.

    Parameters
    ----------
    acc : dict[str, Any]
        The accumulator populated while scanning one ``# h1`` block.
    source_file : str
        Basename of the originating Markdown file, for provenance.

    Returns
    -------
    Intent
        The assembled intent with its response text joined and trimmed.
    """
    # Join the collected response lines and strip surrounding blank lines so
    # the answer starts and ends cleanly when spoken or displayed.
    response = "\n".join(acc["response_lines"]).strip()
    return Intent(
        intent_id=acc["id"],
        title=acc["title"],
        examples=list(acc["examples"]),
        response=response,
        service=acc["service"],
        action=acc["action"],
        metadata=dict(acc["metadata"]),
        source_file=source_file,
    )
