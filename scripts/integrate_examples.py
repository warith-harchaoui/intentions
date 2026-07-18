"""Integrate generated examples into the KB and the evaluation datasets.

Module summary
--------------
One-off maintenance script. It reads ``eval/generated_examples.json``
(produced during dataset expansion) and:

* appends the new **training** utterances under each intent's
  ``## Exemples`` section in the ``knowledge_base/*.md`` files
  (deduplicated against what is already there);
* rewrites ``eval/dataset.jsonl`` with the held-out **paraphrase** test set;
* rewrites ``eval/dataset_oos.jsonl`` with the out-of-scope utterances.

Idempotent-ish: re-running never duplicates a training example (it checks
membership before appending), and always overwrites the two JSONL datasets.

Usage
-----
    python scripts/integrate_examples.py

Author
------
Project maintainers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Resolve project paths relative to this script so it runs from any CWD.
_ROOT = Path(__file__).resolve().parent.parent
_KB_DIR = _ROOT / "knowledge_base"
_GENERATED = _ROOT / "eval" / "generated_examples.json"
_EVAL = _ROOT / "eval" / "dataset.jsonl"
_OOS = _ROOT / "eval" / "dataset_oos.jsonl"

# Same heading patterns the KB parser uses, so we split intents identically.
_H1_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$")
_H2_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(?P<item>.+?)\s*$")


def _slug(text: str) -> str:
    """Lower-case, ASCII-ish slug matching the KB parser's intent ids.

    Parameters
    ----------
    text : str
        Heading text.

    Returns
    -------
    str
        A snake_case identifier.
    """
    import unicodedata

    # Strip accents then collapse non-alphanumerics to underscores.
    ascii_text = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def _append_examples_to_file(path: Path, new_by_intent: dict[str, list[str]]) -> int:
    """Append new training bullets under each intent's ## Exemples section.

    Parameters
    ----------
    path : Path
        A knowledge-base Markdown file.
    new_by_intent : dict[str, list[str]]
        Mapping intent id -> new example strings to add.

    Returns
    -------
    int
        Number of example lines actually appended in this file.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    added = 0

    # State machine over the file: which intent we are in, whether we are in
    # its Exemples section, and which of its new examples remain to insert.
    current_intent = ""
    in_examples = False
    existing: set[str] = set()
    pending: list[str] = []

    def _flush() -> None:
        """Insert any pending new examples for the current intent."""
        nonlocal added, pending
        # Append only the examples not already present (dedup), one bullet each.
        for example in pending:
            if example not in existing:
                out.append(f"- {example}")
                existing.add(example)
                added += 1
        pending = []

    for line in lines:
        h1 = _H1_RE.match(line)
        h2 = _H2_RE.match(line)

        # A new intent (or any new heading) closes the previous Exemples list:
        # flush pending inserts *before* emitting the heading line.
        if (h1 or h2) and in_examples:
            _flush()
            in_examples = False

        if h1:
            # Entering a new intent: reset per-intent state and load its new
            # examples (if any) to insert when we reach its Exemples section.
            current_intent = _slug(h1.group("title"))
            existing = set()
            pending = list(new_by_intent.get(current_intent, []))
            out.append(line)
            continue

        if h2:
            # Track whether this section is the Exemples list for the intent.
            in_examples = _slug(h2.group("name")).startswith("exemple")
            out.append(line)
            continue

        # Inside an Exemples section, remember existing bullets for dedup.
        if in_examples:
            bullet = _BULLET_RE.match(line)
            if bullet:
                existing.add(bullet.group("item").strip())
        out.append(line)

    # End of file: flush any pending inserts for the last intent.
    if in_examples:
        _flush()

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return added


def main() -> int:
    """Run the integration and print a short summary to stdout.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    data = json.loads(_GENERATED.read_text(encoding="utf-8"))
    train: dict[str, list[str]] = data["train"]

    # 1) Append training examples across every KB file (skip underscore docs).
    total_added = 0
    for path in sorted(_KB_DIR.glob("*.md")):
        if path.name.startswith("_"):
            continue
        total_added += _append_examples_to_file(path, train)
    print(f"Exemples d'entraînement ajoutés : {total_added}")

    # 2) Rewrite the held-out paraphrase test set (one JSON object per line).
    eval_lines = [
        json.dumps({"text": r["text"], "expected": r["expected"]}, ensure_ascii=False)
        for r in data["eval"]
    ]
    _EVAL.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
    print(f"Jeu de test (paraphrases) écrit : {len(eval_lines)} exemples")

    # 3) Rewrite the out-of-scope set.
    oos_lines = [json.dumps({"text": t}, ensure_ascii=False) for t in data["oos"]]
    _OOS.write_text("\n".join(oos_lines) + "\n", encoding="utf-8")
    print(f"Jeu hors-périmètre écrit : {len(oos_lines)} exemples")
    return 0


if __name__ == "__main__":  # pragma: no cover - one-off script
    raise SystemExit(main())
