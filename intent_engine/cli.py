"""Command-line interface for the intent engine.

Module summary
--------------
A thin ``argparse`` front end so you can exercise the whole project from a
terminal without the web UI — perfect for a live "here is how it works"
demo to colleagues. Sub-commands:

* ``classify`` — run one engine on one sentence and print the ranking.
* ``compare``  — run all engines on one sentence, side by side.
* ``execute``  — classify then show the concrete routing action + slots.
* ``intents``  — list the intentions parsed from the knowledge base.

This is the one place the coding standard allows user-facing ``print``:
a CLI's entire job is to write to stdout. The library modules it calls
stay ``print``-free.

Usage example
-------------
>>> # From a shell:
>>> # python -m intent_engine compare "je veux assurer ma voiture"

Author
------
Project maintainers.
"""

from __future__ import annotations

import argparse
import logging

from .config import get_settings
from .router import IntentRouter


def _format_result_line(engine: str, result_repr: str) -> str:
    """Format one engine's one-line summary for the terminal.

    Parameters
    ----------
    engine : str
        Engine name, left-padded for column alignment.
    result_repr : str
        Pre-rendered summary of the result.

    Returns
    -------
    str
        A single aligned line.

    Examples
    --------
    >>> _format_result_line("tfidf", "x")
    'tfidf   | x'
    """
    # Pad the engine name to a fixed width so the columns line up when several
    # engines are printed one under another in ``compare``.
    return f"{engine:<7} | {result_repr}"


def _print_result(engine: str, result) -> None:  # noqa: ANN001 - CLI glue
    """Pretty-print one :class:`IntentResult` to stdout.

    Parameters
    ----------
    engine : str
        The engine name to label the block with.
    result : IntentResult
        The result to render.

    Notes
    -----
    ``result`` is intentionally untyped in the signature to avoid importing
    the type just for a CLI helper; it is duck-typed as an ``IntentResult``.
    """
    top = result.top()
    # Header line: engine, latency, and the winning intent (or an abstention).
    if top is None or not result.confident:
        summary = f"(abstention) — {result.latency_ms:.0f} ms"
    else:
        summary = f"{top.intent}  [{top.score:.2f}]  — {result.latency_ms:.0f} ms"
    print(_format_result_line(engine, summary))
    # Show the ranked runners-up indented, so the demo makes the model's
    # uncertainty visible rather than hiding it behind a single label.
    for prediction in result.ranked[1:]:
        print(f"        · {prediction.intent}  [{prediction.score:.2f}]")
    # Slots (LLM only) are the payload that makes execution possible — print
    # them when present so the audience sees the entity extraction.
    if result.slots:
        print(f"        slots: {result.slots}")


def _cmd_classify(router: IntentRouter, args: argparse.Namespace) -> int:
    """Handle the ``classify`` sub-command.

    Parameters
    ----------
    router : IntentRouter
        The router bound to the knowledge base.
    args : argparse.Namespace
        Parsed CLI arguments (``text``, ``engine``).

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    # One engine, one sentence — the simplest path.
    result = router.classify(args.text, args.engine)
    _print_result(args.engine or get_settings().default_engine, result)
    return 0


def _cmd_compare(router: IntentRouter, args: argparse.Namespace) -> int:
    """Handle the ``compare`` sub-command.

    Parameters
    ----------
    router : IntentRouter
        The router bound to the knowledge base.
    args : argparse.Namespace
        Parsed CLI arguments (``text``).

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    # Run everything available and print one block per engine — this is the
    # money shot for the "three approaches" demo.
    print(f'Phrase : "{args.text}"\n')
    for name, result in router.compare(args.text).items():
        _print_result(name, result)
    return 0


def _cmd_execute(router: IntentRouter, args: argparse.Namespace) -> int:
    """Handle the ``execute`` sub-command.

    Parameters
    ----------
    router : IntentRouter
        The router bound to the knowledge base.
    args : argparse.Namespace
        Parsed CLI arguments (``text``, ``engine``).

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    # Classify then act: show the concrete routing decision a downstream
    # system would receive.
    execution = router.execute(args.text, args.engine)
    if execution.handoff:
        # No confident intent: this is where a real IVR would transfer to a
        # human agent.
        print("→ Transfert vers un conseiller humain (intention incertaine).")
        return 0
    # Confident: print the routing action, target service and any slots.
    print(f"→ Intention : {execution.title} ({execution.intent_id})")
    print(f"→ Service   : {execution.service}")
    print(f"→ Action    : {execution.action}")
    if execution.slots:
        print(f"→ Slots     : {execution.slots}")
    print(f"\n{execution.message}")
    return 0


def _cmd_intents(router: IntentRouter, args: argparse.Namespace) -> int:
    """Handle the ``intents`` sub-command (list the catalogue).

    Parameters
    ----------
    router : IntentRouter
        The router bound to the knowledge base.
    args : argparse.Namespace
        Parsed CLI arguments (unused).

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    # A quick inventory of what the KB currently teaches the engines.
    print(f"{len(router.kb)} intentions dans la base de connaissance :\n")
    for intent in router.kb.intents:
        # ``len(examples)`` hints at how much training signal each intent has.
        print(
            f"  {intent.intent_id:<32} {intent.title}  "
            f"({len(intent.examples)} exemples)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with its sub-commands.

    Returns
    -------
    argparse.ArgumentParser
        The fully configured parser.

    Examples
    --------
    >>> parser = build_parser()
    >>> parser.prog is not None
    True
    """
    parser = argparse.ArgumentParser(
        prog="intent_engine",
        description=(
            "Déraison Assurances — moteur d'intentions en 3 approches "
            "(TF-IDF, BERT, LLM)."
        ),
    )
    # ``--kb`` lets a user point at an alternative knowledge base folder.
    parser.add_argument(
        "--kb",
        default=str(get_settings().knowledge_base_dir),
        help="Dossier de la base de connaissance Markdown.",
    )
    # Sub-commands are mutually exclusive verbs; ``required`` forces one.
    sub = parser.add_subparsers(dest="command", required=True)

    # ``classify`` — single engine.
    p_classify = sub.add_parser("classify", help="Classer avec un moteur.")
    p_classify.add_argument("text", help="La phrase du client.")
    p_classify.add_argument("--engine", choices=["tfidf", "bert", "llm"], default=None)
    p_classify.set_defaults(func=_cmd_classify)

    # ``compare`` — all engines.
    p_compare = sub.add_parser("compare", help="Comparer les 3 moteurs.")
    p_compare.add_argument("text", help="La phrase du client.")
    p_compare.set_defaults(func=_cmd_compare)

    # ``execute`` — classify + act.
    p_execute = sub.add_parser("execute", help="Exécuter la requête.")
    p_execute.add_argument("text", help="La requête en langage naturel.")
    p_execute.add_argument("--engine", choices=["tfidf", "bert", "llm"], default=None)
    p_execute.set_defaults(func=_cmd_execute)

    # ``intents`` — list the KB.
    p_intents = sub.add_parser("intents", help="Lister les intentions.")
    p_intents.set_defaults(func=_cmd_intents)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument vector (defaults to ``sys.argv[1:]``), injectable for tests.

    Returns
    -------
    int
        Process exit code.

    Examples
    --------
    >>> build_parser() is not None
    True
    """
    # Configure logging once, at the entry point, so library modules that use
    # ``logging`` produce visible output without each configuring it.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    # Build the router against the chosen KB folder, then dispatch to the
    # sub-command handler attached via ``set_defaults(func=...)``.
    router = IntentRouter.from_directory(args.kb)
    return args.func(router, args)


# Standard ``python -m intent_engine`` / ``python cli.py`` entry guard.
if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
