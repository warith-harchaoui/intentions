"""Dependency-free accuracy + latency harness over the eval dataset.

Module summary
--------------
The always-runnable core of the evaluation layer: no DeepEval, no Giskard,
no network beyond whatever the chosen engine already uses. It runs every
labelled example through an engine, computes top-1 accuracy and mean
latency, and compares them to the versioned thresholds. Both the CI gate
(``eval/test_eval_metrics.py``) and the ``python -m eval.harness`` CLI build
on it.

Usage example
-------------
>>> from eval.harness import evaluate_engine
>>> report = evaluate_engine("tfidf")   # doctest: +SKIP
>>> 0.0 <= report.accuracy <= 1.0       # doctest: +SKIP
True

Author
------
Project maintainers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from intent_engine.config import get_settings
from intent_engine.router import IntentRouter

from .thresholds import (
    MIN_ABSTENTION_RATE,
    THRESHOLDS,
    EvalCase,
    load_dataset,
    load_oos_dataset,
)

# Module logger — the CLI section prints; the library section does not.
logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Aggregate metrics for one engine over the dataset.

    Parameters
    ----------
    engine : str
        Engine name evaluated.
    total : int
        Number of examples scored.
    correct : int
        Number of top-1 correct predictions.
    accuracy : float
        ``correct / total`` (0 when ``total`` is 0).
    mean_latency_ms : float
        Mean per-utterance latency in milliseconds.
    failures : list[tuple[str, str, str]]
        ``(text, expected, predicted)`` triples for the misclassified
        examples, for eyeball error analysis.
    """

    # Which engine these numbers describe.
    engine: str
    # Denominator and numerator of the accuracy.
    total: int = 0
    correct: int = 0
    # Derived headline metrics.
    accuracy: float = 0.0
    mean_latency_ms: float = 0.0
    # Concrete misses, so a failing gate is actionable, not just a number.
    failures: list[tuple[str, str, str]] = field(default_factory=list)

    def passed(self) -> bool:
        """Whether this report meets its engine's versioned thresholds.

        Returns
        -------
        bool
            ``True`` if accuracy and latency both satisfy the bar.
        """
        # An unknown engine has no bar; treat that as a failure to be safe.
        bar = THRESHOLDS.get(self.engine)
        if bar is None:
            return False
        # Both criteria must hold: accurate enough AND fast enough.
        return (
            self.accuracy >= bar["min_accuracy"]
            and self.mean_latency_ms <= bar["max_latency_ms"]
        )


def evaluate_engine(engine: str, router: IntentRouter | None = None) -> EvalReport:
    """Run every dataset example through one engine and aggregate metrics.

    Parameters
    ----------
    engine : str
        Engine name (``"tfidf"``, ``"bert"``, ``"llm"``).
    router : IntentRouter | None, optional
        A pre-built router (to share the parsed KB / fitted engines across
        several calls). When ``None``, one is built from the configured KB.

    Returns
    -------
    EvalReport
        Accuracy, mean latency and the list of misclassifications.
    """
    # Build a router lazily if the caller did not supply one. Reusing a
    # router across engines avoids re-parsing the KB once per engine.
    if router is None:
        router = IntentRouter.from_directory(get_settings().knowledge_base_dir)
    cases: list[EvalCase] = load_dataset()

    correct = 0
    total_latency = 0.0
    failures: list[tuple[str, str, str]] = []
    # Score each example: predict, compare the top intent to the gold label,
    # and accumulate latency for the mean.
    for case in cases:
        result = router.classify(case["text"], engine)
        top = result.top()
        # ``predicted`` is the empty string on abstention, which counts as a
        # miss against a gold label (the dataset has no out-of-scope rows).
        predicted = top.intent if top is not None else ""
        total_latency += result.latency_ms
        if predicted == case["expected"]:
            correct += 1
        else:
            failures.append((case["text"], case["expected"], predicted))

    total = len(cases)
    # Guard the zero-example case so the harness never divides by zero.
    accuracy = correct / total if total else 0.0
    mean_latency = total_latency / total if total else 0.0
    return EvalReport(
        engine=engine,
        total=total,
        correct=correct,
        accuracy=accuracy,
        mean_latency_ms=mean_latency,
        failures=failures,
    )


def abstention_rate(engine: str, router: IntentRouter | None = None) -> float:
    """Fraction of out-of-scope inputs the engine correctly abstains on.

    An intent engine that confidently routes "quelle est la capitale de
    l'Australie" is dangerous: it would send a caller to the wrong service.
    This measures the safety net — how often the engine says "je ne sais
    pas" (``result.confident is False``) on off-topic input.

    Parameters
    ----------
    engine : str
        Engine name to evaluate.
    router : IntentRouter | None, optional
        A pre-built router; one is created from config when ``None``.

    Returns
    -------
    float
        Abstention rate in ``[0, 1]`` over the out-of-scope dataset.
    """
    # Reuse a shared router when provided to avoid re-parsing the KB.
    if router is None:
        router = IntentRouter.from_directory(get_settings().knowledge_base_dir)
    texts = load_oos_dataset()
    if not texts:
        return 0.0
    # Count how many off-topic inputs the engine (correctly) refuses to route.
    abstained = 0
    for text in texts:
        result = router.classify(text, engine)
        # ``confident is False`` is the abstention signal shared by all engines.
        if not result.confident:
            abstained += 1
    return abstained / len(texts)


def _print_report(report: EvalReport) -> None:
    """Print a human-readable summary of one engine's report to stdout.

    Parameters
    ----------
    report : EvalReport
        The report to render.

    Notes
    -----
    This is a CLI helper, the sanctioned place for ``print`` per rule 6.
    """
    # Headline line: accuracy, latency, and pass/fail against the bar.
    verdict = "PASS" if report.passed() else "FAIL"
    bar = THRESHOLDS.get(report.engine, {})
    print(
        f"[{verdict}] {report.engine:<6} "
        f"accuracy={report.accuracy:.0%} "
        f"({report.correct}/{report.total}) "
        f"mean_latency={report.mean_latency_ms:.0f}ms "
        f"(bars: acc≥{bar.get('min_accuracy', 0):.0%}, "
        f"lat≤{bar.get('max_latency_ms', 0):.0f}ms)"
    )
    # List the misses so a failing run is immediately actionable.
    for text, expected, predicted in report.failures:
        shown = predicted or "(abstention)"
        print(f"    ✗ {text!r} → {shown} (attendu: {expected})")


def main(argv: list[str] | None = None) -> int:
    """CLI entry: evaluate one or all engines and print a report.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` if every evaluated engine passed its thresholds, else ``1``.
    """
    import argparse

    # Configure logging once at the entry point.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="eval.harness",
        description="Évalue l'exactitude et la latence des moteurs.",
    )
    # ``--engine`` restricts the run; omit it to evaluate all engines.
    parser.add_argument(
        "--engine",
        choices=["tfidf", "fasttext_custom", "fasttext_pretrained", "bert", "llm"],
        default=None,
        help="Moteur à évaluer (défaut : tous).",
    )
    args = parser.parse_args(argv)

    # Build one router and reuse it so the KB is parsed a single time.
    router = IntentRouter.from_directory(get_settings().knowledge_base_dir)
    engines = (
        [args.engine]
        if args.engine
        else ["tfidf", "fasttext_custom", "fasttext_pretrained", "bert", "llm"]
    )

    all_passed = True
    # Evaluate each requested engine and track the overall verdict.
    for engine in engines:
        report = evaluate_engine(engine, router=router)
        _print_report(report)
        # Also report the out-of-scope abstention rate — the safety net that
        # keeps the engine from confidently routing off-topic requests.
        rate = abstention_rate(engine, router=router)
        floor = MIN_ABSTENTION_RATE.get(engine, 0.0)
        ok = "PASS" if rate >= floor else "FAIL"
        print(f"    [{ok}] abstention hors-périmètre: {rate:.0%} (bar ≥{floor:.0%})")
        all_passed = all_passed and report.passed() and rate >= floor
    # Non-zero exit on any failure so CI can gate on it.
    return 0 if all_passed else 1


# Standard ``python -m eval.harness`` entry guard.
if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
