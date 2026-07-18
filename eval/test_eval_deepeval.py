"""DeepEval evaluation of the LLM intent engine (coding standard rule 14).

Module summary
--------------
Uses `DeepEval <https://github.com/confident-ai/deepeval>`_ to structure the
LLM engine's evaluation as test cases with an explicit metric. The metric
here is a **deterministic, no-API-key** custom metric: it scores 1.0 when
the engine's predicted intent matches the gold label, 0.0 otherwise. We
deliberately avoid DeepEval's LLM-as-judge metrics so the gate needs no
external API and stays cheap and reproducible.

Cost control (rule 14): only a small, fixed subset of the dataset is sent
to the local model, and the whole module is marked ``slow`` so it never
runs in the fast lane.

Run it with:
    pip install ".[eval]"
    pytest -m slow eval/test_eval_deepeval.py

Author
------
Project maintainers.
"""

from __future__ import annotations

import pytest

# DeepEval is an optional, heavy dependency. Skip the whole module cleanly
# when it (or the eval extra) is not installed, so the default suite is
# unaffected.
deepeval = pytest.importorskip("deepeval")
from deepeval.metrics import BaseMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from intent_engine.config import get_settings  # noqa: E402
from intent_engine.router import IntentRouter  # noqa: E402

from .thresholds import THRESHOLDS, load_dataset  # noqa: E402

# Cap the number of local-LLM calls to keep the eval fast and cheap. A dozen
# cases is enough to catch a regression without a multi-minute run.
_MAX_CASES = 12


class IntentMatchMetric(BaseMetric):
    """Deterministic exact-intent-match metric (no LLM judge, no API key).

    The metric compares the intent id the engine predicted (carried in the
    test case's ``actual_output``) with the gold id (``expected_output``).
    This keeps the DeepEval gate free, offline and reproducible.
    """

    def __init__(self) -> None:
        """Initialise with the LLM engine's accuracy bar as the threshold."""
        # Reuse the versioned bar so DeepEval and the harness agree on what
        # "good enough" means for the LLM engine.
        self.threshold: float = THRESHOLDS["llm"]["min_accuracy"]
        # DeepEval reads these attributes after ``measure``.
        self.score: float = 0.0
        self.success: bool = False

    def measure(self, test_case: LLMTestCase) -> float:
        """Score one test case 1.0 on an exact intent match, else 0.0.

        Parameters
        ----------
        test_case : LLMTestCase
            Case whose ``actual_output`` is the predicted intent id and
            ``expected_output`` is the gold id.

        Returns
        -------
        float
            ``1.0`` if the ids match, ``0.0`` otherwise.
        """
        # Exact string match on the intent id — the only thing that matters
        # for routing correctness.
        self.score = (
            1.0 if test_case.actual_output == test_case.expected_output else 0.0
        )
        # Per-case success is the score itself (binary here).
        self.success = self.score >= 1.0
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async shim required by DeepEval; delegates to :meth:`measure`."""
        # No real async work; just forward so DeepEval's async paths work.
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Return whether the last measured case passed."""
        return self.success

    @property
    def __name__(self) -> str:
        """Human-readable metric name shown in DeepEval reports."""
        return "IntentMatch"


@pytest.mark.slow
def test_llm_intent_accuracy_with_deepeval() -> None:
    """Aggregate exact-match accuracy over a subset must clear the bar."""
    # Build the router once and take a bounded slice of the dataset.
    router = IntentRouter.from_directory(get_settings().knowledge_base_dir)
    cases = load_dataset()[:_MAX_CASES]

    metric = IntentMatchMetric()
    scores: list[float] = []
    # Turn each labelled example into a DeepEval test case: run the engine,
    # record predicted vs expected, and score with our deterministic metric.
    for case in cases:
        result = router.classify(case["text"], "llm")
        top = result.top()
        predicted = top.intent if top is not None else ""
        test_case = LLMTestCase(
            input=case["text"],
            actual_output=predicted,
            expected_output=case["expected"],
        )
        scores.append(metric.measure(test_case))

    # Aggregate accuracy across the subset and gate on the versioned bar.
    accuracy = sum(scores) / len(scores) if scores else 0.0
    assert accuracy >= metric.threshold, (
        f"LLM intent accuracy {accuracy:.0%} below bar {metric.threshold:.0%}"
    )
