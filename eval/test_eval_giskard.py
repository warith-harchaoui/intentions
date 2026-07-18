"""Giskard scan of the TF-IDF intent classifier (coding standard rule 14).

Module summary
--------------
Uses `Giskard <https://github.com/Giskard-AI/giskard>`_ to wrap the TF-IDF
engine as a classification model and scan it for ML issues (robustness to
perturbations, performance bias across slices, overconfidence). Giskard's
value here is the *vulnerability scan* that plain accuracy tests miss —
e.g. whether adding a typo or switching to uppercase flips the prediction,
a known weakness of surface-level TF-IDF worth surfacing explicitly.

The module is marked ``slow`` and skips cleanly when Giskard (the ``eval``
extra) is not installed, so the default test suite is unaffected.

.. note::
   **Python version.** Giskard currently ships wheels for Python 3.10–3.11
   only. On newer interpreters (e.g. 3.13, this repo's dev machine) ``pip
   install giskard`` fails with *"No matching distribution found"*, so this
   scan is **written but not executed here** — the ``importorskip`` below
   turns that into a clean skip. Run it on a Python 3.11 environment to
   exercise it for real. (DeepEval, by contrast, installs on 3.13 and the
   LLM eval in ``test_eval_deepeval.py`` was run end-to-end.)

Run it with (Python ≤ 3.11):
    pip install ".[eval]"
    pytest -m slow eval/test_eval_giskard.py

Author
------
Project maintainers.
"""

from __future__ import annotations

import pytest

# Giskard and pandas are optional, heavy dependencies. Skip the module when
# absent so the default suite never requires them.
giskard = pytest.importorskip("giskard")
pd = pytest.importorskip("pandas")

from intent_engine.config import get_settings  # noqa: E402
from intent_engine.router import IntentRouter  # noqa: E402

from .thresholds import load_dataset  # noqa: E402


@pytest.mark.slow
def test_tfidf_giskard_scan_has_no_critical_issues() -> None:
    """A Giskard scan of the TF-IDF classifier finds no critical issue."""
    # Build the router and fit the TF-IDF engine against the real KB.
    router = IntentRouter.from_directory(get_settings().knowledge_base_dir)
    engine = router.engine("tfidf")
    # The full set of intent ids is the classifier's label space; Giskard
    # needs it to interpret the prediction columns.
    labels = router.kb.intent_ids()

    def _predict(df: pd.DataFrame) -> "list[list[float]]":
        """Return per-row class-probability vectors for Giskard.

        Parameters
        ----------
        df : pd.DataFrame
            A frame with a ``text`` column of utterances.

        Returns
        -------
        list[list[float]]
            One probability row per input, aligned to ``labels``.
        """
        rows: list[list[float]] = []
        # Giskard passes a batch as a DataFrame; classify each row and
        # project the ranked predictions back onto the full label vector.
        for text in df["text"].tolist():
            result = engine.classify(text, top_k=len(labels))
            # Start from zeros so unranked classes get probability 0.
            scores = {label: 0.0 for label in labels}
            for prediction in result.ranked:
                scores[prediction.intent] = prediction.score
            rows.append([scores[label] for label in labels])
        return rows

    # Wrap the prediction function as a Giskard classification model.
    model = giskard.Model(
        model=_predict,
        model_type="classification",
        classification_labels=labels,
        feature_names=["text"],
        name="tfidf-intent-classifier",
    )
    # Build the evaluation dataset Giskard perturbs and slices.
    frame = pd.DataFrame(
        [{"text": c["text"], "expected": c["expected"]} for c in load_dataset()]
    )
    dataset = giskard.Dataset(frame, target="expected", name="intent-eval")

    # Run the scan. It probes robustness (typos, casing), calibration and
    # data-slice performance — exactly the failure modes TF-IDF is prone to.
    report = giskard.scan(model, dataset)
    # Gate on the absence of CRITICAL-severity issues; lower-severity
    # findings are informative but non-blocking for a teaching demo.
    critical = [
        issue for issue in report.issues if getattr(issue, "level", "") == "major"
    ]
    assert not critical, f"Giskard found critical issues: {critical}"
