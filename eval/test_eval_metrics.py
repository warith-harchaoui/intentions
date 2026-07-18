"""CI gate: accuracy + latency thresholds via the dependency-free harness.

This is the always-runnable evaluation gate. The TF-IDF check needs no
heavy dependency and runs in the fast lane; the BERT and LLM checks are
marked ``slow`` because they need an embedding backend (SBERT or Ollama) or
the Ollama LLM. A drop below the versioned thresholds fails CI exactly like
a broken unit test (coding standard rule 14: CI gating).

Author
------
Project maintainers.
"""

from __future__ import annotations

import pytest

from .harness import abstention_rate, evaluate_engine
from .thresholds import MIN_ABSTENTION_RATE


def test_tfidf_meets_thresholds() -> None:
    """TF-IDF must hit its accuracy and latency bars on the dataset."""
    # Fast and dependency-free: safe for the default CI lane.
    report = evaluate_engine("tfidf")
    # ``passed`` bundles both the accuracy and latency criteria.
    assert report.passed(), (
        f"TF-IDF below threshold: acc={report.accuracy:.0%}, "
        f"lat={report.mean_latency_ms:.0f}ms, misses={report.failures}"
    )


def test_tfidf_abstains_on_out_of_scope() -> None:
    """TF-IDF must abstain on most off-topic inputs (the safety net)."""
    # Fast, dependency-free: an engine that confidently routes "recette de
    # tarte" would send callers to the wrong service — gate against that.
    rate = abstention_rate("tfidf")
    assert rate >= MIN_ABSTENTION_RATE["tfidf"], (
        f"TF-IDF abstention rate {rate:.0%} below {MIN_ABSTENTION_RATE['tfidf']:.0%}"
    )


@pytest.mark.slow
def test_bert_meets_thresholds() -> None:
    """BERT must hit its bars (needs SBERT or the Ollama embedder).

    Marked ``slow``: the default ``auto`` backend downloads an SBERT model
    or calls Ollama for embeddings. Install ``.[sbert]`` for the higher-
    accuracy path that clears the 0.80 bar.
    """
    report = evaluate_engine("bert")
    assert report.passed(), (
        f"BERT below threshold: acc={report.accuracy:.0%}, "
        f"lat={report.mean_latency_ms:.0f}ms, misses={report.failures}"
    )


@pytest.mark.slow
def test_llm_meets_thresholds() -> None:
    """LLM must hit its bars (needs a running Ollama with the model).

    Marked ``slow``: dozens of seconds of local inference. Skipped by the
    fast lane; run explicitly with ``pytest -m slow eval/``.
    """
    report = evaluate_engine("llm")
    assert report.passed(), (
        f"LLM below threshold: acc={report.accuracy:.0%}, "
        f"lat={report.mean_latency_ms:.0f}ms, misses={report.failures}"
    )
