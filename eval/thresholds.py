"""Versioned evaluation thresholds and dataset loading.

Module summary
--------------
Single source of truth for the pass/fail bars the evaluation gates on, plus
the loader for the committed labelled dataset. Keeping the thresholds here
(not scattered in test files) makes them reviewable in one diff when we
tighten or relax a bar, which is exactly what "versioned thresholds" in the
coding standard asks for.

The bars are intentionally conservative for a tiny demo KB and per engine,
because the three approaches have genuinely different accuracy/latency
profiles (see ``PROS_CONS.md``):

* TF-IDF is fast but surface-level → a modest accuracy bar.
* BERT generalises to paraphrases → a higher accuracy bar.
* LLM is strong but slow → high accuracy, a generous latency ceiling.

Author
------
Project maintainers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

# Absolute path to the committed datasets, resolved relative to this file so
# the loaders work regardless of the current working directory.
_DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
# Out-of-scope utterances: the engine should *abstain* (hand off to a human)
# rather than route these anywhere. Loaded by :func:`load_oos_dataset`.
_OOS_DATASET_PATH = Path(__file__).resolve().parent / "dataset_oos.jsonl"

# Minimum fraction of out-of-scope inputs an engine must correctly abstain on.
# Kept modest for the tiny toy KB: with only 21 classes the softmax mass does
# not always fall under the floor, so we require a majority, not perfection.
MIN_ABSTENTION_RATE: dict[str, float] = {
    "tfidf": 0.60,
    "fasttext_custom": 0.30,
    "fasttext_pretrained": 0.30,
    "bert": 0.55,
    "llm": 0.75,
}


class EvalCase(TypedDict):
    """One labelled evaluation example.

    Parameters
    ----------
    text : str
        The user utterance to classify.
    expected : str
        The gold intent id the engine should predict.
    """

    # The input sentence.
    text: str
    # The gold-standard intent id.
    expected: str


class EngineThreshold(TypedDict):
    """Pass/fail bars for one engine.

    Parameters
    ----------
    min_accuracy : float
        Minimum top-1 accuracy on the dataset, in ``[0, 1]``.
    max_latency_ms : float
        Maximum acceptable mean per-utterance latency, in milliseconds.
    """

    # Lower bound on correctness.
    min_accuracy: float
    # Upper bound on mean latency (the LLM gets a generous ceiling).
    max_latency_ms: float


# Versioned thresholds, keyed by engine name. Bump these deliberately in a
# reviewed commit; a drop below them fails CI exactly like a broken test.
# The held-out test set is paraphrase-heavy (low lexical overlap), so these
# reflect *generalisation*, not vocabulary memorisation — expect the lexical
# engines to sit lower than the semantic ones here.
THRESHOLDS: dict[str, EngineThreshold] = {
    # TF-IDF + Random Forest: surface-level; on a paraphrase test set it is
    # the weakest, so a modest bar.
    "tfidf": {"min_accuracy": 0.45, "max_latency_ms": 200.0},
    # fastText supervised (learned subword embeddings on our examples):
    # a step up from pure bag-of-words.
    "fasttext_custom": {"min_accuracy": 0.55, "max_latency_ms": 200.0},
    # fastText pretrained (cc.fr.300): transfer learning from Common Crawl,
    # should generalise better to unseen phrasings.
    "fasttext_pretrained": {"min_accuracy": 0.65, "max_latency_ms": 500.0},
    # BERT (SBERT + MLP): the strongest non-generative approach on paraphrases.
    "bert": {"min_accuracy": 0.75, "max_latency_ms": 2000.0},
    # LLM: high accuracy, but slow — seconds per call on a local model.
    "llm": {"min_accuracy": 0.80, "max_latency_ms": 40000.0},
}


def load_dataset() -> list[EvalCase]:
    """Load the committed JSONL evaluation dataset.

    Returns
    -------
    list[EvalCase]
        The labelled examples, one per non-empty line.

    Examples
    --------
    >>> cases = load_dataset()
    >>> {"text", "expected"} <= set(cases[0])
    True
    """
    cases: list[EvalCase] = []
    # Read line by line; JSONL keeps the dataset diff-friendly (one example
    # per line) and streamable if it ever grows large.
    for line in _DATASET_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        # Skip blank lines so a trailing newline never produces an empty case.
        if not stripped:
            continue
        row = json.loads(stripped)
        cases.append({"text": row["text"], "expected": row["expected"]})
    return cases


def load_oos_dataset() -> list[str]:
    """Load the out-of-scope utterances the engines should abstain on.

    Returns
    -------
    list[str]
        The off-topic sentences (no gold label — the only correct behaviour
        is abstention).

    Examples
    --------
    >>> texts = load_oos_dataset()
    >>> all(isinstance(t, str) for t in texts)
    True
    """
    texts: list[str] = []
    # Same JSONL, one ``{"text": ...}`` per line; no ``expected`` because the
    # expected outcome is "no confident intent".
    for line in _OOS_DATASET_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        texts.append(json.loads(stripped)["text"])
    return texts
