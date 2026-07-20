"""Statistical evaluation base: bootstrap CIs and k-fold cross-validation.

Module summary
--------------
A single accuracy number (97 % vs 82 % vs 94 %) is not a fair verdict on a
33-example test set — it hides the sampling uncertainty. This module builds
a **real test base** that reports *distributions*, not point estimates, so
we can honestly say whether the engines are actually different or just
noise apart on a small set.

Two complementary methods:

* **Bootstrap on the held-out set** (all engines). Each engine is run
  once over the labelled eval set (``dataset.jsonl``); the per-example
  correctness vector is then resampled with replacement ``B`` times to get
  the sampling distribution of accuracy — mean, standard deviation and a
  95 % percentile interval. This is the only method that includes the
  zero-shot LLM fairly and cheaply (it is classified once, then resampled).
  The LLM predictions are cached on disk so re-runs cost nothing.

* **Repeated stratified k-fold cross-validation on the KB** (the two
  *trainable* engines, TF-IDF and BERT). The engine is retrained on each
  train fold and scored on the held-out fold, giving a distribution of
  fold accuracies that measures train/test robustness. The LLM does not
  train, so it is excluded here.

Both write to ``eval/crossval_results.json`` (consumed by
``eval/violin.py`` to render the violin plot).

Usage example
-------------
>>> from eval.crossval import bootstrap_accuracy
>>> import numpy as np
>>> dist = bootstrap_accuracy(np.array([1, 1, 0, 1]), n_boot=100, seed=0)
>>> 0.0 <= dist.mean() <= 1.0
True

Author
------
Project maintainers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from intent_engine.config import get_settings
from intent_engine.kb import KnowledgeBase
from intent_engine.router import IntentRouter
from intent_engine.tfidf_engine import TfidfIntentEngine

from .thresholds import load_dataset

logger = logging.getLogger(__name__)

# Reproducibility: every random draw (bootstrap, fold shuffling) is seeded
# from this so the reported numbers are identical run to run.
_SEED = 20260718

# Where per-engine results land, and where LLM predictions are cached so the
# expensive ~20 s/call model is queried at most once per utterance.
_RESULTS_PATH = Path(__file__).resolve().parent / "crossval_results.json"
_LLM_CACHE_PATH = Path(__file__).resolve().parent / ".llm_cache.json"


def bootstrap_accuracy(
    correct: np.ndarray, n_boot: int = 2000, seed: int = _SEED
) -> np.ndarray:
    """Bootstrap the accuracy distribution from a correctness vector.

    Parameters
    ----------
    correct : np.ndarray
        1-D array of 0/1 flags, one per test example (1 == predicted the
        gold intent).
    n_boot : int, optional
        Number of bootstrap resamples, by default 2000.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        ``n_boot`` bootstrap accuracies in ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> d = bootstrap_accuracy(np.array([1, 0, 1, 1]), n_boot=50, seed=1)
    >>> d.shape
    (50,)
    """
    rng = np.random.default_rng(seed)
    n = len(correct)
    # Resample indices with replacement, ``n_boot`` times at once, then take
    # the mean correctness of each resample — that is one bootstrap accuracy.
    # Vectorised so 2000 resamples of a tiny vector cost microseconds.
    idx = rng.integers(0, n, size=(n_boot, n))
    return correct[idx].mean(axis=1)


def _engine_correctness(
    router: IntentRouter, engine: str, use_cache: bool
) -> np.ndarray:
    """Run one engine over the eval set and return its 0/1 correctness vector.

    Parameters
    ----------
    router : IntentRouter
        The router holding the fitted engines.
    engine : str
        Engine name to score.
    use_cache : bool
        When ``True`` and ``engine == "llm"``, reuse/refresh an on-disk
        prediction cache so the slow model is queried at most once per text.

    Returns
    -------
    np.ndarray
        0/1 flags aligned with the eval dataset order.
    """
    cases = load_dataset()
    # The LLM is slow (~20 s/call); cache its predictions keyed by utterance
    # so repeated experiments (bootstrap needs the vector only once, but the
    # user may re-run) never pay the cost twice.
    cache: dict[str, str] = {}
    if engine == "llm" and use_cache and _LLM_CACHE_PATH.exists():
        cache = json.loads(_LLM_CACHE_PATH.read_text(encoding="utf-8"))

    flags: list[int] = []
    # Classify each example; a correct top-1 intent scores 1, anything else
    # (wrong intent or abstention) scores 0 — the same rule the harness uses.
    for case in cases:
        text = case["text"]
        if engine == "llm" and text in cache:
            predicted = cache[text]
        else:
            result = router.classify(text, engine)
            top = result.top()
            predicted = top.intent if top is not None else ""
            # Populate the cache for the LLM as we go, and persist it
            # **incrementally** after each call: the LLM bootstrap is ~20 min
            # and memory-heavy, so a crash mid-way must not lose progress —
            # a re-run then resumes from the on-disk cache.
            if engine == "llm" and use_cache:
                cache[text] = predicted
                _LLM_CACHE_PATH.write_text(
                    json.dumps(cache, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        flags.append(1 if predicted == case["expected"] else 0)
    return np.asarray(flags, dtype=float)


# The trainable engines that get a k-fold CV distribution (the LLM is
# zero-shot, so it has no training/CV — it only appears in the bootstrap).
_CV_ENGINES = ("tfidf", "fasttext_custom", "fasttext_pretrained", "bert")


def cross_validate_engine(
    engine_name: str, kb: KnowledgeBase, n_splits: int = 5, n_repeats: int = 5
) -> np.ndarray:
    """Repeated stratified k-fold CV accuracy for a *trainable* engine.

    The classifier is retrained on each train fold and scored on the held-out
    fold, measuring how stable its accuracy is across true train/test splits
    of the KB examples — the honest generalisation picture a single fixed
    split hides.

    For the two embedding-based engines (``fasttext_pretrained``, ``bert``)
    the heavy backbone is loaded/applied **once** to embed every example, and
    only the light classifier head (logistic regression / MLP) is refit per
    fold — otherwise reloading a multi-GB model 25 times would be absurd. The
    text-based engines (``tfidf``, ``fasttext_custom``) are refit whole per
    fold since their vectoriser must see raw text.

    Parameters
    ----------
    engine_name : str
        One of :data:`_CV_ENGINES`.
    kb : KnowledgeBase
        Knowledge base whose examples provide ``(texts, labels)``.
    n_splits : int, optional
        Folds per repeat, by default 5 (min class count is 6, so 5 is safe).
    n_repeats : int, optional
        Number of shuffled repeats, by default 5 (→ 25 fold scores).

    Returns
    -------
    np.ndarray
        One accuracy per fold (``n_splits * n_repeats`` values).
    """
    texts, labels = kb.training_pairs()
    x = np.asarray(texts, dtype=object)
    y = np.asarray(labels, dtype=object)
    # Stratified so every fold keeps each intent's proportion; repeated with a
    # fixed seed so the 25 splits are reproducible.
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=_SEED
    )

    # Text-based engines: refit the whole engine per fold on the fold's texts.
    if engine_name in ("tfidf", "fasttext_custom"):
        return _cv_text_engine(engine_name, x, y, splitter)
    # Embedding-based engines: embed once, then CV the light head on vectors.
    if engine_name in ("fasttext_pretrained", "bert"):
        return _cv_embedding_engine(engine_name, x, y, splitter)
    raise ValueError(f"cross_validate_engine: engine {engine_name!r} is not trainable")


def _cv_text_engine(
    engine_name: str, x: np.ndarray, y: np.ndarray, splitter
) -> np.ndarray:
    """CV a text-based engine by refitting it whole on each fold.

    Parameters
    ----------
    engine_name : str
        ``"tfidf"`` or ``"fasttext_custom"``.
    x : np.ndarray
        Object array of utterance strings.
    y : np.ndarray
        Object array of intent-id labels.
    splitter : RepeatedStratifiedKFold
        The configured fold generator.

    Returns
    -------
    np.ndarray
        One accuracy per fold.
    """
    from intent_engine.fasttext_engine import FastTextSupervisedEngine

    scores: list[float] = []
    # Each split gives disjoint train/test indices over the KB examples.
    for train_idx, test_idx in splitter.split(x, y):
        # Build a throwaway KB from the training slice so we reuse the engine's
        # normal ``fit(kb)`` path unchanged.
        train_kb = _kb_from_pairs(x[train_idx].tolist(), y[train_idx].tolist())
        if engine_name == "tfidf":
            engine = TfidfIntentEngine().fit(train_kb)
        else:
            engine = FastTextSupervisedEngine().fit(train_kb)
        # Score the held-out fold via the engine's own classify path.
        correct = sum(
            1
            for text, gold in zip(x[test_idx], y[test_idx])
            if (top := engine.classify(text).top()) is not None and top.intent == gold
        )
        scores.append(correct / len(test_idx))
    return np.asarray(scores, dtype=float)


def _cv_embedding_engine(
    engine_name: str, x: np.ndarray, y: np.ndarray, splitter
) -> np.ndarray:
    """CV an embedding-based engine by embedding once, refitting the head.

    Parameters
    ----------
    engine_name : str
        ``"fasttext_pretrained"`` or ``"bert"``.
    x : np.ndarray
        Object array of utterance strings.
    y : np.ndarray
        Object array of intent-id labels.
    splitter : RepeatedStratifiedKFold
        The configured fold generator.

    Returns
    -------
    np.ndarray
        One accuracy per fold.
    """
    # Embed every utterance ONCE with the (heavy) backbone, then only the
    # light head is refit per fold.
    features = _embed_all(engine_name, x.tolist())

    scores: list[float] = []
    for train_idx, test_idx in splitter.split(x, y):
        # A fresh head each fold: logistic regression for fastText vectors,
        # the PyTorch MLP for SBERT (matching each engine's real classifier).
        head = _make_head(engine_name)
        head.fit(features[train_idx], y[train_idx])
        proba = head.predict_proba(features[test_idx])
        predicted = head.classes_[proba.argmax(axis=1)]
        scores.append(float((predicted == y[test_idx]).mean()))
    return np.asarray(scores, dtype=float)


def _embed_all(engine_name: str, texts: list[str]) -> np.ndarray:
    """Embed all texts once with the engine's pretrained backbone.

    Parameters
    ----------
    engine_name : str
        ``"fasttext_pretrained"`` or ``"bert"``.
    texts : list[str]
        Utterances to embed.

    Returns
    -------
    np.ndarray
        ``(n, dim)`` embedding matrix.
    """
    if engine_name == "bert":
        # SBERT (or the Ollama fallback) sentence embeddings.
        from intent_engine.embeddings import build_embedder

        return build_embedder().encode(texts)
    # fastText pretrained: reuse the engine's own loaded vectors + normaliser.
    from intent_engine.fasttext_engine import FastTextPretrainedEngine

    engine = FastTextPretrainedEngine()
    # Load the big model once via a throwaway 2-intent fit is wasteful; instead
    # load vectors directly and embed. We borrow the engine's private embedder.
    import fasttext

    engine._vectors = fasttext.load_model(str(engine._model_path))
    return engine._embed(texts)


def _make_head(engine_name: str):
    """Return a fresh classifier head matching the engine's real classifier.

    Parameters
    ----------
    engine_name : str
        ``"fasttext_pretrained"`` (logistic regression) or ``"bert"`` (MLP).

    Returns
    -------
    object
        An estimator exposing ``fit`` / ``predict_proba`` / ``classes_``.
    """
    if engine_name == "bert":
        from intent_engine.mlp import TorchMLPClassifier

        return TorchMLPClassifier()
    # fastText pretrained uses a classic logistic regression on the vectors.
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=10.0, max_iter=1000)


def _kb_from_pairs(texts: list[str], labels: list[str]) -> KnowledgeBase:
    """Build a minimal in-memory KB from ``(texts, labels)`` for a CV fold.

    Parameters
    ----------
    texts : list[str]
        Training utterances for this fold.
    labels : list[str]
        Their intent ids.

    Returns
    -------
    KnowledgeBase
        A KB whose intents carry only the fold's examples (enough to fit a
        classifier; responses/metadata are irrelevant for scoring accuracy).
    """
    from intent_engine.kb import Intent

    # Group examples by label into throwaway Intent records.
    by_label: dict[str, list[str]] = {}
    for text, label in zip(texts, labels):
        by_label.setdefault(label, []).append(text)
    intents = [
        Intent(intent_id=label, title=label, examples=examples)
        for label, examples in by_label.items()
    ]
    return KnowledgeBase(intents)


def run(engines: list[str] | None = None) -> dict[str, object]:
    """Run the full statistical evaluation and persist the results.

    Parameters
    ----------
    engines : list[str] | None, optional
        Engines to include; defaults to all.

    Returns
    -------
    dict[str, object]
        The results dict (also written to ``crossval_results.json``).
    """
    engines = engines or [
        "tfidf",
        "fasttext_custom",
        "fasttext_pretrained",
        "bert",
        "llm",
    ]
    settings = get_settings()
    router = IntentRouter.from_directory(settings.knowledge_base_dir)
    kb = router.kb

    bootstrap: dict[str, list[float]] = {}
    cv: dict[str, list[float]] = {}
    summary: dict[str, dict[str, float]] = {}

    # 1) Held-out bootstrap for every engine (the cross-engine comparison).
    for engine in engines:
        logger.info("Bootstrapping %s ...", engine)
        correct = _engine_correctness(router, engine, use_cache=True)
        dist = bootstrap_accuracy(correct)
        bootstrap[engine] = dist.tolist()
        # Point accuracy plus the bootstrap spread and 95 % interval.
        summary[engine] = {
            "point_accuracy": float(correct.mean()),
            "boot_mean": float(dist.mean()),
            "boot_std": float(dist.std()),
            "ci_low": float(np.percentile(dist, 2.5)),
            "ci_high": float(np.percentile(dist, 97.5)),
            "n_test": int(len(correct)),
        }

    # 2) Repeated k-fold CV for the trainable engines only.
    for engine in engines:
        if engine == "llm":
            # Zero-shot: no training, so KB cross-validation does not apply.
            continue
        logger.info("Cross-validating %s ...", engine)
        folds = cross_validate_engine(engine, kb)
        cv[engine] = folds.tolist()
        summary[engine]["cv_mean"] = float(folds.mean())
        summary[engine]["cv_std"] = float(folds.std())

    results: dict[str, object] = {
        "seed": _SEED,
        "n_bootstrap": len(next(iter(bootstrap.values()))),
        "bootstrap": bootstrap,
        "cv": cv,
        "summary": summary,
    }
    # Persist for the violin renderer and for the docs.
    _RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI: run the statistical evaluation and print a summary table.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Always ``0`` (this is a reporting tool, not a gate).
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="eval.crossval",
        description="Base de test statistique (bootstrap + validation croisée).",
    )
    parser.add_argument(
        "--engine",
        choices=["tfidf", "fasttext_custom", "fasttext_pretrained", "bert", "llm"],
        action="append",
        help="Restreindre aux moteurs donnés (répétable).",
    )
    args = parser.parse_args(argv)
    results = run(args.engine)

    # Print a compact mean ± std table so the terminal alone tells the story.
    summary = results["summary"]  # type: ignore[index]
    print("\nmoteur | held-out (moyenne ± σ, IC95%)        | CV k-fold (moyenne ± σ)")
    print("-------|--------------------------------------|------------------------")
    for engine, s in summary.items():  # type: ignore[union-attr]
        cv_txt = (
            f"{s['cv_mean']:.0%} ± {s['cv_std']:.0%}"
            if "cv_mean" in s
            else "n/a (zero-shot)"
        )
        print(
            f"{engine:<6} | {s['boot_mean']:.0%} ± {s['boot_std']:.0%} "
            f"[{s['ci_low']:.0%}–{s['ci_high']:.0%}]        | {cv_txt}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
