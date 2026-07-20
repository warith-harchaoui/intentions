"""Rigorous evaluation of the four *trainable* intent engines with **skore**.

Why this module exists
----------------------
The historical :mod:`eval.crossval` measured accuracy by hand — a
``RepeatedStratifiedKFold`` loop with a home-grown ``correct / total``. That
works, but it hides the honest generalisation picture behind a single number
and re-implements what the scientific stack already does well. This module
follows the *probabl* methodology instead:

* the sklearn-side pipeline is **declared as a skrub DataOps graph**
  (:func:`build_tfidf_learner`), not a bare ``sklearn.Pipeline`` reached for
  ad hoc;
* every engine is scored through **skore report objects**
  (:class:`skore.CrossValidationReport`, :class:`skore.EstimatorReport`,
  :class:`skore.ComparisonReport`) — one uniform, audited path that yields
  accuracy *and* macro precision / recall, per-class breakdowns, timings and
  confidence spreads for free.

Scope
-----
Four **trainable** classifiers, each exposed as a scikit-learn-compatible
estimator over ``text -> intent id`` so skore can treat them identically:

1. **TF-IDF + Random Forest** — native sklearn, declared via skrub DataOps.
2. **fastText (learned)** — retrained per fold on the fold's own text.
3. **fastText (pretrained)** — frozen cc.fr.300 vectors + logistic regression.
4. **BERT (SBERT) + MLP** — frozen sentence embeddings + a PyTorch MLP head.

The **LLM** engine has no ``fit`` (it is zero/few-shot), so it is *not* an
estimator here — it stays a held-out baseline scored elsewhere
(:mod:`eval.llm_shootout`).

What skore measures here
------------------------
The reports score the classifier's **raw argmax accuracy** — no abstention
threshold. That is the discriminative ML question ("does the model separate
the 21 intents?") and it keeps all four engines comparable. The *product*
question ("when should the router abstain?") is an operating-point choice
measured separately by :mod:`eval.harness` / :mod:`eval.crossval`.

Splitter choice
---------------
The training set is **perfectly balanced** (48 utterances per intent, 21
intents). Stratification therefore buys nothing — every fold already sees
every class in proportion — so we use a plain
:class:`~sklearn.model_selection.RepeatedKFold`
rather than the stratified variant, matching the probabl guidance to not
reach for ``Stratified*`` by reflex.

Usage
-----
    python -m eval.skore_eval                    # all four engines, CV + held-out
    python -m eval.skore_eval --fast             # only the two backbone-free engines

Author
------
Project maintainers.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import skrub
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from intent_engine.config import get_settings
from intent_engine.kb import Intent, KnowledgeBase

from .thresholds import load_dataset

logger = logging.getLogger(__name__)

# Reproducibility + fold budget. 5 splits × 5 repeats = 25 fold scores, the
# same budget the legacy crossval used, so the two are comparable.
_SEED = 20260718
_N_SPLITS = 5
_N_REPEATS = 5

# Frozen sentence embeddings are expensive to compute (a 4.5 GB fastText model
# load, or an SBERT forward pass) but do NOT depend on the training fold, so we
# compute them once for every utterance and cache them on disk. CV then only
# does dictionary look-ups.
_EMB_CACHE_DIR = Path(__file__).resolve().parent / ".skore_emb"

# Human-facing engine names, in the pedagogical progression order.
ENGINE_LABELS: dict[str, str] = {
    "tfidf": "TF-IDF + Random Forest",
    "fasttext_custom": "fastText (learned)",
    "fasttext_pretrained": "fastText (pretrained)",
    "bert": "BERT + MLP",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_train() -> tuple[np.ndarray, np.ndarray]:
    """Return the training ``(texts, labels)`` from the knowledge base.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Object arrays of utterance strings and their intent ids (1008 rows,
        48 per intent).
    """
    kb = KnowledgeBase.from_directory(get_settings().knowledge_base_dir)
    texts, labels = kb.training_pairs()
    return np.asarray(texts, dtype=object), np.asarray(labels, dtype=object)


def load_heldout() -> tuple[np.ndarray, np.ndarray]:
    """Return the held-out ``(texts, labels)`` from ``eval/dataset.jsonl``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Object arrays of paraphrase utterances and gold intent ids (210 rows,
        10 per intent), disjoint from the training set.
    """
    cases = load_dataset()
    texts = np.asarray([c["text"] for c in cases], dtype=object)
    labels = np.asarray([c["expected"] for c in cases], dtype=object)
    return texts, labels


def _kb_from_pairs(texts: list[str], labels: list[str]) -> KnowledgeBase:
    """Build a throwaway KB from ``(texts, labels)`` for one training fold.

    Parameters
    ----------
    texts : list[str]
        Fold training utterances.
    labels : list[str]
        Their intent ids.

    Returns
    -------
    KnowledgeBase
        A KB carrying only these examples — enough to fit an engine.
    """
    by_label: dict[str, list[str]] = {}
    for text, label in zip(texts, labels):
        by_label.setdefault(label, []).append(text)
    intents = [
        Intent(intent_id=label, title=label, examples=examples)
        for label, examples in by_label.items()
    ]
    return KnowledgeBase(intents)


# --------------------------------------------------------------------------- #
# Engine 1 — TF-IDF + Random Forest, declared as a skrub DataOps graph
# --------------------------------------------------------------------------- #
def _tfidf_pipeline() -> Pipeline:
    """Return the exact vectoriser + forest used by ``TfidfIntentEngine``.

    Kept identical to :class:`intent_engine.tfidf_engine.TfidfIntentEngine` so
    the skore numbers describe the shipped engine, not a look-alike.

    Returns
    -------
    sklearn.pipeline.Pipeline
        ``char_wb`` TF-IDF → balanced random forest.
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        lowercase=True,
        strip_accents="unicode",
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def build_tfidf_learner(texts_preview: np.ndarray | None = None) -> skrub.SkrubLearner:
    """Declare the TF-IDF + RF pipeline as a **skrub DataOps** learner.

    This is the *pipeline entry point* the ``build-ml-pipeline`` methodology
    asks for: the graph is rooted on ``skrub.var`` source nodes, the text
    column is marked as ``X`` and the labels as ``y``, and the sklearn pipeline
    is attached at the tail with ``.skb.apply``. Text classification is a flat,
    i.i.d. problem — one source frame, no cross-row features — so the ``X``
    marker sits directly on the loaded source (the canonical IID placement).

    Parameters
    ----------
    texts_preview : np.ndarray | None, optional
        Sample utterances shown in ``learner.skb.preview()`` during
        interactive work. ``None`` for production fit / cross-validate.

    Returns
    -------
    skrub.SkrubLearner
        Unfit learner; drive it with an env dict
        ``{"texts": ..., "labels": ...}`` via ``skore.evaluate`` / ``fit``.
    """
    if texts_preview is not None:
        texts = skrub.var("texts", texts_preview)
        placeholder = np.asarray(["_"] * len(texts_preview), dtype=object)
        labels = skrub.var("labels", placeholder)
    else:
        texts = skrub.var("texts")
        labels = skrub.var("labels")
    x = texts.skb.mark_as_X()
    y = labels.skb.mark_as_y()
    predictions = x.skb.apply(_tfidf_pipeline(), y=y)
    return predictions.skb.make_learner()


# --------------------------------------------------------------------------- #
# Engines 3 & 4 — frozen sentence embeddings + a light head
# --------------------------------------------------------------------------- #
def _embed_backbone(backbone: str, texts: list[str]) -> np.ndarray:
    """Embed ``texts`` with the heavy pretrained backbone (called once).

    Parameters
    ----------
    backbone : str
        ``"fasttext_pretrained"`` (cc.fr.300 vectors) or ``"bert"`` (SBERT).
    texts : list[str]
        Utterances to embed.

    Returns
    -------
    np.ndarray
        ``(n, dim)`` embedding matrix.
    """
    if backbone == "bert":
        from intent_engine.embeddings import build_embedder

        return np.asarray(build_embedder().encode(texts), dtype=np.float32)
    # fastText pretrained: reuse the engine's own loader + normalising embed.
    import fasttext

    from intent_engine.fasttext_engine import FastTextPretrainedEngine

    engine = FastTextPretrainedEngine()
    engine._vectors = fasttext.load_model(str(engine._model_path))
    return np.asarray(engine._embed(texts), dtype=np.float32)


def frozen_embeddings(backbone: str, texts: np.ndarray) -> dict[str, np.ndarray]:
    """Return a ``text -> vector`` cache for ``texts``, persisted to disk.

    The pretrained embeddings do not depend on the labels or the CV fold, so
    computing them once and looking them up per fold is both correct (no
    leakage) and fast. The cache is stored as an ``.npz`` keyed by backbone.

    Parameters
    ----------
    backbone : str
        ``"fasttext_pretrained"`` or ``"bert"``.
    texts : np.ndarray
        Every utterance that will be embedded (train + held-out).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from utterance to its frozen embedding vector.
    """
    _EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _EMB_CACHE_DIR / f"{backbone}.npz"
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        stored = np.load(path, allow_pickle=True)
        keys = stored["keys"].tolist()
        vecs = stored["vecs"]
        cache = {k: vecs[i] for i, k in enumerate(keys)}

    missing = [t for t in dict.fromkeys(texts.tolist()) if t not in cache]
    if missing:
        logger.info("Embedding %d new texts with %s ...", len(missing), backbone)
        new_vecs = _embed_backbone(backbone, missing)
        for t, v in zip(missing, new_vecs):
            cache[t] = v
        keys = list(cache.keys())
        np.savez(
            path,
            keys=np.asarray(keys, dtype=object),
            vecs=np.asarray([cache[k] for k in keys], dtype=np.float32),
        )
    return cache


class FrozenEmbedder(BaseEstimator, TransformerMixin):
    """Stateless transformer: utterance -> its cached frozen embedding.

    The embedding depends only on the pretrained backbone, never on the
    training fold, so this transformer learns nothing at ``fit`` — it is a pure
    look-up into a precomputed cache. That is exactly why the backbone counts
    as a *frozen feature extractor* and the CV stays leakage-free.

    Parameters
    ----------
    cache : dict[str, np.ndarray]
        The ``text -> vector`` mapping from :func:`frozen_embeddings`.
    """

    def __init__(self, cache: dict[str, np.ndarray]):
        self.cache = cache

    def fit(self, X, y=None):  # noqa: D102 - trivial (stateless)
        return self

    def transform(self, X) -> np.ndarray:  # noqa: D102 - cache look-up
        return np.vstack([self.cache[t] for t in X])


def build_pretrained_pipeline(cache: dict[str, np.ndarray]) -> Pipeline:
    """Frozen cc.fr.300 vectors → logistic regression (matches the engine)."""
    return Pipeline(
        [
            ("emb", FrozenEmbedder(cache)),
            ("clf", LogisticRegression(C=10.0, max_iter=1000)),
        ]
    )


def build_bert_pipeline(cache: dict[str, np.ndarray]) -> Pipeline:
    """Frozen SBERT embeddings → the project's PyTorch MLP head."""
    from intent_engine.mlp import TorchMLPClassifier

    return Pipeline([("emb", FrozenEmbedder(cache)), ("clf", TorchMLPClassifier())])


# --------------------------------------------------------------------------- #
# Engine 2 — fastText, retrained on every fold's own text
# --------------------------------------------------------------------------- #
class FastTextCustomClf(BaseEstimator, ClassifierMixin):
    """scikit-learn wrapper around ``fasttext.train_supervised``.

    Unlike the pretrained engines, the *learned* fastText model builds its
    subword embeddings from the training text itself, so it must be **refit on
    each fold** — no frozen cache. Hyper-parameters mirror
    :class:`intent_engine.fasttext_engine.FastTextSupervisedEngine`.
    """

    def fit(self, X, y):
        """Train the supervised model on ``(X, y)`` for this fold."""
        import fasttext

        from intent_engine.fasttext_engine import _LABEL_PREFIX, _normalise

        self.classes_ = np.unique(y)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as handle:
            for text, label in zip(X, y):
                handle.write(f"{_LABEL_PREFIX}{label} {_normalise(text)}\n")
            train_path = handle.name
        # ``thread=1`` + a fixed ``seed`` make training deterministic: multi-
        # threaded fastText updates in a non-reproducible order, which would
        # jitter the held-out accuracy run to run. Evaluation must be exactly
        # repeatable, so we trade the shipped engine's speed for determinism.
        self.model_ = fasttext.train_supervised(
            input=train_path,
            epoch=40,
            lr=0.5,
            wordNgrams=2,
            dim=100,
            minn=3,
            maxn=5,
            loss="softmax",
            thread=1,
            seed=_SEED,
            verbose=0,
        )
        Path(train_path).unlink(missing_ok=True)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return per-class probabilities aligned to ``classes_``."""
        from intent_engine.fasttext_engine import _LABEL_PREFIX, _normalise

        index = {c: i for i, c in enumerate(self.classes_)}
        proba = np.zeros((len(X), len(self.classes_)), dtype=float)
        k = len(self.classes_)
        for row, text in enumerate(X):
            # Low-level C++ predictor: the Python wrapper's ``predict`` does
            # ``np.array(probs, copy=False)`` which raises under NumPy 2.x.
            # ``f.predict`` returns ``(probability, label)`` tuples.
            ranked = self.model_.f.predict(_normalise(text), k, 0.0, "strict")
            for prob, label in ranked:
                proba[row, index[label[len(_LABEL_PREFIX) :]]] = prob
        return proba

    def predict(self, X) -> np.ndarray:
        """Return the argmax intent id for every utterance in ``X``."""
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _splitter() -> RepeatedKFold:
    """Return the shared fold generator (balanced data → plain KFold)."""
    return RepeatedKFold(n_splits=_N_SPLITS, n_repeats=_N_REPEATS, random_state=_SEED)


def _estimators(fast: bool) -> dict[str, object]:
    """Build the estimator for each engine (skipping backbones when ``fast``).

    Parameters
    ----------
    fast : bool
        When ``True``, only the two backbone-free engines (TF-IDF and learned
        fastText) are built — useful for a quick machinery check.

    Returns
    -------
    dict[str, object]
        Engine id → sklearn-compatible estimator (or skrub learner for TF-IDF).
    """
    train_texts, _ = load_train()
    heldout_texts, _ = load_heldout()
    every_text = np.concatenate([train_texts, heldout_texts])

    estimators: dict[str, object] = {
        "tfidf": build_tfidf_learner(),
        "fasttext_custom": FastTextCustomClf(),
    }
    if not fast:
        estimators["fasttext_pretrained"] = build_pretrained_pipeline(
            frozen_embeddings("fasttext_pretrained", every_text)
        )
        estimators["bert"] = build_bert_pipeline(frozen_embeddings("bert", every_text))
    return estimators


def cross_validation_report(engine: str, estimator: object):
    """Return a skore CrossValidationReport for one engine on the train set.

    Parameters
    ----------
    engine : str
        Engine id (drives the skrub-vs-sklearn calling convention).
    estimator : object
        The estimator or skrub learner from :func:`_estimators`.

    Returns
    -------
    skore.CrossValidationReport
        The 25-fold report over the 1008 balanced training utterances.
    """
    from skore import CrossValidationReport, evaluate

    x, y = load_train()
    if engine == "tfidf":
        # SkrubLearner takes an environment dict, not (X, y).
        return evaluate(estimator, data={"texts": x, "labels": y}, splitter=_splitter())
    return CrossValidationReport(estimator, X=x, y=y, splitter=_splitter())


def heldout_report(engine: str, estimator: object):
    """Return a skore EstimatorReport: fit on the 1008 train, score on 210.

    Parameters
    ----------
    engine : str
        Engine id.
    estimator : object
        The estimator or skrub learner from :func:`_estimators`.

    Returns
    -------
    skore.EstimatorReport
        Held-out report; ``.metrics.accuracy()`` is the paraphrase accuracy.
    """
    from skore import EstimatorReport

    x_tr, y_tr = load_train()
    x_te, y_te = load_heldout()
    if engine == "tfidf":
        # SkrubLearner fits from an env dict; EstimatorReport fits it for us.
        return EstimatorReport(
            estimator,
            train_data={"texts": x_tr, "labels": y_tr},
            test_data={"texts": x_te, "labels": y_te},
        )
    return EstimatorReport(
        estimator,
        X_train=x_tr,
        y_train=y_tr,
        X_test=x_te,
        y_test=y_te,
    )


def _accuracy(report) -> float:
    """Pull the mean accuracy scalar out of a skore report's metric frame."""
    frame = report.metrics.accuracy()
    return float(np.asarray(frame).ravel()[0])


def run(fast: bool = False) -> dict[str, dict[str, float]]:
    """Run CV + held-out skore reports for every engine and persist a summary.

    Parameters
    ----------
    fast : bool, optional
        Only the two backbone-free engines, by default ``False``.

    Returns
    -------
    dict[str, dict[str, float]]
        Engine id → ``{"cv_accuracy", "cv_std", "heldout_accuracy"}``.
    """
    from skore import ComparisonReport

    estimators = _estimators(fast)
    summary: dict[str, dict[str, float]] = {}
    heldout_reports: dict[str, object] = {}
    for engine, estimator in estimators.items():
        logger.info("skore CV report: %s ...", engine)
        cv = cross_validation_report(engine, estimator)
        cv_frame = cv.metrics.accuracy(aggregate=("mean", "std"))
        cv_arr = np.asarray(cv_frame).ravel()
        # The per-fold scores (aggregate=None → one column per split) are what
        # the violin figure draws, so the plotted density and the mean/std
        # reported here come from the *same* skore report — one source of truth.
        cv_folds = np.asarray(cv.metrics.accuracy(aggregate=None)).ravel().tolist()

        logger.info("skore held-out report: %s ...", engine)
        # A fresh estimator for the held-out fit (CV already consumed clones).
        ho = heldout_report(engine, _estimators(fast)[engine])
        heldout_reports[ENGINE_LABELS[engine]] = ho
        summary[engine] = {
            "label": ENGINE_LABELS[engine],
            "cv_accuracy": float(cv_arr[0]),
            "cv_std": float(cv_arr[1]) if cv_arr.size > 1 else 0.0,
            "cv_folds": [float(v) for v in cv_folds],
            "heldout_accuracy": _accuracy(ho),
        }
        logger.info(
            "%-22s CV %.3f ± %.3f | held-out %.3f",
            engine,
            summary[engine]["cv_accuracy"],
            summary[engine]["cv_std"],
            summary[engine]["heldout_accuracy"],
        )

    # One skore ComparisonReport over every engine's held-out report — the
    # single object that lines the classifiers up on identical metrics.
    comparison = ComparisonReport(heldout_reports)
    here = Path(__file__).resolve().parent
    (here / "skore_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_comparison_markdown(here / "skore_comparison.md", summary, comparison)
    logger.info("Wrote %s and skore_comparison.md", here / "skore_results.json")
    return summary


def _write_comparison_markdown(path: Path, summary: dict, comparison) -> None:
    """Render the CV + held-out accuracy table (and skore's own summary).

    Parameters
    ----------
    path : Path
        Destination markdown file.
    summary : dict
        The per-engine ``{cv_accuracy, cv_std, heldout_accuracy}`` mapping.
    comparison : skore.ComparisonReport
        The held-out comparison whose ``metrics.summarize()`` frame is appended
        as skore's authoritative cross-engine view.
    """
    lines = [
        "# skore evaluation — trainable intent engines",
        "",
        "Raw argmax accuracy (no abstention). CV = 25-fold RepeatedKFold on the",
        "1008 balanced training utterances; held-out = fit on 1008, scored on the",
        "210 disjoint paraphrases. Produced by `python -m eval.skore_eval`.",
        "",
        "| Engine | CV accuracy | Held-out accuracy |",
        "|---|---|---|",
    ]
    for row in summary.values():
        lines.append(
            f"| {row['label']} | {row['cv_accuracy']:.1%} ± {row['cv_std']:.1%} "
            f"| {row['heldout_accuracy']:.1%} |"
        )
    lines += [
        "",
        "## skore ComparisonReport — held-out metrics",
        "",
        "```",
        comparison.metrics.summarize().frame().round(3).to_string(),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """CLI entry point for the skore evaluation."""
    parser = argparse.ArgumentParser(description="skore evaluation of the engines")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="only the two backbone-free engines (TF-IDF, learned fastText)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(fast=args.fast)


if __name__ == "__main__":
    main()
