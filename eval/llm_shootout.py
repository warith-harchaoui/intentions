"""LLM shootout: models × prompts, one honest comparison.

Module summary
--------------
The LLM *approach* is one row of the five-engine progression, but under it the
**prompt** changes everything. This module runs a **2×2 prompt-engineering
experiment** — prompt *quality* (bad → good) × *examples* (zero-shot →
few-shot) — across several candidate local models, and collects, per config:

* the **accuracy** on a fixed held-out sample, plus a bootstrap distribution
  (via :func:`eval.crossval.bootstrap_accuracy`) for confidence intervals;
* everything is **cached per config** on disk, written incrementally, so the
  long (~30-40 min) run is resumable and re-runs are free.

The point: find the **single model** whose accuracy climbs the most cleanly
across the four prompts (:func:`_pick_best_model`), so the docs can tell one
uncluttered "de mieux en mieux" story instead of a busy multi-model table. The
lesson that falls out — prompt engineering lifts a *weak* model far more than a
*strong* one already near its ceiling.

Usage
-----
    python -m eval.llm_shootout                 # the 2×2 across candidate models
    python -m eval.llm_shootout --sample 30

Author
------
Project maintainers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from intent_engine.config import get_settings
from intent_engine.llm_engine import LlmIntentEngine, experiment_prompt
from intent_engine.router import IntentRouter

from .crossval import bootstrap_accuracy
from .thresholds import load_dataset

logger = logging.getLogger(__name__)

# --- The 2×2 prompt-engineering experiment --------------------------------
# Two INDEPENDENT switches — prompt *quality* (bad → good) and *examples*
# (zero-shot → few-shot) — give four prompts along one "de mieux en mieux"
# axis. Internal keys are stable (they name the on-disk caches); the human
# wording lives in ``eval/violin.py`` (bilingual). ``experiment_prompt`` builds
# each variant compositionally so the ONLY differences are the rules and the
# examples.
_PROMPT_SWITCHES: dict[str, tuple[bool, bool]] = {
    "bad-zs": (False, False),  # bare task + schema, no examples
    "bad-fs": (False, True),  # + three worked examples
    "good-zs": (True, False),  # + error-driven disambiguation rules
    "good-fs": (True, True),  # rules AND examples (the full treatment)
}
_PROMPTS: dict[str, str] = {
    key: experiment_prompt(good=good, fewshot=fewshot)
    for key, (good, fewshot) in _PROMPT_SWITCHES.items()
}
# The pedagogical progression order (the figure's x-axis).
_PROMPT_ORDER: list[str] = ["bad-zs", "bad-fs", "good-zs", "good-fs"]

# We run the FULL 2×2 on several candidate models to *find* the one whose
# accuracy climbs the most cleanly (monotone, wide span) across the four
# prompts — the user asked for the single model that best shows the progression
# ("prend un LLM qui fait le mieux la différence de progrès … sinon c'est
# touffu"). The docs then present only that winner; the rest is diagnostic.
_CANDIDATE_MODELS: list[str] = ["qwen2.5:3b", "gemma3:4b", "gemma4:e2b-mlx"]
_CONFIGS: list[dict[str, str]] = [
    {"model": model, "prompt": prompt}
    for model in _CANDIDATE_MODELS
    for prompt in _PROMPT_ORDER
]

# Held-out sample size (a fixed prefix of the paraphrase test set). Smaller
# than the full 88 to keep the multi-model run within ~30-40 min; still enough
# for a bootstrap comparison.
_DEFAULT_SAMPLE = 30

# Per-config prediction cache directory + aggregated results file.
_CACHE_DIR = Path(__file__).resolve().parent / ".llm_shootout"
_RESULTS_PATH = Path(__file__).resolve().parent / "llm_shootout_results.json"


def _config_key(model: str, prompt: str) -> str:
    """Return a stable, human-readable key for a (model, prompt) config.

    Parameters
    ----------
    model : str
        Ollama model tag.
    prompt : str
        One of the 2×2 prompt keys (``"bad-zs"`` … ``"good-fs"``).

    Returns
    -------
    str
        e.g. ``"gemma3:4b · good-fs"``.
    """
    # Space-dot-space separator reads well in the table and as a violin label.
    # The internal prompt key (bad-zs … good-fs) is stable; the bilingual human
    # wording is applied later, in the figure renderer.
    return f"{model} · {prompt}"


def _cache_path(model: str, prompt: str) -> Path:
    """Filesystem-safe cache path for a (model, prompt) config."""
    # Replace path-unsafe characters in the model tag.
    safe = model.replace("/", "_").replace(":", "_")
    return _CACHE_DIR / f"{safe}__{prompt}.json"


def classify_sample(model: str, prompt: str, kb, sample: int) -> dict[str, str]:
    """Classify the held-out sample with one (model, prompt), caching as we go.

    Parameters
    ----------
    model : str
        Ollama model tag.
    prompt : str
        One of the 2×2 prompt keys (``"bad-zs"`` … ``"good-fs"``).
    kb : KnowledgeBase
        The knowledge base grounding the prompt.
    sample : int
        Number of held-out examples (a fixed prefix) to classify.

    Returns
    -------
    dict[str, str]
        Utterance → predicted intent id ("" on abstention).
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(model, prompt)
    cache: dict[str, str] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    )
    # Pick the 2×2 prompt variant (bad-zs … good-fs).
    engine = LlmIntentEngine(model=model, system_prompt=_PROMPTS[prompt]).fit(kb)

    for case in load_dataset()[:sample]:
        text = case["text"]
        if text in cache:
            continue
        top = engine.classify(text).top()
        cache[text] = top.intent if top is not None else ""
        # Persist after every (slow) call so a crash never loses progress.
        path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return cache


def _correctness(cache: dict[str, str], sample: int) -> np.ndarray:
    """Build the 0/1 correctness vector for a config over the sample.

    Parameters
    ----------
    cache : dict[str, str]
        Utterance → predicted intent id.
    sample : int
        Number of held-out examples scored.

    Returns
    -------
    np.ndarray
        Aligned 0/1 flags over the sampled dataset order.
    """
    return np.asarray(
        [
            1.0 if cache.get(c["text"], "") == c["expected"] else 0.0
            for c in load_dataset()[:sample]
        ],
        dtype=float,
    )


def _progression_score(accuracies: list[float]) -> float:
    """Score how cleanly a model's accuracy climbs across the four prompts.

    We want the clearest "de mieux en mieux" story: a **monotone** climb from
    ``bad-zs`` to ``good-fs``. A big span is worthless if the middle collapses
    (e.g. 23 → 77 → 40 → 80 has a huge span but is a rollercoaster, the opposite
    of the lesson). So every backward step is penalised **3×** its size — a
    single dip sinks a model below any genuinely monotone one. Among monotone
    progressions, the wider span wins.

    Parameters
    ----------
    accuracies : list[float]
        The four accuracies in :data:`_PROMPT_ORDER`.

    Returns
    -------
    float
        ``span - 3·dips``. Higher is a cleaner, wider, more monotone climb.
    """
    span = accuracies[-1] - accuracies[0]
    dips = sum(
        max(0.0, accuracies[i] - accuracies[i + 1]) for i in range(len(accuracies) - 1)
    )
    # Weight dips heavily so a monotone climb always beats a high-variance one.
    return span - 3.0 * dips


def _pick_best_model(summary: dict[str, dict]) -> str:
    """Return the candidate model with the cleanest prompt progression.

    Parameters
    ----------
    summary : dict[str, dict]
        The per-config summary produced by :func:`run`.

    Returns
    -------
    str
        The winning model tag (the one the docs will present).
    """
    by_model: dict[str, dict[str, float]] = {}
    for s in summary.values():
        by_model.setdefault(s["model"], {})[s["prompt"]] = s["point_accuracy"]
    best_model, best_score = "", float("-inf")
    for model, accs in by_model.items():
        # Only models that ran the full 2×2 are eligible for the headline chart.
        if not all(p in accs for p in _PROMPT_ORDER):
            continue
        score = _progression_score([accs[p] for p in _PROMPT_ORDER])
        logger.info("Progression %-16s score=%+.3f", model, score)
        if score > best_score:
            best_model, best_score = model, score
    return best_model


def run(sample: int = _DEFAULT_SAMPLE) -> dict[str, object]:
    """Run every configuration and persist accuracy + bootstrap samples.

    Parameters
    ----------
    sample : int, optional
        Held-out sample size, by default :data:`_DEFAULT_SAMPLE`.

    Returns
    -------
    dict[str, object]
        Results with per-config bootstrap samples and a summary.
    """
    router = IntentRouter.from_directory(get_settings().knowledge_base_dir)
    kb = router.kb

    bootstrap: dict[str, list[float]] = {}
    summary: dict[str, dict[str, object]] = {}
    for cfg in _CONFIGS:
        model, prompt = cfg["model"], cfg["prompt"]
        key = _config_key(model, prompt)
        logger.info("Shootout: %s ...", key)
        cache = classify_sample(model, prompt, kb, sample)
        correct = _correctness(cache, sample)
        dist = bootstrap_accuracy(correct)
        bootstrap[key] = dist.tolist()
        summary[key] = {
            "model": model,
            "prompt": prompt,
            "point_accuracy": float(correct.mean()),
            "boot_mean": float(dist.mean()),
            "boot_std": float(dist.std()),
            "ci_low": float(np.percentile(dist, 2.5)),
            "ci_high": float(np.percentile(dist, 97.5)),
            "n_test": int(len(correct)),
        }

    results: dict[str, object] = {
        "sample": sample,
        "bootstrap": bootstrap,
        "summary": summary,
        # The model whose accuracy climbs the most cleanly across the four
        # prompts — the one the headline figure presents.
        "best_model": _pick_best_model(summary),
    }
    _RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI: run the shootout and print a per-config accuracy table.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="eval.llm_shootout",
        description="Compare des LLM locaux et des prompts sur la tâche d'intention.",
    )
    parser.add_argument(
        "--sample", type=int, default=_DEFAULT_SAMPLE, help="Taille de l'échantillon."
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    results = run(args.sample)
    elapsed = time.perf_counter() - started

    print(f"\nmodèle · prompt          | exactitude | IC95%       (n={args.sample})")
    print("-------------------------|------------|------------------")
    summary = results["summary"]  # type: ignore[index]
    for key, s in summary.items():  # type: ignore[union-attr]
        print(
            f"{key:24} | {s['boot_mean']:9.0%}  | {s['ci_low']:.0%}–{s['ci_high']:.0%}"
        )
    best = results.get("best_model", "")  # type: ignore[union-attr]
    print(f"\nMeilleure progression : {best} (présenté dans les figures)")
    print(f"({elapsed / 60:.1f} min ; prédictions mises en cache par config)")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
