"""LLM shootout: models × prompts, one honest comparison.

Module summary
--------------
The LLM *approach* is one row of the five-engine progression, but under it two
knobs change everything: **which local model** you run, and **which prompt**
you feed it. This module runs the same intent task through several
(model, prompt) configurations and collects, per config:

* top-1 **accuracy** on a fixed held-out sample, plus a **bootstrap**
  distribution for the violin plot (via :func:`eval.crossval.bootstrap_accuracy`);
* everything is **cached per config** on disk, written incrementally, so the
  long (~30-40 min) run is resumable and re-runs are free.

Two stories fall out of one table:

* **Model matters** — a light Qwen vs the compact Gemma vs the two bigger
  Gemma builds, all on the *same* (improved) prompt.
* **The prompt matters** — the same fast models on a **baseline** vs an
  **engineered** prompt (few-shot + reason-first), showing the accuracy lift
  that prompt engineering alone buys.

Usage
-----
    python -m eval.llm_shootout                 # default config line-up
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
from intent_engine.llm_engine import (
    _IMPROVED_SYSTEM_PROMPT,
    _NAIVE_SYSTEM_PROMPT,
    LlmIntentEngine,
)
from intent_engine.router import IntentRouter

from .crossval import bootstrap_accuracy
from .thresholds import load_dataset

logger = logging.getLogger(__name__)

# Prompt variants along the prompt-quality axis. Internal keys are stable (they
# name the on-disk caches); ``_PROMPT_LABEL`` maps them to the human wording.
# ``naive`` is the quick-and-dirty prompt ("à la va-vite"); ``improved`` is the
# engineered one (few-shot + reason-first). ``None`` = the engine's default.
_PROMPTS: dict[str, str | None] = {
    "naive": _NAIVE_SYSTEM_PROMPT,
    "improved": _IMPROVED_SYSTEM_PROMPT,
}
_PROMPT_LABEL: dict[str, str] = {"naive": "va-vite", "improved": "soigné"}

# The configurations, in display order. Every model runs the **improved**
# (soigné) prompt (the model comparison); the two *fast* models also run the
# **naive** (va-vite) prompt (the prompt-engineering comparison), so the slow
# Gemma builds are queried once. Two takeaways from one table: what the *model*
# buys, and what a *good prompt* buys.
_CONFIGS: list[dict[str, str]] = [
    {"model": "qwen2.5:3b", "prompt": "naive"},
    {"model": "qwen2.5:3b", "prompt": "improved"},
    {"model": "gemma3:4b", "prompt": "naive"},
    {"model": "gemma3:4b", "prompt": "improved"},
    {"model": "gemma4:e2b-mlx", "prompt": "improved"},
    {"model": "gemma4:e4b-mlx", "prompt": "improved"},
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
        ``"baseline"`` or ``"improved"``.

    Returns
    -------
    str
        e.g. ``"gemma3:4b · improved"``.
    """
    # Space-dot-space separator reads well in the table and as a violin label;
    # show the human wording (va-vite / soigné) rather than the internal key.
    return f"{model} · {_PROMPT_LABEL.get(prompt, prompt)}"


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
        ``"baseline"`` or ``"improved"``.
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
    # Pick the prompt variant (va-vite / soigné); ``None`` → engine default.
    system_prompt = _PROMPTS.get(prompt)
    engine = LlmIntentEngine(model=model, system_prompt=system_prompt).fit(kb)

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
    print(f"\n({elapsed / 60:.1f} min ; prédictions mises en cache par config)")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
