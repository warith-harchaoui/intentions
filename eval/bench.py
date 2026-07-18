"""Robust timing: CPU time for local engines, Ollama compute for the LLM.

Module summary
--------------
Wall-clock latency on a shared dev machine is noisy — other apps steal the
CPU and skew the numbers. This tool measures the *useful* work instead:

* **Local engines** (TF-IDF, fastText ×2, BERT) — **CPU time** via
  ``time.process_time()`` (user+system CPU consumed by *this* process),
  which is immune to unrelated background activity. For BERT we also report
  ``wall − CPU`` as a proxy for the time spent off-CPU (the MPS/Metal **GPU**
  path plus any I/O), so the CPU/GPU split is visible.

* **LLM** (Gemma via Ollama) — the model runs in a *separate* process, so
  our CPU time is ~0 (we just wait on HTTP). Instead we read the durations
  Ollama reports itself: ``prompt_eval_duration`` + ``eval_duration`` are the
  real prompt-processing + token-generation **compute** on the server
  (nanoseconds), excluding the one-off model ``load_duration``. That is the
  honest "useful GPU time" for the LLM.

Accuracy is the reproducible, seeded metric (see ``eval/crossval.py``);
these timings are order-of-magnitude, but CPU-time / Ollama-compute are far
steadier than wall-clock under load.

Usage
-----
    python -m eval.bench            # all available engines
    python -m eval.bench --repeats 20

Author
------
Project maintainers.
"""

from __future__ import annotations

import logging
import math
import time

import httpx

from intent_engine.config import get_settings
from intent_engine.router import IntentRouter

from .thresholds import load_dataset

logger = logging.getLogger(__name__)

# The purely-local engines whose compute we can time with process_time.
_LOCAL_ENGINES = ("tfidf", "fasttext_custom", "fasttext_pretrained", "bert")


def format_duration(ms: float) -> str:
    """Format a millisecond duration without ever collapsing to a bare ``0``.

    Picks the largest time unit (s / ms / µs / ns) in which the value is at
    least 1, then prints ~3 significant figures. A tiny-but-nonzero measurement
    — fastText classifying in ~8 microseconds — then shows as ``8.00 µs``
    instead of a misleading ``0.00 ms``. A genuine zero (e.g. no off-CPU/GPU
    work for a CPU-only engine) prints as ``—``, never ``0``.

    Parameters
    ----------
    ms : float
        A duration in milliseconds.

    Returns
    -------
    str
        e.g. ``"1.20 s"``, ``"500 ms"``, ``"8.00 µs"``, ``"—"``.

    Examples
    --------
    >>> format_duration(0.008)
    '8.00 µs'
    >>> format_duration(500.0)
    '500 ms'
    >>> format_duration(0.0)
    '—'
    """
    if not math.isfinite(ms) or ms <= 0:
        return "—"
    # (unit label, multiplier from ms to that unit), largest first.
    for name, factor in (("s", 1e-3), ("ms", 1.0), ("µs", 1e3), ("ns", 1e6)):
        value = ms * factor
        if value >= 1.0:
            # ~3 significant figures: fewer decimals as the value grows.
            if value >= 100:
                return f"{value:.0f} {name}"
            if value >= 10:
                return f"{value:.1f} {name}"
            return f"{value:.2f} {name}"
    # Sub-nanosecond but still positive: show it rather than rounding to zero.
    return f"{ms * 1e6:.2f} ns"


def bench_local_engine(
    router: IntentRouter, engine: str, texts: list[str], repeats: int
) -> dict[str, float]:
    """Measure warm wall and CPU time per classify for a local engine.

    Parameters
    ----------
    router : IntentRouter
        Router holding the fitted engines.
    engine : str
        Local engine name.
    texts : list[str]
        Utterances to classify (cycled over ``repeats`` passes).
    repeats : int
        Number of passes over ``texts`` (more = steadier mean).

    Returns
    -------
    dict[str, float]
        ``wall_ms``, ``cpu_ms`` and ``off_cpu_ms`` (wall − cpu) per call.
    """
    # Warm up first so model loads / lazy allocations do not pollute timing.
    router.engine(engine).classify("préchauffage")

    # ``perf_counter`` is wall-clock; ``process_time`` is CPU consumed by this
    # process (user+system) — the metric that ignores other apps' load.
    n = len(texts) * repeats
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for _ in range(repeats):
        for text in texts:
            router.classify(text, engine)
    wall = (time.perf_counter() - wall0) / n * 1000.0
    cpu = (time.process_time() - cpu0) / n * 1000.0
    # ``off_cpu`` is time the wall clock advanced while this process was NOT
    # burning CPU — for BERT that is mostly the MPS GPU doing the matmuls.
    return {"wall_ms": wall, "cpu_ms": cpu, "off_cpu_ms": max(0.0, wall - cpu)}


def bench_llm_ollama(sample: list[str], max_calls: int = 5) -> dict[str, float]:
    """Measure the LLM's server-side compute via Ollama's own durations.

    Parameters
    ----------
    sample : list[str]
        A few utterances to classify (kept small — the LLM is slow).
    max_calls : int, optional
        Cap on the number of Ollama calls, by default 5.

    Returns
    -------
    dict[str, float]
        Mean ``wall_ms`` (round-trip), ``prompt_ms`` and ``gen_ms`` (the
        server-reported prompt-eval and generation compute), all excluding
        the one-off model load.
    """
    settings = get_settings()
    endpoint = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    model = settings.llm_model

    walls: list[float] = []
    prompts: list[float] = []
    gens: list[float] = []
    # One warm-up call so ``load_duration`` is paid before we measure.
    with httpx.Client(timeout=settings.request_timeout_s) as client:
        client.post(
            endpoint,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "format": "json",
            },
        )
        # Time a handful of real short classifications, reading Ollama's
        # nanosecond counters for the *compute* (prompt + generation).
        for text in sample[:max_calls]:
            started = time.perf_counter()
            resp = client.post(
                endpoint,
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": f'Intention de : "{text}" ?'}
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.0},
                },
            )
            walls.append((time.perf_counter() - started) * 1000.0)
            data = resp.json()
            # ns → ms. These exclude ``load_duration`` (the model is warm).
            prompts.append(data.get("prompt_eval_duration", 0) / 1e6)
            gens.append(data.get("eval_duration", 0) / 1e6)

    # Guard the empty case; otherwise report the means.
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return {"wall_ms": mean(walls), "prompt_ms": mean(prompts), "gen_ms": mean(gens)}


def main(argv: list[str] | None = None) -> int:
    """CLI: print a robust wall/CPU/compute timing table for the engines.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0).
    """
    import argparse

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="eval.bench",
        description="Chronométrage robuste (CPU + compute Ollama, pas wall-clock).",
    )
    parser.add_argument(
        "--repeats", type=int, default=15, help="Passes sur l'échantillon local."
    )
    args = parser.parse_args(argv)

    router = IntentRouter.from_directory(get_settings().knowledge_base_dir)
    available = router.available_engines()
    # A small, fixed sample of held-out utterances to time on.
    texts = [c["text"] for c in load_dataset()[:8]]

    print("moteur              |      wall |       CPU | hors-CPU≈GPU")
    print("--------------------|-----------|-----------|-------------")
    # Local engines: CPU time is the robust headline. Units are adaptive so a
    # microsecond-scale engine never prints a misleading "0.00 ms".
    for engine in _LOCAL_ENGINES:
        if engine not in available:
            print(f"{engine:19} | (indisponible)")
            continue
        t = bench_local_engine(router, engine, texts, args.repeats)
        print(
            f"{engine:19} | {format_duration(t['wall_ms']):>9} | "
            f"{format_duration(t['cpu_ms']):>9} | "
            f"{format_duration(t['off_cpu_ms']):>12}"
        )

    # LLM: report Ollama's server-side compute (prompt + generation).
    if "llm" in available:
        t = bench_llm_ollama(texts)
        print(
            f"\nllm ({get_settings().llm_model}) — compute Ollama (hors chargement) :"
        )
        print(
            f"  wall aller-retour {format_duration(t['wall_ms'])} | "
            f"prompt {format_duration(t['prompt_ms'])} | "
            f"génération {format_duration(t['gen_ms'])}"
        )
        print(
            "  (le wall-clock inclut l'attente/HTTP ; le calcul GPU utile "
            "est prompt+génération)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
