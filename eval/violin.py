"""Render violin plots of the engines' accuracy distributions (Vega-Lite).

Module summary
--------------
Turns ``eval/crossval_results.json`` (produced by :mod:`eval.crossval`) into
a **violin plot** comparing the five engines' bootstrap accuracy
distributions on the held-out paraphrase test set, and exports it to PNG via
``vl-convert``. A violin (a mirrored density) is the right picture for the
question the reader actually has — *"are these engines really different, or
are the scores just noise apart?"* — because it shows the whole spread and
overlap, not a single bar.

House style: Roboto, rounded, no chart-junk spines, engine colours matching
the web UI's chips.

Usage
-----
    python -m eval.violin        # writes docs/img/violin-accuracy.png

Author
------
Project maintainers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths: read the stats, write the figure next to the other doc images.
_RESULTS_PATH = Path(__file__).resolve().parent / "crossval_results.json"
_OUT_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "img" / "violin-accuracy.png"
)

# Display order (the pedagogical progression) + human labels + UI-matching
# hues (the same dark, WCAG-friendly chip colours the front end uses).
_ENGINE_META: dict[str, dict[str, str]] = {
    "tfidf": {"label": "TF-IDF\n+RandomForest", "color": "#0055CC"},
    "fasttext_custom": {"label": "fastText\n(appris)", "color": "#0E7490"},
    "fasttext_pretrained": {"label": "fastText\n(pré-entraîné)", "color": "#4338CA"},
    "bert": {"label": "BERT\n+MLP", "color": "#15803D"},
    "llm": {"label": "LLM\n(Gemma)", "color": "#B45309"},
}


def _long_rows(results: dict) -> list[dict[str, object]]:
    """Flatten the per-engine bootstrap samples into long-format rows.

    Parameters
    ----------
    results : dict
        The parsed ``crossval_results.json``.

    Returns
    -------
    list[dict[str, object]]
        One ``{"engine": label, "accuracy": value}`` row per bootstrap draw,
        for every engine present in the results.
    """
    rows: list[dict[str, object]] = []
    bootstrap: dict[str, list[float]] = results.get("bootstrap", {})
    # Emit rows in the fixed progression order so the violins line up left→right
    # from the simplest to the most powerful engine.
    for engine, meta in _ENGINE_META.items():
        for value in bootstrap.get(engine, []):
            rows.append({"engine": meta["label"], "accuracy": value})
    return rows


def build_spec(results: dict) -> dict:
    """Build a Vega-Lite violin-plot spec from the crossval results.

    Parameters
    ----------
    results : dict
        The parsed ``crossval_results.json``.

    Returns
    -------
    dict
        A Vega-Lite v5 specification (faceted density areas == violins).
    """
    rows = _long_rows(results)
    # Preserve progression order and keep colours aligned with the labels
    # actually present in the data.
    present = [
        m["label"] for e, m in _ENGINE_META.items() if e in results.get("bootstrap", {})
    ]
    colours = [
        m["color"] for e, m in _ENGINE_META.items() if e in results.get("bootstrap", {})
    ]

    # The canonical Vega-Lite violin: a horizontal density area per engine,
    # faceted into one narrow column each, stacked to centre so it mirrors.
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {
            "text": "Distribution d'exactitude par moteur (bootstrap, n=88)",
            "subtitle": "Jeu de test paraphrases · 2000 rééchantillonnages",
            "font": "Roboto",
            "subtitleFont": "Roboto",
            "anchor": "start",
            "fontSize": 15,
            "subtitleFontSize": 11,
            "subtitleColor": "#6E6E73",
        },
        # House style: Roboto everywhere, no view border (no chart-junk box).
        "config": {
            "font": "Roboto",
            "view": {"stroke": None},
            "axis": {"labelFont": "Roboto", "titleFont": "Roboto", "grid": False},
            "header": {"labelFont": "Roboto", "titleFont": "Roboto"},
        },
        "data": {"values": rows},
        "transform": [
            # Kernel-density-estimate the accuracy within each engine, over the
            # full [0,1] range so the violins are directly comparable.
            {
                "density": "accuracy",
                "groupby": ["engine"],
                "extent": [0, 1],
                "as": ["accuracy", "density"],
            }
        ],
        "mark": {"type": "area", "orient": "horizontal"},
        "width": 90,
        "height": 420,
        "encoding": {
            # Vertical axis: the accuracy (shared across facets for comparison).
            "y": {
                "field": "accuracy",
                "type": "quantitative",
                "title": "Exactitude (top-1)",
                "axis": {"format": "%"},
                "scale": {"domain": [0, 1]},
            },
            # Horizontal within a facet: the mirrored density (no axis — the
            # width is just the shape of the distribution).
            "x": {
                "field": "density",
                "type": "quantitative",
                "stack": "center",
                "impute": None,
                "title": None,
                "axis": {"labels": False, "ticks": False, "grid": False, "values": []},
            },
            # One narrow column per engine, in progression order.
            "column": {
                "field": "engine",
                "type": "nominal",
                "sort": present,
                "header": {
                    "titleOrient": "bottom",
                    "labelOrient": "bottom",
                    "labelPadding": 6,
                },
                "title": None,
            },
            "color": {
                "field": "engine",
                "type": "nominal",
                "scale": {"domain": present, "range": colours},
                "legend": None,
            },
        },
    }


# --- LLM shootout violin (models × prompts) ------------------------------
_SHOOTOUT_RESULTS = Path(__file__).resolve().parent / "llm_shootout_results.json"
_SHOOTOUT_OUT = (
    Path(__file__).resolve().parent.parent / "docs" / "img" / "violin-llm-shootout.png"
)
# One hue per model family (harchaoui palette); baseline vs improved share the
# model colour and are told apart by the x-axis label.
_MODEL_HUE: dict[str, str] = {
    "qwen2.5:3b": "#007AFF",
    "gemma3:4b": "#28CD41",
    "gemma4:e2b-mlx": "#FF9500",
    "gemma4:e4b-mlx": "#AF52DE",
}


def build_shootout_spec(results: dict) -> dict:
    """Build a Vega-Lite violin spec for the LLM shootout results.

    Parameters
    ----------
    results : dict
        Parsed ``llm_shootout_results.json`` (``bootstrap`` keyed by
        ``"model · prompt"``, plus ``summary`` carrying the model tag).

    Returns
    -------
    dict
        A Vega-Lite v5 faceted-density (violin) specification.
    """
    bootstrap: dict[str, list[float]] = results.get("bootstrap", {})
    summary: dict[str, dict] = results.get("summary", {})
    sample = results.get("sample", 0)
    # Long-format rows + per-config colour (by the config's model family).
    rows: list[dict[str, object]] = []
    order: list[str] = []
    colours: list[str] = []
    for key, samples in bootstrap.items():
        order.append(key)
        model = summary.get(key, {}).get("model", "")
        colours.append(_MODEL_HUE.get(model, "#6E6E73"))
        for value in samples:
            rows.append({"config": key, "accuracy": value})

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {
            "text": "LLM shootout — modèle × prompt (bootstrap)",
            "subtitle": f"paraphrases · n={sample} · baseline vs prompt amélioré",
            "font": "Roboto",
            "subtitleFont": "Roboto",
            "anchor": "start",
            "fontSize": 15,
            "subtitleFontSize": 11,
            "subtitleColor": "#6E6E73",
        },
        "config": {
            "font": "Roboto",
            "view": {"stroke": None},
            "axis": {"labelFont": "Roboto", "titleFont": "Roboto", "grid": False},
            "header": {"labelFont": "Roboto", "titleFont": "Roboto"},
        },
        "data": {"values": rows},
        "transform": [
            {
                "density": "accuracy",
                "groupby": ["config"],
                "extent": [0, 1],
                "as": ["accuracy", "density"],
            }
        ],
        "mark": {"type": "area", "orient": "horizontal"},
        "width": 80,
        "height": 400,
        "encoding": {
            "y": {
                "field": "accuracy",
                "type": "quantitative",
                "title": "Exactitude (top-1)",
                "axis": {"format": "%"},
                "scale": {"domain": [0, 1]},
            },
            "x": {
                "field": "density",
                "type": "quantitative",
                "stack": "center",
                "impute": None,
                "title": None,
                "axis": {"labels": False, "ticks": False, "grid": False, "values": []},
            },
            "column": {
                "field": "config",
                "type": "nominal",
                "sort": order,
                "header": {
                    "titleOrient": "bottom",
                    "labelOrient": "bottom",
                    "labelAngle": -35,
                    "labelPadding": 6,
                },
                "title": None,
            },
            "color": {
                "field": "config",
                "type": "nominal",
                "scale": {"domain": order, "range": colours},
                "legend": None,
            },
        },
    }


def render_shootout(results: dict | None = None) -> Path:
    """Render the LLM-shootout violin plot to PNG and return its path.

    Parameters
    ----------
    results : dict | None, optional
        Pre-loaded shootout results; loaded from disk when ``None``.

    Returns
    -------
    Path
        The written PNG path.

    Raises
    ------
    FileNotFoundError
        If the shootout results file is missing.
    """
    import vl_convert as vlc

    if results is None:
        if not _SHOOTOUT_RESULTS.is_file():
            raise FileNotFoundError(
                f"{_SHOOTOUT_RESULTS} manquant — lancez `python -m eval.llm_shootout`."
            )
        results = json.loads(_SHOOTOUT_RESULTS.read_text(encoding="utf-8"))
    png = vlc.vegalite_to_png(
        vl_spec=json.dumps(build_shootout_spec(results)), scale=2.0
    )
    _SHOOTOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    _SHOOTOUT_OUT.write_bytes(png)
    return _SHOOTOUT_OUT


def render(results: dict | None = None) -> Path:
    """Render the violin plot to PNG and return its path.

    Parameters
    ----------
    results : dict | None, optional
        Pre-loaded results; loaded from disk when ``None``.

    Returns
    -------
    Path
        The written PNG path.

    Raises
    ------
    FileNotFoundError
        If the results file is missing and none were passed.
    """
    import vl_convert as vlc

    # Load the stats if not supplied.
    if results is None:
        if not _RESULTS_PATH.is_file():
            raise FileNotFoundError(
                f"{_RESULTS_PATH} introuvable — lancez `python -m eval.crossval`."
            )
        results = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))

    spec = build_spec(results)
    # 2× scale for a crisp retina PNG in the docs.
    png = vlc.vegalite_to_png(vl_spec=json.dumps(spec), scale=2.0)
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_bytes(png)
    return _OUT_PATH


def main(argv: list[str] | None = None) -> int:
    """CLI: render the violin plot and print where it landed.

    Parameters
    ----------
    argv : list[str] | None, optional
        Unused; present for a uniform CLI signature.

    Returns
    -------
    int
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = render()
    print(f"Violin plot (5 moteurs) écrit : {path}")
    # Also render the LLM shootout violin when its results are present.
    if _SHOOTOUT_RESULTS.is_file():
        print(f"Violin plot (LLM shootout) écrit : {render_shootout()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
