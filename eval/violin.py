"""Render the evaluation figures (Vega-Lite → PNG), bilingual and clean.

Module summary
--------------
Two figures, each produced in **French and English** (``-fr`` / ``-en``) so
each README variant shows a chart in its own language:

* **Cross-validation violins** (``violin-accuracy-{lang}.png``) — for the four
  *trainable* engines (TF-IDF, the two fastText engines, BERT), one violin
  per engine built from the **real** accuracies of a repeated 5-fold
  cross-validation (5 folds × 5 shuffles = 25 measurements). No resampling
  tricks: every point in the density is a genuine train-on-4/5, test-on-1/5
  score. The LLM is zero-shot (it never trains), so it has no cross-validation
  and is discussed in the text, not shown here.

* **LLM shootout bars** (``shootout-{lang}.png``) — a plain bar chart of each
  (model, prompt) configuration's accuracy on the held-out sample. The LLM
  cannot be cross-validated either, so a clean bar (one number per bar) is the
  honest picture; the sample size and caveats live in the surrounding text.

The charts are deliberately **minimal** (title + axis, house palette). All the
methodology — how many intents, how many training examples, what the test set
is — is explained in the prose around the figures, not crammed into them.

Usage
-----
    python -m eval.violin        # writes the -fr and -en PNGs to docs/img/

Author
------
Project maintainers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Result files produced by the eval, and where the figures land.
_CV_RESULTS = Path(__file__).resolve().parent / "crossval_results.json"
_SHOOTOUT_RESULTS = Path(__file__).resolve().parent / "llm_shootout_results.json"
_IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"

# The four *trainable* engines shown in the cross-validation violins, in the
# pedagogical progression order, with house-palette colours (matching the
# Mermaid progression) and bilingual labels.
_ENGINE_META: dict[str, dict[str, str]] = {
    "tfidf": {"fr": "TF-IDF", "en": "TF-IDF", "color": "#007AFF"},
    "fasttext_custom": {
        "fr": "fastText\n(appris)",
        "en": "fastText\n(learned)",
        "color": "#79DBDC",
    },
    "fasttext_pretrained": {
        "fr": "fastText\n(pré-entraîné)",
        "en": "fastText\n(pretrained)",
        "color": "#AF52DE",
    },
    "bert": {"fr": "BERT", "en": "BERT", "color": "#28CD41"},
}

# One hue per LLM family (house palette) for the shootout bars.
_MODEL_HUE: dict[str, str] = {
    "qwen2.5:3b": "#007AFF",
    "gemma3:4b": "#28CD41",
    "gemma4:e2b-mlx": "#FF9500",
    "gemma4:e4b-mlx": "#AF52DE",
}

# Bilingual chart strings (kept tiny — the real explanation is in the docs).
_TEXT: dict[str, dict[str, str]] = {
    "fr": {
        "cv_title": "Exactitude par moteur (validation croisée 5 blocs)",
        "axis": "Exactitude (plus c'est haut, mieux c'est)",
        "shoot_title": "Le modèle compte plus que le prompt",
    },
    "en": {
        "cv_title": "Accuracy by engine (5-fold cross-validation)",
        "axis": "Accuracy (higher is better)",
        "shoot_title": "The model matters more than the prompt",
    },
}

# Bilingual prompt-variant labels for the shootout bar x-axis. Keys are the
# internal prompt names stored in the shootout summary ("naive" / "improved").
_PROMPT_LABEL: dict[str, dict[str, str]] = {
    "fr": {"naive": "va-vite", "improved": "soigné"},
    "en": {"naive": "quick", "improved": "polished"},
}

# House-style base config shared by every chart (Roboto, no chart-junk).
_BASE_CONFIG = {
    "font": "Roboto",
    "view": {"stroke": None},
    "axis": {"labelFont": "Roboto", "titleFont": "Roboto", "grid": False},
    "header": {"labelFont": "Roboto", "titleFont": "Roboto"},
}


def build_cv_spec(results: dict, lang: str) -> dict:
    """Build the cross-validation violin spec in ``lang``.

    Parameters
    ----------
    results : dict
        Parsed ``crossval_results.json`` (uses the ``cv`` field: a list of
        real fold accuracies per trainable engine).
    lang : str
        ``"fr"`` or ``"en"``.

    Returns
    -------
    dict
        A Vega-Lite v5 faceted-density (violin) specification.
    """
    cv: dict[str, list[float]] = results.get("cv", {})
    # Long-format rows: one per (engine, fold accuracy). Only engines present
    # in the CV results are drawn, in progression order.
    rows: list[dict[str, object]] = []
    present: list[str] = []
    colours: list[str] = []
    for engine, meta in _ENGINE_META.items():
        if engine not in cv:
            continue
        present.append(meta[lang])
        colours.append(meta["color"])
        for value in cv[engine]:
            rows.append({"engine": meta[lang], "accuracy": value})

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {
            "text": _TEXT[lang]["cv_title"],
            "font": "Roboto",
            "anchor": "start",
            "fontSize": 16,
        },
        "config": _BASE_CONFIG,
        "data": {"values": rows},
        "transform": [
            {
                "density": "accuracy",
                "groupby": ["engine"],
                "extent": [0, 1],
                "as": ["accuracy", "density"],
            }
        ],
        "mark": {"type": "area", "orient": "horizontal"},
        "width": 95,
        "height": 420,
        "encoding": {
            "y": {
                "field": "accuracy",
                "type": "quantitative",
                "title": _TEXT[lang]["axis"],
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


def build_shootout_spec(results: dict, lang: str) -> dict:
    """Build the LLM-shootout bar-chart spec in ``lang``.

    A bar chart (not a violin): the LLM is zero-shot, so there is no
    cross-validation and no honest per-config distribution to show — one clear
    bar per (model, prompt) configuration is the right picture. Sample size
    and caveats are explained in the surrounding text.

    Parameters
    ----------
    results : dict
        Parsed ``llm_shootout_results.json``.
    lang : str
        ``"fr"`` or ``"en"``.

    Returns
    -------
    dict
        A Vega-Lite v5 bar-chart specification.
    """
    summary: dict[str, dict] = results.get("summary", {})
    rows: list[dict[str, object]] = []
    order: list[str] = []
    colours: list[str] = []
    for _key, s in summary.items():
        model = s.get("model", "")
        prompt = s.get("prompt", "")
        # Two-line x label: model (shortened, no "-mlx") on top, the prompt
        # variant below — split on the newline by the axis ``labelExpr``.
        model_short = model.replace("-mlx", "")
        label = f"{model_short}\n{_PROMPT_LABEL[lang].get(prompt, prompt)}"
        order.append(label)
        colours.append(_MODEL_HUE.get(model, "#6E6E73"))
        rows.append({"config": label, "accuracy": s.get("point_accuracy", 0.0)})

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {
            "text": _TEXT[lang]["shoot_title"],
            "font": "Roboto",
            "anchor": "start",
            "fontSize": 16,
        },
        "config": {**_BASE_CONFIG, "axis": {**_BASE_CONFIG["axis"]}},
        "data": {"values": rows},
        "width": 520,
        "height": 300,
        "layer": [
            {
                "mark": {"type": "bar", "cornerRadiusEnd": 3},
                "encoding": {
                    "x": {
                        "field": "config",
                        "type": "nominal",
                        "sort": order,
                        "title": None,
                        "axis": {
                            "labelAngle": 0,
                            "labelFontSize": 11,
                            # Split "model\nprompt" into two stacked lines so the
                            # six labels never overlap.
                            "labelExpr": "split(datum.value, '\\n')",
                        },
                    },
                    "y": {
                        "field": "accuracy",
                        "type": "quantitative",
                        "title": _TEXT[lang]["axis"],
                        "axis": {"format": "%"},
                        "scale": {"domain": [0, 1]},
                    },
                    "color": {
                        "field": "config",
                        "type": "nominal",
                        "scale": {"domain": order, "range": colours},
                        "legend": None,
                    },
                },
            },
            {
                # The exact % printed above each bar — no guessing from the axis.
                "mark": {"type": "text", "dy": -6, "font": "Roboto", "fontSize": 11},
                "encoding": {
                    "x": {"field": "config", "type": "nominal", "sort": order},
                    "y": {"field": "accuracy", "type": "quantitative"},
                    "text": {
                        "field": "accuracy",
                        "type": "quantitative",
                        "format": ".0%",
                    },
                },
            },
        ],
    }


def _render(spec: dict, out: Path) -> Path:
    """Render a Vega-Lite spec to a 2× PNG at ``out``.

    Parameters
    ----------
    spec : dict
        A Vega-Lite specification.
    out : Path
        Destination PNG path.

    Returns
    -------
    Path
        ``out``.
    """
    import vl_convert as vlc

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(vlc.vegalite_to_png(vl_spec=json.dumps(spec), scale=2.0))
    return out


def render_all() -> list[Path]:
    """Render every figure in both languages and return the written paths.

    Returns
    -------
    list[Path]
        Paths of the PNGs written (CV violins, plus shootout bars if the
        shootout results are present).
    """
    written: list[Path] = []
    # Cross-validation violins (require the crossval results).
    if _CV_RESULTS.is_file():
        cv = json.loads(_CV_RESULTS.read_text(encoding="utf-8"))
        for lang in ("fr", "en"):
            written.append(
                _render(
                    build_cv_spec(cv, lang), _IMG_DIR / f"violin-accuracy-{lang}.png"
                )
            )
    # Shootout bars (only when the shootout has been run).
    if _SHOOTOUT_RESULTS.is_file():
        shoot = json.loads(_SHOOTOUT_RESULTS.read_text(encoding="utf-8"))
        for lang in ("fr", "en"):
            written.append(
                _render(
                    build_shootout_spec(shoot, lang), _IMG_DIR / f"shootout-{lang}.png"
                )
            )
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI: render all figures (both languages) and print the paths.

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
    for path in render_all():
        print(f"Figure écrite : {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
