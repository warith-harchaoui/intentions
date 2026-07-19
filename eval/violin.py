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

# ── THE 8 ENGINES: single source of truth for colour + order ───────────────
# One entry per "engine", used by EVERY figure (the Vega charts here and the
# Mermaid diagrams in the docs) so a given engine is ALWAYS the same colour.
# Colours are the 8 chromatic hues of the house palette
# (harchaoui.org/warith/colors), one per engine, chosen distinct.
#
#   * ``kind: "cv"``  — a trainable classifier: shown as a smooth **violin**
#     (repeated cross-validation, 25 folds).
#   * ``kind: "llm"`` — an LLM configuration (model × examples): a single
#     held-out accuracy → a **Dirac** drawn as one horizontal line (in the
#     accuracy figure) and as a bar (in the shootout figure). ``model``/``prompt``
#     locate its number in ``llm_shootout_results.json``.
ENGINES: list[dict[str, str]] = [
    {"key": "tfidf", "fr": "TF-IDF", "en": "TF-IDF", "color": "#007AFF", "kind": "cv"},
    {
        "key": "fasttext_custom",
        "fr": "fastText\n(appris)",
        "en": "fastText\n(learned)",
        "color": "#79DBDC",  # Turquoise
        "kind": "cv",
    },
    {
        "key": "fasttext_pretrained",
        "fr": "fastText\n(pré-entraîné)",
        "en": "fastText\n(pretrained)",
        "color": "#AF52DE",  # Purple
        "kind": "cv",
    },
    {"key": "bert", "fr": "BERT", "en": "BERT", "color": "#28CD41", "kind": "cv"},
    {
        "key": "qwen-zs",
        "fr": "qwen2.5:3b\nzero shot",
        "en": "qwen2.5:3b\nzero shot",
        "color": "#FFCC00",  # Yellow
        "kind": "llm",
        "model": "qwen2.5:3b",
        "prompt": "good-zs",
    },
    {
        "key": "qwen-fs",
        "fr": "qwen2.5:3b\nfew shots",
        "en": "qwen2.5:3b\nfew shots",
        "color": "#FF9500",  # Orange
        "kind": "llm",
        "model": "qwen2.5:3b",
        "prompt": "good-fs",
    },
    {
        "key": "gemma-zs",
        "fr": "gemma3:4b\nzero shot",
        "en": "gemma3:4b\nzero shot",
        "color": "#FF2D55",  # Pink
        "kind": "llm",
        "model": "gemma3:4b",
        "prompt": "good-zs",
    },
    {
        "key": "gemma-fs",
        "fr": "gemma3:4b\nfew shots",
        "en": "gemma3:4b\nfew shots",
        "color": "#FF3B30",  # Red
        "kind": "llm",
        "model": "gemma3:4b",
        "prompt": "good-fs",
    },
]

# Bilingual chart strings (kept tiny — the real explanation is in the docs).
_TEXT: dict[str, dict[str, str]] = {
    "fr": {
        "cv_title": "Exactitude par moteur",
        "cv_subtitle": (
            "violons = validation croisée (25 mesures) · "
            "lignes LLM = point held-out unique (Dirac)"
        ),
        "axis": "Exactitude (plus c'est haut, mieux c'est)",
        "shoot_title": "Modèle et few shots : de mieux en mieux",
    },
    "en": {
        "cv_title": "Accuracy by engine",
        "cv_subtitle": (
            "violins = cross-validation (25 measurements) · "
            "LLM lines = single held-out point (a Dirac)"
        ),
        "axis": "Accuracy (higher is better)",
        "shoot_title": "Model and few shots: better and better",
    },
}


def _llm_accuracies(shootout: dict) -> dict[str, float]:
    """Map each LLM engine ``key`` to its held-out accuracy from the shootout.

    Parameters
    ----------
    shootout : dict
        Parsed ``llm_shootout_results.json`` (``summary`` field).

    Returns
    -------
    dict[str, float]
        ``{engine_key: accuracy}`` for the four ``kind == "llm"`` engines.
    """
    by_cfg = {
        (s.get("model", ""), s.get("prompt", "")): s.get("point_accuracy", 0.0)
        for s in shootout.get("summary", {}).values()
    }
    out: dict[str, float] = {}
    for eng in ENGINES:
        if eng["kind"] == "llm":
            acc = by_cfg.get((eng["model"], eng["prompt"]))
            if acc is not None:
                out[eng["key"]] = acc
    return out


# House-style base config shared by every chart (Roboto, no chart-junk).
_BASE_CONFIG = {
    "font": "Roboto",
    "view": {"stroke": None},
    "axis": {"labelFont": "Roboto", "titleFont": "Roboto", "grid": False},
    "header": {"labelFont": "Roboto", "titleFont": "Roboto"},
}


def build_cv_spec(cv_results: dict, shootout_results: dict, lang: str) -> dict:
    """Build the all-8-engines accuracy figure in ``lang``.

    One column per engine, in :data:`ENGINES` order, coloured by the shared
    palette. The four **trainable** engines are smooth CV **violins** (25 folds
    each); the four **LLM** configs are single held-out accuracies, each drawn
    as a **Dirac** — one horizontal line (no width = no distribution).

    Parameters
    ----------
    cv_results : dict
        Parsed ``crossval_results.json`` (``cv`` field: fold accuracies).
    shootout_results : dict
        Parsed ``llm_shootout_results.json`` (LLM config accuracies).
    lang : str
        ``"fr"`` or ``"en"``.

    Returns
    -------
    dict
        A Vega-Lite v5 faceted, layered specification.
    """
    cv: dict[str, list[float]] = cv_results.get("cv", {})
    llm_accs = _llm_accuracies(shootout_results)

    # One shared dataset; ``kind`` tells violins (many rows) from Diracs (one
    # row) so the two layers below never collide. Engines kept in ENGINES order.
    rows: list[dict[str, object]] = []
    present: list[str] = []
    colours: list[str] = []
    for eng in ENGINES:
        label = eng[lang]
        if eng["kind"] == "cv":
            if eng["key"] not in cv:
                continue
            present.append(label)
            colours.append(eng["color"])
            for value in cv[eng["key"]]:
                rows.append({"engine": label, "accuracy": value, "kind": "cv"})
        else:  # llm
            if eng["key"] not in llm_accs:
                continue
            present.append(label)
            colours.append(eng["color"])
            rows.append(
                {"engine": label, "accuracy": llm_accs[eng["key"]], "kind": "llm"}
            )

    color_enc = {
        "field": "engine",
        "type": "nominal",
        "scale": {"domain": present, "range": colours},
        "legend": None,
    }
    y_enc = {
        "field": "accuracy",
        "type": "quantitative",
        "title": _TEXT[lang]["axis"],
        "axis": {"format": "%"},
        "scale": {"domain": [0, 1]},
    }

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {
            "text": _TEXT[lang]["cv_title"],
            "subtitle": _TEXT[lang]["cv_subtitle"],
            "subtitleColor": "#6E6E73",
            "font": "Roboto",
            "subtitleFont": "Roboto",
            "anchor": "start",
            "fontSize": 16,
        },
        "config": _BASE_CONFIG,
        "data": {"values": rows},
        "facet": {
            "column": {
                "field": "engine",
                "type": "nominal",
                "sort": present,
                "header": {
                    "titleOrient": "bottom",
                    "labelOrient": "bottom",
                    "labelPadding": 6,
                    "labelFontSize": 11,
                    # Two-line engine labels (model / examples) never overlap.
                    "labelExpr": "split(datum.value, '\\n')",
                },
                "title": None,
            }
        },
        "spec": {
            "width": 70,
            "height": 420,
            "layer": [
                {
                    # Violins: the trainable engines, smoothed by KDE.
                    "transform": [
                        {"filter": "datum.kind === 'cv'"},
                        {
                            "density": "accuracy",
                            "groupby": ["engine"],
                            "extent": [0, 1],
                            "as": ["accuracy", "density"],
                        },
                    ],
                    "mark": {"type": "area", "orient": "horizontal"},
                    "encoding": {
                        "y": y_enc,
                        "x": {
                            "field": "density",
                            "type": "quantitative",
                            "stack": "center",
                            "impute": None,
                            "title": None,
                            "axis": {
                                "labels": False,
                                "ticks": False,
                                "grid": False,
                                "values": [],
                            },
                        },
                        "color": color_enc,
                    },
                },
                {
                    # LLM Diracs: one horizontal line per config at its held-out
                    # accuracy (no width = no distribution).
                    "transform": [{"filter": "datum.kind === 'llm'"}],
                    "mark": {"type": "rule", "size": 4},
                    "encoding": {"y": y_enc, "color": color_enc},
                },
            ],
        },
    }


def build_shootout_spec(results: dict, lang: str) -> dict:
    """Build the model × few-shot bar chart in ``lang``.

    A 2×2: a weaker model (``qwen2.5:3b``) vs a stronger one (``gemma3:4b``),
    each **zero-shot** and **few-shot**, at a fixed (engineered) prompt. Four
    bars, left→right = "de mieux en mieux": the bigger model buys the large
    jump, few-shot adds a little on top. A bar chart (not a violin): the LLM is
    zero-shot, so there is no cross-validation to show — one clear number per
    configuration is the honest picture; sample size lives in the text.

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
    llm_accs = _llm_accuracies(results)
    rows: list[dict[str, object]] = []
    order: list[str] = []
    colours: list[str] = []
    # The four LLM engines, in ENGINES order, each with its palette colour — the
    # SAME colour it carries in the accuracy figure and everywhere else.
    for eng in ENGINES:
        if eng["kind"] != "llm" or eng["key"] not in llm_accs:
            continue
        label = eng[lang]
        order.append(label)
        colours.append(eng["color"])
        rows.append({"config": label, "accuracy": llm_accs[eng["key"]]})

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
        "width": 460,
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
                            # Split the two-line label (model / examples) into
                            # stacked lines so the four labels never overlap.
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
    cv = (
        json.loads(_CV_RESULTS.read_text(encoding="utf-8"))
        if _CV_RESULTS.is_file()
        else {}
    )
    shoot = (
        json.loads(_SHOOTOUT_RESULTS.read_text(encoding="utf-8"))
        if _SHOOTOUT_RESULTS.is_file()
        else {}
    )
    # The all-8-engines accuracy figure (violins + LLM Dirac lines) needs both
    # the crossval and the shootout results.
    if cv:
        for lang in ("fr", "en"):
            written.append(
                _render(
                    build_cv_spec(cv, shoot, lang),
                    _IMG_DIR / f"violin-accuracy-{lang}.png",
                )
            )
    # Shootout bars (only when the shootout has been run).
    if shoot:
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
