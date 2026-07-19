"""Confusion-matrix heatmaps (Vega-Lite → PNG), one per engine — all 8.

For each of the 8 engines we count, over a held-out test set, which intent it
predicted (columns) for every true intent (rows), plus an ``Abstention`` /
``Abstain`` column for the "hand off to a human" cases. Rendered as a
**heatmap**: the diagonal is where the engine is right, off-diagonal cells are
its confusions. Each matrix uses that engine's own house colour (white →
colour), matching every other figure in the repo.

* the four **trainable** classifiers are run live over the 88 held-out
  paraphrases;
* the four **LLM** configs (model × examples) reuse the cached predictions of
  the prompt experiment (``eval/.llm_shootout/``), a 30-example sample.

Intent labels are prettified for the axes: ``vol_vehicule`` → ``Vol Véhicule``.

Usage
-----
    python -m eval.confusion            # all 8 engines, both languages

Author
------
Project maintainers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from intent_engine.config import get_settings
from intent_engine.router import IntentRouter

from .thresholds import load_dataset
from .violin import _IMG_DIR, ENGINES, _render

logger = logging.getLogger(__name__)

_SHOOT_CACHE = Path(__file__).resolve().parent / ".llm_shootout"

# Per-word French spelling with the right accents (lowercase). French uses
# *sentence case*: an intent id like ``vol_vehicule`` reads ``Vol véhicule``
# and ``faire_reclamation`` reads ``Faire réclamation`` (only the first word is
# capitalised, accents restored, no underscores).
_ACCENT: dict[str, str] = {
    "vehicule": "véhicule", "declarer": "déclarer", "depannage": "dépannage",
    "resilier": "résilier", "probleme": "problème", "degat": "dégât",
    "sante": "santé", "prevoyance": "prévoyance", "a": "à",
    "reclamation": "réclamation",
}

_ABSTAIN_KEY = "__abstain__"  # sentinel id (never a real intent)
_ABSTAIN_LABEL = {"fr": "Abstention", "en": "Abstain"}

_TEXT = {
    "fr": {"title": "Matrice de confusion", "x": "Prédit", "y": "Réel"},
    "en": {"title": "Confusion matrix", "x": "Predicted", "y": "True"},
}

_ENGINE_BY_KEY = {e["key"]: e for e in ENGINES}


def _pretty(intent_id: str, lang: str) -> str:
    """Return a human, accented, space-separated label for an intent id.

    Parameters
    ----------
    intent_id : str
        Snake-case intent id (or the abstain sentinel).
    lang : str
        ``"fr"`` or ``"en"``.

    Returns
    -------
    str
        e.g. ``"vol_vehicule"`` → ``"Vol Véhicule"``.
    """
    if intent_id == _ABSTAIN_KEY:
        return _ABSTAIN_LABEL[lang]
    # Sentence case: accents restored, words lowercase, only the FIRST word
    # capitalised ("faire_reclamation" → "Faire réclamation").
    words = [_ACCENT.get(w, w) for w in intent_id.split("_")]
    label = " ".join(words)
    return label[:1].upper() + label[1:]


def _confusion_cv(engine: str, router: IntentRouter) -> dict:
    """Confusion counts for a trainable engine, run live over the 88-set."""
    counts: dict[tuple[str, str], int] = {}
    for case in load_dataset():
        result = router.classify(case["text"], engine)
        top = result.ranked[0] if result.ranked else None
        pred = top.intent if (top is not None and result.confident) else _ABSTAIN_KEY
        counts[(case["expected"], pred)] = counts.get((case["expected"], pred), 0) + 1
    return counts


def _confusion_llm(engine: dict, intents: list[str]) -> dict:
    """Confusion counts for an LLM config, from its cached shootout predictions."""
    safe = engine["model"].replace("/", "_").replace(":", "_")
    path = _SHOOT_CACHE / f"{safe}__{engine['prompt']}.json"
    if not path.is_file():
        return {}
    cache: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[tuple[str, str], int] = {}
    # Only the examples the LLM actually classified (the cached ones) — the LLM
    # runs on a smaller sample than the classifiers (it is far slower).
    for case in load_dataset():
        if case["text"] not in cache:
            continue
        pred = cache[case["text"]]
        if pred not in intents:  # empty, hors_perimetre or unknown → abstain
            pred = _ABSTAIN_KEY
        counts[(case["expected"], pred)] = counts.get((case["expected"], pred), 0) + 1
    return counts


def build_heatmap_spec(
    intents: list[str], counts: dict, engine_key: str, lang: str
) -> dict:
    """Build a Vega-Lite confusion-matrix heatmap spec in ``lang``.

    Parameters
    ----------
    intents : list[str]
        Intent ids, in matrix order.
    counts : dict
        ``{(true_id, pred_id): count}`` (pred may be the abstain sentinel).
    engine_key : str
        Engine key (drives colour + title).
    lang : str
        ``"fr"`` or ``"en"``.

    Returns
    -------
    dict
        A Vega-Lite v5 heatmap specification.
    """
    engine = _ENGINE_BY_KEY[engine_key]
    colour = engine["color"]
    label = engine[lang].replace("\n", " ")

    id_x = intents + [_ABSTAIN_KEY]
    x_labels = [_pretty(i, lang) for i in id_x]
    y_labels = [_pretty(i, lang) for i in intents]
    rows = [
        {
            "true": _pretty(t, lang),
            "pred": _pretty(p, lang),
            "count": counts.get((t, p), 0),
        }
        for t in intents
        for p in id_x
    ]

    x_enc = {
        "field": "pred",
        "type": "nominal",
        "sort": x_labels,
        "title": _TEXT[lang]["x"],
        "axis": {"labelAngle": -45, "labelFontSize": 9},
    }
    y_enc = {
        "field": "true",
        "type": "nominal",
        "sort": y_labels,
        "title": _TEXT[lang]["y"],
        "axis": {"labelFontSize": 9},
    }

    total = sum(counts.values())
    sub = {
        "fr": f"jeu tenu à l'écart · {total} phrases · {len(intents)} intentions",
        "en": f"held-out test · {total} utterances · {len(intents)} intents",
    }[lang]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": {
            "text": f"{_TEXT[lang]['title']} — {label}",
            "subtitle": sub,
            "subtitleColor": "#6E6E73",
            "font": "Roboto",
            "subtitleFont": "Roboto",
            "anchor": "start",
            "fontSize": 16,
        },
        "data": {"values": rows},
        "config": {
            "font": "Roboto",
            "view": {"stroke": None},
            "axis": {"labelFont": "Roboto", "titleFont": "Roboto"},
        },
        "width": 480,
        "height": 480,
        "layer": [
            {
                "mark": {"type": "rect", "stroke": "#EDEDED", "strokeWidth": 0.5},
                "encoding": {
                    "x": x_enc,
                    "y": y_enc,
                    "color": {
                        "condition": {"test": "datum.count === 0", "value": "#FFFFFF"},
                        "field": "count",
                        "type": "quantitative",
                        "scale": {"range": ["#FFFFFF", colour], "domainMin": 0},
                        "legend": {"title": None},
                    },
                },
            },
            {
                "transform": [{"filter": "datum.count > 0"}],
                "mark": {"type": "text", "font": "Roboto", "fontSize": 8},
                "encoding": {
                    "x": x_enc,
                    "y": y_enc,
                    "text": {"field": "count", "type": "quantitative"},
                    "color": {"value": "#1C1C1E"},
                },
            },
        ],
    }


def run() -> list[Path]:
    """Compute + render a heatmap for every one of the 8 engines, both languages.

    Returns
    -------
    list[Path]
        The written PNG paths.
    """
    router = IntentRouter.from_directory(get_settings().knowledge_base_dir)
    available = router.available_engines()
    intents = list(router.kb.intent_ids())
    written: list[Path] = []
    for engine in ENGINES:
        if engine["kind"] == "cv":
            if engine["key"] not in available:
                continue
            counts = _confusion_cv(engine["key"], router)
        else:  # llm config from the shootout cache
            counts = _confusion_llm(engine, intents)
            if not counts:
                continue
        for lang in ("fr", "en"):
            spec = build_heatmap_spec(intents, counts, engine["key"], lang)
            out = _IMG_DIR / f"confusion-{engine['key']}-{lang}.png"
            _render(spec, out)
            written.append(out)
            logger.info("wrote %s", out)
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: render every confusion-matrix heatmap."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in run():
        print(f"Figure écrite : {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
