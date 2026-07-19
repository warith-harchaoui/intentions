"""Generate an equi-distributed dataset (train + test) with the local LLM.

Targets a **balanced** corpus: ``--per-train`` utterances per intent for
training (appended to the KB) and ``--per-test`` held-out utterances per intent
for the paraphrase test set — disjoint from training. Paraphrases are produced
by the local Ollama model (one batched call per intent), lightly cleaned and
deduplicated (case-insensitive) against the existing KB and each other.

Output is ``eval/generated_examples.json`` in the shape
``scripts/integrate_examples.py`` consumes: ``{"train": {id: [...]},
"eval": {id: [...]}, "oos": [...]}``. Run that script next to integrate.

Usage
-----
    python scripts/generate_examples.py --per-train 48 --per-test 10

Author
------
Project maintainers.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from intent_engine.config import get_settings
from intent_engine.ollama_client import OllamaClient
from intent_engine.router import IntentRouter

_ROOT = Path(__file__).resolve().parent.parent
_GENERATED = _ROOT / "eval" / "generated_examples.json"
_OOS = _ROOT / "eval" / "dataset_oos.jsonl"


def _norm(text: str) -> str:
    """Casefold + strip accents/punctuation for dedup comparison."""
    ascii_text = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    return re.sub(r"[^a-z0-9 ]+", "", ascii_text.lower()).strip()


def _generate_for_intent(
    client: OllamaClient, model: str, intent, want: int
) -> list[str]:
    """Ask the LLM for ``want`` fresh, varied French paraphrases for an intent.

    Parameters
    ----------
    client : OllamaClient
        Ollama client.
    model : str
        Model tag.
    intent : Intent
        The intent (title + a few seed examples ground the prompt).
    want : int
        How many paraphrases to request.

    Returns
    -------
    list[str]
        Cleaned candidate utterances (order preserved, not yet deduped).
    """
    seeds = " / ".join(intent.examples[:5])
    prompt = (
        f"Tu écris des phrases de clients d'assurance (français familier, oral et "
        f"écrit, avec parfois des fautes de frappe). Intention : « {intent.title} ». "
        f"Exemples : {seeds}.\n"
        f"Génère {want} NOUVELLES phrases variées et réalistes pour CETTE intention, "
        f"différentes des exemples. Une phrase par ligne, sans numéro ni tiret, "
        f"sans commentaire."
    )
    raw = client.chat(model, [{"role": "user", "content": prompt}], temperature=0.9)
    out: list[str] = []
    for line in raw.splitlines():
        # Strip leading numbering / bullets / quotes the model may add.
        clean = re.sub(r'^\s*(?:\d+[.)]\s*|[-*•]\s*|["«»]\s*)', "", line).strip()
        clean = clean.strip('"«»').strip()
        if 3 <= len(clean) <= 160 and " " in clean:
            out.append(clean)
    return out


def run(per_train: int, per_test: int) -> dict:
    """Generate the balanced dataset and write ``generated_examples.json``.

    Parameters
    ----------
    per_train : int
        Target training utterances per intent (KB total after integration).
    per_test : int
        Held-out test utterances per intent.

    Returns
    -------
    dict
        The written ``generated_examples`` structure.
    """
    settings = get_settings()
    router = IntentRouter.from_directory(settings.knowledge_base_dir)
    client = OllamaClient(
        settings.ollama_base_url, timeout_s=settings.request_timeout_s
    )
    model = settings.llm_model

    train_new: dict[str, list[str]] = {}
    eval_by: dict[str, list[str]] = {}

    for intent in router.kb.intents:
        existing = {_norm(e) for e in intent.examples}
        need_train = max(0, per_train - len(intent.examples))
        # Over-generate so dedup still leaves enough for train + test.
        want = need_train + per_test + 20
        seen = set(existing)
        pool: list[str] = []
        # Up to two batched calls to reach the pool size after dedup.
        for _ in range(2):
            for cand in _generate_for_intent(client, model, intent, want):
                key = _norm(cand)
                if key and key not in seen:
                    seen.add(key)
                    pool.append(cand)
            if len(pool) >= need_train + per_test:
                break
        train_new[intent.intent_id] = pool[:need_train]
        eval_by[intent.intent_id] = pool[need_train : need_train + per_test]
        got_t = len(intent.examples) + len(train_new[intent.intent_id])
        got_e = len(eval_by[intent.intent_id])
        print(f"{intent.intent_id:32} train={got_t:3d} test={got_e:2d}", flush=True)

    # Preserve the existing out-of-scope set (rewritten verbatim by integrate).
    oos = [json.loads(line)["text"] for line in _OOS.read_text().splitlines() if line]
    data = {"train": train_new, "eval": eval_by, "oos": oos}
    _GENERATED.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_train = sum(len(i.examples) for i in router.kb.intents) + sum(
        len(v) for v in train_new.values()
    )
    total_test = sum(len(v) for v in eval_by.values())
    print(f"\nTOTAL after integrate: train~{total_train}  test~{total_test}")
    return data


def main(argv: list[str] | None = None) -> int:
    """CLI: generate the balanced dataset."""
    parser = argparse.ArgumentParser(prog="generate_examples")
    parser.add_argument("--per-train", type=int, default=48)
    parser.add_argument("--per-test", type=int, default=10)
    args = parser.parse_args(argv)
    run(args.per_train, args.per_test)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
