# Examples — Déraison Assurances intent engine

A runnable cookbook for the five-approaches intent engine. Every snippet is
copy-pasteable. English per the coding standard; the domain data and answers
are French (it is a French insurance assistant).

See [`README.md`](README.md) for install, and [`PROS_CONS.md`](PROS_CONS.md)
for the sourced comparison of the five approaches.

---

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional SBERT path for the BERT engine (else it uses the Ollama fallback):
pip install "sentence-transformers>=3.0.0"
```

For the LLM engine, run Ollama and pull the models:

```bash
ollama pull gemma3:4b           # compact, fast LLM (~5 s/call warm)
ollama pull nomic-embed-text    # embedding fallback for the BERT engine
```

---

## 1. CLI — list the intents

```bash
python -m intent_engine intents
```

```text
21 intentions dans la base de connaissance :

  souscrire_assurance_auto         Souscrire une assurance auto  (10 exemples)
  declarer_sinistre_auto           Déclarer un sinistre automobile  (11 exemples)
  ...
```

## 2. CLI — compare the five engines

```bash
python -m intent_engine compare "on s'est rentrés dedans à un carrefour"
```

```text
tfidf               | (abstention)                 — 28 ms
fasttext_custom     | declarer_sinistre_auto [0.33] — 0 ms
fasttext_pretrained | (abstention)                 — 0 ms
bert                | declarer_sinistre_auto [0.98] — 18 ms
llm                 | declarer_sinistre_auto [0.95] — 4725 ms
        slots: {'type_bien': 'auto', 'urgence': 'haute'}
```

Watch the lexical engines abstain on a paraphrase while the semantic ones
nail it — the whole pedagogical point in one command.

## 3. CLI — one engine, natural-language execution

```bash
python -m intent_engine execute --engine tfidf "je voudrais résilier mon assurance auto"
```

```text
→ Intention : Résilier un contrat (resilier_contrat)
→ Service   : Gestion des contrats
→ Action    : route:gestion_contrats

Je comprends. Grâce à la loi Hamon, après un an d'engagement vous pouvez...
```

---

## 4. Python — use an engine directly

```python
from intent_engine import KnowledgeBase, TfidfIntentEngine

kb = KnowledgeBase.from_directory("knowledge_base")
engine = TfidfIntentEngine().fit(kb)

result = engine.classify("mon pare-brise est fissuré")
print(result.top().intent)      # declarer_sinistre_auto
print(round(result.top().score, 2))
# 0.7
```

## 5. Python — the router (compare + execute)

```python
from intent_engine import IntentRouter

router = IntentRouter.from_directory("knowledge_base")

# Which engines can run right now (fastText-pretrained only if cc.fr.300 is
# downloaded; LLM only if Ollama answers)?
print(router.available_engines())
# ['tfidf', 'fasttext_custom', 'fasttext_pretrained', 'bert', 'llm']

# Run all engines on one utterance:
for name, res in router.compare("j'ai un dégât des eaux chez moi").items():
    top = res.top()
    print(name, top.intent if top else "(abstention)", f"{res.latency_ms:.0f}ms")

# Turn a request into a routing action + slots:
execution = router.execute("je dois être hospitalisé, prise en charge svp", "llm")
print(execution.action)   # route:remboursements_sante
print(execution.slots)    # {'type_service': 'hospitalisation', ...}
```

## 6. Python — the LLM engine with strict JSON

```python
from intent_engine import KnowledgeBase, LlmIntentEngine

kb = KnowledgeBase.from_directory("knowledge_base")
engine = LlmIntentEngine().fit(kb)      # no training — captures the catalogue

result = engine.classify("quelqu'un a embouti ma portière sur le parking")
print(result.top().intent)              # declarer_sinistre_auto
print(result.slots)                     # {'urgence': ...}
print(result.meta["reformulation"])     # the model's one-line summary
```

## 7. Python — force a specific embedding backend for BERT

```python
from intent_engine import KnowledgeBase, BertIntentEngine
from intent_engine.embeddings import build_embedder

kb = KnowledgeBase.from_directory("knowledge_base")

# Force the Ollama embedding backend (no PyTorch needed):
embedder = build_embedder(backend="ollama")
engine = BertIntentEngine(embedder=embedder).fit(kb)
print(engine.classify("ma voiture est abîmée").meta["backend"])
# ollama:nomic-embed-text
```

---

## 8. Run the web app

```bash
uvicorn intent_engine.api:app --reload --port 8000
# open http://localhost:8000
```

Type a request, compare the five engines with confidence bars and latencies,
and browse the knowledge base.

## 9. HTTP API directly

```bash
# Health + which engines are usable now
curl -s localhost:8000/api/health

# Compare all engines
curl -s -X POST localhost:8000/api/compare \
  -H 'Content-Type: application/json' \
  -d '{"text":"je veux assurer ma voiture"}'

# Execute (routing action + slots)
curl -s -X POST localhost:8000/api/execute \
  -H 'Content-Type: application/json' \
  -d '{"text":"on m'\''a cambriolé","engine":"llm"}'
```

---

## 10. Evaluation

```bash
# Accuracy + latency vs versioned thresholds, all engines
python -m eval.harness

# One engine
python -m eval.harness --engine bert
```

```text
[PASS] tfidf               accuracy=49% mean_latency=30ms (bars: acc≥45%)
    [PASS] abstention hors-périmètre: 93% (bar ≥60%)
[PASS] fasttext_custom     accuracy=67% mean_latency=0ms  (bars: acc≥55%)
[PASS] fasttext_pretrained accuracy=73% mean_latency=1ms  (bars: acc≥65%)
[PASS] bert                accuracy=88% mean_latency=15ms  (bars: acc≥75%)
```

Distributions (bootstrap CIs + k-fold CV) and the violin plot:

```bash
python -m eval.crossval     # writes eval/crossval_results.json + prints mean ± std
python -m eval.violin       # renders docs/img/violin-accuracy.png
```

DeepEval (LLM) and Giskard (ML) integrations run with the eval extra:

```bash
pip install ".[eval]"
pytest -m slow eval/test_eval_deepeval.py   # DeepEval, exact-intent-match metric
# Giskard needs Python <= 3.11 (no 3.13 wheel); run it in a 3.11 venv:
pytest -m slow eval/test_eval_giskard.py    # Giskard vulnerability scan (TF-IDF)
```

---

## 11. Tests

```bash
pytest -m "not slow"     # fast, deterministic, no network
pytest                   # full suite (real BERT backend + Ollama)
```

## 12. Lint (PEP 8, enforced)

```bash
pip install ruff
ruff check intent_engine/ tests/ eval/
ruff format --check intent_engine/ tests/ eval/
```
