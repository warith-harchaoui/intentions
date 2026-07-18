# Examples — Déraison Assurances intent engine

A runnable cookbook for the three-approaches intent engine. Every snippet is
copy-pasteable. English per the coding standard; the domain data and answers
are French (it is a French insurance assistant).

See [`README.md`](README.md) for install, and [`PROS_CONS.md`](PROS_CONS.md)
for the sourced comparison of the three approaches.

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
ollama pull gemma4:e4b          # on Apple Silicon, gemma4:e4b-mlx is auto-picked
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

## 2. CLI — compare the three engines

```bash
python -m intent_engine compare "on m'a volé ma voiture cette nuit"
```

```text
tfidf  | vol_vehicule  [0.86]  — 0 ms
bert   | vol_vehicule  [0.61]  — 30 ms
llm    | vol_vehicule  [1.00]  — 10394 ms
        slots: {'type_bien': 'véhicule', 'urgence': 'haute'}
```

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

# Which engines can run right now (LLM only if Ollama answers)?
print(router.available_engines())
# ['tfidf', 'bert', 'llm']

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

Type or dictate a request, compare the three engines with confidence bars and
latencies, toggle "read the answer aloud", and browse the knowledge base.

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
[PASS] tfidf  accuracy=97% (32/33) mean_latency=0ms (bars: acc≥75%, lat≤50ms)
    [PASS] abstention hors-périmètre: 100% (bar ≥75%)
[PASS] bert   accuracy=82% (27/33) mean_latency=15ms (bars: acc≥80%, lat≤2000ms)
    [PASS] abstention hors-périmètre: 75% (bar ≥50%)
```

DeepEval (LLM) and Giskard (ML) integrations run with the eval extra:

```bash
pip install ".[eval]"
pytest -m slow eval/test_eval_deepeval.py   # DeepEval, exact-intent-match metric
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
