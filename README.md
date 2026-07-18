# Déraison Assurances — one *intent engine*, three ways

[🇫🇷 Français](LISEZMOI.md) · [🇬🇧 English](README.md) — 📖 User guide: [🇫🇷 MODEDEMPLOI](MODEDEMPLOI.md) · [🇬🇧 USERGUIDE](USERGUIDE.md)

> "My colleagues keep asking me **how you build** an intent-detection engine."
> — This repo answers by showing **three ways** to do it, side by side, on a
> concrete case: the routing chatbot of a (fictional) insurance company that
> helps its customers on the **phone** (voice) and in **writing**.

![Three-engine comparator](docs/img/02-comparateur-3-moteurs.png)

A customer says *"I had an accident this morning, my car is dented"* and the
system must understand the **intent** (`declarer_sinistre_auto`), route to the
right **department**, and ideally extract the **useful entities** (urgency,
kind of asset). Three engines do this job, from the most "old-school" to the
most "brute-force":

| # | Engine | Tech | The trade-off |
|---|--------|------|---------------|
| 1 | **TF-IDF** | scikit-learn (n-grams + logistic regression) | Instant, tiny, offline. Sticks to words. |
| 2 | **BERT** | Sentence embeddings (SBERT) + ML classifier | Understands meaning, generalises to paraphrases. |
| 3 | **LLM** | Gemma, local via Ollama, prompt + **strict JSON** | Zero training, extracts slots. The slowest. |

> The UI is French because it is a French insurance assistant; the code,
> docstrings and this README are English. A full French mirror lives in
> [`LISEZMOI.md`](LISEZMOI.md).

📊 The detailed, sourced comparison (benchmarks, GDPR, costs): **[`PROS_CONS.md`](PROS_CONS.md)**.
📖 The step-by-step user guide (with screenshots): **[`USERGUIDE.md`](USERGUIDE.md)**.
🍳 The runnable cookbook: **[`EXAMPLES.md`](EXAMPLES.md)**.
📐 The coding standard followed everywhere: **[`CODING.md`](CODING.md)**.

---

## The core idea: knowledge lives in Markdown

**One `# h1` heading = one intent.** A domain expert adds an intent by writing
Markdown in `knowledge_base/`, **without touching any code**:

```markdown
# declarer_sinistre_auto

> **Titre** : Déclarer un sinistre automobile
> **Service** : Gestion des sinistres auto
> **Action** : route:sinistres_auto

## Exemples
- J'ai eu un accident de voiture
- Mon pare-brise est fissuré
- On m'a rentré dedans au feu rouge

## Réponse
Je vous mets en relation avec le service sinistres auto…
```

The `## Exemples` are the **training data** for TF-IDF and BERT, and the
**few-shot** examples for the LLM. The `## Réponse` is the scripted answer read
back. Full format: [`knowledge_base/_FORMAT.md`](knowledge_base/_FORMAT.md).

---

## Install

Requirements: **Python ≥ 3.10**. For the LLM engine (and the BERT embedding
fallback), a local **Ollama**.

### 1. Ollama (for the LLM engine)

- macOS 🍎 : `brew install ollama` (install `brew` via [brew.sh](https://brew.sh/)), then `ollama serve`
- Ubuntu 🐧 : `curl -fsSL https://ollama.com/install.sh | sh`
- Windows 🪟 : `winget install Ollama.Ollama`

Then pull the models:

```bash
ollama pull gemma4:e4b          # LLM engine (on Apple Silicon: gemma4:e4b-mlx)
ollama pull nomic-embed-text    # embedding fallback for the BERT engine
```

### 2. The project

```bash
python -m venv .venv
source .venv/bin/activate         # Windows 🪟 : .venv\Scripts\activate
pip install -r requirements.txt

# Optional — the "proper" SBERT path for the BERT engine (pulls PyTorch, ~2 GB):
pip install "sentence-transformers>=3.0.0"
# Optional — the evaluation layer (DeepEval + Giskard):
pip install ".[eval]"
```

> Without `sentence-transformers`, the BERT engine **automatically falls back**
> to the Ollama embeddings (`nomic-embed-text`): the demo still runs.

---

## Quickstart

### The web app (the nice front end)

```bash
./start.sh                        # or: uvicorn intent_engine.api:app --port 8000
# then open http://localhost:8000
```

Type **or dictate** a request (browser speech recognition — *vocal-helper*),
compare the three engines with confidence bars and latencies, **read the answer
aloud** (speech synthesis — *speech-helper*), and browse the knowledge base.

### Command line

```bash
python -m intent_engine intents                       # list the intents
python -m intent_engine compare "j'ai eu un accident, ma voiture est cabossée"
python -m intent_engine classify --engine tfidf "je veux résilier"
python -m intent_engine execute "il me faut une prise en charge pour l'hôpital"
```

Example `compare` output:

```text
tfidf   | declarer_sinistre_auto  [0.86]  — 1 ms
bert    | declarer_sinistre_auto  [0.59]  — 39 ms
llm     | declarer_sinistre_auto  [1.00]  — 16565 ms
        slots: {'type_bien': 'auto', 'urgence': 'haute'}
```

---

## Measured results (21-intent KB, 33-example eval set)

Reproducible with `python -m eval.harness`.

| Engine | Accuracy | Mean latency | Slots |
|---|---|---|---|
| **TF-IDF** | 97 % (32/33) | ~1 ms | ❌ |
| **BERT — SBERT** | 82 % (27/33) | ~15 ms | ❌ |
| **BERT — Ollama fallback** | 79 % (26/33) | ~28 ms | ❌ |
| **LLM — gemma4:e4b** | 94 % (31/33) | ~20 s | ✅ |

> Plus an **out-of-scope abstention** check: on 8 off-topic inputs (weather,
> maths, cooking…), TF-IDF and the LLM abstain 100 % of the time — they say
> "je ne sais pas" instead of misrouting.

> At equal accuracy, the LLM is **~20,000× slower** than TF-IDF but extracts
> slots. *Heavier ≠ better: choose by need.* Details and sources in
> [`PROS_CONS.md`](PROS_CONS.md).

---

## Architecture

```
intent_engine/
  kb.py            # Markdown parser: # h1 = intent
  base.py          # shared contracts: IntentEngine, IntentResult
  tfidf_engine.py  # Approach 1 — scikit-learn
  embeddings.py    # pluggable embedding backends (SBERT / Ollama)
  bert_engine.py   # Approach 2 — embeddings + classifier
  llm_engine.py    # Approach 3 — Ollama + strict JSON + anti-hallucination
  ollama_client.py # synchronous Ollama client (JSON chat + embeddings)
  router.py        # engine registry + comparison + execution
  api.py           # FastAPI app
  cli.py           # terminal interface
knowledge_base/    # the knowledge (Markdown, h1 = intent)
web/               # vanilla JS + Tailwind front (+ self-hosted fonts)
eval/              # labelled dataset + thresholds + harness + DeepEval + Giskard
tests/             # pytest
```

All three engines implement **the same contract** (`IntentEngine`), so the
router, the API and the front end treat them identically. That is the teaching
point: only the **representation** changes.

---

## Tests & evaluation

```bash
pytest -m "not slow"                   # fast suite (deterministic, no network)
pytest                                 # full suite (real BERT backend + Ollama)
python -m eval.harness                 # accuracy/latency of all three engines
```

The evaluation layer (coding standard rule 14) ships a **labelled dataset**
(`eval/dataset.jsonl`), **versioned thresholds** (`eval/thresholds.py`), a
dependency-free **harness**, and **DeepEval** (LLM) + **Giskard** (ML)
integrations enabled by `pip install ".[eval]"`.

---

## Privacy

The **intent engine** runs **locally** (scikit-learn, self-hosted SBERT, LLM
via Ollama): the text of a request never leaves the machine — a deliberate
choice, because in insurance requests can contain **health data** (GDPR art. 9).

> ⚠️ **Honest caveat about voice.** The web UI's speech features use the
> browser's Web Speech API. In Chrome, **speech recognition sends audio to
> Google's servers** — so the *voice* path is *not* local, unlike the NLU. For a
> genuinely local voice pipeline, plug a server-side *vocal-helper*
> (whisper.cpp) for speech-to-text and an OSS TTS for the reply. This is called
> out in [`ASSESSMENT.md`](ASSESSMENT.md) and in the UI itself.

Details and the GDPR discussion in [`PROS_CONS.md`](PROS_CONS.md).

---

## Acknowledgements

Special thanks to the contributors, reviewers, and users who helped improve
this project.

AI tools may have been used during development, but authorship and
responsibility remain with the human maintainers.
