# Déraison Assurances — one *intent engine*, five ways

[🇫🇷 Français](LISEZMOI.md) · [🇬🇧 English](README.md) — 📖 User guide: [🇫🇷 MODEDEMPLOI](MODEDEMPLOI.md) · [🇬🇧 USERGUIDE](USERGUIDE.md)

> "My colleagues keep asking me **how you build** an intent-detection engine."
> — This repo answers by showing **five ways** to do it, side by side, on a
> concrete case: the routing chatbot of a (fictional) insurance company that
> helps its customers on the **phone** (voice) and in **writing**.

![Five-engine comparator](docs/img/02-comparateur-5-moteurs.png)

A customer says *"I had an accident this morning, my car is dented"* and the
system must understand the **intent** (`declarer_sinistre_auto`), route to the
right **department**, and ideally extract the **useful entities** (urgency,
kind of asset). Five engines do this job — a deliberate **walk through the
history of NLP**, from bag-of-words to generative LLM:

| # | Engine | Representation | Classifier | The trade-off |
|---|--------|---------------|-----------|---------------|
| 1 | **TF-IDF** | sparse char/word n-grams | **Random Forest** | Instant, tiny, offline. Memorises surface forms. |
| 2 | **fastText (custom)** | subword embeddings **learned on our examples** | fastText softmax | Light; a step up from bag-of-words. |
| 3 | **fastText (pretrained)** | **cc.fr.300** French vectors (Common Crawl) | logistic regression | Transfer learning: already knows *voiture* ≈ *véhicule*. |
| 4 | **BERT** | contextual sentence embeddings (**SBERT**) | **PyTorch MLP** | Understands meaning; wins on paraphrases. Local. |
| 5 | **LLM** | — (prompt) | **Gemma** via Ollama, **strict JSON** | Zero training, extracts slots. The slowest, the smartest. |

> The UI is French because it is a French insurance assistant; the code,
> docstrings and this README are English. A full French mirror lives in
> [`LISEZMOI.md`](LISEZMOI.md).

## Why this project exists — the pedagogical goal

This is a **teaching artefact for Data Science / Machine Learning / AI**. The
point is not to ship the best classifier; it is to make a group of colleagues
who *don't* practise ML **feel, in one screen**, the single most important
idea in applied NLP: **the representation matters more than the classifier.**

Read the engine table top to bottom and you are walking the field's history:

1. **Bag-of-words (TF-IDF)** — count n-grams; the model sees *strings*, not
   *meaning*. A synonym it never saw is invisible to it.
2. **Learned subword embeddings (fastText, trained on our data)** — the model
   starts to place similar words near each other, from a few hundred examples.
3. **Pretrained embeddings (fastText cc.fr.300)** — transfer learning: knowledge
   from billions of words of French is poured in for free.
4. **Contextual embeddings (BERT/SBERT) + a neural net** — meaning that depends
   on context, plus a non-linear classifier.
5. **Generative LLM (Gemma)** — no training at all; reasoning from a prompt,
   and — uniquely — pulling structured *slots* out of the sentence.

The comparator then shows the **pay-off** with real, measured numbers (not
opinions): on a **paraphrase-heavy** test set, accuracy climbs monotonically
**49 % → 63 % → 73 % → 88 %** across engines 1→4, and the LLM adds slot
extraction on top. And crucially, it shows the **honest caveats** an ML
practitioner cares about — sampling uncertainty (bootstrap violin plots),
train/test-split variance (k-fold cross-validation), model mis-calibration
(neural nets are over-confident on out-of-scope input), and privacy (why it
all runs locally). The goal is that a non-ML colleague leaves understanding
*why* you would pick one approach over another — which is the judgement this
project exists to transmit.

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

# Optional — the SBERT + PyTorch MLP path for the BERT engine (~2 GB):
pip install "sentence-transformers>=3.0.0" torch
# Optional — the pretrained fastText engine: download cc.fr.300 (~4.5 GB):
python scripts/download_fasttext.py
# Optional — the evaluation layer (DeepEval; Giskard needs Python ≤ 3.11):
pip install ".[eval]"
```

> The demo **degrades gracefully**: without `sentence-transformers`+`torch` the
> BERT engine is unavailable; without `cc.fr.300.bin` the pretrained-fastText
> engine is hidden; without Ollama the LLM engine is hidden. TF-IDF and
> fastText-custom always run.

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

Example `compare` output (on a paraphrase — note how the lexical engines fall
apart while the semantic ones hold):

```text
tfidf               | (abstention)                — 28 ms
fasttext_custom     | declarer_sinistre_auto [0.33] — 0 ms
fasttext_pretrained | (abstention)                — 0 ms
bert                | declarer_sinistre_auto [0.98] — 18 ms
llm                 | declarer_sinistre_auto [0.95] — 14378 ms
        slots: {'type_bien': 'auto', 'urgence': 'haute'}
```

---

## Measured results (21-intent KB, 88-example paraphrase test set)

Reproducible with `python -m eval.harness` (accuracy/latency) and
`python -m eval.crossval` (bootstrap + cross-validation distributions).

The held-out test set is deliberately **paraphrase-heavy** (low lexical
overlap with the training phrasings), so it measures *generalisation*, not
vocabulary memorisation — which is exactly where the representation shows its
worth:

| # | Engine | Accuracy (held-out) | Mean latency | Slots |
|---|--------|--------------------:|-------------:|:-----:|
| 1 | **TF-IDF + Random Forest** | 49 % | ~30 ms | ❌ |
| 2 | **fastText (custom)** | 67 % | ~0 ms | ❌ |
| 3 | **fastText (pretrained cc.fr.300)** | 73 % | ~1 ms | ❌ |
| 4 | **BERT (SBERT + MLP)** | **88 %** | ~15 ms | ❌ |
| 5 | **LLM (Gemma via Ollama)** | ~90 % | ~20 s | ✅ |

**The distributions, not just the point estimates** — bootstrap resampling of
the test set (2000×) shows the engines are *genuinely* different on this hard
set (TF-IDF and BERT distributions don't even overlap), not noise apart:

![Accuracy distribution per engine (violin plot)](docs/img/violin-accuracy.png)

> **Two lenses, one honest story.** On the paraphrase set above, accuracy
> climbs 49 → 67 → 73 → 88 %. But under **k-fold cross-validation** on the
> in-distribution KB examples, the engines are *closer* (~72 / 69 / — / 82 %):
> lexical methods do fine when the test looks like the training, and collapse
> under paraphrase shift — the whole reason semantic representations exist.
>
> Out-of-scope safety net: on 15 off-topic inputs (weather, maths, cooking…),
> TF-IDF abstains ~93 % of the time; the neural BERT is more over-confident
> (~73 % after tuning its threshold) — a real lesson on **neural-net
> calibration**. Full analysis + sources in [`PROS_CONS.md`](PROS_CONS.md).

---

## Architecture

```
intent_engine/
  kb.py              # Markdown parser: # h1 = intent
  base.py            # shared contracts: IntentEngine, IntentResult
  tfidf_engine.py    # 1 — TF-IDF + Random Forest (scikit-learn)
  fasttext_engine.py # 2 & 3 — fastText supervised + pretrained cc.fr.300
  embeddings.py      # pluggable embedding backends (SBERT / Ollama)
  mlp.py             # PyTorch MLP head (scikit-learn-like fit/predict_proba)
  bert_engine.py     # 4 — SBERT embeddings + PyTorch MLP
  llm_engine.py      # 5 — Ollama + strict JSON + anti-hallucination + slots
  ollama_client.py   # synchronous Ollama client (JSON chat + embeddings)
  router.py          # engine registry + comparison + execution
  api.py             # FastAPI app
  cli.py             # terminal interface
knowledge_base/      # the knowledge (Markdown, h1 = intent)
web/                 # vanilla JS + Tailwind front (+ self-hosted fonts)
eval/                # datasets + thresholds + harness + crossval + violin
                     # + DeepEval (LLM) + Giskard (ML) integrations
tests/               # pytest
```

All five engines implement **the same contract** (`IntentEngine`), so the
router, the API and the front end treat them identically. That is the teaching
point: only the **representation + classifier** changes — the plumbing is
constant, so you can watch quality move as you climb the progression.

---

## Tests & evaluation

```bash
pytest -m "not slow"                   # fast suite (deterministic, no network)
pytest                                 # full suite (real BERT/fastText + Ollama)
python -m eval.harness                 # accuracy/latency of all five engines
python -m eval.crossval                # bootstrap + k-fold distributions
python -m eval.violin                  # render the violin plot to docs/img/
```

The evaluation layer (coding standard rule 14) ships a **labelled dataset**
(`eval/dataset.jsonl`), an **out-of-scope set** (`eval/dataset_oos.jsonl`),
**versioned thresholds** (`eval/thresholds.py`), a dependency-free **harness**,
a **statistical base** (`eval/crossval.py`: bootstrap CIs + k-fold CV) with
**violin plots** (`eval/violin.py`), and **DeepEval** (LLM) + **Giskard** (ML)
integrations enabled by `pip install ".[eval]"`. See the testing strategy in
[`ASSESSMENT.md`](ASSESSMENT.md).

---

## Privacy

The **intent engine** runs **locally** (scikit-learn, self-hosted fastText &
SBERT, LLM via Ollama): the text of a request never leaves the machine — a
deliberate choice, because in insurance a single sentence can be **sensitive
health data** under GDPR art. 9. *"Il me faut une prise en charge pour l'Institut
de cancérologie"* reveals a cancer diagnosis; sending it to a cloud LLM would
exfiltrate exactly the data the law protects most. Here it stays on the box.

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
