# Experimental protocol

This document fixes the **scientific protocol** for every number in this repo,
so results are reproducible and not "story-driven". Storytelling (README) comes
*after* this; the numbers come from here.

---

## 1. Task

Single-label **intent classification** over **K = 21** insurance intents
(French). One utterance in, one intent out (or an explicit *abstention* that
hands off to a human). The catalogue lives in `knowledge_base/*.md`
(one `# h1` = one intent).

---

## 2. Data (the foundation : everything else depends on it)

Generated with a local LLM (`scripts/generate_examples.py`, gemma3:4b), then
integrated by `scripts/integrate_examples.py`. Design constraints, all
**verifiable** with the command in §2.1:

| Split | Size | Per intent | File |
|-------|-----:|-----------:|------|
| **Train** | **1008** | **48** (exactly, all 21) | `knowledge_base/*.md` |
| **Test** (held-out) | **210** | **10** (exactly, all 21) | `eval/dataset.jsonl` |
| **Out-of-scope** | 15 |, (off-topic) | `eval/dataset_oos.jsonl` |

Guarantees:

- **Balanced** : every intent has exactly 48 train / 10 test (no majority-class
  inflation, no intent starved).
- **Disjoint** : train ∩ test = ∅ (**0** overlap, case-insensitive). No leakage.
- **Deduplicated** : the 1008 training utterances are unique.
- **Held-out** : the test set never enters training or few-shot prompts.

### 2.1 Verify it yourself

```bash
python - <<'PY'
import json
from collections import Counter
from intent_engine.router import IntentRouter
from intent_engine.config import get_settings
r = IntentRouter.from_directory(get_settings().knowledge_base_dir)
train = {e.strip().lower() for i in r.kb.intents for e in i.examples}
test = [json.loads(l) for l in open("eval/dataset.jsonl")]
tc = Counter(t["expected"] for t in test)
trc = {i.intent_id: len(i.examples) for i in r.kb.intents}
assert set(trc.values()) == {48}, trc          # balanced train
assert set(tc.values()) == {10}, tc             # balanced test
assert train & {t["text"].strip().lower() for t in test} == set()  # no leakage
print("OK: 21×48 train, 21×10 test, balanced, disjoint, deduplicated")
PY
```

### 2.2 Honest caveats

- The utterances are **synthetic** (LLM-generated), styled as realistic French
  customer messages with registers, typos and some genuine ambiguity. Ambiguity
  is a property of the *domain* (real messages are ambiguous), so a perfect
  score is neither expected nor desirable.
- Generation temperature is non-zero, so the *generator* is not bit-reproducible;
  reproducibility is guaranteed **from the saved data** (`knowledge_base/`,
  `eval/*.jsonl`), which is versioned.

---

## 3. Engines under test

Four **trainable** classifiers (trained on the 1008-train) and the **LLM**
(zero-/few-shot, no training):

1. **TF-IDF + Random Forest** : char/word n-grams → forest.
2. **fastText (learned)** : subword embeddings trained on our examples → softmax.
3. **fastText (pretrained)** : cc.fr.300 sentence vectors → logistic regression.
4. **BERT + MLP** : SBERT sentence embeddings → PyTorch MLP.
5. **LLM** : `gemma3:4b` (and `qwen2.5:3b` for the prompt study) via Ollama,
   strict-JSON output. Prompt kept in `locales/i18n.yaml`, per language.

---

## 4. Evaluation

Every engine is scored **on the same held-out 210-test, spanning all 21
intents**. No engine sees a different sample.

- **Held-out accuracy** : fraction correct on the 210-test (`eval/harness.py`,
  `eval/crossval.py`). This is the headline generalisation number.
- **Repeated 5-fold cross-validation** : for the *trainable* engines only:
  5 folds × 5 shuffles = **25 fits**, stratified, on the training pool
  (`eval/crossval.py`). Measures variance and in-distribution accuracy. The LLM
  has no CV (it is not trained), it contributes a single held-out point.
- **Confusion matrices** : K×K counts on the 210-test, one per engine, with an
  `Abstain` column (`eval/confusion.py`).
- **Out-of-scope abstention** : fraction of the 15 off-topic inputs each engine
  correctly refuses (`eval/thresholds.py`). Refusing to guess is a first-class
  metric.

### 4.1 Prompt sub-study (LLM only)

A **2×2**: model (qwen2.5:3b vs gemma3:4b) × examples (zero-shot vs few-shot),
at a fixed engineered prompt, on the same test set. Few-shot examples are
*fresh* (never in train or test) to avoid leakage. Isolates what a bigger model
vs a few examples each buy.

---

## 5. Reproducibility

- Seeds fixed in `eval/crossval.py` (fold shuffles) and the classifiers.
- LLM decoding at temperature 0; predictions **cached per configuration**
  (`eval/.llm_shootout/`, `eval/.llm_cache.json`) so re-runs are free and the
  reported numbers are stable across runs.
- Figures regenerate from the saved results: `python -m eval.crossval`,
  `python -m eval.violin`, `python -m eval.confusion`.

---

## 6. What we do *not* claim

- Not a production system (no auth, no rate-limiting, single-turn).
- Synthetic data ≠ real traffic; absolute numbers are indicative, the
  **relative** comparison under one fixed protocol is the point.
- Scale beyond 21 intents / 1008 examples is untested.
