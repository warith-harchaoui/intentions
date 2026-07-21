"""Tests for the three intent engines and the LLM parsing logic.

The TF-IDF engine is exercised for real (fast, deterministic). The BERT
engine's classifier logic is exercised with a fake embedder so we avoid
downloading a model or hitting Ollama. The LLM engine is exercised with a
fake Ollama client so its JSON parsing and anti-hallucination gate are
tested without a server. A ``slow`` test covers the real BERT path on
demand.

Author
------
Project maintainers.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import FakeOllamaClient

from intent_engine.bert_engine import BertIntentEngine
from intent_engine.kb import KnowledgeBase
from intent_engine.llm_engine import LlmIntentEngine
from intent_engine.tfidf_engine import TfidfIntentEngine


class HashingEmbedder:
    """A deterministic, network-free fake embedder for the BERT engine.

    It maps each text to a small vector of normalised character-class
    counts. It is not semantically meaningful, but it is stable and lets a
    linear classifier separate the three clearly-distinct sample intents,
    which is all the classifier-plumbing test needs.
    """

    # Backend label surfaced in the result diagnostics.
    name = "fake:hashing"

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed texts into a tiny deterministic feature matrix.

        Parameters
        ----------
        texts : list[str]
            Input sentences.

        Returns
        -------
        np.ndarray
            An ``(n, 8)`` array of normalised token-hash counts.
        """
        rows: list[list[float]] = []
        # For each text, bucket its word hashes into 8 bins. Same text →
        # same vector, and lexically-different intents land in different
        # regions, which is enough for a separable toy problem.
        for text in texts:
            bins = [0.0] * 8
            for token in text.lower().split():
                bins[hash(token) % 8] += 1.0
            rows.append(bins)
        matrix = np.asarray(rows, dtype=float)
        # L2-normalise so the geometry matches the real embedders.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def test_tfidf_classifies_and_routes(kb: KnowledgeBase) -> None:
    """TF-IDF picks the right intent on a near-training phrase."""
    engine = TfidfIntentEngine().fit(kb)
    result = engine.classify("je veux assurer mon auto")
    top = result.top()
    # The obvious souscription phrasing must win.
    assert top is not None
    assert top.intent == "assurer_voiture"
    # A confident hit carries the scripted answer.
    assert result.confident is True
    assert "devis auto" in result.response
    # Latency is measured and non-negative.
    assert result.latency_ms >= 0.0


def test_tfidf_lower_confidence_out_of_scope(kb: KnowledgeBase) -> None:
    """Out-of-scope input scores lower than an in-scope query.

    On a tiny 3-class toy KB the absolute confidence floor is easily
    exceeded (softmax over few classes), so we assert the *relative*
    property that always holds: an off-topic sentence is less confident
    than an on-topic one. The absolute-floor abstention is covered on the
    real 20-intent KB in ``test_real_kb_abstains``.
    """
    engine = TfidfIntentEngine().fit(kb)
    # A clearly on-topic phrase should score high on its intent.
    in_scope = engine.classify("je veux assurer ma voiture").top().score
    # A cooking question shares almost no n-grams with any intent.
    out_scope = (
        engine.classify("quelle est la recette de la tarte aux pommes").top().score
    )
    # The core, class-count-independent invariant.
    assert out_scope < in_scope


def test_real_kb_abstains(real_kb_dir) -> None:
    """On the real 20-intent KB, off-topic input abstains outright."""
    # With enough classes the softmax mass spreads out and the top score
    # drops below the confidence floor — the production abstention path.
    kb = KnowledgeBase.from_directory(real_kb_dir)
    engine = TfidfIntentEngine().fit(kb)
    result = engine.classify("quelle est la recette de la tarte aux pommes")
    # Off-topic → no confident routing, no scripted answer.
    assert result.confident is False
    assert result.response == ""


def test_engine_requires_two_intents(tmp_path) -> None:
    """A single-intent KB cannot train a classifier."""
    # One intent is degenerate for classification; expect a clear error.
    (tmp_path / "one.md").write_text(
        "# only\n\n## Exemples\n- a\n- b\n", encoding="utf-8"
    )
    solo = KnowledgeBase.from_directory(tmp_path)
    with pytest.raises(ValueError):
        TfidfIntentEngine().fit(solo)


def test_bert_engine_with_fake_embedder(kb: KnowledgeBase) -> None:
    """The BERT engine trains and predicts through the embedder seam."""
    # Inject the deterministic fake so no model/network is involved.
    engine = BertIntentEngine(embedder=HashingEmbedder()).fit(kb)
    result = engine.classify("je veux résilier mon contrat")
    top = result.top()
    assert top is not None
    # The resiliation phrasing should map to the resilier intent.
    assert top.intent == "resilier"
    # Diagnostics report which backend produced the vectors.
    assert result.meta["backend"] == "fake:hashing"


def test_llm_engine_parses_json(kb: KnowledgeBase) -> None:
    """The LLM engine parses a valid JSON answer into a result + slots."""
    # Scripted model answer: a valid intent with slots and high confidence.
    reply = json.dumps(
        {
            "intent": "declarer_sinistre",
            "confidence": 0.95,
            "slots": {"urgence": "haute"},
            "reformulation": "Le client déclare un sinistre.",
        }
    )
    engine = LlmIntentEngine(client=FakeOllamaClient(reply)).fit(kb)
    result = engine.classify("j'ai eu un accident")
    top = result.top()
    assert top is not None
    # The parsed intent and slots flow through to the result.
    assert top.intent == "declarer_sinistre"
    assert result.slots == {"urgence": "haute"}
    assert result.confident is True
    # The prompt must have carried the intent catalogue to the model.
    assert "declarer_sinistre" in engine._client.last_messages[1]["content"]


def test_llm_engine_rejects_hallucinated_intent(kb: KnowledgeBase) -> None:
    """An intent id absent from the catalogue triggers abstention."""
    # The model invents an id that is not in the KB — must not be trusted.
    reply = json.dumps({"intent": "acheter_une_pizza", "confidence": 0.99})
    engine = LlmIntentEngine(client=FakeOllamaClient(reply)).fit(kb)
    result = engine.classify("je voudrais une margherita")
    # Hallucinated id → no confident prediction, flagged in meta.
    assert result.confident is False
    assert result.ranked == []
    assert result.meta["error"] == "unknown_id"


def test_llm_engine_sanitizes_slots(kb: KnowledgeBase) -> None:
    """Messy model slots are flattened, capped and urgency-normalised."""
    # The model returns a nested object, an uppercase urgency, and an empty
    # key — all of which must be cleaned before reaching a downstream system.
    reply = json.dumps(
        {
            "intent": "declarer_sinistre",
            "confidence": 0.9,
            "slots": {
                "urgence": "TRÈS URGENT",
                "type_bien": "voiture",
                "": "ignored",
                "meta": {"nested": True},
            },
        }
    )
    engine = LlmIntentEngine(client=FakeOllamaClient(reply)).fit(kb)
    slots = engine.classify("j'ai eu un accident").slots
    # Urgency is mapped onto the controlled vocabulary.
    assert slots["urgence"] == "haute"
    # A plain string slot passes through untouched.
    assert slots["type_bien"] == "voiture"
    # The empty-key entry is dropped; the nested object is stringified, not
    # left as a dict (downstream expects a flat str -> str map).
    assert "" not in slots
    assert isinstance(slots["meta"], str)


def test_llm_engine_strips_markdown_fences(kb: KnowledgeBase) -> None:
    """A JSON answer wrapped in a ```json fence is still parsed."""
    # Small local models often ignore JSON mode and fence their answer; the
    # engine must recover it rather than abstaining.
    fenced = (
        '```json\n{"intent": "declarer_sinistre", "confidence": 0.9, "slots": {}}\n```'
    )
    engine = LlmIntentEngine(client=FakeOllamaClient(fenced)).fit(kb)
    top = engine.classify("j'ai eu un accident").top()
    assert top is not None
    assert top.intent == "declarer_sinistre"


def test_llm_engine_handles_invalid_json(kb: KnowledgeBase) -> None:
    """Non-JSON model output degrades to a clean abstention."""
    # Even in JSON mode a tiny model can misbehave; we must not crash.
    engine = LlmIntentEngine(client=FakeOllamaClient("pas du json")).fit(kb)
    result = engine.classify("bonjour")
    assert result.confident is False
    assert result.meta["error"] == "invalid_json"


def test_fasttext_custom_classifies(kb: KnowledgeBase) -> None:
    """The fastText supervised engine learns the toy intents."""
    pytest.importorskip("fasttext", reason="fasttext-wheel not installed")
    from intent_engine.fasttext_engine import FastTextSupervisedEngine

    # Trains fastText's own classifier on the KB examples (no download).
    engine = FastTextSupervisedEngine().fit(kb)
    top = engine.classify("je veux résilier mon contrat").top()
    assert top is not None
    # A near-training phrasing must land on the resiliation intent.
    assert top.intent == "resilier"
    # Diagnostics identify the fastText backend.
    assert "fastText" in engine.classify("bonjour").meta["backend"]


def test_mlp_classifier_learns_separable_data() -> None:
    """The PyTorch MLP head fits a simple separable two-class problem."""
    from intent_engine.mlp import TorchMLPClassifier

    # Two clusters around distinct, per-feature-varied centroids (so the
    # network's first LayerNorm has real within-vector variation to work
    # with, as it would on real embeddings). Trivially separable → a
    # correctly wired MLP must reach 100 % training accuracy.
    rng = np.random.RandomState(0)
    centroid_a = np.array([3.0, -1.0, 2.0, 0.5])
    centroid_b = np.array([-2.0, 1.0, -3.0, 0.2])
    class_a = centroid_a + rng.normal(scale=0.1, size=(15, 4))
    class_b = centroid_b + rng.normal(scale=0.1, size=(15, 4))
    x = np.vstack([class_a, class_b])
    y = np.array(["a"] * 15 + ["b"] * 15)
    clf = TorchMLPClassifier(epochs=100, seed=0).fit(x, y)
    proba = clf.predict_proba(x)
    # Probability matrix shape and normalisation.
    assert proba.shape == (30, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # Predictions recover the labels on this easy problem.
    predicted = clf.classes_[proba.argmax(axis=1)]
    assert (predicted == y).mean() == 1.0


@pytest.mark.slow
def test_fasttext_pretrained_if_available(kb: KnowledgeBase) -> None:
    """Integration: the pretrained French fastText engine, if downloaded.

    Marked ``slow`` and skipped when the ~4.5 GB ``cc.fr.300.bin`` model is
    not present, so the fast suite never needs the big download.
    """
    from intent_engine.fasttext_engine import FastTextPretrainedEngine

    # Skip cleanly when the model has not been downloaded on this machine.
    if not FastTextPretrainedEngine.is_model_available():
        pytest.skip("Modèle fastText FR (cc.fr.300.bin) absent.")
    engine = FastTextPretrainedEngine().fit(kb)
    top = engine.classify("je souhaite mettre fin à mon engagement").top()
    assert top is not None
    # Pretrained vectors should map this paraphrase to the resiliation intent.
    assert top.intent == "resilier"


@pytest.mark.slow
def test_bert_engine_real_backend(kb: KnowledgeBase) -> None:
    """Integration: the real embedding backend classifies correctly.

    Marked ``slow`` because it either downloads an SBERT model or calls a
    running Ollama server. Skipped by the fast CI lane (``-m "not slow"``).
    """
    # No embedder injected → build_embedder picks the configured backend.
    engine = BertIntentEngine().fit(kb)
    result = engine.classify("ma voiture est abîmée après un choc")
    top = result.top()
    assert top is not None
    # A semantic backend should map this paraphrase to the sinistre intent.
    assert top.intent == "declarer_sinistre"
