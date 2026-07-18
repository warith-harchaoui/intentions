"""Approach 5 — the "brute force" LLM engine with a strict JSON contract.

Module summary
--------------
No training, no vectors, no classifier. We hand a local generative model
(Gemma via Ollama) the catalogue of intents and a carefully engineered
prompt, and ask it to *reason* about the user's sentence and answer with a
single, strictly-shaped JSON object. Ollama's ``format="json"`` grammar
guarantees the output parses; the prompt guarantees it means something.

Why bother, if TF-IDF and BERT already work?
--------------------------------------------
Two things the classifiers cannot do out of the box:

* **Zero-shot coverage** — a brand-new intent works the moment it is added
  to the Markdown KB, with no retraining.
* **Slot filling** — the same call that picks the intent also extracts
  structured entities (policy number, urgency, channel), which is what you
  actually need to *execute* the request, not just label it.

The price is real: it is the slowest and heaviest of the three, it can
hallucinate an intent that is not in the catalogue (we defend against that
by rejecting unknown ids), and it depends on a running model server. This
engine is the demonstration of "prompt engineering, done properly" — a
pinned schema, few-shot examples, temperature 0, and defensive parsing.

Usage example
-------------
>>> from intent_engine.kb import KnowledgeBase
>>> from intent_engine.llm_engine import LlmIntentEngine
>>> kb = KnowledgeBase.from_directory("knowledge_base")   # doctest: +SKIP
>>> engine = LlmIntentEngine().fit(kb)                     # doctest: +SKIP
>>> engine.classify("j'ai eu un accident ce matin").engine  # doctest: +SKIP
'llm'

Author
------
Project maintainers.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .base import IntentEngine, IntentPrediction, IntentResult
from .config import get_settings
from .i18n import system_prompt as i18n_system_prompt
from .i18n import user_template as i18n_user_template
from .kb import KnowledgeBase
from .lang import detect_language
from .ollama_client import OllamaClient, OllamaError

# Module logger — no bare prints in library code (coding standard rule 6).
logger = logging.getLogger(__name__)

# The system prompt is the heart of this approach. It fixes the role, the
# task, the exact JSON schema, and the anti-hallucination rules. Everything
# that makes the "brute force" approach reliable lives in this string, which
# is why it is heavily commented at the point of use below.
# The default system prompt (French) and its English mirror now live in the
# single localized source of truth, ``locales/i18n.yaml``, alongside the GUI
# copy. They are fetched per query via :func:`i18n_system_prompt` /
# :func:`i18n_user_template` so a translator edits one file, and the model is
# instructed in the query's own language. The intent ids stay identical across
# languages (they are language-neutral snake_case) — only the wording changes.

# --- Prompt-engineering experiment (2×2) ---------------------------------
# Four prompts built from two INDEPENDENT switches, so the shootout can
# isolate what each buys:
#   * quality:  a **bad** prompt (task + schema, nothing else) vs a **good**
#     prompt that adds error-driven disambiguation rules (below);
#   * examples: **zero-shot** vs **few-shot** (three worked examples).
# The few-shot examples use *fresh* utterances that are NOT in the test set —
# using test queries as examples would be cheating (leakage).

# Shared task + output schema (constant across all four, so the only
# differences are the rules and the examples).
_EXP_TASK = (
    "Tu es l'aiguilleur d'intentions de « Déraison Assurances ». Pour la "
    "phrase du client, choisis UNE intention du catalogue fourni et réponds "
    "en JSON."
)
_EXP_SCHEMA = (
    'Format de sortie : {"intent": "<id du catalogue>", "confidence": '
    '<nombre entre 0 et 1>, "slots": {<clé>: <valeur>}, "reformulation": '
    '"<phrase>"}'
)

# The **good** half: explicit disambiguation of the exact intent pairs an
# error analysis showed the model confuses (souscrire vs modifier vs
# probleme_paiement; theft vs accident). This is prompt engineering *driven
# by the mistakes*, and it is what actually moves the accuracy.
_EXP_RULES = (
    "Distinctions à NE PAS confondre (les erreurs viennent de là) :\n"
    "- souscrire_* = NOUVEAU contrat pour un bien pas encore assuré "
    '("je viens d\'acheter", "il me faut une assurance pour…").\n'
    "- modifier_contrat = changer un contrat DÉJÀ existant (véhicule, "
    "conducteur, adresse, garantie). Changer de voiture sur son contrat = "
    "modifier, PAS souscrire.\n"
    "- probleme_paiement = paiement / prélèvement / RIB / banque, y compris "
    '"j\'ai changé de banque, mettez à jour mes coordonnées".\n'
    "- vol_vehicule = on a DÉROBÉ le véhicule ou des objets dedans. "
    "declarer_sinistre_auto = DÉGÂTS par accident / choc, sans vol.\n"
    "- resilier_contrat = arrêter / ne pas reconduire un contrat."
)

# The **few-shot** half: three worked examples on FRESH queries (never in the
# test set — that would be leakage), each showing the exact JSON to produce.
_EXP_FEWSHOT = (
    "Exemples :\n"
    "Phrase : \"un camion m'a accroché en reculant, l'aile est enfoncée\" → "
    '{"intent":"declarer_sinistre_auto","confidence":0.95,'
    '"slots":{"type_bien":"auto"},"reformulation":"Accident matériel auto."}\n'
    'Phrase : "je viens de prendre un scooter, il me faut une assurance" → '
    '{"intent":"souscrire_assurance_auto","confidence":0.93,'
    '"slots":{"type_bien":"deux-roues"},"reformulation":"Assurer un scooter '
    'neuf."}\n'
    'Phrase : "on a forcé mon coffre et volé mes outils dans la voiture" → '
    '{"intent":"vol_vehicule","confidence":0.9,"slots":{"urgence":"haute"},'
    '"reformulation":"Vol d\'objets dans la voiture."}'
)


def experiment_prompt(good: bool, fewshot: bool) -> str:
    """Assemble one of the four experiment prompts from two switches.

    Parameters
    ----------
    good : bool
        Include the error-driven disambiguation rules (the "good" half).
    fewshot : bool
        Include the three worked few-shot examples.

    Returns
    -------
    str
        The assembled system prompt.

    Examples
    --------
    >>> "Distinctions" in experiment_prompt(good=True, fewshot=False)
    True
    >>> "Distinctions" in experiment_prompt(good=False, fewshot=False)
    False
    """
    # Order: task → (rules if good) → (examples if few-shot) → schema.
    parts = [_EXP_TASK]
    if good:
        parts.append(_EXP_RULES)
    if fewshot:
        parts.append(_EXP_FEWSHOT)
    parts.append(_EXP_SCHEMA)
    return "\n\n".join(parts)


def _extract_json(raw: str) -> str:
    """Pull a JSON object out of a possibly fenced / chatty model answer.

    Small local models sometimes ignore JSON mode and wrap the object in a
    Markdown ```json … ``` fence, or add a sentence around it. This peels off
    the fence and, as a last resort, returns the substring from the first
    ``{`` to the last ``}`` so ``json.loads`` sees a bare object.

    Parameters
    ----------
    raw : str
        The raw model output.

    Returns
    -------
    str
        A best-effort JSON string (may still be invalid; the caller catches).

    Examples
    --------
    >>> _extract_json('{"a": 1}')
    '{"a": 1}'
    >>> _extract_json('Voici la réponse : {"a": 1} voilà')
    '{"a": 1}'
    """
    text = raw.strip()
    # Strip an opening code fence (``` or ```json) and any closing fence.
    if text.startswith("```"):
        # Drop the first line (``` or ```json) and a trailing ``` if present.
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    # If it already looks like a bare object, use it as-is.
    if text.startswith("{") and text.endswith("}"):
        return text
    # Otherwise, carve out the outermost {...} span (handles stray prose).
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    # Nothing object-like found; return the original so the caller's
    # json.loads raises and we abstain cleanly.
    return raw


# Below this self-reported confidence we abstain and hand off to a human,
# mirroring the two classifier engines so the comparison is apples-to-apples.
_CONFIDENCE_FLOOR = 0.40

# Caps for slot sanitisation: at most this many slots, each key/value no
# longer than this, so a runaway model answer cannot bloat the payload sent
# to a downstream CRM/IVR.
_MAX_SLOTS = 12
_MAX_SLOT_LEN = 120


class LlmIntentEngine(IntentEngine):
    """Zero-shot intent + slot extraction via a local LLM and JSON mode.

    Parameters
    ----------
    client : OllamaClient | None, optional
        Injected Ollama client (useful for tests/mocks). When ``None``
        (default) one is built from configuration at :meth:`fit` time.
    model : str | None, optional
        Override the Ollama model tag; defaults to the configured model
        (host-aware, MLX on Apple Silicon).

    Attributes
    ----------
    name : str
        Always ``"llm"``.
    """

    # Engine tag surfaced in results and the comparison UI.
    name: str = "llm"

    def __init__(
        self,
        client: OllamaClient | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Store the (optional) injected client, model and prompt overrides."""
        settings = get_settings()
        # Allow dependency injection for tests; otherwise build a real client.
        self._client: OllamaClient = client or OllamaClient(
            settings.ollama_base_url, timeout_s=settings.request_timeout_s
        )
        # Model tag: explicit override wins, else the host-aware default.
        self._model: str = model or settings.llm_model
        # System-prompt override: when set (e.g. the prompt-engineering eval),
        # it wins for every query. When ``None`` (the product default), the
        # prompt is picked per query to match the *detected* language.
        self._system_prompt_override: str | None = system_prompt
        # Retained for answer/routing lookup and to build the catalogue.
        self._kb: KnowledgeBase | None = None
        # The pre-rendered catalogue block injected into every prompt. Built
        # once at fit time so we do not re-serialise it on every request.
        self._catalogue_block: str = ""
        # Set of valid ids for O(1) hallucination checks at predict time.
        self._valid_ids: set[str] = set()

    def fit(self, kb: KnowledgeBase) -> LlmIntentEngine:
        """Capture the intent catalogue used to ground the prompt.

        There is no model training here — "fit" simply memorises the KB and
        pre-renders the catalogue the prompt will show the LLM.

        Parameters
        ----------
        kb : KnowledgeBase
            The parsed knowledge base.

        Returns
        -------
        LlmIntentEngine
            ``self``, ready to :meth:`classify`.
        """
        self._kb = kb
        # Remember every legal id so we can reject a hallucinated one later.
        self._valid_ids = set(kb.intent_ids())
        # Pre-render the catalogue as a compact, readable block: one line per
        # intent with a couple of examples. This is the few-shot grounding.
        self._catalogue_block = self._render_catalogue(kb)
        return self

    def _render_catalogue(self, kb: KnowledgeBase) -> str:
        """Serialise the intent catalogue into a compact prompt block.

        Parameters
        ----------
        kb : KnowledgeBase
            The knowledge base whose catalogue we render.

        Returns
        -------
        str
            One human-readable line per intent, with a couple of examples,
            small enough to keep the local model fast.
        """
        lines: list[str] = []
        # One bullet per intent: id, human title, and up to two example
        # utterances. Examples act as few-shot anchors that sharpen the
        # model's mapping from phrasing to id without bloating the prompt.
        for entry in kb.catalogue():
            examples = " | ".join(entry["examples"])
            lines.append(f"- {entry['id']} — {entry['title']} (ex: {examples})")
        # Always advertise the escape hatch id so the model has a legal way to
        # say "none of these" instead of forcing a wrong pick.
        lines.append("- hors_perimetre — Aucune intention du catalogue ne correspond")
        return "\n".join(lines)

    def classify(self, text: str, top_k: int = 3) -> IntentResult:
        """Classify one utterance via the LLM and parse its JSON answer.

        Parameters
        ----------
        text : str
            The user's sentence.
        top_k : int, optional
            Unused by this engine (the LLM commits to a single intent), kept
            for interface compatibility with the other engines.

        Returns
        -------
        IntentResult
            The chosen intent (or an abstention), extracted slots, the
            scripted answer, timing and the raw model output for debugging.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        # Guard against use-before-fit.
        if self._kb is None:
            raise RuntimeError("LlmIntentEngine.classify called before fit().")

        # Match the prompt to the query's language: detect it, then pick the
        # system prompt and user-message template. An explicit override (the
        # prompt-engineering eval) short-circuits the system-prompt choice.
        lang = detect_language(text)
        system = self._system_prompt_override or i18n_system_prompt(lang)
        template = i18n_user_template(lang)

        # Assemble the two-message chat: the system prompt, then a user message
        # carrying the catalogue and the sentence to classify.
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": template.format(catalogue=self._catalogue_block, text=text),
            },
        ]

        # Time the whole round-trip: this is the number the comparison UI uses
        # to show how much slower the LLM is than the classifiers.
        started = time.perf_counter()
        try:
            # JSON mode + temperature 0: deterministic, grammar-constrained.
            raw = self._client.chat(
                self._model, messages, json_mode=True, temperature=0.0
            )
        except OllamaError as exc:
            # Server down / model missing: abstain gracefully so the demo
            # keeps working and the UI can show "LLM engine offline".
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.warning("LLM engine unavailable: %s", exc)
            return self._abstention(text, elapsed_ms, error=str(exc))
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # Parse and validate the model's JSON into a clean result.
        return self._build_result(text, raw, elapsed_ms)

    def _build_result(self, text: str, raw: str, elapsed_ms: float) -> IntentResult:
        """Parse the raw JSON answer into a validated :class:`IntentResult`.

        Parameters
        ----------
        text : str
            The original user utterance (echoed into the result).
        raw : str
            The raw string returned by the model (expected to be JSON).
        elapsed_ms : float
            Measured inference latency in milliseconds.

        Returns
        -------
        IntentResult
            A validated result, or an abstention if parsing/validation fails.
        """
        # Defensive parse: even in JSON mode a tiny local model can wrap its
        # answer in a Markdown ```json fence or add stray prose. We strip
        # fences and, failing that, extract the outermost {...} block before
        # giving up — a JSONDecodeError becomes a clean abstention, not a crash.
        try:
            payload: dict[str, Any] = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("LLM returned non-JSON output: %r", raw[:200])
            return self._abstention(text, elapsed_ms, error="invalid_json")

        # Pull the fields with safe defaults; the model may omit some.
        intent_id = str(payload.get("intent", "")).strip()
        confidence = self._coerce_confidence(payload.get("confidence"))
        # Slots are model-generated free-form JSON; sanitise them before they
        # flow to any downstream system (a raw dict could carry nested
        # objects, huge strings, or a bogus ``urgence`` value).
        slots = self._sanitize_slots(payload.get("slots"))
        reformulation = str(payload.get("reformulation", "")).strip()

        # Anti-hallucination gate: if the model returned an id that is not in
        # the catalogue (and not our explicit escape hatch), we cannot trust
        # it — abstain instead of routing the caller somewhere invented.
        assert self._kb is not None  # narrowed by the guard in ``classify``
        known = self._kb.get(intent_id)
        if known is None:
            # ``hors_perimetre`` is a *legitimate* "no match", distinct from a
            # hallucinated id; both lead to abstention but we label them apart.
            reason = "out_of_scope" if intent_id == "hors_perimetre" else "unknown_id"
            return self._abstention(
                text,
                elapsed_ms,
                error=reason,
                slots=dict(slots),
                reformulation=reformulation,
                raw=raw,
            )

        # Valid intent. Decide trust from the model's self-reported confidence.
        confident = confidence >= _CONFIDENCE_FLOOR
        response = known.response if confident else ""
        ranked = [IntentPrediction(intent=intent_id, score=confidence)]
        return IntentResult(
            engine=self.name,
            query=text,
            ranked=ranked,
            response=response,
            slots=dict(slots),
            latency_ms=elapsed_ms,
            confident=confident,
            meta={
                "backend": f"ollama:{self._model}",
                "reformulation": reformulation,
                "raw": raw,
                "confidence_floor": _CONFIDENCE_FLOOR,
            },
        )

    def _coerce_confidence(self, value: Any) -> float:
        """Coerce the model's ``confidence`` field into a clamped float.

        Parameters
        ----------
        value : Any
            Whatever the model put in the ``confidence`` slot (float, str,
            or missing).

        Returns
        -------
        float
            A value clamped to ``[0.0, 1.0]``; ``0.0`` if uninterpretable.

        Examples
        --------
        >>> from intent_engine.llm_engine import LlmIntentEngine
        >>> e = LlmIntentEngine.__new__(LlmIntentEngine)
        >>> e._coerce_confidence("0.8")
        0.8
        >>> e._coerce_confidence(None)
        0.0
        """
        # The model sometimes returns confidence as a string ("0.9"); accept
        # both and reject anything non-numeric by defaulting to 0.0.
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        # Clamp into the unit interval so a stray 1.5 or -0.2 cannot break the
        # threshold logic or the confidence bar in the UI.
        return max(0.0, min(1.0, number))

    def _sanitize_slots(self, value: Any) -> dict[str, str]:
        """Coerce the model's free-form ``slots`` into a safe flat dict.

        The LLM can return anything under ``slots`` (nested objects, numbers,
        megabyte strings, a hundred keys). Downstream systems (CRM, IVR)
        expect a small flat map of ``str -> str``, so we defensively flatten,
        cap and normalise before letting it out.

        Parameters
        ----------
        value : Any
            Whatever the model put under ``slots`` (ideally a dict).

        Returns
        -------
        dict[str, str]
            At most :data:`_MAX_SLOTS` entries, string keys and values,
            each value truncated to :data:`_MAX_SLOT_LEN`, with ``urgence``
            normalised to the ``{faible, moyenne, haute}`` vocabulary.

        Examples
        --------
        >>> from intent_engine.llm_engine import LlmIntentEngine
        >>> e = LlmIntentEngine.__new__(LlmIntentEngine)
        >>> e._sanitize_slots({"urgence": "HAUTE", "x": {"nested": 1}})
        {'urgence': 'haute', 'x': "{'nested': 1}"}
        >>> e._sanitize_slots("not a dict")
        {}
        """
        # A non-dict ``slots`` (the model ignored the schema) yields no slots
        # rather than an error — abstaining on slots must never crash routing.
        if not isinstance(value, dict):
            return {}
        clean: dict[str, str] = {}
        # Iterate deterministically and stop at the cap so a pathological
        # answer with hundreds of keys cannot bloat the payload.
        for key, raw_value in list(value.items())[:_MAX_SLOTS]:
            # Keys and values are coerced to trimmed, length-capped strings.
            slot_key = str(key).strip()[:_MAX_SLOT_LEN]
            slot_value = str(raw_value).strip()[:_MAX_SLOT_LEN]
            # Drop empty keys/values — they carry no routing signal.
            if not slot_key or not slot_value:
                continue
            # Normalise the urgency vocabulary so downstream code can branch on
            # a closed set instead of the model's free spelling ("Haute", "URGENT").
            if slot_key.lower() == "urgence":
                slot_value = self._normalise_urgency(slot_value)
            clean[slot_key] = slot_value
        return clean

    def _normalise_urgency(self, value: str) -> str:
        """Map a free-form urgency string onto ``{faible, moyenne, haute}``.

        Parameters
        ----------
        value : str
            The raw urgency value emitted by the model.

        Returns
        -------
        str
            One of ``"faible"``, ``"moyenne"``, ``"haute"``; the lower-cased
            input is returned unchanged if it matches none (kept for
            visibility rather than silently dropped).
        """
        lowered = value.lower()
        # Cheap keyword mapping: real deployments would use a controlled
        # vocabulary, but this covers the model's common spellings/synonyms.
        if any(w in lowered for w in ("haut", "urgent", "critique", "grave")):
            return "haute"
        if any(w in lowered for w in ("moyen", "normal", "modere", "modéré")):
            return "moyenne"
        if any(w in lowered for w in ("faible", "bas", "low")):
            return "faible"
        # Unknown wording: keep it (lower-cased) so a human can still see it.
        return lowered

    def _abstention(
        self,
        text: str,
        elapsed_ms: float,
        *,
        error: str,
        slots: dict[str, Any] | None = None,
        reformulation: str = "",
        raw: str = "",
    ) -> IntentResult:
        """Build a uniform "no confident intent" result.

        Parameters
        ----------
        text : str
            The original utterance.
        elapsed_ms : float
            Measured latency in milliseconds.
        error : str
            Machine-readable reason (``"invalid_json"``, ``"unknown_id"``,
            ``"out_of_scope"``, or an Ollama error string).
        slots : dict[str, Any] | None, optional
            Any slots recovered before abstaining, by default ``None``.
        reformulation : str, optional
            Any reformulation recovered before abstaining.
        raw : str, optional
            The raw model output, kept for the debugging panel.

        Returns
        -------
        IntentResult
            An empty-ranked, non-confident result carrying the reason.
        """
        # Centralising abstention keeps every failure path returning the exact
        # same shape, so the router/UI never special-cases the LLM engine.
        return IntentResult(
            engine=self.name,
            query=text,
            ranked=[],
            response="",
            slots=dict(slots or {}),
            latency_ms=elapsed_ms,
            confident=False,
            meta={
                "backend": f"ollama:{self._model}",
                "error": error,
                "reformulation": reformulation,
                "raw": raw,
            },
        )
