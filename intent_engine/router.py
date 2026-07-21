"""Engine registry, head-to-head comparison, and request execution.

Module summary
--------------
The orchestration layer that the CLI and the API sit on top of. It owns
the parsed knowledge base, lazily builds and caches the engines
(TF-IDF, fastText, BERT, LLM), and exposes three things everyone needs:

* :meth:`IntentRouter.classify` — run one engine on one utterance.
* :meth:`IntentRouter.compare` — run every available engine and return all
  their results side by side (this powers the teaching comparison UI).
* :meth:`IntentRouter.execute` — the "natural-language request execution":
  classify, then turn the winning intent into a concrete routing action
  (send to a department, open a form) enriched with any extracted slots.

Lazy construction matters: the BERT engine loads a hundreds-of-MB model,
so we only build an engine the first time it is actually asked for, and
cache it thereafter.

Usage example
-------------
>>> from intent_engine.router import IntentRouter
>>> router = IntentRouter.from_directory("knowledge_base")   # doctest: +SKIP
>>> router.classify("je veux résilier mon contrat", "tfidf").engine  # +SKIP
'tfidf'

Author
------
Project maintainers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .base import IntentEngine, IntentResult
from .bert_engine import BertIntentEngine
from .config import get_settings
from .fasttext_engine import FastTextPretrainedEngine, FastTextSupervisedEngine
from .kb import KnowledgeBase
from .llm_engine import LlmIntentEngine
from .ollama_client import OllamaClient
from .tfidf_engine import TfidfIntentEngine

# Module logger — no bare prints (coding standard rule 6).
logger = logging.getLogger(__name__)

# The canonical engine names, in the order we want them shown in the UI —
# the pedagogical progression from sparse bag-of-words to generative LLM:
#   tfidf                → TF-IDF + Random Forest (lexical)
#   fasttext_custom      → fastText supervised, learned on our examples
#   fasttext_pretrained  → pretrained French fastText vectors + classifier
#   bert                 → SBERT embeddings + PyTorch MLP (contextual)
#   llm                  → Gemma via Ollama, JSON (generative, zero-shot)
# Mapping each to a zero-arg factory lets the router build them lazily.
_ENGINE_FACTORIES: dict[str, Callable[[], IntentEngine]] = {
    "tfidf": TfidfIntentEngine,
    "fasttext_custom": FastTextSupervisedEngine,
    "fasttext_pretrained": FastTextPretrainedEngine,
    "bert": BertIntentEngine,
    "llm": LlmIntentEngine,
}


# The explicit "Je ne sais pas" script read back when we escalate to a human.
# Phrased in the first person so it works verbatim as a spoken/written reply:
# admitting uncertainty and handing over to a person is safer than a confident
# wrong guess — the whole point of the safety net the user asked for.
_HANDOFF_MESSAGE = (
    "Je ne suis pas sûr d'avoir bien compris votre demande et je préfère "
    "ne pas vous orienter au hasard. Je vous passe un conseiller — une vraie "
    "personne — qui va prendre le relais. Un instant, je vous transfère."
)


def _human_handoff(slots: dict[str, Any]) -> Execution:
    """Build the uniform "I don't know → escalate to a human" execution.

    Centralising this keeps every uncertain path (low confidence, missing
    KB record) returning the same explicit, non-AI-fallback payload: a
    ``handoff`` flag, a real routing action/service so a downstream IVR
    knows where to send the call, the scripted admission message, and any
    slots recovered before abstaining.

    Parameters
    ----------
    slots : dict[str, Any]
        Entities extracted before we gave up (empty for non-LLM engines).

    Returns
    -------
    Execution
        The human-escalation action.
    """
    # ``handoff=True`` tells the UI/IVR to transfer to a person; the action
    # and service still point at the human-agent queue so the escalation is
    # actionable, not a dead end.
    return Execution(
        intent_id="escalade_humain",
        title="Je ne sais pas — transfert à un conseiller humain",
        action="route:conseiller_humain",
        service="Accueil téléphonique",
        slots=dict(slots),
        handoff=True,
        message=_HANDOFF_MESSAGE,
    )


@dataclass
class Execution:
    """The concrete outcome of "executing" a detected intent.

    This is the bridge between *understanding* the request and *acting* on
    it. The demo does not really open tickets, but it produces the exact
    payload a downstream system (CRM, IVR, web form) would consume.

    Parameters
    ----------
    intent_id : str
        The intent that was acted upon (empty when handed off to a human).
    title : str
        Human label of that intent.
    action : str
        Machine-readable routing action from the KB metadata (e.g.
        ``"route:sinistres_auto"`` or ``"form:souscription_auto"``).
    service : str
        Department the caller is routed to.
    slots : dict[str, Any]
        Structured entities extracted from the utterance (LLM engine).
    handoff : bool
        ``True`` when no confident intent was found and a human should take
        over.
    message : str
        The scripted answer to display or speak back to the caller.

    Examples
    --------
    >>> Execution(intent_id="x", title="X").handoff
    True
    """

    # What we acted on and how.
    intent_id: str = ""
    title: str = ""
    action: str = ""
    service: str = ""
    # Entities that let a downstream system actually process the request.
    slots: dict[str, Any] = field(default_factory=dict)
    # Whether we bailed to a human (default True until a confident hit sets it
    # False), and the message to read back.
    handoff: bool = True
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary for the API layer.

        Returns
        -------
        dict[str, Any]
            Primitive-only representation of this execution.
        """
        # Straight field mirror; every field is already a primitive/dict.
        return {
            "intent_id": self.intent_id,
            "title": self.title,
            "action": self.action,
            "service": self.service,
            "slots": self.slots,
            "handoff": self.handoff,
            "message": self.message,
        }


class IntentRouter:
    """Own the KB, cache the engines, and drive classify/compare/execute.

    Parameters
    ----------
    kb : KnowledgeBase
        The parsed knowledge base shared by every engine.

    Attributes
    ----------
    kb : KnowledgeBase
        The knowledge base (exposed for the API's ``/kb`` endpoint).
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        """Store the KB and prepare the (empty) engine cache."""
        # Shared KB — every engine is fitted against this one instance.
        self.kb: KnowledgeBase = kb
        # Lazily-populated cache of fitted engines, keyed by name. Building an
        # engine (especially BERT) is expensive, so we do it at most once.
        self._engines: dict[str, IntentEngine] = {}

    @classmethod
    def from_directory(cls, directory: str | Path) -> IntentRouter:
        """Build a router by parsing a knowledge-base directory.

        Parameters
        ----------
        directory : str | Path
            Folder of Markdown intent files.

        Returns
        -------
        IntentRouter
            A router ready to serve classify/compare/execute calls.
        """
        # Delegate parsing to the KB, then wrap it in a router.
        return cls(KnowledgeBase.from_directory(directory))

    def engine(self, name: str) -> IntentEngine:
        """Return the fitted engine for ``name``, building it on first use.

        Parameters
        ----------
        name : str
            One of ``"tfidf"``, ``"fasttext_custom"``, ``"fasttext_pretrained"``,
            ``"bert"``, ``"llm"``.

        Returns
        -------
        IntentEngine
            The cached, fitted engine.

        Raises
        ------
        KeyError
            If ``name`` is not a known engine.
        """
        # Unknown engine name is a programming error — surface it clearly.
        if name not in _ENGINE_FACTORIES:
            raise KeyError(
                f"Unknown engine {name!r}; choose from {sorted(_ENGINE_FACTORIES)}"
            )
        # Build-and-fit on first request, then reuse. ``fit`` is what trains
        # the classifiers / captures the catalogue, so it must run once.
        if name not in self._engines:
            logger.info("Building and fitting engine %r ...", name)
            engine = _ENGINE_FACTORIES[name]().fit(self.kb)
            self._engines[name] = engine
        return self._engines[name]

    def available_engines(self) -> list[str]:
        """List engine names that can currently run.

        The TF-IDF and BERT engines are always listable; the LLM engine is
        only listed when the Ollama server actually answers, so the UI can
        grey it out honestly instead of offering a doomed button.

        Returns
        -------
        list[str]
            Names of engines usable right now, in display order.
        """
        settings = get_settings()
        usable: list[str] = []
        # TF-IDF, fastText-supervised and BERT are self-contained (BERT may
        # fall back to Ollama for embeddings, handled inside the engine).
        usable.append("tfidf")
        usable.append("fasttext_custom")
        # fastText-pretrained needs the big cc.fr.300 model downloaded; only
        # advertise it when the file is present, so the UI never offers a
        # column that would fail — the same honesty as the LLM gating below.
        if FastTextPretrainedEngine.is_model_available(settings.fasttext_model_path):
            usable.append("fasttext_pretrained")
        else:
            logger.info("Modèle fastText FR absent ; moteur pretrained masqué.")
        usable.append("bert")
        # Probe Ollama once for the LLM engine so we do not advertise it when
        # the server is down. ``is_available`` is a cheap 5s liveness check.
        if OllamaClient(settings.ollama_base_url).is_available():
            usable.append("llm")
        else:
            logger.info("Ollama unreachable; LLM engine hidden from UI.")
        return usable

    def classify(self, text: str, engine: str | None = None) -> IntentResult:
        """Classify one utterance with a single engine.

        Parameters
        ----------
        text : str
            The user's sentence.
        engine : str | None, optional
            Engine name; defaults to the configured default engine.

        Returns
        -------
        IntentResult
            The chosen engine's result.
        """
        # Fall back to the configured default (TF-IDF) when unspecified.
        name = engine or get_settings().default_engine
        return self.engine(name).classify(text)

    def compare(self, text: str) -> dict[str, IntentResult]:
        """Run every available engine on the same utterance.

        This is the core of the teaching demo: identical input, one set of
        representations, verdicts and latencies to eyeball.

        Parameters
        ----------
        text : str
            The user's sentence.

        Returns
        -------
        dict[str, IntentResult]
            Mapping of engine name to its result, in display order.
        """
        results: dict[str, IntentResult] = {}
        # Iterate the currently-usable engines so a downed Ollama simply omits
        # the LLM column instead of raising. Each engine times itself.
        for name in self.available_engines():
            # Isolate failures per engine: one crashing engine must not sink
            # the whole comparison the user is trying to learn from.
            try:
                results[name] = self.engine(name).classify(text)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Engine %r failed on %r: %s", name, text, exc)
        return results

    def execute(self, text: str, engine: str | None = None) -> Execution:
        """Classify then turn the winning intent into a routing action.

        This is the "write your request in natural language and have it
        executed" path. We classify with the chosen engine, and if the
        result is confident we assemble the concrete :class:`Execution`
        (service, action, slots, scripted message). Otherwise we hand off
        to a human.

        Parameters
        ----------
        text : str
            The natural-language request.
        engine : str | None, optional
            Engine to use; defaults to the configured default.

        Returns
        -------
        Execution
            The action a downstream system would carry out.
        """
        # First, understand the request with the requested engine.
        result = self.classify(text, engine)
        top = result.top()
        # No confident top intent → escalate to a human agent. The scripted
        # message stays empty; the UI shows a "transferring you" state.
        if not result.confident or top is None:
            # This is the explicit "Je ne sais pas → human" safety net: rather
            # than guess and route the caller to the wrong department, we say
            # so plainly and escalate to a non-AI agent. We still emit a real
            # routing action/service so a downstream IVR knows where to send
            # the call, and carry any slots recovered before abstaining.
            return _human_handoff(result.slots)
        # Confident hit: look up the KB record for its routing metadata.
        intent = self.kb.get(top.intent)
        # ``intent`` should exist (the engine picked a KB label), but guard
        # anyway so a stale cache can never crash execution.
        if intent is None:  # pragma: no cover - defensive
            return _human_handoff(result.slots)
        # Assemble the concrete action. ``handoff=False`` means "the bot can
        # take it from here"; the slots ride along so the target system has the
        # entities it needs (policy number, urgency, ...).
        return Execution(
            intent_id=intent.intent_id,
            title=intent.title,
            action=intent.action,
            service=intent.service,
            slots=result.slots,
            handoff=False,
            message=result.response,
        )
