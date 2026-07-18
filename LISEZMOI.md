# Déraison Assurances — un *intent engine*, en 3 approches

[🇫🇷 Français](LISEZMOI.md) · [🇬🇧 English](README.md) — 📖 Mode d'emploi : [🇫🇷 MODEDEMPLOI](MODEDEMPLOI.md) · [🇬🇧 USERGUIDE](USERGUIDE.md)

> « Mes collègues me demandent **comment on fait** un moteur de détection
> d'intention. » — Ce dépôt répond, en montrant **trois façons** de le faire,
> côte à côte, sur un cas concret : le chatbot d'aiguillage d'une compagnie
> d'assurance (fictive) qui oriente ses clients au **téléphone** (voix) comme
> à l'**écrit**.

![Comparateur des 3 moteurs](docs/img/02-comparateur-3-moteurs.png)

Un client dit *« j'ai eu un accident ce matin, ma voiture est cabossée »* et le
système doit comprendre l'**intention** (`declarer_sinistre_auto`), aiguiller
vers le bon **service** et, idéalement, extraire les **informations utiles**
(urgence, type de bien). Trois moteurs font ce travail, du plus « à l'ancienne »
au plus « bourrin » :

| # | Moteur | Techno | Le compromis |
|---|--------|--------|--------------|
| 1 | **TF-IDF** | scikit-learn (n-grammes + régression logistique) | Instantané, minuscule, hors-ligne. Colle aux mots. |
| 2 | **BERT** | Embeddings de phrases (SBERT) + classifieur ML | Comprend le sens, généralise aux paraphrases. |
| 3 | **LLM** | Gemma en local via Ollama, prompt + **JSON strict** | Zéro entraînement, extrait les slots. Le plus lent. |

📊 Le comparatif détaillé et sourcé (benchmarks, RGPD, coûts) : **[`PROS_CONS.md`](PROS_CONS.md)**.
📖 Le mode d'emploi pas à pas (avec captures) : **[`MODEDEMPLOI.md`](MODEDEMPLOI.md)**.
🍳 Le cookbook exécutable : **[`EXAMPLES.md`](EXAMPLES.md)**.
📐 Le standard de code suivi partout : **[`CODING.md`](CODING.md)**.

---

## Le principe : la connaissance vit dans du Markdown

**Un titre `# h1` = une intention.** Un·e expert·e métier ajoute une intention
en écrivant du Markdown dans `knowledge_base/`, **sans toucher au code** :

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

Les `## Exemples` servent de **données d'entraînement** à TF-IDF et BERT, et
d'exemples **few-shot** au LLM. La `## Réponse` est le script lu/affiché. Format
complet : [`knowledge_base/_FORMAT.md`](knowledge_base/_FORMAT.md).

---

## Installation

Pré-requis : **Python ≥ 3.10**. Pour le moteur LLM (et le repli d'embeddings
BERT), **Ollama** en local.

### 1. Ollama (pour le moteur LLM)

- macOS 🍎 : `brew install ollama` (installez `brew` via [brew.sh](https://brew.sh/)), puis `ollama serve`
- Ubuntu 🐧 : `curl -fsSL https://ollama.com/install.sh | sh`
- Windows 🪟 : `winget install Ollama.Ollama`

Puis récupérez les modèles :

```bash
ollama pull gemma4:e4b          # moteur LLM (sur Apple Silicon : gemma4:e4b-mlx)
ollama pull nomic-embed-text    # repli d'embeddings pour le moteur BERT
```

### 2. Le projet

```bash
python -m venv .venv
source .venv/bin/activate        # Windows 🪟 : .venv\Scripts\activate
pip install -r requirements.txt

# Optionnel — le "vrai" chemin SBERT du moteur BERT (tire PyTorch, ~2 Go) :
pip install "sentence-transformers>=3.0.0"
# Optionnel — la couche d'évaluation (DeepEval + Giskard) :
pip install ".[eval]"
```

> Sans `sentence-transformers`, le moteur BERT **bascule automatiquement** sur
> les embeddings Ollama (`nomic-embed-text`) : le démo tourne quand même.

---

## Démarrage rapide

### L'interface web (le joli front)

```bash
./start.sh                       # ou : uvicorn intent_engine.api:app --port 8000
# puis ouvrez http://localhost:8000
```

Écrivez **ou dictez** une demande (reconnaissance vocale du navigateur —
*vocal-helper*), comparez les 3 moteurs avec barres de confiance et latences,
**lisez la réponse à voix haute** (synthèse vocale — *speech-helper*) et
parcourez la base de connaissance.

### En ligne de commande

```bash
python -m intent_engine intents                       # lister les intentions
python -m intent_engine compare "j'ai eu un accident, ma voiture est cabossée"
python -m intent_engine classify --engine tfidf "je veux résilier"
python -m intent_engine execute "il me faut une prise en charge pour l'hôpital"
```

Exemple de sortie de `compare` :

```text
tfidf   | declarer_sinistre_auto  [0.86]  — 1 ms
bert    | declarer_sinistre_auto  [0.59]  — 39 ms
llm     | declarer_sinistre_auto  [1.00]  — 16565 ms
        slots: {'type_bien': 'auto', 'urgence': 'haute'}
```

---

## Résultats mesurés (base de 21 intentions, 33 exemples d'éval)

Reproductibles avec `python -m eval.harness`.

| Moteur | Exactitude | Latence moyenne | Slots |
|---|---|---|---|
| **TF-IDF** | 97 % (32/33) | ~1 ms | ❌ |
| **BERT — SBERT** | 82 % (27/33) | ~15 ms | ❌ |
| **BERT — repli Ollama** | 79 % (26/33) | ~28 ms | ❌ |
| **LLM — gemma4:e4b** | 94 % (31/33) | ~20 s | ✅ |

> Plus un contrôle d'**abstention hors-périmètre** : sur 8 phrases hors sujet
> (météo, calcul, cuisine…), TF-IDF et le LLM s'abstiennent 100 % du temps — ils
> disent « je ne sais pas » au lieu de mal aiguiller.

> À exactitude égale, le LLM est **~20 000× plus lent** que TF-IDF mais extrait
> des slots. *Plus lourd ≠ meilleur : on choisit selon le besoin.* Détails et
> sources dans [`PROS_CONS.md`](PROS_CONS.md).

---

## Architecture

```
intent_engine/
  kb.py            # parseur Markdown : # h1 = intention
  base.py          # contrats communs : IntentEngine, IntentResult
  tfidf_engine.py  # Approche 1 — scikit-learn
  embeddings.py    # backends d'embeddings enfichables (SBERT / Ollama)
  bert_engine.py   # Approche 2 — embeddings + classifieur
  llm_engine.py    # Approche 3 — Ollama + JSON strict + anti-hallucination
  ollama_client.py # client Ollama synchrone (chat JSON + embeddings)
  router.py        # registre des moteurs + comparaison + exécution
  api.py           # API FastAPI
  cli.py           # interface terminal
knowledge_base/    # la connaissance (Markdown, h1 = intention)
web/               # front vanilla JS + Tailwind (+ polices self-hostées)
eval/              # dataset étiqueté + seuils + banc d'essai + DeepEval + Giskard
tests/             # pytest
```

Les trois moteurs implémentent **le même contrat** (`IntentEngine`), donc le
routeur, l'API et le front les traitent de façon identique. C'est tout l'intérêt
pédagogique : seule la **représentation** change.

---

## Tests & évaluation

```bash
pytest -m "not slow"                   # suite rapide (déterministe, sans réseau)
pytest                                 # suite complète (BERT réel + Ollama)
python -m eval.harness                 # exactitude/latence des 3 moteurs
```

---

## Confidentialité

Le **moteur d'intention** tourne **en local** (scikit-learn, SBERT auto-hébergé,
LLM via Ollama) : le texte d'une requête ne quitte pas la machine — un choix
délibéré, car en assurance les requêtes peuvent contenir des **données de
santé** (art. 9 RGPD).

> ⚠️ **Réserve honnête sur la voix.** Les fonctions vocales de l'interface
> utilisent l'API Web Speech du navigateur. Sous Chrome, **la reconnaissance
> vocale envoie l'audio aux serveurs de Google** — le chemin *voix* n'est donc
> *pas* local, contrairement au NLU. Pour une voix vraiment locale, brancher un
> *vocal-helper* (whisper.cpp) côté serveur pour la transcription et une TTS OSS
> pour la réponse. C'est signalé dans [`ASSESSMENT.md`](ASSESSMENT.md) et dans
> l'interface.

Détails et discussion RGPD dans [`PROS_CONS.md`](PROS_CONS.md#le-point-qui-décide-en-assurance--rgpd--données-de-santé).

---

## Remerciements

Remerciements chaleureux aux contributrices, contributeurs, relectrices,
relecteurs et utilisateurs qui ont aidé à améliorer ce projet.

Des outils d'IA ont pu être utilisés pendant le développement, mais la
paternité et la responsabilité restent aux mainteneurs humains.
