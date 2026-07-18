# Déraison Assurances — un *intent engine*, en 5 approches

[🇫🇷 Français](LISEZMOI.md) · [🇬🇧 English](README.md) — 📖 Mode d'emploi : [🇫🇷 MODEDEMPLOI](MODEDEMPLOI.md) · [🇬🇧 USERGUIDE](USERGUIDE.md)

> « Mes collègues me demandent **comment on fait** un moteur de détection
> d'intention. » — Ce dépôt répond, en montrant **cinq façons** de le faire,
> côte à côte, sur un cas concret : le chatbot d'aiguillage d'une compagnie
> d'assurance (fictive) qui oriente ses clients au **téléphone** comme
> à l'**écrit**.

![Comparateur des 5 moteurs](docs/img/02-comparateur-5-moteurs.png)

Un client dit *« j'ai eu un accident ce matin, ma voiture est cabossée »* et le
système doit comprendre l'**intention** (`declarer_sinistre_auto`), aiguiller
vers le bon **service** et, idéalement, extraire les **informations utiles**
(urgence, type de bien). Cinq moteurs font ce travail — une **traversée
volontaire de l'histoire du NLP**, du sac-de-mots au LLM génératif :

| # | Moteur | Représentation | Classifieur | Le compromis |
|---|--------|----------------|-------------|--------------|
| 1 | **TF-IDF** | n-grammes creux (car./mots) | **Random Forest** | Instantané, minuscule. Mémorise les formes de surface. |
| 2 | **fastText (appris)** | sous-mots **appris sur nos exemples** | softmax fastText | Léger ; un cran au-dessus du sac-de-mots. |
| 3 | **fastText (pré-entraîné)** | vecteurs **cc.fr.300** (Common Crawl) | régression logistique | Transfert : sait déjà que *voiture* ≈ *véhicule*. |
| 4 | **BERT** | embeddings contextuels (**SBERT**) | **MLP PyTorch** | Comprend le sens ; gagne sur les paraphrases. Local. |
| 5 | **LLM** | — (prompt) | **Gemma** via Ollama, **JSON strict** | Zéro entraînement, extrait les slots. Le plus lent, le plus malin. |

📊 Le comparatif détaillé et sourcé (benchmarks, RGPD, coûts) : **[`PROS_CONS.md`](PROS_CONS.md)**.
📖 Le mode d'emploi pas à pas (avec captures) : **[`MODEDEMPLOI.md`](MODEDEMPLOI.md)**.
🍳 Le cookbook exécutable : **[`EXAMPLES.md`](EXAMPLES.md)**.
🎯 Mon avis honnête sur le projet : **[`ASSESSMENT.md`](ASSESSMENT.md)**.
📐 Le standard de code suivi partout : **[`CODING.md`](CODING.md)**.

## Pourquoi ce projet — l'objectif pédagogique

C'est un **artefact d'enseignement Data Science / Machine Learning / IA**. Le
but n'est pas de livrer le meilleur classifieur ; c'est de faire **ressentir,
en un écran**, à des collègues qui ne pratiquent *pas* le ML, l'idée la plus
importante du NLP appliqué : **la représentation compte plus que le
classifieur.**

Lisez le tableau des moteurs de haut en bas et vous parcourez l'histoire du
domaine :

1. **Sac-de-mots (TF-IDF)** — on compte des n-grammes ; le modèle voit des
   *chaînes*, pas du *sens*. Un synonyme jamais vu lui est invisible.
2. **Sous-mots appris (fastText, sur nos données)** — le modèle commence à
   rapprocher les mots proches, à partir de quelques centaines d'exemples.
3. **Vecteurs pré-entraînés (fastText cc.fr.300)** — transfert : la
   connaissance de milliards de mots de français est versée gratuitement.
4. **Embeddings contextuels (BERT/SBERT) + réseau de neurones** — du sens qui
   dépend du contexte, plus un classifieur non-linéaire.
5. **LLM génératif (Gemma)** — aucun entraînement ; du raisonnement à partir
   d'un prompt, et — c'est unique — l'extraction de *slots* structurés.

```mermaid
flowchart LR
    A["1 · TF-IDF<br/>sac-de-mots<br/>51 %"] --> B["2 · fastText<br/>sous-mots appris<br/>66 %"] --> C["3 · fastText<br/>pré-entraîné cc.fr.300<br/>74 %"] --> D["4 · BERT + MLP<br/>contextuel<br/>86 %"] --> E["5 · LLM Gemma<br/>génératif + slots<br/>82 %"]
    style A fill:#CCE4FF,stroke:#007AFF,color:#1C1C1E
    style B fill:#79DBDC,stroke:#0E7490,color:#1C1C1E
    style C fill:#EFDCF8,stroke:#AF52DE,color:#1C1C1E
    style D fill:#D4F5D9,stroke:#28CD41,color:#1C1C1E
    style E fill:#FFEACC,stroke:#FF9500,color:#1C1C1E
```

Le comparateur montre ensuite le **gain** avec des chiffres réels mesurés (pas
des opinions) : sur un jeu de test **riche en paraphrases**, l'exactitude monte
**51 % → 66 % → 74 % → 86 %** des moteurs 1→4, et le LLM ajoute l'extraction de
slots. Et surtout, il montre les **réserves honnêtes** qui comptent pour un·e
praticien·ne : l'incertitude d'échantillonnage (**violin plots** bootstrap), la
variance train/test (**validation croisée** k-fold), la mauvaise calibration
des réseaux de neurones (trop sûrs d'eux hors-périmètre) et la confidentialité
(pourquoi tout tourne en local). But : qu'un·e collègue non-ML reparte en
comprenant *pourquoi* choisir une approche plutôt qu'une autre.

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
ollama pull gemma3:4b           # moteur LLM (compact + rapide ; ~5 s/appel à chaud)
ollama pull nomic-embed-text    # repli d'embeddings pour le moteur BERT
```

### 2. Le projet

```bash
python -m venv .venv
source .venv/bin/activate        # Windows 🪟 : .venv\Scripts\activate
pip install -r requirements.txt

# Optionnel — le chemin SBERT + MLP PyTorch du moteur BERT (~2 Go) :
pip install "sentence-transformers>=3.0.0" torch
# Optionnel — le moteur fastText pré-entraîné : télécharger cc.fr.300 (~4,5 Go) :
python scripts/download_fasttext.py
# Optionnel — la couche d'évaluation (DeepEval ; Giskard exige Python ≤ 3.11) :
pip install ".[eval]"
```

> La démo **se dégrade gracieusement** : sans `sentence-transformers`+`torch`,
> le moteur BERT est indisponible ; sans `cc.fr.300.bin`, le fastText
> pré-entraîné est masqué ; sans Ollama, le LLM est masqué. TF-IDF et fastText
> appris tournent toujours.

---

## Démarrage rapide

### L'interface web (le joli front)

```bash
./start.sh                       # ou : uvicorn intent_engine.api:app --port 8000
# puis ouvrez http://localhost:8000
```

Écrivez une demande, choisissez un moteur (ou **Comparer tout**), et voyez les
prédictions des 5 moteurs côte à côte : barres de confiance, latences, slots
extraits et action d'aiguillage. Parcourez la base de connaissance pour essayer
des exemples.

### En ligne de commande

```bash
python -m intent_engine intents                       # lister les intentions
python -m intent_engine compare "j'ai eu un accident, ma voiture est cabossée"
python -m intent_engine classify --engine tfidf "je veux résilier"
python -m intent_engine execute "il me faut une prise en charge pour l'hôpital"
```

Exemple de sortie de `compare` (sur une paraphrase — voyez les moteurs lexicaux
s'abstenir tandis que les sémantiques trouvent) :

| Moteur | Prédiction | Confiance | CPU / appel |
|--------|-----------|:---------:|------------:|
| `tfidf` | *(s'abstient)* | — | ~50 ms |
| `fasttext_custom` | `declarer_sinistre_auto` | 0.33 | ~33 µs |
| `fasttext_pretrained` | *(s'abstient)* | — | ~250 µs |
| `bert` | `declarer_sinistre_auto` | **0.98** | ~20 ms |
| `llm` | `declarer_sinistre_auto` | **0.95** | ~4,7 s |

Le LLM extrait en plus des **slots** — `type_bien: auto`, `urgence: haute` — ce
qu'aucun classifieur ne fait. Les deux moteurs lexicaux s'abstiennent ou passent
tout juste la barre sur cette paraphrase ; les sémantiques sont confiants. *Voilà*
la leçon en une requête.

---

## Résultats mesurés (21 intentions, 88 paraphrases tenues à l'écart)

Reproductibles : `python -m eval.harness` (exactitude/latence) et
`python -m eval.crossval` (distributions bootstrap + validation croisée).

Le jeu de test est volontairement **riche en paraphrases** (faible recouvrement
lexical avec l'entraînement) : il mesure la **généralisation**, pas la
mémorisation du vocabulaire — c'est là que la représentation prouve sa valeur.

| # | Moteur | Exactitude (held-out) | CPU / appel | Slots |
|---|--------|----------------------:|------------:|:-----:|
| 1 | **TF-IDF + RandomForest** | 51 % | ~50 ms | ❌ |
| 2 | **fastText (appris)** | 66 % | ~33 µs | ❌ |
| 3 | **fastText (pré-entraîné cc.fr.300)** | 74 % | ~250 µs | ❌ |
| 4 | **BERT (SBERT + MLP)** | **86 %** | ~20 ms | ❌ |
| 5 | **LLM (Gemma via Ollama)** | 82 % | ~5 s | ✅ |

> **Une surprise de latence à remarquer.** Le *classique* `TF-IDF + RandomForest`
> (~50 ms) est en fait le **moteur non-LLM le plus lent** — les centaines d'arbres
> de la forêt coûtent plus cher que la tête MLP de BERT à deux produits matriciels
> (~20 ms) ou la recherche de vecteurs de fastText (~33 µs). « À l'ancienne » ne
> veut pas dire « rapide », et « neuronal » ne veut pas dire « lent » : on mesure,
> on ne suppose pas. (Temps CPU via `process_time`, insensible aux autres apps —
> voir `eval/bench.py` ; le chiffre du LLM est le calcul propre d'Ollama.)

**Les distributions, pas juste les points** — chaque violon ci-dessous est une
**validation croisée répétée 5 blocs** : 5 blocs × 5 mélanges = **25 mesures
réelles** par moteur (apprendre sur 4/5 des K = 21 intentions / N = 500
exemples, tester sur le 1/5 restant). Chaque point est un vrai score de
généralisation — aucun artifice de rééchantillonnage — et les densités se
chevauchent à peine : les moteurs sont *réellement* différents, pas du bruit :

![Distribution d'exactitude par moteur (violin plot)](docs/img/violin-accuracy-fr.png)

> **Deux angles, une histoire honnête.** Sur les paraphrases ci-dessus,
> l'exactitude monte 51 → 66 → 74 → 86 %. Mais en **validation croisée** sur les
> exemples in-distribution de la KB, les moteurs sont *plus proches* : le lexical
> s'en sort quand le test ressemble à l'entraînement, et s'effondre sous le
> changement de distribution (paraphrases) — la raison d'être des représentations
> sémantiques.
>
> Filet hors-périmètre : sur 15 phrases hors sujet, TF-IDF s'abstient ~93 % du
> temps ; le réseau BERT est plus sûr de lui (~73 % après réglage du seuil) —
> une vraie leçon sur la **calibration des réseaux de neurones**. Analyse
> complète et sources dans [`PROS_CONS.md`](PROS_CONS.md).
>
> **Sur le choix du LLM.** Le défaut est le compact `gemma3:4b` (~5 s à chaud) :
> il atteint **82 %**, *sous* BERT — un petit LLM local troque de l'exactitude
> contre de la vitesse, et son vrai atout est l'**extraction de slots + le
> zero-shot**, pas la précision brute. Le plus gros `gemma4:e4b` monte à ~93 %
> mais à ~40 s/appel (`INTENT_LLM_MODEL` le rebranche). Plus lourd ≠ meilleur —
> on choisit selon le besoin.

### L'ingénierie de prompt, ça sert vraiment ? Une expérience 2×2

La leçon *représentation* ci-dessus parle du **modèle**. Celle-ci parle du
**prompt**. On croise deux réglages indépendants — la **qualité** du prompt (un
prompt *mauvais* : une tâche + un schéma simples mais raisonnables, pas un
épouvantail — vs un prompt *bon* qui ajoute des règles de désambiguïsation issues
de l'analyse d'erreurs) et les **exemples** (zéro-shot vs trois exemples few-shot,
sur des phrases **fraîches** hors jeu de test, donc sans tricher) — soit quatre
prompts sur un seul axe. On lance le 2×2 complet sur plusieurs modèles locaux et
on ne montre **que** celui où le prompt *bon* gagne vraiment, avec la montée la
plus nette :

![Ingénierie de prompt : de mieux en mieux](docs/img/shootout-fr.png)

> **Soigner le prompt paie surtout quand le modèle est faible.** Sur le petit
> `qwen2.5:3b`, les quatre prompts grimpent **60 → 60 → 67 → 77 %** — +17 points
> par les mots et les exemples seuls. Sur `gemma3:4b`, déjà à ~90 %, le même
> effort ne bouge presque rien (il est proche de son plafond). La morale qu'un·e
> praticien·ne reconnaît : *avant de sortir un modèle plus gros, corrige le
> prompt — mais n'attends pas de miracle quand le modèle est déjà fort.*
> (Échantillon tenu à l'écart de 30 ; prédictions mises en cache par config dans
> `eval/.llm_shootout/`.)

---

## Architecture

La base de connaissance Markdown alimente tous les moteurs ; les cinq
implémentent le même contrat `IntentEngine`, donc le routeur, l'API, la CLI et
le front les traitent à l'identique. Seuls la **représentation et le
classifieur** changent.

```mermaid
flowchart LR
    KB["📄 knowledge_base/<br/>Markdown, h1 = intention"] --> Moteurs

    subgraph Moteurs["Cinq moteurs — même contrat IntentEngine"]
        direction TB
        E1["1 · TF-IDF · Random Forest"]
        E2["2 · fastText · appris"]
        E3["3 · fastText · pré-entraîné"]
        E4["4 · BERT · SBERT + MLP"]
        E5["5 · LLM · Gemma · JSON"]
    end

    Moteurs --> R["router.py<br/>registre · comparaison · exécution"]
    R --> API["api.py · FastAPI"]
    R --> CLI["cli.py · terminal"]
    API --> WEB["web/ — vanilla JS + Tailwind"]

    style KB fill:#FFEACC,stroke:#FF9500,color:#1C1C1E
    style R fill:#F8F8F8,stroke:#808080,color:#1C1C1E
    style E1 fill:#CCE4FF,stroke:#007AFF,color:#1C1C1E
    style E2 fill:#79DBDC,stroke:#0E7490,color:#1C1C1E
    style E3 fill:#EFDCF8,stroke:#AF52DE,color:#1C1C1E
    style E4 fill:#D4F5D9,stroke:#28CD41,color:#1C1C1E
    style E5 fill:#FFEACC,stroke:#FF9500,color:#1C1C1E
    style API fill:#CCE4FF,stroke:#007AFF,color:#1C1C1E
    style CLI fill:#CCE4FF,stroke:#007AFF,color:#1C1C1E
    style WEB fill:#CCE4FF,stroke:#007AFF,color:#1C1C1E
```

Modules-clés : `kb.py` (parseur), `base.py` (contrats), `tfidf_engine.py`,
`fasttext_engine.py`, `embeddings.py` + `mlp.py`, `bert_engine.py`,
`llm_engine.py` + `ollama_client.py`, `router.py`, `api.py`, `cli.py` ;
`eval/` contient les datasets, le banc d'essai, `crossval.py`, `violin.py` et
les intégrations DeepEval/Giskard.

Les cinq moteurs partagent la même « tuyauterie » : on voit la qualité bouger
le long de la progression, seule la représentation change.

---

## Tests & évaluation

```bash
pytest -m "not slow"                   # suite rapide (déterministe, sans réseau)
pytest                                 # suite complète (BERT réel + Ollama)
python -m eval.harness                 # exactitude/latence des 3 moteurs
```

---

## Confidentialité

Le **moteur d'intention** tourne **en local** (scikit-learn, fastText & SBERT
auto-hébergés, LLM via Ollama) : le texte d'une requête ne quitte pas la
machine — un choix délibéré, car en assurance une seule phrase peut être une
**donnée de santé sensible** au sens de l'art. 9 RGPD. *« Il me faut une prise
en charge pour l'Institut de cancérologie »* révèle un diagnostic de cancer ;
l'envoyer à un LLM cloud exfiltrerait exactement la donnée que la loi protège
le plus. Ici, elle reste sur la machine.

Détails et discussion RGPD dans [`PROS_CONS.md`](PROS_CONS.md#le-point-qui-décide-en-assurance--rgpd--données-de-santé).


