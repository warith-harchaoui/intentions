# Stack technique — Moteur d'intention « Déraison Assurances »

> Carte de la stack : *quelle brique, pour quel rôle, et pourquoi.* Document de référence
> pour reprendre le projet ou l'auditer. Tout est **100 % local** (scikit-learn, fastText, SBERT,
> LLM via Ollama) — le texte d'une requête ne quitte jamais la machine (enjeu RGPD art. 9 :
> une phrase d'assurance peut être une donnée de santé).

## Philosophie

Artefact **pédagogique** (Data Science / ML / IA) : faire *sentir*, en un écran, l'idée centrale
du NLP appliqué — **la représentation compte plus que le classifieur** — en comparant **cinq
moteurs** côte à côte, du sac-de-mots au LLM génératif, sur le **même** jeu, avec des chiffres
**mesurés rigoureusement** (skore) et des **réserves honnêtes** (incertitude, calibration).

## Langage & environnement

| Élément | Choix | Notes |
|---|---|---|
| Langage | **Python 3.11** | CI sur 3.11 |
| Env | **pip + venv** (`.venv`) | choix validé : *pas* de migration pixi (projet déjà publié) |
| Lint/format | **Ruff** (PEP 8 + pydocstyle + isort) | garde bloquante |
| Tests | **pytest** (suite rapide déterministe + lents marqués) | CI GitHub Actions |

## Les cinq moteurs (progression historique du NLP)

Tous exposent le même contrat `IntentEngine` (`base.py`) ; seule change la **représentation**.

| # | Moteur | Représentation | Classifieur | Stack |
|---|---|---|---|---|
| 1 | **TF-IDF + Random Forest** | n-grammes creux | Random Forest | **scikit-learn** |
| 2 | **fastText (appris)** | sous-mots appris sur nos exemples | softmax fastText | **fasttext-wheel** |
| 3 | **fastText (pré-entraîné)** | vecteurs **cc.fr.300** (Common Crawl, ~4,5 Go) | régression logistique | fasttext-wheel + scikit-learn |
| 4 | **BERT + MLP** | embeddings de phrase **SBERT** (contextuels) | MLP **PyTorch** | **sentence-transformers** + **torch** |
| 5 | **LLM Gemma** | (prompt) | `gemma3:4b` via **Ollama**, **JSON strict** | httpx → Ollama |

> Le LLM est le seul à **extraire des slots** (champs structurés : urgence, type de bien…),
> prêts pour un CRM/IVR. Embeddings de secours : `nomic-embed-text` via Ollama.

## Données (la fondation)

- **Base de connaissance en Markdown** : un titre `# h1` = une intention (`knowledge_base/*.md`).
  Un expert métier ajoute une intention **sans toucher au code**. 21 intentions.
- **Train = 1008** (48/intention × 21) · **held-out = 210** (10/intention × 21) · **OOS = 15**.
- 0 doublon, 0 fuite train∩test (vérifié). Génération : `scripts/generate_examples.py` (gemma3:4b),
  protocole documenté dans **`PROTOCOL.md`**.

## Évaluation rigoureuse (méthodologie *probabl* : skore + skrub)

| Outil | Rôle |
|---|---|
| **skore** | `CrossValidationReport` (25-fold **RepeatedKFold**, données équilibrées) + `EstimatorReport` (held-out 210) + `ComparisonReport` sur les 4 classifieurs → `eval/skore_results.json` |
| **skrub** | graphe **DataOps** déclarant le pipeline côté sklearn (TF-IDF+RF) |
| **DeepEval** | éval orientée LLM |
| **Giskard** | scan de robustesse ML |

- Les 4 classifieurs sont exposés en **estimateurs sklearn-compatibles** (`FrozenEmbedder` pour
  cacher les embeddings, `TorchMLPClassifier(BaseEstimator, ClassifierMixin)`) pour un traitement
  homogène par skore. skore mesure l'**accuracy argmax brute** (sans abstention) = la vraie
  question ML ; l'abstention/seuil est un choix **produit**, mesuré à part (`eval/harness.py`).
- Le LLM (zéro/few-shot) n'a pas de `fit` → **baseline held-out** séparée (pas de CV),
  caches résumables `eval/.llm_shootout/`.

## Figures (Vega-Lite → PNG)

- **Violin** : distributions des 25 folds skore (4 classifieurs) + lignes Dirac (LLM zéro-entraînement).
- **8 matrices de confusion** (une par moteur), **encre adaptative** (noir/blanc selon la luminance
  de la cellule, seuil WCAG 0.5), labels sentence-case accentués, colonne **Abstention**.
- Rendu via **vl-convert** ; boucle *export PNG → analyse → correction*. **Identité couleur par
  moteur** (palette harchaoui.org) : TF-IDF #007AFF · fastText-appris #1D8C8D · **fastText-pré #28CD41
  (vert)** · **BERT #AF52DE (violet)** · qwen-zs #FFCC00 · qwen-fs #FF9500 · gemma-zs #FF8AC4 · gemma-fs #FF3B30.

## API, Front, i18n

| Couche | Stack |
|---|---|
| **API** | **FastAPI** + **uvicorn** + **pydantic** / **pydantic-settings** (`intent_engine/api.py`) ; `/api/i18n` |
| **Front** | **vanilla JS** + **Tailwind** (house style *front-ui*, trois-Roboto) ; bilingue **FR/EN** (drapeau 🇫🇷/🇬🇧) + thème clair/sombre ; a11y lint 0 |
| **i18n** | **`locales/i18n.yaml`** = source unique (copie GUI **et** prompts LLM) |
| **Langue** | **langdetect** — détecte la langue de la requête → prompt LLM FR ou EN |

## Chronométrage

`eval/bench.py` `format_duration` (s/ms/µs/ns, jamais « 0 ms ») : temps **CPU** (`process_time`,
insensible aux autres apps) + compute **Ollama** (`eval_duration`). Surprise mesurée : le
**TF-IDF + Random Forest (~50 ms)** est le plus **lent** des non-LLM (des centaines d'arbres > la
tête MLP de BERT ~20 ms). *Mesurer, ne pas supposer.*

## Récapitulatif des dépendances par rôle

```
Web/API        fastapi · uvicorn · pydantic · pydantic-settings · python-multipart
ML classique   scikit-learn · numpy
Éval rigoureuse skore · skrub          (+ deepeval · giskard en extra [eval])
fastText       fasttext-wheel          (+ cc.fr.300 ~4.5 Go téléchargé à la demande)
BERT (option)  sentence-transformers · torch
LLM/embeddings httpx → Ollama          (gemma3:4b · nomic-embed-text)
Bilingue/i18n  langdetect · PyYAML
Dev/CI         ruff · pytest
Figures        vl-convert (Vega-Lite → PNG)
```

> Dégradation **gracieuse** : sans `sentence-transformers`+`torch`, le moteur BERT est masqué ;
> sans `cc.fr.300.bin`, le fastText pré-entraîné est masqué ; sans Ollama, le LLM est masqué.
> TF-IDF et fastText-appris tournent toujours.
