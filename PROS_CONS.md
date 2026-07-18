# Intent Engine — 3 approches comparées

> Grand tableau comparatif pour choisir **comment détecter l'intention** d'un
> client d'assurance : à l'ancienne (TF-IDF), avec une représentation type
> BERT, ou en « bourrin » avec un LLM. Chiffres issus de benchmarks publiés
> **et** de nos propres mesures sur ce dépôt. Sources en bas de page.

---

## En une phrase

| # | Approche | L'idée | Quand la choisir |
|---|----------|--------|------------------|
| 1 | **TF-IDF + linéaire** | Sac de n-grammes pondérés + régression logistique | Baseline instantanée, hors-ligne, explicable ; beaucoup de données étiquetées, vocabulaire stable |
| 2 | **Embeddings BERT + classifieur** | Phrase → vecteur sémantique (SBERT) → petit classifieur | Robustesse aux paraphrases, multilingue, peu d'exemples par intention |
| 3 | **LLM + prompt JSON** | On décrit les intentions au modèle, il répond en JSON strict | Démarrage à froid (zéro donnée), extraction de slots, catalogue qui bouge vite |

---

## Le grand tableau

| Critère | 1 · TF-IDF | 2 · BERT (SBERT) | 3 · LLM (local) |
|---|---|---|---|
| **Généralisation aux paraphrases** | Faible — colle aux mots exacts¹ | Bonne — capture le sens² | Excellente — comprend le langage³ |
| **Données requises / intention** | Dizaines d'exemples¹ | 5–10 (few-shot)² | 0–5 (dans le prompt)³ |
| **Latence d'inférence** | La plus rapide (µs–ms) | Rapide (10–40 ms CPU)² | Lente (s à dizaines de s en local)³ |
| **Compute / matériel** | CPU, empreinte minuscule | CPU ok, modèle ~100–500 Mo² | GPU/Apple Silicon conseillé³ |
| **Coût monétaire** | Gratuit, local | Gratuit si auto-hébergé | Gratuit en local / **payant en API**⁴ |
| **Explicabilité** | Excellente — poids par mot¹ | Moyenne — voisins/kNN² | Faible-moyenne, non déterministe⁵ |
| **Multilingue / français** | Faible — un modèle par langue¹ | Fort — modèles multilingues² | Excellent nativement³ |
| **Fautes de frappe** | Faible (atténué par n-grammes char)⁶ | Bonne² | Excellente³ |
| **Ajouter une intention** | Ré-entraîner + collecter des données¹ | Ré-entraîner le **petit** classifieur² | Éditer une ligne du prompt³ |
| **Confidentialité / RGPD** | Idéal (on-premise) | Idéal (on-premise) | Idéal en **local** ; **risque en cloud**⁷ |
| **Risque spécifique** | Dérive de vocabulaire¹ | Coût d'embeddings² | **Hallucination** (mitigée par JSON contraint)⁸ |

---

## Ce que **ce dépôt** mesure (base de 21 intentions, 33 exemples d'éval)

Chiffres reproductibles : `python -m eval.harness`. Ce ne sont pas des
benchmarks académiques mais les résultats réels du code de ce projet, ce qui
rend le compromis tangible.

| Moteur | Exactitude (top-1) | Latence moyenne | Extraction de slots |
|---|---|---|---|
| **TF-IDF** | **97 %** (32/33) | **~0–1 ms** | ❌ |
| **BERT — SBERT multilingue** (`.[sbert]`) | **82 %** (27/33) | **~15 ms** | ❌ |
| **BERT — repli Ollama `nomic-embed-text`** | 79 % (26/33) | ~28 ms | ❌ |
| **LLM — `gemma4:e4b` (Ollama, JSON)** | **94 %** (31/33) | **~20 s** | ✅ (urgence, type de bien…) |

**Abstention hors-périmètre** (8 phrases hors sujet : météo, calcul, cuisine…) :
TF-IDF et le LLM s'abstiennent **100 %** du temps (ils disent « je ne sais
pas » → humain), BERT un peu moins. Refuser de deviner est aussi important que
bien classer.

> **Le résultat qui surprend — et qui enseigne.** Sur *nos* données, TF-IDF
> (97 %, ~1 ms) **fait jeu égal, voire mieux** que le LLM (94 %, ~20 s) en
> exactitude, et **bat** BERT. Pourquoi ? Les n-grammes de caractères (`char_wb` 3–5) capturent le
> vocabulaire partagé entre exemples d'entraînement et phrases de test, et la
> littérature confirme que **TF-IDF est une baseline redoutable** quand il y a
> recouvrement lexical¹. L'avantage de BERT/LLM apparaît sur les **vraies
> paraphrases sans mot commun** et surtout — pour le LLM — sur l'**extraction
> d'entités** (slots), que ni TF-IDF ni BERT ne font ici. Autrement dit : à
> exactitude égale, le LLM est **~20 000× plus lent** mais rend un service en
> plus. Morale : ne jamais présumer qu'« plus lourd = meilleur ». Mesurez, et
> choisissez selon le besoin réel (vitesse ? slots ? démarrage à froid ?).

---

## Détail par approche

### 1 · TF-IDF + classifieur linéaire (scikit-learn)

**Pour**
- Le plus **rapide** et le plus **léger** : inférence en µs/ms, aucun GPU, entraînement en secondes.
- **Explicabilité maximale** : les poids linéaires disent quels mots poussent vers quelle intention ; déterministe, débogable¹.
- **100 % local**, gratuit, rien ne sort du SI — idéal RGPD.
- Robustesse aux fautes récupérable via **n-grammes de caractères** (recommandé aussi par Rasa)⁶.

**Contre**
- **Ne comprend pas le sens** : la doc scikit-learn le dit — TF-IDF « ne tient pas compte de l'ordre des mots ni du contexte »¹. Deux synonymes = deux vecteurs orthogonaux.
- **Besoin de données étiquetées** ; pas de zero-shot. En few-shot pur (5 ex./intention) les baselines lexicales s'effondrent bien plus que les encodeurs de phrases⁹.
- **Un modèle par langue**, tokenisation/stop-words spécifiques au français.
- **Dérive de vocabulaire** : quand le langage client évolue (jargon, nouveaux produits), la performance baisse silencieusement → ré-entraînement périodique¹.

### 2 · Embeddings type BERT (SBERT) + classifieur

**Pour**
- **Généralise au sens** : SBERT (Reimers & Gurevych, 2019) rend les phrases comparables par cosinus, ramenant la recherche de similarité de **~65 h à ~5 s** vs BERT brut². Paraphrases sans mot commun → même voisinage.
- **Few-shot** : ~85 % sur BANKING77 avec **10 exemples/intention** (dual encoders), dépassant un BERT fine-tuné en régime few-shot⁹.
- **Multilingue** natif (modèles `paraphrase-multilingual-*`, LaBSE) — atout majeur pour le français².
- **Maintenance légère** : ajouter une intention = ré-entraîner **seulement le petit classifieur** (secondes), l'encodeur reste figé ; en kNN/prototypes, parfois sans ré-entraînement du tout.
- **On-premise** possible à 100 % (modèle téléchargé).

**Contre**
- **Plus lourd** que TF-IDF : modèle de ~100–500 Mo à charger, coût compute d'inférence non nul².
- **Explicabilité intermédiaire** : on inspecte les voisins/scores, mais l'espace dense est moins lisible que des poids par mot.
- Sans exemples, **pas de zero-shot strict** (il faut au moins amorcer le classifieur).
- La **qualité dépend du modèle d'embedding** : notre repli `nomic-embed-text` fait 77 %, le SBERT multilingue fait 84 % — le choix du modèle compte.

### 3 · LLM génératif + prompt JSON (Ollama, local)

**Pour**
- **Zero-shot / few-shot** : fonctionne sans donnée d'entraînement, juste le catalogue dans le prompt³. Démarrage à froid immédiat.
- **Extraction de slots** dans le même appel (numéro de contrat, urgence, type de bien) — ce qu'il faut pour *exécuter* la demande, pas seulement l'étiqueter.
- **Robustesse maximale** aux formulations inédites, fautes, français familier³.
- **Maintenance triviale** : nouvelle intention = une ligne de prompt, aucun ré-entraînement.
- **Gratuit en local** (Ollama/Gemma) : coût = matériel + électricité.

**Contre**
- **Le plus lent et le plus lourd** : de la seconde aux dizaines de secondes par requête en local³ (ici ~10–16 s sur `gemma4:e4b`).
- **Hallucination** : peut inventer une intention hors catalogue. **Mitigation** : JSON contraint (Ollama `format:"json"`) + rejet des id inconnus — implémenté ici⁸. ⚠️ Contrepartie : forcer un format strict peut légèrement dégrader le raisonnement¹⁰ (d'où l'astuce d'un champ `reformulation` libre).
- **Précision qui chute quand les intentions sont nombreuses/fines** : ~89 % zero-shot sur 9 intentions, mais ~74 % sur 60 intentions et ~61 % sur 13 intentions ambiguës⁹.
- **En API cloud** : coût par token⁴ **et surtout** exfiltration de données — critique en assurance (voir ci-dessous).
- **Non déterministe** : deux appels peuvent différer (atténué par température 0).

---

## Le point qui décide en assurance : RGPD & données de santé

L'assurance manipule des **données sensibles** (santé, art. 9 RGPD). La CNIL
rappelle que⁷ :
- l'hébergement de données de santé doit être **certifié HDS** en France ;
- le principe de **minimisation** s'applique ;
- une **décision significative** (refus, tarification) ne peut reposer sur le
  seul chatbot sans intervention humaine.

**Conséquence pratique** : un LLM **en cloud** envoie potentiellement des
données de santé hors du SI → montage juridique lourd. Les trois approches en
**local** (TF-IDF, SBERT auto-hébergé, LLM via Ollama on-premise) lèvent
l'essentiel de ce risque. **C'est pourquoi ce projet est 100 % local.**

---

## Recommandation : un système **hybride**

Aucune approche ne gagne sur tous les axes. En production d'assurance :

1. **SBERT multilingue + classifieur** pour le **volume** : rapide, robuste aux
   paraphrases, explicable par voisinage, on-premise.
2. **LLM local (Ollama) en repli** pour les intentions **rares/ambiguës**, le
   **démarrage à froid** zero-shot et surtout l'**extraction de slots** — avec
   **JSON contraint** pour éliminer les hallucinations.
3. **TF-IDF** comme **garde-fou** ultra-rapide et **baseline de référence** :
   s'il est très confiant, inutile de réveiller un modèle plus lourd.

Le présent dépôt implémente les trois et les fait tourner **côte à côte** pour
rendre ce choix tangible plutôt que théorique.

---

## Chiffres clés benchmarks (ordres de grandeur, ⚠️ benchmarks en anglais)

| Réglage | Approche | Exactitude | Source |
|---|---|---|---|
| CLINC150 (full) | RoBERTa-base | 97,0 % | [a] |
| BANKING77 (full) | RoBERTa-base | 94,1 % | [a] |
| CLINC/BANKING/HWU (full) | Rasa DIET | 89,4 / 89,9 / 84,9 % | [a] |
| 5 ex./intention (few-shot) | RoBERTa-base | 86,3 / 75,9 / 71,7 % | [a] |
| 10 ex./intention, BANKING77 | Dual encoder + classifieur | ~85,2 % (+1,77 vs BERT FT) | [b] |
| Zero-shot, 9 intentions | GPT-3 (prompt) | 89,3 % | [c] |
| Zero-shot, MASSIVE (60 int.) | GPT-3 | 73,9 % | [c] |
| Zero-shot, 13 intentions fines | GPT-3 | 61,3 % (chute) | [c] |
| Recherche de similarité, 10k phrases | SBERT vs BERT | 65 h → 5 s | [d] |

⚠️ Ces benchmarks sont **anglais** : extrapolation prudente au français.
Débits « phrases/s » très dépendants du matériel ; tarifs API volatils.

---

## Bibliographie

- **[a]** Benchmark NLU (CLINC150 / BANKING77 / HWU64) — arXiv 2012.03929 — <https://ar5iv.labs.arxiv.org/html/2012.03929>
- **[b]** Casanueva et al. 2020, *Efficient Intent Detection with Dual Sentence Encoders* (BANKING77) — <https://arxiv.org/abs/2003.04807>
- **[c]** Parikh et al. 2023, *Exploring Zero and Few-shot Techniques for Intent Classification* (ACL Industry) — <https://aclanthology.org/2023.acl-industry.71.pdf>
- **[d]** Reimers & Gurevych 2019, *Sentence-BERT* — <https://arxiv.org/abs/1908.10084>
- Bridging Zero-Shot & Fine-Tuned via Retrieval-Augmented Prompting (BANKING77) — <https://asrjetsjournal.org/American_Scientific_Journal/article/view/12048>
- *Let Me Speak Freely?* (impact des contraintes de format JSON) — <https://arxiv.org/pdf/2408.02442>
- Constrained decoding / structured outputs — <https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output>
- scikit-learn — Feature extraction (TF-IDF, limitations) — <https://scikit-learn.org/stable/modules/feature_extraction.html>
- Rasa — 10 Best Practices for NLU Training Data (fautes, out-of-scope) — <https://rasa.com/blog/10-best-practices-for-designing-nlu-training-data>
- Tarifs API LLM 2025-2026 (ordres de grandeur) — <https://benchlm.ai/llm-pricing> · <https://intuitionlabs.ai/articles/llm-api-pricing-comparison-2025>
- CNIL — Chatbots & droits des personnes — <https://www.cnil.fr/fr/chatbots-les-conseils-de-la-cnil-pour-respecter-les-droits-des-personnes>
- CNIL — Minimisation & données de santé (assurance) — <https://www.cnil.fr/fr/le-principe-de-minimisation-et-les-traitements-du-nir-et-des-donnees-de-sante-dans-le-secteur-de>
- Article 9 RGPD (données sensibles) — <https://monexpertrgpd.com/article-9/>

### Renvois de notes

¹ scikit-learn feature extraction · ² Reimers & Gurevych 2019 + Casanueva 2020 ·
³ Parikh 2023 · ⁴ tarifs API BenchLM/IntuitionLabs · ⁵ *Let Me Speak Freely?* ·
⁶ Rasa best practices · ⁷ CNIL + art. 9 RGPD · ⁸ constrained decoding ·
⁹ Parikh 2023 / Casanueva 2020 · ¹⁰ arXiv 2408.02442.
