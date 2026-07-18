<!--
Ce fichier commence par un underscore : le parseur (intent_engine/kb.py)
l'ignore. C'est de la documentation pour les rédacteurs métier, pas une
intention. Toute la connaissance vit dans les autres fichiers *.md.
-->

# Format de la base de connaissance

> **Ceci n'est pas une intention** — fichier de documentation, ignoré par le moteur.

La règle d'or : **chaque titre de niveau 1 (`# ...`) est UNE intention.**
Un·e expert·e métier ajoute une intention en écrivant du Markdown, sans
jamais toucher au code Python.

## Anatomie d'une intention

```markdown
# declarer_sinistre_auto

> **Titre** : Déclarer un sinistre automobile
> **Service** : Gestion des sinistres auto
> **Action** : route:sinistres_auto
> **Urgence** : haute

## Exemples
- J'ai eu un accident de voiture
- Je veux déclarer un accrochage
- Quelqu'un a embouti ma portière sur le parking

## Réponse
Je suis navré pour cet incident. Je vous mets en relation avec le service
sinistres auto. Munissez-vous de votre numéro de contrat et du constat.
```

## Les champs

| Élément | Rôle |
|---|---|
| `# identifiant` | **Id machine** de l'intention (snake_case, sans accent). C'est la classe apprise par les moteurs TF-IDF et BERT, et l'id que le LLM doit renvoyer. |
| `> **Titre**` | Libellé lisible affiché dans l'interface. |
| `> **Service**` | Service vers lequel aiguiller l'appel. |
| `> **Action**` | Action machine exécutée : `route:<service>` (transfert) ou `form:<formulaire>` (ouverture d'un formulaire). |
| `> **Urgence**` | Métadonnée libre (`faible` / `moyenne` / `haute`). |
| `## Exemples` | Phrases d'entraînement (une par puce). **Le carburant des moteurs TF-IDF et BERT**, et exemples few-shot du prompt LLM. Plus il y en a, mieux c'est. |
| `## Réponse` | Le script lu/affiché au client quand l'intention est détectée avec confiance. |

## Bonnes pratiques pour les exemples

- **Variez les formulations** : question, affirmation, langage parlé, écrit.
- **Incluez des fautes et du langage naturel** (« j'ai calé ma bagnole »).
- **Couvrez les paraphrases** : c'est là que TF-IDF échoue et que BERT/LLM brillent.
- Visez **au moins 6-8 exemples** par intention pour un entraînement décent.
- Un identifiant spécial existe côté LLM : `hors_perimetre` (aucune
  intention ne correspond). Inutile de le déclarer, il est géré nativement.
