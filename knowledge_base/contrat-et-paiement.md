# resilier_contrat

> **Titre** : Résilier un contrat
> **Service** : Gestion des contrats
> **Action** : route:gestion_contrats
> **Urgence** : moyenne

## Exemples
- Je veux résilier mon contrat
- Comment mettre fin à mon assurance
- J'aimerais annuler mon assurance auto
- Résiliation de ma mutuelle
- Je souhaite arrêter mon contrat habitation
- Stopper mon assurance, je pars à l'étranger
- Je veux résilier avec la loi Hamon
- Comment ne pas être reconduit automatiquement

## Réponse
Je comprends. Grâce à la loi Hamon, après un an d'engagement vous pouvez
résilier à tout moment, sans frais. Je prépare votre demande de résiliation ;
elle prendra effet un mois après réception. Souhaitez-vous que je vérifie
d'abord si une nouvelle formule pourrait mieux vous convenir ?


# modifier_contrat

> **Titre** : Modifier un contrat
> **Service** : Gestion des contrats
> **Action** : route:gestion_contrats
> **Urgence** : faible

## Exemples
- Je veux modifier mon contrat
- J'ai changé de voiture, il faut mettre à jour l'assurance
- Ajouter un conducteur secondaire
- Je déménage, comment changer l'adresse de mon assurance
- Changer ma formule pour être mieux couvert
- Mettre à jour mes garanties
- Retirer une option de mon contrat
- Ajouter mon conjoint sur la mutuelle

## Réponse
Pas de problème. Dites-moi ce que vous souhaitez changer (véhicule, adresse,
conducteur, garanties) et votre numéro de contrat. Je transmets la
modification au service gestion des contrats ; un avenant vous sera envoyé
pour validation.


# probleme_paiement

> **Titre** : Régler un problème de paiement / prélèvement
> **Service** : Service comptabilité
> **Action** : route:comptabilite
> **Urgence** : moyenne

## Exemples
- J'ai un problème avec mon prélèvement
- On m'a prélevé deux fois
- Mon paiement a été rejeté
- Je veux changer ma date de prélèvement
- Comment mettre à jour mon RIB
- Je n'arrive pas à payer ma cotisation ce mois-ci
- Ma carte a expiré, comment mettre à jour le paiement
- Contester un prélèvement

## Réponse
Je vais regarder cela avec vous. Pour un changement de RIB ou de date de
prélèvement, j'ai besoin de votre numéro de contrat et de vos nouvelles
coordonnées bancaires. En cas de difficulté ponctuelle, le service
comptabilité peut étudier un échéancier.


# demander_attestation

> **Titre** : Obtenir une attestation ou un document
> **Service** : Gestion des contrats
> **Action** : form:demande_document
> **Urgence** : faible

## Exemples
- J'ai besoin de mon attestation d'assurance
- Pouvez-vous m'envoyer ma carte verte
- Il me faut une attestation habitation pour mon propriétaire
- Où trouver mon attestation scolaire
- Envoyez-moi mon relevé d'information
- J'ai besoin d'un justificatif d'assurance pour la préfecture
- Attestation responsabilité civile

## Réponse
Je vous prépare cela. Précisez le contrat concerné et le type d'attestation
(auto, habitation, responsabilité civile, scolaire). Le document est généré
immédiatement et vous est envoyé par e-mail, ou disponible dans votre espace
client.
