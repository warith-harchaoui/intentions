# skore evaluation : trainable intent engines

Raw argmax accuracy (no abstention). CV = 25-fold RepeatedKFold on the
1008 balanced training utterances; held-out = fit on 1008, scored on the
210 disjoint paraphrases. Produced by `python -m eval.skore_eval`.

| Engine | CV accuracy | Held-out accuracy |
|---|---|---|
| TF-IDF + Random Forest | 71.9% ± 3.2% | 68.1% |
| fastText (learned) | 75.2% ± 3.2% | 71.4% |
| fastText (pretrained) | 75.6% ± 3.5% | 72.9% |
| BERT + MLP | 78.2% ± 3.4% | 77.1% |

## skore ComparisonReport : held-out metrics

```
Estimator                                        TF-IDF + Random Forest  fastText (learned)  fastText (pretrained)  BERT + MLP
Metric           Label                                                                                                        
Score                                                             0.681                 NaN                    NaN         NaN
Accuracy                                                          0.681               0.714                  0.729       0.771
Precision        assistance_depannage                             0.857               0.667                  0.667       0.778
                 cambriolage                                      0.667               0.800                  0.833       0.833
                 declarer_sinistre_auto                           0.857               0.750                  1.000       0.900
                 degat_des_eaux                                   0.714               0.769                  0.818       0.909
                 demander_attestation                             1.000               1.000                  1.000       0.900
                 escalade_humain                                  0.364               0.412                  0.316       0.500
                 faire_reclamation                                0.667               0.667                  0.833       0.750
                 horaires_et_contact                              0.714               0.818                  0.909       1.000
                 incendie_habitation                              1.000               0.875                  0.800       1.000
                 modifier_contrat                                 0.500               0.625                  0.500       0.833
                 parler_a_conseiller                              0.615               0.875                  0.667       0.583
                 probleme_paiement                                0.778               0.857                  0.833       0.667
                 remboursement_soins                              0.714               1.000                  0.667       0.545
                 resilier_contrat                                 0.875               0.769                  0.833       0.875
                 souscrire_assurance_auto                         0.400               0.500                  0.636       0.727
                 souscrire_assurance_habitation                   0.375               0.455                  0.400       0.571
                 souscrire_mutuelle_sante                         0.818               1.000                  0.900       0.750
                 souscrire_prevoyance                             0.750               0.667                  0.667       0.800
                 suivre_dossier                                   0.545               0.538                  0.700       0.636
                 tiers_payant                                     0.818               1.000                  0.818       1.000
                 vol_vehicule                                     0.667               0.667                  0.857       0.875
Recall           assistance_depannage                             0.600               0.600                  0.800       0.700
                 cambriolage                                      0.800               0.800                  1.000       1.000
                 declarer_sinistre_auto                           0.600               0.600                  0.600       0.900
                 degat_des_eaux                                   1.000               1.000                  0.900       1.000
                 demander_attestation                             0.700               0.800                  1.000       0.900
                 escalade_humain                                  0.400               0.700                  0.600       0.500
                 faire_reclamation                                0.200               0.400                  0.500       0.600
                 horaires_et_contact                              1.000               0.900                  1.000       1.000
                 incendie_habitation                              1.000               0.700                  0.800       1.000
                 modifier_contrat                                 0.300               0.500                  0.400       0.500
                 parler_a_conseiller                              0.800               0.700                  0.800       0.700
                 probleme_paiement                                0.700               0.600                  0.500       0.800
                 remboursement_soins                              0.500               0.500                  0.600       0.600
                 resilier_contrat                                 0.700               1.000                  1.000       0.700
                 souscrire_assurance_auto                         0.800               0.700                  0.700       0.800
                 souscrire_assurance_habitation                   0.300               0.500                  0.200       0.800
                 souscrire_mutuelle_sante                         0.900               0.900                  0.900       0.600
                 souscrire_prevoyance                             0.900               0.800                  0.800       0.800
                 suivre_dossier                                   0.600               0.700                  0.700       0.700
                 tiers_payant                                     0.900               0.800                  0.900       0.900
                 vol_vehicule                                     0.600               0.800                  0.600       0.700
ROC AUC          assistance_depannage                             0.986                 NaN                  0.974       0.995
                 cambriolage                                      0.989                 NaN                  0.996       0.999
                 declarer_sinistre_auto                           0.968                 NaN                  0.984       0.999
                 degat_des_eaux                                   1.000                 NaN                  0.997       1.000
                 demander_attestation                             0.991                 NaN                  1.000       1.000
                 escalade_humain                                  0.918                 NaN                  0.937       0.956
                 faire_reclamation                                0.953                 NaN                  0.948       0.976
                 horaires_et_contact                              0.996                 NaN                  1.000       1.000
                 incendie_habitation                              0.999                 NaN                  0.997       1.000
                 modifier_contrat                                 0.905                 NaN                  0.947       0.870
                 parler_a_conseiller                              0.987                 NaN                  0.969       0.983
                 probleme_paiement                                0.967                 NaN                  0.974       0.940
                 remboursement_soins                              0.978                 NaN                  0.951       0.966
                 resilier_contrat                                 0.992                 NaN                  1.000       0.993
                 souscrire_assurance_auto                         0.953                 NaN                  0.982       0.989
                 souscrire_assurance_habitation                   0.922                 NaN                  0.930       0.963
                 souscrire_mutuelle_sante                         0.989                 NaN                  0.965       0.987
                 souscrire_prevoyance                             0.985                 NaN                  0.959       0.986
                 suivre_dossier                                   0.940                 NaN                  0.984       0.978
                 tiers_payant                                     0.998                 NaN                  0.998       0.997
                 vol_vehicule                                     0.960                 NaN                  0.950       0.984
Log loss                                                          1.160               0.897                  1.171       0.889
Fit time (s)                                                      0.470               0.467                  0.049       0.716
Predict time (s)                                                  0.055               0.004                  0.000       0.001
ROC AUC                                                             NaN                 NaN                    NaN         NaN
```
