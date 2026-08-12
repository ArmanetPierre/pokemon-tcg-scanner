# Audit ML — rapport critique et plan d'expertise

*Revue externe du 11 août 2026. Toutes les mesures citées ont été refaites sur
le dépôt, avec l'index et les photos présents localement ; les commandes sont en
annexe.*

---

## Verdict en une page

Le projet est **au-dessus du niveau d'un projet de portfolio**, et il l'est pour
une raison précise : chaque constante du code est justifiée par une mesure, et
les limites sont écrites avant les résultats. `MIN_QUAD_AREA`, `FALLBACK_SCORE`,
`SHAPE_BUCKET`, la zone morte d'orientation — aucune n'est un nombre magique.
C'est rare, et c'est le marqueur le plus fiable d'un travail sérieux.

Ce qui l'empêche aujourd'hui d'être un projet **d'expert ML**, c'est que le
travail d'ingénierie a été fait, et pas le travail scientifique :

| | |
|---|---|
| **Ce qui est de niveau expert** | architecture retrieval justifiée, parité PyTorch↔Core ML testée, profilage par étage, refus explicite (`incertain` / `hors_index`), portage device mesuré, documentation des pièges |
| **Ce qui ne l'est pas encore** | protocole d'évaluation (n=31, sans IC, sans séparation calibration/test), aucune ablation chiffrée, confiance non calibrée, géométrie de l'espace d'embedding jamais étudiée, aucune adaptation de domaine, zéro test automatisé |

Trois chiffres résument le diagnostic, tous mesurés dans cet audit :

1. **La similarité seule fait 26/31, pas 31/31.** L'OCR du bandeau porte 5
   identifications sur 31, soit **16 % du résultat annoncé**.
2. **90,7 % top-1 sur 1 650 requêtes synthétiques** couvrant tout l'index, avec
   **79,3 % sur l'ère WotC contre 98,7 % sur Diamond & Pearl** — un écart de
   19 points que le banc actuel ne peut pas voir.
3. **La marge intrinsèque médiane entre une carte et sa plus proche voisine dans
   l'index est de 0,0099**, sous le seuil de décision de 0,03. Pour la moitié de
   l'index, la similarité ne peut structurellement pas produire un verdict ferme,
   même sur un scan parfait.

---

## 1. Le protocole d'évaluation est le maillon faible

### 1.1 n=31 ne mesure pas ce que le README laisse entendre

« 31 identifications correctes sur 31 » est vrai et vérifié — je l'ai reproduit.
Mais l'intervalle de confiance de Wilson à 95 % pour 31/31 est **[89,0 % ; 100 %]**.
La phrase honnête est « la précision réelle est probablement supérieure à 89 % »,
ce qui est un énoncé très différent de « 100 % ».

S'y ajoutent trois biais que la doc mentionne mais dont elle ne tire pas les
conséquences :

- **toutes les cartes photographiées sont récentes**, alors que 47 % de l'index
  précède Sun & Moon ;
- **un seul modèle de téléphone**, donc un seul pipeline ISP, une seule optique ;
- **deux collections**, donc deux styles de prise de vue.

### 1.2 Les seuils de confiance n'ont jamais vu de données de test

`FIRM_ID_MARGIN = 0.03` et `FIRM_NAME_MARGIN = 0.04` ont été calibrés sur 18
photos, puis « tenus » sur 14 de plus. Le commentaire du code le dit sans détour
(« à reconfirmer sur 40+ photos »), ce qui est à porter au crédit du projet —
mais il n'existe aucune séparation calibration / test, et les 14 photos de
confirmation ont été ajoutées *après* avoir vu qu'elles passaient. C'est du
réglage sur l'ensemble complet, et l'estimation de précision qui en découle est
optimiste par construction.

### 1.3 Ce que donne une évaluation à grande échelle

J'ai construit un banc synthétique : partir du scan de référence, lui appliquer
les dégradations d'une photo de téléphone (perspective résiduelle, sous-échantillonnage,
flou, exposition, balance des blancs, reflet holographique, bruit, JPEG), puis
chercher. La vérité terrain est gratuite et exacte, donc l'échantillon peut
couvrir tout l'index.

**1 650 requêtes, 150 par ère :**

| Ère | n | top-1 | top-5 | marge médiane |
|---|---:|---:|---:|---:|
| WotC 1990s (Base, Jungle, Fossil) | 150 | **79,3 %** | 99,3 % | **0,0118** |
| WotC 2000s | 150 | 90,0 % | 98,7 % | 0,0500 |
| EX | 150 | 88,0 % | 96,7 % | 0,0546 |
| Diamond & Pearl / Platinum | 150 | **98,7 %** | 99,3 % | 0,0724 |
| HGSS | 150 | 92,0 % | 98,0 % | 0,0554 |
| Black & White | 150 | 94,0 % | 98,7 % | 0,0491 |
| XY | 150 | 90,7 % | 97,3 % | 0,0537 |
| Sun & Moon | 150 | 90,0 % | 98,7 % | 0,0445 |
| Sword & Shield | 150 | 91,3 % | 98,0 % | 0,0598 |
| Scarlet & Violet | 150 | 92,0 % | 98,0 % | 0,0639 |
| promos / autres | 150 | 91,3 % | 98,7 % | 0,0595 |
| **TOTAL** | **1 650** | **90,7 %** | **98,3 %** | |

Deux enseignements que 31 photos ne peuvent pas donner :

- **L'ère WotC décroche de 19 points.** Illustrations de basse résolution
  d'origine, mise en page uniforme, et surtout des réimpressions massives entre
  Base, Base Set 2, Jungle et Legendary Collection.
- **Sa marge médiane, 0,0118, est sous le seuil `FIRM_ID_MARGIN` de 0,03.** Les
  seuils calibrés sur des cartes récentes produiront donc massivement des
  « incertain » sur les cartes anciennes — pas des erreurs, mais un refus de
  répondre, sur 8,7 % de l'index.

Ce banc **ne remplace pas** les vraies photos : les augmentations ne reproduisent
ni l'optique du capteur, ni la vraie géométrie de détection. C'est une mesure
*relative*, qui compare des ères, des sets et des variantes de modèle entre elles
sur des milliers de cartes — là où 31 photos ne mesurent que du bruit.

---

## 2. Aucune ablation : on ne sait pas ce que chaque étage apporte

C'est la lacune la plus coûteuse, parce qu'elle empêche de savoir où investir.

J'ai isolé l'étage de similarité seul (mêmes crops, même logique de sélection,
sans la promotion par le numéro lu) :

| Configuration | top-1 | top-5 |
|---|---:|---:|
| similarité seule | **26/31** (83,9 %) | 29/31 |
| chaîne complète (avec lecture du bandeau) | **31/31** | 31/31 |

**L'OCR du bandeau porte 5 identifications sur 31.** Le README attribue 31/31 à
la chaîne mais la présente sous le titre « reconnaissance de cartes », ce qui
laisse croire que l'embedding fait le travail. Il en fait 84 %. Le sixième
restant repose sur l'étage le plus fragile de la chaîne — un OCR sur une bande
de 14 % de la hauteur d'un crop, sujet aux confusions de caractères, et dont
`OVERVIEW.md` reconnaît par ailleurs qu'il ne peut résoudre directement que 38 %
de l'index.

C'est une architecture parfaitement défendable — combiner un signal visuel et un
signal symbolique est la bonne idée — mais elle doit être **présentée et mesurée
comme telle**, avec le tableau d'ablation en évidence.

---

## 3. La confiance est un seuil, pas une calibration

Le projet a déjà l'essentiel : il refuse de répondre, à deux niveaux, et il sait
dire « hors index ». C'est plus que ce que font la plupart des systèmes en
production. Mais la décision reste une comparaison à une constante.

Ce qu'un travail d'expert produit à la place, ce sont des **courbes
risque/couverture**. Mesuré sur le banc :

| espace | top-1 | précision @100 % couverture | @75 % | @50 % | @25 % | AURC |
|---|---:|---:|---:|---:|---:|---:|
| brut | 26/31 | 83,9 % | 87,0 % | 100 % | 100 % | **0,047** |

Et surtout : **le seuil qui garantit zéro erreur sur le banc est 0,0141, et il
retient 65 % des photos** — pas 0,03. Le seuil actuel est donc *conservateur*,
ce qui est le bon sens du signe, mais il a été choisi sur 18 points et son
incertitude est large dans les deux directions.

Ce qui manque :

- reporter **AURC** et **couverture à précision cible** plutôt qu'un seuil ;
- une **probabilité calibrée** (régression logistique puis calibration isotonique
  sur `[marge_id, marge_nom, verdict OCR, écart-type du crop, aire du quad]`),
  validée en leave-one-out, plutôt qu'une cascade de `if` ;
- des seuils **conditionnés à l'ère**, puisque la section 1.3 montre que la
  distribution des marges change d'un facteur 5 entre WotC et DP.

---

## 4. La géométrie de l'espace d'embedding n'a jamais été étudiée

Le projet documente comme un piège le fait que « deux cartes sans rapport se
ressemblent déjà à ~0,79 ». C'est exact, mais présenté comme une fatalité alors
que c'est un **phénomène nommé et mesurable** : l'anisotropie des espaces
CLIP. Les embeddings n'occupent pas la sphère, ils occupent un cône étroit.

Mesuré sur l'index :

```
‖vecteur moyen‖              = 0,8146      (0 = isotrope, 1 = tous colinéaires)
cos(carte, direction moyenne) : médiane 0,846, p5 0,602, p95 0,901
similarité entre deux cartes quelconques : médiane 0,697, p99 0,841
```

Un espace dont la moyenne a une norme de 0,81 est un espace où **83 % du signal
de similarité est une constante partagée par toutes les cartes**. D'où le « 0,79
plancher ». Ce n'est pas une propriété des cartes Pokémon, c'est une propriété du
modèle.

### Ce qui est intéressant : les corrections classiques échouent ici

J'ai testé les deux corrections standard de la littérature en recherche
d'instance :

| espace | top-1 (banc réel) | marge médiane | AURC | seuil sans erreur → couverture |
|---|---:|---:|---:|---|
| brut | **26/31** | 0,0241 | **0,047** | 0,0141 → **65 %** |
| centré (`e − µ`, renormalisé) | 23/31 | 0,0865 | 0,097 | 0,1318 → 32 % |
| CSLS (correction de hubness) | 22/31 | — | 0,190 | 0,2117 → 23 % |

Le centrage multiplie la marge intrinsèque médiane par 3,55 et écrase la hubness
(le hub maximal passe de 284 à 77 apparitions dans les top-10), mais **il dégrade
la précision**. Il amplifie les directions de bruit en même temps que les
directions discriminantes, et les marges gonflées ne sont plus comparables au
seuil calibré.

**C'est un résultat négatif, et c'est le plus utile de cet audit** : il ferme la
porte à la correction non supervisée, et il désigne la seule voie qui reste —
une correction **apprise**. Voir chantier C.

*(Piège méthodologique à retenir : une marge n'est jamais comparable entre deux
espaces métriques. Comparer 0,03 dans l'espace brut à 0,03 dans l'espace centré
n'a aucun sens. Seules les courbes risque/couverture se comparent.)*

---

## 5. La structure de l'index n'est pas diagnostiquée

Aucun script ne regarde l'index comme un objet en soi. Mesuré :

```
plus proche voisin de chaque carte :  p50 0,894   p90 0,962   p99 0,999
cartes ayant un voisin ≥ 0,99 :    884  (4,3 %)
cartes ayant un voisin ≥ 0,98 :  1 334  (6,5 %)
cartes ayant un voisin ≥ 0,95 :  2 766  (13,5 %)
marge intrinsèque 1er/2e voisin : médiane 0,0099 — 77,6 % des cartes sous 0,03
le plus proche voisin porte le même nom : 57,5 % des cas
```

Trois conséquences directes :

1. **4,3 % de l'index est visuellement indiscernable.** Pour ces cartes, aucune
   photo, aussi bonne soit-elle, ne produira une marge exploitable. La lecture du
   numéro n'est pas un raffinement, c'est la seule voie.
2. **Le seuil de 0,03 est au-dessus de la marge intrinsèque de 77,6 % de
   l'index.** Le verdict `edition` par similarité est donc structurellement rare,
   et le système repose bien plus sur l'OCR que la doc ne le laisse voir — ce que
   confirme l'ablation du §2.
3. **Quelques cartes sont des attracteurs.** 26 cartes apparaissent plus de 100
   fois dans le top-10 d'autres cartes, contre une médiane de 6 :

   | apparitions | carte | set |
   |---:|---|---|
   | 284 | `sm3-10` Ledian | Burning Shadows |
   | 239 | `swsh6-52` Thundurus | Chilling Reign |
   | 168 | `swsh8-140` Gligar | Fusion Strike |
   | 158 | `swsh4-105` Sableye | Vivid Voltage |

Les sets les plus auto-confusants confirment le décrochage WotC du §1.3 :
Legendary Collection (90 % des cartes ont un voisin ≥ 0,97), Base Set 2 (84,6 %),
Jungle (76,6 %), Base (62,7 %) — ce sont les réimpressions croisées.

Cette information devrait **sortir du diagnostic et entrer dans le produit** :
un champ par carte disant « discriminable par l'image seule : oui / non ».

---

## 6. Le modèle n'a jamais été touché

MobileCLIP2-S2 est utilisé *zero-shot*. Le `REGISTRY` prévoit six modèles
candidats, mais aucune comparaison chiffrée n'est versionnée, et surtout :

- **le décalage de domaine n'est ni mesuré ni réduit.** L'index est fait de scans
  éditeur propres, à plat, en éclairage studio ; les requêtes sont des photos de
  téléphone. C'est exactement le problème que 20 ans de littérature en recherche
  d'instance traitent par apprentissage métrique, et rien n'est tenté ;
- **un seul vecteur par carte.** L'enrôlement multi-vues (encoder chaque
  référence sous k augmentations) est la technique la moins chère du domaine :
  aucun réentraînement, index × k, gain immédiat sur la robustesse ;
- **aucune tête de projection.** Une matrice 512×512 apprise en contrastif sur
  des paires (référence augmentée → référence) coûte quelques minutes de GPU,
  s'exporte en Core ML pour un coût d'inférence nul, et c'est précisément la
  correction supervisée que le §4 désigne.

C'est la frontière entre « assemblage intelligent de briques Apple » — ce qu'est
le projet aujourd'hui, et il l'assemble très bien — et « travail de ML ».

---

## 7. Ingénierie : ce qui manque au dossier

- **Zéro test automatisé** sur ~5 600 lignes de Python. Les invariants les plus
  critiques sont documentés comme des pièges (§3 d'`AGENTS.md`) mais rien
  n'empêche leur régression : géométrie du prétraitement, alignement
  embeddings ↔ `card_ids`, non-application de l'EXIF, distance chiffre à chiffre,
  ordre marges-avant-réordonnancement.
- **Pas de CI.** Le banc et la parité Core ML se lancent à la main. Un runner
  macOS ferait tourner Vision.
- **Pas de traçabilité des artefacts.** `index.json` porte le nom du modèle mais
  aucune empreinte : rien n'empêche l'app de charger un `index.bin` construit
  avec un autre encodeur, et le symptôme serait des réponses *plausibles et
  fausses* — exactement la classe de bug que le projet documente ailleurs avec
  soin. Il manque un hash du `.mlpackage` dans `index.json`, et un refus de
  démarrage en cas de discordance.
- **Reproductibilité partielle.** `pyproject.toml` n'a que des bornes basses, pas
  de lock ; la révision de `pokemon-tcg-data` n'est pas épinglée. L'index n'est
  donc pas reproductible bit à bit, alors que c'est l'artefact central.
- **Pas de boucle de retour.** Aucun mécanisme de collecte des échecs depuis
  l'app. C'est pourtant le seul moyen de sortir de n=31.
- **Pas de model card.** Un projet d'expert publie : données d'entraînement de
  l'encodeur, données de l'index, population testée, populations *non* testées,
  métriques avec IC, usages déconseillés. Les éléments existent tous, dispersés
  dans `OVERVIEW.md` ; ils ne sont pas rassemblés sous une forme citable.

---

## 8. Plan d'amélioration, par rapport gain / effort

### A — Banc synthétique stratifié, en garde anti-régression
**Effort : faible (script déjà écrit et exécuté).**
Intégrer le générateur d'augmentations et l'évaluation par ère comme
`scripts/evaluate_synthetic.py`. 1 650 requêtes en ~4 min. Sortie JSON, pas du
texte imprimé.
*Critère de sortie :* toute modification de l'index, du prétraitement ou du
modèle produit un tableau par ère comparable au précédent ; une régression de
plus de 2 points sur une ère bloque.

### B — Protocole d'évaluation défendable — ✅ **fait**
**Effort : faible. Impact : c'est ce qui rend tout le reste crédible.**

Livré :

1. **Split inscrit dans `truth.json`** — `calibration` (21 photos, 1ʳᵉ
   collection) et `test` (11 photos, 2ᵉ collection). Le bloc `_protocol` dit
   précisément ce dont le split test est du hold-out (les seuils de confiance,
   figés en `a240b84`, avant l'arrivée de ces photos en `99ba7d2`) et ce dont il
   n'en est **pas** (`MIN_QUAD_AREA`, `MIN_MULTI_QUAD_AREA`, `SHAPE_BUCKET`,
   introduits dans le commit même qui ajoute ces photos — toute précision de
   détection mesurée dessus est optimiste).
2. **`ml/src/stats.py`** — intervalle de Wilson, courbe risque/couverture, AURC,
   couverture à précision cible. Vérifié : `wilson(31, 31) = (0.8897, 1.0)`.
3. **`evaluate_real.py` refondu** — rapport par split avec intervalles, `--ablation`
   (trois bras), `--json` pour une sortie exploitable en CI, et la courbe
   risque/couverture en lieu et place du seuil.
4. **`identify(..., use_band=False)`** dans `src/pipeline.py` — l'ablation se
   mesure sur le vrai code, pas sur une copie divergente.
5. **README et OVERVIEW** portent le tableau d'ablation et l'IC.

Résultat mesuré par le nouveau banc (`--ablation`) :

| Configuration | Top-1 | Δ |
|---|---:|---:|
| photo entière, sans détection ni orientation | 4/31 (13 %, IC95 5-29) | |
| + détection, filtrage, orientation (similarité seule) | 26/31 (84 %, IC95 67-93) | **+22** |
| + lecture du numéro imprimé (chaîne complète) | 31/31 (100 %, IC95 89-100) | **+5** |

Le split test se révèle plus dur que le split de calibration sur la similarité
seule : **7/10 contre 19/21**. L'écart n'est pas significatif à cette taille
(les intervalles se recouvrent largement), mais il va dans le sens attendu d'un
réglage sur la calibration, et il justifie à lui seul d'avoir séparé les deux.

*Ce qui reste ouvert :* le banc n'a plus de hold-out pour les paramètres de
détection. Il faut un 3ᵉ lot jamais vu — 40+ photos, d'autres appareils, et des
cartes d'avant Sun & Moon, dont le banc ne contient à ce jour aucun exemplaire.

### C — Tête de projection apprise (le chantier qui change la nature du projet)
**Effort : moyen. Impact : le plus élevé.**
Entraîner une projection linéaire 512→512 (ou 512→256, qui diviserait aussi
l'index par deux) en contrastif — InfoNCE, négatifs durs pris parmi les plus
proches voisins mesurés au §5 — sur des paires (référence augmentée → référence),
avec les augmentations du chantier A. Quelques minutes de GPU, export Core ML à
coût d'inférence nul, aucun changement d'architecture.
*Pourquoi c'est le bon chantier :* le §4 a montré que la correction non
supervisée dégrade ; la correction supervisée est la seule qui reste, et elle
attaque simultanément l'anisotropie, le décalage de domaine et le décrochage
WotC.
*Critère de sortie :* mesuré sur les 1 650 requêtes synthétiques **et** sur les
14 photos de test jamais vues. Objectif : +5 points sur l'ère WotC sans
régression ailleurs, et similarité seule de 26/31 à 29/31.

### D — Enrôlement multi-vues
**Effort : faible. À faire avant C, pour savoir ce que C doit battre.**
Encoder chaque référence sous k=4 augmentations, comparer trois variantes :
centroïde renormalisé (index inchangé, 21 Mo), k vecteurs avec max-pooling au
scoring (index ×4, 84 Mo), et k=1 actuel.
*Critère de sortie :* décision chiffrée sur le banc synthétique, ventilée par
ère et par coût mémoire.

### E — Exposer le plafond structurel dans le produit
**Effort : faible. Impact produit direct.**
Calculer pour chaque carte sa marge intrinsèque (§5) et l'écrire dans
`cards.json` : `image_discriminable: true | false`. L'app sait alors *avant* de
répondre qu'une carte donnée exige la lecture du numéro, et peut demander un
cadrage du bandeau plutôt que d'échouer en silence.
*Critère de sortie :* les 884 cartes à voisin ≥ 0,99 sont étiquetées, et le
verdict `edition` par similarité seule leur est interdit.

### F — Tests et CI
**Effort : moyen.**
`pytest` sur les invariants : parité géométrique du prétraitement contre un
tenseur de référence figé, alignement embeddings ↔ `card_ids`, `_digit_distance`,
`match_edition` sur cas limites, `classify_confidence` sur ordres réordonnés,
`known_pair`. Plus un banc synthétique réduit (150 requêtes) en CI sur runner
macOS.
*Critère de sortie :* le seuil de parité Core ML de 0,98 est vérifié par la CI,
pas par un humain qui pense à lancer le script.

### G — Traçabilité et reproductibilité
**Effort : faible.**
Hash SHA-256 du `.mlpackage` dans `index.json` et `encoder_meta.json`, vérifié
au chargement côté app ; verrou de dépendances ; révision épinglée de la source
de métadonnées ; graine fixée partout.

### H — Boucle de retour depuis l'app
**Effort : moyen, dépend de l'app.**
Remontée opt-in des photos ayant donné `incertain` ou corrigées par
l'utilisateur. C'est ce qui fait passer l'évaluation de 31 photos à quelques
milliers, et c'est la seule proposition dont le gain croît avec le temps.

### I — Model card
**Effort : très faible.**
Rassembler l'existant sous forme citable : données, population testée,
populations non testées, métriques avec IC, usages déconseillés, résultats
négatifs (§4).

---

## Annexe — reproduire les mesures

Les trois scripts d'audit sont versionnés dans `ml/scripts/audit/`. Ils sont en
lecture seule sur les données et n'écrivent rien.

```bash
cd ml
.venv/bin/python scripts/audit/audit_index.py       # §4 §5 — structure de l'index
.venv/bin/python scripts/audit/exp_riskcov.py       # §3 §4 — risque/couverture, 3 espaces
.venv/bin/python scripts/audit/exp_synthetic.py 150 # §1.3 — banc synthétique par ère
```

Chiffres de référence de la chaîne existante, revérifiés :

```
scripts/evaluate_real.py  ->  top-1 31/31, top-5 31/31, refus corrects 1/1
similarité seule (même sélection, sans OCR du bandeau)  ->  top-1 26/31
```
