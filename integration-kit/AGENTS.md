# Reconnaissance de cartes Pokémon TCG — kit d'intégration iOS

Ce kit identifie une carte Pokémon à partir d'une photo, **entièrement sur
l'appareil**, sans réseau. Il contient le modèle, l'index des 20 394 cartes, et
la logique de décision validée.

> **Statut mesuré** : 18 identifications correctes sur 18 photos iPhone réelles
> (cartes françaises, index anglais, conditions ordinaires : contre-jour,
> pochette, fond chargé, cartes inclinées). Mesures faites sur Mac avec le
> modèle Core ML exporté ici.
>
> **Ça tourne maintenant sur iPhone.** Portage Swift intégré à une app Expo, mesuré
> sur iPhone 13 Pro : voir §8. L'encodeur y fait **5,5 ms** sur le Neural Engine.
> Le §10 rassemble ce que le passage sur appareil a appris — dont une lacune du
> pipeline que seul un vrai jeu de photos a fait apparaître.

> **Récupérer le modèle et l'index** : ils ne sont pas dans le dépôt (94 Mo).
> Une archive est publiée en release, `kit-v1` — c'est ce que télécharge le
> `sync-model.mjs` de l'app. Les régénérer demande les 5,5 Go d'images de
> référence.

> **Flux vidéo** : adapté. L'embedding tourne en **3,3 ms** sur le Neural
> Engine et ne représente que 3 % du coût total ; le poste dominant est l'OCR de
> Vision. Lire le §9 avant de concevoir la boucle vidéo — l'architecture y est
> plus déterminante que le choix du modèle.

---

## 1. Contenu

| Fichier | Taille | Rôle |
|---|---|---|
| `CardEncoder.mlpackage` | 69 Mo | encodeur d'image → vecteur de 512 dimensions |
| `index.bin` | 21 Mo | 20 394 vecteurs float16, L2-normalisés |
| `index.json` | 0,2 Mo | forme de `index.bin` + `card_ids` dans l'ordre des lignes |
| `cards.json` | 4,3 Mo | métadonnées d'affichage, même ordre que l'index |
| `encoder_meta.json` | — | géométrie du prétraitement à reproduire |
| `benchmarks.json` | — | mesures de référence, à comparer aux vôtres (§8) |
| `test/` | — | fixtures pour vérifier l'intégration (voir §6) |
| `reference-python/` | — | implémentation de référence, commentée |

Total à embarquer : **~95 Mo**.

Le dossier `reference-python/` n'est pas à porter tel quel : c'est la source de
vérité du comportement. En cas de doute sur un détail, il fait foi.

---

## 2. Le pipeline

Sept étapes. Chacune a un équivalent natif — rien n'est à réimplémenter à la
main.

```
photo
  1. détection des quadrilatères   VNDetectDocumentSegmentationRequest
                                   + VNDetectRectanglesRequest
  2. filtrage des candidats        géométrie + variance
  3. redressement (homographie)    CIPerspectiveCorrection
  4. orientation 0° / 180°         VNRecognizeTextRequest (.fast)
  5. embedding                     CardEncoder.mlpackage
  6. recherche                     produit matriciel (Accelerate)
  7. édition exacte                VNRecognizeTextRequest (.accurate)
→ carte + niveau de confiance
```

### 1. Détection

Lancer les **deux** détecteurs et fusionner leurs résultats. La segmentation
document cadre mieux la carte ; le détecteur de rectangles rattrape les cas
qu'elle rate.

**Les candidats issus de la segmentation document doivent passer en premier** :
au dédoublonnage (étape 2), le premier vu gagne, et l'ordre inverse a coûté
deux photos sur le banc d'essai.

### 2. Filtrage

Trois filtres, dans cet ordre :

- **Dédoublonnage par centre** : deux quadrilatères dont les centres sont
  distants de moins de 5 % du grand côté de l'image visent la même carte.
- **Variance** : rejeter un crop dont l'écart-type des niveaux de gris est
  `< 8`. Ça élimine le ciel, une dalle, une touche de clavier.
  ⚠️ Ne pas monter ce seuil. À 25 il éliminait une vraie carte à contre-jour
  (écart-type 10,5). Rater une carte coûte bien plus cher que d'encoder un
  parasite, que la sélection éliminera de toute façon.
- **Repli photo entière** : si aucun crop n'atteint 0,65 de score, ajouter la
  photo non découpée comme candidat. Nécessaire quand la carte remplit le
  cadre et que ses bords sortent de l'image.
  ⚠️ Ne **pas** mettre la photo entière en concurrence systématique : elle vole
  la sélection aux crops légitimes à marge serrée.

### 3. Redressement

Homographie des 4 coins vers un portrait 734 × 1024. Si le côté haut est plus
long que le côté gauche, la carte est couchée : décaler les coins d'un cran
avant de calculer la transformation.

### 4. Orientation

Une seule passe d'OCR rapide sur le crop, dont on ne garde que **la position**
des boîtes de texte, pas leur contenu :

- centre de masse vertical du texte (0 = haut, 1 = bas), chaque boîte pondérée
  par son nombre de caractères ;
- `≥ 0,55` → carte à l'endroit ; `≤ 0,45` → la retourner à 180° ;
- moins de 20 caractères lus, ou centre entre les deux → indécis : garder les
  deux orientations comme candidats concurrents.

### 5. Embedding

Entrée : image RVB **256 × 256**, pixels 0-255.
Sortie : 512 flottants, **déjà L2-normalisés**.

La normalisation (échelle, biais) est intégrée au graphe Core ML. Ne rien
soustraire ni diviser côté Swift.

### 6. Recherche

Similarité cosinus = produit scalaire, puisque tout est normalisé. Un produit
matrice-vecteur `(20394 × 512) · (512)` via Accelerate suffit. Aucune
bibliothèque vectorielle nécessaire.

Descendre à **10 candidats** minimum : la marge de nom (§5) a besoin de trouver
un candidat portant un nom différent, or le haut du classement est souvent
saturé de réimpressions.

### 7. Édition exacte

OCR **précis** (`.accurate`) sur les **14 % inférieurs** du crop redressé, pour
lire le motif « numéro/total » (ex. `043/084`).

Puis, parmi les 10 candidats : le total imprimé doit correspondre exactement
(c'est lui qui identifie le set), le numéro tolère **une** erreur de chiffre.
Le candidat qui correspond est promu en tête.

Appliquer une table de confusions avant extraction — l'OCR se trompe de façon
prévisible sur ce bandeau : `O Q D → 0`, `I l | ) ] → 1`, `Z → 2`, `S → 5`,
`T ? → 7`, `B → 8`. La liste complète est dans `reference-python/edition.py`.

**Garde-fou hors index** : si un numéro est lu proprement mais qu'aucune carte
de toute la base ne porte ce couple numéro/total, afficher « carte inconnue »
plutôt qu'une attribution confiante. Trois cartes testées étaient dans ce cas
(promos et énergies trop récentes pour la base publique).

---

## 3. Les pièges qui coûtent des heures

Chacun a réellement coûté du temps pendant la mise au point.

### L'orientation de l'image entre Vision et le découpage

Vision travaille sur les pixels bruts. Si la couche qui découpe applique la
rotation EXIF et pas celle qui détecte (ou l'inverse), les quadrilatères
désignent une zone qui n'existe pas — les crops sortent hors cadre et le
résultat est aléatoire.

En Swift : passer explicitement la même `CGImagePropertyOrientation` à
`VNImageRequestHandler` que celle du buffer utilisé pour le découpage. Une
photo iPhone en portrait est stockée en paysage avec un drapeau d'orientation ;
ignorer ce drapeau des deux côtés est parfaitement valide, à condition de le
faire des deux côtés.

*Symptôme* : scores autour de 0,4-0,5, résultats identiques sur des photos
différentes (les crops sont noirs).

### La géométrie du prétraitement doit être identique au pixel près

L'index a été construit avec : **redimensionner le petit côté à 256 en
bilinéaire avec antialiasing, puis recadrer au centre en 256 × 256**.

⚠️ Ce recadrage **rogne volontairement le haut et le bas** d'une carte
portrait. C'est le comportement d'origine du modèle, l'index entier est
construit ainsi, et l'app doit faire exactement pareil. Redimensionner
directement en 256 × 256 (écrasement) fait chuter la parité à **0,65**.

Le filtre compte aussi : bilinéaire, pas bicubique.

*Symptôme* : identifications plausibles mais souvent fausses, scores tièdes.

### Ne jamais laisser l'embedding arbitrer l'orientation

Tentant : encoder les deux orientations et garder le meilleur score. **Ça ne
marche pas.** Une carte à l'envers ressemble encore à une carte et peut scorer
plus haut sur une *mauvaise* carte que le crop droit sur la bonne — mesuré :
0,834 (faux) contre 0,788 (juste).

L'orientation se décide sur la position du texte (étape 4), jamais sur le
score.

### Ne jamais seuiller ni afficher le score absolu

Toutes les cartes partagent une mise en page : deux cartes sans aucun rapport
se ressemblent déjà à ~0,79. L'échelle utile va de 0,79 à 1, pas de 0 à 1.

Le pire échec rencontré affichait le **deuxième meilleur score de tout le jeu**
(0,8598). C'est l'**écart** entre le 1er et le 2e qui porte l'information.

### Calculer les marges avant réordonnancement

L'étape 7 promeut un candidat en tête. Si les marges sont calculées après, elles
deviennent négatives et absurdes. Ordre correct : calculer la confiance sur le
classement par similarité, **puis** relever le niveau si l'OCR a tranché.

---

## 4. Format des fichiers

### `index.bin`

Tableau brut, **sans en-tête** : `count × dim` valeurs `float16`, ligne par
ligne (row-major). `index.json` donne `count` (20394), `dim` (512) et
`card_ids` — le n-ième identifiant correspond à la n-ième ligne.

```swift
let meta = try JSONDecoder().decode(IndexMeta.self, from: Data(contentsOf: indexJSON))
let raw = try Data(contentsOf: indexBin)          // 20 883 456 octets
// raw.count == meta.count * meta.dim * 2
```

Les vecteurs sont déjà normalisés (norme 1,0000 ± 0,0001). Convertir en
`Float32` pour le produit matriciel, ou utiliser directement les fonctions
half-precision d'Accelerate.

Le passage float32 → float16 a été vérifié : **top-1 identique sur 200/200**
requêtes de contrôle.

### `cards.json`

Tableau aligné sur `index.json.card_ids`, un objet par carte :

```json
{
  "id": "me5-36", "name": "Litwick", "number": "36", "rarity": "Common",
  "set_id": "me5", "set_name": "Pitch Black", "set_printed_total": 84,
  "image_small": "https://images.pokemontcg.io/me5/36.png"
}
```

`image_small` est une URL distante — la seule chose du kit qui demande le
réseau, et uniquement pour afficher la vignette officielle.

---

## 5. Confiance et affichage

Deux marges, calculées sur le classement par similarité :

- **marge d'édition** = score du 1er − score du 2e ;
- **marge de nom** = score du 1er − score du premier candidat portant un *nom
  différent*.

| Condition | Niveau | Affichage suggéré |
|---|---|---|
| le numéro a été lu sur la carte (§7) | `edition` | carte et set fermes |
| marge d'édition ≥ 0,03 | `edition` | carte et set fermes |
| marge de nom ≥ 0,04 | `nom` | « Primeape — édition à confirmer » |
| sinon | `incertain` | proposer le top-5, ou inviter à stabiliser |

Ces seuils sont **calibrés sur 18 photos d'un seul joueur**. Ils tiennent sur ce
jeu (aucune sur-vente : les deux vrais échecs étaient classés « incertain »),
mais méritent d'être reconfirmés sur un échantillon plus large avant d'être
figés.

Pourquoi deux niveaux : la confusion vit presque entièrement *à l'intérieur du
même nom de carte* — plusieurs tirages de la même illustration dans des sets
différents. Une marge serrée signifie souvent « bonne carte, édition
incertaine », ce qui reste utile à l'utilisateur, et non « je ne sais pas ».

---

## 6. Vérifier l'intégration

Le dossier `test/` permet de valider par étapes plutôt que de déboguer la chaîne
entière. `test/expected.json` contient les valeurs attendues.

**Étape 1 — le prétraitement seul.** Encoder `test/reference_card.jpg` (un scan
officiel présent dans l'index) et comparer au vecteur donné dans
`expected.json`. Un cosinus `≥ 0,98` valide la géométrie ; en dessous, le
prétraitement est faux et rien d'autre ne marchera. La recherche doit renvoyer
`me5-36` avec un score ≈ 0,995.

**Étape 2 — la chaîne complète.** Trois photos réelles, choisies pour exercer
des chemins différents :

| Fichier | Ce qu'elle teste |
|---|---|
| `IMG_5033.jpeg` | cas nominal |
| `IMG_5028.jpeg` | l'embedding seul se trompe — c'est l'OCR du bandeau qui donne la bonne réponse |
| `IMG_5021.jpeg` | contre-jour, carte tenue à la main, bandeau illisible |

Les champs sous `attendu` doivent être reproduits. Ceux sous `indicatif`
(crop retenu, valeurs de marge) peuvent varier légèrement selon
l'implémentation sans que ce soit un problème.

---

## 7. Limites connues

- **Cartes qui se chevauchent** : sur une photo de 5 cartes dont 4 se
  recouvrent, une seule a été isolée (correctement identifiée). Les détecteurs
  d'Apple ont besoin de voir 4 bords fermés. Pour un scan de collection,
  demander d'espacer les cartes — ou entraîner un détecteur dédié.
- **Cartes absentes de l'index** : 3 des 21 cartes photographiées (promos
  « MEP », énergies « MEE ») ne sont dans aucune source publique **avec
  image**. Le garde-fou du §7 les signale au lieu de les attribuer à tort.
- **Cartes anciennes non testées** : toutes les photos portent sur des cartes
  récentes, alors que **47 % de l'index est antérieur à Sun & Moon**. Les mises
  en page des séries Base, Neo ou EX diffèrent nettement — bordures, cadre
  d'illustration, position du bandeau. Être dans l'index ne prouve rien sur la
  reconnaissance : la couverture est un décompte, pas une mesure.
- **Réimpressions** : 92 % des cartes partagent leur nom avec une autre (Pikachu
  apparaît 99 fois). C'est la raison d'être des deux niveaux de confiance du §5.
- **Langue** : l'index est en anglais, mais les cartes françaises sont
  reconnues (l'illustration prime largement sur le texte). Non testé sur
  japonais.

---

## 8. Performance

Toutes les mesures ci-dessous ont été faites **sur Mac M3 Pro**, avec le
`.mlpackage` de ce kit. Elles sont reprises dans `benchmarks.json` pour
comparaison.

### Sur iPhone 13 Pro

Portage Swift, photos 1080×1920 issues du flux vidéo, par appel :

| Étape | iPhone 13 Pro | Mac M3 Pro |
|---|---|---|
| détection (les deux détecteurs) | 24 ms | 26 ms |
| redressement | 6,4 ms | 2,9 ms |
| OCR d'orientation | 11 ms | 9,5 ms |
| prétraitement géométrique | 1,2 ms | 1,0 ms |
| **embedding (Neural Engine)** | **5,5 ms** | **4,1 ms** |
| recherche sur 20 394 vecteurs | 1,9 ms | 0,6 ms |
| OCR du bandeau | 65 ms | 60 ms |

Le modèle tient donc largement la promesse du §9 sur du matériel réel.

⚠️ **Compiler le portage en `-O`, même en Debug.** Le reste de l'app peut rester
non optimisé, pas ce code. Les deux boucles pixel par pixel — le filtre de
variance et la rotation à 180° — coûtent **124 ms et 114 ms par crop** en
`-Onone` contre moins d'une milliseconde optimisées. Sur huit crops, c'est 1,8 s
d'un scan de 2,3 s, entièrement imputable à la configuration de build. Tout ce
qui passe par Vision, Core ML ou Accelerate est insensible : ce sont des
frameworks précompilés.

⚠️ **Le nombre de crops est le vrai poste de coût, pas le modèle.** Une photo
12 Mpx produit jusqu'à **15 hypothèses** (8 quadrilatères dédoublonnés, dont
plusieurs à orientation indécise qui comptent double). Le « 96 ms de bout en
bout » ci-dessous suppose une image 720p où les détecteurs trouvent bien moins de
candidats. Instrumenter par appel et non en cumul, sans quoi les chiffres ne
veulent rien dire.

### Sur Mac M3 Pro

### Réglage obligatoire : forcer le Neural Engine

```swift
let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine
let encoder = try CardEncoder(configuration: config)
```

Laisser Core ML choisir seul (`.all`) coûte presque le double :

| Unité de calcul | Latence | Débit |
|---|---|---|
| `.cpuAndNeuralEngine` | **3,3 ms** | 306 img/s |
| `.all` | 5,7 ms | 175 img/s |
| `.cpuAndGPU` | 8,7 ms | 115 img/s |
| `.cpuOnly` | 22,7 ms | 44 img/s |

### Coût de chaque étape

| Étape | Coût |
|---|---|
| détection document — 720p | 12,7 ms |
| détection document — 1080p | 24,1 ms |
| détection document — 12 Mpx | 113,5 ms |
| détection rectangles — 720p | 17,9 ms |
| redressement (homographie) | 0,5 ms |
| prétraitement géométrique | 1,8 ms |
| **embedding (Neural Engine)** | **3,3 ms** |
| recherche sur 20 394 vecteurs | 0,5 ms |
| OCR d'orientation | 16,1 ms |
| OCR du bandeau | 59,6 ms |

### De bout en bout, depuis une image 720p

| | Coût |
|---|---|
| identification complète, OCR du bandeau compris | **96 ms** |
| identification sans OCR du bandeau | **36,5 ms** |

La répartition est le fait marquant : **le modèle pèse 3 % du total, les deux
OCR en pèsent 79 %.** C'est ce qui dicte l'architecture du §9.

---

## 9. Flux vidéo en direct

Le modèle convient largement : **3,3 ms par image sur le Neural Engine**, soit
306 images/s. Il représente 3 % du coût d'une identification. Le poste dominant
est l'OCR de Vision (79 %).

> ⚠️ **Ne pas passer à un modèle plus léger.** MobileCLIP2-S0 ferait gagner
> environ 2 ms sur 96 et coûterait de la précision. Optimiser l'embedding, c'est
> optimiser ce qui ne limite pas.

### Répartition du travail

La règle : **ce qui coûte cher ne doit tourner qu'une fois par carte, jamais par
image.**

| Fréquence | Étapes | Coût |
|---|---|---|
| chaque image | détection du quadrilatère + suivi | 12,7 ms |
| à l'apparition d'une carte | redressement, orientation, embedding, recherche | ~36 ms |
| une fois, en tâche de fond | OCR du bandeau (confirme l'édition) | ~60 ms |
| carte déjà identifiée et suivie | rien | 0 ms |

À 30 images/s (33 ms par image), seule la détection tourne en continu : elle
laisse 20 ms pour le rendu et le suivi. L'identification d'une nouvelle carte
part sur une file d'arrière-plan — l'aperçu ne doit jamais l'attendre.

### Ce qui fait la différence

- **Détecter à 720p, pas à la résolution photo.** 12,7 ms contre 113 ms en
  12 Mpx, soit un facteur 9 pour un résultat équivalent : une carte occupe assez
  de pixels à 720p. Le crop pour l'embedding, lui, peut être extrait du buffer
  pleine résolution si disponible.
- **Suivre les cartes entre les images** (IoU sur les boîtes, ou
  `VNTrackObjectRequest`) pour ne ré-identifier que les nouvelles. En régime
  stable, le coût retombe à la seule détection.
- **N'utiliser que la segmentation document par image** (12,7 ms) et réserver le
  détecteur de rectangles (17,9 ms) à l'identification.
- **Stabiliser par vote** : plutôt qu'afficher le résultat de la première image,
  accumuler 3 à 5 identifications d'une même carte suivie et retenir la
  majoritaire. Gratuit, puisque la carte reste dans le champ.

### Limite à connaître

`VNDetectDocumentSegmentationRequest` ne renvoie **qu'un seul objet**. Pour
plusieurs cartes simultanées, il faut passer par
`VNDetectRectanglesRequest` — plus coûteux et moins fiable sur fond chargé (voir
§7). Un flux « une carte à la fois » est le cas nominal ; le scan d'un étalage
complet reste le cas difficile.

---

## 10. Ce que le passage sur iPhone a appris

Trois choses que seul un vrai portage a fait apparaître. Elles ne sont pas dans
`reference-python/`, qui reste par ailleurs la référence de comportement.

### La promotion par le bandeau ne suffit pas — il faut la recherche par numéro

L'étape 7 réordonne les candidats du top-k. Elle est donc impuissante dans un cas
précis : **le numéro est lu proprement, la carte existe, et l'embedding ne l'a
jamais fait remonter**. Mesuré sur une photo d'Inkay (`me5-51`), dont le bandeau
a donné `051/084` trois fois de suite, pendant que le classement était mené par
un Dresseur sans rapport situé **0,089 au-dessus**.

Le correctif : si aucun candidat ne porte le numéro lu et qu'**une seule carte de
tout l'index** le porte, aller la chercher directement. Un couple unique parmi
20 394 identifie la carte sans discussion, et c'est une preuve plus forte que
n'importe quel score de similarité. Se restreindre au cas unique n'est pas de la
prudence sur la lecture : un numéro partagé par plusieurs tirages ne dit pas
lequel c'est.

C'est le pendant positif du garde-fou « hors index » du §2 — la même table,
utilisée pour trouver plutôt que pour disqualifier.

⚠️ **Ce rattrapage ne couvre que 38 % des cartes.** 63 % des couples
numéro/total sont uniques, mais ils ne concernent que 7 773 cartes sur 20 394 —
les autres partagent leur couple avec jusqu'à 9 cartes, et un couple partagé ne
désigne rien. Pour ces 62 %, une carte que la recherche visuelle manque reste
manquée. C'est la limite à garder en tête avant de considérer le problème réglé :
le correctif est utile, il n'est pas un filet.

### Le filtre de rééchantillonnage, côté Core Graphics

PIL n'a pas d'équivalent exact. Les quatre filtres, mesurés sur
`test/reference_card.jpg` — qui est dans l'index, donc sa propre similarité est
la note à battre (Python obtient 0,9951) :

| filtre | similarité |
|---|---|
| `.medium` | **0,9938** |
| `.high` | 0,9926 |
| `.low` | 0,9854 |
| `.none` | 0,9656 |

`.medium` gagne, et c'est aussi celui qui est bilinéaire plutôt que bicubique —
la distinction que le §3 demande de respecter.

### Le simulateur iOS ne vaut rien pour juger la précision

Core ML s'y comporte à l'identique (self-test à 0,9944 contre 0,9938 sur Mac),
**mais pas Vision** : le segmenteur de document cadre autrement et le
recognizer lit moins. Sur les trois fixtures du §6, une seule reproduit
exactement, une perd son numéro de bandeau et une donne une mauvaise carte. Le
simulateur sert à vérifier que le module charge et que la chaîne tourne ; la
précision se juge sur appareil.

---

## 11. Mettre à jour l'index

Une nouvelle extension ne demande **aucun réentraînement** : il suffit de
régénérer `index.bin` et `cards.json` et de les remplacer. Le modèle ne change
pas. Cette régénération se fait côté projet ML, pas dans l'app.

Prévoir que l'index puisse être téléchargé plutôt qu'embarqué, pour livrer une
nouvelle extension sans passer par l'App Store.
