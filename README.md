# Pokémon TCG Scanner

Reconnaissance de cartes Pokémon à partir d'une photo, **entièrement sur
l'appareil**, destinée à une app iOS.

> En anglais : [`OVERVIEW.md`](OVERVIEW.md) — architecture, mesures, pièges et
> limites, en un seul document.
>
> Ce que le système vaut, où il échoue, et ce qu'il ne faut pas lui demander :
> [`docs/model-card.md`](docs/model-card.md).

**État actuel : 34 identifications correctes sur 39 photos iPhone réelles**
(87,2 %, IC de Wilson à 95 % : 73-94 %), et **une seule affirmation ferme
fausse** sur 29, aucun des 7 négatifs n'étant affirmé. La configuration est
réglée contre les faux positifs : mieux vaut demander une autre photo
qu'annoncer une carte fausse avec assurance. Le banc annonçait 31/31 jusqu'à ce
qu'un troisième lot — cartes anciennes, cadrages ratés, et sept photos sans
bonne réponse possible — le ramène à cette valeur. Ce lot a aussi produit
**trois attributions fermes et fausses**, dont une sur un flou que personne ne
peut identifier : voir `docs/audit-ml.md` §0. Les photos viennent de trois
collections photographiées par plusieurs personnes (cartes françaises, index
anglais, conditions ordinaires — contre-jour, pochette, fond chargé, cartes
inclinées, un lot entièrement en paysage, et des cartes en classeur). Sept
autres photos n'ont **aucune bonne réponse possible** : dos de carte, cartes
One Piece, carte coréenne, pochon porte-cartes, flou illisible. La chaîne se
tait correctement sur six d'entre elles. Mesuré sur Mac.

**Ce que chaque étage apporte**, mesuré par ablation sur ce même banc — c'est la
chaîne qui identifie, pas le modèle seul :

| Configuration | Top-1 |
|---|---|
| photo entière, sans détection ni orientation | 5/39 (13 %, IC95 6-27) |
| + détection, filtrage, orientation → **similarité seule** | 29/39 (74 %, IC95 59-85) |
| + lecture du numéro imprimé → **chaîne complète** | 35/39 (90 %, IC95 76-96) |

Le cadrage vaut 24 identifications, l'embedding ne travaille que sur ce qu'on
lui donne, et la lecture du numéro imprimé en rattrape 6 de plus. Reproductible
par `scripts/evaluate_real.py --ablation`.

Le rapport entre les étages tient sur le banc élargi : la similarité seule
plafonne à 74 %, et l'OCR du bandeau en récupère 16 points. Chercher un meilleur
encodeur, c'est optimiser l'étage qui pèse le moins.

**Ça tourne sur iPhone.** Le portage Swift est intégré à une app Expo,
[hugo-heer/poke-scanner](https://github.com/hugo-heer/poke-scanner), comme
module natif `expo-card-encoder`, à côté du chemin OCR + TCGdex existant. Sur
iPhone 13 Pro, l'encodeur fait **5,5 ms** sur le Neural Engine. Ce que le passage
sur appareil a appris est en [`integration-kit/AGENTS.md`](integration-kit/AGENTS.md)
§10.

**Récupérer le modèle et l'index** sans reconstruire la chaîne ML — 94 Mo,
publiés en release :

```bash
gh release download kit-v4 --repo ArmanetPierre/pokemon-tcg-scanner
tar -xzf card-encoder-kit.tar.gz -C integration-kit/
```

## Principe

Pas de classifieur à 20 000 classes. L'architecture est
**détection + embedding + recherche par similarité** : une nouvelle extension se
règle en régénérant l'index, sans réentraînement.

```
photo → détection du quadrilatère (Vision) → redressement (homographie)
     → orientation par position du texte → embedding (MobileCLIP2-S2, 512 dims)
     → recherche cosinus sur 20 512 cartes → lecture du numéro imprimé
     → carte + niveau de confiance
```

## Organisation

| Dossier | Contenu |
|---|---|
| `ml/src/` | pipeline, détection, orientation, lecture du bandeau, recherche |
| `ml/scripts/` | dataset, embeddings, évaluation, exports Core ML et index |
| `integration-kit/` | doc d'intégration iOS (`AGENTS.md`) et mesures de référence |
| `docs/` | [fiche modèle](docs/model-card.md), [audit ML](docs/audit-ml.md), rapport de tests et plan d'amélioration |
| `plan-ios.md` | plan d'origine et résultats mesurés |

## Ce qui n'est pas versionné

Tout le lourd est régénérable et volontairement absent du dépôt : les 5,5 Go
d'images de référence, la banque d'embeddings, le modèle Core ML, l'index
binaire et les photos de test.

```bash
cd ml
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt   # versions exactes des mesures
.venv/bin/pip install -e . --no-deps

.venv/bin/python scripts/build_metadata.py      # métadonnées (pokemon-tcg-data)
.venv/bin/python scripts/add_missing_cards.py   # sets que cette source ignore (TCGdex)
.venv/bin/python scripts/download_images.py     # images haute résolution (~15 min)
.venv/bin/python scripts/build_embeddings.py    # index vectoriel (~6 min sur M3)
```

Puis, pour reconstituer le kit d'intégration :

```bash
.venv/bin/python scripts/export_coreml.py       # CardEncoder.mlpackage + parité
.venv/bin/python scripts/export_index.py        # index.bin + cards.json
```

`scripts/refresh_index.py --check` signale les sets sortis depuis la dernière
construction ; `scripts/add_missing_cards.py --check` dit lesquels sont
réellement ajoutables et lesquels n'ont d'image nulle part.

**L'index est reconstructible à l'identique.** La source de métadonnées est
épinglée à une révision précise — lire `master` en ferait une cible mouvante, et
deux reconstructions à un mois d'écart donneraient deux index différents sans
que rien ne le signale. `scripts/build_metadata.py --check` compare l'épingle à
l'amont, `--ref <sha>` la déplace délibérément. Les versions de paquets sont
figées dans `requirements.lock.txt`.

**Index : 20 512 cartes, 176 sets.** Plus aucune carte des métadonnées n'est
sans image. **Plus aucune carte de l'index n'est sans image.**

Les 49 dernières — collections McDonald's 2014/2015/2017/2018 et une promo HGSS —
ont été récupérées à la main et intégrées par
[`scripts/import_local_images.py`](ml/scripts/import_local_images.py), aucune
source publique ne les servant.

## Vérifier

```bash
.venv/bin/python scripts/evaluate_real.py            # banc d'essai, PyTorch
.venv/bin/python scripts/evaluate_real.py --ablation # ce qu'apporte chaque étage
.venv/bin/python scripts/evaluate_real.py --coreml   # avec le modèle embarqué
.venv/bin/python scripts/scan.py photo.jpeg          # identifier une photo
```

Le banc s'appuie sur `ml/data/eval/truth.json`, la vérité terrain établie en
lisant nom, numéro et code de set imprimés sur chaque carte. Les photos
correspondantes sont hors dépôt.

**Protocole.** Les 46 photos sont réparties en deux splits, inscrits dans
`truth.json` avec le détail de ce dont chacun est — et n'est pas — du hold-out :
`calibration` (25 photos, dont 4 sans bonne réponse) et `test` (21 photos, dont
3 sans bonne réponse). Les seuils de confiance ont été fixés avant l'arrivée du
2ᵉ lot et n'ont pas bougé ; les filtres de détection, eux, ont été réglés en le
voyant, et le 3ᵉ lot leur rend ce hold-out perdu. Le banc rapporte les deux
splits séparément, avec leurs intervalles.

Les négatifs sont majoritairement en calibration, et c'est délibéré : jusqu'au
3ᵉ lot, **aucune** photo de calibration n'était sans réponse, donc le
comportement de refus n'était calibré sur rien. `--calibrate` en tient compte —
un seuil doit atteindre 100 % de précision sur les positifs *et* passer au-dessus
de ce que les négatifs obtiennent.

Chaque taux sort avec son intervalle de Wilson, et la confiance est rapportée
par une **courbe risque/couverture** plutôt que par un seuil : un seuil n'est
comparable ni entre deux modèles ni entre deux espaces métriques, une courbe
l'est toujours.

**Le banc synthétique**, lui, couvre tout l'index — 31 photos de cartes récentes
ne peuvent pas voir un décrochage par ère, alors que 47 % de l'index précède Sun
& Moon :

```bash
.venv/bin/python scripts/evaluate_synthetic.py            # 150 cartes par ère
```

La requête est un scan de référence dégradé ([`src/augment.py`](ml/src/augment.py) :
perspective résiduelle, sous-échantillonnage, flou, exposition, reflet holo,
bruit, JPEG), donc la vérité terrain est gratuite et exacte. C'est une mesure
**relative** — elle compare des ères, des sets et des variantes de modèle entre
elles ; elle ne prédit pas la précision sur de vraies photos, faute de
reproduire l'optique du capteur et les erreurs de la détection. Le juge de paix
reste `evaluate_real.py`.

## Trois choses à savoir avant de toucher au code

Elles ont chacune coûté du temps, et sont détaillées dans
[`integration-kit/AGENTS.md`](integration-kit/AGENTS.md) §3 :

- **Le score absolu ne discrimine rien.** Deux cartes sans rapport se
  ressemblent déjà à ~0,79. C'est l'écart entre le 1er et le 2e candidat qui
  porte l'information.
- **L'embedding ne doit jamais arbitrer l'orientation.** Une carte à l'envers
  peut scorer plus haut sur une mauvaise carte que le crop droit sur la bonne.
  L'orientation se décide sur la position du texte.
- **La géométrie du prétraitement doit être identique au pixel près** entre
  l'index et les requêtes : petit côté à 256 en bilinéaire, puis recadrage
  centré. Un écrasement direct en 256×256 fait chuter la parité à 0,65.

## Pour intégrer dans une app iOS

Tout est dans [`integration-kit/AGENTS.md`](integration-kit/AGENTS.md) :
pipeline étape par étape avec l'API native correspondante, pièges,
seuils de confiance, format des fichiers, fixtures de validation, et budget
temps pour un flux vidéo (§9 — le modèle ne pèse que 5 % du coût d'une
identification ; les deux OCR de Vision en pèsent les deux tiers). Le §8 donne le
profil d'une photo entière, où ce qui décide vraiment du temps de traitement est
le nombre de crops candidats : **287 ms → 159 ms par photo** en écartant les
quadrilatères trop petits avant tout traitement, et en lisant le bandeau en deux
passes.

Un portage existe déjà et sert de référence :
[`modules/expo-card-encoder`](https://github.com/hugo-heer/poke-scanner/tree/main/modules/expo-card-encoder)
dans l'app poke-scanner — Swift, avec son propre README.
