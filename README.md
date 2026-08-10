# Pokémon TCG Scanner

Reconnaissance de cartes Pokémon à partir d'une photo, **entièrement sur
l'appareil**, destinée à une app iOS.

> En anglais : [`OVERVIEW.md`](OVERVIEW.md) — architecture, mesures, pièges et
> limites, en un seul document.

**État actuel : 18 identifications correctes sur 18 photos iPhone réelles**
(cartes françaises, index anglais, conditions ordinaires — contre-jour,
pochette, fond chargé, cartes inclinées). Mesuré avec le modèle Core ML exporté,
sur Mac.

**Ça tourne sur iPhone.** Le portage Swift est intégré à une app Expo,
[hugo-heer/poke-scanner](https://github.com/hugo-heer/poke-scanner), comme
module natif `expo-card-encoder`, à côté du chemin OCR + TCGdex existant. Sur
iPhone 13 Pro, l'encodeur fait **5,5 ms** sur le Neural Engine. Ce que le passage
sur appareil a appris est en [`integration-kit/AGENTS.md`](integration-kit/AGENTS.md)
§10.

**Récupérer le modèle et l'index** sans reconstruire la chaîne ML — 94 Mo,
publiés en release :

```bash
gh release download kit-v3 --repo ArmanetPierre/pokemon-tcg-scanner
tar -xzf card-encoder-kit.tar.gz -C integration-kit/
```

## Principe

Pas de classifieur à 20 000 classes. L'architecture est
**détection + embedding + recherche par similarité** : une nouvelle extension se
règle en régénérant l'index, sans réentraînement.

```
photo → détection du quadrilatère (Vision) → redressement (homographie)
     → orientation par position du texte → embedding (MobileCLIP2-S2, 512 dims)
     → recherche cosinus sur 20 504 cartes → lecture du numéro imprimé
     → carte + niveau de confiance
```

## Organisation

| Dossier | Contenu |
|---|---|
| `ml/src/` | pipeline, détection, orientation, lecture du bandeau, recherche |
| `ml/scripts/` | dataset, embeddings, évaluation, exports Core ML et index |
| `integration-kit/` | doc d'intégration iOS (`AGENTS.md`) et mesures de référence |
| `docs/` | rapport de tests et plan d'amélioration (pages HTML) |
| `plan-ios.md` | plan d'origine et résultats mesurés |

## Ce qui n'est pas versionné

Tout le lourd est régénérable et volontairement absent du dépôt : les 5,5 Go
d'images de référence, la banque d'embeddings, le modèle Core ML, l'index
binaire et les photos de test.

```bash
cd ml
python3.11 -m venv .venv && .venv/bin/pip install -e .

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

**Index : 20 504 cartes, 175 sets.** Plus aucune carte des métadonnées n'est
sans image. Il ne reste dehors que les **8 énergies du set `mee`**, absent de
`pokemon-tcg-data` et sans visuel nulle part.

Les 49 dernières — collections McDonald's 2014/2015/2017/2018 et une promo HGSS —
ont été récupérées à la main et intégrées par
[`scripts/import_local_images.py`](ml/scripts/import_local_images.py), aucune
source publique ne les servant.

## Vérifier

```bash
.venv/bin/python scripts/evaluate_real.py            # banc d'essai, PyTorch
.venv/bin/python scripts/evaluate_real.py --coreml   # avec le modèle embarqué
.venv/bin/python scripts/scan.py photo.jpeg          # identifier une photo
```

Le banc s'appuie sur `ml/data/eval/truth.json`, la vérité terrain établie en
lisant nom, numéro et code de set imprimés sur chaque carte. Les photos
correspondantes sont hors dépôt.

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
temps pour un flux vidéo (§9 — le modèle ne pèse que 3 % du coût total ; les
deux OCR de Vision en pèsent 79 %).

Un portage existe déjà et sert de référence :
[`modules/expo-card-encoder`](https://github.com/hugo-heer/poke-scanner/tree/main/modules/expo-card-encoder)
dans l'app poke-scanner — Swift, avec son propre README.
