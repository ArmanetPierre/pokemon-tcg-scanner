# Plan détaillé — Pokémon TCG Scanner sur iOS

> Complément de `pokemon-tcg-live-scanner.md`, adapté à l'objectif : **une app iOS testable sur iPhone**, avec reconnaissance embarquée (pas de serveur nécessaire à l'usage).

---

## 0. Décision d'architecture clé

La spec d'origine vise un pipeline Python temps réel sur GPU. Pour iOS, deux options :

| | Option A — Tout sur l'iPhone (on-device) | Option B — iPhone caméra + serveur d'inférence |
|---|---|---|
| Latence | Excellente (Neural Engine) | Dépend du réseau |
| Fonctionne hors ligne | ✅ | ❌ |
| Complexité de déploiement | Une seule app | App + serveur à héberger |
| Itération ML | Conversion Core ML à chaque changement | Immédiate |

**Recommandation : Option A (on-device)**, avec une phase de prototypage en Python sur Mac pour choisir et valider les modèles *avant* de les convertir en Core ML. C'est réaliste car :

- **La détection de cartes ne nécessite probablement pas de YOLO** : une carte est un rectangle d'aspect ratio connu (63×88 mm ≈ 0,716). Le framework **Vision** d'Apple (`VNDetectRectanglesRequest`) détecte les rectangles et donne les **4 coins** — ce qui règle aussi la correction de perspective (`CIPerspectiveCorrection`). Zéro entraînement, zéro dataset annoté pour le MVP.
- **La recherche vectorielle ne nécessite pas FAISS sur iPhone** : ~20 000 embeddings × 512 dims en float16 ≈ **20 Mo**. Une similarité cosinus brute-force via Accelerate/MPS prend quelques millisecondes. L'index est un simple fichier binaire embarqué dans l'app.
- **L'embedding tourne sur le Neural Engine** : MobileCLIP (Apple) ou DINOv2 ViT-S converti via `coremltools` tournent à plusieurs dizaines de FPS sur un iPhone récent.

Le principe de la spec reste intact : **détection + embedding + recherche par similarité**, pas de classifieur à 20 000 classes. Une nouvelle extension = régénérer le fichier d'index, pas de réentraînement.

---

## 1. Architecture du repo (monorepo ML + iOS)

```text
pokemon-card-scanner/
│
├── README.md
├── plan-ios.md
│
├── ml/                          # Prototypage Python sur Mac
│   ├── pyproject.toml
│   ├── data/
│   │   ├── raw/                 # images + JSON bruts
│   │   ├── processed/           # dataset nettoyé
│   │   └── embeddings/          # index + metadata
│   ├── scripts/
│   │   ├── download_dataset.py
│   │   ├── build_metadata.py    # normalisation IDs, dédoublonnage
│   │   ├── build_embeddings.py
│   │   ├── evaluate.py          # top-1/top-5 sur vues augmentées
│   │   ├── export_coreml.py     # modèle → .mlpackage
│   │   └── export_index.py      # embeddings → fichier binaire + metadata JSON
│   └── src/
│       ├── encoder.py           # abstraction sur CLIP/DINOv2/MobileCLIP
│       ├── augment.py           # rotation, perspective, éclairage, flou, sleeve
│       ├── search.py            # FAISS (Python uniquement)
│       └── identify.py          # image → top-5 candidats
│
├── ios/
│   └── PokeScanner/             # projet Xcode (SwiftUI)
│       ├── PokeScanner.xcodeproj
│       ├── App/
│       ├── Camera/              # AVCaptureSession
│       ├── Detection/           # VNDetectRectanglesRequest + filtre aspect ratio
│       ├── Perspective/         # CIPerspectiveCorrection
│       ├── Embedding/           # modèle Core ML
│       ├── Search/              # cosine similarity (Accelerate)
│       ├── Tracking/            # association inter-frames
│       ├── UI/                  # overlays, fiche carte, historique
│       └── Resources/
│           ├── CardEncoder.mlpackage
│           ├── index.bin        # embeddings float16
│           └── cards.json       # metadata (id, nom, set, numéro, rareté, URL image)
│
└── docs/
```

---

## 2. Prérequis côté iPhone (à faire une fois)

1. **Xcode** installé sur le Mac (App Store), avec la plateforme iOS.
2. **Apple ID** ajouté dans Xcode → Settings → Accounts.
   - Compte **gratuit** : suffisant pour installer sur votre propre iPhone. Limite : l'app expire au bout de **7 jours** (il suffit de la redéployer), max 3 apps.
   - Compte développeur payant (99 €/an) : nécessaire seulement pour TestFlight / App Store, pas pour tester.
3. **iPhone en mode développeur** : Réglages → Confidentialité et sécurité → Mode développeur (apparaît après un premier branchement à Xcode).
4. Premier lancement : brancher l'iPhone en USB, sélectionner l'appareil dans Xcode, Run. Ensuite le déploiement marche aussi en Wi-Fi.

⚠️ **Le simulateur iOS n'a pas de caméra.** L'UI et le pipeline (sur photos importées) se testent en simulateur ; tout ce qui touche au flux caméra se teste sur l'iPhone physique.

---

## 3. Phases

### Phase 1 — Dataset (Python, sur Mac)

Identique à la spec d'origine :

- [ ] Télécharger les images (dataset Kaggle « Pokémon TCG All Image Cards » ou directement les `image_large` de l'API pokemontcg.io).
- [ ] Récupérer les métadonnées structurées : repo GitHub **PokemonTCG/pokemon-tcg-data** (JSON complet, pas de rate-limit).
- [ ] Normaliser les IDs (`swsh1-25`), joindre images ↔ metadata, dédoublonner.
- [ ] Script d'augmentation : rotation, perspective, éclairage, ombres, reflets, flou, simulation de sleeve — servira **uniquement à l'évaluation** (mesurer la robustesse), pas à l'entraînement.

**Livrable : `data/processed/` propre + `cards.json`.**

### Phase 2 — Identification offline (Python, sur Mac)

Le cœur du projet. Ne pas toucher à iOS tant que ça ne marche pas.

- [ ] Implémenter `encoder.py` avec 2–3 candidats : **MobileCLIP-S2** (Apple, pensé pour le Neural Engine), **DINOv2 ViT-S/14**, CLIP ViT-B/32 comme baseline.
- [ ] `build_embeddings.py` : encoder les ~20 000 cartes de référence.
- [ ] `identify.py` : image → embedding → top-5 (FAISS en local).
- [ ] `evaluate.py` : top-1/top-5 accuracy sur des **vues augmentées** (et idéalement 50–100 vraies photos de vos cartes prises à l'iPhone — c'est le vrai test).
- [ ] Mesurer la confusion entre cartes visuellement proches (même illustration dans plusieurs sets) ; si besoin, stratégie de départage (crop sur le bandeau numéro/set, ou re-ranking).

**Critère de sortie : top-1 > ~90 % sur photos réelles d'une carte bien cadrée.**
**Le modèle retenu est celui qui offre le meilleur compromis accuracy / vitesse Core ML.**

### Phase 3 — Export vers iOS

- [ ] `export_coreml.py` : conversion du modèle retenu en `.mlpackage` via `coremltools` (float16, cible Neural Engine).
- [ ] **Test de parité** : vérifier que l'embedding Core ML ≈ embedding PyTorch (similarité cosinus > 0,99 sur un échantillon) — c'est le piège classique de la conversion (préprocessing : resize, normalisation, ordre des canaux).
- [ ] `export_index.py` : embeddings → `index.bin` (float16, ~20 Mo) + `cards.json`.
- [ ] Benchmark de l'inférence Core ML sur Mac (`coremltools` permet de prédire) puis sur iPhone.

**Livrables : `CardEncoder.mlpackage`, `index.bin`, `cards.json`.**

### Phase 4 — App iOS MVP : une photo → une carte

Première app sur l'iPhone, volontairement minimale :

- [ ] Projet Xcode SwiftUI `PokeScanner`.
- [ ] Écran : bouton photo (ou import depuis la photothèque) → détection du rectangle (Vision) → correction de perspective (Core Image) → embedding (Core ML) → cosine top-5 (Accelerate) → affichage **nom, set, numéro, confiance, miniature**.
- [ ] Si la détection de rectangle échoue : fallback = utiliser l'image entière (carte bien cadrée).
- [ ] Déployer sur l'iPhone, tester sur de vraies cartes.

**C'est le jalon « ça marche dans ma main ». Tout le reste est de l'amélioration.**

### Phase 5 — Flux caméra temps réel

- [ ] `AVCaptureSession` avec preview plein écran.
- [ ] `VNDetectRectanglesRequest` sur chaque frame (filtré par aspect ratio ~0,716 ± tolérance, aire minimale) → gère plusieurs cartes par frame.
- [ ] Homographie via les 4 coins → crop normalisé → identification.
- [ ] Overlay par carte : cadre + nom + confiance.
- [ ] Budget perf : détection à chaque frame, identification seulement quand nécessaire (voir Phase 6). Viser 30 fps de preview, identification < 100 ms.

**Fallback si Vision ne suffit pas** (fonds très chargés, cartes qui se chevauchent) : entraîner un petit YOLO (~200–500 images annotées, Roboflow ou CVAT) et l'exporter en Core ML. C'est le plan B, pas le point de départ.

### Phase 6 — Tracking et stabilisation

- [ ] Association inter-frames simple par IoU des bounding boxes (ou `VNTrackObjectRequest`).
- [ ] Une carte trackée n'est **pas ré-identifiée** à chaque frame : identification 1×, puis vote majoritaire sur ~5 identifications espacées pour stabiliser.
- [ ] Gestion apparition/disparition ; lissage de l'overlay.

### Phase 7 — UX et collection

- [ ] Historique des cartes scannées (SwiftData).
- [ ] Mode « scan de collection » : ajout automatique quand une carte est identifiée avec confiance stable.
- [ ] Export CSV/JSON.
- [ ] Fiche carte détaillée (image officielle haute résolution, rareté).

### Phase 8 — Prix

- [ ] Source de prix (l'API pokemontcg.io expose les prix TCGplayer/Cardmarket).
- [ ] Requête réseau à la demande (les prix, contrairement à l'identification, justifient d'être en ligne).
- [ ] Affichage prix sur la fiche et valeur totale de la collection.

---

## 4. Mises à jour pour les nouvelles extensions

```text
Nouvelle extension
    ↓
ml/scripts: download → build_embeddings → export_index
    ↓
Nouveaux index.bin + cards.json
    ↓
Mise à jour de l'app (ou téléchargement de l'index à distance, plus tard)
```

Aucun réentraînement. À terme, l'app peut télécharger l'index depuis un CDN au lieu de l'embarquer.

---

## 4bis. Résultats mesurés (14 photos iPhone, cartes FR)

Sur les 6 premières photos, en isolant l'apport de chaque étape :

| Étape | top-1 |
|---|---|
| Photo iPhone brute | 1/6 |
| Détecteur OpenCV maison + redressement | 3/6 |
| Vision d'Apple + redressement + sélection score+marge | **4/6** |

Sur les 12 photos mono-carte identifiables (2 autres cartes sont absentes de
l'index, voir plus bas) : **top-1 7/12, top-5 9/12**.

### La marge est un signal de confiance fiable

Classées par marge décroissante, les 6 premières sont toutes correctes :

| seuil de marge | résultats retenus | précision |
|---|---|---|
| ≥ 0,04 | 6/12 | **100 %** |
| ≥ 0,02 | 7/12 | 86 % |
| aucun seuil | 12/12 | 58 % |

Conséquence directe pour l'app : afficher un résultat ferme au-dessus de 0,04,
et en dessous proposer le top-5 ou demander de stabiliser la caméra. Le score
absolu, lui, ne discrimine rien — le pire échec (5030) avait le 2e meilleur
score de tout le jeu, à 0,8598.

Ce que ces mesures ont tranché :

- **L'embedding n'était pas le problème.** Sur un crop correct, la bonne carte
  sort à 0,84-0,87 avec une marge nette. Sur photo brute, le vecteur décrit
  surtout la table et l'arrière-plan.
- **Les cartes françaises sont reconnues depuis l'index anglais**, et de loin :
  l'illustration est identique d'une langue à l'autre, seul le texte diffère.
  Ça évite d'avoir à constituer un index par langue.
- **Vision d'Apple bat nettement un détecteur OpenCV maison** : il trouve la
  carte sur les 6 photos, y compris à contre-jour et tenue à la main. Le YOLO
  prévu par la spec n'est pas nécessaire pour la carte isolée. `find_card_quads`
  dans `ml/src/detect.py` n'est plus qu'une référence de comparaison.
- **Ne jamais sélectionner sur le score absolu seul.** Une carte à l'envers
  garde un score élevé ; un crop uni (table, écran) se détache nettement sur une
  carte Énergie, elle aussi presque unie. La somme score + marge départage
  mieux que l'un ou l'autre.
- **Piège de conversion**: `cv2.imread` applique la rotation EXIF,
  `CGImageSource` non. Sur iOS le même piège se pose avec l'orientation du
  `CVPixelBuffer` passée à `VNImageRequestHandler`.

- **Le vrai point faible est l'orientation, pas le type de carte.** Une première
  lecture accusait les full-art holo. C'était faux : en inspectant les crops
  retenus, 4 des 5 échecs portaient sur une carte redressée à l'envers. Avec un
  choix parfait de variante, Méga-Blizzaroi ex — le pire échec — remonte en
  top-1. Les full-art se comportent bien une fois à l'endroit ; c'est la carte
  qui a obtenu la meilleure marge de tout le jeu (Ordres du Boss, 0,1954).
- **Deux cartes sont absentes de l'index** (set promo « MEP », non couvert par
  pokemon-tcg-data). Aucun modèle ne peut les trouver : c'est un problème de
  fraîcheur de la source de données, à surveiller à chaque nouvelle sortie.
- **Le multi-cartes ne marche que si les cartes ne se touchent pas.** Sur une
  photo de 5 cartes dont 4 se chevauchent, une seule a été isolée — la seule
  bien séparée — mais elle a été correctement identifiée. Vision a besoin de
  voir les 4 bords. Pour un scan de collection, soit on impose d'espacer les
  cartes, soit il faut le YOLO du plan B.

Les échecs restants sont soit des erreurs d'orientation à 180° (la bonne
orientation existait et scorait 0,77-0,79), soit des confusions entre cartes
consécutives d'un même set partageant un style d'illustration (Cryodo 209 vs
Baxcalibur 210 : la bonne réponse était en rang 2, à 0,006 d'écart).

## 5. Risques et parades

| Risque | Parade |
|---|---|
| Cartes quasi identiques entre sets (même illustration) | Re-ranking sur le crop du bandeau bas (numéro/set), métadonnées |
| Conversion Core ML dégrade l'embedding | Test de parité systématique (Phase 3) |
| Vision rate les rectangles sur fond chargé | Plan B : petit YOLO → Core ML |
| Reflets sur cartes holo/sleeves | Évaluer sur vraies photos dès la Phase 2 ; vote multi-frames en Phase 6 |
| Taille de l'app | Index 20 Mo + modèle 30–80 Mo : acceptable ; sinon téléchargement au premier lancement |

---

## 6. Ordre de bataille (résumé)

1. **Phase 1–2 (Python)** : dataset + identification fiable sur image fixe. *Rien d'autre avant ça.*
2. **Phase 3** : export Core ML + test de parité.
3. **Phase 4** : app iPhone photo → carte. Premier test réel dans votre main.
4. **Phase 5–6** : caméra temps réel + tracking.
5. **Phase 7–8** : collection + prix.
