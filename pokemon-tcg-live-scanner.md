# Pokémon TCG Live Scanner

## 1. Objectif

Construire une application capable de reconnaître automatiquement des cartes Pokémon TCG à partir d'un **flux caméra en temps réel**.

L'utilisateur filme une table contenant une ou plusieurs cartes. L'application doit :

1. détecter les cartes présentes dans l'image ;
2. corriger leur perspective ;
3. identifier chaque carte parmi l'ensemble des cartes Pokémon TCG connues ;
4. suivre les cartes entre les frames pour éviter de refaire inutilement l'identification ;
5. afficher le nom, le set, le numéro et le niveau de confiance ;
6. à terme, permettre d'enrichir le résultat avec des informations de prix.

Exemple :

```text
📷 Flux caméra
      ↓
Détection des cartes
      ↓
Correction perspective
      ↓
Extraction embedding
      ↓
Recherche vectorielle
      ↓
Identification
      ↓
┌─────────────────────────────┐
│ Pikachu                     │
│ Base Set — 58/102           │
│ Confidence: 98.7%           │
└─────────────────────────────┘
```

---

## 2. Principe technique

Ne pas entraîner un classifieur avec ~20 000 classes.

Le système doit utiliser une architecture **détection + recherche par similarité**.

### Pipeline

```text
Camera
  │
  ▼
Card Detector
YOLO / modèle équivalent
  │
  ├── Card bounding box
  │
  ▼
Perspective correction
  │
  ▼
Image embedding model
DINO / CLIP / modèle vision spécialisé
  │
  ▼
Vector similarity search
FAISS / Qdrant
  │
  ▼
Card database
  │
  ▼
Card ID + metadata
```

### Pourquoi cette approche ?

Une nouvelle extension Pokémon ne doit idéalement pas nécessiter de réentraîner le modèle.

Si une nouvelle carte est ajoutée :

```text
Nouvelle carte
    ↓
Image officielle
    ↓
Embedding
    ↓
Ajout dans la vector database
```

Le système peut alors reconnaître la nouvelle carte.

---

## 3. Dataset

Utiliser comme base un dataset contenant les images des cartes Pokémon TCG.

Dataset identifié :

**Pokémon TCG – All Image Cards**
- environ 20 000+ images ;
- nombreuses extensions ;
- images organisées par set.

Compléter les images avec les données structurées du projet **Pokémon TCG API / Pokémon TCG Data**.

Pour chaque carte, conserver au minimum :

```json
{
  "id": "swsh1-25",
  "name": "Pikachu",
  "set": "Sword & Shield",
  "set_id": "swsh1",
  "number": "25",
  "rarity": "...",
  "image_small": "...",
  "image_large": "..."
}
```

Le dataset doit être considéré comme une **base de référence**, pas comme un dataset d'entraînement final.

---

## 4. Données réalistes à générer

Les scans officiels sont propres et parfaitement cadrés.

Pour fonctionner avec une caméra réelle, le système devra être robuste à :

- rotation ;
- perspective ;
- changement d'échelle ;
- éclairage différent ;
- ombres ;
- reflets ;
- flou léger ;
- cartes sous sleeve ;
- cartes légèrement masquées ;
- arrière-plans complexes ;
- plusieurs cartes dans la même image ;
- cartes partiellement hors champ.

Dans un premier temps, utiliser de la **data augmentation**.

Plus tard, créer un dataset synthétique à partir des scans officiels.

---

## 5. Détection

Commencer avec **YOLO**.

Objectif du détecteur :

```text
Input:
  frame caméra

Output:
  [
    {
      x1,
      y1,
      x2,
      y2,
      confidence
    }
  ]
```

Le détecteur n'a pas besoin de savoir quelle carte est présente.

Il doit uniquement répondre :

> "Il y a une carte Pokémon à cet endroit."

### Première étape MVP

Créer un petit dataset annoté de cartes dans différentes configurations et entraîner un modèle de détection.

Ne pas chercher immédiatement la perfection.

Objectif initial :

- détecter 1 carte ;
- puis plusieurs cartes ;
- puis cartes partiellement masquées.

---

## 6. Correction de perspective

Après détection, transformer le bounding box en image de carte normalisée.

Objectif :

```text
Photo inclinée
      ↓
┌───────────╲
│            ╲
│   CARD      │
│             │
└─────────────┘

      ↓

┌─────────────┐
│             │
│    CARD     │
│             │
└─────────────┘
```

À terme, utiliser une détection des **4 coins** de la carte et une homographie plutôt qu'un simple crop rectangulaire.

---

## 7. Identification

Tester plusieurs modèles d'embedding :

- CLIP ;
- DINOv2 / DINOv3 ;
- modèle vision spécialisé ;
- éventuellement un modèle entraîné spécifiquement sur les cartes Pokémon.

Pour chaque image de référence :

```text
card image
    ↓
embedding
    ↓
vector
```

Construire une base :

```text
card_id → embedding + metadata
```

Lorsqu'une carte est détectée :

```text
camera image
    ↓
embedding
    ↓
nearest neighbors
    ↓
top 5 candidates
```

Exemple :

```text
1. pikachu-base-58     0.982
2. pikachu-xy-42       0.914
3. pikachu-swsh-063    0.901
4. pikachu-sv-051      0.887
5. pikachu-promo-12    0.861
```

Utiliser ensuite les métadonnées ou un second modèle pour départager les cartes visuellement proches.

---

## 8. Vector database

Pour le prototype local :

**FAISS** est suffisant.

Plus tard :

- Qdrant ;
- pgvector ;
- autre vector database.

Avec ~20 000 cartes, la recherche est très légère.

Il faut conserver :

```text
embedding
card_id
set_id
set_name
card_number
name
image URL
```

---

## 9. Tracking vidéo

Ne pas refaire l'identification complète à chaque frame.

Pipeline :

```text
Frame 1
  ↓
Detect → Identify → Track

Frame 2
  ↓
Track existing cards
  ↓
Pas de nouvelle identification si nécessaire

Frame N
  ↓
Nouvelle carte détectée
  ↓
Identify
```

Utiliser par exemple :

- ByteTrack ;
- BoT-SORT ;
- tracker intégré à YOLO.

Cela permettra de conserver un FPS élevé.

---

## 10. Interface MVP

Pour commencer, faire une application simple.

### Option recommandée

**Python + OpenCV + FastAPI/Gradio/Streamlit**

Interface :

```text
┌────────────────────────────────────────────┐
│                                            │
│             CAMERA STREAM                 │
│                                            │
│    ┌────────────┐                          │
│    │ Pikachu    │                          │
│    │ 98.7%      │                          │
│    └────────────┘                          │
│                                            │
│                 ┌────────────┐             │
│                 │ Charizard  │             │
│                 │ 96.1%      │             │
│                 └────────────┘             │
│                                            │
└────────────────────────────────────────────┘
```

Chaque carte détectée doit afficher :

- nom ;
- set ;
- numéro ;
- confidence ;
- éventuellement miniature.

---

## 11. Architecture du repository

Proposition :

```text
pokemon-card-scanner/
│
├── README.md
├── pyproject.toml
├── docker-compose.yml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── embeddings/
│
├── models/
│   ├── detector/
│   └── embedding/
│
├── src/
│   ├── detection/
│   │   └── detector.py
│   │
│   ├── perspective/
│   │   └── transform.py
│   │
│   ├── embedding/
│   │   └── encoder.py
│   │
│   ├── search/
│   │   └── vector_search.py
│   │
│   ├── tracking/
│   │   └── tracker.py
│   │
│   ├── camera/
│   │   └── camera.py
│   │
│   └── pipeline.py
│
├── scripts/
│   ├── download_dataset.py
│   ├── build_embeddings.py
│   ├── build_index.py
│   └── evaluate.py
│
└── tests/
```

---

## 12. Roadmap

### Phase 1 — Dataset

- [ ] récupérer les images ;
- [ ] récupérer les métadonnées ;
- [ ] normaliser les IDs ;
- [ ] supprimer les doublons ;
- [ ] créer un dataset propre ;
- [ ] générer quelques augmentations.

### Phase 2 — Identification offline

Avant la caméra, faire fonctionner :

```text
image carte → embedding → nearest neighbor → carte
```

Mesurer :

- Top-1 accuracy ;
- Top-5 accuracy ;
- confusion entre cartes similaires ;
- temps d'inférence.

### Phase 3 — Détection

- [ ] annoter des images multi-cartes ;
- [ ] entraîner YOLO ;
- [ ] détecter une carte ;
- [ ] détecter plusieurs cartes ;
- [ ] tester différents angles.

### Phase 4 — Pipeline caméra

```text
Camera
 → Detection
 → Perspective
 → Embedding
 → Search
 → Result
```

Objectif initial : fonctionnement quasi temps réel sur RTX 3080.

### Phase 5 — Tracking

- [ ] intégrer ByteTrack/BoT-SORT ;
- [ ] éviter les recalculs ;
- [ ] stabiliser les résultats ;
- [ ] gérer apparition/disparition des cartes.

### Phase 6 — UX

- [ ] interface web ;
- [ ] overlays ;
- [ ] historique des cartes détectées ;
- [ ] scan automatique ;
- [ ] export CSV/JSON.

### Phase 7 — Prix

Ajouter une source de prix et afficher :

```text
Pikachu
Base Set 58/102
Near Mint

💰 Price:
€XX.XX
```

---

## 13. Contraintes importantes

### Ne pas commencer par tout construire

Le premier objectif doit être extrêmement simple :

```text
UNE IMAGE
   ↓
UNE CARTE
   ↓
IDENTIFICATION CORRECTE
```

Ensuite seulement :

```text
UNE IMAGE
   ↓
PLUSIEURS CARTES
```

Puis :

```text
FLUX VIDÉO
   ↓
PLUSIEURS CARTES
   ↓
TRACKING
```

### Ne pas entraîner un modèle avec 20 000 classes

Privilégier :

```text
Detector
+
Embedding model
+
Vector search
```

Cette architecture sera beaucoup plus flexible pour les nouvelles extensions.

---

## 14. Objectif final

Le produit final doit permettre de prendre un téléphone, une webcam ou un flux vidéo et de faire :

```text
                 📱 CAMERA
                     │
                     ▼
              ┌──────────────┐
              │ Card detector │
              └──────┬───────┘
                     │
           ┌─────────┼─────────┐
           ▼         ▼         ▼
         Card 1    Card 2    Card 3
           │         │         │
           ▼         ▼         ▼
       Embedding  Embedding  Embedding
           │         │         │
           └─────────┼─────────┘
                     ▼
                Vector Search
                     │
                     ▼
             Pokémon TCG Data
                     │
                     ▼
          ┌────────────────────┐
          │ Pikachu             │
          │ Base Set 58/102     │
          │ 98.7%               │
          ├────────────────────┤
          │ Charizard           │
          │ Base Set 4/102      │
          │ 97.2%               │
          └────────────────────┘
```

À terme, le scanner pourrait devenir un outil de **scan automatique de collection**, capable d'identifier les cartes, constituer une collection et récupérer leur valeur.

---

## 15. Première tâche pour Claude Code

Commencer par demander à Claude Code de :

1. créer le repository selon l'architecture ci-dessus ;
2. mettre en place Python + PyTorch ;
3. créer un script de téléchargement/import du dataset ;
4. normaliser les métadonnées ;
5. créer un premier script `build_embeddings.py` ;
6. utiliser un modèle d'embedding vision pré-entraîné ;
7. construire un index FAISS ;
8. créer `identify_card.py` qui prend une image en entrée et retourne les 5 cartes les plus proches ;
9. créer quelques tests ;
10. mesurer la précision et le temps d'inférence avant de commencer la partie caméra.

**Important : ne pas implémenter la caméra avant d'avoir obtenu une identification fiable sur des images individuelles.**
