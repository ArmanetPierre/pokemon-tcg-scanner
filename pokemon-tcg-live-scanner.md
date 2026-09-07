# Pokémon TCG Live Scanner

## 1. Goal

Build an application able to automatically recognize Pokémon TCG cards from a **real-time camera stream**.

The user films a table containing one or more cards. The application must:

1. detect the cards present in the image;
2. correct their perspective;
3. identify each card among the set of known Pokémon TCG cards;
4. track the cards between frames to avoid redoing the identification needlessly;
5. display the name, the set, the number and the confidence level;
6. eventually, allow the result to be enriched with price information.

Example:

```text
📷 Camera stream
      ↓
Card detection
      ↓
Perspective correction
      ↓
Embedding extraction
      ↓
Vector search
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

## 2. Technical principle

Do not train a classifier with ~20,000 classes.

The system must use a **detection + similarity search** architecture.

### Pipeline

```text
Camera
  │
  ▼
Card Detector
YOLO / equivalent model
  │
  ├── Card bounding box
  │
  ▼
Perspective correction
  │
  ▼
Image embedding model
DINO / CLIP / specialized vision model
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

### Why this approach?

Ideally, a new Pokémon set should not require retraining the model.

If a new card is added:

```text
New card
    ↓
Official image
    ↓
Embedding
    ↓
Add to the vector database
```

The system can then recognize the new card.

---

## 3. Dataset

Use as a base a dataset containing the images of the Pokémon TCG cards.

Dataset identified:

**Pokémon TCG – All Image Cards**
- about 20,000+ images;
- many sets;
- images organized by set.

Complete the images with the structured data of the **Pokémon TCG API / Pokémon TCG Data** project.

For each card, keep at minimum:

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

The dataset must be considered a **reference base**, not a final training dataset.

---

## 4. Realistic data to generate

The official scans are clean and perfectly framed.

To work with a real camera, the system will have to be robust to:

- rotation;
- perspective;
- scale change;
- different lighting;
- shadows;
- glare;
- slight blur;
- cards in a sleeve;
- slightly masked cards;
- complex backgrounds;
- several cards in the same image;
- cards partially out of frame.

At first, use **data augmentation**.

Later, create a synthetic dataset from the official scans.

---

## 5. Detection

Start with **YOLO**.

Goal of the detector:

```text
Input:
  camera frame

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

The detector does not need to know which card is present.

It must only answer:

> "There is a Pokémon card at this spot."

### First MVP step

Create a small annotated dataset of cards in different configurations and train a detection model.

Do not immediately aim for perfection.

Initial goal:

- detect 1 card;
- then several cards;
- then partially masked cards.

---

## 6. Perspective correction

After detection, transform the bounding box into a normalized card image.

Goal:

```text
Tilted photo
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

Eventually, use detection of the card's **4 corners** and a homography rather than a simple rectangular crop.

---

## 7. Identification

Test several embedding models:

- CLIP;
- DINOv2 / DINOv3;
- specialized vision model;
- possibly a model trained specifically on Pokémon cards.

For each reference image:

```text
card image
    ↓
embedding
    ↓
vector
```

Build a base:

```text
card_id → embedding + metadata
```

When a card is detected:

```text
camera image
    ↓
embedding
    ↓
nearest neighbors
    ↓
top 5 candidates
```

Example:

```text
1. pikachu-base-58     0.982
2. pikachu-xy-42       0.914
3. pikachu-swsh-063    0.901
4. pikachu-sv-051      0.887
5. pikachu-promo-12    0.861
```

Then use the metadata or a second model to decide between visually close cards.

---

## 8. Vector database

For the local prototype:

**FAISS** is enough.

Later:

- Qdrant;
- pgvector;
- another vector database.

With ~20,000 cards, the search is very light.

You need to keep:

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

## 9. Video tracking

Do not redo the full identification on every frame.

Pipeline:

```text
Frame 1
  ↓
Detect → Identify → Track

Frame 2
  ↓
Track existing cards
  ↓
No new identification if not needed

Frame N
  ↓
New card detected
  ↓
Identify
```

Use for example:

- ByteTrack;
- BoT-SORT;
- tracker built into YOLO.

This will keep a high FPS.

---

## 10. MVP interface

To start, make a simple application.

### Recommended option

**Python + OpenCV + FastAPI/Gradio/Streamlit**

Interface:

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

Each detected card must display:

- name;
- set;
- number;
- confidence;
- possibly a thumbnail.

---

## 11. Repository architecture

Proposal:

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

- [ ] retrieve the images;
- [ ] retrieve the metadata;
- [ ] normalize the IDs;
- [ ] remove the duplicates;
- [ ] create a clean dataset;
- [ ] generate a few augmentations.

### Phase 2 — Offline identification

Before the camera, get this working:

```text
card image → embedding → nearest neighbor → card
```

Measure:

- Top-1 accuracy;
- Top-5 accuracy;
- confusion between similar cards;
- inference time.

### Phase 3 — Detection

- [ ] annotate multi-card images;
- [ ] train YOLO;
- [ ] detect one card;
- [ ] detect several cards;
- [ ] test different angles.

### Phase 4 — Camera pipeline

```text
Camera
 → Detection
 → Perspective
 → Embedding
 → Search
 → Result
```

Initial goal: near-real-time operation on an RTX 3080.

### Phase 5 — Tracking

- [ ] integrate ByteTrack/BoT-SORT;
- [ ] avoid recomputations;
- [ ] stabilize the results;
- [ ] handle appearance/disappearance of cards.

### Phase 6 — UX

- [ ] web interface;
- [ ] overlays;
- [ ] history of detected cards;
- [ ] automatic scan;
- [ ] CSV/JSON export.

### Phase 7 — Prices

Add a price source and display:

```text
Pikachu
Base Set 58/102
Near Mint

💰 Price:
€XX.XX
```

---

## 13. Important constraints

### Do not start by building everything

The first goal must be extremely simple:

```text
ONE IMAGE
   ↓
ONE CARD
   ↓
CORRECT IDENTIFICATION
```

Only then:

```text
ONE IMAGE
   ↓
SEVERAL CARDS
```

Then:

```text
VIDEO STREAM
   ↓
SEVERAL CARDS
   ↓
TRACKING
```

### Do not train a model with 20,000 classes

Favour:

```text
Detector
+
Embedding model
+
Vector search
```

This architecture will be much more flexible for new sets.

---

## 14. Final goal

The final product must allow you to take a phone, a webcam or a video stream and do:

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

Eventually, the scanner could become an **automatic collection scan** tool, able to identify the cards, build a collection and retrieve their value.

---

## 15. First task for Claude Code

Start by asking Claude Code to:

1. create the repository following the architecture above;
2. set up Python + PyTorch;
3. create a dataset download/import script;
4. normalize the metadata;
5. create a first `build_embeddings.py` script;
6. use a pre-trained vision embedding model;
7. build a FAISS index;
8. create `identify_card.py` that takes an image as input and returns the 5 closest cards;
9. create a few tests;
10. measure the accuracy and inference time before starting the camera part.

**Important: do not implement the camera before obtaining reliable identification on individual images.**
