# Detailed plan — Pokémon TCG Scanner on iOS

> Complement to `pokemon-tcg-live-scanner.md`, adapted to the goal: **an iOS app testable on an iPhone**, with on-device recognition (no server needed in use).

---

## 0. Key architecture decision

The original spec targets a real-time Python pipeline on a GPU. For iOS, two options:

| | Option A — All on the iPhone (on-device) | Option B — iPhone camera + inference server |
|---|---|---|
| Latency | Excellent (Neural Engine) | Depends on the network |
| Works offline | ✅ | ❌ |
| Deployment complexity | A single app | App + server to host |
| ML iteration | Core ML conversion on every change | Immediate |

**Recommendation: Option A (on-device)**, with a prototyping phase in Python on a Mac to choose and validate the models *before* converting them to Core ML. This is realistic because:

- **Card detection probably does not need a YOLO**: a card is a rectangle of known aspect ratio (63×88 mm ≈ 0.716). Apple's **Vision** framework (`VNDetectRectanglesRequest`) detects rectangles and gives the **4 corners** — which also settles perspective correction (`CIPerspectiveCorrection`). Zero training, zero annotated dataset for the MVP.
- **Vector search does not need FAISS on an iPhone**: ~20,000 embeddings × 512 dims in float16 ≈ **20 MB**. A brute-force cosine similarity via Accelerate/MPS takes a few milliseconds. The index is a simple binary file bundled in the app.
- **The embedding runs on the Neural Engine**: MobileCLIP (Apple) or DINOv2 ViT-S converted via `coremltools` run at several dozen FPS on a recent iPhone.

The principle of the spec stays intact: **detection + embedding + similarity search**, no 20,000-class classifier. A new set = regenerate the index file, no retraining.

---

## 1. Repo architecture (ML + iOS monorepo)

```text
pokemon-card-scanner/
│
├── README.md
├── plan-ios.md
│
├── ml/                          # Python prototyping on a Mac
│   ├── pyproject.toml
│   ├── data/
│   │   ├── raw/                 # raw images + JSON
│   │   ├── processed/           # cleaned dataset
│   │   └── embeddings/          # index + metadata
│   ├── scripts/
│   │   ├── download_dataset.py
│   │   ├── build_metadata.py    # ID normalization, deduplication
│   │   ├── build_embeddings.py
│   │   ├── evaluate.py          # top-1/top-5 on augmented views
│   │   ├── export_coreml.py     # model → .mlpackage
│   │   └── export_index.py      # embeddings → binary file + JSON metadata
│   └── src/
│       ├── encoder.py           # abstraction over CLIP/DINOv2/MobileCLIP
│       ├── augment.py           # rotation, perspective, lighting, blur, sleeve
│       ├── search.py            # FAISS (Python only)
│       └── identify.py          # image → top-5 candidates
│
├── ios/
│   └── PokeScanner/             # Xcode project (SwiftUI)
│       ├── PokeScanner.xcodeproj
│       ├── App/
│       ├── Camera/              # AVCaptureSession
│       ├── Detection/           # VNDetectRectanglesRequest + aspect ratio filter
│       ├── Perspective/         # CIPerspectiveCorrection
│       ├── Embedding/           # Core ML model
│       ├── Search/              # cosine similarity (Accelerate)
│       ├── Tracking/            # inter-frame association
│       ├── UI/                  # overlays, card sheet, history
│       └── Resources/
│           ├── CardEncoder.mlpackage
│           ├── index.bin        # float16 embeddings
│           └── cards.json       # metadata (id, name, set, number, rarity, image URL)
│
└── docs/
```

---

## 2. iPhone-side prerequisites (do once)

1. **Xcode** installed on the Mac (App Store), with the iOS platform.
2. **Apple ID** added in Xcode → Settings → Accounts.
   - **Free** account: enough to install on your own iPhone. Limit: the app expires after **7 days** (just redeploy it), max 3 apps.
   - Paid developer account (€99/year): needed only for TestFlight / App Store, not for testing.
3. **iPhone in developer mode**: Settings → Privacy & Security → Developer Mode (appears after a first connection to Xcode).
4. First launch: plug the iPhone in over USB, select the device in Xcode, Run. After that, deployment also works over Wi-Fi.

⚠️ **The iOS Simulator has no camera.** The UI and the pipeline (on imported photos) are tested in the Simulator; anything touching the camera stream is tested on the physical iPhone.

---

## 3. Phases

### Phase 1 — Dataset (Python, on the Mac)

Identical to the original spec:

- [ ] Download the images (Kaggle dataset "Pokémon TCG All Image Cards" or directly the `image_large` from the pokemontcg.io API).
- [ ] Retrieve the structured metadata: GitHub repo **PokemonTCG/pokemon-tcg-data** (full JSON, no rate limit).
- [ ] Normalize the IDs (`swsh1-25`), join images ↔ metadata, deduplicate.
- [ ] Augmentation script: rotation, perspective, lighting, shadows, glare, blur, sleeve simulation — will serve **only for evaluation** (measuring robustness), not for training.

**Deliverable: a clean `data/processed/` + `cards.json`.**

### Phase 2 — Offline identification (Python, on the Mac)

The heart of the project. Do not touch iOS until this works.

- [ ] Implement `encoder.py` with 2–3 candidates: **MobileCLIP-S2** (Apple, designed for the Neural Engine), **DINOv2 ViT-S/14**, CLIP ViT-B/32 as a baseline.
- [ ] `build_embeddings.py`: encode the ~20,000 reference cards.
- [ ] `identify.py`: image → embedding → top-5 (FAISS locally).
- [ ] `evaluate.py`: top-1/top-5 accuracy on **augmented views** (and ideally 50–100 real photos of your cards taken with the iPhone — that is the real test).
- [ ] Measure the confusion between visually close cards (same artwork across several sets); if needed, a tie-breaking strategy (crop on the number/set strip, or re-ranking).

**Exit criterion: top-1 > ~90 % on real photos of a well-framed card.**
**The model chosen is the one that offers the best accuracy / Core ML speed trade-off.**

### Phase 3 — Export to iOS

- [ ] `export_coreml.py`: conversion of the chosen model to `.mlpackage` via `coremltools` (float16, Neural Engine target).
- [ ] **Parity test**: check that the Core ML embedding ≈ PyTorch embedding (cosine similarity > 0.99 on a sample) — this is the classic conversion trap (preprocessing: resize, normalization, channel order).
- [ ] `export_index.py`: embeddings → `index.bin` (float16, ~20 MB) + `cards.json`.
- [ ] Benchmark the Core ML inference on the Mac (`coremltools` can predict) then on the iPhone.

**Deliverables: `CardEncoder.mlpackage`, `index.bin`, `cards.json`.**

### Phase 4 — iOS MVP app: one photo → one card

First app on the iPhone, deliberately minimal:

- [ ] SwiftUI Xcode project `PokeScanner`.
- [ ] Screen: photo button (or import from the photo library) → rectangle detection (Vision) → perspective correction (Core Image) → embedding (Core ML) → cosine top-5 (Accelerate) → display **name, set, number, confidence, thumbnail**.
- [ ] If rectangle detection fails: fallback = use the whole image (well-framed card).
- [ ] Deploy to the iPhone, test on real cards.

**This is the "it works in my hand" milestone. Everything else is improvement.**

### Phase 5 — Real-time camera stream

- [ ] `AVCaptureSession` with full-screen preview.
- [ ] `VNDetectRectanglesRequest` on each frame (filtered by aspect ratio ~0.716 ± tolerance, minimum area) → handles several cards per frame.
- [ ] Homography via the 4 corners → normalized crop → identification.
- [ ] Overlay per card: frame + name + confidence.
- [ ] Perf budget: detection every frame, identification only when necessary (see Phase 6). Aim for 30 fps preview, identification < 100 ms.

**Fallback if Vision is not enough** (very busy backgrounds, overlapping cards): train a small YOLO (~200–500 annotated images, Roboflow or CVAT) and export it to Core ML. That is plan B, not the starting point.

### Phase 6 — Tracking and stabilization

- [ ] Simple inter-frame association by IoU of the bounding boxes (or `VNTrackObjectRequest`).
- [ ] A tracked card is **not re-identified** on every frame: identification once, then a majority vote over ~5 spaced identifications to stabilize.
- [ ] Handling of appearance/disappearance; overlay smoothing.

### Phase 7 — UX and collection

- [ ] History of scanned cards (SwiftData).
- [ ] "Collection scan" mode: automatic addition when a card is identified with stable confidence.
- [ ] CSV/JSON export.
- [ ] Detailed card sheet (official high-resolution image, rarity).

### Phase 8 — Prices

- [ ] Price source (the pokemontcg.io API exposes TCGplayer/Cardmarket prices).
- [ ] On-demand network request (prices, unlike identification, justify being online).
- [ ] Price display on the sheet and total value of the collection.

---

## 4. Updates for new sets

```text
New set
    ↓
ml/scripts: download → build_embeddings → export_index
    ↓
New index.bin + cards.json
    ↓
App update (or remote index download, later)
```

No retraining. Eventually, the app can download the index from a CDN instead of bundling it.

---

## 4bis. Measured results (14 iPhone photos, FR cards)

On the first 6 photos, isolating the contribution of each step:

| Step | top-1 |
|---|---|
| Raw iPhone photo | 1/6 |
| Home-grown OpenCV detector + rectification | 3/6 |
| Apple's Vision + rectification + score+margin selection | **4/6** |

On the 12 identifiable single-card photos (2 other cards are absent from the
index, see below): **top-1 7/12, top-5 9/12**.

### The margin is a reliable confidence signal

Sorted by decreasing margin, the first 6 are all correct:

| margin threshold | results kept | precision |
|---|---|---|
| ≥ 0.04 | 6/12 | **100 %** |
| ≥ 0.02 | 7/12 | 86 % |
| no threshold | 12/12 | 58 % |

Direct consequence for the app: show a firm result above 0.04, and below it offer
the top-5 or ask to steady the camera. The absolute score, for its part,
discriminates nothing — the worst failure (5030) had the 2nd-best score of the
whole set, at 0.8598.

What these measurements settled:

- **The embedding was not the problem.** On a correct crop, the right card comes
  out at 0.84-0.87 with a clear margin. On a raw photo, the vector mostly
  describes the table and the background.
- **French cards are recognized from the English index**, and by a wide margin:
  the artwork is identical from one language to the other, only the text differs.
  This avoids having to build a per-language index.
- **Apple's Vision clearly beats a home-grown OpenCV detector**: it finds the
  card on all 6 photos, including backlit and hand-held. The YOLO planned by the
  spec is not needed for the isolated card. `find_card_quads` in
  `ml/src/detect.py` is now just a comparison reference.
- **Never select on the absolute score alone.** An upside-down card keeps a high
  score; a plain crop (table, screen) stands out clearly against an Energy card,
  itself almost plain. The sum score + margin decides better than either one.
- **Conversion trap**: `cv2.imread` applies the EXIF rotation, `CGImageSource`
  does not. On iOS the same trap arises with the `CVPixelBuffer` orientation
  passed to `VNImageRequestHandler`.

- **The real weak point is the orientation, not the card type.** A first reading
  blamed the full-art holos. That was wrong: inspecting the retained crops, 4 of
  the 5 failures were on a card rectified upside down. With a perfect choice of
  variant, Mega Blastoise ex — the worst failure — comes back to top-1. Full-arts
  behave well once the right way up; it is the card that got the best margin of
  the whole set (Boss's Orders, 0.1954).
- **Two cards are absent from the index** (promo set "MEP", not covered by
  pokemon-tcg-data). No model can find them: it is a data-source freshness
  problem, to watch on every new release.
- **Multi-card only works if the cards do not touch.** On a photo of 5 cards
  where 4 overlapped, only one was isolated — the only well-separated one — but
  it was correctly identified. Vision needs to see the 4 edges. For a collection
  scan, either you require the cards to be spaced out, or you need plan B's YOLO.

The remaining failures are either 180° orientation errors (the right orientation
existed and scored 0.77-0.79), or confusions between consecutive cards of one set
sharing an artwork style (Cryogonal 209 vs Baxcalibur 210: the right answer was
at rank 2, 0.006 apart).

## 5. Risks and countermeasures

| Risk | Countermeasure |
|---|---|
| Near-identical cards across sets (same artwork) | Re-ranking on the bottom-strip crop (number/set), metadata |
| Core ML conversion degrades the embedding | Systematic parity test (Phase 3) |
| Vision misses the rectangles on a busy background | Plan B: small YOLO → Core ML |
| Glare on holo cards / sleeves | Evaluate on real photos from Phase 2 on; multi-frame vote in Phase 6 |
| App size | Index 20 MB + model 30–80 MB: acceptable; otherwise download on first launch |

---

## 6. Battle order (summary)

1. **Phase 1–2 (Python)**: dataset + reliable identification on a still image. *Nothing else before that.*
2. **Phase 3**: Core ML export + parity test.
3. **Phase 4**: iPhone app photo → card. First real test in your hand.
4. **Phase 5–6**: real-time camera + tracking.
5. **Phase 7–8**: collection + prices.
