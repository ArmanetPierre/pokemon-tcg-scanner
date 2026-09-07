# Pokémon TCG card recognition — iOS integration kit

This kit identifies a Pokémon card from a photo, **entirely on device**, with no
network. It contains the model, the index of the 20,512 cards, and the validated
decision logic.

> **Measured status**: 34 correct identifications out of 39 real iPhone photos
> (87 %, CI95 73-94), from three collections photographed by several people
> (French cards, English index, ordinary conditions: backlight, sleeve, busy
> background, tilted cards, binder, one batch entirely in landscape). **29 firm
> assertions, of which 28 correct.** Seven other photos have no possible right
> answer — card back, cards from another game, Korean card, card-holder pouch,
> illegible blur — and the chain refuses **all seven**. Measured on a Mac with
> the model exported here.
>
> The figure has dropped since `kit-v4`, which announced 31/31: this is not a
> regression, it is a widened bench finally measuring what the previous one did
> not see. Read `docs/model-card.md` before announcing a performance.
>
> **It now runs on iPhone.** Swift port integrated into an Expo app, measured on
> an iPhone 13 Pro: see §8. The encoder there takes **5.5 ms** on the Neural
> Engine. §10 gathers what the move onto device taught — including a pipeline gap
> that only a real photo set brought out.

> **Get the model and the index**: they are not in the repo (94 MB). An archive
> is published as a release, **`kit-v5`** — it is what the app's `sync-model.mjs`
> downloads, and you must point at it: `kit-v4` carries the old enrolment and has
> neither the thresholds in `index.json`, nor the fingerprints, nor
> `needs_printed_number`. Regenerating them requires the 5.5 GB of reference
> images.

> **Video stream**: suitable. The embedding runs in **3.3 ms** on the Neural
> Engine and represents only 3 % of the total cost; the dominant item is Vision's
> OCR. Read §9 before designing the video loop — the architecture matters more
> there than the choice of model.

---

## 1. Contents

| File | Size | Role |
|---|---|---|
| `CardEncoder.mlpackage` | 69 MB | image encoder → 512-dimension vector |
| `index.bin` | 21 MB | 20,512 float16 vectors, L2-normalised |
| `index.json` | 0.2 MB | shape of `index.bin` + `card_ids` in row order |
| `cards.json` | 5.0 MB | display metadata + `needs_printed_number`, same order as the index |
| `encoder_meta.json` | — | preprocessing geometry to reproduce |
| `benchmarks.json` | — | reference measurements, to compare to yours (§8) |
| `test/` | — | fixtures to verify the integration (see §6) |
| `reference-python/` | — | reference implementation, commented |

Total to bundle: **~95 MB**.

The `reference-python/` folder is not to be ported as is: it is the source of
truth for behaviour. In case of doubt about a detail, it is authoritative.

---

## 2. The pipeline

Seven steps. Each has a native equivalent — nothing is to be reimplemented by
hand.

```
photo
  1. quadrilateral detection       VNDetectDocumentSegmentationRequest
                                   + VNDetectRectanglesRequest
  2. candidate filtering           geometry + variance
  3. rectification (homography)    CIPerspectiveCorrection
  4. orientation 0° / 180°         VNRecognizeTextRequest (.fast)
  5. embedding                     CardEncoder.mlpackage
  6. search                        matrix product (Accelerate)
  7. exact edition                 VNRecognizeTextRequest (.accurate)
→ card + confidence level
```

### 1. Detection

Run **both** detectors and merge their results. Document segmentation frames the
card better; the rectangle detector catches the cases it misses.

**At comparable shape, the candidates from document segmentation go first**: at
deduplication (step 2), the first seen wins, and the reverse order cost two
photos on the test bench. It is only a default priority, which the quadrilateral's
geometric quality can overturn — see step 2.

### 2. Filtering

Four filters, in this order. **This is where the processing time is decided**:
everything that survives costs a rectification, an OCR pass and an embedding, and
`VNDetectRectanglesRequest` returns up to 8 observations, most of which are
artefacts (§8).

- **Area**: reject a quadrilateral covering less than **2 %** of the photo,
  before even rectifying. On the bench, 4 of the 6.3 quadrilaterals per photo are
  less than 1 % of the area, whereas the finally retained candidate never dropped
  below 6.6 %. This single filter takes the number of crops from 10.2 to 2.4 per
  photo, i.e. **−37 % processing time**.
  ⚠️ An **area** filter, definitely not an aspect-ratio one: perspective squashes
  the apparent ratio of a card (0.66 to 0.94 measured on the winning
  quadrilaterals, against 63/88 = 0.72 flat). A width/height ratio filter
  discards real cards.
  ⚠️ To scan a display of several cards, go down to **0.5 %**: one card among ten
  necessarily covers a small fraction of the frame.
- **Deduplication by centre**: two quadrilaterals whose centres are less than
  5 % of the image's long side apart target the same card. The group is
  represented by **the best-formed quadrilateral**, measured by the equality of
  its opposite sides — `min(short side / long side)` over the two pairs of
  opposite sides, 1 for a rectangle, 0 for a degenerate quadrilateral. Compare in
  steps of 0.10 and break ties by the detector order (segmentation first).
  This criterion assumes nothing about the card's aspect ratio: the perspective
  of a hand-held photo leaves the opposite sides roughly equal, whereas a
  misplaced corner collapses the measure. Measured on one bench photo: the
  detectors returned one perfect quadrilateral (0.98) and two broken ones (0.50
  and 0.68), and keeping "the first seen" retained a broken one — crop rotated by
  90°, illegible strip, right species but wrong edition.
- **Variance**: reject a crop whose grey-level standard deviation is `< 8`. That
  eliminates the sky, a slab, a keyboard key.
  ⚠️ Do not raise this threshold. At 25 it eliminated a real backlit card
  (standard deviation 10.5). Missing a card costs far more than encoding an
  artefact, which the selection will eliminate anyway.
- **Whole-photo fallback**: if no crop reaches a score of 0.65, add the uncropped
  photo as a candidate. Needed when the card fills the frame and its edges leave
  the image.
  ⚠️ Do **not** put the whole photo in systematic competition: it steals the
  selection from legitimate crops with a tight margin.
  ⚠️ If it is the one retained, **no firm verdict is allowed** (§5). Over 46
  photos, this fallback produced no correct identification and exactly one firm
  false positive.

### 3. Rectification

Homography of the 4 corners to a portrait 734 × 1024. If the top side is longer
than the left side, the card is lying down: shift the corners by one notch before
computing the transformation.

### 4. Orientation

A single fast OCR pass over the crop, of which only **the position** of the text
boxes is kept, not their content:

- vertical centre of mass of the text (0 = top, 1 = bottom), each box weighted by
  its number of characters;
- `≥ 0.55` → card the right way up; `≤ 0.45` → flip it 180°;
- fewer than 20 characters read, or centre between the two → undecided: keep both
  orientations as competing candidates.

### 5. Embedding

Input: RGB image **256 × 256**, pixels 0-255.
Output: 512 floats, **already L2-normalised**.

The normalisation (scale, bias) is folded into the Core ML graph. Subtract or
divide nothing on the Swift side.

### 6. Search

Cosine similarity = dot product, since everything is normalised. A matrix-vector
product `(20512 × 512) · (512)` via Accelerate is enough. No vector library
needed.

Go down to **20 candidates** minimum, for two reasons. The name margin (§5) needs
to find a candidate bearing a different name, and the top of the ranking is often
saturated with reprints. And above all, step 7 only reorders what the search
gives it: **a card outside this window is lost even if its printed number is
perfectly legible.**

⚠️ Do not keep the old value of 10. One bench card came out at rank 12, with its
"113/193" perfectly clear: the search discarded it before the OCR could save it.
15 was enough, 20 leaves margin, with no regression up to 30 — the search costs
1.6 ms.

### 7. Exact edition

OCR over the **bottom 14 %** of the rectified crop, to read the "number/total"
pattern (e.g. `043/084`).

**Two cascaded passes, not one.** The `.accurate` OCR costs 51 ms and remains the
first item of the chain; the `.fast` mode on that same strip **enlarged ×2**
costs 12 ms. On the bench, the fast one settles 16 strips out of 21 against 17
for the accurate one, and above all **the two never contradict each other**: the
fast one reads the right number, or reads nothing. Hence the rule — fast pass
first, accurate pass only if the fast one designated no candidate. Average cost
24 ms instead of 51, without losing anything.

The ×2 enlargement is not optional: at scale 1, the `.fast` mode gives out on
these small characters and reads only 13 strips out of 21.

Then, among the 20 candidates: the printed total must match exactly (it is what
identifies the set), the number tolerates **one** digit error. The matching
candidate is promoted to the top.

Apply a confusion table before extraction — the OCR errs predictably on this
strip: `O Q D → 0`, `I l | ) ] → 1`, `Z → 2`, `S → 5`, `T ? → 7`, `B → 8`. The
full list is in `reference-python/edition.py`.

**Out-of-index safeguard**: if a number is read cleanly but no card in the whole
base bears this number/total pair, display "unknown card" rather than a confident
attribution. Three bench cards were in this case; they have since all joined the
index — the 60 MEP promos, then the 8 MEE energies — and are correctly identified
today. The safeguard stays indispensable for anything released after this kit.

⚠️ Only pronounce "out of index" on the **accurate pass**. Declaring a card
absent from the base on a fast reading is asserting a lot from the least reliable
mode.

---

## 3. The traps that cost hours

Each one genuinely cost time during development.

### The image orientation between Vision and the cropping

Vision works on the raw pixels. If the layer that crops applies the EXIF rotation
and not the one that detects (or the reverse), the quadrilaterals designate a
zone that does not exist — the crops come out of frame and the result is random.

In Swift: explicitly pass the same `CGImagePropertyOrientation` to
`VNImageRequestHandler` as that of the buffer used for the cropping. A portrait
iPhone photo is stored in landscape with an orientation flag; ignoring this flag
on both sides is perfectly valid, as long as you do it on both sides.

*Symptom*: scores around 0.4-0.5, identical results on different photos (the
crops are black).

### The preprocessing geometry must be identical to the pixel

The index was built with: **resize the short side to 256 in bilinear with
antialiasing, then centre-crop to 256 × 256**.

⚠️ This crop **deliberately trims the top and bottom** of a portrait card. It is
the model's original behaviour, the whole index is built this way, and the app
must do exactly the same. Resizing directly to 256 × 256 (squash) drops the
parity to **0.65**.

The filter matters too: bilinear, not bicubic.

*Symptom*: plausible but often wrong identifications, lukewarm scores.

### Never let the embedding arbitrate the orientation

Tempting: encode both orientations and keep the best score. **It does not work.**
An upside-down card still looks like a card and can score higher on a *wrong*
card than the upright crop on the right one — measured: 0.834 (wrong) against
0.788 (right).

The orientation is decided from the text position (step 4), never from the score.

### Never threshold or display the absolute score

All cards share a layout: two entirely unrelated cards already resemble each
other at ~0.79. The useful scale is 0.79 to 1, not 0 to 1.

The worst failure encountered showed the **second-best score of the whole set**
(0.8598). It is the **gap** between the 1st and the 2nd that carries the
information.

### Compute the margins before reordering

Step 7 promotes a candidate to the top. If the margins are computed afterwards,
they become negative and absurd. Correct order: compute the confidence on the
similarity ranking, **then** raise the level if the OCR settled it.

---

## 4. File formats

### `index.bin`

Raw array, **with no header**: `count × dim` `float16` values, row by row
(row-major). `index.json` gives `count` (20512), `dim` (512) and `card_ids` — the
nth identifier corresponds to the nth row.

```swift
let meta = try JSONDecoder().decode(IndexMeta.self, from: Data(contentsOf: indexJSON))
let raw = try Data(contentsOf: indexBin)          // 20,883,456 bytes
// raw.count == meta.count * meta.dim * 2
```

The vectors are already normalised (norm 1.0000 ± 0.0001). Convert to `Float32`
for the matrix product, or use Accelerate's half-precision functions directly.

The float32 → float16 pass is checked on every export, over 200 control queries:
**top-1 identical on 199/200, top-5 on 193/200**. The residual reordering
concerns already-undecidable neighbourhoods — cards separated by less than the
quantisation noise, which are a matter of reading the number anyway (see
`needs_printed_number` below).

### Check that the index and the model go together

`index.json` carries `encoder_sha256`, the fingerprint of the
`CardEncoder.mlpackage` that produced the vectors. The same value is in
`encoder_meta.json`.

**The app must compare them at startup and refuse to continue if they differ.**
An index built with one encoder and queried by another raises no error: the
search returns neighbours, the scores stay in their usual range, and the answers
are plausible and wrong. It is the same kind of silent failure as the
preprocessing geometry (§3), and the only one that was not tooled.

```swift
guard indexMeta.encoderSHA256 == encoderMeta.encoderSHA256 else {
    throw KitError.mismatchedArtifacts   // do not degrade: refuse
}
```

### `cards.json`

Array aligned with `index.json.card_ids`, one object per card:

```json
{
  "id": "me5-36", "name": "Litwick", "number": "36", "rarity": "Common",
  "set_id": "me5", "set_name": "Pitch Black", "set_printed_total": 84,
  "image_small": "https://images.pokemontcg.io/me5/36.png",
  "needs_printed_number": false
}
```

`image_small` is a remote URL — the only thing in the kit that needs the network,
and only to display the official thumbnail.

### `needs_printed_number` — the ceiling, known in advance

**4,787 cards out of 20,512 (23.3 %) cannot be settled by the image alone.**
These are reprints of the same artwork: 884 of them have a neighbour in the index
at 0.99 or above, and for those no photo, however good, will produce a usable
margin.

The proportion depends strongly on the era:

| Series | Cards | Number required |
|---|---:|---:|
| Base | 494 | **78.3 %** |
| E-Card | 529 | 29.3 % |
| Sword & Shield | 3,667 | 26.7 % |
| Sun & Moon | 2,973 | 23.1 % |
| Scarlet & Violet | 3,595 | 16.9 % |
| Diamond & Pearl | 900 | 13.6 % |
| Platinum | 517 | 7.5 % |

The point of the field is that it is available **before** answering. When the
first candidate carries it, a tight margin is not an anomaly but the expected
behaviour: the app can go and get the strip, ask for a shot of the bottom of the
card, or offer the candidate editions — rather than displaying an "uncertain" it
could have predicted.

The label is **measured** (`ml/scripts/label_discriminability.py` replays
synthetic degradations of each card against the whole index), not deduced from a
rule. Two limits: the queries are synthetic, so it is a difficulty indicator and
not a failure prediction; and the label is computed in the space of the current
index, so it is stale if the encoder changes — hence the fingerprint shipped with
it.

---

## 5. Confidence and display

Two margins, computed on the similarity ranking:

- **edition margin** = 1st candidate's score − 2nd candidate's score;
- **name margin** = 1st candidate's score − the score of the first candidate
  bearing a *different name*.

| Condition | Level | Suggested display |
|---|---|---|
| the selection retained the **whole photo** | `uncertain` | ask for a shot framed on the card |
| the number was read off the card (§7) | `edition` | card and set, firm |
| edition margin ≥ `firm_id_margin` | `edition` | card and set, firm |
| name margin ≥ `firm_name_margin` | `name` | "Primeape — edition to confirm" |
| otherwise | `uncertain` | offer the top-5, or invite to steady the shot |

**Do not hard-code the thresholds.** They are in `index.json`, under
`confidence`, and they **depend on the index**: two enrolment strategies do not
have the same margin scale. An app that kept its old values against a regenerated
index would assert wrongly, with no error raised — exactly the problem the
encoder fingerprint avoids for the model (§4).

```swift
let thresholds = meta.confidence          // firm_id_margin, firm_name_margin
```

Values of the current index: **0.045** and **0.06**. They are higher than what
the calibration requires (0.028): it is a policy choice — the product prefers to
stay wrongly silent than to assert wrongly — and a bound established on four
negatives has no safety margin.

**The whole-photo fallback must never produce a firm verdict**
(`no_firm_verdict_on_full_photo`). When no quadrilateral is retained, the vector
describes a scene and not a rectified card: the margin there compares two wrong
answers against each other. Measured over 46 photos, this fallback produced **no
correct identification** and exactly one firm false positive.

**What these thresholds are worth, and what they are not.** On the 46-photo
bench: 29 firm assertions, of which 28 correct, and the 7 photos with no right
answer (card back, cards from another game, Korean card, pouch, illegible blur)
all refused. But the system **cannot say "this is not a Pokémon card"**: it
answers "uncertain", which invites the user to retake a photo that will never
work. Plan an exit message after two or three consecutive refusals.

Why two levels: the confusion lives almost entirely *within the same card name* —
several printings of the same artwork in different sets. A tight margin often
means "right card, uncertain edition", which is still useful to the user, and not
"I do not know".

---

## 6. Verify the integration

The `test/` folder allows validation in stages rather than debugging the whole
chain. `test/expected.json` contains the expected values.

**Step 1 — preprocessing alone.** Encode `test/reference_card.jpg` (an official
scan present in the index) and compare to the vector given in `expected.json`. A
cosine `≥ 0.98` validates the geometry; below it, the preprocessing is wrong and
nothing else will work. The search must return `me5-36` with a score ≈ 0.995.

**Step 2 — the full chain.** Three real photos, chosen to exercise different
paths:

| File | What it tests |
|---|---|
| `IMG_5033.jpeg` | nominal case |
| `IMG_5028.jpeg` | the embedding alone gets it wrong — it is the bottom-strip OCR that gives the right answer |
| `IMG_5021.jpeg` | backlight, hand-held card, illegible strip |

The fields under `expected` must be reproduced. Those under `indicative`
(retained crop, margin values) can vary slightly depending on the implementation
without it being a problem.

---

## 7. Known limits

- **Overlapping cards**: on a photo of 5 cards where 4 overlap, only one was
  isolated (correctly identified). Apple's detectors need to see 4 closed edges.
  For a collection scan, ask for the cards to be spaced out — or train a
  dedicated detector.
- **Cards absent from the index**: **57 cards** have no image in any public
  source — 8 "MEE" energies, 48 McDonald's cards, one HGSS promo. Without an
  image, no embedding; the safeguard of §7 flags them instead of attributing them
  wrongly. The "MEP" promos, long in this case, entered the index in kit-v4.
  ⚠️ The fallback CDN used to retrieve them (`images.scrydex.com`) **never**
  returns 404: it serves a placeholder over HTTP 200 for any unknown identifier.
  Adding them on the strength of the status code would have injected 57 identical
  card backs, universal attractors in the embedding space. Availability is
  decided on the content fingerprint.
- **Old cards untested**: all the photos are of recent cards, whereas **47 % of
  the index predates Sun & Moon**. The layouts of the Base, Neo or EX series
  differ markedly — borders, artwork frame, strip position. Being in the index
  proves nothing about recognition: coverage is a count, not a measurement.
- **Reprints**: 92 % of cards share their name with another (Pikachu appears 99
  times). It is the reason for the two confidence levels of §5.
- **Language**: the index is in English, but French cards are recognized (the
  artwork dominates the text by a wide margin). Untested on Japanese.

---

## 8. Performance

All the measurements below were made **on a Mac M3 Pro**, with the `.mlpackage`
of this kit. They are reproduced in `benchmarks.json` for comparison.

### On an iPhone 13 Pro

Swift port, 1080×1920 photos from the video stream, per call:

| Step | iPhone 13 Pro | Mac M3 Pro |
|---|---|---|
| detection (both detectors) | 24 ms | 26 ms |
| rectification | 6.4 ms | 2.9 ms |
| orientation OCR | 11 ms | 9.5 ms |
| geometric preprocessing | 1.2 ms | 1.0 ms |
| **embedding (Neural Engine)** | **5.5 ms** | **4.1 ms** |
| search over 20,512 vectors | 1.9 ms | 0.6 ms |
| bottom-strip OCR (`.accurate`) | 65 ms | 60 ms |

The model therefore comfortably keeps the promise of §9 on real hardware.

These costs are **per call** and have not moved: it is the number of calls that
changed (see below). The bottom-strip OCR is now paid only one time in five, the
fast pass handling the rest at 12 ms.

⚠️ **Compile the port with `-O`, even in Debug.** The rest of the app can stay
unoptimised, not this code. The two pixel-by-pixel loops — the variance filter
and the 180° rotation — cost **124 ms and 114 ms per crop** in `-Onone` against
under a millisecond optimised. Over eight crops, that is 1.8 s of a 2.3 s scan,
entirely attributable to the build configuration. Anything going through Vision,
Core ML or Accelerate is insensitive: those are precompiled frameworks.

⚠️ **The number of crops is the real cost item, not the model.** Without the area
filter of §2, a 12-megapixel photo produces up to **15 hypotheses** (8
deduplicated quadrilaterals, several of them with undecided orientation which
count double), and each one pays rectification, orientation and embedding. With
the filter: **2.4 on average, 7 at worst**. Instrument per call and not
cumulatively, otherwise the figures mean nothing.

### The real profile of a photo

Measured over the 32 bench photos, Core ML encoder, Mac M3 Pro, 12-megapixel
photos — this is the profile of a **whole photo**, not of a single-crop
identification. Reproducible with `ml/scripts/profile_pipeline.py`.

| Step | ms per photo | calls per photo |
|---|---|---|
| embedding | 33.1 | 1.0 (per batch of variants) |
| photo decoding | 32.0 | 1.0 |
| orientation (incl. OCR) | 27.2 | 1.9 |
| bottom-strip OCR | 26.3 | 1.25 |
| rectangle detection | 17.0 | 1.0 |
| document detection | 13.3 | 1.0 |
| search | 1.7 | 1.0 |
| rectification | 1.4 | 2.0 |
| **total** | **159 ms** | |

Three things to take from it:

- **The starting point was 287 ms.** The area filter removed 37 % of it, the
  bottom-strip OCR cascade 8 % more, without losing a single identification.
- **The "79 % OCR" split of the previous kit was true for a single-crop
  identification, not for a photo.** On a real photo, the two OCRs weigh 34 % —
  and the apparent first item, the embedding, is only so because it was paid ten
  times. The two have the same cause: the number of crops.
- **The photo decoding (32 ms) does not exist in the app.** It is a bench
  artefact, which starts from a JPEG on disk; a video-stream image arrives
  already decoded. To subtract before comparing your measurements to ours.

⚠️ **Do not shrink the input photo to go faster.** Tested: at 6 megapixels the
decoding drops from 38 to 11 ms but the bench loses one card, at 3 megapixels it
loses three. Detecting on a shrunk version stays fine (§9) — what must not be
shrunk is the image the crop is extracted from.

⚠️ **Coordinate trap.** If the detection reads the file and the rectification
works on a separately decoded array, the two must have exactly the same
resolution. Shifting the two by a factor of 2, the bench dropped from 18/18 to
1/18 — **with no error raised at all**, the quadrilaterals simply being applied
in the wrong frame.

### On a Mac M3 Pro

### Mandatory setting: force the Neural Engine

```swift
let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine
let encoder = try CardEncoder(configuration: config)
```

Letting Core ML choose on its own (`.all`) costs almost double:

| Compute unit | Latency | Throughput |
|---|---|---|
| `.cpuAndNeuralEngine` | **3.3 ms** | 306 img/s |
| `.all` | 5.7 ms | 175 img/s |
| `.cpuAndGPU` | 8.7 ms | 115 img/s |
| `.cpuOnly` | 22.7 ms | 44 img/s |

### Cost of each step

| Step | Cost |
|---|---|
| document detection — 720p | 12.7 ms |
| document detection — 1080p | 24.1 ms |
| document detection — 12 Mpx | 113.5 ms |
| rectangle detection — 720p | 17.9 ms |
| rectification (homography) | 0.5 ms |
| geometric preprocessing | 1.8 ms |
| **embedding (Neural Engine)** | **3.3 ms** |
| search over 20,512 vectors | 0.5 ms |
| orientation OCR | 16.1 ms |
| bottom-strip OCR — `.accurate` | 59.6 ms |
| bottom-strip OCR — `.fast`, enlarged ×2 | 12.0 ms |
| bottom-strip OCR — cascade, on average | 24.0 ms |

### End to end, from a 720p image

| | Cost |
|---|---|
| full identification, bottom-strip OCR included | **~60 ms** (96 ms before the cascade) |
| identification without bottom-strip OCR | **36.5 ms** |

The total with the strip is deduced from the table above, the cascade costing
24 ms on average instead of 59.6. The end-to-end profile measured on real photos
rather than a single-crop 720p image is higher up in this §8.

The split remains the striking fact: **the model weighs 5 % of the total, the two
OCRs weigh two thirds of it.** It is what dictates the architecture of §9.

---

## 9. Live video stream

The model is more than suitable: **3.3 ms per frame on the Neural Engine**, i.e.
306 frames/s. It represents 5 % of the cost of an identification. The dominant
item stays Vision's OCR (two thirds).

> ⚠️ **Do not move to a lighter model.** MobileCLIP2-S0 would save about 2 ms out
> of 60 and would cost precision. Optimising the embedding means optimising what
> does not limit. What limits is the number of crops (§2) and the OCR frequency.

### Splitting the work

The rule: **what is costly must run only once per card, never per frame.**

| Frequency | Steps | Cost |
|---|---|---|
| every frame | quadrilateral detection + tracking | 12.7 ms |
| when a card appears | rectification, orientation, embedding, search | ~36 ms |
| once, in the background | bottom-strip OCR (confirms the edition) | ~24 ms |
| card already identified and tracked | nothing | 0 ms |

At 30 frames/s (33 ms per frame), only the detection runs continuously: it leaves
20 ms for rendering and tracking. The identification of a new card goes onto a
background queue — the preview must never wait for it.

### What makes the difference

- **Detect at 720p, not at photo resolution.** 12.7 ms against 113 ms at 12
  megapixels, i.e. a factor of 9 for an equivalent result: a card occupies enough
  pixels at 720p. The crop for the embedding, meanwhile, can be extracted from
  the full-resolution buffer if available.
- **Track the cards between frames** (IoU on the boxes, or `VNTrackObjectRequest`)
  to re-identify only the new ones. In steady state, the cost falls back to the
  detection alone.
- **Use only document segmentation per frame** (12.7 ms) and reserve the
  rectangle detector (17.9 ms) for identification.
- **Stabilize by vote**: rather than displaying the result of the first frame,
  accumulate 3 to 5 identifications of one tracked card and keep the majority
  one. Free, since the card stays in view.

### Limit to know

`VNDetectDocumentSegmentationRequest` returns **only one object**. For several
simultaneous cards, you have to go through `VNDetectRectanglesRequest` — more
costly and less reliable on a busy background (see §7). A "one card at a time"
stream is the nominal case; scanning a full display remains the hard case.

---

## 10. What the move onto iPhone taught

Three things that only a real port brought out. They are not in
`reference-python/`, which otherwise remains the behaviour reference.

### Promotion by the strip is not enough — you need the number search

Step 7 reorders the top-k candidates. It is therefore powerless in one precise
case: **the number is read cleanly, the card exists, and the embedding never
brought it up**. Measured on a photo of Inkay (`me5-51`), whose strip gave
`051/084` three times in a row, while the ranking was led by an unrelated Trainer
sitting **0.089 above**.

The fix: if no candidate bears the read number and **a single card in the whole
index** bears it, go and get it directly. A unique pair among 20,512 identifies
the card without discussion, and it is a stronger proof than any similarity
score. Restricting to the unique case is not caution about the reading: a number
shared by several printings does not say which one it is.

It is the positive counterpart of the "out of index" safeguard of §2 — the same
table, used to find rather than to disqualify.

**Since then, the search window has gone from 10 to 20 candidates** (§6), which
settles the benign variant of the same problem: a card fallen just below the bar.
A Hariyama came out at rank 12 with its number perfectly legible. Widening the
window makes this recovery less often needed, but does not replace it: the Inkay
case, led by 0.089 by an unrelated candidate, is not a matter of two or three
ranks.

⚠️ **This recovery only covers 38 % of cards.** 63 % of number/total pairs are
unique, but they concern only 7,833 cards out of 20,512 — the others share their
pair with up to 9 cards, and a shared pair designates nothing. For those 62 %, a
card the visual search misses stays missed. It is the limit to keep in mind
before considering the problem settled: the fix is useful, it is not a net.

### The resampling filter, on the Core Graphics side

PIL has no exact equivalent. The four filters, measured on
`test/reference_card.jpg` — which is in the index, so its own similarity is the
score to beat (Python gets 0.9951):

| filter | similarity |
|---|---|
| `.medium` | **0.9938** |
| `.high` | 0.9926 |
| `.low` | 0.9854 |
| `.none` | 0.9656 |

`.medium` wins, and it is also the one that is bilinear rather than bicubic — the
distinction §3 asks to respect.

### The iOS Simulator is worthless for judging accuracy

Core ML behaves identically there (self-test at 0.9944 against 0.9938 on the
Mac), **but not Vision**: the document segmenter frames differently and the
recognizer reads less. On the three fixtures of §6, only one reproduces exactly,
one loses its strip number and one gives a wrong card. The Simulator serves to
check that the module loads and the chain runs; accuracy is judged on device.

---

## 11. Updating the index

A new set requires **no retraining**: just regenerate `index.bin` and
`cards.json` and replace them. The model does not change. This regeneration is
done on the ML project side, not in the app.

Plan for the index to be downloadable rather than bundled, to ship a new set
without going through the App Store.
