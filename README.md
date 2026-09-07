# Pokémon TCG Scanner

Recognition of Pokémon cards from a photo, **entirely on device**, meant for an
iOS app.

> [`OVERVIEW.md`](OVERVIEW.md) — architecture, measurements, traps and limits, in
> a single document.
>
> What the system is worth, where it fails, and what not to ask of it:
> [`docs/model-card.md`](docs/model-card.md).

**Current state: 35 correct identifications out of 39 real iPhone photos**
(89.7 %, 95 % Wilson CI: 76-96 %), and **no firm-and-wrong assertion** among the
29 firm verdicts. The configuration is tuned against false positives: better to
ask for another photo than confidently announce a wrong card. The bench reported
31/31 until a third batch — old cards, botched framing, and seven photos with no
possible right answer — brought it back to this value. That batch first produced
**three firm-and-wrong attributions**; only one remains, on a motion blur nobody
can identify. The other two, the cluttered shelf and the 2006 card, no longer
happen. The full story is in `docs/audit-ml.md` §0. The photos come from three
collections photographed by several people (French cards, English index,
ordinary conditions — backlight, sleeve, busy background, tilted cards, one batch
entirely in landscape, and cards in a binder). Seven other photos have **no
possible right answer**: card back, One Piece cards, Korean card, card-holder
pouch, illegible blur. The chain correctly stays silent on **six of the seven**:
the blur is still asserted. Measured on a Mac.

**What each stage contributes**, measured by ablation on this same bench — it is
the chain that identifies, not the model alone:

| Configuration | Top-1 |
|---|---|
| whole photo, no detection or orientation | 5/39 (12.8 %, CI95 6-27) |
| + detection, filtering, orientation → **similarity alone** | 29/39 (74.4 %, CI95 59-85) |
| + reading the printed number → **full chain** | 35/39 (89.7 %, CI95 76-96) |

The framing is worth 24 identifications, the embedding only works on what it is
given, and reading the printed number recovers 6 more. Reproducible with
`scripts/evaluate_real.py --ablation`.

Looking for a better encoder means optimizing the stage that weighs the least:
similarity alone tops out at 72 %, and the OCR of the bottom strip recovers 15
points. Of the chain's 29 firm assertions, **25 come from the printed number
read**.

**It runs on iPhone.** The Swift port is integrated into an Expo app,
[hugo-heer/poke-scanner](https://github.com/hugo-heer/poke-scanner), as a native
module `expo-card-encoder`, alongside the existing OCR + TCGdex path. On an
iPhone 13 Pro, the encoder takes **5.5 ms** on the Neural Engine. What the move
onto device taught is in [`integration-kit/AGENTS.md`](integration-kit/AGENTS.md)
§10.

**Get the model and the index** without rebuilding the ML chain — 94 MB,
published as a release:

```bash
gh release download kit-v5 --repo ArmanetPierre/pokemon-tcg-scanner
tar -xzf card-encoder-kit.tar.gz -C integration-kit/
```

## Principle

No 20,000-class classifier. The architecture is
**detection + embedding + similarity search**: a new set is handled by
regenerating the index, without retraining.

```
photo → quadrilateral detection (Vision) → rectification (homography)
     → orientation from text position → embedding (MobileCLIP2-S2, 512 dims)
     → cosine search over 20,512 cards → reading the printed number
     → card + confidence level
```

## Layout

| Folder | Content |
|---|---|
| `ml/src/` | pipeline, detection, orientation, bottom-strip reading, search |
| `ml/scripts/` | dataset, embeddings, evaluation, Core ML and index exports |
| `integration-kit/` | iOS integration doc (`AGENTS.md`) and reference measurements |
| `docs/` | [model card](docs/model-card.md), [ML audit](docs/audit-ml.md), test report and improvement plan |
| `plan-ios.md` | original plan and measured results |

## What is not versioned

Everything heavy is regenerable and deliberately absent from the repo: the 5.5 GB
of reference images, the embedding bank, the Core ML model, the binary index and
the test photos.

```bash
cd ml
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt   # exact versions of the measurements
.venv/bin/pip install -e . --no-deps

.venv/bin/python scripts/build_metadata.py      # metadata (pokemon-tcg-data)
.venv/bin/python scripts/add_missing_cards.py   # sets this source ignores (TCGdex)
.venv/bin/python scripts/download_images.py     # high-resolution images (~15 min)
.venv/bin/python scripts/build_embeddings.py    # vector index (~6 min on M3)
```

Then, to rebuild the integration kit:

```bash
.venv/bin/python scripts/export_coreml.py       # CardEncoder.mlpackage + parity
.venv/bin/python scripts/export_index.py        # index.bin + cards.json
```

`scripts/refresh_index.py --check` reports the sets released since the last build;
`scripts/add_missing_cards.py --check` says which are actually addable and which
have no image anywhere.

**The index is rebuildable identically.** The metadata source is pinned to a
precise revision — reading `master` would make it a moving target, and two
rebuilds a month apart would give two different indexes with nothing to signal
it. `scripts/build_metadata.py --check` compares the pin to upstream, `--ref
<sha>` moves it deliberately. The package versions are frozen in
`requirements.lock.txt`.

**Index: 20,512 cards, 176 sets.** No card in the metadata is without an image
any longer. **No card in the index is without an image any longer.**

The last 49 — the McDonald's collections 2014/2015/2017/2018 and one HGSS promo —
were retrieved by hand and integrated by
[`scripts/import_local_images.py`](ml/scripts/import_local_images.py), no public
source serving them.

## Verify

```bash
.venv/bin/python scripts/evaluate_real.py            # test bench, PyTorch
.venv/bin/python scripts/evaluate_real.py --ablation # what each stage contributes
.venv/bin/python scripts/evaluate_real.py --coreml   # with the embedded model
.venv/bin/python scripts/scan.py photo.jpeg          # identify a photo
```

The bench relies on `ml/data/eval/truth.json`, the ground truth established by
reading the name, number and set code printed on each card. The matching photos
are outside the repo.

**Protocol.** The 46 photos are split into two splits, recorded in `truth.json`
with the detail of what each one is — and is not — a hold-out of: `calibration`
(25 photos, of which 4 with no right answer) and `test` (21 photos, of which 3
with no right answer). The confidence thresholds were fixed before the 2nd batch
arrived and have not moved; the detection filters, on the other hand, were tuned
while seeing it, and the 3rd batch makes this hold-out lost to them. The bench
reports the two splits separately, with their intervals.

The negatives are mostly in calibration, and that is deliberate: until the 3rd
batch, **no** calibration photo was without an answer, so the refusal behaviour
was calibrated on nothing. `--calibrate` takes this into account — a threshold
must reach 100 % precision on the positives *and* rank above what the negatives
get.

Every rate comes out with its Wilson interval, and confidence is reported by a
**risk/coverage curve** rather than by a threshold: a threshold is comparable
neither between two models nor between two metric spaces, a curve always is.

**The synthetic bench**, for its part, covers the whole index — 31 photos of
recent cards cannot see an era-by-era drop-off, whereas 47 % of the index
predates Sun & Moon:

```bash
.venv/bin/python scripts/evaluate_synthetic.py            # 150 cards per era
```

The query is a degraded reference scan ([`src/augment.py`](ml/src/augment.py):
residual perspective, downsampling, blur, exposure, holo glare, noise, JPEG), so
the ground truth is free and exact. It is a **relative** measurement — it
compares eras, sets and model variants against each other; it does not predict
the accuracy on real photos, since it does not reproduce the sensor's optics or
the detection's errors. The referee stays `evaluate_real.py`.

## Three things to know before touching the code

Each one cost time, and they are detailed in
[`integration-kit/AGENTS.md`](integration-kit/AGENTS.md) §3:

- **The absolute score discriminates nothing.** Two unrelated cards already
  resemble each other at ~0.79. It is the gap between the 1st and the 2nd
  candidate that carries the information.
- **The embedding must never arbitrate the orientation.** An upside-down card can
  score higher on a wrong card than the upright crop on the right one.
  Orientation is decided from the text position.
- **The preprocessing geometry must be identical to the pixel** between the index
  and the queries: short side to 256 in bilinear, then centre crop. A direct
  squash to 256×256 drops the parity to 0.65.

## To integrate into an iOS app

Everything is in [`integration-kit/AGENTS.md`](integration-kit/AGENTS.md):
step-by-step pipeline with the matching native API, traps, confidence thresholds,
file formats, validation fixtures, and time budget for a video stream (§9 — the
model is only 5 % of the cost of an identification; Vision's two OCRs weigh two
thirds of it). §8 gives the profile of a whole photo, where what really decides
the processing time is the number of candidate crops: **287 ms → 159 ms per
photo** by discarding the too-small quadrilaterals before any processing, and by
reading the bottom strip in two passes.

A port already exists and serves as a reference:
[`modules/expo-card-encoder`](https://github.com/hugo-heer/poke-scanner/tree/main/modules/expo-card-encoder)
in the poke-scanner app — Swift, with its own README.
