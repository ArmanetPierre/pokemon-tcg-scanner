# Model card — on-device Pokémon TCG card recognition

*Version of 12 August 2026. All figures are reproducible with the scripts cited;
none is carried over from an earlier run without having been redone.*

---

## What it is

A **card identification** system: from a photo, it returns which card it is, from
which set, and how reliable the answer is. Entirely on device, with no network
call.

It is **not** a classifier. The architecture is *detect → rectify → encode →
search → read the number*, and the "knowledge" lives in a vector index, not in
weights. A new set is handled by regenerating the index; the model never changes.

| | |
|---|---|
| Encoder | MobileCLIP2-S2, **zero-shot** (never retrained or fine-tuned) |
| Output | 512-dimension vector, L2-normalised |
| On-device format | Core ML fp16, 69 MB, **5.5 ms** on the Neural Engine of an iPhone 13 Pro |
| Index | 20,512 cards, 176 sets, float16, 21 MB |
| Search | exact matrix product (no FAISS, no approximate index) |
| Unlearned stages | detection and OCR (Vision), orientation, reading the number |

**Most of the performance does not come from the model.** Ablation on the
46-photo bench:

| Configuration | Top-1 |
|---|---|
| whole photo, no detection or orientation | 5/39 (13 %, CI95 6-27) |
| + detection, filtering, orientation → similarity alone | 29/39 (74 %, CI95 59-85) |
| + reading the printed number → full chain | **35/39 (90 %, CI95 76-96)** |

The framing is worth 24 identifications, the bottom-strip OCR 6. The model
represents about **5 % of the compute time** of an identification; Vision's two
OCR passes represent two thirds of it.

---

## Intended use

**Intended.** To help a person identify a card they hold, in an application where
they see the result and can correct it. The system is designed to **stay silent**
when it does not know, and this property matters more than its raw rate.

**Out of scope, and discouraged.**

- **Any money decision** — valuation, purchase, resale, insurance,
  authentication. The system does not tell an authentic card from a counterfeit,
  nor one condition from another, and it sometimes firmly attributes a wrong
  printing (see *Known failures*).
- **Automatic inventory without human review.** At 90 % precision, a collection
  of 1,000 cards contains about a hundred errors, some of them asserted
  confidently.
- **Japanese or Korean cards.** The index is English. French cards are recognized
  — the artwork dominates the text by a wide margin — but the Japanese and Korean
  lines have their own numbering and are not covered.
- **Counterfeit detection, condition grading, price estimation.** None of these
  tasks is measured, and nothing in the design prepares for them.

---

## Data

### The index

20,512 cards, 176 sets, from Base Set (1999) to Pitch Black (2026).

| Era | Cards | Share |
|---|---:|---:|
| Base / WotC (1999-2003) | 1,780 | 8.7 % |
| EX (2003-2007) | 1,722 | 8.4 % |
| Diamond & Pearl / Platinum | 1,417 | 6.9 % |
| HGSS / Call of Legends | 544 | 2.7 % |
| Black & White | 1,416 | 6.9 % |
| XY | 1,775 | 8.7 % |
| Sun & Moon | 2,955 | 14.5 % |
| Sword & Shield | 3,529 | 17.3 % |
| Scarlet & Violet | 3,249 | 15.9 % |
| Mega Evolution | 1,039 | 5.1 % |
| promos, decks, specials | 1,028 | 5.0 % |

**Provenance.** Metadata from `PokemonTCG/pokemon-tcg-data`, pinned to revision
`8b4e3879` (2026-07-17). High-resolution images from the official CDN, except for
49 cards (McDonald's collections 2014-2018 and one HGSS promo) retrieved by hand,
no public source serving them. The encoder's fingerprint is recorded in
`index.json`: an index and a model out of tune produce no error, only confident
wrong answers.

**A collection trap, resolved.** The fallback CDN never returns 404: it answers
any unknown identifier with a generic image over HTTP 200. Trusting the status
code would have added 57 identical card backs to the index, which would have
behaved as universal attractors. Availability is therefore decided on a content
fingerprint.

### The index is not of uniform quality

Measured directly (`scripts/audit/audit_index.py`):

- **4.3 % of cards have a neighbour in the index at ≥ 0.99** — visually
  indistinguishable, whatever the quality of the photo.
- The **median intrinsic margin** between a card and its nearest neighbour is
  **0.0099**, below the decision threshold of 0.03. For **77.6 %** of the index,
  similarity structurally cannot produce a firm verdict.
- 26 cards behave as **attractors**, appearing more than 100 times in the top-10
  of other cards (up to 284 for `sm3-10`), against a median of 6.

As a result, each card carries in `cards.json` a `needs_printed_number` field,
measured and not deduced: **4,787 cards (23.3 %)** cannot be settled by the image
alone. Very uneven by era: 78.3 % for the Base series, 7.5 % for Platinum.

---

## Evaluation

### Real-photo bench — the referee

46 photos, of which **39 with a right answer** and **7 without** (card back, One
Piece cards, Korean card, card-holder pouch, illegible blur). Three collections,
several people, ordinary conditions: backlight, sleeve, busy background, tilted
cards, in a binder, one batch entirely in landscape.

| | Calibration | Test | Total |
|---|---|---|---|
| photos with an answer | 21 | 18 | 39 |
| top-1 | 21/21 (100 %, CI95 85-100) | **14/18 (78 %, CI95 55-91)** | **35/39 (90 %, CI95 76-96)** |
| top-5 | 21/21 | 16/18 | 37/39 |
| firm verdicts, and correct | | | **29/39, of which 29/29** |
| correct refusals | 3/4 | 3/3 | **6/7** |

**The protocol, and what it does not guarantee.** The confidence thresholds were
fixed before the 2nd batch arrived and have not moved: for them, that batch is
hold-out. The detection filters, on the other hand, were tuned while seeing it;
the 3rd batch gives them back this hold-out. The detail is recorded in
`truth.json`.

**39 photos cannot demonstrate better than their lower bound.** Even a flawless
run would top out at a 91 % lower bound. Any figure presented without its
interval, on this bench, is an over-claim.

### Synthetic bench — to see by era

1,209 queries on held-out cards, obtained by degrading the reference scan
(`src/augment.py`). A **relative** measurement: it compares eras and spaces
against each other, it does not predict the accuracy on real photos.

| Era | top-1 |
|---|---:|
| WotC 1990s | **75.6 %** |
| WotC 2000s | 84.7 % |
| EX | 94.2 % |
| Diamond & Pearl / Platinum | **95.8 %** |
| Sword & Shield | 90.0 % |
| Scarlet & Violet | 95.0 % |
| **Total** | **91.1 %** |

**20 points of spread between the best and worst era.** The real-photo bench
could not see it: it contained no card older than Sun & Moon before the 3rd
batch, whereas 47 % of the index is that old.

### Confidence

Two levels, on margins computed **before** any promotion by the OCR:

| Condition | Level |
|---|---|
| the printed number was read and designates a candidate | `edition` |
| 1st/2nd margin ≥ 0.03 | `edition` |
| margin to the first candidate of a different name ≥ 0.04 | `name` |
| otherwise | `uncertain` |

**The absolute score must never be thresholded or shown.** Two unrelated cards
already resemble each other at 0.697 on median; the useful range is not 0-1. It
is a property of the model, not of the cards: the norm of the index's mean vector
is **0.8146**, so most of the similarity is a shared constant.

**These thresholds are insufficient, and that is measured.** On the widened
bench, precision at 25 % coverage (80 %) is **lower** than at 100 % (89.7 %):
sorting by margin is worse than not sorting. AURC = 0.101.

---

## Known failures

**One firm-and-wrong attribution** over 46 photos — the worst possible outcome
for a user, who has no way of knowing the answer is invented:

- **A motion blur no human can identify**, announced firmly.

The third batch had produced **three**. The other two no longer happen, and none
of the 39 photos with a right answer is asserted wrongly today: **29 firm
verdicts, 29 correct**. They are still noted here because they say where the
chain gives out:

1. Card small in a cluttered shelf: no quadrilateral isolates it, the
   **whole-photo fallback** won and asserted an unrelated card, the right answer
   not even in the top-5.
2. 2006 card (Crystal Guardians): rank 2, and another card asserted. It was the
   old-era drop-off, confirmed on a real photo.

**The threshold is not fixed by recalibration.** For the blur to stop being
asserted, `FIRM_ID_MARGIN` must go from 0.03 to 0.0558 — which drops firm
verdicts from 21/21 to 6/21. Blocking one false positive costs **71 % of the
coverage**.

Other limits:

- **Overlapping cards.** Apple's detectors need four closed edges. On a photo of
  five cards where four overlapped, only one was isolated.
- **Cards outside the index.** When a number reads cleanly and belongs to no
  known card, the system answers "unknown card" rather than attributing it to
  whatever ranked first.
- **The embedding can miss entirely.** On one card, the right answer was not in
  the top-10, 0.089 behind an unrelated Trainer.
- **The iOS Simulator is worthless for judging accuracy.** Core ML behaves
  identically there, Vision does not.

---

## Negative results

They are published because they close doors, and a door closed with a
measurement is worth more than a lead reopened every six months.

- **Correcting the anisotropy without supervision degrades.** Centring (`e − µ`,
  renormalised) multiplies the median intrinsic margin by 3.55 and crushes the
  hubness (max hub 284 → 77), but drops the top-1 from 26/31 to 23/31 — measured
  on the similarity-alone arm of the 32-photo bench, before its widening. CSLS
  does worse still, at 22/31.
- **A learned projection wins, but is not shippable.** Trained contrastively on
  pairs (degraded scan → scan), it takes the synthetic bench from 91.1 % to
  95.2 % on never-seen cards, and markedly improves the sorting (AURC 0.047 →
  0.016). But it degrades the refusal, and that threshold cannot be calibrated
  honestly for lack of negatives in quantity. It also revealed that **four
  absolute quantities of the pipeline** do not survive a change of space, one of
  them written nowhere: `selection_score` adds a score and a margin, which
  assumes they live on the same scale.
- **A sharpness criterion does not screen out non-cards.** Laplacian variance on
  the retained crop: negatives from 70 to 1,082, real cards from 44 to 6,106,
  distributions entirely overlapping.
- **The synthetic bench cannot compare two enrolment strategies.** It gives
  99.9 % to a centroid index against 91.1 % to the reference index, a gain that
  does not carry over to real photos: enrolment and query come out of the same
  degradation model.

---

## Shipped configuration, and measured alternatives

What is on device: **reference index** (one vector per card, the scan) and
thresholds 0.03 / 0.04. Two alternatives are measured and **not shipped**, their
artefacts staying outside the repo:

| | shipped | multi-view centroid | learned projection |
|---|---|---|---|
| top-1 (39 photos) | **35/39** | 34/39 | not measured on this bench |
| correct refusals | 6/7 | **7/7** | degraded |
| AURC | 0.101 | **0.077** | 0.016 (32-photo bench) |
| cost of refusal with no false positive | 71 % of coverage | **24 %** | not calibratable |
| operating cost | — | **nil** | nil |

⚠️ **Read this table in columns.** The `README` announced "34/39 and 7/7" as the
current state for a while: those are the figures of the **multi-view centroid**,
which is not on device. The shipped one does 35/39 and 6/7. The left column is
the only one that describes what the application does.

The centroid loses one identification and fixes the confidence. The trade-off is
a product choice, not a technical one: one fewer identification against one fewer
confident wrong attribution.

---

## Reproduce

```bash
cd ml
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/pip install -e . --no-deps

.venv/bin/python scripts/evaluate_real.py --ablation   # the referee
.venv/bin/python scripts/evaluate_synthetic.py         # by era, over the whole index
.venv/bin/python scripts/audit/audit_index.py          # index structure
.venv/bin/python -m pytest -q                          # 43 invariants
```

The bench photos are not in the repo. The full analysis, with the remaining work,
is in [`audit-ml.md`](audit-ml.md).
