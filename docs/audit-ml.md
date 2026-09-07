# ML audit — critical report and expert-review plan

*External review of 11 August 2026. All the measurements cited were redone on the
repo, with the index and photos present locally; the commands are in the
appendix.*

---

## Verdict on one page

The project is **above the level of a portfolio project**, and it is so for a
precise reason: every constant in the code is justified by a measurement, and the
limits are written before the results. `MIN_QUAD_AREA`, `FALLBACK_SCORE`,
`SHAPE_BUCKET`, the orientation dead zone — none is a magic number. That is rare,
and it is the most reliable marker of serious work.

What stops it from being an **expert ML** project today is that the engineering
work has been done, and not the scientific work:

| | |
|---|---|
| **What is at expert level** | justified retrieval architecture, PyTorch↔Core ML parity tested, per-stage profiling, explicit refusal (`uncertain` / `out_of_index`), measured device port, documentation of the traps |
| **What is not yet** | evaluation protocol (n=31, no CI, no calibration/test separation), no quantified ablation, uncalibrated confidence, embedding-space geometry never studied, no domain adaptation, zero automated tests |

Three figures sum up the diagnosis, all measured in this audit:

1. **Similarity alone does 26/31, not 31/31.** The bottom-strip OCR carries 5
   identifications out of 31, i.e. **16 % of the announced result**.
2. **90.7 % top-1 over 1,650 synthetic queries** covering the whole index, with
   **79.3 % on the WotC era against 98.7 % on Diamond & Pearl** — a gap of 19
   points the current bench cannot see.
3. **The median intrinsic margin between a card and its nearest neighbour in the
   index is 0.0099**, below the decision threshold of 0.03. For half the index,
   similarity structurally cannot produce a firm verdict, even on a perfect scan.

---

## 0. What a third batch of photos changed (12 August 2026)

**The 31/31 was a property of the tested population, not of the system.**

Fourteen photos were added: old cards, botched framing, and above all **seven
negatives** — three One Piece cards, a Korean card, an illegible blur, a
card-holder pouch, a card back. The bench goes from 32 to 46 photos. The result:

| | before (32 photos) | after (46 photos) |
|---|---:|---:|
| top-1 | 31/31 (100 %, CI95 89-100) | **35/39 (89.7 %, CI95 76-96)** |
| top-1 on the test split alone | 10/10 | **14/18 (77.8 %)** |
| correct firm verdicts | 30/30 | **33/35** |
| correct refusals | 1/1 | **6/7** |
| AURC | 0.000 | **0.101** |

**Three firm false positives**, where there were none:

- `IMG_5121` — the card is small in a cluttered shelf, no quadrilateral isolates
  it, the **whole-photo fallback** wins and asserts `sv1-250` with a margin of
  0.0676. The right answer is not even in the top-5. The same card, better framed
  (`IMG_5123`), comes out at top-1.
- `IMG_5127` — Dynavolt, Crystal Guardians **2006**, the first pre-Sun & Moon
  card of the bench. Rank 2, and `dpp-DP54` asserted firmly. This is exactly the
  drop-off the synthetic bench announced for the old eras, confirmed on a real
  photo.
- `IMG_5124` — a motion blur **no human can identify**, announced firmly.

And precision at 25 % coverage (80 %) is **lower** than at 100 % (89.7 %): on
this population, sorting by margin is worse than not sorting. The confidence
signal is not merely insufficient, it is misleading.

**The cost of the threshold, and a correction.** With the negatives finally
present in calibration, `evaluate_real.py --calibrate` quantifies the problem:
for the blur of `IMG_5124` (margin 0.0557) to stop being asserted,
`FIRM_ID_MARGIN` must go from 0.03 to **0.0558**, which leaves only **6 firm
verdicts out of 21** in calibration.

This sentence was first written here as "blocking one false positive costs 71 %
of the coverage", and that was an over-claim. The threshold only governs verdicts
from **similarity**: on the full bench, **25 of the 29 firm assertions come from
the printed number read**, a path that does not go through it. The real product
cost is therefore far less than the calibration coverage suggests. What stays
true is the ranking of the options — see task D.

**A hypothesis tested and refuted.** If the negatives were simply blurry or flat,
a sharpness criterion would screen them out at no cost. Measured (Laplacian
variance on the retained crop): the negatives range from 70 to 1,082, the real
cards from 44 to 6,106 — distributions entirely overlapping. Sharpness is not the
missing signal. The crop of `IMG_5124` is in fact sharp: it is the wall behind
the card that is in the focal plane.

**What it says about the system.** The refusal rests on a single quantity, the
margin, which measures "are these two candidates close?" and not "am I even
looking at a card?". These are two different questions, and no threshold on the
first answers the second. The underlying task stays **B2**.

### What was delivered in response (12 August 2026)

On a product decision — "let's avoid false positives, we can ask for a
better-framed photo" — three changes, each measured:

1. **The whole-photo fallback can no longer assert anything.** It produces no
   correct identification on the 46 photos, and exactly one firm false positive:
   capping it costs nothing.
2. **The centroid enrolment is on device** (task D). At equal safety, it takes
   the firm verdicts from the margin from 6/21 to 16/21.
3. **Thresholds raised to 0.045 / 0.06**, against 0.028 that the calibration
   gives. The gap is a **deliberate policy choice**, not a calibrated value: a
   bound established on four negatives has no safety margin.

Result: **34/39 top-1, 29 firm assertions of which 28 correct, and 7/7 negatives
refused** — the blur and the pouch included. One wrong assertion remains, the
2006 card, which is a matter of the old-era drop-off and not of framing.

**A lead ruled out, with measurements to back it.** A threshold on the
quadrilateral's area seemed the natural lever for "require better framing". It
does not reach these negatives: the hand-held One Piece cards cover 0.44 to 0.53
of the photo, more than most correctly identified real cards. They are
**well-framed** objects that are not Pokémon cards. The area is still exposed on
`Variant`, because it remains the right signal to guide the user — but it does
not screen out non-cards.

### B2 — A learned confidence rather than a threshold
**Effort: medium. It is now the most urgent task.**

Replace the cascade of `if` with a probability calibrated over several signals —
edition margin, name margin, absolute score, crop standard deviation, sharpness,
quadrilateral area, OCR verdict, and **the fact that the whole-photo fallback was
used**, which on its own explains one of the three false positives. Logistic
regression then isotonic calibration, validated leave-one-out.

*What is missing to do it:* negatives. Seven is enough to reveal the problem, not
to train the solution. It would take **thirty to fifty**, varied: other card
games, rectangular objects (books, cases, phones), cards outside the index,
framing failures.

---

## 1. The evaluation protocol is the weak link

### 1.1 n=31 does not measure what the README implies

"31 correct identifications out of 31" is true and verified — I reproduced it.
But the 95 % Wilson confidence interval for 31/31 is **[89.0 % ; 100 %]**. The
honest sentence is "the real precision is probably above 89 %", which is a very
different statement from "100 %".

To this add three biases that the doc mentions but whose consequences it does not
draw:

- **all the photographed cards are recent**, whereas 47 % of the index predates
  Sun & Moon;
- **a single phone model**, therefore a single ISP pipeline, a single optic;
- **two collections**, therefore two shooting styles.

### 1.2 The confidence thresholds have never seen test data

`FIRM_ID_MARGIN = 0.03` and `FIRM_NAME_MARGIN = 0.04` were calibrated on 18
photos, then "held" on 14 more. The code comment says so plainly ("to reconfirm
on 40+ photos"), which is to the project's credit — but there is no
calibration / test separation, and the 14 confirmation photos were added *after*
seeing that they passed. That is tuning on the full set, and the precision
estimate that follows is optimistic by construction.

### 1.3 What a large-scale evaluation gives

I built a synthetic bench: start from the reference scan, apply the degradations
of a phone photo to it (residual perspective, downsampling, blur, exposure, white
balance, holographic glare, noise, JPEG), then search. The ground truth is free
and exact, so the sample can cover the whole index.

**1,650 queries, 150 per era:**

| Era | n | top-1 | top-5 | median margin |
|---|---:|---:|---:|---:|
| WotC 1990s (Base, Jungle, Fossil) | 150 | **79.3 %** | 99.3 % | **0.0118** |
| WotC 2000s | 150 | 90.0 % | 98.7 % | 0.0500 |
| EX | 150 | 88.0 % | 96.7 % | 0.0546 |
| Diamond & Pearl / Platinum | 150 | **98.7 %** | 99.3 % | 0.0724 |
| HGSS | 150 | 92.0 % | 98.0 % | 0.0554 |
| Black & White | 150 | 94.0 % | 98.7 % | 0.0491 |
| XY | 150 | 90.7 % | 97.3 % | 0.0537 |
| Sun & Moon | 150 | 90.0 % | 98.7 % | 0.0445 |
| Sword & Shield | 150 | 91.3 % | 98.0 % | 0.0598 |
| Scarlet & Violet | 150 | 92.0 % | 98.0 % | 0.0639 |
| promos / others | 150 | 91.3 % | 98.7 % | 0.0595 |
| **TOTAL** | **1,650** | **90.7 %** | **98.3 %** | |

Two lessons that 31 photos cannot give:

- **The WotC era drops off by 19 points.** Originally low-resolution artwork,
  uniform layout, and above all massive reprints across Base, Base Set 2, Jungle
  and Legendary Collection.
- **Its median margin, 0.0118, is below the `FIRM_ID_MARGIN` threshold of
  0.03.** Thresholds calibrated on recent cards will therefore massively produce
  "uncertain" on old cards — not errors, but a refusal to answer, on 8.7 % of the
  index.

This bench **does not replace** real photos: the augmentations reproduce neither
the sensor's optics nor the real detection geometry. It is a *relative*
measurement, comparing eras, sets and model variants against each other over
thousands of cards — where 31 photos only measure noise.

---

## 2. No ablation: we do not know what each stage contributes

This is the most costly gap, because it prevents knowing where to invest.

I isolated the similarity stage alone (same crops, same selection logic, without
the promotion by the read number):

| Configuration | top-1 | top-5 |
|---|---:|---:|
| similarity alone | **26/31** (83.9 %) | 29/31 |
| full chain (with bottom-strip reading) | **31/31** | 31/31 |

**The bottom-strip OCR carries 5 identifications out of 31.** The README
attributes 31/31 to the chain but presents it under the title "card recognition",
which suggests the embedding does the work. It does 84 % of it. The remaining
sixth rests on the most fragile stage of the chain — an OCR on a band of 14 % of
a crop's height, prone to character confusions, and which `OVERVIEW.md` elsewhere
acknowledges can only directly resolve 38 % of the index.

It is a perfectly defensible architecture — combining a visual signal and a
symbolic signal is the right idea — but it must be **presented and measured as
such**, with the ablation table in plain sight.

---

## 3. The confidence is a threshold, not a calibration

The project already has the essentials: it refuses to answer, at two levels, and
it can say "out of index". That is more than most systems in production do. But
the decision remains a comparison to a constant.

What expert work produces instead is **risk/coverage curves**. Measured on the
bench:

| space | top-1 | precision @100 % coverage | @75 % | @50 % | @25 % | AURC |
|---|---:|---:|---:|---:|---:|---:|
| raw | 26/31 | 83.9 % | 87.0 % | 100 % | 100 % | **0.047** |

And above all: **the threshold that guarantees zero error on the bench is 0.0141,
and it retains 65 % of the photos** — not 0.03. The current threshold is
therefore *conservative*, which is the right sign, but it was chosen on 18 points
and its uncertainty is wide in both directions.

What is missing:

- report **AURC** and **coverage at target precision** rather than a threshold;
- a **calibrated probability** (logistic regression then isotonic calibration on
  `[id_margin, name_margin, OCR verdict, crop standard deviation, quad area]`),
  validated leave-one-out, rather than a cascade of `if`;
- **era-conditioned** thresholds, since section 1.3 shows the margin distribution
  changes by a factor of 5 between WotC and DP.

---

## 4. The geometry of the embedding space has never been studied

The project documents as a trap the fact that "two unrelated cards already
resemble each other at ~0.79". That is exact, but presented as a fatality whereas
it is a **named and measurable phenomenon**: the anisotropy of CLIP spaces. The
embeddings do not occupy the sphere, they occupy a narrow cone.

Measured on the index:

```
‖mean vector‖               = 0.8146      (0 = isotropic, 1 = all collinear)
cos(card, mean direction) : median 0.846, p5 0.602, p95 0.901
similarity between two arbitrary cards : median 0.697, p99 0.841
```

A space whose mean has a norm of 0.81 is a space where **83 % of the similarity
signal is a constant shared by all cards**. Hence the "0.79 floor". This is not a
property of Pokémon cards, it is a property of the model.

### What is interesting: the classic corrections fail here

I tested the two standard corrections from the instance-retrieval literature:

| space | top-1 (real bench) | median margin | AURC | zero-error threshold → coverage |
|---|---:|---:|---:|---|
| raw | **26/31** | 0.0241 | **0.047** | 0.0141 → **65 %** |
| centred (`e − µ`, renormalised) | 23/31 | 0.0865 | 0.097 | 0.1318 → 32 % |
| CSLS (hubness correction) | 22/31 | — | 0.190 | 0.2117 → 23 % |

Centring multiplies the median intrinsic margin by 3.55 and crushes the hubness
(the max hub goes from 284 to 77 appearances in the top-10s), but **it degrades
the precision**. It amplifies the noise directions at the same time as the
discriminative directions, and the inflated margins are no longer comparable to
the calibrated threshold.

**This is a negative result, and it is the most useful one of this audit**: it
closes the door to the unsupervised correction, and it points to the only way
left — a **learned** correction. See task C.

*(Methodological trap to remember: a margin is never comparable between two
metric spaces. Comparing 0.03 in the raw space to 0.03 in the centred space makes
no sense. Only risk/coverage curves compare.)*

---

## 5. The structure of the index is not diagnosed

No script looks at the index as an object in itself. Measured:

```
nearest neighbour of each card :  p50 0.894   p90 0.962   p99 0.999
cards with a neighbour ≥ 0.99 :    884  (4.3 %)
cards with a neighbour ≥ 0.98 :  1,334  (6.5 %)
cards with a neighbour ≥ 0.95 :  2,766  (13.5 %)
intrinsic margin 1st/2nd neighbour : median 0.0099 — 77.6 % of cards below 0.03
the nearest neighbour bears the same name : 57.5 % of cases
```

Three direct consequences:

1. **4.3 % of the index is visually indistinguishable.** For these cards, no
   photo, however good, will produce a usable margin. Reading the number is not a
   refinement, it is the only way.
2. **The threshold of 0.03 is above the intrinsic margin of 77.6 % of the
   index.** The `edition` verdict by similarity is therefore structurally rare,
   and the system rests far more on the OCR than the doc shows — which the
   ablation of §2 confirms.
3. **A few cards are attractors.** 26 cards appear more than 100 times in the
   top-10 of other cards, against a median of 6:

   | appearances | card | set |
   |---:|---|---|
   | 284 | `sm3-10` Ledian | Burning Shadows |
   | 239 | `swsh6-52` Thundurus | Chilling Reign |
   | 168 | `swsh8-140` Gligar | Fusion Strike |
   | 158 | `swsh4-105` Sableye | Vivid Voltage |

The most self-confusing sets confirm the WotC drop-off of §1.3: Legendary
Collection (90 % of the cards have a neighbour ≥ 0.97), Base Set 2 (84.6 %),
Jungle (76.6 %), Base (62.7 %) — these are the cross-reprints.

This information should **leave the diagnosis and enter the product**: a field
per card saying "discriminable by the image alone: yes / no".

---

## 6. The model has never been touched

MobileCLIP2-S2 is used *zero-shot*. The `REGISTRY` provides six candidate models,
but no quantified comparison is versioned, and above all:

- **the domain shift is neither measured nor reduced.** The index is made of
  clean publisher scans, flat, in studio lighting; the queries are phone photos.
  This is exactly the problem that 20 years of instance-retrieval literature
  treat with metric learning, and nothing is attempted;
- **a single vector per card.** Multi-view enrolment (encode each reference under
  k augmentations) is the cheapest technique in the field: no retraining, index
  × k, immediate gain on robustness;
- **no projection head.** A 512×512 matrix learned contrastively on pairs
  (augmented reference → reference) costs a few minutes of GPU, exports to Core
  ML at zero inference cost, and it is precisely the supervised correction that
  §4 points to.

This is the border between "intelligent assembly of Apple bricks" — what the
project is today, and it assembles them very well — and "ML work".

---

## 7. Engineering: what is missing from the case

- **Zero automated tests** on ~5,600 lines of Python. The most critical
  invariants are documented as traps (§3 of `AGENTS.md`) but nothing prevents
  their regression: preprocessing geometry, embeddings ↔ `card_ids` alignment,
  non-application of the EXIF, digit-to-digit distance, margins-before-reordering
  order.
- **No CI.** The bench and the Core ML parity are run by hand. A macOS runner
  would run Vision.
- **No artefact traceability.** `index.json` carries the model name but no
  fingerprint: nothing prevents the app from loading an `index.bin` built with
  another encoder, and the symptom would be *plausible and wrong* answers —
  exactly the class of bug the project documents elsewhere with care. A hash of
  the `.mlpackage` is missing from `index.json`, and a refusal to start on a
  mismatch.
- **Partial reproducibility.** `pyproject.toml` has only lower bounds, no lock;
  the `pokemon-tcg-data` revision is not pinned. The index is therefore not
  bit-for-bit reproducible, whereas it is the central artefact.
- **No feedback loop.** No mechanism to collect the failures from the app. Yet it
  is the only way out of n=31.
- **No model card.** An expert project publishes: the encoder's training data,
  the index's data, the tested population, the *untested* populations, metrics
  with CI, discouraged uses. The elements all exist, scattered in `OVERVIEW.md`;
  they are not gathered in a citable form.

---

## 8. Improvement plan, by gain / effort ratio

### A — Stratified synthetic bench, as an anti-regression guard — ✅ **done**
**Effort: low.**

Delivered: the degradation model is isolated in `ml/src/augment.py` (each stage
documented by the real condition it reproduces) and the bench in
`ml/scripts/evaluate_synthetic.py` — rate per era with Wilson intervals, JSON
output, deterministic draw, and a `--projection` flag to compare a space variant.

A flaw of the initial audit is corrected along the way: the measurement script
drew its seeds from `hash()`, randomised between processes by `PYTHONHASHSEED`.
The figures of §1.3 stay valid as a measurement, but were not bit-for-bit
reproducible; `augment.view_rng` now derives the seed from the string itself.
`scripts/audit/exp_synthetic.py` is removed in favour of the production script.

*Left to do:* wire it into CI with a per-era regression threshold.

### B — Defensible evaluation protocol — ✅ **done**
**Effort: low. Impact: it is what makes everything else credible.**

Delivered:

1. **Split recorded in `truth.json`** — `calibration` (21 photos, 1st
   collection) and `test` (11 photos, 2nd collection). The `_protocol` block says
   precisely what the test split is a hold-out of (the confidence thresholds,
   frozen in `a240b84`, before these photos arrived in `99ba7d2`) and what it is
   **not** (`MIN_QUAD_AREA`, `MIN_MULTI_QUAD_AREA`, `SHAPE_BUCKET`, introduced in
   the very commit that adds these photos — any detection precision measured on
   them is optimistic).
2. **`ml/src/stats.py`** — Wilson interval, risk/coverage curve, AURC, coverage
   at target precision. Checked: `wilson(31, 31) = (0.8897, 1.0)`.
3. **`evaluate_real.py` reworked** — report per split with intervals, `--ablation`
   (three arms), `--json` for a CI-usable output, and the risk/coverage curve in
   place of the threshold.
4. **`identify(..., use_band=False)`** in `src/pipeline.py` — the ablation is
   measured on the real code, not on a divergent copy.
5. **README and OVERVIEW** carry the ablation table and the CI.

Result measured by the new bench (`--ablation`):

| Configuration | Top-1 | Δ |
|---|---:|---:|
| whole photo, no detection or orientation | 4/31 (13 %, CI95 5-29) | |
| + detection, filtering, orientation (similarity alone) | 26/31 (84 %, CI95 67-93) | **+22** |
| + reading the printed number (full chain) | 31/31 (100 %, CI95 89-100) | **+5** |

The test split turns out harder than the calibration split on similarity alone:
**7/10 against 19/21**. The gap is not significant at this size (the intervals
overlap widely), but it goes in the direction expected from tuning on the
calibration set, and it alone justifies having separated the two.

*What stays open:* the bench no longer has a hold-out for the detection
parameters. A 3rd never-seen batch is needed — 40+ photos, other devices, and
pre-Sun & Moon cards, of which the bench so far contains no example.

### C — Learned projection head — 🔶 **trained, measured, not shippable as is**
**Effort: medium. Impact: real, but blocked by something other than the model.**

Delivered: `ml/scripts/train_projection.py` trains a 512→512 linear projection in
InfoNCE on pairs (degraded scan → scan), with hard negatives taken from the
nearest neighbours measured in §5. Initialisation to identity, so that any
measured gap is an effect of the learning and not of the starting point.
Selection on the validation, never on the last epoch. **Per-card** separation:
3,076 cards held out of training. No real photo enters the learning.

#### What it gains

Synthetic bench, **cards never seen in training**, same degradations
(n = 1,209):

| Era | without | with | Δ |
|---|---:|---:|---:|
| WotC 1990s | 75.6 % | 80.5 % | +4.9 |
| WotC 2000s | 84.7 % | 88.1 % | +3.4 |
| EX | 94.2 % | 95.8 % | +1.6 |
| DP/Pt | 95.8 % | 95.8 % | = |
| HGSS | 91.1 % | 97.8 % | **+6.7** |
| BW | 91.7 % | 98.3 % | **+6.6** |
| XY | 91.7 % | 95.8 % | +4.1 |
| SM | 94.2 % | 100 % | +5.8 |
| SWSH | 90.0 % | 99.2 % | **+9.2** |
| SV | 95.0 % | 95.0 % | = |
| promos | 87.5 % | 91.7 % | +4.2 |
| **TOTAL** | **91.1 %** | **95.2 %** | **+4.1** |

Top-5: 98.2 % → **99.9 %**. No era regresses.

On the real bench, at corrected configuration (see below): top-1 **31/31
unchanged**, but the quality of the **sorting** progresses markedly on the
similarity-alone arm — AURC **0.047 → 0.016**, and the 100 % precision holds up
to **84 % coverage instead of 65 %**. The no-detection arm goes from 4/31 to
9/31.

#### What it broke, and what it reveals

Applied as is, the projection **dropped the chain from 31/31 to 29/31** and
firmly announced a card back. The model was not to blame: **four absolute
quantities of the pipeline do not survive a change of space**, and only three
were identified.

| Constant | Old space | Projected space |
|---|---:|---:|
| `FIRM_ID_MARGIN` | 0.03 | 0.0698 |
| `FIRM_NAME_MARGIN` | 0.04 | 0.0222 |
| `FALLBACK_SCORE` | 0.65 | < 0.4387 |
| **`selection_score`** | — | **margin weight ÷ 12** |

The fourth is the most interesting because it is written nowhere:
`selection_score` computes `score + (score − score₂)`, which **implicitly assumes
that score and margin live on the same scale**. The projection multiplies the
margins by 5.6 without touching the scores: the formula therefore becomes "margin
alone" — precisely the regime the project had measured as failing. The two lost
photos were exactly the two whose orientation was ambiguous, therefore those
where the selection arbitrates between 0° and 180°. Restoring the balance
(`--selection-margin-weight 0.08`) recovers both.

Thresholds re-proposed by `evaluate_real.py --calibrate`, **from the calibration
split alone** — the test split was used for nothing above.

#### Why it is not shippable

The card back (`Nothing.jpeg`) goes from "uncertain" to "name": an asserted
attribution where the system must stay silent. And this threshold **cannot be
calibrated honestly today** — the calibration split contains **no negative
example**: its 21 photos all have a right answer. The bench's only negative is in
the test split.

In other words: the current system's refusal behaviour is calibrated on nothing.
That it correctly refuses the card back is a happy accident of the thresholds,
not a measured property. Task C only makes it visible.

*Delivery condition:* about ten negatives in the calibration split — card backs,
non-cards, cards outside the index, blurry photos — then recalibration of the
four quantities, then re-verification on the test split. Until that is done, the
projection stays in `data/export/`, measured and not on device.

*Also left to do, if it is shipped:* fold `W` into the Core ML graph as a final
layer (zero inference cost) and regenerate `index.bin` with the same matrix — an
index and an encoder out of tune would give plausible and wrong answers, which is
the argument of task G.

### D — Multi-view enrolment — 🔶 **measured: no precision gain, but makes the refusal practicable**
**Effort: low. Operating cost: nil.**

`ml/scripts/build_multiview_index.py` replaces each card's reference vector with
the **centroid** of {scan, 2 degradations}, renormalised. The index keeps its
exact shape: 21 MB, same format, same search, no Swift change.

| | reference index | centroid |
|---|---:|---:|
| top-1 (real bench, 39 photos) | **35/39 (89.7 %)** | 34/39 (87.2 %) |
| top-5 | 37/39 | 36/39 |
| correct refusals | 6/7 | **7/7** |
| firm false positives | 3 | **2** |
| AURC | 0.101 | **0.077** |
| precision at 25 % coverage | 80 % | **100 %** |

It loses one identification and fixes the confidence. The decisive figure is
elsewhere: **the cost of the refusal.** For no negative to be asserted,
`FIRM_ID_MARGIN` must go to 0.0558 with the reference index, which leaves **6
firm verdicts out of 21**. With the centroid, 0.0278 is enough, and **16 out of
21** remain. The problem goes from "impracticable" to "costs 24 % of the
coverage".

The blur of `IMG_5124`, asserted firmly by the reference index, becomes
"uncertain" again. It is the only change that fixes one of the three false
positives without any hand-tuning.

**A limit of my own instrument, discovered while using it.** The synthetic bench
gives 99.9 % to the centroid against 91.1 % to the reference index — a gain that
does not carry over to real photos at all. The reason is structural: enrolment
and query come out of the **same degradation model**, so the index was moved
toward exactly the family of images it is then shown. Changing the salt does not
change it: the salt changes the draw, not the family. The synthetic bench
honestly compares two **spaces**; it cannot compare two **enrolment** strategies.
It is written at the top of `evaluate_synthetic.py` so that nobody is caught out.

*Left to do:* the multi-vector variant (k vectors per card, score = maximum over
the views) is more expressive, but it multiplies the index by k and breaks the
row-to-row correspondence with `card_ids`. To open only if the centroid is
adopted — and the decision depends on a product trade-off: one fewer
identification against one fewer firm false positive.

### E — Expose the structural ceiling in the product — ✅ **done**
**Effort: low. Direct product impact.**

`ml/scripts/label_discriminability.py` labels each card, **by measurement and not
by a rule**: it replays the cached degraded views against the whole index and
looks at what the search returns. A card is marked `needs_printed_number` if none
of its views retrieves it at the top with a usable margin. `export_index.py`
writes the field into `cards.json`.

**4,787 cards out of 20,512 (23.3 %)** cannot be settled by the image alone —
i.e. five times more than the 884 near-duplicates suggested. Reassuring cross-
validation: **all 884 are captured.**

| Series | Cards | Number required |
|---|---:|---:|
| Base | 494 | **78.3 %** |
| E-Card | 529 | 29.3 % |
| Sword & Shield | 3,667 | 26.7 % |
| Scarlet & Violet | 3,595 | 16.9 % |
| Diamond & Pearl | 900 | 13.6 % |
| Platinum | 517 | 7.5 % |

The point is that the information arrives **before** the answer: when the first
candidate carries the flag, a tight margin is not an anomaly but the expected
behaviour, and the app can ask for a shot of the bottom strip instead of
displaying an "uncertain" it could have predicted.

### G — Traceability and reproducibility — 🔶 **fingerprints done, locks to do**
**Effort: low.**

`ml/src/fingerprint.py` computes the SHA-256 fingerprint of an artefact,
directory included (relative paths sorted and folded into the digest, so that a
moved weight changes the fingerprint). `export_coreml.py` writes it into
`encoder_meta.json`, `export_index.py` replicates it in `index.json`, and
`AGENTS.md` §4 tells the app to compare the two and **refuse to start** on a
mismatch.

It was the only trap of the kit that was not tooled: an index and an encoder out
of tune raise no error, the search returns neighbours, the scores stay in their
range, and the answers are plausible and wrong.

An out-of-date figure is corrected along the way: `AGENTS.md` announced "top-1
identical on 200/200" for the float32 → float16 pass; the measurement on the
current index gives **199/200 top-1 and 193/200 top-5**. The residual reordering
is on already-undecidable neighbourhoods, which are precisely a matter of task E.

**Revision of the pinned source.** `build_metadata.py` read `pokemon-tcg-data` on
`master` — a moving target: two rebuilds a month apart gave two different indexes
with nothing to signal it. The revision is now pinned to `8b4e3879` (2026-07-17,
the commit that adds Pitch Black, so exactly the state that produced the measured
20,512 cards). `--check` compares the pin to upstream, `--ref` allows it to be
moved deliberately, and `data/processed/source.json` notes the revision used,
which `export_index.py` copies into `index.json`. The shipped kit therefore says
where its two halves come from: which encoder produced the vectors, which
revision produced the cards.

**Dependency lock.** `scripts/lock_dependencies.py` writes
`requirements.lock.txt` — 59 packages, transitive closure of the *declared*
dependencies, and not a `pip freeze` of the working environment that would make
pandas, umap and Embedding Atlas look necessary to the identification chain.

**A packaging bug discovered while doing it:** `pyproject.toml` declared neither
`opencv-python-headless`, nor `pyobjc-framework-Vision`/`Quartz`, nor
`pillow-heif`, whereas `src/detect.py`, `src/orient.py`, `src/edition.py` and
`src/augment.py` import them. **The README's installation procedure produced a
broken environment** — nobody noticed, the development environment having
received them by another path. The extras are now separated: `coreml`, `dev`, and
`atlas` for exploration.

**Seeds.** Checked: all the randomness in the repo is already seeded —
`default_rng(0)` for the card draws, `torch.manual_seed` in training, and
`augment.view_rng` which derives its seed from the card identifier.

*Left to do:* recording the provenance in the shipped index requires a full
rebuild of the metadata. It has **not** been run here, and that is deliberate:
`build_metadata.py` rewrites `cards.json` from scratch, which would erase the 49
cards added by hand by `add_missing_cards.py` and misalign the index from its
embedding bank. To be done at the next rebuild, in the order documented by the
README. Meanwhile, `export_index.py` signals it instead of inventing a
provenance.

### F — Tests and CI — 🔶 **tests done, CI to do**
**Effort: medium.**

Delivered: **36 tests** in `ml/tests/`, runnable everywhere — none needs Vision,
the model or the 42 MB index.

- `test_stats.py` — the statistical base. If these values drift, all the
  published figures drift with them. Including the scale invariance of the AURC,
  which is the property that makes two spaces comparable.
- `test_invariants.py` — the traps of `AGENTS.md` §3, each turned into a test:
  the centre crop does trim the top of a portrait card, the direct squash does
  produce a different image, the printed total must match exactly, the margins
  are computed before reordering, the read number takes precedence over the
  margins.

An undocumented behaviour appeared while writing the tests, and it is now
recorded: **an OCR error on the leading zero is not recoverable.** Leading zeros
are stripped before comparison, so a "143" reading for a printed "043" becomes
143 against 43 — two different lengths, therefore no match, whereas it is indeed
a single-digit substitution. The system refuses rather than risk a wrong
attribution, which is the right sign of the error, but it costs recall on the
sets with three-digit printed numbers.

*Left to do:* the CI. A macOS runner so that the real bench and the Core ML
parity run; the 36 tests and the reduced synthetic bench can run on Linux.

### H — Feedback loop from the app
**Effort: medium, depends on the app. Now the shortest path to B2.**
Opt-in reporting of the photos that gave "uncertain" or were corrected by the
user. It is the only proposal whose gain grows with time — and since the 3rd
batch, it is also the most realistic way to gather the thirty to fifty negatives
that B2 demands: the refusals are precisely what this reporting collects.

### I — Model card — ✅ **done**
**Effort: very low.**

`docs/model-card.md`: intended use, out-of-scope, metrics with intervals, known
failures, negative results. The part that existed nowhere before is **what NOT to
ask of the system** — no money decision, no inventory without review, no Japanese
or Korean, no condition grading or counterfeit detection.

---

## Appendix — reproducing the measurements

The three audit scripts are versioned in `ml/scripts/audit/`. They are read-only
on the data and write nothing.

```bash
cd ml
.venv/bin/python scripts/audit/audit_index.py       # §4 §5 — index structure
.venv/bin/python scripts/audit/exp_riskcov.py       # §3 §4 — risk/coverage, 3 spaces
.venv/bin/python scripts/evaluate_synthetic.py      # §1.3 — synthetic bench by era
```

Reference figures of the existing chain, re-checked:

```
scripts/evaluate_real.py  ->  top-1 31/31, top-5 31/31, correct refusals 1/1
similarity alone (same selection, without the bottom-strip OCR)  ->  top-1 26/31
```
