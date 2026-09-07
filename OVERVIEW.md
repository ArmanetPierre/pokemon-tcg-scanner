# Pokémon TCG card recognition, on-device

Identifies a Pokémon trading card from a photograph, entirely on the phone, with
no network call. Point a camera at a card and get back which card it is, which
set it was printed in, and how much that answer can be trusted.

This file is an overview of the whole project. `docs/model-card.md` is the
citable summary: intended use, what is out of scope,
metrics with confidence intervals, known failures, and published negative
results.

**Status.** 35 correct identifications out of 39 real iPhone photographs
(89.7 %, Wilson 95 % interval 76-96 %). The bench read 31/31 until a third batch
— older cards, botched framing, and seven photographs with **no correct answer
at all** — brought it down to this. That batch also produced **three confident
wrong attributions**, one of them on a motion blur no human can identify. The
photographs come from three collections shot by several people (French cards
against an English index, in ordinary conditions — backlight, sleeves, cluttered
backgrounds, tilted cards, binder pages, one batch entirely in landscape),
measured on a Mac. It also runs on an iPhone, inside a real app, where the
encoder takes **5.5 ms** on the Neural Engine.

Do not read the headline number without `docs/model-card.md`: what the system
must not be used for is as measured as what it does.

**What each stage contributes**, by ablation on that same bench:

| Configuration | Top-1 |
|---|---|
| whole photograph, no detection or orientation | 5/39 (13 %, CI95 6-27) |
| + detection, filtering, orientation → **similarity alone** | 29/39 (74 %, CI95 59-85) |
| + reading the printed number → **full chain** | 35/39 (90 %, CI95 76-96) |

Framing is worth 24 identifications; the embedding only ever works on what it is
handed; reading the printed collector number recovers 6 more. The headline
number is a property of the *chain*, not of the model — reach for a better
encoder and you are optimising the stage that contributes least.

---

## The idea

There is no 20 000-class classifier here. The architecture is **detect, embed,
search**:

```
photo
  → find the card's quadrilateral        (Vision)
  → flatten it                           (homography)
  → decide which way up                  (position of text, never the embedding)
  → turn it into a 512-number vector     (Core ML, MobileCLIP2-S2)
  → find the closest of 20 512 vectors   (cosine similarity)
  → read the printed collector number    (Vision, on the bottom edge)
→ card + confidence level
```

That choice is what makes the thing maintainable. A new set does not need
retraining, or fine-tuning, or any GPU time at all: regenerate the index and
ship it. The model never changes.

It also means the system can only ever return a card that is in the index — a
limit worth knowing, and one the pipeline detects rather than hides (see
*Confidence*).

---

## What it can recognise

**20 512 cards across 176 sets** — the English game, from Base Set (1999) to
Pitch Black.

| Era | Cards | Share |
|---|---|---|
| Base / WotC (1999–2003) | 1 780 | 8.7 % |
| EX (2003–2007) | 1 722 | 8.4 % |
| Diamond & Pearl / Platinum | 1 417 | 6.9 % |
| HeartGold SoulSilver / Call of Legends | 544 | 2.7 % |
| Black & White | 1 416 | 6.9 % |
| XY | 1 775 | 8.7 % |
| Sun & Moon | 2 955 | 14.5 % |
| Sword & Shield | 3 529 | 17.3 % |
| Scarlet & Violet | 3 249 | 15.9 % |
| Mega Evolution (incl. MEP promos) | 1 039 | 5.1 % |
| promos, decks, specials | 1 028 | 5.0 % |

Not covered: the Japanese-only game (the index is entirely English, though
French cards are recognised — the artwork dominates the text by a wide margin),
Every card the index knows now has an image — the last gap, the eight Mega
Evolution energies, is closed. That set is absent from the primary source
entirely, metadata included, so its cards were sourced by hand and their
metadata taken from a secondary catalogue.

The 48 McDonald's Collection cards and one HGSS promo that were missing for the
same reason are now in, sourced by hand. Their trap is worth knowing: these
cards are reprints, and public sources serve only the *original* printing. A
McDonald's promo and its original measure 0.928 apart, where two entirely
unrelated cards already sit at 0.79 — so filing the wrong printing under the
right identifier would place the index 0.07 from the truth, more than twice the
margin that decides an identification.

Filling that gap needs care rather than effort. The fallback CDN never returns
404: it answers any unknown identifier with a placeholder over HTTP 200. Trusting
the status code would have added 57 identical card backs to the index, which
would then behave as universal attractors — every poor photograph would resemble
them. Availability is therefore decided on a content digest, in
`ml/src/fallback_images.py`.

**Being in the index is not the same as being tested.** The bench holds exactly
**one** card older than Sun & Moon, while **47 % of the index predates it** —
and that one card is among the failures, confidently misattributed. Coverage is
a count; it is not evidence.

The synthetic bench measures the blind spot the real one cannot reach: over
1 209 queries on held-out cards, the WotC era scores **75.6 %** against **95.8 %**
for Diamond & Pearl — twenty points of spread.

Two numbers explain much of the design:

- **92 % of cards share their name with another card.** Pikachu appears 99 times,
  Eevee 63. This is why confidence has two levels rather than one: the confusion
  lives *inside* a name, and "right card, unsure which printing" is a useful
  answer rather than a failure.
- **Only 38 % of cards are identifiable by their printed number alone.** 63 % of
  number/total pairs are unique, but those cover just 7833 cards; the rest share
  a pair with up to nine others.

---

## The two repositories

| | |
|---|---|
| **[pokemon-tcg-scanner](https://github.com/ArmanetPierre/pokemon-tcg-scanner)** (this one) | The model: dataset assembly, embedding index, evaluation bench, Core ML export, and the integration kit. Python. |
| **[poke-scanner](https://github.com/hugo-heer/poke-scanner)** | An Expo / React Native iOS app that uses it, as a native module (`modules/expo-card-encoder`). Swift + TypeScript. |

The bridge between them is `integration-kit/`, which holds the exported model,
the index, a reference Python implementation that is the source of truth for
behaviour, and `AGENTS.md` — a step-by-step porting guide with the matching
native API for each stage. Read that before writing platform code.

The app keeps **two** identifiers side by side. Its original one reads the
printed name with OCR and looks it up on TCGdex over the network; this one
recognises the artwork offline. Neither strictly beats the other, so both are
wired up and the app's mode button switches between them. That is deliberate:
comparing them on the same photograph is the only way to see where each gives
out.

---

## Getting the model

The encoder and the index are ~94 MB and are not in git. A built copy is
published as a release:

```bash
gh release download kit-v5 --repo ArmanetPierre/pokemon-tcg-scanner
tar -xzf card-encoder-kit.tar.gz -C integration-kit/
```

| File | Size | Role |
|---|---|---|
| `CardEncoder.mlpackage` | 69 MB | image → 512-dimension vector |
| `index.bin` | 21 MB | 20 512 float16 vectors, L2-normalised |
| `index.json` | 0.2 MB | index shape and `card_ids`, in row order |
| `cards.json` | 5.0 MB | display metadata, aligned with the index, incl. `needs_printed_number` |

The iOS app fetches this for you: `npm run sync:model`.

## Rebuilding it from scratch

Everything heavy is reproducible and deliberately absent from the repository —
5.5 GB of reference images, the embedding bank, the Core ML model, the binary
index, and the test photographs.

```bash
cd ml
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt   # the exact versions measured
.venv/bin/pip install -e . --no-deps

.venv/bin/python scripts/build_metadata.py     # metadata, at the pinned revision
.venv/bin/python scripts/add_missing_cards.py  # sets that source lacks, from TCGdex
.venv/bin/python scripts/download_images.py    # high-resolution images, ~15 min
.venv/bin/python scripts/build_embeddings.py   # vector index, ~6 min on an M3

.venv/bin/python scripts/export_coreml.py      # CardEncoder.mlpackage + parity check
.venv/bin/python scripts/export_index.py       # index.bin + cards.json
```

`scripts/refresh_index.py --check` reports sets released since the index was
last built; `scripts/add_missing_cards.py --check` says which of them can
actually be added, and which have no image anywhere.

## Checking it

```bash
.venv/bin/python scripts/evaluate_real.py            # the bench, on PyTorch
.venv/bin/python scripts/evaluate_real.py --ablation # what each stage contributes
.venv/bin/python scripts/evaluate_real.py --coreml   # on the exported model
.venv/bin/python scripts/scan.py photo.jpeg          # identify one photograph
```

Every rate is reported per split (`calibration` / `test`) with its Wilson
interval, and confidence as a risk-coverage curve with its AURC. A threshold is
comparable neither across models nor across metric spaces; a curve always is.

`integration-kit/test/` holds fixtures for validating a port in stages, with
expected values in `expected.json`. Start with the preprocessing alone: encode
`reference_card.jpg`, which is itself in the index, and confirm it comes back as
`me5-36` at about 0.995. Nothing downstream is worth debugging until it does.

---

## Four things that will cost you time

Each of these cost real time during development, and each fails quietly rather
than loudly. They are covered at length in `integration-kit/AGENTS.md` §3.

**The absolute score discriminates nothing.** Every card in the game shares a
layout, so two entirely unrelated cards already sit at about 0.79 — the useful
range is 0.79 to 1, not 0 to 1. The worst failure on the bench carried the
second-highest absolute score of the whole set. Never threshold on it and never
show it to a user; what carries information is the **gap** between the first
candidate and the next.

**The embedding must never decide orientation.** An upside-down card still looks
like a card, and it can out-score the upright crop against the *wrong* card —
measured at 0.834 for a wrong answer against 0.788 for the right one. Orientation
is decided on where the text sits: a card the right way up keeps its attack text
and its footer in the lower half.

**The preprocessing geometry has to match to the pixel.** The index was built by
scaling each reference scan's *shortest* side to 256 with antialiased bilinear
resampling, then cropping the centre — which deliberately trims the top and
bottom off a portrait card. Squashing an image straight to 256×256 instead drops
measured parity to 0.65, and the symptom is not a crash but plausible, confident,
wrong answers.

**EXIF rotation must be handled the same on both sides.** Whatever finds the card
and whatever crops it must work in the same frame. Ignoring the flag on both
sides is perfectly valid; applying it to one and not the other points the
outlines at pixels that are not there, and you get black crops and scores around
0.4 that look like a bad model.

---

## Confidence

Two margins, both computed on the similarity ranking *before* the printed number
promotes anything — the other order measures gaps on an already-reordered list,
where they come out negative and mean nothing.

- **edition margin** — first candidate minus second.
- **name margin** — first candidate minus the first one bearing a *different*
  card name.

| Condition | Level | What to show |
|---|---|---|
| the collector number was read off the card | `edition` | card and set, firmly |
| edition margin ≥ 0.03 | `edition` | card and set, firmly |
| name margin ≥ 0.04 | `name` | "Primeape — set to confirm" |
| otherwise | `uncertain` | offer the top five, or ask for a steadier shot |

Two levels rather than one because nearly all the confusion lives *within* a
single card name — several printings of one artwork across different sets. A
tight margin usually means "right card, unsure which printing", which is worth
telling someone, and is not the same as not knowing.

These thresholds were calibrated on the first collection and have not moved
since; the second player's 11 photographs plus one card back are therefore
genuine held-out data **for the thresholds** — no over-claiming on either, and
the card back correctly lands in `uncertain`. They are *not* held out for the
quadrilateral filters, which were tuned in the same commit that added them; the
split annotations in `truth.json` say exactly which is which.

**These thresholds are not good enough, and that is measured.** On the widened
bench, precision at 25 % coverage (80 %) is *lower* than at full coverage
(89.7 %): sorting by margin is worse than not sorting. AURC is 0.101. One
photograph still gets a firm, wrong answer: a motion blur that carries a larger
margin than fifteen of the twenty-one correct identifications. The widened
bench first produced three; the other two no longer occur, and none of the 39
photographs that have a right answer is asserted wrongly today (29 firm
verdicts, 29 of them correct).

Raising the threshold does not fix it. To stop that blur being asserted,
`FIRM_ID_MARGIN` must go from 0.03 to 0.0558, which drops firm verdicts from
21/21 to **6/21**. Blocking one false positive costs 71 % of coverage. Nothing
was changed in production on the strength of seven negatives.

The root cause is that refusal rests on a single quantity, the margin, which
answers "are these two candidates close?" and not "am I even looking at a
card?". No threshold on the first answers the second. `docs/audit-ml.md` §0
carries the full measurement and the proposed fix — a confidence learned over
several signals, which needs thirty to fifty negatives rather than seven.

---

## Performance

| Stage | iPhone 13 Pro | Mac M3 Pro |
|---|---|---|
| detection (both detectors) | 24 ms | 26 ms |
| rectification | 6.4 ms | 2.9 ms |
| orientation OCR | 11 ms | 9.5 ms |
| geometric preprocessing | 1.2 ms | 1.0 ms |
| **embedding (Neural Engine)** | **5.5 ms** | **4.1 ms** |
| search over 20 512 vectors | 1.9 ms | 0.6 ms |
| collector-number OCR (`.accurate`) | 65 ms | 60 ms |

Per call. **What decides the cost of a photograph is how many times each of these
runs**, and the answer used to be "far too many": the rectangle detector returns
up to 8 observations, most of them tiny artefacts, and every one of them was paid
for in rectification, orientation OCR and embedding. Rejecting any quadrilateral
covering less than 2 % of the frame — before rectifying anything — takes a
photograph from 10.2 crops to 2.4. A second pass over the collector-number OCR
(the `.fast` recogniser on a 2× enlarged strip first, `.accurate` only when it
comes back empty) takes that stage from 60 ms to 24 ms on average. Together:
**287 ms to 159 ms per 12-megapixel photograph, with no identification lost**,
measured over the 32-photograph bench with `ml/scripts/profile_pipeline.py`.

The split on what remains: the model is about 5 % of an identification, and
Vision's two OCR passes about two thirds. Reaching for a lighter model still
optimises the thing that is not the constraint. What matters for a live video
loop is how often each stage runs — detection every frame, identification once
per card, the collector-number read once in the background. `AGENTS.md` §9 has
the budget.

Three things that are easy to get wrong and expensive to diagnose:

- **Compile the port with `-O`, even in Debug.** Framework work is unaffected,
  but hand-written pixel loops are not: measured at 124 ms and 114 ms per crop
  unoptimised against under a millisecond optimised — 1.8 seconds of a 2.3
  second scan, purely an artefact of the build configuration.
- **Instrument per call, not cumulatively.** A per-stage total is a sum over
  every crop hypothesis — up to 15 of them before the area filter, up to 7
  after. Without the call count the numbers cannot be compared between two
  scans, let alone against this table.
- **Detection and rectification must see the same image.** Detection reads the
  file and returns quadrilaterals in its pixel coordinates; rectification works
  on a decoded array. Decoding that array at half resolution while the detectors
  still read the full-resolution file took the bench from 18/18 to 1/18 — and
  raised no error at all, the quadrilaterals simply landing in the wrong frame.

---

## Known limits

- **Overlapping cards.** Apple's detectors need to see four closed edges. On a
  photograph of five cards where four overlapped, only one was isolated. For
  scanning a collection, ask for the cards to be spread out.
- **Cards outside the index.** Three of 21 photographed cards (recent promos,
  energies) are in no public source *with an image*. When a collector number
  reads cleanly and belongs to no card in the index, the pipeline reports
  "unknown card" rather than attributing it to whatever ranked first.
- **The embedding can miss entirely.** On one card the correct answer was not in
  the top ten at all, 0.089 behind an unrelated Trainer, while its number read
  cleanly three times running. Promotion can only reorder candidates the search
  returned, so the port resolves a uniquely-printed number directly instead.
  This is an improvement over the reference implementation, not a port of it —
  but it only reaches the **38 %** of cards whose number/total pair is unique.
  For the other 62 %, a card the search misses stays missed.
- **Older cards are untested.** Every test photograph is of a recent card. Base,
  Neo and EX era layouts differ substantially.
- **Language.** The index is English, and French cards are recognised — the
  artwork dominates the text by a wide margin. Japanese is untested.
- **The iOS Simulator is useless for judging accuracy.** Core ML behaves
  identically there; Vision does not. Use it to check that the module loads and
  the chain runs, and judge accuracy on a device.

---

## Updating the index

A new set needs no retraining. Regenerate `index.bin` and `cards.json`, publish
them, replace them in the app. The model does not change.

Worth planning for the index to be downloadable rather than bundled, so a new
set can ship without going through the App Store.
