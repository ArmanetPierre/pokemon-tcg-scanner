# Evaluation photos

Real photos taken with an iPhone. This is the only test set that matters: the
official scans are in the index, so identifying them proves nothing.

## Where to put the files

One folder per physical card, in `photos/`:

```
photos/
  pikachu_58-102/
    01_plat.jpg
    02_incline.jpg
    03_pivote.jpg
    ...
  charizard_4-102/
    01_plat.jpg
    ...
```

**Folder name**: what is printed at the bottom of the card, i.e. the number and
the total (`58/102` → `58-102`), preceded by the Pokémon's name. That is enough
to find the exact ID unambiguously. If the number is illegible or absent
(promos), put what you can, we will resolve it case by case.

**File name**: the numbered prefix does not matter, only the condition keyword
counts (see below). The iPhone format (HEIC) is accepted.

## Conditions to cover

In order of importance. The first 5 are enough for a first verdict.

| Keyword | Condition |
|---|---|
| `plat` | flat, well framed, correct light — this is the reference |
| `incline` | angled view, 30-45°, strong perspective |
| `pivote` | rotated in the plane (45°, 90°, upside down) |
| `reflet` | harsh light or a window in the axis — the hard case for holos |
| `sombre` | low light, or card in the shade |
| `sleeve` | in a plastic sleeve |
| `loin` | card small in the frame (less than 1/4 of the image) |
| `masque` | a corner or an edge hidden by a finger / another card |
| `fond` | placed on a cluttered table, busy background |
| `flou` | slight motion blur |

## Which cards to choose

The choice of cards matters as much as the choice of conditions:

- **2-3 holographic or reverse holo** — reflections are the hardest case, and the
  index only contains matte, perfect scans.
- **1-2 very common cards** (a Pikachu, a base Charizard). These are the ones that
  have been reprinted across several sets, so the ones where the identification
  comes down to a 0.02 gap.
- **1-2 recent cards** (Scarlet & Violet or more recent) and **1-2 old ones**
  (Base, Neo, EX) — the layouts differ a lot.
- **1 non-French card if you have one**: the index is in English. Knowing whether
  a French card is recognized changes the scope of the project.

## How many

**Start small: 2 or 3 cards with the first 5 conditions**, i.e. about fifteen
photos. That validates the protocol and the naming before you spend an hour on
it. We then extend to about ten cards.

## Useful bonus, later

A few photos with **several cards visible** (3-5 spread on a table, overlapping a
bit). They are not used for identification evaluation, but they will say whether
Apple's rectangle detection is enough or whether a YOLO will have to be trained.
To be put in `photos/_multi/`.
