"""Lecture du bandeau bas : tranche l'édition exacte parmi les candidats.

Le numéro imprimé (« 043/084 ») et lui seul distingue deux tirages de la même
illustration — c'est la raison d'être des images haute résolution. L'OCR
confond certains caractères (7↔1, 2↔), O↔0...), mais l'espace des valeurs
valides est fermé : on ne lit pas un nombre, on cherche lequel des candidats
du top-k correspond le mieux à ce qui est lu.

Même API que l'app iOS (`VNRecognizeTextRequest` en mode accurate, sur un crop
de bandeau minuscule — coût ~50 ms, une fois par carte).
"""

from __future__ import annotations

import re

import numpy as np
import Vision

from src.orient import _cgimage_from_bgr

# Fraction basse de la carte contenant la ligne numéro/set.
BAND_FRACTION = 0.14

# Confusions OCR observées sur le banc, appliquées avant extraction des motifs.
CONFUSIONS = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1", ")": "1", "]": "1", "i": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7", "?": "7",
    "B": "8",
    "g": "9", "q": "9",
})

NUMBER_PATTERN = re.compile(r"(\d{1,3})\s*/\s*(\d{1,3})")


def _ocr_strings(bgr: np.ndarray) -> list[str]:
    cgimage = _cgimage_from_bgr(bgr)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(0)  # accurate : le bandeau est petit et fin
    request.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
    ok, _ = handler.performRequests_error_([request], None)
    if not ok:
        return []
    out = []
    for obs in request.results() or []:
        candidates = obs.topCandidates_(1)
        if candidates:
            out.append(candidates[0].string())
    return out


def read_number_pairs(card_bgr: np.ndarray) -> list[tuple[str, str]]:
    """Lit les motifs « numéro/total » du bandeau bas d'une carte redressée."""
    height = card_bgr.shape[0]
    band = card_bgr[int((1 - BAND_FRACTION) * height):, :]
    pairs = []
    for raw in _ocr_strings(band):
        cleaned = raw.translate(CONFUSIONS)
        pairs.extend(NUMBER_PATTERN.findall(cleaned))
    return pairs


def _digit_distance(a: str, b: str) -> int:
    """Distance entre deux nombres imprimés : substitutions chiffre à chiffre.

    Les zéros de tête sont normalisés (« 043 » ≡ « 43 ») ; une différence de
    longueur au-delà rend la correspondance invalide.
    """
    a, b = a.lstrip("0") or "0", b.lstrip("0") or "0"
    if len(a) != len(b):
        return 99
    return sum(x != y for x, y in zip(a, b))


def match_edition(hits, pairs: list[tuple[str, str]]):
    """Cherche quel candidat du top-k porte un des numéros lus.

    Retourne (indice du candidat, nombre d'erreurs OCR tolérées) ou None. Le
    total imprimé doit correspondre exactement — c'est lui qui identifie le
    set ; le numéro tolère une confusion résiduelle d'un chiffre.
    """
    best: tuple[int, int] | None = None
    for i, hit in enumerate(hits):
        number = str(hit.card.get("number") or "")
        total = str(hit.card.get("set_printed_total") or "")
        if not number.isdigit() or not total:
            continue
        for read_number, read_total in pairs:
            if _digit_distance(read_total, total) != 0:
                continue
            errors = _digit_distance(read_number, number)
            if errors <= 1 and (best is None or errors < best[1]):
                best = (i, errors)
    return best


def _normalize(value: str) -> str:
    return value.lstrip("0") or "0"


def known_pair(index, pairs: list[tuple[str, str]]) -> bool:
    """Un des numéros lus existe-t-il quelque part dans l'index ?

    Si l'OCR lit proprement un couple numéro/total inconnu de tout l'index, la
    carte est probablement hors index : mieux vaut le dire que d'afficher une
    fausse attribution confiante.
    """
    if not hasattr(index, "_pair_set"):
        index._pair_set = {
            (_normalize(str(c.get("number"))), _normalize(str(c.get("set_printed_total"))))
            for c in index.cards_by_id.values()
            if c.get("set_printed_total")
        }
    return any(
        (_normalize(n), _normalize(t)) in index._pair_set for n, t in pairs
    )
