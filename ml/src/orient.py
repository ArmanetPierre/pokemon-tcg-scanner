"""Détection de l'orientation d'une carte redressée, par position du texte.

Deux résultats de mesure fondent ce module :

1. Vision lit le texte inversé aussi bien que le texte droit (ratios de volume
   ~1,0 en mode accurate) — le *sens* du texte ne discrimine donc rien.
2. La *position* du texte discrimine : sur une carte à l'endroit, la masse de
   texte (attaques, bandeau) est dans la moitié basse ; l'illustration occupe
   le haut. Centre de masse mesuré à 0,65-0,82 sur les cartes du banc d'essai.

Corollaire mesuré sur 5020/5024/5032 : l'embedding ne doit jamais arbitrer
l'orientation — un crop inversé peut scorer plus haut sur une mauvaise carte
que le crop droit sur la bonne (0,834 Spiritomb contre 0,788 Sinistcha).

Une seule passe d'OCR par carte. Même API que l'app iOS.
"""

from __future__ import annotations

import cv2
import numpy as np
import Quartz
import Vision

# Zone morte autour de 0,5 : si le centre de masse du texte est trop proche du
# milieu, ou s'il y a trop peu de texte, on ne tranche pas.
UP_THRESHOLD = 0.55
DOWN_THRESHOLD = 0.45
MIN_CHARS = 20


def _cgimage_from_bgr(bgr: np.ndarray):
    """Convertit un tableau BGR OpenCV en CGImage, sans passer par le disque."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    data = rgb.tobytes()
    provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    return Quartz.CGImageCreate(
        width, height,
        8, 24, width * 3,
        Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGImageAlphaNone,
        provider, None, False,
        Quartz.kCGRenderingIntentDefault,
    )


def text_mass_center(bgr: np.ndarray) -> tuple[float, float]:
    """Centre de masse vertical du texte (0 = haut, 1 = bas) et volume lu.

    Mode rapide sans correction linguistique : seules les positions des boîtes
    comptent, pas le contenu. Chaque boîte pèse le nombre de caractères lus.
    """
    cgimage = _cgimage_from_bgr(bgr)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(1)  # fast
    request.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
    ok, _ = handler.performRequests_error_([request], None)
    if not ok:
        return float("nan"), 0.0

    mass = weight_sum = 0.0
    for obs in request.results() or []:
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        weight = len(candidates[0].string().strip())
        box = obs.boundingBox()  # coordonnées normalisées, origine en bas
        y_from_top = 1.0 - (box.origin.y + box.size.height / 2)
        mass += weight * y_from_top
        weight_sum += weight

    if weight_sum == 0:
        return float("nan"), 0.0
    return mass / weight_sum, weight_sum


def pick_orientation(warped: np.ndarray) -> tuple[str, np.ndarray | None]:
    """Tranche entre le crop tel quel et sa rotation à 180°.

    Retourne ("0°" | "180°", image orientée) si la position du texte est
    décisive, ("ambigu", None) sinon — l'appelant garde alors les deux
    hypothèses en concurrence.
    """
    center, chars = text_mass_center(warped)
    if chars >= MIN_CHARS:
        if center >= UP_THRESHOLD:
            return "0°", warped
        if center <= DOWN_THRESHOLD:
            return "180°", cv2.rotate(warped, cv2.ROTATE_180)
    return "ambigu", None
