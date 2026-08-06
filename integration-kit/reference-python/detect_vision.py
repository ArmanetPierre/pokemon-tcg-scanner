"""Détection de carte via Vision.framework d'Apple, pilotée depuis Python.

Même API que celle qu'utilisera l'app iOS (`VNDetectRectanglesRequest` et
`VNDetectDocumentSegmentationRequest`), ce qui permet de valider l'approche sur
de vraies photos avant d'écrire du Swift. Si Vision suffit ici, il suffira dans
l'app — et le YOLO de la spec devient inutile.

Vision renvoie les 4 coins en coordonnées normalisées avec l'origine en bas à
gauche ; on les repasse en pixels avec l'origine en haut à gauche.
"""

from __future__ import annotations

import numpy as np
import Quartz
import Vision
from Foundation import NSURL

CARD_ASPECT = 63.0 / 88.0


def _load_cgimage(path: str):
    url = NSURL.fileURLWithPath_(path)
    source = Quartz.CGImageSourceCreateWithURL(url, None)
    if source is None:
        raise ValueError(f"image illisible : {path}")
    return Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)


def _observation_to_quad(obs, width: int, height: int) -> np.ndarray:
    """VNRectangleObservation -> quadrilatère pixels (HG, HD, BD, BG)."""
    corners = [obs.topLeft(), obs.topRight(), obs.bottomRight(), obs.bottomLeft()]
    return np.array(
        [[p.x * width, (1.0 - p.y) * height] for p in corners],
        dtype=np.float32,
    )


def detect_rectangles(path: str, min_confidence: float = 0.5) -> list[np.ndarray]:
    """Détecteur classique de rectangles, contraint au format d'une carte."""
    cgimage = _load_cgimage(path)
    width = Quartz.CGImageGetWidth(cgimage)
    height = Quartz.CGImageGetHeight(cgimage)

    request = Vision.VNDetectRectanglesRequest.alloc().init()
    # Vision définit le rapport comme petit côté / grand côté. On laisse une
    # marge large : la perspective écrase fortement le rapport apparent.
    request.setMinimumAspectRatio_(0.45)
    request.setMaximumAspectRatio_(0.95)
    request.setMinimumSize_(0.05)
    request.setMinimumConfidence_(min_confidence)
    request.setMaximumObservations_(8)
    request.setQuadratureTolerance_(30.0)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
    success, error = handler.performRequests_error_([request], None)
    if not success:
        raise RuntimeError(f"Vision a échoué : {error}")

    results = request.results() or []
    return [_observation_to_quad(obs, width, height) for obs in results]


def detect_document(path: str) -> list[np.ndarray]:
    """Segmentation de document : modèle appris, bien plus robuste au fond.

    Ne renvoie qu'un seul objet — utile pour la photo d'une carte isolée, pas
    pour un plateau de plusieurs cartes.
    """
    cgimage = _load_cgimage(path)
    width = Quartz.CGImageGetWidth(cgimage)
    height = Quartz.CGImageGetHeight(cgimage)

    request = Vision.VNDetectDocumentSegmentationRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
    success, error = handler.performRequests_error_([request], None)
    if not success:
        raise RuntimeError(f"Vision a échoué : {error}")

    results = request.results() or []
    return [_observation_to_quad(obs, width, height) for obs in results]
