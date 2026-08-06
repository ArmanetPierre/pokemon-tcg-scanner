"""Détection de carte et correction de perspective.

Sert de banc d'essai pour ce que fera `VNDetectRectanglesRequest` +
`CIPerspectiveCorrection` sur iOS. L'objectif n'est pas d'écrire le détecteur
définitif en Python, mais de savoir si une détection de quadrilatère générique
suffit sur des photos réelles — ou s'il faudra entraîner un YOLO.

Une carte Pokémon fait 63x88 mm, soit un rapport largeur/hauteur de 0,716.
C'est la contrainte qui permet de rejeter les rectangles parasites (écrans,
carrelage, dalles de table).
"""

from __future__ import annotations

import cv2
import numpy as np

CARD_ASPECT = 63.0 / 88.0
ASPECT_TOLERANCE = 0.18
OUTPUT_SIZE = (734, 1024)  # même résolution que les images de référence


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordonne 4 points en (haut-gauche, haut-droite, bas-droite, bas-gauche)."""
    pts = pts.reshape(4, 2).astype(np.float32)
    summed = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [
            pts[np.argmin(summed)],   # haut-gauche : x+y minimal
            pts[np.argmin(diff)],     # haut-droite : x-y maximal
            pts[np.argmax(summed)],   # bas-droite
            pts[np.argmax(diff)],     # bas-gauche
        ],
        dtype=np.float32,
    )


def _quad_aspect(quad: np.ndarray) -> float:
    """Rapport petit côté / grand côté du quadrilatère."""
    widths = [
        np.linalg.norm(quad[1] - quad[0]),
        np.linalg.norm(quad[2] - quad[3]),
    ]
    heights = [
        np.linalg.norm(quad[3] - quad[0]),
        np.linalg.norm(quad[2] - quad[1]),
    ]
    w, h = float(np.mean(widths)), float(np.mean(heights))
    if w == 0 or h == 0:
        return 0.0
    return min(w, h) / max(w, h)


def find_card_quads(bgr: np.ndarray, max_results: int = 3) -> list[np.ndarray]:
    """Retourne les quadrilatères plausibles, en coordonnées de l'image d'entrée."""
    height, width = bgr.shape[:2]
    scale = 1024 / max(height, width)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    candidates: list[tuple[float, np.ndarray]] = []
    small_area = small.shape[0] * small.shape[1]

    # Plusieurs seuils de Canny : un seuil unique rate soit les cartes sur fond
    # sombre, soit celles à contraste faible (carte claire sur table claire).
    for low, high in ((30, 90), (50, 150), (75, 200)):
        edges = cv2.Canny(gray, low, high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < small_area * 0.02:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            quad = order_corners(approx)
            aspect = _quad_aspect(quad)
            if abs(aspect - CARD_ASPECT) > ASPECT_TOLERANCE:
                continue

            candidates.append((area, quad / scale))

    # Dédoublonner : les trois passes de Canny retrouvent souvent le même bord.
    kept: list[np.ndarray] = []
    for _, quad in sorted(candidates, key=lambda c: -c[0]):
        centre = quad.mean(axis=0)
        if any(np.linalg.norm(centre - k.mean(axis=0)) < 0.05 * max(height, width) for k in kept):
            continue
        kept.append(quad)
        if len(kept) >= max_results:
            break
    return kept


def warp_card(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Redresse le quadrilatère en une image de carte portrait."""
    w, h = OUTPUT_SIZE
    aspect = _quad_aspect(quad)
    del aspect  # l'orientation est déduite des longueurs de côtés ci-dessous

    side_top = np.linalg.norm(quad[1] - quad[0])
    side_left = np.linalg.norm(quad[3] - quad[0])
    if side_top > side_left:
        # Carte couchée : on tourne la correspondance d'un quart de tour pour
        # sortir un portrait plutôt qu'un paysage écrasé.
        quad = np.roll(quad, 1, axis=0)

    target = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(bgr, matrix, (w, h))
