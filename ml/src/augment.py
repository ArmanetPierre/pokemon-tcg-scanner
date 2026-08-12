"""Modèle de dégradation : d'un scan éditeur vers une photo de téléphone.

L'index est bâti sur des scans propres — à plat, éclairage studio, pleine
résolution. Les requêtes sont des photos de téléphone d'une carte déjà
redressée par la chaîne. Entre les deux il y a un décalage de domaine que le
banc de 31 photos constate sans jamais le mesurer ni le réduire.

Ce module le simule. Il sert à deux choses, et il faut les distinguer :

- **mesurer** (`scripts/evaluate_synthetic.py`) — la vérité terrain est gratuite
  et exacte, donc l'évaluation peut couvrir les 20 512 cartes au lieu de 31.
  C'est une mesure *relative* : elle compare des ères, des sets et des variantes
  de modèle entre elles. Elle ne prédit pas la précision sur de vraies photos,
  parce qu'elle ne reproduit ni l'optique du capteur, ni son traitement d'image,
  ni les erreurs de la détection en amont.
- **entraîner** (`scripts/train_projection.py`) — les paires (référence dégradée,
  référence) fournissent la supervision qu'aucune donnée annotée ne donnerait.

Le risque de la seconde usage est réel et doit être surveillé : un modèle
entraîné sur ces augmentations peut apprendre à les inverser plutôt qu'à
généraliser. C'est pourquoi le juge de paix reste le banc de photos réelles, qui
n'entre jamais dans l'entraînement.

Chaque dégradation est tirée d'un générateur local, jamais du hasard global :
deux exécutions donnent exactement les mêmes images.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image

# Chaque étage reproduit une dégradation observée sur le banc réel.
#   perspective  — le redressement par homographie n'est jamais exact au pixel
#   échelle      — un crop de carte fait 300-800 px de large, le scan 700+
#   flou         — mise au point et bougé, très présents sur les photos tenues
#   exposition   — contre-jour et sur-exposition sont deux conditions du banc
#   balance      — lumière chaude d'intérieur contre scan en lumière neutre
#   reflet       — les holos sont un cas majeur, cité dans truth.json
#   bruit + JPEG — capteur en basse lumière, puis compression de l'appareil
JITTER = 0.02
SCALE = (0.25, 0.45)
BLUR_PROB, BLUR_SIGMA = 0.6, (0.6, 1.6)
EXPOSURE = (0.75, 1.25)
WHITE_BALANCE = (0.90, 1.10)
GLARE_PROB, GLARE_STRENGTH = 0.5, (25, 70)
NOISE_SIGMA = (1.0, 5.0)
JPEG_QUALITY = (55, 88)


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Applique le modèle de dégradation à une image de référence.

    L'entrée est le scan pleine résolution, la sortie une image plus petite et
    abîmée. Le prétraitement géométrique de l'encodeur (petit côté à 256 puis
    recadrage centré) s'applique APRÈS, comme pour une vraie requête.
    """
    a = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = a.shape[:2]

    # Perspective résiduelle : la carte est déjà redressée, mais imparfaitement.
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-JITTER, JITTER) * w,
                             rng.uniform(-JITTER, JITTER) * h] for _ in range(4)])
    a = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h),
                            borderMode=cv2.BORDER_REPLICATE)

    # Résolution réelle d'un crop, bien plus basse que celle du scan.
    k = rng.uniform(*SCALE)
    a = cv2.resize(a, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)

    if rng.random() < BLUR_PROB:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(*BLUR_SIGMA))

    a = a.astype(np.float32)
    a *= rng.uniform(*EXPOSURE)
    a *= np.float32([rng.uniform(*WHITE_BALANCE) for _ in range(3)])

    # Reflet holographique : nappe claire en diagonale, pas une tache ronde —
    # c'est ainsi que la lumière balaie une carte inclinée devant une fenêtre.
    if rng.random() < GLARE_PROB:
        hh, ww = a.shape[:2]
        yy, xx = np.mgrid[0:hh, 0:ww]
        d = (xx - rng.uniform(0, ww)) + (yy - rng.uniform(0, hh))
        nappe = np.exp(-(d ** 2) / (2 * (0.45 * ww) ** 2)).astype(np.float32)
        a += (nappe * rng.uniform(*GLARE_STRENGTH))[..., None]

    # Bruit de capteur, tiré d'un générateur numpy dérivé du même état : la
    # reproductibilité doit tenir sur les deux sources d'aléa.
    noise_rng = np.random.default_rng(rng.getrandbits(63))
    a += noise_rng.normal(0, rng.uniform(*NOISE_SIGMA), a.shape).astype(np.float32)
    a = np.clip(a, 0, 255).astype(np.uint8)

    _, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY,
                                      rng.randint(*JPEG_QUALITY)])
    a = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))


def view_rng(card_id: str, view: int, salt: int = 0) -> random.Random:
    """Générateur déterministe pour une vue donnée d'une carte donnée.

    `hash()` de Python est randomisé entre processus (PYTHONHASHSEED) : l'utiliser
    donnerait des images différentes à chaque exécution, et des workers de
    DataLoader incohérents entre eux. La graine est donc dérivée de la chaîne
    elle-même.
    """
    seed = int.from_bytes(f"{card_id}#{view}#{salt}".encode(), "little") % (2 ** 63)
    return random.Random(seed)
