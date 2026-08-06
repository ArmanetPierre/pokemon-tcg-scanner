"""Encodeur Core ML, interface compatible avec `src.encoder.Encoder`.

Permet de faire tourner le banc d'essai avec le modèle réellement embarqué
plutôt qu'avec PyTorch. C'est la seule mesure qui compte pour décider si la
quantification fp16 est acceptable : une parité cosinus de 0,989 ne dit rien
tant qu'on n'a pas vérifié qu'aucune identification ne bascule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import coremltools as ct
import numpy as np
from PIL import Image

from src.encoder import PreprocessSpec, geometric_preprocess

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "export"


class CoreMLEncoder:
    """Même contrat que `Encoder` : `.encode(images) -> vecteurs L2-normalisés`."""

    def __init__(self, package: Path | None = None):
        package = package or EXPORT_DIR / "CardEncoder.mlpackage"
        meta = json.loads((EXPORT_DIR / "encoder_meta.json").read_text())

        self.model = ct.models.MLModel(str(package))
        self.name = f"{meta['model']}-coreml-{meta['precision']}"
        self.dim = meta["dim"]
        geometry = meta.get("geometry", {})
        self.preprocess_spec = PreprocessSpec(
            size=meta["input_size"],
            mean=(0.0, 0.0, 0.0),  # intégrée au graphe Core ML
            std=(1.0, 1.0, 1.0),
            interpolation=geometry.get("interpolation", "bilinear"),
        )

    def encode(self, images: Sequence[Image.Image], batch_size: int = 1) -> np.ndarray:
        del batch_size  # Core ML prédit image par image ici
        out = np.empty((len(images), self.dim), dtype=np.float32)
        for i, image in enumerate(images):
            prepared = geometric_preprocess(image.convert("RGB"), self.preprocess_spec)
            vector = self.model.predict({"image": prepared})["embedding"]
            vector = np.asarray(vector, dtype=np.float32).ravel()
            out[i] = vector / np.linalg.norm(vector)
        return out
