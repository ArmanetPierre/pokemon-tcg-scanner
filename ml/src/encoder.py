"""Abstraction sur les modèles d'embedding candidats.

Tous les encodeurs renvoient des vecteurs **L2-normalisés**, ce qui permet de
traiter la similarité cosinus comme un simple produit scalaire — côté Python
comme côté iOS (Accelerate).

Le champ `preprocess_spec` documente le prétraitement exact (taille, mode de
redimensionnement, moyenne/écart-type de normalisation). Il devra être reproduit
à l'identique en Swift : c'est la cause n°1 de divergence après conversion
Core ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class PreprocessSpec:
    """Prétraitement à répliquer côté Swift, au pixel près.

    `interpolation` est lu sur la transform réelle, jamais supposé : le modèle
    a été entraîné avec un filtre donné et en changer suffit à faire dériver
    l'embedding.
    """

    size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    resize_mode: str = "resize_shortest_then_center_crop"
    interpolation: str = "bilinear"
    antialias: bool = True


def geometric_preprocess(img: Image.Image, spec: PreprocessSpec) -> Image.Image:
    """Redimensionnement + recadrage seuls, sans normalisation.

    Sert au test de parité Core ML (où la normalisation est dans le graphe) et
    documente la géométrie exacte à reproduire en Swift. Attention : le
    recadrage centré retire le haut et le bas d'une carte portrait — c'est le
    comportement d'origine, l'index a été construit ainsi, et l'app doit faire
    pareil sous peine d'incohérence.
    """
    resample = {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "nearest": Image.NEAREST,
    }[spec.interpolation]

    width, height = img.size
    scale = spec.size / min(width, height)
    resized = img.resize((round(width * scale), round(height * scale)), resample)

    left = (resized.width - spec.size) // 2
    top = (resized.height - spec.size) // 2
    return resized.crop((left, top, left + spec.size, top + spec.size))


class _HFPreprocess:
    """Adaptateur processor HuggingFace -> transform façon torchvision.

    Une classe de module plutôt qu'une closure : les workers de DataLoader
    utilisent `spawn`, qui ne sait pas sérialiser une fonction locale.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.processor(img, return_tensors="pt")["pixel_values"][0]


@dataclass
class Encoder:
    name: str
    dim: int
    preprocess_spec: PreprocessSpec
    _model: torch.nn.Module = field(repr=False)
    _preprocess: object = field(repr=False)
    _device: str = field(repr=False)
    _kind: str = field(repr=False)

    @property
    def transform(self):
        """Prétraitement seul, sans référence au modèle.

        À passer aux workers d'un DataLoader : le redimensionnement PIL est
        mono-thread et sature sinon le CPU pendant que le GPU attend. Passer
        `preprocess_image` (méthode liée) sérialiserait l'encodeur entier, donc
        le modèle sur MPS — ce que `spawn` refuse.
        """
        return self._preprocess

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """Image PIL -> tenseur CHW prêt pour le modèle."""
        return self._preprocess(img.convert("RGB"))

    @torch.inference_mode()
    def encode_batch(self, tensors: torch.Tensor) -> np.ndarray:
        """Batch de tenseurs prétraités -> vecteurs L2-normalisés."""
        tensors = tensors.to(self._device)
        if self._kind == "open_clip":
            feats = self._model.encode_image(tensors)
        else:  # dinov2 via transformers : token CLS poolé
            feats = self._model(pixel_values=tensors).pooler_output

        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def encode(self, images: Sequence[Image.Image], batch_size: int = 64) -> np.ndarray:
        """Encode des images PIL en vecteurs L2-normalisés (float32)."""
        out: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = torch.stack([self.preprocess_image(img) for img in batch])
            out.append(self.encode_batch(tensors))

        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(out)


# Modèles candidats. MobileCLIP2 est la piste principale : conçu par Apple pour
# le Neural Engine, donc conversion Core ML sans mauvaise surprise. DINOv2 sert
# de comparaison (souvent meilleur en recherche d'instance pure), CLIP B/32 de
# référence basse.
REGISTRY = {
    "mobileclip2-s0": ("open_clip", "MobileCLIP2-S0", "dfndr2b"),
    "mobileclip2-s2": ("open_clip", "MobileCLIP2-S2", "dfndr2b"),
    "mobileclip2-s4": ("open_clip", "MobileCLIP2-S4", "dfndr2b"),
    "clip-b32": ("open_clip", "ViT-B-32", "openai"),
    "dinov2-s": ("hf", "facebook/dinov2-small", None),
    "dinov2-b": ("hf", "facebook/dinov2-base", None),
}


def load_encoder(key: str, device: str | None = None) -> Encoder:
    if key not in REGISTRY:
        raise KeyError(f"Modèle inconnu : {key}. Disponibles : {sorted(REGISTRY)}")

    kind, model_id, pretrained = REGISTRY[key]
    device = device or pick_device()

    if kind == "open_clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained
        )
        model = model.to(device).eval()

        cfg = open_clip.get_model_config(model_id)
        size = cfg["vision_cfg"]["image_size"]
        size = size if isinstance(size, int) else size[0]
        # open_clip range la normalisation dans le dernier transform de la chaîne.
        norm = preprocess.transforms[-1]
        resize = next(
            (t for t in preprocess.transforms if type(t).__name__ == "Resize"), None
        )
        interpolation = (
            resize.interpolation.value if resize is not None else "bilinear"
        )
        spec = PreprocessSpec(
            size=size,
            mean=tuple(norm.mean),
            std=tuple(norm.std),
            interpolation=interpolation,
        )

        with torch.inference_mode():
            dim = model.encode_image(torch.zeros(1, 3, size, size, device=device)).shape[-1]

    else:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()

        size = processor.crop_size["height"]
        spec = PreprocessSpec(
            size=size,
            mean=tuple(processor.image_mean),
            std=tuple(processor.image_std),
        )

        preprocess = _HFPreprocess(processor)

        with torch.inference_mode():
            dim = model(pixel_values=torch.zeros(1, 3, size, size, device=device)).pooler_output.shape[-1]

    return Encoder(
        name=key,
        dim=dim,
        preprocess_spec=spec,
        _model=model,
        _preprocess=preprocess,
        _device=device,
        _kind=kind,
    )
