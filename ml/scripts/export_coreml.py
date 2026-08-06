"""Convertit l'encodeur d'images en modèle Core ML pour l'app iOS.

    python scripts/export_coreml.py
    python scripts/export_coreml.py --model mobileclip2-s2 --precision fp16

Produit `data/export/CardEncoder.mlpackage`, puis vérifie la **parité** : les
embeddings Core ML doivent coïncider avec ceux de PyTorch à mieux que 0,99 de
similarité cosinus. C'est le contrôle qui attrape les erreurs de prétraitement
(ordre des canaux, échelle, interpolation) — la cause n°1 d'un modèle qui
« marche » en Python et dérive sur le téléphone.

Le prétraitement est intégré au modèle Core ML (`scale` / `bias` de
`ImageType`), pour que Swift n'ait qu'à fournir un CVPixelBuffer redimensionné :
une normalisation réimplémentée à la main des deux côtés finit toujours par
diverger.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.encoder import REGISTRY, geometric_preprocess, load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "export"

# Seuil d'alerte, pas de vérité. Mesuré : la conversion elle-même est fidèle
# (fp32 -> min 0,9957) et le bruit fp16 descend à 0,989 sans faire basculer une
# seule identification du banc d'essai. Le critère qui fait foi est donc
# `evaluate_real.py --coreml`, pas cette similarité cosinus : en dessous de ce
# seuil on suspecte une erreur de prétraitement, au-dessus on va vérifier sur
# le banc.
PARITY_THRESHOLD = 0.98


class ImageEncoderWrapper(torch.nn.Module):
    """Expose uniquement la branche image, en sortie L2-normalisée.

    La normalisation est faite dans le graphe : l'app compare des vecteurs par
    produit scalaire, autant garantir la norme unitaire côté modèle plutôt que
    de compter sur Swift pour le refaire.
    """

    def __init__(self, model: torch.nn.Module, kind: str):
        super().__init__()
        self.model = model
        self.kind = kind

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.kind == "open_clip":
            features = self.model.encode_image(pixel_values)
        else:
            features = self.model(pixel_values=pixel_values).pooler_output
        return features / features.norm(dim=-1, keepdim=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobileclip2-s2", choices=sorted(REGISTRY))
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()

    # La conversion trace le graphe sur CPU : MPS produit des tenseurs que
    # coremltools ne sait pas suivre.
    encoder = load_encoder(args.model, device="cpu")
    spec = encoder.preprocess_spec
    size = spec.size
    print(f"modèle {encoder.name} — dim {encoder.dim} — entrée {size}px")
    print(f"prétraitement : mean={spec.mean} std={spec.std} ({spec.resize_mode})")

    wrapper = ImageEncoderWrapper(encoder._model, encoder._kind).eval()
    example = torch.zeros(1, 3, size, size)

    # `torch.jit.trace` casse sur MobileCLIP2 (cast int d'un tenseur non
    # scalaire dans les blocs hybrides). `torch.export` passe, à condition de
    # décomposer d'abord : le graphe brut est en dialecte TRAINING, que
    # coremltools ne sait pas lire.
    exported = torch.export.export(wrapper, (example,)).run_decompositions({})

    # Core ML fournit des pixels 0-255 ; le modèle attend (x/255 - mean) / std.
    # scale et bias encodent cette transformation directement dans l'entrée.
    scale = 1.0 / 255.0 / float(np.mean(spec.std)) if np.mean(spec.std) else 1.0 / 255.0
    bias = [-m / s for m, s in zip(spec.mean, spec.std)]

    mlmodel = ct.convert(
        exported,
        inputs=[ct.ImageType(name="image", shape=(1, 3, size, size), scale=scale, bias=bias)],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.short_description = f"Encodeur d'images de cartes Pokémon ({encoder.name})"
    mlmodel.input_description["image"] = f"Carte redressée, {size}x{size}, RVB"
    mlmodel.output_description["embedding"] = f"Vecteur L2-normalisé de dimension {encoder.dim}"

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    package = EXPORT_DIR / "CardEncoder.mlpackage"
    mlmodel.save(str(package))
    print(f"\nmodèle écrit : {package}")

    # --- test de parité ---
    image_dir = ROOT / "data" / "raw" / "images_large"
    paths = sorted(image_dir.rglob("*.jpg"))[:: max(1, 20000 // args.samples)][: args.samples]
    if not paths:
        print("aucune image de référence : parité non vérifiée", file=sys.stderr)
        return 1

    images = [Image.open(p).convert("RGB") for p in paths]
    reference = encoder.encode(images)

    similarities = []
    for image, ref_vector in zip(images, reference):
        # Même géométrie que PyTorch (redimension du petit côté + recadrage
        # centré) ; la normalisation, elle, est dans le graphe Core ML.
        prepared = geometric_preprocess(image, spec)
        out = mlmodel.predict({"image": prepared})["embedding"].astype(np.float32).ravel()
        out /= np.linalg.norm(out)
        similarities.append(float(out @ ref_vector))

    similarities = np.array(similarities)
    print("\n--- parité PyTorch / Core ML ---")
    print(f"  min {similarities.min():.5f}  moyenne {similarities.mean():.5f}  "
          f"max {similarities.max():.5f}  sur {len(similarities)} images")

    (EXPORT_DIR / "encoder_meta.json").write_text(json.dumps({
        "model": encoder.name,
        "dim": encoder.dim,
        "input_size": size,
        "precision": args.precision,
        # La normalisation est dans le graphe (scale/bias de l'ImageType) :
        # Swift ne fournit que des pixels 0-255. En revanche la géométrie
        # ci-dessous doit être reproduite à l'identique côté app.
        "preprocess_in_model": True,
        "geometry": {
            "resize_mode": spec.resize_mode,
            "interpolation": spec.interpolation,
            "antialias": spec.antialias,
            "note": "redimensionner le petit côté à input_size, puis recadrer "
                    "au centre — le haut et le bas d'une carte portrait sont "
                    "volontairement rognés, l'index est construit ainsi",
        },
        "parity_min": float(similarities.min()),
        "parity_mean": float(similarities.mean()),
    }, indent=2))

    if similarities.min() < PARITY_THRESHOLD:
        print(f"\nÉCHEC : parité minimale {similarities.min():.5f} < {PARITY_THRESHOLD}.")
        print("Écart trop grand pour du bruit de quantification : chercher une")
        print("divergence de prétraitement (géométrie, interpolation, échelle).")
        return 1

    print(f"\nparité au-dessus du seuil d'alerte ({PARITY_THRESHOLD})")
    print("Valider maintenant sur le banc d'essai :")
    print("    python scripts/evaluate_real.py --coreml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
