"""Écrit l'index d'embeddings et les métadonnées au format attendu par l'app.

    python scripts/export_index.py

Produit dans `data/export/` :
  index.bin    embeddings float16 contigus, N x dim, sans en-tête
  index.json   forme et ordre des cartes, pour lire index.bin sans deviner
  cards.json   métadonnées d'affichage, allégées

Le float16 divise la taille par deux et se lit directement en Swift : la
recherche est un produit matriciel via Accelerate, pas une bibliothèque
vectorielle. À 20 000 cartes, c'est exact et instantané — l'index Python fait
exactement la même chose, ce qui garantit des résultats identiques entre le
banc d'essai et l'app.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline  # noqa: E402
from src.fingerprint import fingerprint, short  # noqa: E402
from src.search import CardIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "export"
LABELS = ROOT / "data" / "processed" / "discriminability.json"
SOURCE = ROOT / "data" / "processed" / "source.json"

# Champs nécessaires à l'affichage et au départage par numéro (chantier D).
# Le reste (attaques, texte, légalité) n'a rien à faire dans le bundle.
CARD_FIELDS = (
    "id", "name", "number", "rarity",
    "set_id", "set_name", "set_printed_total",
    "image_small",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobileclip2-s2")
    parser.add_argument("--embeddings", type=Path,
                        help="enrôlement alternatif (ex. centroïde multi-vues)")
    args = parser.parse_args()

    index = CardIndex(args.model)
    embeddings = index.embeddings.astype(np.float32)

    # Enrôlement multi-vues (scripts/build_multiview_index.py). Le format ne
    # change pas — même forme, même ordre de lignes, même recherche — seul le
    # contenu des vecteurs diffère. C'est ce qui permet de changer de stratégie
    # d'enrôlement sans toucher au portage Swift.
    if args.embeddings:
        embeddings = np.load(args.embeddings).astype(np.float32)
        if embeddings.shape != index.embeddings.shape:
            print(f"forme {embeddings.shape} incompatible avec l'index "
                  f"{index.embeddings.shape}", file=sys.stderr)
            return 1
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        print(f"enrôlement : {args.embeddings.name}")
    print(f"index source : {embeddings.shape} ({embeddings.nbytes / 1e6:.0f} Mo en float32)")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    half = embeddings.astype(np.float16)
    (EXPORT_DIR / "index.bin").write_bytes(half.tobytes())

    # Provenance des métadonnées (scripts/build_metadata.py). Avec l'empreinte
    # de l'encodeur ci-dessous, le kit livré dit d'où viennent ses deux moitiés :
    # quel modèle a produit les vecteurs, et quelle révision de la source a
    # produit les cartes.
    source = json.loads(SOURCE.read_text()) if SOURCE.exists() else None
    if source is None:
        print("provenance des métadonnées absente : relancer build_metadata.py "
              "pour l'inscrire dans l'index.")
    elif not source.get("matches_pin"):
        print(f"ATTENTION : métadonnées construites hors épingle "
              f"({source['revision'][:12]}).")

    # Empreinte de l'encodeur qui a produit ces vecteurs. Un index interrogé par
    # un autre encodeur ne lève aucune erreur : la recherche rend des voisins,
    # les scores restent dans leur plage habituelle, et les réponses sont
    # plausibles et fausses. L'app doit refuser de démarrer sur un désaccord.
    package = EXPORT_DIR / "CardEncoder.mlpackage"
    encoder_sha = fingerprint(package) if package.exists() else None
    if encoder_sha is None:
        print("ATTENTION : CardEncoder.mlpackage absent, index sans empreinte "
              "d'encodeur — exporter le modèle d'abord (export_coreml.py).")

    (EXPORT_DIR / "index.json").write_text(json.dumps({
        "model": index.meta["model"],
        "count": int(half.shape[0]),
        "dim": int(half.shape[1]),
        "dtype": "float16",
        "layout": "row_major",
        "normalized": True,
        "note": "vecteurs L2-normalisés : la similarité cosinus est un simple "
                "produit scalaire",
        "encoder_sha256": encoder_sha,
        "data_source": source,
        # Les seuils voyagent AVEC l'index, et ne sont pas recopiés à la main
        # côté app. Ils dépendent de l'enrôlement : le centroïde et l'index de
        # référence n'ont pas la même échelle de marges, et une app qui
        # garderait les anciennes valeurs face à un nouvel index affirmerait
        # à tort — sans qu'aucune erreur ne soit levée. Même raisonnement que
        # l'empreinte de l'encodeur ci-dessus.
        "confidence": {
            "firm_id_margin": pipeline.FIRM_ID_MARGIN,
            "firm_name_margin": pipeline.FIRM_NAME_MARGIN,
            "no_firm_verdict_on_full_photo": True,
            "note": "marges calculées AVANT toute promotion par le numéro lu. "
                    "Le repli photo-entière ne peut produire aucun verdict "
                    "ferme : aucune identification correcte n'en est jamais "
                    "sortie sur le banc, et un faux positif ferme si.",
        },
        "encoder_note": "empreinte de CardEncoder.mlpackage. L'app doit la "
                        "comparer au modèle qu'elle embarque et refuser de "
                        "démarrer si elle diffère : un index et un encodeur "
                        "désaccordés ne produisent pas d'erreur, seulement de "
                        "fausses réponses confiantes.",
        "card_ids": index.card_ids,
    }))

    # Étiquette de discriminabilité (scripts/label_discriminability.py) : dit à
    # l'app, AVANT de répondre, si l'image seule peut trancher pour cette carte
    # ou s'il faut aller lire le numéro imprimé. Optionnelle — l'export reste
    # possible sans, mais l'app perd cette information.
    labels = {}
    if LABELS.exists():
        payload = json.loads(LABELS.read_text())
        if payload.get("model") != index.meta["model"]:
            print(f"ATTENTION : étiquettes calculées pour {payload.get('model')}, "
                  f"index en {index.meta['model']} — ignorées.")
        else:
            labels = payload["labels"]
    else:
        print("étiquettes de discriminabilité absentes : "
              "lancer scripts/label_discriminability.py pour les produire.")

    cards = []
    for card_id in index.card_ids:
        card = {k: index.cards_by_id[card_id].get(k) for k in CARD_FIELDS}
        if card_id in labels:
            card["needs_printed_number"] = labels[card_id]["needs_printed_number"]
        cards.append(card)
    (EXPORT_DIR / "cards.json").write_text(json.dumps(cards, ensure_ascii=False))

    if labels:
        flagged = sum(c.get("needs_printed_number", False) for c in cards)
        print(f"  {flagged} cartes sur {len(cards)} ({flagged / len(cards):.1%}) "
              f"marquées « numéro imprimé requis »")

    # Vérifier que la perte de précision ne réordonne aucun voisinage : on
    # rejoue une recherche sur un échantillon et on compare les top-5.
    rng = np.random.default_rng(0)
    sample = rng.choice(half.shape[0], size=200, replace=False)
    reference = embeddings[sample] @ embeddings.T
    degraded = half[sample].astype(np.float32) @ half.astype(np.float32).T
    top_ref = np.argsort(-reference, axis=1)[:, :5]
    top_deg = np.argsort(-degraded, axis=1)[:, :5]
    identical = int(np.sum(np.all(top_ref == top_deg, axis=1)))
    top1_same = int(np.sum(top_ref[:, 0] == top_deg[:, 0]))

    for name in ("index.bin", "index.json", "cards.json"):
        size = (EXPORT_DIR / name).stat().st_size
        print(f"  {name:<12} {size / 1e6:7.1f} Mo")
    total = sum((EXPORT_DIR / n).stat().st_size for n in ("index.bin", "index.json", "cards.json"))
    print(f"  {'total':<12} {total / 1e6:7.1f} Mo")
    if encoder_sha:
        print(f"\nencodeur : {short(encoder_sha)}  (à retrouver dans encoder_meta.json)")

    print(f"\nfloat32 -> float16 sur {len(sample)} requêtes :")
    print(f"  top-1 identique : {top1_same}/{len(sample)}")
    print(f"  top-5 identique : {identical}/{len(sample)}")
    if top1_same < len(sample):
        print("  (des réordonnancements en tête : vérifier avant d'embarquer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
