"""Recherche par similarité sur la banque d'embeddings.

Les vecteurs étant L2-normalisés, le produit scalaire *est* la similarité
cosinus. À 20 000 vecteurs de dimension 512, un produit matriciel numpy est
exact et prend moins d'une milliseconde : pas besoin de FAISS.

C'est un choix délibéré. FAISS entre en conflit OpenMP avec PyTorch sur macOS,
et surtout la version iOS fera exactement la même chose via Accelerate — garder
les deux implémentations identiques évite les écarts de résultats entre le
prototype et l'app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Hit:
    card_id: str
    score: float
    card: dict


class CardIndex:
    def __init__(self, model: str, cards_json: Path | None = None):
        emb_dir = ROOT / "data" / "embeddings" / model
        self.model = model
        self.embeddings: np.ndarray = np.load(emb_dir / "embeddings.npy")
        self.card_ids: list[str] = json.loads((emb_dir / "card_ids.json").read_text())
        self.meta: dict = json.loads((emb_dir / "meta.json").read_text())

        cards = json.loads((cards_json or ROOT / "data" / "processed" / "cards.json").read_text())
        self.cards_by_id = {c["id"]: c for c in cards}

        if len(self.card_ids) != self.embeddings.shape[0]:
            raise ValueError("card_ids et embeddings désalignés")

        self._id_to_row = {cid: i for i, cid in enumerate(self.card_ids)}

    def __len__(self) -> int:
        return len(self.card_ids)

    def search(self, queries: np.ndarray, k: int = 5) -> list[list[Hit]]:
        if queries.ndim == 1:
            queries = queries[None, :]

        sims = queries.astype(np.float32) @ self.embeddings.T

        # argpartition trouve les k meilleurs en O(N), on ne trie que ceux-là.
        k = min(k, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]

        results = []
        for row, idx in enumerate(top):
            idx = idx[np.argsort(-sims[row, idx])]
            results.append(
                [
                    Hit(self.card_ids[i], float(sims[row, i]), self.cards_by_id[self.card_ids[i]])
                    for i in idx
                ]
            )
        return results

    def embedding_of(self, card_id: str) -> np.ndarray:
        return self.embeddings[self._id_to_row[card_id]]
