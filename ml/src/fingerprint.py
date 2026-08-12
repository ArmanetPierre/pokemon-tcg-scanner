"""Empreintes des artefacts livrés : encodeur, index, métadonnées.

Le kit d'intégration livre trois fichiers qui doivent avoir été produits par le
**même** encodeur : `CardEncoder.mlpackage`, `index.bin` et `cards.json`. Rien,
aujourd'hui, ne l'impose. `index.json` porte le nom du modèle — une chaîne que
deux exports différents partagent forcément.

Or c'est le pire type de désaccord possible pour ce projet : un index construit
avec un encodeur et interrogé par un autre ne lève aucune erreur. La recherche
rend des voisins, les scores restent dans la plage habituelle, et les réponses
sont plausibles et fausses. C'est exactement la classe de bug que le kit
documente ailleurs avec soin (géométrie du prétraitement, rotation EXIF), et la
seule qui n'était pas outillée.

Une empreinte par artefact, vérifiée au chargement côté app, transforme cette
panne silencieuse en refus de démarrage explicite.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1 << 20


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_tree(root: Path) -> str:
    """Empreinte d'un répertoire, indépendante de l'ordre du système de fichiers.

    `CardEncoder.mlpackage` est un répertoire. Concaténer les fichiers dans
    l'ordre rendu par le système donnerait une empreinte différente d'une
    machine à l'autre ; le chemin relatif est donc trié, et intégré au condensat
    pour qu'un simple renommage change aussi l'empreinte.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(sha256_of_file(path).encode())
    return digest.hexdigest()


def fingerprint(path: Path) -> str:
    """Empreinte d'un fichier ou d'un répertoire, préfixée de son algorithme."""
    if not path.exists():
        raise FileNotFoundError(path)
    value = sha256_of_tree(path) if path.is_dir() else sha256_of_file(path)
    return f"sha256:{value}"


def short(value: str) -> str:
    """12 caractères : assez pour comparer à l'œil dans un journal de build."""
    return value.split(":", 1)[-1][:12]
