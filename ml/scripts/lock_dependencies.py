"""Écrit le verrou de dépendances : ce qui a réellement tourné.

    .venv/bin/python scripts/lock_dependencies.py > requirements.lock.txt

`pyproject.toml` ne porte que des bornes basses : il dit ce qui est *compatible*,
pas ce qui a produit les mesures. Or l'index est l'artefact central du projet et
une mesure ne se compare à une autre que si l'environnement se compare aussi —
une montée de version de torch ou d'open_clip peut déplacer les embeddings sans
rien casser de visible.

Fermeture transitive des dépendances **déclarées**, et non `pip freeze` de
l'environnement de travail : celui-ci contient aussi les paquets d'exploration
(Embedding Atlas, pandas, umap) qui n'ont aucun rôle dans la chaîne
d'identification, et les faire passer pour nécessaires induirait en erreur.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, requires, version

from packaging.requirements import Requirement

# Racines : les dépendances de `[project]` plus les extras `coreml` et `dev`.
# L'extra `atlas` est volontairement exclu — il sert à explorer l'index, pas à
# le construire ni à le mesurer.
ROOTS = [
    "torch", "torchvision", "open_clip_torch", "transformers", "numpy",
    "pillow", "requests", "opencv-python-headless", "pillow-heif",
    "pyobjc-framework-Vision", "pyobjc-framework-Quartz",
    "coremltools", "pytest", "ruff",
]

HEADER = """\
# Verrou de dépendances — l'environnement exact qui a produit les mesures
# publiées : index de 20 512 cartes, banc réel 31/31, banc synthétique 95,2 %.
#
# Généré par scripts/lock_dependencies.py, à régénérer après toute montée de
# version délibérée. Portée : fermeture transitive des dépendances DÉCLARÉES,
# pas un pip freeze de l'environnement de travail.
#
#     python3.11 -m venv .venv
#     .venv/bin/pip install -r requirements.lock.txt
#     .venv/bin/pip install -e . --no-deps
#
# macOS, Python {python}. Les paquets pyobjc-* n'existent que sur Darwin ;
# ailleurs, seuls les tests d'invariants et src/stats.py restent utilisables.
"""


def closure(roots: list[str]) -> dict[str, str]:
    """Paquets atteignables depuis `roots`, avec leur version installée.

    Les dépendances conditionnelles sont évaluées dans l'environnement courant :
    un paquet requis seulement sous Windows, ou seulement pour un extra qu'on
    n'installe pas, n'a rien à faire dans le verrou.
    """
    found: dict[str, str] = {}

    def walk(name: str, depth: int = 0) -> None:
        key = name.lower().replace("_", "-")
        if key in found or depth > 8:
            return
        try:
            found[key] = version(name)
        except PackageNotFoundError:
            print(f"absent de l'environnement, ignoré : {name}", file=sys.stderr)
            return
        for raw in requires(name) or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            walk(requirement.name, depth + 1)

    for root in roots:
        walk(root)
    return found


def main() -> int:
    packages = closure(ROOTS)
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(HEADER.format(python=python))
    for name, ver in sorted(packages.items()):
        print(f"{name}=={ver}")
    print(f"\n{len(packages)} paquets", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
