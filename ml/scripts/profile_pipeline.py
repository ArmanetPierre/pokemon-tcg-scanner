"""Profile la chaîne d'identification étape par étape, sur les photos du banc.

    python scripts/profile_pipeline.py            # PyTorch
    python scripts/profile_pipeline.py --coreml   # modèle embarqué
    python scripts/profile_pipeline.py --json out.json

Répond à une seule question : **où part le temps sur une vraie photo**, et non
sur une identification théorique à un seul crop. C'est la distinction qui
compte, parce que l'orientation, le redressement et l'embedding sont payés une
fois par crop candidat — 13 à 15 fois sur une photo 12 Mpx (AGENTS.md §8).

Les étapes sont mesurées en enveloppant les fonctions réelles de `src/`, jamais
en réécrivant le pipeline : ce qui est chronométré est ce qui tourne. Le total
par étape n'a de sens qu'avec son **nombre d'appels** — c'est l'avertissement
explicite du kit d'intégration, et la raison de la colonne « appels ».

Le profileur vérifie aussi le top-1 contre `truth.json` : toute optimisation se
juge sur deux axes, les millisecondes gagnées et la justesse conservée.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.detect_vision  # noqa: E402
import src.edition  # noqa: E402
import src.orient  # noqa: E402
import src.pipeline  # noqa: E402
from src.encoder import REGISTRY, load_encoder  # noqa: E402
from src.pipeline import identify, load_bgr  # noqa: E402
from src.search import CardIndex  # noqa: E402

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
TRUTH = ML_ROOT / "data" / "eval" / "truth.json"


@dataclass
class Stage:
    """Cumul d'une étape : total, appels, et détail par photo."""

    label: str
    nested: bool          # contenue dans une autre étape mesurée -> hors total
    calls: int = 0
    total_ms: float = 0.0
    per_photo: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


class Profiler:
    """Registre d'étapes + enveloppe de chronométrage."""

    def __init__(self) -> None:
        self.stages: dict[str, Stage] = {}
        self.counters: dict[str, list[int]] = defaultdict(list)
        self._photo_start: dict[str, tuple[int, float]] = {}

    def stage(self, label: str, nested: bool = False) -> Stage:
        if label not in self.stages:
            self.stages[label] = Stage(label, nested)
        return self.stages[label]

    def wrap(self, module, name: str, label: str, nested: bool = False,
             count: str | None = None):
        """Remplace `module.name` par une version chronométrée.

        `count` : nom d'un compteur alimenté par la longueur du résultat (le
        nombre de quadrilatères ou de variantes, qui multiplie tout l'aval).
        """
        stage = self.stage(label, nested)
        original = getattr(module, name)

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return_value = original(*args, **kwargs)
            finally:
                stage.calls += 1
                stage.total_ms += (time.perf_counter() - start) * 1000.0
            if count is not None:
                try:
                    self.counters[count][-1] += len(return_value)
                except (TypeError, IndexError):
                    pass
            return return_value

        setattr(module, name, timed)
        return original

    def open_photo(self) -> None:
        """Ouvre une photo : mémorise l'état de chaque étape pour l'isoler."""
        self._photo_start = {
            label: (s.calls, s.total_ms) for label, s in self.stages.items()
        }
        for values in self.counters.values():
            values.append(0)

    def close_photo(self) -> dict[str, float]:
        """Ferme la photo et renvoie le coût par étape sur cette photo seule."""
        share = {}
        for label, stage in self.stages.items():
            _, before_ms = self._photo_start.get(label, (0, 0.0))
            spent = stage.total_ms - before_ms
            stage.per_photo.append(spent)
            share[label] = spent
        return share


def install(profiler: Profiler, encoder, index) -> None:
    """Instrumente les étapes réelles du pipeline.

    Ordre d'importance décroissant, tel qu'il apparaît dans une identification :
    décodage, détection, redressement, orientation, embedding, recherche,
    bandeau. Les étapes marquées `nested=True` sont incluses dans une autre et
    ne comptent pas deux fois dans le total.
    """
    # Étapes disjointes — leur somme fait le temps d'une identification.
    profiler.wrap(src.pipeline, "load_bgr", "décodage photo (OpenCV)")
    profiler.wrap(src.pipeline, "detect_document", "détection document (Vision)",
                  count="quads_doc")
    profiler.wrap(src.pipeline, "detect_rectangles", "détection rectangles (Vision)",
                  count="quads_rect")
    profiler.wrap(src.pipeline, "warp_card", "redressement (homographie)")
    profiler.wrap(src.pipeline, "pick_orientation", "orientation (dont OCR)")
    profiler.wrap(src.edition, "read_number_pairs", "OCR bandeau (Vision, accurate)")
    profiler.wrap(type(encoder), "encode", "embedding")
    profiler.wrap(type(index), "search", "recherche")

    # Étapes imbriquées — pour savoir ce qui, dans les précédentes, coûte.
    profiler.wrap(src.orient, "text_mass_center", "  ↳ OCR orientation (Vision, fast)",
                  nested=True)
    profiler.wrap(src.orient, "_cgimage_from_bgr", "  ↳ conversion BGR -> CGImage",
                  nested=True)
    profiler.wrap(src.detect_vision, "_load_cgimage", "  ↳ décodage JPEG pour Vision",
                  nested=True)
    profiler.wrap(src.pipeline, "build_variants", "  ↳ construction des variantes",
                  nested=True, count="variantes")


def render(profiler: Profiler, wall_ms: list[float], score: str, label: str) -> None:
    disjoint = [s for s in profiler.stages.values() if not s.nested]
    nested = [s for s in profiler.stages.values() if s.nested]
    measured = sum(s.total_ms for s in disjoint)
    wall = sum(wall_ms)
    photos = len(wall_ms)

    print(f"\n=== Profil par étape — {label}, {photos} photos ===\n")
    header = f"{'Étape':<36}{'appels':>8}{'ms/appel':>10}{'ms/photo':>10}{'% total':>9}"
    print(header)
    print("-" * len(header))
    for stage in sorted(disjoint, key=lambda s: -s.total_ms):
        print(
            f"{stage.label:<36}{stage.calls:>8}{stage.mean_ms:>10.1f}"
            f"{stage.total_ms / photos:>10.1f}{100 * stage.total_ms / wall:>8.1f}%"
        )
    rest = wall - measured
    print(f"{'reste (non instrumenté)':<36}{'':>8}{'':>10}{rest / photos:>10.1f}"
          f"{100 * rest / wall:>8.1f}%")
    print("-" * len(header))
    print(f"{'TOTAL par photo':<36}{'':>8}{'':>10}{wall / photos:>10.1f}{100.0:>8.1f}%")

    print("\nDétail des étapes imbriquées (déjà comptées ci-dessus) :\n")
    for stage in sorted(nested, key=lambda s: -s.total_ms):
        print(
            f"{stage.label:<36}{stage.calls:>8}{stage.mean_ms:>10.1f}"
            f"{stage.total_ms / photos:>10.1f}{100 * stage.total_ms / wall:>8.1f}%"
        )

    print("\nCe qui multiplie les étapes par crop :\n")
    for name, values in profiler.counters.items():
        if not values:
            continue
        print(f"  {name:<24} total {sum(values):>4}   "
              f"moyenne {sum(values) / len(values):>5.1f}   max {max(values):>3}")

    ocr = sum(s.total_ms for s in profiler.stages.values()
              if "OCR" in s.label and not s.nested)
    print(f"\nPart des deux OCR Vision : {100 * ocr / wall:.0f} % du temps total.")
    print(f"Top-1 exact : {score}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobileclip2-s2", choices=sorted(REGISTRY))
    parser.add_argument("--coreml", action="store_true",
                        help="encodeur Core ML embarqué plutôt que PyTorch")
    parser.add_argument("--limit", type=int, help="ne profiler que les N premières photos")
    parser.add_argument("--json", type=Path, help="écrit le détail chiffré dans ce fichier")
    args = parser.parse_args()

    if args.coreml:
        from src.coreml_encoder import CoreMLEncoder

        encoder = CoreMLEncoder()
    else:
        encoder = load_encoder(args.model)
    index = CardIndex(args.model)

    truth = json.loads(TRUTH.read_text())["photos"]
    entries = [(p, e) for p, e in truth.items() if (PROJECT_ROOT / p).exists()]
    if args.limit:
        entries = entries[: args.limit]

    # Préchauffage : le premier appel à Vision et à Core ML charge des modèles,
    # coût unique qui n'appartient à aucune étape.
    warm_path, _ = entries[0]
    identify(str(PROJECT_ROOT / warm_path), load_bgr(str(PROJECT_ROOT / warm_path)),
             encoder, index)

    profiler = Profiler()
    install(profiler, encoder, index)

    wall_ms: list[float] = []
    correct = identifiable = 0
    per_photo_rows = []
    for rel_path, expected in entries:
        path = str(PROJECT_ROOT / rel_path)
        profiler.open_photo()
        start = time.perf_counter()
        bgr = src.pipeline.load_bgr(path)
        result = identify(path, bgr, encoder, index)
        elapsed = (time.perf_counter() - start) * 1000.0
        share = profiler.close_photo()
        wall_ms.append(elapsed)

        # card_id null : carte absente de l'index, non identifiable par
        # construction. Elle est profilée (son coût est réel) mais exclue du
        # score, sans quoi le banc afficherait un plafond à 18/21.
        hit = result.card_id == expected["card_id"]
        if expected["card_id"] is None:
            verdict = "hors index (attendu)"
        else:
            identifiable += 1
            correct += hit
            verdict = "ok" if hit else "ÉCHEC"
        megapixels = bgr.shape[0] * bgr.shape[1] / 1e6
        print(f"{Path(rel_path).name:<16} {elapsed:>7.0f} ms  {megapixels:>4.1f} Mpx  "
              f"{verdict:<21}[{result.crop_label}]")
        per_photo_rows.append({
            "photo": rel_path,
            "megapixels": round(megapixels, 1),
            "total_ms": round(elapsed, 1),
            "identifiable": expected["card_id"] is not None,
            "correct": bool(hit),
            "stages_ms": {k: round(v, 1) for k, v in share.items()},
        })

    label = "Core ML" if args.coreml else f"PyTorch {args.model}"
    score = f"{correct}/{identifiable} ({len(entries) - identifiable} hors index)"
    render(profiler, wall_ms, score, label)

    if args.json:
        args.json.write_text(json.dumps({
            "encodeur": label,
            "photos": len(entries),
            "top1": score,
            "ms_par_photo": round(sum(wall_ms) / len(wall_ms), 1),
            "etapes": {
                s.label.strip(): {
                    "appels": s.calls,
                    "ms_par_appel": round(s.mean_ms, 2),
                    "ms_par_photo": round(s.total_ms / len(wall_ms), 1),
                    "imbriquee": s.nested,
                }
                for s in profiler.stages.values()
            },
            "compteurs": {k: v for k, v in profiler.counters.items()},
            "detail": per_photo_rows,
        }, ensure_ascii=False, indent=2))
        print(f"\nDétail écrit dans {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
