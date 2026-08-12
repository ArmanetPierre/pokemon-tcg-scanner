"""Statistiques d'évaluation : intervalles de confiance et courbes risque/couverture.

Deux problèmes que le banc d'essai avait, et que ce module corrige.

**Un score sans intervalle n'est pas une mesure.** « 31/31 » se lit comme
100 %, alors que l'intervalle de Wilson à 95 % pour 31 succès sur 31 essais
descend à 89 %. Ce n'est pas de la modestie de façade : sur un échantillon de
cette taille, la précision réelle *peut* être de 90 % sans que le banc s'en
aperçoive. Wilson plutôt que l'intervalle normal parce que ce dernier donne
[100 % ; 100 %] quand p = 1, ce qui est faux et le fait de manière invisible.

**Un seuil n'est pas une calibration.** Le système ne répond pas toujours : il
peut dire « incertain ». La question utile n'est donc pas « quelle précision ? »
mais « quelle précision à quelle couverture ? ». La courbe risque/couverture
répond aux deux à la fois, et l'AURC la résume en un nombre comparable entre
deux variantes du système.

Piège associé, mesuré pendant l'audit : une marge n'est **jamais** comparable
d'un espace métrique à un autre (centrer l'index multiplie les marges par 3,5
sans rien améliorer). Comparer deux systèmes par leurs seuils n'a pas de sens ;
seules les courbes se comparent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 1,959964 = quantile normal à 97,5 %, soit un intervalle bilatéral à 95 %.
Z95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Intervalle de confiance de Wilson pour une proportion.

    Retourne (borne basse, borne haute) dans [0, 1]. Se comporte correctement
    aux extrêmes, contrairement à l'intervalle normal : wilson(31, 31) rend
    (0.890, 1.0) là où la formule normale rendrait (1.0, 1.0).
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def fmt_ratio(successes: int, n: int) -> str:
    """« 26/31 (83,9 %, IC95 67-93) » — format unique pour tout le banc."""
    if n == 0:
        return "0/0"
    low, high = wilson(successes, n)
    return (f"{successes}/{n} ({successes / n:.1%}, "
            f"IC95 {low:.0%}-{high:.0%})")


@dataclass
class RiskCoverage:
    """Courbe risque/couverture d'un score de confiance.

    `coverage[i]` / `risk[i]` : en ne répondant que sur les i+1 photos les plus
    confiantes, quelle fraction du banc est traitée et quelle fraction de ces
    réponses est fausse.
    """

    coverage: np.ndarray
    risk: np.ndarray
    aurc: float               # aire sous la courbe : 0 = parfait, plus bas = mieux
    thresholds: np.ndarray    # score de confiance au point de coupure

    def coverage_at(self, min_precision: float) -> tuple[float, float]:
        """Couverture maximale atteignable à une précision donnée.

        Retourne (couverture, seuil). La couverture est cherchée du plus
        permissif au plus strict et on garde le point le plus à droite qui
        respecte encore la précision : répondre plus souvent à précision égale
        est toujours préférable.
        """
        ok = self.risk <= 1 - min_precision
        if not ok.any():
            return (0.0, float("inf"))
        i = int(np.max(np.flatnonzero(ok)))
        return (float(self.coverage[i]), float(self.thresholds[i]))


def risk_coverage(scores, correct) -> RiskCoverage:
    """Trie par confiance décroissante et mesure le risque à chaque couverture.

    `scores` est n'importe quelle grandeur croissante avec la confiance (ici la
    marge 1er/2e). `correct` dit si la réponse était juste. Aucune hypothèse
    n'est faite sur l'échelle de `scores` — c'est tout l'intérêt : deux systèmes
    dont les marges vivent sur des échelles différentes restent comparables.
    """
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if scores.size == 0:
        empty = np.zeros(0)
        return RiskCoverage(empty, empty, float("nan"), empty)

    order = np.argsort(-scores)
    ranked = correct[order]
    n = len(ranked)

    errors = np.cumsum(~ranked)
    counts = np.arange(1, n + 1)
    risk = errors / counts
    coverage = counts / n
    # AURC : risque moyen sur tous les points de coupure. Un système qui trie
    # parfaitement (toutes les erreurs en dernier) minimise cette aire.
    return RiskCoverage(coverage, risk, float(risk.mean()), scores[order])


def render_risk_coverage(rc: RiskCoverage, points=(1.0, 0.75, 0.5, 0.25)) -> str:
    """Une ligne de tableau : précision à quelques couvertures remarquables."""
    if rc.coverage.size == 0:
        return "—"
    cells = []
    for cov in points:
        i = min(len(rc.risk) - 1, max(0, int(round(cov * len(rc.risk))) - 1))
        cells.append(f"{1 - rc.risk[i]:>7.1%}")
    return " ".join(cells)
