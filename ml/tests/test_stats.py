"""Le socle statistique du banc : si ces valeurs dérivent, tous les chiffres publiés dérivent."""

from __future__ import annotations

import numpy as np
import pytest

from src.stats import fmt_ratio, risk_coverage, wilson


class TestWilson:
    def test_borne_basse_a_saturation(self):
        """31/31 ne vaut pas 100 % : c'est la raison d'être de ce module.

        L'intervalle normal rendrait [1.0, 1.0], ce qui est faux et invisible.
        Valeur de référence calculée à la main.
        """
        low, high = wilson(31, 31)
        assert low == pytest.approx(0.8897, abs=1e-4)
        assert high == 1.0

    def test_symetrie_a_un_demi(self):
        low, high = wilson(5, 10)
        assert low + high == pytest.approx(1.0, abs=1e-9)

    def test_zero_succes(self):
        low, high = wilson(0, 10)
        assert low == 0.0
        assert high == pytest.approx(0.2775, abs=1e-4)

    def test_echantillon_vide_ne_leve_pas(self):
        assert wilson(0, 0) == (0.0, 1.0)

    def test_intervalle_se_resserre_avec_n(self):
        largeurs = [wilson(n, n)[1] - wilson(n, n)[0] for n in (10, 30, 100, 1000)]
        assert largeurs == sorted(largeurs, reverse=True)

    def test_toujours_dans_zero_un(self):
        for succes, n in ((0, 1), (1, 1), (3, 7), (999, 1000)):
            low, high = wilson(succes, n)
            assert 0.0 <= low <= high <= 1.0

    def test_rend_des_flottants_natifs(self):
        """Le résumé du banc part en JSON : un np.float64 ferait échouer json.dumps."""
        low, high = wilson(26, 31)
        assert type(low) is float and type(high) is float


def test_fmt_ratio_porte_toujours_son_intervalle():
    texte = fmt_ratio(26, 31)
    assert "26/31" in texte and "83.9%" in texte and "IC95" in texte


class TestRiskCoverage:
    def test_tri_parfait_bat_tri_inverse(self):
        """L'AURC doit récompenser le classement, pas la précision brute.

        Les deux jeux ont la même précision globale ; seul l'ordre change.
        """
        correct = [True, True, False]
        parfait = risk_coverage([3, 2, 1], correct)
        inverse = risk_coverage([1, 2, 3], correct)
        assert parfait.aurc < inverse.aurc

    def test_aurc_nul_quand_tout_est_juste(self):
        rc = risk_coverage([1, 2, 3], [True, True, True])
        assert rc.aurc == 0.0

    def test_couverture_a_precision_cible(self):
        # marges décroissantes : les deux meilleures sont justes, la 3e est fausse
        rc = risk_coverage([0.9, 0.5, 0.1], [True, True, False])
        couverture, seuil = rc.coverage_at(1.0)
        assert couverture == pytest.approx(2 / 3)
        assert seuil == pytest.approx(0.5)

    def test_couverture_nulle_si_precision_inatteignable(self):
        rc = risk_coverage([0.9, 0.5], [False, True])
        couverture, seuil = rc.coverage_at(1.0)
        assert couverture == 0.0
        assert seuil == float("inf")

    def test_garde_le_point_le_plus_a_droite(self):
        """À précision égale, répondre plus souvent est toujours préférable.

        Ici la précision retombe à 100 % à pleine couverture après un creux :
        c'est la couverture maximale qui doit être rendue, pas la première
        rencontrée.
        """
        rc = risk_coverage([0.9, 0.8, 0.7], [True, True, True])
        assert rc.coverage_at(1.0)[0] == pytest.approx(1.0)

    def test_invariance_a_l_echelle(self):
        """Multiplier toutes les marges par 3,5 (ce que fait un centrage) ne
        doit rien changer : c'est la propriété qui rend deux espaces
        comparables, là où un seuil ne l'est pas."""
        correct = [True, False, True, True]
        scores = [0.10, 0.02, 0.30, 0.05]
        a = risk_coverage(scores, correct)
        b = risk_coverage([s * 3.5 for s in scores], correct)
        assert a.aurc == pytest.approx(b.aurc)
        np.testing.assert_allclose(a.risk, b.risk)

    def test_entree_vide(self):
        rc = risk_coverage([], [])
        assert np.isnan(rc.aurc)
