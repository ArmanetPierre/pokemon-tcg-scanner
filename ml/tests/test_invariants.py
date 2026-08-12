"""Les pièges d'AGENTS.md §3, transformés en tests.

Chacun de ces invariants a coûté du temps à trouver, et chacun échoue
*silencieusement* : pas de crash, juste des réponses plausibles et fausses.
Ils étaient documentés ; rien ne les empêchait de régresser.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.edition import _digit_distance, known_pair, match_edition
from src.encoder import PreprocessSpec, geometric_preprocess
from src.pipeline import classify_confidence, selection_score


class _Hit:
    """Substitut de `src.search.Hit`, sans avoir à charger l'index de 42 Mo."""

    def __init__(self, card_id: str, score: float, name: str = "X",
                 number: str = "1", total: int | None = 100):
        self.card_id = card_id
        self.score = score
        self.card = {"name": name, "number": number, "set_printed_total": total}


class TestGeometriePretraitement:
    """« La géométrie doit être identique au pixel près » — parité mesurée à 0,65 sinon."""

    def test_petit_cote_puis_recadrage_centre(self):
        spec = PreprocessSpec(size=256, mean=(0, 0, 0), std=(1, 1, 1))
        out = geometric_preprocess(Image.new("RGB", (734, 1024)), spec)
        assert out.size == (256, 256)

    def test_recadrage_rogne_le_haut_et_le_bas_d_une_carte_portrait(self):
        """Comportement volontaire : l'index est construit ainsi.

        Une bande horizontale unique placée en haut de la carte doit DISPARAÎTRE
        du recadrage centré. Si un jour le recadrage cessait de rogner, ce test
        tombe — et c'est une rupture de compatibilité avec tout l'index.
        """
        img = Image.new("RGB", (700, 1000), (0, 0, 0))
        img.paste(Image.new("RGB", (700, 100), (255, 0, 0)), (0, 0))
        spec = PreprocessSpec(size=256, mean=(0, 0, 0), std=(1, 1, 1))
        out = np.array(geometric_preprocess(img, spec))
        assert out[:, :, 0].max() == 0, "le haut de la carte n'a pas été rogné"

    def test_ecrasement_direct_donne_une_autre_image(self):
        """Le raccourci qui fait chuter la parité à 0,65, matérialisé."""
        img = Image.radial_gradient("L").convert("RGB").resize((700, 1000))
        spec = PreprocessSpec(size=256, mean=(0, 0, 0), std=(1, 1, 1))
        correct = np.array(geometric_preprocess(img, spec), dtype=np.int16)
        ecrase = np.array(img.resize((256, 256), Image.BILINEAR), dtype=np.int16)
        assert np.abs(correct - ecrase).mean() > 5


class TestDistanceChiffres:
    def test_zeros_de_tete_normalises(self):
        assert _digit_distance("043", "43") == 0

    def test_une_substitution(self):
        assert _digit_distance("152", "132") == 1

    def test_longueurs_differentes_invalides(self):
        """« 43 » et « 143 » ne sont pas à une erreur d'OCR l'un de l'autre."""
        assert _digit_distance("143", "43") >= 99

    def test_erreur_sur_le_zero_de_tete_non_rattrapable(self):
        """Limite connue, et volontairement du côté prudent.

        Les zéros de tête sont retirés AVANT la comparaison : une lecture
        « 143 » pour un « 043 » imprimé devient donc 143 contre 43, deux
        longueurs différentes, donc pas de correspondance — alors que c'est bien
        une substitution d'un chiffre. Le système refuse au lieu de risquer une
        fausse attribution ; c'est le bon signe d'erreur, mais cela coûte du
        rappel sur les sets dont les numéros sont imprimés sur trois chiffres.
        """
        assert _digit_distance("143", "043") >= 99

    def test_zero_et_chaine_de_zeros(self):
        assert _digit_distance("000", "0") == 0


class TestMatchEdition:
    def test_le_total_doit_correspondre_exactement(self):
        """Le total identifie le set : le tolérer casserait l'attribution d'édition."""
        hits = [_Hit("a", 0.9, number="43", total=84)]
        assert match_edition(hits, [("43", "85")]) is None
        assert match_edition(hits, [("43", "84")]) == (0, 0)

    def test_une_erreur_toleree_sur_le_numero(self):
        hits = [_Hit("a", 0.9, number="43", total=84)]
        assert match_edition(hits, [("48", "84")]) == (0, 1)
        assert match_edition(hits, [("58", "84")]) is None

    def test_prefere_la_lecture_sans_erreur(self):
        hits = [_Hit("a", 0.9, number="48", total=84), _Hit("b", 0.8, number="43", total=84)]
        assert match_edition(hits, [("43", "84")]) == (1, 0)

    def test_numero_non_numerique_ignore(self):
        """Les promos ont des numéros comme « XY142 » : ne pas planter dessus."""
        hits = [_Hit("a", 0.9, number="XY142", total=None)]
        assert match_edition(hits, [("43", "84")]) is None


class TestConfiance:
    """« Calculer les marges AVANT réordonnancement » — sinon elles sont négatives."""

    def test_marge_edition_franche(self):
        hits = [_Hit("a", 0.90, name="Pika"), _Hit("b", 0.80, name="Rai")]
        c = classify_confidence(hits)
        assert c.level == "edition" and c.source == "similarité"

    def test_marge_serree_mais_nom_homogene_donne_le_nom(self):
        """« Bonne carte, édition incertaine » n'est pas « je ne sais pas »."""
        hits = [_Hit("a", 0.90, name="Pika"), _Hit("b", 0.895, name="Pika"),
                _Hit("c", 0.80, name="Rai")]
        c = classify_confidence(hits)
        assert c.level == "nom"
        assert c.margin_id < 0.03 <= c.margin_name

    def test_tout_serre_donne_incertain(self):
        hits = [_Hit("a", 0.90, name="Pika"), _Hit("b", 0.895, name="Rai")]
        assert classify_confidence(hits).level == "incertain"

    def test_numero_lu_prime_sur_les_marges(self):
        hits = [_Hit("a", 0.90, name="Pika"), _Hit("b", 0.8999, name="Rai")]
        c = classify_confidence(hits, band_verdict="ocr")
        assert c.level == "edition" and c.source == "numéro lu"

    def test_top_k_homogene_en_nom(self):
        """Sans candidat d'un autre nom, la marge de nom vaut au moins celle d'édition."""
        hits = [_Hit("a", 0.90, name="Pika"), _Hit("b", 0.60, name="Pika")]
        c = classify_confidence(hits)
        assert c.margin_name == pytest.approx(c.margin_id)

    def test_selection_score_recompense_la_marge(self):
        """Score seul : piégé par une carte à l'envers. Marge seule : piégée par
        un crop plat. La somme des deux est ce qui a été mesuré meilleur."""
        serre = [_Hit("a", 0.88), _Hit("b", 0.87)]
        franc = [_Hit("c", 0.85), _Hit("d", 0.70)]
        assert selection_score(franc) > selection_score(serre)


class TestHorsIndex:
    def test_couple_connu(self):
        index = type("I", (), {"cards_by_id": {
            "x": {"number": "043", "set_printed_total": 84}}})()
        assert known_pair(index, [("43", "84")]) is True

    def test_couple_inconnu(self):
        """Le verdict « hors index » repose entièrement sur cette fonction."""
        index = type("I", (), {"cards_by_id": {
            "x": {"number": "043", "set_printed_total": 84}}})()
        assert known_pair(index, [("99", "84")]) is False


class TestLectureBandeau:
    def test_confusions_ocr_corrigees(self):
        """« O43/O84 » doit se lire 43/84 : l'espace des valeurs valides est fermé."""
        band = np.zeros((100, 400, 3), np.uint8)
        # read_number_pairs appelle Vision ; on teste ici la table de confusion
        # seule, via le même chemin de normalisation.
        from src.edition import CONFUSIONS, NUMBER_PATTERN

        assert NUMBER_PATTERN.findall("O43/O84".translate(CONFUSIONS)) == [("043", "084")]
        assert NUMBER_PATTERN.findall("l52/l32".translate(CONFUSIONS)) == [("152", "132")]
        assert band.shape == (100, 400, 3)  # garde-fou de forme, sans appel Vision
