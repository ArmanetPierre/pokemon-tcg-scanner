"""L'empreinte doit changer quand l'artefact change — et seulement alors."""

from __future__ import annotations

import pytest

from src.fingerprint import fingerprint, short


@pytest.fixture
def package(tmp_path):
    """Imite la forme d'un .mlpackage : un répertoire, plusieurs fichiers."""
    root = tmp_path / "CardEncoder.mlpackage"
    (root / "Data" / "com.apple.CoreML" / "weights").mkdir(parents=True)
    (root / "Manifest.json").write_bytes(b'{"a": 1}')
    (root / "Data" / "com.apple.CoreML" / "model.mlmodel").write_bytes(b"graphe")
    (root / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").write_bytes(b"poids")
    return root


def test_stable_entre_deux_appels(package):
    assert fingerprint(package) == fingerprint(package)


def test_change_si_un_poids_change(package):
    avant = fingerprint(package)
    (package / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").write_bytes(b"poidt")
    assert fingerprint(package) != avant


def test_change_si_un_fichier_est_renomme(package):
    """Le chemin relatif entre dans le condensat : déplacer un poids compte.

    Sans cela, deux paquets aux mêmes octets répartis différemment auraient la
    même empreinte, et la vérification laisserait passer un modèle mal assemblé.
    """
    avant = fingerprint(package)
    weights = package / "Data" / "com.apple.CoreML" / "weights"
    (weights / "weight.bin").rename(weights / "weight2.bin")
    assert fingerprint(package) != avant


def test_change_si_un_fichier_est_ajoute(package):
    avant = fingerprint(package)
    (package / "extra.txt").write_bytes(b"")
    assert fingerprint(package) != avant


def test_fichier_seul(tmp_path):
    path = tmp_path / "index.bin"
    path.write_bytes(b"\x00\x01\x02")
    assert fingerprint(path).startswith("sha256:")


def test_artefact_absent_leve(tmp_path):
    with pytest.raises(FileNotFoundError):
        fingerprint(tmp_path / "nexiste_pas")


def test_short_lisible_dans_un_journal():
    assert short("sha256:" + "a" * 64) == "a" * 12
