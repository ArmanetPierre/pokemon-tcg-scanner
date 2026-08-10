"""Source d'images de secours, pour les cartes que pokemon-tcg-data ne couvre pas.

Deux trous distincts, tous deux mesurés :

- des sets entiers absents de pokemon-tcg-data alors qu'ils existent — les promos
  « MEP » de 2026, référencées par TCGdex mais **sans visuel** chez lui ;
- des cartes présentes dans les métadonnées dont l'image répond 404 sur
  `images.pokemontcg.io` — 50 au dernier relevé, surtout les collections
  McDonald's.

`images.scrydex.com` couvre une partie des deux. Mais il ne renvoie **jamais**
404 : pour un identifiant inconnu il sert un placeholder en HTTP 200. Prendre le
code de statut pour une réponse ajouterait au dataset des dos de carte
identiques, qui deviendraient un attracteur universel dans l'espace des
embeddings — chaque photo un peu ratée leur ressemblerait. D'où la vérification
par empreinte ci-dessous : c'est elle qui rend cette source utilisable, pas
l'URL.

Relevé du 2026-08-10 : les 60 promos MEP et `svp-102` ont de vraies images ;
les 8 énergies MEE, les 48 cartes McDonald's et `hsp-HGSS18` n'ont que le
placeholder, et restent donc hors index.
"""

from __future__ import annotations

import hashlib

import requests

SCRYDEX = "https://images.scrydex.com/pokemon"

# Empreinte SHA-256 du placeholder servi pour tout identifiant inconnu.
# Vérifiée en demandant des identifiants volontairement absurdes.
PLACEHOLDER_SHA256 = "01f03f71564c567abf1b8f38b87e16e42bd02ebbd77ddebb9671178646b1c1f5"

# Ce que l'on compare réellement : le préfixe suffit à distinguer un placeholder
# d'une vraie carte, et rester sur un préfixe évite de casser au moindre
# réencodage du fichier côté CDN.
PLACEHOLDER_PREFIX = "01f03f71564c"


def scrydex_url(card_id: str, size: str = "small") -> str:
    """`size` vaut `small` (245x342) ou `large` (734x1024)."""
    return f"{SCRYDEX}/{card_id}/{size}"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_placeholder(data: bytes) -> bool:
    return _digest(data).startswith(PLACEHOLDER_PREFIX)


def fetch_if_real(
    card_id: str,
    size: str = "small",
    session: requests.Session | None = None,
    timeout: int = 30,
) -> bytes | None:
    """Les octets de l'image, ou None si le CDN n'a servi que son placeholder.

    None veut dire « cette carte n'a pas d'image ici », pas « erreur réseau » —
    une panne lève, pour ne pas retirer silencieusement des cartes de l'index au
    prochain passage.
    """
    get = (session or requests).get
    resp = get(scrydex_url(card_id, size), timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    return None if is_placeholder(resp.content) else resp.content


def has_real_image(
    card_id: str,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> bool:
    """Teste la disponibilité sur la petite image, qui coûte ~45 Ko."""
    return fetch_if_real(card_id, "small", session, timeout) is not None
