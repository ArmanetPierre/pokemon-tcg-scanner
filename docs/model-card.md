# Fiche modèle — reconnaissance de cartes Pokémon TCG sur appareil

*Version du 12 août 2026. Tous les chiffres sont reproductibles par les scripts
cités ; aucun n'est repris d'une exécution antérieure sans avoir été refait.*

---

## Ce que c'est

Un système d'**identification de carte** : à partir d'une photo, il rend quelle
carte c'est, de quelle extension, et à quel point la réponse est fiable.
Entièrement sur l'appareil, sans appel réseau.

Ce n'est **pas** un classifieur. L'architecture est *détecter → redresser →
encoder → chercher → lire le numéro*, et la « connaissance » vit dans un index
de vecteurs, pas dans des poids. Une nouvelle extension se règle en régénérant
l'index ; le modèle ne change jamais.

| | |
|---|---|
| Encodeur | MobileCLIP2-S2, **zero-shot** (jamais réentraîné ni affiné) |
| Sortie | vecteur de 512 dimensions, L2-normalisé |
| Format embarqué | Core ML fp16, 69 Mo, **5,5 ms** sur le Neural Engine d'un iPhone 13 Pro |
| Index | 20 512 cartes, 176 extensions, float16, 21 Mo |
| Recherche | produit matriciel exact (pas de FAISS, pas d'index approché) |
| Étages non appris | détection et OCR (Vision), orientation, lecture du numéro |

**L'essentiel de la performance ne vient pas du modèle.** Ablation sur le banc
de 46 photos :

| Configuration | Top-1 |
|---|---|
| photo entière, sans détection ni orientation | 5/39 (13 %, IC95 6-27) |
| + détection, filtrage, orientation → similarité seule | 29/39 (74 %, IC95 59-85) |
| + lecture du numéro imprimé → chaîne complète | **35/39 (90 %, IC95 76-96)** |

Le cadrage vaut 24 identifications, l'OCR du bandeau 6. Le modèle représente
environ **5 % du temps de calcul** d'une identification ; les deux passes d'OCR
de Vision en représentent les deux tiers.

---

## Usage prévu

**Prévu.** Aider une personne à identifier une carte qu'elle a en main, dans une
application où elle voit le résultat et peut le corriger. Le système est conçu
pour **se taire** quand il ne sait pas, et cette propriété compte plus que son
taux brut.

**Hors périmètre, et déconseillé.**

- **Toute décision d'argent** — estimation, achat, revente, assurance,
  authentification. Le système ne distingue pas une carte authentique d'une
  contrefaçon, ni un état de conservation d'un autre, et il attribue parfois
  fermement une mauvaise édition (voir *Défaillances connues*).
- **Inventaire automatique sans relecture humaine.** À 90 % de précision, une
  collection de 1 000 cartes contient une centaine d'erreurs, dont certaines
  affirmées avec assurance.
- **Cartes japonaises ou coréennes.** L'index est anglais. Les cartes françaises
  sont reconnues — l'illustration domine largement le texte — mais les lignes
  japonaise et coréenne ont une numérotation propre et ne sont pas couvertes.
- **Détection de contrefaçon, notation d'état, estimation de prix.** Aucune de
  ces tâches n'est mesurée, et rien dans la conception n'y prépare.

---

## Données

### L'index

20 512 cartes, 176 extensions, de Base Set (1999) à Pitch Black (2026).

| Ère | Cartes | Part |
|---|---:|---:|
| Base / WotC (1999-2003) | 1 780 | 8,7 % |
| EX (2003-2007) | 1 722 | 8,4 % |
| Diamond & Pearl / Platinum | 1 417 | 6,9 % |
| HGSS / Call of Legends | 544 | 2,7 % |
| Black & White | 1 416 | 6,9 % |
| XY | 1 775 | 8,7 % |
| Sun & Moon | 2 955 | 14,5 % |
| Sword & Shield | 3 529 | 17,3 % |
| Scarlet & Violet | 3 249 | 15,9 % |
| Mega Evolution | 1 039 | 5,1 % |
| promos, decks, spéciaux | 1 028 | 5,0 % |

**Provenance.** Métadonnées de `PokemonTCG/pokemon-tcg-data`, épinglée à la
révision `8b4e3879` (2026-07-17). Images haute résolution du CDN officiel, sauf
49 cartes (collections McDonald's 2014-2018 et une promo HGSS) récupérées à la
main, aucune source publique ne les servant. L'empreinte de l'encodeur est
inscrite dans `index.json` : un index et un modèle désaccordés ne produisent
aucune erreur, seulement de fausses réponses confiantes.

**Un piège de collecte, résolu.** Le CDN de repli ne renvoie jamais 404 : il
répond à tout identifiant inconnu par une image générique en HTTP 200. Faire
confiance au code de statut aurait ajouté 57 dos de cartes identiques à l'index,
qui se seraient comportés en attracteurs universels. La disponibilité se décide
donc sur une empreinte de contenu.

### L'index n'est pas de qualité uniforme

Mesuré directement (`scripts/audit/audit_index.py`) :

- **4,3 % des cartes ont dans l'index un voisin à ≥ 0,99** — visuellement
  indiscernables, quelle que soit la qualité de la photo.
- La **marge intrinsèque médiane** entre une carte et sa plus proche voisine est
  de **0,0099**, sous le seuil de décision de 0,03. Pour **77,6 %** de l'index,
  la similarité ne peut structurellement pas produire un verdict ferme.
- 26 cartes se comportent en **attracteurs**, apparaissant plus de 100 fois dans
  le top-10 d'autres cartes (jusqu'à 284 pour `sm3-10`), contre une médiane de 6.

En conséquence, chaque carte porte dans `cards.json` un champ
`needs_printed_number`, mesuré et non déduit : **4 787 cartes (23,3 %)** ne
peuvent pas être tranchées par l'image seule. Très inégal selon l'époque : 78,3 %
pour la série Base, 7,5 % pour Platinum.

---

## Évaluation

### Banc de photos réelles — le juge de paix

46 photos, dont **39 avec une bonne réponse** et **7 sans** (dos de carte,
cartes One Piece, carte coréenne, pochon porte-cartes, flou illisible). Trois
collections, plusieurs personnes, conditions ordinaires : contre-jour, pochette,
fond chargé, cartes inclinées, en classeur, un lot entièrement en paysage.

| | Calibration | Test | Total |
|---|---|---|---|
| photos avec réponse | 21 | 18 | 39 |
| top-1 | 21/21 (100 %, IC95 85-100) | **14/18 (78 %, IC95 55-91)** | **35/39 (90 %, IC95 76-96)** |
| top-5 | 21/21 | 16/18 | 37/39 |
| verdicts fermes, et justes | | | **29/39, dont 29/29** |
| refus corrects | 3/4 | 3/3 | **6/7** |

**Le protocole, et ce qu'il ne garantit pas.** Les seuils de confiance ont été
fixés avant l'arrivée du 2ᵉ lot et n'ont pas bougé : pour eux, ce lot est du
hold-out. Les filtres de détection, eux, ont été réglés en le voyant ; le 3ᵉ lot
leur rend ce hold-out. Le détail est inscrit dans `truth.json`.

**39 photos ne peuvent pas démontrer mieux que leur borne basse.** Même un
sans-faute plafonnerait à 91 % de borne inférieure. Tout chiffre présenté sans
son intervalle, sur ce banc, est une sur-affirmation.

### Banc synthétique — pour voir par ère

1 209 requêtes sur des cartes tenues à l'écart, obtenues en dégradant le scan de
référence (`src/augment.py`). Mesure **relative** : elle compare des ères et des
espaces entre eux, elle ne prédit pas la précision sur de vraies photos.

| Ère | top-1 |
|---|---:|
| WotC 1990s | **75,6 %** |
| WotC 2000s | 84,7 % |
| EX | 94,2 % |
| Diamond & Pearl / Platinum | **95,8 %** |
| Sword & Shield | 90,0 % |
| Scarlet & Violet | 95,0 % |
| **Total** | **91,1 %** |

**20 points d'écart entre la meilleure et la pire ère.** Le banc de photos
réelles ne pouvait pas le voir : il ne contenait aucune carte antérieure à
Sun & Moon avant le 3ᵉ lot, alors que 47 % de l'index en relève.

### Confiance

Deux niveaux, sur des marges calculées **avant** toute promotion par l'OCR :

| Condition | Niveau |
|---|---|
| le numéro imprimé a été lu et désigne un candidat | `edition` |
| marge 1er/2e ≥ 0,03 | `edition` |
| marge au premier candidat d'un autre nom ≥ 0,04 | `nom` |
| sinon | `incertain` |

**Le score absolu ne doit jamais être seuillé ni montré.** Deux cartes sans
rapport se ressemblent déjà à 0,697 en médiane ; la plage utile n'est pas 0-1.
C'est une propriété du modèle, pas des cartes : la norme du vecteur moyen de
l'index vaut **0,8146**, donc l'essentiel de la similarité est une constante
partagée.

**Ces seuils sont insuffisants, et c'est mesuré.** Sur le banc élargi, la
précision à 25 % de couverture (80 %) est **plus basse** qu'à 100 % (89,7 %) :
trier par la marge est pire que ne pas trier. AURC = 0,101.

---

## Défaillances connues

**Une attribution ferme et fausse** sur 46 photos — le pire résultat possible
pour un utilisateur, qui n'a aucun moyen de savoir que la réponse est inventée :

- **Un flou de bougé qu'aucun humain ne peut identifier**, annoncé fermement.

Le troisième lot en avait produit **trois**. Les deux autres ne se produisent
plus, et aucune des 39 photos ayant une bonne réponse n'est aujourd'hui affirmée
à tort : **29 verdicts fermes, 29 justes**. Elles restent notées ici parce
qu'elles disent où la chaîne cède :

1. Carte petite dans une étagère encombrée : aucun quadrilatère ne l'isole, le
   **repli photo-entière** l'emportait et affirmait une carte sans rapport, la
   bonne réponse n'étant même pas dans le top-5.
2. Carte de 2006 (Crystal Guardians) : rang 2, et une autre carte affirmée.
   C'était le décrochage des ères anciennes, confirmé sur une vraie photo.

**Le seuil ne se répare pas par recalibration.** Pour que le flou cesse d'être
affirmé, `FIRM_ID_MARGIN` doit passer de 0,03 à 0,0558 — ce qui fait tomber les
verdicts fermes de 21/21 à 6/21. Bloquer un faux positif coûte **71 % de la
couverture**.

Autres limites :

- **Cartes qui se chevauchent.** Les détecteurs d'Apple ont besoin de quatre
  arêtes fermées. Sur une photo de cinq cartes dont quatre se recouvraient, une
  seule a été isolée.
- **Cartes hors index.** Quand un numéro se lit proprement et n'appartient à
  aucune carte connue, le système répond « carte inconnue » plutôt que
  d'attribuer au premier venu.
- **L'embedding peut manquer complètement.** Sur une carte, la bonne réponse
  n'était pas dans le top-10, à 0,089 derrière un Dresseur sans rapport.
- **Le simulateur iOS ne vaut rien pour juger la précision.** Core ML s'y comporte
  à l'identique, Vision non.

---

## Résultats négatifs

Ils sont publiés parce qu'ils ferment des portes, et qu'une porte fermée avec une
mesure vaut mieux qu'une piste rouverte tous les six mois.

- **Corriger l'anisotropie sans supervision dégrade.** Le centrage
  (`e − µ`, renormalisé) multiplie la marge intrinsèque médiane par 3,55 et
  écrase la hubness (hub maximal 284 → 77), mais fait tomber le top-1 de 26/31 à
  23/31 — mesuré sur le bras similarité seule du banc de 32 photos, avant son
  élargissement. CSLS fait pire encore, à 22/31.
- **Une projection apprise gagne, mais n'est pas livrable.** Entraînée en
  contrastif sur des paires (scan dégradé → scan), elle porte le banc synthétique
  de 91,1 % à 95,2 % sur des cartes jamais vues, et améliore nettement le tri
  (AURC 0,047 → 0,016). Mais elle dégrade le refus, et ce seuil-là ne peut pas
  être calibré honnêtement faute de négatifs en quantité. Elle a aussi révélé que
  **quatre grandeurs absolues du pipeline** ne survivent pas à un changement
  d'espace, dont une qui n'était écrite nulle part : `selection_score` additionne
  un score et une marge, ce qui suppose qu'ils vivent sur la même échelle.
- **Un critère de netteté n'écarte pas les non-cartes.** Variance du laplacien
  sur le crop retenu : négatifs de 70 à 1 082, vraies cartes de 44 à 6 106,
  distributions entièrement superposées.
- **Le banc synthétique ne peut pas comparer deux stratégies d'enrôlement.** Il
  donne 99,9 % à un index centroïde contre 91,1 % à l'index de référence, un gain
  qui ne se retrouve pas sur les vraies photos : enrôlement et requête sortent du
  même modèle de dégradation.

---

## Configuration livrée, et alternatives mesurées

Ce qui est embarqué : **index de référence** (un vecteur par carte, le scan) et
seuils 0,03 / 0,04. Deux alternatives sont mesurées et **non embarquées**, leurs
artefacts restant hors du dépôt :

| | livré | centroïde multi-vues | projection apprise |
|---|---|---|---|
| top-1 (39 photos) | **35/39** | 34/39 | non mesuré sur ce banc |
| refus corrects | 6/7 | **7/7** | dégradé |
| AURC | 0,101 | **0,077** | 0,016 (banc de 32) |
| coût du refus sans faux positif | 71 % de couverture | **24 %** | non calibrable |
| coût d'exploitation | — | **nul** | nul |

⚠️ **Lire ce tableau en colonnes.** Le `README` a annoncé pendant un temps
« 34/39 et 7/7 » comme état courant : ce sont les chiffres du **centroïde
multi-vues**, qui n'est pas embarqué. Le livré fait 35/39 et 6/7. La colonne de
gauche est la seule qui décrit ce que l'application fait.

Le centroïde perd une identification et répare la confiance. L'arbitrage est un
choix produit, pas technique : une identification de moins contre une
attribution fausse et confiante de moins.

---

## Reproduire

```bash
cd ml
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/pip install -e . --no-deps

.venv/bin/python scripts/evaluate_real.py --ablation   # le juge de paix
.venv/bin/python scripts/evaluate_synthetic.py         # par ère, sur tout l'index
.venv/bin/python scripts/audit/audit_index.py          # structure de l'index
.venv/bin/python -m pytest -q                          # 43 invariants
```

Les photos du banc ne sont pas dans le dépôt. L'analyse complète, avec les
chantiers restants, est dans [`audit-ml.md`](audit-ml.md).
