# Photos d'évaluation

Photos réelles prises à l'iPhone. C'est le seul jeu de test qui compte : les
scans officiels sont dans l'index, donc les identifier ne prouve rien.

## Où déposer les fichiers

Un dossier par carte physique, dans `photos/` :

```
photos/
  pikachu_58-102/
    01_plat.jpg
    02_incline.jpg
    03_pivote.jpg
    ...
  charizard_4-102/
    01_plat.jpg
    ...
```

**Nom du dossier** : ce qui est imprimé en bas de la carte, c'est-à-dire le
numéro et le total (`58/102` → `58-102`), précédé du nom du Pokémon. Ça suffit à
retrouver l'ID exact sans ambiguïté. Si le numéro est illisible ou absent
(promos), mettez ce que vous pouvez, on résoudra au cas par cas.

**Nom du fichier** : le préfixe numéroté n'a pas d'importance, seul le mot-clé
de condition compte (voir ci-dessous). Le format iPhone (HEIC) est accepté.

## Conditions à couvrir

Par ordre d'importance. Les 5 premières suffisent pour un premier verdict.

| Mot-clé | Condition |
|---|---|
| `plat` | à plat, bien cadrée, lumière correcte — c'est la référence |
| `incline` | vue de biais, 30-45°, forte perspective |
| `pivote` | tournée dans le plan (45°, 90°, à l'envers) |
| `reflet` | lumière crue ou fenêtre dans l'axe — le cas dur pour les holo |
| `sombre` | lumière faible, ou carte à l'ombre |
| `sleeve` | sous pochette plastique |
| `loin` | carte petite dans le cadre (moins d'1/4 de l'image) |
| `masque` | un coin ou un bord caché par un doigt / une autre carte |
| `fond` | posée sur une table encombrée, fond chargé |
| `flou` | léger bougé |

## Quelles cartes choisir

Le choix des cartes compte autant que celui des conditions :

- **2-3 holographiques ou reverse holo** — les reflets sont le cas le plus dur,
  et l'index ne contient que des scans mats et parfaits.
- **1-2 cartes très communes** (un Pikachu, un Dracaufeu de base). Ce sont
  celles qui ont été réimprimées dans plusieurs sets, donc celles où
  l'identification se joue à 0,02 d'écart.
- **1-2 cartes récentes** (Scarlet & Violet ou plus récent) et **1-2 anciennes**
  (Base, Neo, EX) — les mises en page diffèrent beaucoup.
- **1 carte non française si vous en avez** : l'index est en anglais. Savoir si
  une carte française est reconnue change le périmètre du projet.

## Combien

**Commencez petit : 2 ou 3 cartes avec les 5 premières conditions**, soit une
quinzaine de photos. Ça valide le protocole et le nommage avant que vous y
passiez une heure. On étend ensuite à une dizaine de cartes.

## Bonus utile plus tard

Quelques photos avec **plusieurs cartes visibles** (3-5 étalées sur une table,
qui se chevauchent un peu). Elles ne servent pas à l'évaluation
d'identification, mais elles diront si la détection de rectangles d'Apple suffit
ou s'il faudra entraîner un YOLO. À déposer dans `photos/_multi/`.
