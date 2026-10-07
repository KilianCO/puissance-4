# Approche et résultats

Ce document résume ce qui a été fait dans ce dépôt et pourquoi. Le modèle est jouable sur [kilianco.github.io](https://kilianco.github.io/projets/puissance-4/).

## Objectif

Apprendre à une machine à jouer au Puissance 4 sans lui donner de stratégie : elle joue, reçoit +1 pour une victoire et −1 pour une défaite, et doit en déduire quels coups mènent à la victoire. Puis mener le modèle jusqu'au bout : le mesurer, l'exporter, le rendre jouable dans un navigateur.

## Trois tentatives

| Version | Principe | Résultat |
|---|---|---|
| Q-learning tabulaire | une valeur mémorisée par position rencontrée | apprend contre un joueur aléatoire, mais ne généralise pas : le jeu compte environ 4,5 × 10¹² positions |
| DQN V1 | un réseau dense remplace le tableau ; entraîné 10 000 parties contre un joueur aléatoire | 4 % de victoires contre un adversaire qui pare seulement les menaces immédiates |
| DQN V2 | réseau convolutif, entraîné 40 000 parties contre lui-même et des adversaires variés | modèle publié, résultats ci-dessous |

Les deux premières versions sont conservées dans le dépôt comme points de comparaison.

## Ce qui a fait échouer la V1

Le taux de victoire affiché pendant l'entraînement paraissait correct. Une évaluation rigoureuse a montré que le modèle jouait à peine mieux que le hasard, et pourquoi :

- l'exploration ne diminuait pas assez vite : 61 % des coups étaient encore aléatoires à la fin de l'entraînement ;
- le seul adversaire était un joueur aléatoire, qui ne crée presque jamais de menace à parer ;
- l'entraînement était trop court ;
- le plateau était codé par 0, 1 ou 2 par case, ce qui suggère un ordre entre « moi » et « adversaire » qui n'existe pas.

## Les choix de la V2

| Choix | Raison |
|---|---|
| Plateau codé en deux plans 6 × 7 (mes pions, ceux de l'adversaire), toujours vu par le joueur qui doit jouer | pas d'ordre artificiel, et le réseau n'a pas à apprendre séparément à jouer en premier et en second |
| Réseau convolutif à blocs résiduels, 267 000 paramètres | un alignement est le même motif partout sur le plateau ; assez petit pour un navigateur |
| Chaque coup de chaque joueur sert d'exemple, avec une cible de signe opposé pour la position suivante | deux fois plus de données, et l'agent apprend aussi des coups de ses adversaires |
| Double DQN, réseau cible, replay buffer | stabilité de l'apprentissage, correction du biais d'optimisme |
| Adversaires tirés au sort : lui-même (50 %), minimax (25 %), tactique (15 %), aléatoire (10 %) | le jeu contre soi-même fait progresser sans plafond ; les autres évitent qu'il ne sache jouer que contre son propre style |
| Positions apprises aussi en miroir gauche-droite | le jeu est symétrique : chaque partie en vaut deux |
| Évaluation sans exploration toutes les 2 000 parties, conservation du meilleur modèle | la progression n'est pas régulière : le dernier modèle n'est pas toujours le meilleur |

## Mesurer avant de conclure

Le banc d'évaluation (`P4/arena.py`) fait affronter à n'importe quel joueur les mêmes adversaires, la moitié des parties dans chaque position, puisque le premier joueur a un avantage.

| Adversaire | Description |
|---|---|
| Aléatoire | joue au hasard |
| Tactique | gagne ou bloque quand c'est possible en un coup, au hasard sinon |
| Minimax 1, 4, 6 | recherche alpha-bêta à 1, 4 ou 6 coups d'avance, avec une évaluation écrite à la main ; ce sont les niveaux facile, moyen et difficile du site |

## Résultats

Part de victoires sur 200 parties par adversaire, sans exploration.

| Adversaire | DQN V1 | DQN V2, plateau vide | DQN V2, 4 premiers coups au hasard |
|---|---|---|---|
| Aléatoire | 65 % | 100 % | 100 % |
| Tactique | 4 % | 98,5 % | 95 % |
| Minimax 1 (facile) | 6 % | 99,5 % | 96 % |
| Minimax 4 (moyen) | 0 % | 88 % | 58 % |
| Minimax 6 (difficile) | 0 % | 65,5 % | 45,5 % |

*La V1 a été mesurée sur 100 parties par adversaire.*

Les deux colonnes de la V2 ne disent pas la même chose. En partant du plateau vide, le modèle et le minimax sont presque déterministes et rejouent souvent des parties proches de celles vues à l'entraînement. Avec quatre premiers coups aléatoires, le modèle joue des positions nouvelles : c'est la mesure la plus représentative, et celle à retenir.

## Un troisième type de joueur : la recherche de Monte-Carlo

`P4/mcts.py` ajoute un joueur qui n'a rien appris et ne contient aucune stratégie écrite à la main. Pour juger un coup, il termine la partie au hasard un grand nombre de fois et compte les victoires. Un arbre de recherche concentre les simulations sur les coups prometteurs (formule UCT), et le coup joué est le plus visité.

Il n'y a rien à entraîner : sa force se règle par le nombre de simulations accordées à chaque coup. C'est un point de comparaison utile, parce qu'il ne doit rien ni à l'apprentissage ni à une formule d'évaluation.

## Classement Elo

`P4/league.py` fait s'affronter tous les joueurs deux à deux, puis calcule les notes Elo qui expliquent le mieux l'ensemble des résultats. Les notes ne sont pas mises à jour partie après partie, ce qui les rendrait dépendantes de l'ordre des parties.

Tournoi du 7 octobre 2026 : 40 parties par confrontation, la moitié dans chaque position, quatre premiers coups au hasard. Le détail est dans [ligue.json](ligue.json).

| Rang | Joueur | Méthode | Elo |
|---|---|---|---|
| 1 | Monte-Carlo, 10 000 simulations | recherche | 1856 |
| 2 | **IA entraînée (DQN V2)** | apprentissage | 1832 |
| 3 | Minimax moyen | recherche | 1809 |
| 4 | Minimax difficile | recherche | 1806 |
| 5 | Monte-Carlo, 1 000 simulations | recherche | 1717 |
| 6 | Minimax à 2 coups d'avance | recherche | 1685 |
| 7 | Minimax facile | recherche | 1297 |
| 8 | Tactique | règle simple | 1280 |
| 9 | Premier réseau (DQN V1) | apprentissage | 917 |
| 10 | Aléatoire | hasard | 801 |

200 points d'écart correspondent à environ 76 % de score attendu. Les quatre premiers se tiennent en 50 points, ce qui n'est pas significatif à 40 parties par confrontation : le réseau, le minimax à 4 ou 6 coups d'avance et le Monte-Carlo à 10 000 simulations sont de force comparable. Le classement confirme en revanche l'écart entre les deux versions du DQN.

L'Elo suppose une force transitive, ce qui n'est pas garanti entre des méthodes aussi différentes ; le détail des confrontations reste donc utile.

## Limites

- Le modèle ne calcule pas : il répond en une seule passe du réseau. Il perd encore 5 % des parties contre le joueur tactique.
- Un seul entraînement, une seule graine aléatoire : la variabilité d'un entraînement à l'autre n'est pas mesurée.
- Tous les choix de la V2 ont été introduits ensemble : leur contribution individuelle n'est pas isolée.
- Le Puissance 4 est un jeu résolu ; un solveur exact reste plus fort que n'importe quel modèle appris.

## Déploiement

Le réseau est exporté au format ONNX. L'export relit le fichier avec ONNX Runtime et vérifie, sur 200 positions, qu'il calcule la même chose que le modèle PyTorch. Le site l'exécute ensuite dans le navigateur du visiteur avec ONNX Runtime Web : pas de serveur, pas de coût, pas de latence réseau.

## Suite envisagée

- Réunir les deux briques présentes : guider la recherche de Monte-Carlo par le réseau au lieu de parties aléatoires (principe d'AlphaZero).
- Ajouter d'autres méthodes d'apprentissage à la ligue.
- Évaluer automatiquement chaque nouveau modèle avant sa mise en ligne.
