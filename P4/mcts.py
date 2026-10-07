# -*- coding: utf-8 -*-

"""
Recherche arborescente de Monte-Carlo (MCTS) pure.

Ni apprentissage, ni fonction d'évaluation écrite à la main : la valeur
d'un coup est estimée en jouant un grand nombre de parties aléatoires à
partir de lui, et en comptant les victoires.

Chaque simulation fait quatre choses :

1. sélection   : descendre dans l'arbre déjà construit en choisissant à
                 chaque étage le coup au meilleur compromis entre « a bien
                 marché jusqu'ici » et « peu essayé » (formule UCT) ;
2. expansion   : ajouter à l'arbre un coup pas encore essayé ;
3. simulation  : finir la partie au hasard ;
4. rétropropagation : remonter le résultat le long du chemin parcouru.

Le coup joué est le plus visité. La démo du site contient la même
recherche en JavaScript (static/js/puissance4/agents.js).
"""

import math
import random

from .bitboard import has_four, is_full, legal_actions, play


EXPLORATION = 1.4

_ONGOING = 0
_WON = 1
_DRAWN = 2


class _Node:
    """
    Position atteinte après un coup. `wins` est compté pour le joueur
    qui vient de jouer ce coup.
    """

    __slots__ = (
        "current", "mask", "parent", "move",
        "children", "untried", "visits", "wins", "state",
    )

    def __init__(self, current, mask, parent=None, move=None):
        self.current = current
        self.mask = mask
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0.0

        if parent is not None and has_four(current ^ mask):
            self.state = _WON
        elif is_full(mask):
            self.state = _DRAWN
        else:
            self.state = _ONGOING

        self.untried = (
            legal_actions(mask) if self.state == _ONGOING else []
        )


def _rollout(current: int, mask: int) -> float:
    """
    Finit la partie au hasard. Retourne le résultat pour le joueur qui
    doit jouer au départ : 1 victoire, 0 défaite, 0,5 nul.
    """
    starter_to_move = True

    while True:
        columns = legal_actions(mask)

        if not columns:
            return 0.5

        current, mask = play(current, mask, random.choice(columns))

        if has_four(current ^ mask):
            return 1.0 if starter_to_move else 0.0

        starter_to_move = not starter_to_move


def mcts_visits(
    current: int,
    mask: int,
    simulations: int,
) -> dict[int, int]:
    """Nombre de simulations consacrées à chaque coup légal."""
    root = _Node(current, mask)

    for _ in range(simulations):
        node = root

        # 1. Sélection
        while not node.untried and node.children:
            log_visits = math.log(node.visits)

            node = max(
                node.children,
                key=lambda child: (
                    child.wins / child.visits
                    + EXPLORATION * math.sqrt(log_visits / child.visits)
                ),
            )

        # 2. Expansion
        if node.untried:
            column = node.untried.pop(
                random.randrange(len(node.untried))
            )

            child = _Node(
                *play(node.current, node.mask, column),
                parent=node,
                move=column,
            )

            node.children.append(child)
            node = child

        # 3. Simulation : résultat pour le joueur qui vient de jouer.
        if node.state == _WON:
            result = 1.0
        elif node.state == _DRAWN:
            result = 0.5
        else:
            result = 1.0 - _rollout(node.current, node.mask)

        # 4. Rétropropagation : le point de vue s'inverse à chaque étage.
        while node is not None:
            node.visits += 1
            node.wins += result
            result = 1.0 - result
            node = node.parent

    return {child.move: child.visits for child in root.children}


def mcts_move(current: int, mask: int, simulations: int) -> int:
    """Coup le plus visité ; les égalités sont départagées au hasard."""
    visits = mcts_visits(current, mask, simulations)
    best = max(visits.values())

    return random.choice([
        column
        for column, count in visits.items()
        if count == best
    ])
