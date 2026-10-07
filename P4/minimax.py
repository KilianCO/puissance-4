# -*- coding: utf-8 -*-

"""
Adversaires de référence par recherche : minimax alpha-bêta et tactique.

Le minimax reprend exactement l'évaluation et l'ordre de recherche de la
démo du site (static/js/puissance4/agents.js) : une profondeur donnée ici
correspond donc au même niveau que sur le site.
"""

import random

from .bitboard import (
    CENTER_COLUMN,
    CENTER_FIRST,
    WINDOWS,
    can_play,
    has_four,
    is_full,
    legal_actions,
    play,
)


WIN = 1_000_000


def evaluate(current: int, mask: int) -> int:
    """Évalue la position pour le joueur qui doit jouer."""
    opponent = current ^ mask

    score = 3 * (current & CENTER_COLUMN).bit_count()

    for window in WINDOWS:
        mine = (current & window).bit_count()
        theirs = (opponent & window).bit_count()

        if mine and theirs:
            continue

        if mine == 3:
            score += 5
        elif mine == 2:
            score += 2
        elif theirs == 3:
            score -= 4
        elif theirs == 2:
            score -= 1

    return score


def negamax(
    current: int,
    mask: int,
    depth: int,
    alpha: float,
    beta: float,
) -> float:
    """Valeur de la position pour le joueur qui doit jouer."""

    # L'adversaire vient de jouer : a-t-il gagné ?
    if has_four(current ^ mask):
        return -WIN - depth

    if is_full(mask):
        return 0

    if depth == 0:
        return evaluate(current, mask)

    best = -float("inf")

    for column in CENTER_FIRST:
        if not can_play(mask, column):
            continue

        value = -negamax(
            *play(current, mask, column),
            depth - 1,
            -beta,
            -alpha,
        )

        if value > best:
            best = value

        if best > alpha:
            alpha = best

        if alpha >= beta:
            break

    return best


def minimax_scores(
    current: int,
    mask: int,
    depth: int,
) -> dict[int, float]:
    """Score de chaque coup légal, à la profondeur donnée."""
    return {
        column: -negamax(
            *play(current, mask, column),
            depth - 1,
            -float("inf"),
            float("inf"),
        )
        for column in CENTER_FIRST
        if can_play(mask, column)
    }


def minimax_move(
    current: int,
    mask: int,
    depth: int,
    randomness: float = 0.0,
) -> int:
    """
    Meilleur coup selon le minimax.

    Les égalités sont départagées au hasard. `randomness` est la part de
    coups joués complètement au hasard (niveau « facile » du site).
    """
    if randomness and random.random() < randomness:
        return random.choice(legal_actions(mask))

    scores = minimax_scores(current, mask, depth)
    best = max(scores.values())

    return random.choice([
        column
        for column, score in scores.items()
        if score == best
    ])


def tactical_move(current: int, mask: int) -> int:
    """
    Gagne si c'est possible, bloque une menace immédiate, sinon joue au
    hasard.
    """
    columns = legal_actions(mask)
    opponent = current ^ mask

    # `play` rend la position vue par le joueur suivant : les pions de
    # celui qui vient de jouer sont donc new_current ^ new_mask.
    for column in columns:
        new_current, new_mask = play(current, mask, column)

        if has_four(new_current ^ new_mask):
            return column

    for column in columns:
        new_current, new_mask = play(opponent, mask, column)

        if has_four(new_current ^ new_mask):
            return column

    return random.choice(columns)
