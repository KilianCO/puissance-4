# -*- coding: utf-8 -*-

"""
Représentation compacte du plateau, pour la recherche et l'entraînement.

Une position tient en deux entiers :

    current : les pions du joueur qui doit jouer ;
    mask    : tous les pions posés.

Chaque colonne occupe 7 bits (6 cases + 1 bit de garde), la case du bas
en premier. Les alignements se testent par décalages de bits, ce qui est
beaucoup plus rapide que de parcourir la grille de Board.

Board reste la référence pour les règles ; les tests vérifient que les
deux représentations sont toujours d'accord.
"""

import numpy as np

from .board import Board, Cell


ROWS = Board.ROWS
COLUMNS = Board.COLUMNS

_COLUMN_BITS = ROWS + 1

# Colonnes du centre vers les bords : meilleur ordre pour l'élagage.
CENTER_FIRST = (3, 2, 4, 1, 5, 0, 6)


def _bit(row_from_bottom: int, column: int) -> int:
    return 1 << (column * _COLUMN_BITS + row_from_bottom)


_BOTTOM = tuple(_bit(0, column) for column in range(COLUMNS))
_TOP = tuple(_bit(ROWS - 1, column) for column in range(COLUMNS))

CENTER_COLUMN = sum(_bit(row, 3) for row in range(ROWS))

FULL = sum(
    _bit(row, column)
    for row in range(ROWS)
    for column in range(COLUMNS)
)


def _build_windows() -> tuple[int, ...]:
    """Les 69 alignements de quatre cases possibles."""
    windows = []

    for row in range(ROWS):
        for column in range(COLUMNS):
            for delta_row, delta_column in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_row = row + 3 * delta_row
                end_column = column + 3 * delta_column

                if not (0 <= end_row < ROWS and 0 <= end_column < COLUMNS):
                    continue

                windows.append(sum(
                    _bit(row + i * delta_row, column + i * delta_column)
                    for i in range(4)
                ))

    return tuple(windows)


WINDOWS = _build_windows()


def can_play(mask: int, column: int) -> bool:
    return not mask & _TOP[column]


def legal_actions(mask: int) -> list[int]:
    return [
        column
        for column in range(COLUMNS)
        if not mask & _TOP[column]
    ]


def play(current: int, mask: int, column: int) -> tuple[int, int]:
    """
    Joue dans une colonne pour le joueur qui a le trait.

    Retourne la nouvelle position, vue par le joueur suivant.
    """
    return current ^ mask, mask | (mask + _BOTTOM[column])


def has_four(stones: int) -> bool:
    """Retourne True si ces pions contiennent un alignement de quatre."""
    for shift in (1, _COLUMN_BITS - 1, _COLUMN_BITS, _COLUMN_BITS + 1):
        pairs = stones & (stones >> shift)

        if pairs & (pairs >> (2 * shift)):
            return True

    return False


def is_full(mask: int) -> bool:
    return mask == FULL


def from_board(board: Board, piece: Cell) -> tuple[int, int]:
    """Convertit un Board en position vue par le joueur `piece`."""
    current = 0
    mask = 0

    for row in range(ROWS):
        for column in range(COLUMNS):
            cell = board.get(row, column)

            if cell == Cell.EMPTY:
                continue

            bit = _bit(ROWS - 1 - row, column)
            mask |= bit

            if cell == piece:
                current |= bit

    return current, mask


# Bit de chaque case dans l'ordre des plans : ligne 0 en haut.
_PLANE_BITS = tuple(
    _bit(ROWS - 1 - row, column)
    for row in range(ROWS)
    for column in range(COLUMNS)
)


def encode_planes(current: int, mask: int) -> np.ndarray:
    """
    Encode la position pour le réseau : tableau [2, 6, 7].

    Plan 0 : pions du joueur qui doit jouer.
    Plan 1 : pions adverses.
    Ligne 0 en haut, comme Board et comme la démo du site.
    """
    opponent = current ^ mask
    planes = np.zeros(2 * ROWS * COLUMNS, dtype=np.uint8)

    for index, bit in enumerate(_PLANE_BITS):
        if current & bit:
            planes[index] = 1
        elif opponent & bit:
            planes[ROWS * COLUMNS + index] = 1

    return planes.reshape(2, ROWS, COLUMNS)
