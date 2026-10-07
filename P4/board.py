# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:43:02 2026

@author: kcollet

Moteur du jeu
"""

### `P4/board.py`

from enum import IntEnum


class Cell(IntEnum):
    EMPTY = 0
    PLAYER_1 = 1
    PLAYER_2 = 2


class Board:
    ROWS = 6
    COLUMNS = 7
    CONNECT = 4

    def __init__(self):
        self._grid = [
            [Cell.EMPTY for _ in range(self.COLUMNS)]
            for _ in range(self.ROWS)
        ]

    def reset(self) -> None:
        """Remet le plateau dans son état initial."""
        for row in range(self.ROWS):
            for column in range(self.COLUMNS):
                self._grid[row][column] = Cell.EMPTY

    def is_valid_column(self, column: int) -> bool:
        """Retourne True si un pion peut être joué dans cette colonne."""
        return (
            0 <= column < self.COLUMNS
            and self._grid[0][column] == Cell.EMPTY
        )

    def play(self, column: int, player: Cell) -> tuple[int, int]:
        """
        Joue un pion dans une colonne.

        Retourne les coordonnées (row, column) du pion placé.

        Lève ValueError si :
        - la colonne est invalide ;
        - la colonne est pleine ;
        - le joueur est invalide.
        """
        if player not in (Cell.PLAYER_1, Cell.PLAYER_2):
            raise ValueError("Invalid player.")

        if not self.is_valid_column(column):
            raise ValueError("Invalid column.")

        for row in range(self.ROWS - 1, -1, -1):
            if self._grid[row][column] == Cell.EMPTY:
                self._grid[row][column] = player
                return row, column

        # Cette ligne ne devrait jamais être atteinte puisque
        # is_valid_column() a vérifié que la colonne n'était pas pleine.
        raise RuntimeError("Could not place the piece.")

    def get(self, row: int, column: int) -> Cell:
        """Retourne le contenu d'une case."""
        if not (0 <= row < self.ROWS):
            raise IndexError("Invalid row.")

        if not (0 <= column < self.COLUMNS):
            raise IndexError("Invalid column.")

        return self._grid[row][column]

    def is_full(self) -> bool:
        """Retourne True si le plateau est complètement rempli."""
        return all(
            self._grid[0][column] != Cell.EMPTY
            for column in range(self.COLUMNS)
        )

    def check_winner(self, player: Cell) -> bool:
        """Retourne True si le joueur possède un alignement de quatre."""

        if player not in (Cell.PLAYER_1, Cell.PLAYER_2):
            return False

        directions = (
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonale descendante
            (1, -1),  # diagonale montante
        )

        for row in range(self.ROWS):
            for column in range(self.COLUMNS):
                if self._grid[row][column] != player:
                    continue

                for delta_row, delta_column in directions:
                    if self._has_connect_four(
                        row,
                        column,
                        delta_row,
                        delta_column,
                        player,
                    ):
                        return True

        return False

    def _has_connect_four(
        self,
        row: int,
        column: int,
        delta_row: int,
        delta_column: int,
        player: Cell,
    ) -> bool:
        """Vérifie un alignement de quatre depuis une case donnée."""

        for offset in range(self.CONNECT):
            current_row = row + offset * delta_row
            current_column = column + offset * delta_column

            if not (
                0 <= current_row < self.ROWS
                and 0 <= current_column < self.COLUMNS
            ):
                return False

            if self._grid[current_row][current_column] != player:
                return False

        return True

    def legal_actions(self) -> list[int]:
        """
        Retourne les colonnes dans lesquelles un pion peut être joué.

        Cette méthode sera particulièrement importante pour le RL :
        elle permettra à un agent de connaître les actions actuellement
        disponibles.
        """
        return [
            column
            for column in range(self.COLUMNS)
            if self.is_valid_column(column)
        ]

    def copy(self) -> "Board":
        """Retourne une copie indépendante du plateau."""
        board = Board()

        for row in range(self.ROWS):
            for column in range(self.COLUMNS):
                board._grid[row][column] = self._grid[row][column]

        return board