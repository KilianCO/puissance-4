# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:44:30 2026

@author: kcollet
"""

### `P4/game.py`

from enum import Enum

from .board import Board, Cell
from .players import Player


class GameStatus(Enum):
    IN_PROGRESS = "in_progress"
    PLAYER_1_WON = "player_1_won"
    PLAYER_2_WON = "player_2_won"
    DRAW = "draw"


class Game:
    def __init__(self, player_1: Player, player_2: Player):
        if player_1.piece == player_2.piece:
            raise ValueError("Players must have different pieces.")

        self.board = Board()

        self.players = {
            Cell.PLAYER_1: player_1,
            Cell.PLAYER_2: player_2,
        }

        self.current_player = Cell.PLAYER_1
        self.status = GameStatus.IN_PROGRESS

    def reset(self) -> None:
        """Commence une nouvelle partie."""
        self.board.reset()
        self.current_player = Cell.PLAYER_1
        self.status = GameStatus.IN_PROGRESS

    def play_turn(self) -> None:
        """
        Joue un tour.

        Le joueur courant choisit une action.
        Le moteur applique ensuite cette action et met à jour l'état
        de la partie.
        """
        if self.status != GameStatus.IN_PROGRESS:
            raise RuntimeError("The game is already finished.")

        player = self.players[self.current_player]

        action = player.choose_action(self.board)

        if not self.board.is_valid_column(action):
            raise ValueError(
                f"Player selected invalid column: {action}"
            )

        self.board.play(action, self.current_player)

        if self.board.check_winner(self.current_player):
            self.status = (
                GameStatus.PLAYER_1_WON
                if self.current_player == Cell.PLAYER_1
                else GameStatus.PLAYER_2_WON
            )
            return

        if self.board.is_full():
            self.status = GameStatus.DRAW
            return

        self._switch_player()

    def _switch_player(self) -> None:
        if self.current_player == Cell.PLAYER_1:
            self.current_player = Cell.PLAYER_2
        else:
            self.current_player = Cell.PLAYER_1

    def play(self) -> GameStatus:
        """Joue automatiquement la partie jusqu'à son terme."""
        while self.status == GameStatus.IN_PROGRESS:
            self.play_turn()

        return self.status