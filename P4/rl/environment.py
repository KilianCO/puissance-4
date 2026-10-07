# -*- coding: utf-8 -*-

"""
Environnement de reinforcement learning pour le Puissance 4.

L'état est toujours représenté du point de vue de l'agent :

    0 = vide
    1 = agent
    2 = adversaire

L'agent peut être PLAYER_1 ou PLAYER_2.
"""

import random

from P4.board import Board, Cell
from P4.players import Player, RandomPlayer


class Connect4Environment:
    """
    Environnement RL du Puissance 4.

    Parameters
    ----------
    agent_piece : Cell
        Pièce contrôlée par l'agent.

    opponent : Player | None
        Adversaire.
    """

    def __init__(
        self,
        agent_piece: Cell = Cell.PLAYER_1,
        opponent: Player | None = None,
    ):
        if agent_piece not in (
            Cell.PLAYER_1,
            Cell.PLAYER_2,
        ):
            raise ValueError(
                "agent_piece must be PLAYER_1 or PLAYER_2."
            )

        opponent_piece = (
            Cell.PLAYER_2
            if agent_piece == Cell.PLAYER_1
            else Cell.PLAYER_1
        )

        if opponent is None:
            opponent = RandomPlayer(opponent_piece)

        if opponent.piece != opponent_piece:
            raise ValueError(
                "Opponent has an incompatible piece."
            )

        self.agent_piece = agent_piece
        self.opponent_piece = opponent_piece
        self.opponent = opponent

        self.board = Board()
        self.done = False

    def reset(
        self,
        agent_piece: Cell | None = None,
    ) -> tuple[int, ...]:
        """
        Réinitialise la partie.

        Si agent_piece est fourni, il devient la pièce
        contrôlée par l'agent pour cette partie.
        """

        if agent_piece is not None:
            if agent_piece not in (
                Cell.PLAYER_1,
                Cell.PLAYER_2,
            ):
                raise ValueError(
                    "agent_piece must be PLAYER_1 or PLAYER_2."
                )

            self.agent_piece = agent_piece

            self.opponent_piece = (
                Cell.PLAYER_2
                if agent_piece == Cell.PLAYER_1
                else Cell.PLAYER_1
            )

            # L'adversaire reste le même : seule sa pièce change.
            self.opponent.piece = self.opponent_piece

        self.board.reset()
        self.done = False

        # Si l'agent joue second, l'adversaire commence.
        if self.agent_piece == Cell.PLAYER_2:
            self._play_opponent_turn()

        return self.get_state()

    def get_state(self) -> tuple[int, ...]:
        """
        Retourne le plateau vu par l'agent.

        0 = vide
        1 = agent
        2 = adversaire
        """

        state = []

        for row in range(self.board.ROWS):
            for column in range(self.board.COLUMNS):

                cell = self.board.get(row, column)

                if cell == Cell.EMPTY:
                    state.append(0)

                elif cell == self.agent_piece:
                    state.append(1)

                elif cell == self.opponent_piece:
                    state.append(2)

                else:
                    raise RuntimeError(
                        f"Unexpected cell value: {cell}"
                    )

        return tuple(state)

    def legal_actions(self) -> list[int]:
        """Retourne les colonnes actuellement jouables."""
        return self.board.legal_actions()

    def step(
        self,
        action: int,
    ) -> tuple[tuple[int, ...], float, bool, dict]:
        """
        Effectue le coup de l'agent puis celui de l'adversaire.
        """

        if self.done:
            raise RuntimeError(
                "The environment is already finished."
            )

        if action not in self.legal_actions():
            raise ValueError(
                f"Illegal action: {action}"
            )

        # -----------------------------
        # Coup de l'agent
        # -----------------------------

        self.board.play(
            action,
            self.agent_piece,
        )

        if self.board.check_winner(
            self.agent_piece
        ):
            self.done = True

            return (
                self.get_state(),
                1.0,
                True,
                {"winner": self.agent_piece},
            )

        if self.board.is_full():
            self.done = True

            return (
                self.get_state(),
                0.0,
                True,
                {"winner": None},
            )

        # -----------------------------
        # Coup de l'adversaire
        # -----------------------------

        self._play_opponent_turn()

        if self.board.check_winner(
            self.opponent_piece
        ):
            self.done = True

            return (
                self.get_state(),
                -1.0,
                True,
                {"winner": self.opponent_piece},
            )

        if self.board.is_full():
            self.done = True

            return (
                self.get_state(),
                0.0,
                True,
                {"winner": None},
            )

        return (
            self.get_state(),
            0.0,
            False,
            {"winner": None},
        )

    def _play_opponent_turn(self) -> None:
        """Fait jouer l'adversaire."""

        legal_actions = self.board.legal_actions()

        if not legal_actions:
            raise RuntimeError(
                "Opponent has no legal action."
            )

        action = self.opponent.choose_action(
            self.board
        )

        if action not in legal_actions:
            raise RuntimeError(
                f"Opponent selected illegal action: {action}"
            )

        self.board.play(
            action,
            self.opponent_piece,
        )