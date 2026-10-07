# -*- coding: utf-8 -*-

"""
Adaptateur : fait jouer un agent entraîné comme n'importe quel Player.

Les agents choisissent un coup à partir d'un état encodé ; les Player à
partir d'un Board. AgentPlayer fait le lien, ce qui rend les modèles
utilisables dans Game, dans l'arène et dans l'interface en ligne de
commande.
"""

from pathlib import Path
from typing import Callable

import torch

from P4.bitboard import encode_planes, from_board
from P4.board import Board, Cell
from P4.players import Player


def flat_state(board: Board, piece: Cell) -> tuple[int, ...]:
    """Encodage V1 : 42 valeurs, 0 = vide, 1 = moi, 2 = adversaire."""
    state = []

    for row in range(board.ROWS):
        for column in range(board.COLUMNS):
            cell = board.get(row, column)

            if cell == Cell.EMPTY:
                state.append(0)
            elif cell == piece:
                state.append(1)
            else:
                state.append(2)

    return tuple(state)


def planes_state(board: Board, piece: Cell):
    """Encodage V2 : deux plans 6 x 7, vus par le joueur `piece`."""
    return encode_planes(*from_board(board, piece))


class AgentPlayer(Player):
    """
    Joueur piloté par un agent entraîné, sans exploration.

    Parameters
    ----------
    agent
        Objet exposant choose_action(state, legal_actions, training).

    encode
        Fonction (board, piece) -> état attendu par l'agent.
    """

    def __init__(
        self,
        piece: Cell,
        agent,
        encode: Callable,
    ):
        super().__init__(piece)

        self.agent = agent
        self.encode = encode

    def choose_action(self, board: Board) -> int:
        return self.agent.choose_action(
            self.encode(board, self.piece),
            board.legal_actions(),
            training=False,
        )


def load_agent_player(
    path: str | Path,
) -> Callable[[Cell], AgentPlayer]:
    """
    Charge un modèle et retourne une fabrique `piece -> AgentPlayer`.

    Formats reconnus : Q-table (.pkl), DQN V1 et DQN V2 (.pt).
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix == ".pkl":
        from P4.rl.q_learning import QLearningAgent

        agent = QLearningAgent()
        agent.load(path)
        encode = flat_state

    else:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint.get("format") == "dqn_v2":
            from P4.rl.dqn_v2 import DQNv2Agent

            # Jouer ne demande pas de replay buffer.
            agent = DQNv2Agent.load(path, replay_capacity=1)
            agent.online_network.eval()
            encode = planes_state

        else:
            from P4.rl.dqn import DQNAgent

            agent = DQNAgent()
            agent.load(path)
            encode = flat_state

    return lambda piece: AgentPlayer(piece, agent, encode)
