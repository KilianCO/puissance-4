# -*- coding: utf-8 -*-

"""
Tests de l'environnement de reinforcement learning.
"""

import pytest

from P4.board import Board, Cell
from P4.players import Player
from P4.rl.environment import Connect4Environment


class FixedPlayer(Player):
    """
    Adversaire de test qui joue une séquence prédéfinie de colonnes.
    """

    def __init__(self, piece, actions):
        super().__init__(piece)
        self.actions = iter(actions)

    def choose_action(self, board):
        return next(self.actions)


def test_reset_returns_empty_state_when_agent_starts():
    env = Connect4Environment()

    state = env.reset()

    assert state == (0,) * (Board.ROWS * Board.COLUMNS)
    assert not env.done


def test_opponent_plays_first_when_agent_is_player_2():
    env = Connect4Environment(
        agent_piece=Cell.PLAYER_2,
        opponent=FixedPlayer(Cell.PLAYER_1, [3]),
    )

    state = env.reset()

    # Le pion adverse est vu comme un 2, en bas de la colonne 3.
    assert state.count(2) == 1
    assert state[(Board.ROWS - 1) * Board.COLUMNS + 3] == 2


def test_reset_with_agent_piece_keeps_the_opponent():
    opponent = FixedPlayer(Cell.PLAYER_2, [3])
    env = Connect4Environment(opponent=opponent)

    env.reset(agent_piece=Cell.PLAYER_2)

    assert env.opponent is opponent
    assert opponent.piece == Cell.PLAYER_1


def test_agent_win_gives_positive_reward():
    env = Connect4Environment(
        opponent=FixedPlayer(Cell.PLAYER_2, [6, 6, 6]),
    )
    env.reset()

    for _ in range(3):
        _, reward, done, _ = env.step(0)

        assert reward == 0.0
        assert not done

    _, reward, done, info = env.step(0)

    assert reward == 1.0
    assert done
    assert info["winner"] == Cell.PLAYER_1


def test_opponent_win_gives_negative_reward():
    env = Connect4Environment(
        opponent=FixedPlayer(Cell.PLAYER_2, [6, 6, 6, 6]),
    )
    env.reset()

    for column in (0, 1, 2):
        env.step(column)

    _, reward, done, info = env.step(4)

    assert reward == -1.0
    assert done
    assert info["winner"] == Cell.PLAYER_2


def test_step_rejects_illegal_action():
    env = Connect4Environment()
    env.reset()

    with pytest.raises(ValueError):
        env.step(7)


def test_step_after_end_raises_error():
    env = Connect4Environment(
        opponent=FixedPlayer(Cell.PLAYER_2, [6, 6, 6]),
    )
    env.reset()

    for _ in range(4):
        env.step(0)

    with pytest.raises(RuntimeError):
        env.step(1)
