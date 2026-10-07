# -*- coding: utf-8 -*-

"""
Tests des joueurs de Puissance 4.
"""

import pytest

from P4.board import Board, Cell
from P4.players import Player, RandomPlayer


def test_player_accepts_valid_piece():
    player_1 = RandomPlayer(Cell.PLAYER_1)
    player_2 = RandomPlayer(Cell.PLAYER_2)

    assert player_1.piece == Cell.PLAYER_1
    assert player_2.piece == Cell.PLAYER_2


def test_player_rejects_invalid_piece():
    with pytest.raises(ValueError):
        RandomPlayer(Cell.EMPTY)


def test_random_player_returns_legal_action():
    board = Board()
    player = RandomPlayer(Cell.PLAYER_1)

    for _ in range(100):
        action = player.choose_action(board)

        assert action in board.legal_actions()


def test_random_player_does_not_modify_board():
    board = Board()
    player = RandomPlayer(Cell.PLAYER_1)

    before = board.copy()

    player.choose_action(board)

    for row in range(board.ROWS):
        for column in range(board.COLUMNS):
            assert board.get(row, column) == before.get(row, column)


def test_random_player_does_not_choose_full_column():
    board = Board()

    for _ in range(Board.ROWS):
        board.play(3, Cell.PLAYER_1)

    player = RandomPlayer(Cell.PLAYER_2)

    for _ in range(100):
        action = player.choose_action(board)

        assert action != 3
        assert action in board.legal_actions()


def test_random_player_can_choose_all_legal_columns():
    board = Board()
    player = RandomPlayer(Cell.PLAYER_1)

    chosen_actions = set()

    for _ in range(1000):
        chosen_actions.add(player.choose_action(board))

    assert chosen_actions == set(board.legal_actions())


def test_random_player_raises_when_board_is_full():
    board = Board()

    for column in range(board.COLUMNS):
        for _ in range(board.ROWS):
            board.play(column, Cell.PLAYER_1)

    player = RandomPlayer(Cell.PLAYER_1)

    with pytest.raises(RuntimeError):
        player.choose_action(board)
