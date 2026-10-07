# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:50:45 2026

@author: kcollet
"""

import pytest

from P4.board import Board, Cell


def test_board_is_empty_at_creation():
    board = Board()

    for row in range(Board.ROWS):
        for column in range(Board.COLUMNS):
            assert board.get(row, column) == Cell.EMPTY


def test_board_dimensions():
    board = Board()

    assert len(board._grid) == Board.ROWS
    assert all(len(row) == Board.COLUMNS for row in board._grid)


def test_play_piece_falls_to_bottom():
    board = Board()

    row, column = board.play(3, Cell.PLAYER_1)

    assert (row, column) == (5, 3)
    assert board.get(5, 3) == Cell.PLAYER_1


def test_pieces_stack_from_bottom():
    board = Board()

    board.play(3, Cell.PLAYER_1)
    board.play(3, Cell.PLAYER_2)
    board.play(3, Cell.PLAYER_1)

    assert board.get(5, 3) == Cell.PLAYER_1
    assert board.get(4, 3) == Cell.PLAYER_2
    assert board.get(3, 3) == Cell.PLAYER_1


def test_play_invalid_column():
    board = Board()

    with pytest.raises(ValueError):
        board.play(-1, Cell.PLAYER_1)

    with pytest.raises(ValueError):
        board.play(Board.COLUMNS, Cell.PLAYER_1)


def test_play_invalid_player():
    board = Board()

    with pytest.raises(ValueError):
        board.play(0, Cell.EMPTY)


def test_full_column_is_not_valid():
    board = Board()

    for _ in range(Board.ROWS):
        board.play(0, Cell.PLAYER_1)

    assert not board.is_valid_column(0)


def test_play_in_full_column_raises_error():
    board = Board()

    for _ in range(Board.ROWS):
        board.play(0, Cell.PLAYER_1)

    with pytest.raises(ValueError):
        board.play(0, Cell.PLAYER_2)


def test_legal_actions_initially_contains_all_columns():
    board = Board()

    assert board.legal_actions() == list(range(Board.COLUMNS))


def test_legal_actions_excludes_full_columns():
    board = Board()

    for _ in range(Board.ROWS):
        board.play(2, Cell.PLAYER_1)

    assert 2 not in board.legal_actions()
    assert len(board.legal_actions()) == Board.COLUMNS - 1


def test_horizontal_win():
    board = Board()

    for column in range(4):
        board.play(column, Cell.PLAYER_1)

    assert board.check_winner(Cell.PLAYER_1)
    assert not board.check_winner(Cell.PLAYER_2)


def test_vertical_win():
    board = Board()

    for _ in range(4):
        board.play(0, Cell.PLAYER_1)

    assert board.check_winner(Cell.PLAYER_1)


def test_diagonal_descending_win():
    board = Board()

    # Construction de :
    #
    # X
    # O X
    # O O X
    # O O O X

    board.play(0, Cell.PLAYER_2)
    board.play(0, Cell.PLAYER_2)
    board.play(0, Cell.PLAYER_2)
    board.play(0, Cell.PLAYER_1)

    board.play(1, Cell.PLAYER_2)
    board.play(1, Cell.PLAYER_2)
    board.play(1, Cell.PLAYER_1)

    board.play(2, Cell.PLAYER_2)
    board.play(2, Cell.PLAYER_1)

    board.play(3, Cell.PLAYER_1)

    assert board.check_winner(Cell.PLAYER_1)


def test_diagonal_ascending_win():
    board = Board()

    # Construction de :
    #
    #       X
    #     X O
    #   X O O
    # X O O O

    board.play(0, Cell.PLAYER_1)

    board.play(1, Cell.PLAYER_2)
    board.play(1, Cell.PLAYER_1)

    board.play(2, Cell.PLAYER_2)
    board.play(2, Cell.PLAYER_2)
    board.play(2, Cell.PLAYER_1)

    board.play(3, Cell.PLAYER_2)
    board.play(3, Cell.PLAYER_2)
    board.play(3, Cell.PLAYER_2)
    board.play(3, Cell.PLAYER_1)

    assert board.check_winner(Cell.PLAYER_1)


def test_three_in_a_row_is_not_a_win():
    board = Board()

    for column in range(3):
        board.play(column, Cell.PLAYER_1)

    assert not board.check_winner(Cell.PLAYER_1)


def test_other_player_does_not_win():
    board = Board()

    for column in range(4):
        board.play(column, Cell.PLAYER_1)

    assert not board.check_winner(Cell.PLAYER_2)


def test_empty_player_does_not_win():
    board = Board()

    assert not board.check_winner(Cell.EMPTY)


def test_board_is_not_full_initially():
    board = Board()

    assert not board.is_full()


def test_board_is_full():
    board = Board()

    for column in range(Board.COLUMNS):
        for _ in range(Board.ROWS):
            board.play(column, Cell.PLAYER_1)

    assert board.is_full()


def test_copy_creates_independent_board():
    board = Board()
    board.play(3, Cell.PLAYER_1)

    copied_board = board.copy()

    assert copied_board.get(5, 3) == Cell.PLAYER_1

    copied_board.play(3, Cell.PLAYER_2)

    assert board.get(4, 3) == Cell.EMPTY
    assert copied_board.get(4, 3) == Cell.PLAYER_2


def test_reset_clears_board():
    board = Board()

    board.play(0, Cell.PLAYER_1)
    board.play(1, Cell.PLAYER_2)

    board.reset()

    for row in range(Board.ROWS):
        for column in range(Board.COLUMNS):
            assert board.get(row, column) == Cell.EMPTY