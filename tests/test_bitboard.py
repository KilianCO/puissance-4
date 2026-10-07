# -*- coding: utf-8 -*-

"""
Tests de la représentation compacte et des adversaires par recherche.

Board est la référence : on vérifie que bitboard donne toujours les
mêmes réponses sur des parties aléatoires.
"""

import random

import pytest

from P4 import bitboard
from P4.board import Board, Cell
from P4.minimax import minimax_move, minimax_scores, tactical_move, WIN
from P4.players import MinimaxPlayer, TacticalPlayer


def other(piece):
    return Cell.PLAYER_2 if piece == Cell.PLAYER_1 else Cell.PLAYER_1


def build(moves):
    """Joue une suite de colonnes en alternant, PLAYER_1 d'abord."""
    board = Board()
    piece = Cell.PLAYER_1

    for column in moves:
        board.play(column, piece)
        piece = other(piece)

    return board, piece


def test_there_are_69_windows():
    assert len(bitboard.WINDOWS) == 69


def test_bitboard_agrees_with_board_on_random_games():
    rng = random.Random(0)

    for _ in range(300):
        board = Board()
        current, mask = 0, 0
        piece = Cell.PLAYER_1

        while True:
            assert bitboard.from_board(board, piece) == (current, mask)
            assert bitboard.legal_actions(mask) == board.legal_actions()

            column = rng.choice(board.legal_actions())

            board.play(column, piece)
            current, mask = bitboard.play(current, mask, column)

            # Après play, les pions de celui qui vient de jouer
            # sont current ^ mask.
            won = bitboard.has_four(current ^ mask)

            assert won == board.check_winner(piece)
            assert bitboard.is_full(mask) == board.is_full()

            if won or board.is_full():
                break

            piece = other(piece)


def test_encode_planes_matches_board():
    board, piece = build([3, 3, 4, 0, 6])

    planes = bitboard.encode_planes(*bitboard.from_board(board, piece))

    assert planes.shape == (2, Board.ROWS, Board.COLUMNS)

    for row in range(Board.ROWS):
        for column in range(Board.COLUMNS):
            cell = board.get(row, column)

            assert planes[0, row, column] == (cell == piece)
            assert planes[1, row, column] == (cell == other(piece))


def test_tactical_move_takes_the_win():
    # PLAYER_1 a trois pions en colonne 0 et doit jouer.
    board, piece = build([0, 6, 0, 6, 0, 5])

    assert tactical_move(*bitboard.from_board(board, piece)) == 0


def test_tactical_move_blocks_the_threat():
    # PLAYER_2 doit jouer et PLAYER_1 menace en colonne 0.
    board, piece = build([0, 6, 0, 5, 0])

    assert tactical_move(*bitboard.from_board(board, piece)) == 0


def test_tactical_move_prefers_winning_to_blocking():
    # PLAYER_1 peut gagner en 0 ; PLAYER_2 menace en 6.
    board, piece = build([0, 6, 0, 6, 0, 6])

    assert tactical_move(*bitboard.from_board(board, piece)) == 0


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_minimax_takes_the_win(depth):
    board, piece = build([0, 6, 0, 6, 0, 5])

    position = bitboard.from_board(board, piece)

    assert minimax_move(*position, depth) == 0
    assert minimax_scores(*position, depth)[0] >= WIN


@pytest.mark.parametrize("depth", [2, 4])
def test_minimax_blocks_the_threat(depth):
    board, piece = build([0, 6, 0, 5, 0])

    assert minimax_move(*bitboard.from_board(board, piece), depth) == 0


def test_minimax_sees_a_double_threat_two_moves_ahead():
    # PLAYER_1 a deux pions en bas (colonnes 2 et 3), bords libres :
    # jouer en 1 ou en 4 crée une menace double imparable.
    board, piece = build([2, 2, 3, 3])

    scores = minimax_scores(*bitboard.from_board(board, piece), 4)

    assert scores[1] >= WIN
    assert scores[4] >= WIN
    assert scores[0] < WIN


def test_players_return_legal_actions():
    rng = random.Random(1)

    for player_class in (TacticalPlayer, MinimaxPlayer):
        board = Board()
        piece = Cell.PLAYER_1

        for _ in range(12):
            player = player_class(piece)
            column = player.choose_action(board)

            assert column in board.legal_actions()

            board.play(rng.choice(board.legal_actions()), piece)

            if board.check_winner(piece):
                break

            piece = other(piece)


def test_minimax_player_rejects_invalid_depth():
    with pytest.raises(ValueError):
        MinimaxPlayer(Cell.PLAYER_1, depth=0)
