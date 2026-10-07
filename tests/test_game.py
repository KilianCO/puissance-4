# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:51:18 2026

@author: kcollet
"""

import pytest

from P4.board import Cell
from P4.game import Game, GameStatus
from P4.players import Player


class FixedPlayer(Player):
    """
    Joueur de test qui joue une séquence prédéfinie de colonnes.
    """

    def __init__(self, piece, actions):
        super().__init__(piece)
        self.actions = iter(actions)

    def choose_action(self, board):
        return next(self.actions)


def test_game_starts_with_player_1():
    player_1 = FixedPlayer(Cell.PLAYER_1, [])
    player_2 = FixedPlayer(Cell.PLAYER_2, [])

    game = Game(player_1, player_2)

    assert game.current_player == Cell.PLAYER_1
    assert game.status == GameStatus.IN_PROGRESS


def test_players_must_have_different_pieces():
    player_1 = FixedPlayer(Cell.PLAYER_1, [])
    player_2 = FixedPlayer(Cell.PLAYER_1, [])

    with pytest.raises(ValueError):
        Game(player_1, player_2)


def test_turn_switches_after_valid_move():
    player_1 = FixedPlayer(Cell.PLAYER_1, [0])
    player_2 = FixedPlayer(Cell.PLAYER_2, [1])

    game = Game(player_1, player_2)

    game.play_turn()

    assert game.current_player == Cell.PLAYER_2

    game.play_turn()

    assert game.current_player == Cell.PLAYER_1


def test_player_1_can_win():
    player_1 = FixedPlayer(
        Cell.PLAYER_1,
        [0, 1, 2, 3],
    )

    player_2 = FixedPlayer(
        Cell.PLAYER_2,
        [4, 4, 5],
    )

    game = Game(player_1, player_2)

    status = game.play()

    assert status == GameStatus.PLAYER_1_WON


def test_player_2_can_win():
    player_1 = FixedPlayer(
        Cell.PLAYER_1,
        [0, 1, 2, 6],
    )

    player_2 = FixedPlayer(
        Cell.PLAYER_2,
        [4, 4, 4, 4],
    )

    game = Game(player_1, player_2)

    status = game.play()

    assert status == GameStatus.PLAYER_2_WON


def test_game_cannot_continue_after_win():
    player_1 = FixedPlayer(
        Cell.PLAYER_1,
        [0, 1, 2, 3],
    )

    player_2 = FixedPlayer(
        Cell.PLAYER_2,
        [4, 4, 5],
    )

    game = Game(player_1, player_2)

    game.play()

    with pytest.raises(RuntimeError):
        game.play_turn()


def test_invalid_player_action_raises_error():
    player_1 = FixedPlayer(Cell.PLAYER_1, [7])
    player_2 = FixedPlayer(Cell.PLAYER_2, [])

    game = Game(player_1, player_2)

    with pytest.raises(ValueError):
        game.play_turn()


def test_reset_starts_new_game():
    player_1 = FixedPlayer(Cell.PLAYER_1, [0])
    player_2 = FixedPlayer(Cell.PLAYER_2, [1])

    game = Game(player_1, player_2)

    game.play_turn()
    game.play_turn()

    game.reset()

    assert game.status == GameStatus.IN_PROGRESS
    assert game.current_player == Cell.PLAYER_1

    for row in range(game.board.ROWS):
        for column in range(game.board.COLUMNS):
            assert game.board.get(row, column) == Cell.EMPTY