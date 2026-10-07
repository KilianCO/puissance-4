# -*- coding: utf-8 -*-

"""
Tests de la recherche de Monte-Carlo et du classement Elo.
"""

import random

import pytest

from P4 import bitboard
from P4.board import Board, Cell
from P4.league import fit_elo, make_player, standings
from P4.mcts import mcts_move, mcts_visits
from P4.players import MonteCarloPlayer


def position(moves):
    current, mask = 0, 0

    for column in moves:
        current, mask = bitboard.play(current, mask, column)

    return current, mask


def test_mcts_spends_every_simulation_on_a_legal_move():
    random.seed(0)

    # Colonne 3 pleine.
    current, mask = position([3, 3, 3, 3, 3, 3])

    visits = mcts_visits(current, mask, 200)

    assert 3 not in visits
    assert sum(visits.values()) == 200


def test_mcts_takes_the_win():
    random.seed(0)

    # Le joueur au trait a trois pions en colonne 0.
    assert mcts_move(*position([0, 6, 0, 6, 0, 5]), 400) == 0


def test_mcts_blocks_the_threat():
    random.seed(0)

    # L'adversaire a trois pions en colonne 0.
    assert mcts_move(*position([0, 6, 0, 5, 0]), 2_000) == 0


def test_mcts_prefers_the_centre_on_an_empty_board():
    random.seed(0)

    visits = mcts_visits(0, 0, 2_000)

    assert max(visits, key=visits.get) == 3


def test_monte_carlo_player_returns_a_legal_action():
    random.seed(0)

    board = Board()
    board.play(3, Cell.PLAYER_1)

    player = MonteCarloPlayer(Cell.PLAYER_2, simulations=100)

    assert player.choose_action(board) in board.legal_actions()

    with pytest.raises(ValueError):
        MonteCarloPlayer(Cell.PLAYER_1, simulations=0)


def test_make_player_builds_every_kind():
    for spec in ("random", "tactical", "minimax:2:0", "mcts:50"):
        player = make_player(spec)(Cell.PLAYER_1)

        assert player.choose_action(Board()) in range(Board.COLUMNS)

    with pytest.raises(ValueError):
        make_player("inconnu")


def test_elo_orders_players_by_strength():
    # A bat B 30-10, B bat C 30-10, A bat C 36-4.
    results = {
        ("A", "B"): (30, 10, 0),
        ("B", "C"): (30, 10, 0),
        ("A", "C"): (36, 4, 0),
    }

    ratings = fit_elo(["A", "B", "C"], results)

    assert ratings["A"] > ratings["B"] > ratings["C"]
    assert sum(ratings.values()) / 3 == pytest.approx(1500)


def test_elo_matches_the_expected_score_formula():
    # Sans partie fictive, 75 % de score correspond à environ 191 points.
    ratings = fit_elo(["A", "B"], {("A", "B"): (75, 25, 0)}, prior_games=0)

    assert ratings["A"] - ratings["B"] == pytest.approx(190.8, abs=0.5)


def test_elo_stays_finite_for_a_player_who_never_wins():
    ratings = fit_elo(["A", "B"], {("A", "B"): (40, 0, 0)})

    assert 0 < ratings["A"] - ratings["B"] < 2_000


def test_standings_are_sorted_and_count_both_sides():
    roster = {"A": "random", "B": "tactical"}

    table = standings(roster, {("A", "B"): (5, 30, 5)})

    assert [row["name"] for row in table] == ["B", "A"]
    assert table[0]["wins"] == 30 and table[0]["losses"] == 5
    assert table[1]["games"] == 40
