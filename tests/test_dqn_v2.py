# -*- coding: utf-8 -*-

"""
Tests du DQN V2 : réseau, apprentissage, sauvegarde et adaptateur.
"""

import random

import pytest

torch = pytest.importorskip("torch")

from P4 import bitboard
from P4.board import Board, Cell
from P4.rl.agent_player import AgentPlayer, load_agent_player, planes_state
from P4.rl.dqn_v2 import ConvQNetwork, DQNv2Agent
from P4.rl.train_dqn_v2 import epsilon_at, play_training_game


def small_agent(**kwargs):
    return DQNv2Agent(
        batch_size=16,
        replay_capacity=500,
        channels=8,
        blocks=1,
        device="cpu",
        **kwargs,
    )


def test_network_follows_the_site_contract():
    network = ConvQNetwork(channels=8, blocks=1)

    output = network(torch.zeros(3, 2, 6, 7))

    assert output.shape == (3, 7)
    assert output.abs().max() <= 1.0


def test_choose_action_is_always_legal():
    agent = small_agent()
    state = bitboard.encode_planes(0, 0)

    for legal in ([0], [6], [2, 5], list(range(7))):
        assert agent.choose_action(state, legal) in legal

    agent.epsilon = 1.0

    assert agent.choose_action(state, [4], training=True) == 4


def test_agent_learns_a_winning_move():
    """
    Une seule position, où jouer en colonne 0 gagne : après quelques
    centaines d'étapes, l'agent doit préférer ce coup.
    """
    torch.manual_seed(0)
    random.seed(0)

    agent = small_agent(learning_rate=5e-3)

    # Le joueur au trait a trois pions en colonne 0.
    current, mask = 0, 0
    for column in (0, 6, 0, 6, 0, 5):
        current, mask = bitboard.play(current, mask, column)

    state = bitboard.encode_planes(current, mask)

    for column in bitboard.legal_actions(mask):
        after = bitboard.play(current, mask, column)
        won = bitboard.has_four(after[0] ^ after[1])

        for _ in range(8):
            agent.remember(
                state,
                column,
                1.0 if won else 0.0,
                bitboard.encode_planes(*after),
                [] if won else bitboard.legal_actions(after[1]),
                won,
            )

    for _ in range(300):
        assert agent.train_step() is not None

    values = agent.q_values(state)

    assert agent.choose_action(state, bitboard.legal_actions(mask)) == 0
    assert values[0] > 0.8


def test_training_game_fills_the_replay_buffer():
    random.seed(0)

    agent = small_agent()
    agent.epsilon = 1.0

    kind, outcome, _ = play_training_game(agent, train_every=2, warmup=10)

    # Une partie dure au moins 7 coups, et chacun est mémorisé.
    assert len(agent.replay_buffer) >= 7
    assert (outcome is None) == (kind == "self")


def test_save_and_load_round_trip(tmp_path):
    agent = small_agent()
    agent.episodes = 12
    agent.training_steps = 34

    path = tmp_path / "model.pt"
    agent.save(path)

    loaded = DQNv2Agent.load(path, device="cpu")
    state = bitboard.encode_planes(0, 0)

    assert loaded.episodes == 12
    assert loaded.training_steps == 34
    assert loaded.q_values(state) == pytest.approx(agent.q_values(state))


def test_loaded_model_plays_as_a_player(tmp_path):
    path = tmp_path / "model.pt"
    small_agent().save(path)

    player = load_agent_player(path)(Cell.PLAYER_2)
    board = Board()
    board.play(3, Cell.PLAYER_1)

    assert isinstance(player, AgentPlayer)
    assert player.choose_action(board) in board.legal_actions()


def test_planes_state_is_seen_by_the_given_player():
    board = Board()
    board.play(3, Cell.PLAYER_1)

    mine = planes_state(board, Cell.PLAYER_1)
    theirs = planes_state(board, Cell.PLAYER_2)

    assert mine[0, 5, 3] == 1 and mine[1].sum() == 0
    assert theirs[1, 5, 3] == 1 and theirs[0].sum() == 0


def test_epsilon_schedule():
    assert epsilon_at(0, 1000) == 1.0
    assert epsilon_at(500, 1000) == pytest.approx(0.05)
    assert epsilon_at(1000, 1000) == pytest.approx(0.05)
    assert 0.05 < epsilon_at(250, 1000) < 1.0
