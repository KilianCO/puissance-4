# -*- coding: utf-8 -*-

"""
Entraînement du Q-learning v2.

Le modèle apprend alternativement à jouer premier
et second.
"""

import random
from pathlib import Path

from P4.board import Cell
from P4.rl.environment import Connect4Environment
from P4.rl.q_learning import QLearningAgent


# Chemin ancré à la racine du dépôt, quel que soit le dossier courant.
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "q_learning_v2.pkl"


def train(
    number_of_episodes: int = 200_000,
) -> QLearningAgent:

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99995,
    )

    env = Connect4Environment()

    wins = 0
    losses = 0
    draws = 0

    for episode in range(number_of_episodes):

        # Une partie sur deux environ, l'agent commence.
        if random.random() < 0.5:
            agent_piece = Cell.PLAYER_1
        else:
            agent_piece = Cell.PLAYER_2

        state = env.reset(
            agent_piece=agent_piece
        )

        done = False

        while not done:

            legal_actions = env.legal_actions()

            action = agent.choose_action(
                state,
                legal_actions,
                training=True,
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            if done:
                next_legal_actions = []
            else:
                next_legal_actions = env.legal_actions()

            agent.update(
                state,
                action,
                reward,
                next_state,
                next_legal_actions,
                done,
            )

            state = next_state

        winner = info["winner"]

        if winner == agent_piece:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

        agent.decay_epsilon()

        if (episode + 1) % 10_000 == 0:

            total = wins + losses + draws

            print(
                f"Episode {episode + 1:>7} | "
                f"epsilon = {agent.epsilon:.4f} | "
                f"states = {len(agent.q_table):>7} | "
                f"win = {100 * wins / total:6.2f}% | "
                f"loss = {100 * losses / total:6.2f}% | "
                f"draw = {100 * draws / total:6.2f}%"
            )

            wins = 0
            losses = 0
            draws = 0

    return agent


def main() -> None:

    number_of_episodes = 200_000

    print("=" * 60)
    print("          Q-LEARNING V2")
    print("=" * 60)
    print()
    print(
        f"Episodes : {number_of_episodes:,}"
    )

    agent = train(number_of_episodes)

    agent.save(MODEL_PATH)

    print()
    print(
        f"Modèle sauvegardé : {MODEL_PATH}"
    )
    print(
        f"États appris      : "
        f"{len(agent.q_table):,}"
    )
    print(
        f"Epsilon final     : "
        f"{agent.epsilon:.4f}"
    )


if __name__ == "__main__":
    main()