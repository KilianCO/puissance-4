# -*- coding: utf-8 -*-

"""
Entraînement d'un agent Q-learning.
"""

from pathlib import Path

from P4.rl.environment import Connect4Environment
from P4.rl.q_learning import QLearningAgent


# Chemin ancré à la racine du dépôt, quel que soit le dossier courant.
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "q_table.pkl"


def train(
    number_of_episodes: int = 100_000,
) -> QLearningAgent:
    """
    Entraîne un agent Q-learning.

    Parameters
    ----------
    number_of_episodes : int
        Nombre de parties d'entraînement.

    Returns
    -------
    QLearningAgent
        Agent entraîné.
    """

    env = Connect4Environment()

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99995,
    )

    for episode in range(number_of_episodes):

        state = env.reset()
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
                _,
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

        agent.decay_epsilon()

        if (episode + 1) % 10_000 == 0:
            print(
                f"Episode {episode + 1:>7} | "
                f"epsilon = {agent.epsilon:.4f} | "
                f"states = {len(agent.q_table)}"
            )

    return agent


def main() -> None:
    number_of_episodes = 100_000

    print("=" * 50)
    print("          ENTRAÎNEMENT Q-LEARNING")
    print("=" * 50)
    print()
    print(
        f"Nombre d'épisodes : {number_of_episodes:,}"
    )

    agent = train(number_of_episodes)

    agent.save(MODEL_PATH)

    print()
    print(
        f"Modèle sauvegardé dans : {MODEL_PATH}"
    )
    print(
        f"Nombre d'états appris : {len(agent.q_table):,}"
    )
    print(
        f"Epsilon final         : {agent.epsilon:.4f}"
    )


if __name__ == "__main__":
    main()