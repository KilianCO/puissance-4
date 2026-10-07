# -*- coding: utf-8 -*-

"""
Agent Q-learning tabulaire pour le Puissance 4.
"""

import pickle
import random
from collections import defaultdict
from pathlib import Path


class QLearningAgent:
    """
    Agent utilisant le Q-learning tabulaire.

    Parameters
    ----------
    learning_rate : float
        Vitesse d'apprentissage.

    discount_factor : float
        Importance accordée aux récompenses futures.

    epsilon : float
        Probabilité d'exploration pendant l'entraînement.

    epsilon_min : float
        Valeur minimale d'epsilon.

    epsilon_decay : float
        Facteur de décroissance d'epsilon.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99995,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(
            lambda: defaultdict(float)
        )

    def choose_action(
        self,
        state: tuple[int, ...],
        legal_actions: list[int],
        training: bool = True,
    ) -> int:
        """
        Choisit une action selon une politique epsilon-greedy.

        Pendant l'entraînement :
            - exploration avec probabilité epsilon ;
            - exploitation sinon.

        Pendant l'évaluation :
            - aucune exploration ;
            - exploitation uniquement.
        """

        if not legal_actions:
            raise RuntimeError(
                "No legal action available."
            )

        # Exploration
        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Exploitation
        q_values = [
            self.q_table[state][action]
            for action in legal_actions
        ]

        max_q = max(q_values)

        # Plusieurs actions peuvent avoir exactement la même
        # valeur Q. On choisit aléatoirement entre elles afin
        # d'éviter un biais systématique vers la première colonne.
        best_actions = [
            action
            for action in legal_actions
            if self.q_table[state][action] == max_q
        ]

        return random.choice(best_actions)

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        next_legal_actions: list[int],
        done: bool,
    ) -> None:
        """
        Met à jour Q(state, action).
        """

        current_q = self.q_table[state][action]

        if done:
            target = reward

        else:
            if not next_legal_actions:
                raise RuntimeError(
                    "No legal action available in non-terminal state."
                )

            next_q = max(
                self.q_table[next_state][next_action]
                for next_action in next_legal_actions
            )

            target = (
                reward
                + self.discount_factor * next_q
            )

        self.q_table[state][action] = (
            current_q
            + self.learning_rate
            * (target - current_q)
        )

    def decay_epsilon(self) -> None:
        """
        Réduit progressivement l'exploration.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path: str | Path) -> None:
        """
        Sauvegarde la Q-table sur disque.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # On convertit les defaultdicts en dictionnaires
        # classiques avant sauvegarde.
        data = {
            state: dict(actions)
            for state, actions in self.q_table.items()
        }

        with path.open("wb") as file:
            pickle.dump(data, file)

    def load(self, path: str | Path) -> None:
        """
        Charge une Q-table depuis le disque.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        with path.open("rb") as file:
            data = pickle.load(file)

        self.q_table = defaultdict(
            lambda: defaultdict(float)
        )

        for state, actions in data.items():
            self.q_table[state].update(actions)
