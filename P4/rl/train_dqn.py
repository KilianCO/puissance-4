# -*- coding: utf-8 -*-

"""
Entraînement du DQN sur le Puissance 4.

DQN V1.

Affiche :

- epsilon
- taille du replay buffer
- nombre de training steps
- victoires / défaites / nuls
- loss
- vitesse
- temps écoulé
- temps restant estimé
"""

import random
import time
from pathlib import Path

import torch

from P4.board import Cell
from P4.rl.dqn import DQNAgent
from P4.rl.environment import Connect4Environment


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "models/dqn_v1.pt"
)

NUMBER_OF_EPISODES = 10_000

PRINT_EVERY = 1_000


# ============================================================
# Entraînement
# ============================================================

def train(
    number_of_episodes: int,
) -> DQNAgent:

    # --------------------------------------------------------
    # Création de l'agent
    # --------------------------------------------------------

    agent = DQNAgent(
        learning_rate=1e-3,

        discount_factor=0.99,

        epsilon=1.0,

        epsilon_min=0.05,

        epsilon_decay=0.99995,

        batch_size=128,

        replay_capacity=100_000,

        target_update_frequency=1_000,
    )

    # --------------------------------------------------------
    # Environnement
    # --------------------------------------------------------

    env = Connect4Environment()

    # --------------------------------------------------------
    # Statistiques globales
    # --------------------------------------------------------

    total_wins = 0
    total_losses = 0
    total_draws = 0

    # --------------------------------------------------------
    # Statistiques fenêtre
    # --------------------------------------------------------

    window_wins = 0
    window_losses = 0
    window_draws = 0

    window_steps = 0

    window_losses_nn = []

    # --------------------------------------------------------
    # Temps
    # --------------------------------------------------------

    training_start = (
        time.perf_counter()
    )

    window_start = training_start

    # ========================================================
    # Informations système
    # ========================================================

    print()
    print(
        f"Device utilisé : "
        f"{agent.device}"
    )

    if agent.device.type == "cuda":

        print(
            "GPU disponible : "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "GPU non disponible : "
            "entraînement sur CPU"
        )

    print()

    # ========================================================
    # TEST CRITIQUE DES GRADIENTS
    # ========================================================

    print(
        "Vérification du réseau DQN..."
    )

    agent.test_gradients()

    print(
        "Test des gradients : OK"
    )

    print()

    # ========================================================
    # Boucle principale
    # ========================================================

    for episode in range(
        1,
        number_of_episodes + 1,
    ):

        # ----------------------------------------------------
        # Début épisode
        # ----------------------------------------------------

        # On mesure les performances
        # indépendamment de la durée
        # d'affichage.

        episode_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Position de départ aléatoire
        # ----------------------------------------------------

        if random.random() < 0.5:

            agent_piece = (
                Cell.PLAYER_1
            )

        else:

            agent_piece = (
                Cell.PLAYER_2
            )

        # ----------------------------------------------------
        # Reset environnement
        # ----------------------------------------------------

        state = env.reset(
            agent_piece=agent_piece
        )

        done = False

        episode_steps = 0

        episode_losses = []

        # ----------------------------------------------------
        # Partie
        # ----------------------------------------------------

        while not done:

            # ------------------------------------------------
            # Actions légales
            # ------------------------------------------------

            legal_actions = (
                env.legal_actions()
            )

            if not legal_actions:

                raise RuntimeError(
                    "Environment returned no "
                    "legal action while the "
                    "game was not finished."
                )

            # ------------------------------------------------
            # Choix de l'action
            # ------------------------------------------------

            action = (
                agent.choose_action(
                    state,
                    legal_actions,
                    training=True,
                )
            )

            # ------------------------------------------------
            # Action dans l'environnement
            # ------------------------------------------------

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(
                action
            )

            # ------------------------------------------------
            # Actions légales état suivant
            # ------------------------------------------------

            if done:

                next_legal_actions = []

            else:

                next_legal_actions = (
                    env.legal_actions()
                )

            # ------------------------------------------------
            # Stockage expérience
            # ------------------------------------------------

            agent.remember(
                state,
                action,
                reward,
                next_state,
                next_legal_actions,
                done,
            )

            # ------------------------------------------------
            # Apprentissage
            # ------------------------------------------------

            loss = (
                agent.train_step()
            )

            if loss is not None:

                episode_losses.append(
                    loss
                )

                window_losses_nn.append(
                    loss
                )

            # ------------------------------------------------
            # Passage à l'état suivant
            # ------------------------------------------------

            state = next_state

            episode_steps += 1

            window_steps += 1

        # ----------------------------------------------------
        # Résultat de la partie
        # ----------------------------------------------------

        winner = info["winner"]

        if winner == agent_piece:

            total_wins += 1
            window_wins += 1

        elif winner is None:

            total_draws += 1
            window_draws += 1

        else:

            total_losses += 1
            window_losses += 1

        # ----------------------------------------------------
        # Epsilon
        # ----------------------------------------------------

        agent.decay_epsilon()

        agent.episodes += 1

        # ----------------------------------------------------
        # Temps épisode
        # ----------------------------------------------------

        episode_elapsed = (
            time.perf_counter()
            - episode_start
        )

        # Évite que Python considère la variable
        # comme inutilisée et documente le fait
        # qu'elle est disponible pour un futur affichage.

        _ = episode_elapsed

        # ====================================================
        # Affichage
        # ====================================================

        if (
            episode % PRINT_EVERY
            == 0
        ):

            now = time.perf_counter()

            elapsed_total = (
                now
                - training_start
            )

            elapsed_window = (
                now
                - window_start
            )

            # ------------------------------------------------
            # Vitesse
            # ------------------------------------------------

            if elapsed_window > 0:

                episodes_per_second = (
                    PRINT_EVERY
                    / elapsed_window
                )

            else:

                episodes_per_second = 0.0

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            if window_losses_nn:

                average_loss = (
                    sum(
                        window_losses_nn
                    )
                    /
                    len(
                        window_losses_nn
                    )
                )

                min_loss = min(
                    window_losses_nn
                )

                max_loss = max(
                    window_losses_nn
                )

            else:

                average_loss = 0.0
                min_loss = 0.0
                max_loss = 0.0

            # ------------------------------------------------
            # Temps restant
            # ------------------------------------------------

            episodes_remaining = (
                number_of_episodes
                - episode
            )

            if episodes_per_second > 0:

                estimated_remaining = (
                    episodes_remaining
                    / episodes_per_second
                )

            else:

                estimated_remaining = 0.0

            # ------------------------------------------------
            # Statistiques fenêtre
            # ------------------------------------------------

            window_total = (
                window_wins
                + window_losses
                + window_draws
            )

            if window_total > 0:

                win_rate = (
                    100.0
                    * window_wins
                    / window_total
                )

                loss_rate = (
                    100.0
                    * window_losses
                    / window_total
                )

                draw_rate = (
                    100.0
                    * window_draws
                    / window_total
                )

            else:

                win_rate = 0.0
                loss_rate = 0.0
                draw_rate = 0.0

            # ------------------------------------------------
            # Affichage
            # ------------------------------------------------

            print()
            print(
                "-" * 75
            )

            print(
                f"Episode "
                f"{episode:>7,} / "
                f"{number_of_episodes:,}"
            )

            print(
                f"Progression      : "
                f"{100 * episode / number_of_episodes:6.2f} %"
            )

            print(
                f"Epsilon           : "
                f"{agent.epsilon:.4f}"
            )

            print(
                f"Replay buffer     : "
                f"{len(agent.replay_buffer):,}"
            )

            print(
                f"Training steps    : "
                f"{agent.training_steps:,}"
            )

            print()

            print(
                f"Victoires fenêtre : "
                f"{window_wins:>5} "
                f"({win_rate:6.2f} %)"
            )

            print(
                f"Défaites fenêtre  : "
                f"{window_losses:>5} "
                f"({loss_rate:6.2f} %)"
            )

            print(
                f"Nuls fenêtre      : "
                f"{window_draws:>5} "
                f"({draw_rate:6.2f} %)"
            )

            print()

            print(
                f"Loss moyenne      : "
                f"{average_loss:.6f}"
            )

            print(
                f"Loss min/max      : "
                f"{min_loss:.6f} / "
                f"{max_loss:.6f}"
            )

            print()

            print(
                f"Vitesse           : "
                f"{episodes_per_second:.2f} "
                f"episodes/s"
            )

            print(
                f"Temps écoulé      : "
                f"{elapsed_total / 60:.1f} min"
            )

            print(
                f"Temps restant ~   : "
                f"{estimated_remaining / 60:.1f} min"
            )

            print(
                f"Étapes dernière   : "
                f"{episode_steps}"
            )

            print(
                f"Total victoires   : "
                f"{total_wins:,}"
            )

            print(
                f"Total défaites    : "
                f"{total_losses:,}"
            )

            print(
                f"Total nuls        : "
                f"{total_draws:,}"
            )

            print(
                "-" * 75
            )

            # ------------------------------------------------
            # Reset fenêtre
            # ------------------------------------------------

            window_wins = 0
            window_losses = 0
            window_draws = 0

            window_steps = 0

            window_losses_nn = []

            window_start = now

    return agent


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 75
    )

    print(
        "                         DQN V1"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Episodes : "
        f"{NUMBER_OF_EPISODES:,}"
    )

    print(
        f"Affichage tous les "
        f"{PRINT_EVERY:,} épisodes"
    )

    print()

    agent = train(
        NUMBER_OF_EPISODES
    )

    # ========================================================
    # Sauvegarde
    # ========================================================

    agent.save(
        MODEL_PATH
    )

    print()

    print(
        "=" * 75
    )

    print(
        "ENTRAÎNEMENT TERMINÉ"
    )

    print(
        "=" * 75
    )

    print()

    print(
        f"Modèle sauvegardé : "
        f"{MODEL_PATH}"
    )

    print(
        f"Device             : "
        f"{agent.device}"
    )

    print(
        f"Replay buffer      : "
        f"{len(agent.replay_buffer):,}"
    )

    print(
        f"Training steps     : "
        f"{agent.training_steps:,}"
    )

    print(
        f"Epsilon final      : "
        f"{agent.epsilon:.4f}"
    )


if __name__ == "__main__":
    main()