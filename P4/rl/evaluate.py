# -*- coding: utf-8 -*-

"""
Évaluation d'un agent Q-learning contre RandomPlayer.

Le modèle est testé dans les deux positions :

    1. RL commence
    2. Random commence

L'objectif est de mesurer le biais éventuel lié au
fait de jouer en premier.
"""

from pathlib import Path

from P4.board import Cell
from P4.rl.environment import Connect4Environment
from P4.rl.q_learning import QLearningAgent


# Chemin ancré à la racine du dépôt, quel que soit le dossier courant.
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "q_table.pkl"


def evaluate_position(
    agent: QLearningAgent,
    agent_piece: Cell,
    number_of_games: int,
) -> dict:
    """
    Évalue le modèle dans une position donnée.

    Parameters
    ----------
    agent : QLearningAgent
        Agent entraîné.

    agent_piece : Cell
        Position de l'agent :
        PLAYER_1 ou PLAYER_2.

    number_of_games : int
        Nombre de parties.

    Returns
    -------
    dict
        Statistiques.
    """

    env = Connect4Environment(
        agent_piece=agent_piece
    )

    results = {
        "games": number_of_games,
        "wins": 0,
        "losses": 0,
        "draws": 0,
    }

    for _ in range(number_of_games):

        state = env.reset()
        done = False

        while not done:

            legal_actions = env.legal_actions()

            action = agent.choose_action(
                state,
                legal_actions,
                training=False,
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            state = next_state

        winner = info["winner"]

        if winner == agent_piece:
            results["wins"] += 1

        elif winner is None:
            results["draws"] += 1

        else:
            results["losses"] += 1

    return results


def display_position_results(
    title: str,
    results: dict,
) -> None:
    """
    Affiche les résultats d'une position.
    """

    games = results["games"]
    wins = results["wins"]
    losses = results["losses"]
    draws = results["draws"]

    print()
    print("-" * 50)
    print(title)
    print("-" * 50)

    print(
        f"Parties       : {games:,}"
    )

    print(
        f"Victoires RL  : "
        f"{wins:>6} "
        f"({100 * wins / games:6.2f} %)"
    )

    print(
        f"Défaites RL   : "
        f"{losses:>6} "
        f"({100 * losses / games:6.2f} %)"
    )

    print(
        f"Matchs nuls   : "
        f"{draws:>6} "
        f"({100 * draws / games:6.2f} %)"
    )


def combine_results(
    results_1: dict,
    results_2: dict,
) -> dict:
    """
    Combine deux séries de résultats.
    """

    return {
        "games": (
            results_1["games"]
            + results_2["games"]
        ),
        "wins": (
            results_1["wins"]
            + results_2["wins"]
        ),
        "losses": (
            results_1["losses"]
            + results_2["losses"]
        ),
        "draws": (
            results_1["draws"]
            + results_2["draws"]
        ),
    }


def display_global_results(
    results: dict,
) -> None:
    """
    Affiche les résultats globaux.
    """

    games = results["games"]
    wins = results["wins"]
    losses = results["losses"]
    draws = results["draws"]

    print()
    print("=" * 50)
    print("             RÉSULTATS GLOBAUX")
    print("=" * 50)

    print(
        f"Parties       : {games:,}"
    )

    print(
        f"Victoires RL  : "
        f"{wins:>6} "
        f"({100 * wins / games:6.2f} %)"
    )

    print(
        f"Défaites RL   : "
        f"{losses:>6} "
        f"({100 * losses / games:6.2f} %)"
    )

    print(
        f"Matchs nuls   : "
        f"{draws:>6} "
        f"({100 * draws / games:6.2f} %)"
    )

    print("=" * 50)


def main() -> None:

    number_of_games = 1_000

    print("=" * 50)
    print("       ÉVALUATION Q-LEARNING")
    print("=" * 50)

    print()
    print(
        f"Modèle : {MODEL_PATH}"
    )

    print(
        f"Parties par position : "
        f"{number_of_games:,}"
    )

    # --------------------------------------------------
    # Chargement du modèle
    # --------------------------------------------------

    agent = QLearningAgent()

    agent.load(MODEL_PATH)

    print()
    print(
        f"États chargés : "
        f"{len(agent.q_table):,}"
    )

    # --------------------------------------------------
    # RL commence
    # --------------------------------------------------

    print()
    print("Évaluation 1/2 : RL commence...")

    results_first = evaluate_position(
        agent=agent,
        agent_piece=Cell.PLAYER_1,
        number_of_games=number_of_games,
    )

    display_position_results(
        "RL = PLAYER_1 | Random = PLAYER_2",
        results_first,
    )

    # --------------------------------------------------
    # RL joue second
    # --------------------------------------------------

    print()
    print("Évaluation 2/2 : Random commence...")

    results_second = evaluate_position(
        agent=agent,
        agent_piece=Cell.PLAYER_2,
        number_of_games=number_of_games,
    )

    display_position_results(
        "Random = PLAYER_1 | RL = PLAYER_2",
        results_second,
    )

    # --------------------------------------------------
    # Résultats globaux
    # --------------------------------------------------

    global_results = combine_results(
        results_first,
        results_second,
    )

    display_global_results(
        global_results
    )


if __name__ == "__main__":
    main()