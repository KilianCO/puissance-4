# -*- coding: utf-8 -*-

"""
Simulation de parties automatiques de Puissance 4.

Ce module permet de faire jouer différents types de joueurs
les uns contre les autres et de collecter des statistiques.

Cette infrastructure sera utilisée plus tard pour évaluer
les agents d'apprentissage par renforcement.

Exemple :
    python -m P4.simulate
"""

from dataclasses import dataclass
from typing import Type

from P4.board import Cell
from P4.game import Game, GameStatus
from P4.players import Player, RandomPlayer


@dataclass
class SimulationResults:
    """
    Résultats d'une série de simulations.
    """

    number_of_games: int = 0
    player_1_wins: int = 0
    player_2_wins: int = 0
    draws: int = 0
    total_moves: int = 0

    @property
    def player_1_win_rate(self) -> float:
        """Taux de victoire du joueur 1."""
        if self.number_of_games == 0:
            return 0.0

        return self.player_1_wins / self.number_of_games

    @property
    def player_2_win_rate(self) -> float:
        """Taux de victoire du joueur 2."""
        if self.number_of_games == 0:
            return 0.0

        return self.player_2_wins / self.number_of_games

    @property
    def draw_rate(self) -> float:
        """Taux de matchs nuls."""
        if self.number_of_games == 0:
            return 0.0

        return self.draws / self.number_of_games

    @property
    def average_moves(self) -> float:
        """Nombre moyen de coups par partie."""
        if self.number_of_games == 0:
            return 0.0

        return self.total_moves / self.number_of_games


def create_player(
    player_class: Type[Player],
    piece: Cell,
) -> Player:
    """
    Crée un joueur.

    Cette fonction paraît simple actuellement, mais elle nous
    permettra plus tard d'ajouter des joueurs nécessitant des
    paramètres spécifiques.
    """

    return player_class(piece)


def play_game(
    player_1_class: Type[Player],
    player_2_class: Type[Player],
) -> tuple[GameStatus, int]:
    """
    Joue une partie complète entre deux joueurs.

    Parameters
    ----------
    player_1_class : Type[Player]
        Classe du joueur 1.

    player_2_class : Type[Player]
        Classe du joueur 2.

    Returns
    -------
    tuple[GameStatus, int]
        Résultat de la partie et nombre de coups joués.
    """

    player_1 = create_player(
        player_1_class,
        Cell.PLAYER_1,
    )

    player_2 = create_player(
        player_2_class,
        Cell.PLAYER_2,
    )

    game = Game(player_1, player_2)

    number_of_moves = 0

    while game.status == GameStatus.IN_PROGRESS:
        game.play_turn()
        number_of_moves += 1

    return game.status, number_of_moves


def simulate(
    number_of_games: int,
    player_1_class: Type[Player],
    player_2_class: Type[Player],
) -> SimulationResults:
    """
    Simule plusieurs parties entre deux types de joueurs.

    Parameters
    ----------
    number_of_games : int
        Nombre de parties à jouer.

    player_1_class : Type[Player]
        Classe du joueur 1.

    player_2_class : Type[Player]
        Classe du joueur 2.

    Returns
    -------
    SimulationResults
        Statistiques de la simulation.
    """

    if number_of_games <= 0:
        raise ValueError(
            "number_of_games must be positive."
        )

    results = SimulationResults(
        number_of_games=number_of_games
    )

    for _ in range(number_of_games):

        status, number_of_moves = play_game(
            player_1_class,
            player_2_class,
        )

        results.total_moves += number_of_moves

        if status == GameStatus.PLAYER_1_WON:
            results.player_1_wins += 1

        elif status == GameStatus.PLAYER_2_WON:
            results.player_2_wins += 1

        elif status == GameStatus.DRAW:
            results.draws += 1

        else:
            raise RuntimeError(
                f"Unexpected game status: {status}"
            )

    return results


def display_results(
    results: SimulationResults,
    player_1_class: Type[Player],
    player_2_class: Type[Player],
) -> None:
    """
    Affiche les résultats d'une simulation.
    """

    print()
    print("=" * 50)
    print("             RÉSULTATS")
    print("=" * 50)

    print(
        f"Joueur 1 : {player_1_class.__name__}"
    )

    print(
        f"Joueur 2 : {player_2_class.__name__}"
    )

    print()

    print(
        f"Nombre de parties : "
        f"{results.number_of_games:,}"
    )

    print()

    print(
        f"Victoires joueur 1 : "
        f"{results.player_1_wins:>7} "
        f"({results.player_1_win_rate * 100:6.2f} %)"
    )

    print(
        f"Victoires joueur 2 : "
        f"{results.player_2_wins:>7} "
        f"({results.player_2_win_rate * 100:6.2f} %)"
    )

    print(
        f"Matchs nuls         : "
        f"{results.draws:>7} "
        f"({results.draw_rate * 100:6.2f} %)"
    )

    print()

    print(
        f"Coups total         : "
        f"{results.total_moves:>7}"
    )

    print(
        f"Coups moyen / partie: "
        f"{results.average_moves:7.2f}"
    )

    print("=" * 50)


def main() -> None:
    """
    Lance une simulation RandomPlayer vs RandomPlayer.
    """

    number_of_games = 5000

    player_1_class = RandomPlayer
    player_2_class = RandomPlayer

    print("=" * 50)
    print("          SIMULATION PUISSANCE 4")
    print("=" * 50)

    print()

    print(
        f"Simulation de {number_of_games:,} parties :"
    )

    print(
        f"{player_1_class.__name__} "
        f"vs "
        f"{player_2_class.__name__}"
    )

    results = simulate(
        number_of_games,
        player_1_class,
        player_2_class,
    )

    display_results(
        results,
        player_1_class,
        player_2_class,
    )


if __name__ == "__main__":
    main()
