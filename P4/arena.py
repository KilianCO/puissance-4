# -*- coding: utf-8 -*-

"""
Banc d'évaluation : fait s'affronter n'importe quels joueurs.

Un joueur est décrit par une fabrique `piece -> Player`, ce qui permet de
le créer indifféremment en premier ou en second joueur. Chaque match est
joué pour moitié dans chaque position.

Exemples :

    python -m P4.arena                          # références entre elles
    python -m P4.arena --model models/dqn_v2/best.pt
    python -m P4.arena --model models/dqn_v1.pt --games 200
    python -m P4.arena --model models/dqn_v2/best.pt --opening 4

Avec --opening N, les N premiers coups de chaque partie sont joués au
hasard : deux joueurs déterministes ne rejouent alors pas toujours la même
partie, et l'on mesure le jeu sur des positions variées.
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from P4.board import Cell
from P4.game import Game, GameStatus
from P4.players import (
    MinimaxPlayer,
    Player,
    RandomPlayer,
    TacticalPlayer,
)


PlayerFactory = Callable[[Cell], Player]


@dataclass
class MatchResult:
    """Résultat d'un match, du point de vue du premier joueur nommé."""

    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    wins_as_first: int = 0
    games_as_first: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def score(self) -> float:
        """Victoire = 1, nul = 0,5 : entre 0 et 1."""
        if not self.games:
            return 0.0

        return (self.wins + 0.5 * self.draws) / self.games


# Adversaires de référence, du plus faible au plus fort. Les minimax
# 1, 4 et 6 sont les niveaux facile, moyen et difficile du site.
REFERENCES: dict[str, PlayerFactory] = {
    "aléatoire": RandomPlayer,
    "tactique": TacticalPlayer,
    "minimax 1 (facile)": lambda piece: MinimaxPlayer(piece, 1, 0.35),
    "minimax 2": lambda piece: MinimaxPlayer(piece, 2),
    "minimax 4 (moyen)": lambda piece: MinimaxPlayer(piece, 4),
    "minimax 6 (difficile)": lambda piece: MinimaxPlayer(piece, 6),
}


def play_match(
    player: PlayerFactory,
    opponent: PlayerFactory,
    games: int,
    opening_moves: int = 0,
) -> MatchResult:
    """
    Joue `games` parties, en alternant le joueur qui commence.

    Les `opening_moves` premiers coups sont joués au hasard (au plus 6 :
    au-delà, la partie pourrait être finie avant de commencer).
    """
    if not 0 <= opening_moves <= 6:
        raise ValueError("opening_moves must be between 0 and 6.")
    result = MatchResult()

    for index in range(games):
        player_starts = index % 2 == 0

        if player_starts:
            game = Game(
                player(Cell.PLAYER_1),
                opponent(Cell.PLAYER_2),
            )
            player_won = GameStatus.PLAYER_1_WON
        else:
            game = Game(
                opponent(Cell.PLAYER_1),
                player(Cell.PLAYER_2),
            )
            player_won = GameStatus.PLAYER_2_WON

        for _ in range(opening_moves):
            game.board.play(
                random.choice(game.board.legal_actions()),
                game.current_player,
            )
            game._switch_player()

        status = game.play()

        result.games += 1
        result.games_as_first += player_starts

        if status == player_won:
            result.wins += 1
            result.wins_as_first += player_starts
        elif status == GameStatus.DRAW:
            result.draws += 1
        else:
            result.losses += 1

    return result


def benchmark(
    player: PlayerFactory,
    games: int = 100,
    opponents: dict[str, PlayerFactory] | None = None,
    opening_moves: int = 0,
) -> dict[str, MatchResult]:
    """Fait jouer `player` contre chaque adversaire de référence."""
    if opponents is None:
        opponents = REFERENCES

    return {
        name: play_match(player, opponent, games, opening_moves)
        for name, opponent in opponents.items()
    }


def display_benchmark(
    title: str,
    results: dict[str, MatchResult],
) -> None:
    print()
    print(title)
    print("-" * 66)
    print(
        f"{'adversaire':<24}{'parties':>8}"
        f"{'victoires':>11}{'défaites':>10}{'nuls':>7}"
    )

    for name, result in results.items():
        print(
            f"{name:<24}{result.games:>8}"
            f"{100 * result.win_rate:>10.1f}%"
            f"{100 * result.loss_rate:>9.1f}%"
            f"{100 * result.draw_rate:>6.1f}%"
        )


def load_model(path: Path) -> PlayerFactory:
    """
    Charge un modèle entraîné et retourne la fabrique de joueurs
    correspondante. Le format est reconnu d'après le fichier.
    """
    from P4.rl.agent_player import load_agent_player

    return load_agent_player(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évalue un joueur contre les adversaires de référence."
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="modèle entraîné à évaluer (.pt ou .pkl) ; "
             "sans modèle, évalue le joueur tactique",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opening",
        type=int,
        default=0,
        help="nombre de premiers coups joués au hasard (0 à 6)",
    )
    parser.add_argument(
        "--skip-deep",
        action="store_true",
        help="ne joue pas contre le minimax 6, le plus lent",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    opponents = dict(REFERENCES)

    if args.skip_deep:
        del opponents["minimax 6 (difficile)"]

    if args.model is None:
        title = "joueur tactique"
        player = TacticalPlayer
    else:
        title = str(args.model)
        player = load_model(args.model)

    display_benchmark(
        f"{title} — {args.games} parties par adversaire, "
        f"moitié en premier, moitié en second"
        + (f", {args.opening} premiers coups au hasard" if args.opening else ""),
        benchmark(player, args.games, opponents, args.opening),
    )


if __name__ == "__main__":
    main()
