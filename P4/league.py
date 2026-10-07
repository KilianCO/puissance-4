# -*- coding: utf-8 -*-

"""
Ligue : tous les joueurs s'affrontent, puis reçoivent un classement Elo.

Chaque paire joue le même nombre de parties, la moitié dans chaque
position, avec des premiers coups aléatoires pour que deux joueurs
déterministes ne rejouent pas toujours la même partie.

Le classement n'est pas mis à jour partie après partie (le résultat
dépendrait de l'ordre des parties) : on cherche les notes qui expliquent
le mieux l'ensemble des résultats.

Exemples :

    python -m P4.league
    python -m P4.league --games 40 --output docs/ligue.json
"""

import argparse
import itertools
import json
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

from P4.arena import play_match
from P4.players import (
    MinimaxPlayer,
    MonteCarloPlayer,
    RandomPlayer,
    TacticalPlayer,
)


ROOT = Path(__file__).resolve().parents[1]

# Nom affiché -> description du joueur. Une description est un texte
# simple, pour pouvoir être envoyée à un autre processus.
ROSTER = {
    "Aléatoire": "random",
    "Tactique": "tactical",
    "Minimax facile": "minimax:1:0.35",
    "Minimax 2": "minimax:2:0",
    "Minimax moyen": "minimax:4:0",
    "Minimax difficile": "minimax:6:0",
    "Monte-Carlo 1 000": "mcts:1000",
    "Monte-Carlo 10 000": "mcts:10000",
    "DQN V1": "model:models/dqn_v1.pt",
    "DQN V2": "model:models/dqn_v2/best.pt",
}

KIND = {
    "random": "référence",
    "tactical": "référence",
    "minimax": "recherche",
    "mcts": "recherche",
    "model": "appris",
}


def make_player(spec: str):
    """Construit la fabrique `piece -> Player` décrite par `spec`."""
    kind, _, arguments = spec.partition(":")

    if kind == "random":
        return RandomPlayer

    if kind == "tactical":
        return TacticalPlayer

    if kind == "minimax":
        depth, randomness = arguments.split(":")

        return lambda piece: MinimaxPlayer(
            piece, int(depth), float(randomness),
        )

    if kind == "mcts":
        return lambda piece: MonteCarloPlayer(piece, int(arguments))

    if kind == "model":
        from P4.rl.agent_player import load_agent_player

        return load_agent_player(ROOT / arguments)

    raise ValueError(f"Unknown player: {spec}")


def _play_pairing(job: tuple) -> tuple[str, str, float, float, float]:
    """Un match entre deux joueurs, exécuté dans un processus à part."""
    name_a, spec_a, name_b, spec_b, games, opening, seed = job

    random.seed(seed)

    if spec_a.startswith("model") or spec_b.startswith("model"):
        import torch

        torch.set_num_threads(1)

    result = play_match(
        make_player(spec_a),
        make_player(spec_b),
        games,
        opening,
    )

    return name_a, name_b, result.wins, result.losses, result.draws


def play_league(
    roster: dict[str, str],
    games: int,
    opening_moves: int = 4,
    seed: int = 0,
    workers: int | None = None,
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """
    Joue toutes les paires. Retourne, pour chaque paire (a, b), les
    victoires, défaites et nuls de a.
    """
    jobs = [
        (name_a, roster[name_a], name_b, roster[name_b],
         games, opening_moves, seed + index)
        for index, (name_a, name_b)
        in enumerate(itertools.combinations(roster, 2))
    ]

    results = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name_a, name_b, wins, losses, draws in pool.map(
            _play_pairing, jobs,
        ):
            results[(name_a, name_b)] = (wins, losses, draws)
            print(
                f"{name_a} - {name_b} : "
                f"{wins:.0f} / {losses:.0f} / {draws:.0f}",
                flush=True,
            )

    return results


def fit_elo(
    names: list[str],
    results: dict[tuple[str, str], tuple[float, float, float]],
    mean: float = 1500.0,
    prior_games: float = 1.0,
    iterations: int = 20_000,
) -> dict[str, float]:
    """
    Notes Elo qui expliquent le mieux les résultats.

    Avec un écart de note d, le score attendu est 1 / (1 + 10^(-d/400)).
    On ajuste les notes jusqu'à ce que le score attendu de chaque joueur
    égale son score réel (victoire = 1, nul = 0,5).

    `prior_games` ajoute à chaque paire une partie nulle fictive : sans
    elle, un joueur qui gagne ou perd toutes ses parties aurait une note
    infinie.
    """
    points = {name: {} for name in names}
    played = {name: {} for name in names}

    for (name_a, name_b), (wins, losses, draws) in results.items():
        total = wins + losses + draws + prior_games

        points[name_a][name_b] = wins + 0.5 * draws + 0.5 * prior_games
        points[name_b][name_a] = losses + 0.5 * draws + 0.5 * prior_games
        played[name_a][name_b] = total
        played[name_b][name_a] = total

    ratings = {name: 0.0 for name in names}

    for _ in range(iterations):
        largest_step = 0.0

        for name in names:
            actual = sum(points[name].values())
            expected = sum(
                games / (1 + 10 ** ((ratings[other] - ratings[name]) / 400))
                for other, games in played[name].items()
            )
            total = sum(played[name].values())

            # Pas proportionnel à l'écart entre score réel et attendu.
            step = 400 * (actual - expected) / total

            ratings[name] += step
            largest_step = max(largest_step, abs(step))

        if largest_step < 1e-4:
            break

    shift = mean - sum(ratings.values()) / len(ratings)

    return {name: rating + shift for name, rating in ratings.items()}


def standings(
    roster: dict[str, str],
    results: dict[tuple[str, str], tuple[float, float, float]],
) -> list[dict]:
    """Classement trié par note Elo décroissante."""
    ratings = fit_elo(list(roster), results)
    table = []

    for name, spec in roster.items():
        wins = losses = draws = 0.0

        for (name_a, name_b), (a_wins, a_losses, a_draws) in results.items():
            if name == name_a:
                wins, losses, draws = (
                    wins + a_wins, losses + a_losses, draws + a_draws,
                )
            elif name == name_b:
                wins, losses, draws = (
                    wins + a_losses, losses + a_wins, draws + a_draws,
                )

        total = wins + losses + draws

        table.append({
            "name": name,
            "kind": KIND[spec.partition(":")[0]],
            "elo": round(ratings[name]),
            "games": int(total),
            "score": round((wins + 0.5 * draws) / total, 3) if total else 0.0,
            "wins": int(wins),
            "losses": int(losses),
            "draws": int(draws),
        })

    return sorted(table, key=lambda row: -row["elo"])


def display_standings(table: list[dict]) -> None:
    print()
    print(f"{'':>3} {'joueur':<20}{'type':<11}{'Elo':>6}{'score':>8}")
    print("-" * 48)

    for rank, row in enumerate(table, start=1):
        print(
            f"{rank:>3} {row['name']:<20}{row['kind']:<11}"
            f"{row['elo']:>6}{100 * row['score']:>7.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fait s'affronter tous les joueurs et calcule leur Elo."
    )
    parser.add_argument("--games", type=int, default=40,
                        help="parties par paire de joueurs")
    parser.add_argument("--opening", type=int, default=4,
                        help="premiers coups joués au hasard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        help="fichier JSON où écrire le classement")
    args = parser.parse_args()

    roster = {
        name: spec
        for name, spec in ROSTER.items()
        if not spec.startswith("model:")
        or (ROOT / spec.partition(":")[2]).exists()
    }

    missing = sorted(set(ROSTER) - set(roster))

    if missing:
        print(f"Modèles absents, joueurs ignorés : {', '.join(missing)}")

    results = play_league(
        roster, args.games, args.opening, args.seed, args.workers,
    )

    table = standings(roster, results)
    display_standings(table)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        args.output.write_text(
            json.dumps(
                {
                    "date": date.today().isoformat(),
                    "games_per_pairing": args.games,
                    "opening_moves": args.opening,
                    "seed": args.seed,
                    "standings": table,
                    "pairings": [
                        {
                            "player": name_a,
                            "opponent": name_b,
                            "wins": int(wins),
                            "losses": int(losses),
                            "draws": int(draws),
                        }
                        for (name_a, name_b), (wins, losses, draws)
                        in results.items()
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"\nClassement écrit dans {args.output}")


if __name__ == "__main__":
    main()
