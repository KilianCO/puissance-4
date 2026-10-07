# -*- coding: utf-8 -*-

"""
Entraînement du DQN V2.

L'agent joue contre lui-même et contre des adversaires de référence
(aléatoire, tactique, minimax). Chaque coup de chaque joueur devient une
expérience : l'agent apprend aussi des coups de ses adversaires.

Toutes les `--eval-every` parties, l'agent est évalué sans exploration ;
le dernier état et le meilleur modèle sont sauvegardés, et une ligne est
ajoutée au journal CSV.

Exemples :

    python -m P4.rl.train_dqn_v2 --episodes 40000
    python -m P4.rl.train_dqn_v2 --episodes 80000 --resume
    python -m P4.rl.train_dqn_v2 --episodes 200 --eval-every 100 --name essai
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from P4 import bitboard
from P4.arena import REFERENCES, play_match
from P4.minimax import minimax_move, tactical_move
from P4.rl.agent_player import AgentPlayer, planes_state
from P4.rl.dqn_v2 import DQNv2Agent


MODELS_DIRECTORY = Path(__file__).resolve().parents[2] / "models"

# Qui joue contre l'agent pendant l'entraînement, et à quelle fréquence.
OPPONENT_MIX = {
    "self": 0.50,
    "minimax": 0.25,
    "tactical": 0.15,
    "random": 0.10,
}

# Adversaires de l'évaluation périodique et nombre de parties.
EVALUATION = {
    "tactique": 60,
    "minimax 2": 40,
    "minimax 4 (moyen)": 40,
}


def epsilon_at(
    episode: int,
    total_episodes: int,
    start: float = 1.0,
    end: float = 0.05,
    decay_fraction: float = 0.5,
) -> float:
    """
    Exploration : décroît linéairement de `start` à `end` pendant la
    première partie de l'entraînement, puis reste à `end`.
    """
    progress = episode / max(1, decay_fraction * total_episodes)

    return max(end, start - (start - end) * progress)


def opponent_move(kind: str, current: int, mask: int, depth: int) -> int:
    if kind == "minimax":
        return minimax_move(current, mask, depth, randomness=0.1)

    if kind == "tactical":
        return tactical_move(current, mask)

    return random.choice(bitboard.legal_actions(mask))


def play_training_game(
    agent: DQNv2Agent,
    train_every: int,
    warmup: int,
) -> tuple[str, int | None, list[float]]:
    """
    Joue une partie et stocke chaque coup dans le replay buffer.

    Retourne le type d'adversaire, le résultat de l'agent (1 victoire,
    0 nul, -1 défaite ; None en self-play) et les loss de la partie.
    """
    kind = random.choices(
        list(OPPONENT_MIX),
        weights=list(OPPONENT_MIX.values()),
    )[0]

    # Côtés joués par l'agent : les deux en self-play, un seul sinon.
    agent_sides = (0, 1) if kind == "self" else (random.randrange(2),)
    depth = random.randint(1, 4)

    current, mask = 0, 0
    side = 0
    outcome = None
    losses = []

    state = bitboard.encode_planes(current, mask)

    while True:
        legal = bitboard.legal_actions(mask)

        if side in agent_sides:
            action = agent.choose_action(state, legal, training=True)
        else:
            action = opponent_move(kind, current, mask, depth)

        current, mask = bitboard.play(current, mask, action)

        won = bitboard.has_four(current ^ mask)
        done = won or bitboard.is_full(mask)

        next_state = bitboard.encode_planes(current, mask)

        agent.remember(
            state,
            action,
            1.0 if won else 0.0,
            next_state,
            [] if done else bitboard.legal_actions(mask),
            done,
        )

        if (
            agent.replay_buffer.pushes >= warmup
            and agent.replay_buffer.pushes % train_every == 0
        ):
            loss = agent.train_step()

            if loss is not None:
                losses.append(loss)

        if done:
            if kind != "self":
                if not won:
                    outcome = 0
                else:
                    outcome = 1 if side in agent_sides else -1

            return kind, outcome, losses

        state = next_state
        side = 1 - side


def evaluate(agent: DQNv2Agent) -> dict[str, float]:
    """
    Score de l'agent sans exploration contre chaque adversaire de
    l'évaluation (victoire = 1, nul = 0,5).
    """
    agent.online_network.eval()

    def player(piece):
        return AgentPlayer(piece, agent, planes_state)

    scores = {
        name: play_match(player, REFERENCES[name], games).score
        for name, games in EVALUATION.items()
    }

    agent.online_network.train()

    return scores


def train(args: argparse.Namespace) -> None:
    run_directory = MODELS_DIRECTORY / args.name
    last_path = run_directory / "last.pt"
    best_path = run_directory / "best.pt"
    log_path = run_directory / "log.csv"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    best_score = -1.0

    if args.resume:
        agent = DQNv2Agent.load(last_path, replay_capacity=args.replay_capacity)
        best_score = torch.load(
            last_path, map_location="cpu", weights_only=True,
        ).get("best_score", -1.0)

        print(f"Reprise à la partie {agent.episodes:,} depuis {last_path}")

    else:
        if last_path.exists():
            raise SystemExit(
                f"{last_path} existe déjà : utiliser --resume pour "
                f"continuer, ou --name pour un nouvel entraînement."
            )

        agent = DQNv2Agent(replay_capacity=args.replay_capacity)

    run_directory.mkdir(parents=True, exist_ok=True)

    columns = [
        "episode", "epsilon", "loss", "training_steps",
        *EVALUATION, "score", "minutes",
    ]

    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(columns)

    print(f"Device : {agent.device}")
    print(f"Parties : {agent.episodes:,} -> {args.episodes:,}")
    print(f"Sorties : {run_directory}")
    print()

    start = time.perf_counter()
    window_losses = []
    window_results = {kind: [0, 0] for kind in OPPONENT_MIX if kind != "self"}

    def checkpoint() -> None:
        nonlocal best_score, window_losses

        scores = evaluate(agent)
        score = sum(scores.values()) / len(scores)
        minutes = (time.perf_counter() - start) / 60

        if score > best_score:
            best_score = score
            agent.save(best_path, best_score=best_score, scores=scores)

        agent.save(last_path, best_score=best_score, scores=scores)

        loss = (
            sum(window_losses) / len(window_losses)
            if window_losses else float("nan")
        )

        with log_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                agent.episodes,
                f"{agent.epsilon:.4f}",
                f"{loss:.6f}",
                agent.training_steps,
                *(f"{scores[name]:.3f}" for name in EVALUATION),
                f"{score:.3f}",
                f"{minutes:.1f}",
            ])

        training = "  ".join(
            f"{kind} {100 * wins / games:.0f}%"
            for kind, (wins, games) in window_results.items()
            if games
        )

        print(
            f"partie {agent.episodes:>7,} | eps {agent.epsilon:.2f} | "
            f"loss {loss:.4f} | "
            + " | ".join(
                f"{name} {100 * scores[name]:.0f}%" for name in EVALUATION
            )
            + f" | score {score:.3f} (meilleur {best_score:.3f}) | "
            f"{minutes:.1f} min"
        )
        print(f"    victoires à l'entraînement : {training}", flush=True)

        window_losses = []

        for result in window_results.values():
            result[0] = result[1] = 0

    try:
        while agent.episodes < args.episodes:
            agent.epsilon = epsilon_at(agent.episodes, args.episodes)

            kind, outcome, losses = play_training_game(
                agent,
                args.train_every,
                args.warmup,
            )

            agent.episodes += 1
            window_losses.extend(losses)

            if outcome is not None:
                window_results[kind][0] += outcome == 1
                window_results[kind][1] += 1

            if agent.episodes % args.eval_every == 0:
                checkpoint()

    except KeyboardInterrupt:
        print("Interrompu : sauvegarde du dernier état.")
        agent.save(last_path, best_score=best_score)
        return

    if agent.episodes % args.eval_every != 0:
        checkpoint()

    print()
    print(f"Terminé. Meilleur score : {best_score:.3f} -> {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne le DQN V2.")

    parser.add_argument("--episodes", type=int, default=40_000,
                        help="nombre total de parties à atteindre")
    parser.add_argument("--name", default="dqn_v2",
                        help="dossier de sortie dans models/")
    parser.add_argument("--resume", action="store_true",
                        help="reprend depuis models/<name>/last.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--train-every", type=int, default=2,
                        help="une étape d'apprentissage tous les N coups")
    parser.add_argument("--warmup", type=int, default=5_000,
                        help="coups à accumuler avant d'apprendre")
    parser.add_argument("--replay-capacity", type=int, default=300_000)

    train(parser.parse_args())


if __name__ == "__main__":
    main()
