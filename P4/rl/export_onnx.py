# -*- coding: utf-8 -*-

"""
Exporte un modèle DQN V2 au format ONNX pour la démo du site.

Contrat :

    entrée  "board"     float32 [N, 2, 6, 7]
            plan 0 = pions du joueur qui doit jouer, plan 1 = pions
            adverses, ligne 0 en haut ;
    sortie  "q_values"  float32 [N, 7], une valeur par colonne.

Après l'export, le fichier est relu avec ONNX Runtime et comparé au
réseau PyTorch sur des positions aléatoires.

Exemple :

    python -m P4.rl.export_onnx models/dqn_v2/best.pt models/puissance4.onnx
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from P4 import bitboard
from P4.rl.dqn_v2 import DQNv2Agent


def random_positions(count: int, seed: int = 0) -> np.ndarray:
    """Positions atteintes par des parties aléatoires, encodées."""
    rng = random.Random(seed)
    positions = []

    while len(positions) < count:
        current, mask = 0, 0

        for _ in range(rng.randint(0, 30)):
            legal = bitboard.legal_actions(mask)

            if not legal:
                break

            current, mask = bitboard.play(current, mask, rng.choice(legal))

            if bitboard.has_four(current ^ mask):
                break

        else:
            positions.append(bitboard.encode_planes(current, mask))

    return np.stack(positions).astype(np.float32)


def export(checkpoint: Path, output: Path) -> float:
    """
    Exporte le réseau et retourne l'écart maximal entre PyTorch et
    ONNX Runtime sur 200 positions.
    """
    import onnxruntime

    agent = DQNv2Agent.load(checkpoint, device="cpu", replay_capacity=1)
    network = agent.online_network.eval()

    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        network,
        torch.zeros(1, 2, bitboard.ROWS, bitboard.COLUMNS),
        str(output),
        input_names=["board"],
        output_names=["q_values"],
        dynamic_axes={"board": {0: "batch"}, "q_values": {0: "batch"}},
        opset_version=17,
    )

    positions = random_positions(200)

    with torch.no_grad():
        expected = network(torch.from_numpy(positions)).numpy()

    session = onnxruntime.InferenceSession(
        str(output),
        providers=["CPUExecutionProvider"],
    )

    actual = session.run(["q_values"], {"board": positions})[0]

    return float(np.abs(actual - expected).max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporte un modèle DQN V2 en ONNX."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    difference = export(args.checkpoint, args.output)

    print(f"Modèle exporté : {args.output}")
    print(f"Taille         : {args.output.stat().st_size / 1024:.0f} Ko")
    print(f"Écart maximal PyTorch / ONNX Runtime : {difference:.2e}")

    if difference > 1e-4:
        raise SystemExit("Écart trop grand : export à vérifier.")


if __name__ == "__main__":
    main()
