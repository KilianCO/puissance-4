# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:53:37 2026

@author: kcollet
"""

from pathlib import Path

from P4.board import Board, Cell
from P4.game import Game, GameStatus
from P4.players import (
    HumanPlayer,
    MinimaxPlayer,
    RandomPlayer,
    TacticalPlayer,
)


MODELS_DIRECTORY = Path(__file__).resolve().parents[2] / "models"


def display_board(board: Board) -> None:
    """
    Affiche le plateau dans le terminal.
    """

    print()

    # Numéros des colonnes
    print("   " + "   ".join(
        str(column + 1)
        for column in range(board.COLUMNS)
    ))

    print("  " + "+---" * board.COLUMNS + "+")

    for row in range(board.ROWS):
        cells = []

        for column in range(board.COLUMNS):
            cell = board.get(row, column)

            if cell == Cell.PLAYER_1:
                symbol = "X"
            elif cell == Cell.PLAYER_2:
                symbol = "O"
            else:
                symbol = " "

            cells.append(symbol)

        print("  | " + " | ".join(cells) + " |")
        print("  " + "+---" * board.COLUMNS + "+")

    print()


def display_result(status: GameStatus) -> None:
    if status == GameStatus.PLAYER_1_WON:
        print("Le joueur X gagne !")

    elif status == GameStatus.PLAYER_2_WON:
        print("Le joueur O gagne !")

    elif status == GameStatus.DRAW:
        print("Match nul !")


def choose_model():
    """
    Demande quel modèle entraîné affronter et retourne la fabrique de
    joueurs correspondante, ou None s'il n'y en a aucun.
    """
    from P4.rl.agent_player import load_agent_player

    models = sorted(
        path
        for pattern in ("*.pt", "*.pkl")
        for path in MODELS_DIRECTORY.rglob(pattern)
    )

    if not models:
        print(f"Aucun modèle trouvé dans {MODELS_DIRECTORY}.")
        return None

    print()
    print("Modèles disponibles :")

    for index, path in enumerate(models, start=1):
        print(f"{index}. {path.relative_to(MODELS_DIRECTORY)}")

    while True:
        choice = input("Votre choix : ")

        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return load_agent_player(models[int(choice) - 1])

        print("Choix invalide.")


def create_game() -> Game:
    print()
    print("Choisissez le mode de jeu :")
    print("1. Humain vs Humain")
    print("2. Humain vs Random")
    print("3. Humain vs Tactique")
    print("4. Humain vs Minimax moyen (4 coups d'avance)")
    print("5. Humain vs Minimax difficile (6 coups d'avance)")
    print("6. Humain vs modèle entraîné")

    while True:
        choice = input("Votre choix : ")

        if choice == "3":
            return Game(
                HumanPlayer(Cell.PLAYER_1),
                TacticalPlayer(Cell.PLAYER_2),
            )

        if choice in ("4", "5"):
            return Game(
                HumanPlayer(Cell.PLAYER_1),
                MinimaxPlayer(
                    Cell.PLAYER_2,
                    depth=4 if choice == "4" else 6,
                ),
            )

        if choice == "6":
            model = choose_model()

            if model is not None:
                return Game(
                    HumanPlayer(Cell.PLAYER_1),
                    model(Cell.PLAYER_2),
                )

            continue

        if choice == "1":
            return Game(
                HumanPlayer(Cell.PLAYER_1),
                HumanPlayer(Cell.PLAYER_2),
            )

        if choice == "2":
            return Game(
                HumanPlayer(Cell.PLAYER_1),
                RandomPlayer(Cell.PLAYER_2),
            )

        print("Choix invalide.")

def main() -> None:
    print("================================")
    print("          PUISSANCE 4           ")
    print("================================")

    game = create_game()

    while game.status == GameStatus.IN_PROGRESS:
        display_board(game.board)

        current_player = game.players[game.current_player]

        print(
            f"Tour du joueur "
            f"{'X' if current_player.piece == Cell.PLAYER_1 else 'O'}"
        )

        game.play_turn()

    display_board(game.board)
    display_result(game.status)


if __name__ == "__main__":
    main()