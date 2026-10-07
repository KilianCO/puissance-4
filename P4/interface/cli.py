# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 08:53:37 2026

@author: kcollet
"""

from P4.board import Board, Cell
from P4.game import Game, GameStatus
from P4.players import HumanPlayer, RandomPlayer


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


def create_game() -> Game:
    print()
    print("Choisissez le mode de jeu :")
    print("1. Humain vs Humain")
    print("2. Humain vs Random")

    while True:
        choice = input("Votre choix : ")

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