# -*- coding: utf-8 -*-

"""
Created on Mon Aug 10 08:44:33 2026

@author: kcollet

Définition des différents types de joueurs.
"""

from abc import ABC, abstractmethod
import random

from .board import Board, Cell


class Player(ABC):
    """
    Classe abstraite représentant un joueur.

    Un joueur possède une couleur/pion et doit être capable de
    choisir une action à partir de l'état actuel du plateau.

    Le joueur ne modifie jamais directement le plateau.
    """

    def __init__(self, piece: Cell):
        if piece not in (Cell.PLAYER_1, Cell.PLAYER_2):
            raise ValueError("A player must have a valid piece.")

        self.piece = piece

    @abstractmethod
    def choose_action(self, board: Board) -> int:
        """
        Retourne la colonne dans laquelle le joueur souhaite jouer.

        Parameters
        ----------
        board : Board
            État actuel du plateau.

        Returns
        -------
        int
            Index de la colonne choisie (0 à 6).
        """
        pass


class RandomPlayer(Player):
    """
    Joueur choisissant aléatoirement parmi les actions légales.

    Ce joueur constitue notre premier adversaire automatique et
    notre première baseline pour les futurs algorithmes de RL.
    """

    def choose_action(self, board: Board) -> int:
        legal_actions = board.legal_actions()

        if not legal_actions:
            raise RuntimeError("No legal action available.")

        return random.choice(legal_actions)


class HumanPlayer(Player):
    """
    Joueur contrôlé par un utilisateur humain.

    L'interaction avec l'utilisateur se fait dans le terminal.
    """

    def choose_action(self, board: Board) -> int:
        while True:
            try:
                value = input(
                    f"Joueur {self.symbol()}, "
                    f"choisissez une colonne (1-{board.COLUMNS}) : "
                )

                column = int(value) - 1

                if column not in board.legal_actions():
                    print("Cette colonne n'est pas disponible.")
                    continue

                return column

            except ValueError:
                print(
                    "Veuillez entrer un numéro de colonne valide."
                )

    def symbol(self) -> str:
        """
        Retourne le symbole utilisé pour représenter le joueur.
        """
        if self.piece == Cell.PLAYER_1:
            return "X"

        return "O"
