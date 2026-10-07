# -*- coding: utf-8 -*-

"""
Created on Mon Aug 10 08:44:33 2026

@author: kcollet

Définition des différents types de joueurs.
"""

from abc import ABC, abstractmethod
import random

from .bitboard import from_board
from .board import Board, Cell
from .mcts import mcts_move
from .minimax import minimax_move, tactical_move


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


class TacticalPlayer(Player):
    """
    Joueur qui gagne s'il le peut, bloque une menace immédiate, et joue
    au hasard sinon.

    C'est le premier adversaire qu'un agent entraîné doit battre : il ne
    planifie rien, mais ne laisse passer aucune erreur à un coup.
    """

    def choose_action(self, board: Board) -> int:
        return tactical_move(*from_board(board, self.piece))


class MinimaxPlayer(Player):
    """
    Joueur minimax avec élagage alpha-bêta.

    Parameters
    ----------
    depth : int
        Nombre de coups d'avance. Les niveaux du site sont 1 (facile,
        avec randomness=0.35), 4 (moyen) et 6 (difficile).

    randomness : float
        Part de coups joués au hasard.
    """

    def __init__(
        self,
        piece: Cell,
        depth: int = 4,
        randomness: float = 0.0,
    ):
        super().__init__(piece)

        if depth < 1:
            raise ValueError("depth must be at least 1.")

        self.depth = depth
        self.randomness = randomness

    def choose_action(self, board: Board) -> int:
        return minimax_move(
            *from_board(board, self.piece),
            self.depth,
            self.randomness,
        )


class MonteCarloPlayer(Player):
    """
    Joueur par recherche arborescente de Monte-Carlo pure : il estime
    chaque coup en jouant des parties aléatoires, sans rien avoir appris.

    Parameters
    ----------
    simulations : int
        Nombre de parties simulées par coup. Plus il y en a, plus le
        joueur est fort et lent.
    """

    def __init__(
        self,
        piece: Cell,
        simulations: int = 1_000,
    ):
        super().__init__(piece)

        if simulations < 1:
            raise ValueError("simulations must be at least 1.")

        self.simulations = simulations

    def choose_action(self, board: Board) -> int:
        return mcts_move(
            *from_board(board, self.piece),
            self.simulations,
        )
