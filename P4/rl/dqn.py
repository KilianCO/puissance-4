# -*- coding: utf-8 -*-

"""
Deep Q-Network pour le Puissance 4.

DQN V1.

Fonctionnalités :
- réseau neuronal Q
- réseau cible (Target Network)
- Replay Buffer
- epsilon-greedy
- gestion des actions légales
- sauvegarde / chargement
- entraînement par mini-batch
"""

import random
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Constantes
# ============================================================

STATE_SIZE = 42
ACTION_SIZE = 7


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    """
    Mémoire des expériences de l'agent.

    Une expérience contient :

        state
        action
        reward
        next_state
        next_legal_actions
        done
    """

    def __init__(
        self,
        capacity: int = 100_000,
    ) -> None:

        self.buffer = deque(
            maxlen=capacity
        )

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        next_legal_actions,
        done,
    ) -> None:

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                next_legal_actions,
                done,
            )
        )

    def sample(
        self,
        batch_size: int,
    ):
        """
        Retourne un échantillon aléatoire
        du replay buffer.
        """

        return random.sample(
            self.buffer,
            batch_size,
        )

    def __len__(self) -> int:

        return len(self.buffer)


# ============================================================
# Réseau neuronal
# ============================================================

class QNetwork(nn.Module):
    """
    Réseau neuronal approximant :

        Q(state, action)

    Entrée :
        42 valeurs correspondant aux cases du plateau.

    Sortie :
        7 valeurs, une par colonne.
    """

    def __init__(self) -> None:

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                STATE_SIZE,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                ACTION_SIZE,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)


# ============================================================
# Agent DQN
# ============================================================

class DQNAgent:
    """
    Agent Deep Q-Learning.

    L'agent possède :

    - un réseau online ;
    - un réseau target ;
    - un replay buffer ;
    - un optimiseur ;
    - une stratégie epsilon-greedy.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99995,
        batch_size: int = 128,
        replay_capacity: int = 100_000,
        target_update_frequency: int = 1_000,
    ) -> None:

        # ----------------------------------------------------
        # Hyperparamètres
        # ----------------------------------------------------

        self.learning_rate = learning_rate

        self.discount_factor = (
            discount_factor
        )

        self.epsilon = epsilon

        self.epsilon_min = epsilon_min

        self.epsilon_decay = (
            epsilon_decay
        )

        self.batch_size = batch_size

        self.target_update_frequency = (
            target_update_frequency
        )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ----------------------------------------------------
        # Réseau online
        # ----------------------------------------------------

        self.online_network = (
            QNetwork()
            .to(self.device)
        )

        # IMPORTANT :
        # Le réseau online doit être entraînable.

        for parameter in (
            self.online_network.parameters()
        ):
            parameter.requires_grad_(True)

        self.online_network.train()

        # ----------------------------------------------------
        # Réseau target
        # ----------------------------------------------------

        self.target_network = (
            QNetwork()
            .to(self.device)
        )

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        # Le target network ne reçoit jamais
        # directement de gradients.

        for parameter in (
            self.target_network.parameters()
        ):
            parameter.requires_grad_(False)

        self.target_network.eval()

        # ----------------------------------------------------
        # Optimiseur
        # ----------------------------------------------------

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=self.learning_rate,
        )

        # ----------------------------------------------------
        # Fonction de loss
        # ----------------------------------------------------

        self.loss_function = (
            nn.SmoothL1Loss()
        )

        # ----------------------------------------------------
        # Replay Buffer
        # ----------------------------------------------------

        self.replay_buffer = (
            ReplayBuffer(
                capacity=replay_capacity
            )
        )

        # ----------------------------------------------------
        # Compteurs
        # ----------------------------------------------------

        self.training_steps = 0

        self.episodes = 0

    # ========================================================
    # Conversion état -> Tensor
    # ========================================================

    def _tensor_state(
        self,
        state,
    ) -> torch.Tensor:
        """
        Convertit un état du jeu en Tensor PyTorch.
        """

        return torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

    # ========================================================
    # Choix d'action
    # ========================================================

    def choose_action(
        self,
        state,
        legal_actions: list[int],
        training: bool = True,
    ) -> int:
        """
        Choisit une action selon epsilon-greedy.

        Pendant l'entraînement :

            epsilon élevé
                ↓
            exploration

        Pendant l'évaluation :

            training=False
                ↓
            exploitation pure
        """

        if not legal_actions:

            raise RuntimeError(
                "No legal action available."
            )

        # ----------------------------------------------------
        # Exploration
        # ----------------------------------------------------

        if (
            training
            and random.random()
            < self.epsilon
        ):

            return random.choice(
                legal_actions
            )

        # ----------------------------------------------------
        # Exploitation
        # ----------------------------------------------------

        state_tensor = (
            self._tensor_state(
                state
            )
            .unsqueeze(0)
        )

        # Aucun gradient nécessaire
        # pour choisir une action.

        with torch.no_grad():

            q_values = (
                self.online_network(
                    state_tensor
                )[0]
            )

        # ----------------------------------------------------
        # Actions légales uniquement
        # ----------------------------------------------------

        legal_q_values = {
            action:
                q_values[action].item()
            for action in legal_actions
        }

        max_q = max(
            legal_q_values.values()
        )

        # Plusieurs actions peuvent avoir
        # exactement la même Q-value.

        best_actions = [
            action
            for action, value
            in legal_q_values.items()
            if value == max_q
        ]

        return random.choice(
            best_actions
        )

    # ========================================================
    # Stockage d'une expérience
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        next_legal_actions,
        done,
    ) -> None:
        """
        Ajoute une expérience au replay buffer.
        """

        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            next_legal_actions,
            done,
        )

    # ========================================================
    # Entraînement
    # ========================================================

    def train_step(
        self,
    ) -> float | None:
        """
        Effectue une étape d'apprentissage du DQN.

        Retourne :

            float
                Loss de cette étape.

            None
                Si le replay buffer est trop petit.
        """

        # ----------------------------------------------------
        # Pas assez d'expériences
        # ----------------------------------------------------

        if (
            len(self.replay_buffer)
            < self.batch_size
        ):
            return None

        # ====================================================
        # IMPORTANT
        #
        # On force explicitement l'activation de autograd.
        #
        # Cela protège train_step() contre un éventuel
        # contexte extérieur torch.no_grad().
        # ====================================================

        with torch.enable_grad():

            # ------------------------------------------------
            # Le réseau online doit être entraînable.
            # ------------------------------------------------

            self.online_network.train()

            for parameter in (
                self.online_network.parameters()
            ):
                parameter.requires_grad_(True)

            # ------------------------------------------------
            # Batch
            # ------------------------------------------------

            batch = (
                self.replay_buffer.sample(
                    self.batch_size
                )
            )

            (
                states,
                actions,
                rewards,
                next_states,
                next_legal_actions,
                dones,
            ) = zip(*batch)

            # ------------------------------------------------
            # Conversion en Tensor
            # ------------------------------------------------

            states_tensor = torch.tensor(
                states,
                dtype=torch.float32,
                device=self.device,
            )

            next_states_tensor = torch.tensor(
                next_states,
                dtype=torch.float32,
                device=self.device,
            )

            actions_tensor = torch.tensor(
                actions,
                dtype=torch.long,
                device=self.device,
            )

            rewards_tensor = torch.tensor(
                rewards,
                dtype=torch.float32,
                device=self.device,
            )

            dones_tensor = torch.tensor(
                dones,
                dtype=torch.float32,
                device=self.device,
            )

            # =================================================
            # 1. Q(s, a)
            #
            # PAS de no_grad ici.
            # =================================================

            current_q_values_all = (
                self.online_network(
                    states_tensor
                )
            )

            current_q_values = (
                current_q_values_all
                .gather(
                    1,
                    actions_tensor.unsqueeze(1),
                )
                .squeeze(1)
            )

            # ------------------------------------------------
            # Vérification interne
            # ------------------------------------------------

            if not current_q_values.requires_grad:

                raise RuntimeError(
                    "DQN internal error: "
                    "current_q_values does not "
                    "require gradients."
                )

            # =================================================
            # 2. Q_target(s', a')
            #
            # Aucun gradient.
            # =================================================

            with torch.no_grad():

                next_q_values = (
                    self.target_network(
                        next_states_tensor
                    )
                )

                max_next_q_values = (
                    torch.zeros(
                        self.batch_size,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

                for index in range(
                    self.batch_size
                ):

                    legal_actions = (
                        next_legal_actions[index]
                    )

                    # État terminal.

                    if not legal_actions:
                        continue

                    legal_values = (
                        next_q_values[
                            index,
                            legal_actions,
                        ]
                    )

                    max_next_q_values[
                        index
                    ] = legal_values.max()

            # =================================================
            # 3. Équation de Bellman
            # =================================================

            targets = (
                rewards_tensor
                +
                (
                    1.0
                    - dones_tensor
                )
                *
                self.discount_factor
                *
                max_next_q_values
            )

            # =================================================
            # 4. Loss
            # =================================================

            loss = self.loss_function(
                current_q_values,
                targets,
            )

            if not loss.requires_grad:

                raise RuntimeError(
                    "DQN internal error: "
                    "loss does not require "
                    "gradients."
                )

            # =================================================
            # 5. Backpropagation
            # =================================================

            self.optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            # =================================================
            # 6. Gradient clipping
            # =================================================

            torch.nn.utils.clip_grad_norm_(
                self.online_network.parameters(),
                max_norm=1.0,
            )

            # =================================================
            # 7. Mise à jour online network
            # =================================================

            self.optimizer.step()

            self.training_steps += 1

            # =================================================
            # 8. Mise à jour target network
            # =================================================

            if (
                self.training_steps
                %
                self.target_update_frequency
                == 0
            ):

                self.target_network.load_state_dict(
                    self.online_network.state_dict()
                )

            return loss.item()

    # ========================================================
    # Décroissance epsilon
    # ========================================================

    def decay_epsilon(
        self,
    ) -> None:
        """
        Réduit progressivement epsilon.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay,
        )

    # ========================================================
    # Sauvegarde
    # ========================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Sauvegarde le modèle et les paramètres
        importants de l'entraînement.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "online_network":
                    self.online_network.state_dict(),

                "target_network":
                    self.target_network.state_dict(),

                "epsilon":
                    self.epsilon,

                "training_steps":
                    self.training_steps,

                "episodes":
                    self.episodes,

                "learning_rate":
                    self.learning_rate,

                "discount_factor":
                    self.discount_factor,

                "epsilon_min":
                    self.epsilon_min,

                "epsilon_decay":
                    self.epsilon_decay,

                "batch_size":
                    self.batch_size,

                "target_update_frequency":
                    self.target_update_frequency,
            },
            path,
        )

    # ========================================================
    # Chargement
    # ========================================================

    def load(
        self,
        path: str | Path,
    ) -> None:
        """
        Charge un modèle précédemment sauvegardé.
        """

        path = Path(path)

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self.online_network.load_state_dict(
            checkpoint[
                "online_network"
            ]
        )

        self.target_network.load_state_dict(
            checkpoint[
                "target_network"
            ]
        )

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon,
        )

        self.training_steps = (
            checkpoint.get(
                "training_steps",
                0,
            )
        )

        self.episodes = checkpoint.get(
            "episodes",
            0,
        )

        # ----------------------------------------------------
        # Configuration des réseaux après chargement
        # ----------------------------------------------------

        self.online_network.train()

        for parameter in (
            self.online_network.parameters()
        ):
            parameter.requires_grad_(True)

        self.target_network.eval()

        for parameter in (
            self.target_network.parameters()
        ):
            parameter.requires_grad_(False)

    # ========================================================
    # Test des gradients
    # ========================================================

    def test_gradients(
        self,
    ) -> None:
        """
        Teste que le réseau online est bien
        connecté au système de gradients.

        Cette méthode est principalement destinée
        au diagnostic.
        """

        with torch.enable_grad():

            self.online_network.train()

            state = torch.zeros(
                1,
                STATE_SIZE,
                dtype=torch.float32,
                device=self.device,
            )

            output = (
                self.online_network(
                    state
                )
            )

            print()
            print(
                "===== TEST GRADIENTS DQN ====="
            )

            print(
                "Grad enabled        :",
                torch.is_grad_enabled(),
            )

            print(
                "Network training    :",
                self.online_network.training,
            )

            print(
                "Output requires_grad:",
                output.requires_grad,
            )

            print(
                "Output grad_fn     :",
                output.grad_fn,
            )

            loss = output.mean()

            print(
                "Loss requires_grad  :",
                loss.requires_grad,
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            print(
                "Backward            : OK"
            )

            print(
                "=============================="
            )
            print()