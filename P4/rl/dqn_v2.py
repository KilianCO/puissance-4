# -*- coding: utf-8 -*-

"""
Deep Q-Network V2 pour le Puissance 4.

Différences avec la V1 (dqn.py, conservée comme référence) :

- l'état est encodé en deux plans 6 x 7 (pions du joueur qui doit jouer,
  pions adverses) au lieu de 42 valeurs 0/1/2 ;
- le réseau est convolutif : il reconnaît un alignement où qu'il soit ;
- l'agent apprend des deux côtés de chaque partie. La valeur d'une
  position pour un joueur est l'opposé de sa valeur pour l'autre, donc :

      cible = récompense                      si la partie est finie
      cible = -gamma * max Q(état suivant)    sinon

  (l'état suivant est vu par l'adversaire, d'où le signe moins) ;
- Double DQN : le réseau online choisit le coup suivant, le réseau cible
  l'évalue ;
- chaque position est aussi apprise en miroir gauche-droite ;
- le replay buffer vit sur le device et les cibles sont calculées par
  lot, sans boucle Python.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ROWS = 6
COLUMNS = 7

FORMAT = "dqn_v2"


# ============================================================
# Réseau neuronal
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.conv_1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv_1(x))
        y = self.conv_2(y)

        return torch.relu(x + y)


class ConvQNetwork(nn.Module):
    """
    Entrée  : float32 [N, 2, 6, 7].
    Sortie  : float32 [N, 7], une valeur entre -1 et 1 par colonne.

    C'est exactement le contrat attendu par la démo du site.
    """

    def __init__(
        self,
        channels: int = 64,
        blocks: int = 3,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.blocks = blocks

        self.stem = nn.Conv2d(2, channels, 3, padding=1)

        self.body = nn.Sequential(*(
            ResidualBlock(channels)
            for _ in range(blocks)
        ))

        self.reduce = nn.Conv2d(channels, 8, 1)

        self.head = nn.Sequential(
            nn.Linear(8 * ROWS * COLUMNS, 128),
            nn.ReLU(),
            nn.Linear(128, COLUMNS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.stem(x))
        x = self.body(x)
        x = torch.relu(self.reduce(x))
        x = self.head(x.flatten(1))

        # Les valeurs réelles sont entre -1 (défaite) et 1 (victoire).
        return torch.tanh(x)


# ============================================================
# Replay buffer
# ============================================================

class ReplayBuffer:
    """
    Mémoire circulaire stockée directement sur le device.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
    ) -> None:

        self.capacity = capacity
        self.device = device

        self.states = torch.zeros(
            (capacity, 2, ROWS, COLUMNS),
            dtype=torch.uint8,
            device=device,
        )
        self.next_states = torch.zeros_like(self.states)

        self.actions = torch.zeros(
            capacity, dtype=torch.long, device=device,
        )
        self.rewards = torch.zeros(
            capacity, dtype=torch.float32, device=device,
        )
        self.dones = torch.zeros(
            capacity, dtype=torch.float32, device=device,
        )
        self.next_legal = torch.zeros(
            (capacity, COLUMNS), dtype=torch.bool, device=device,
        )

        self.position = 0
        self.size = 0

        # Nombre total d'expériences reçues, y compris celles écrasées.
        self.pushes = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_legal_actions: list[int],
        done: bool,
    ) -> None:

        index = self.position

        self.states[index] = torch.from_numpy(state)
        self.next_states[index] = torch.from_numpy(next_state)
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = float(done)

        self.next_legal[index] = False

        if next_legal_actions:
            self.next_legal[index, next_legal_actions] = True

        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.pushes += 1

    def sample(self, batch_size: int):
        indices = torch.randint(
            0, self.size, (batch_size,), device=self.device,
        )

        return (
            self.states[indices].float(),
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices].float(),
            self.next_legal[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


# ============================================================
# Agent
# ============================================================

class DQNv2Agent:
    """
    Agent DQN V2.

    `epsilon` est piloté par le script d'entraînement : l'agent ne le
    fait pas décroître lui-même.
    """

    def __init__(
        self,
        learning_rate: float = 5e-4,
        discount_factor: float = 0.98,
        batch_size: int = 256,
        replay_capacity: int = 300_000,
        target_update_frequency: int = 1_000,
        channels: int = 64,
        blocks: int = 3,
        device: str | None = None,
    ) -> None:

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.target_update_frequency = target_update_frequency

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.online_network = ConvQNetwork(channels, blocks).to(self.device)
        self.target_network = ConvQNetwork(channels, blocks).to(self.device)

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )
        self.target_network.requires_grad_(False)
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(replay_capacity, self.device)

        self.epsilon = 0.0
        self.training_steps = 0
        self.episodes = 0

    # --------------------------------------------------------
    # Choix d'action
    # --------------------------------------------------------

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> list[float]:
        """Valeur de chaque colonne pour le joueur qui doit jouer."""
        tensor = (
            torch.from_numpy(state)
            .to(self.device, torch.float32)
            .unsqueeze(0)
        )

        return self.online_network(tensor)[0].tolist()

    def choose_action(
        self,
        state: np.ndarray,
        legal_actions: list[int],
        training: bool = False,
    ) -> int:
        """
        Meilleur coup légal ; pendant l'entraînement, coup au hasard
        avec la probabilité epsilon.
        """
        if not legal_actions:
            raise RuntimeError("No legal action available.")

        if training and random.random() < self.epsilon:
            return random.choice(legal_actions)

        values = self.q_values(state)

        return max(legal_actions, key=lambda action: values[action])

    # --------------------------------------------------------
    # Apprentissage
    # --------------------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_legal_actions: list[int],
        done: bool,
    ) -> None:

        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            next_legal_actions,
            done,
        )

    def train_step(self) -> float | None:
        """
        Une étape d'apprentissage. Retourne la loss, ou None si le
        replay buffer est encore trop petit.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            next_legal,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        # Miroir gauche-droite sur la moitié du lot : le jeu est
        # symétrique, chaque partie en vaut deux.
        mirrored = (
            torch.rand(self.batch_size, device=self.device) < 0.5
        )

        states = torch.where(
            mirrored[:, None, None, None], states.flip(-1), states,
        )
        next_states = torch.where(
            mirrored[:, None, None, None], next_states.flip(-1), next_states,
        )
        next_legal = torch.where(
            mirrored[:, None], next_legal.flip(-1), next_legal,
        )
        actions = torch.where(
            mirrored, COLUMNS - 1 - actions, actions,
        )

        current_q = (
            self.online_network(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )

        with torch.no_grad():
            # Double DQN : online choisit, target évalue.
            next_online = self.online_network(next_states)
            next_online = next_online.masked_fill(~next_legal, -2.0)
            next_actions = next_online.argmax(1, keepdim=True)

            next_q = (
                self.target_network(next_states)
                .gather(1, next_actions)
                .squeeze(1)
            )

            # L'état suivant est vu par l'adversaire : signe moins.
            targets = (
                rewards
                - (1.0 - dones) * self.discount_factor * next_q
            )

        loss = self.loss_function(current_q, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=5.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return loss.item()

    # --------------------------------------------------------
    # Sauvegarde / chargement
    # --------------------------------------------------------

    def save(self, path: str | Path, **extra) -> None:
        """
        Sauvegarde les réseaux, l'optimiseur et les compteurs. Le replay
        buffer n'est pas sauvegardé : il se reconstitue à la reprise.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "format": FORMAT,
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "channels": self.online_network.channels,
                "blocks": self.online_network.blocks,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "batch_size": self.batch_size,
                "replay_capacity": self.replay_capacity,
                "target_update_frequency": self.target_update_frequency,
                "training_steps": self.training_steps,
                "episodes": self.episodes,
                **extra,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | None = None,
        replay_capacity: int | None = None,
    ) -> "DQNv2Agent":
        """Recrée un agent à partir d'un checkpoint."""
        checkpoint = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} checkpoint.")

        agent = cls(
            learning_rate=checkpoint["learning_rate"],
            discount_factor=checkpoint["discount_factor"],
            batch_size=checkpoint["batch_size"],
            replay_capacity=(
                replay_capacity
                if replay_capacity is not None
                else checkpoint["replay_capacity"]
            ),
            target_update_frequency=checkpoint["target_update_frequency"],
            channels=checkpoint["channels"],
            blocks=checkpoint["blocks"],
            device=device,
        )

        agent.online_network.load_state_dict(checkpoint["online_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])

        agent.training_steps = checkpoint["training_steps"]
        agent.episodes = checkpoint["episodes"]

        return agent
