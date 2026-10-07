# Puissance 4

Moteur de jeu, adversaires de référence et agents entraînés par apprentissage par renforcement.
Le modèle entraîné est jouable sur [kilianco.github.io](https://kilianco.github.io/projets/puissance-4/).

La démarche, les choix techniques et les résultats mesurés sont résumés dans [docs/APPROCHE.md](docs/APPROCHE.md).

```
P4/board.py, game.py        moteur de jeu
P4/players.py               joueurs : humain, aléatoire, tactique, minimax, Monte-Carlo
P4/bitboard.py              plateau compact
P4/minimax.py, mcts.py      recherche alpha-bêta, recherche de Monte-Carlo
P4/arena.py                 banc d'évaluation entre joueurs
P4/league.py                tournoi entre tous les joueurs et classement Elo
P4/interface/cli.py         partie en ligne de commande
P4/simulate.py              simulations entre joueurs
P4/rl/                      agents, entraînement, export
tests/                      tests
models/                     modèles entraînés (non versionnés)
```

## Installer

Python 3.11. Toutes les commandes se lancent depuis la racine du dépôt.

```bash
pip install -r requirements.txt
python -m pytest
```

## Jouer

```bash
python -m P4.interface.cli
```

Contre un autre humain, l'aléatoire, le joueur tactique, le minimax ou un modèle entraîné présent dans `models/`.

## Entraîner

```bash
python -m P4.rl.train_dqn_v2 --episodes 40000            # nouvel entraînement
python -m P4.rl.train_dqn_v2 --episodes 80000 --resume   # le prolonger
```

L'agent joue contre lui-même et contre les adversaires de référence. Toutes les 2 000 parties, il est évalué sans exploration ; `models/dqn_v2/` reçoit le dernier état (`last.pt`), le meilleur modèle (`best.pt`) et le journal (`log.csv`).

## Évaluer

```bash
python -m P4.arena --model models/dqn_v2/best.pt
```

Chaque joueur affronte les mêmes adversaires, la moitié des parties dans chaque position :

| Adversaire | Description |
|---|---|
| aléatoire | joue au hasard |
| tactique | gagne ou bloque quand c'est possible à un coup, au hasard sinon |
| minimax 1, 4, 6 | alpha-bêta ; ce sont les niveaux facile, moyen et difficile du site |

Avec `--opening 4`, les quatre premiers coups sont joués au hasard : la mesure porte alors sur des positions variées.

## Classer tous les joueurs

```bash
python -m P4.league --games 40 --output docs/ligue.json
```

Tous les joueurs (aléatoire, tactique, minimax, Monte-Carlo, modèles entraînés) s'affrontent deux à deux, puis reçoivent une note Elo. Le dernier classement est dans [docs/ligue.json](docs/ligue.json) et commenté dans [docs/APPROCHE.md](docs/APPROCHE.md).

## Publier sur le site

```bash
python -m P4.rl.export_onnx models/dqn_v2/best.pt models/puissance4.onnx
```

Le réseau est exporté en ONNX (entrée `[N, 2, 6, 7]`, sortie `[N, 7]`) et vérifié avec ONNX Runtime. Le fichier se copie ensuite dans `static/models/` du dépôt du site.

## Agents

| Agent | Fichiers | État |
|---|---|---|
| Monte-Carlo (MCTS) | `mcts.py` | recherche sans apprentissage : rien à entraîner |
| DQN V2 | `rl/dqn_v2.py`, `rl/train_dqn_v2.py` | réseau convolutif, self-play ; modèle du site |
| DQN V1 | `rl/dqn.py`, `rl/train_dqn.py` | conservé comme référence |
| Q-learning tabulaire | `rl/q_learning.py`, `rl/train_qlearning.py` | conservé comme référence |
