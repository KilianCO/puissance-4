# Puissance 4

Moteur de jeu, joueurs et agents entraînés par apprentissage par renforcement (Q-learning, DQN).

```
P4/board.py, game.py, players.py   moteur de jeu
P4/interface/cli.py                partie en ligne de commande
P4/rl/                             environnement, agents et scripts d'entraînement
P4/simulate.py                     simulations entre joueurs
tests/                             tests du moteur
models/                            modèles entraînés (non versionnés)
```

## Lancer

Python 3.11. Toutes les commandes se lancent depuis la racine du dépôt.

```bash
pip install -r requirements.txt
python -m pytest
python -m P4.interface.cli
python -m P4.rl.train_dqn
```

Les modèles entraînés sont écrits dans `models/` et ne sont pas versionnés : ils sont publiés comme artefacts d'une GitHub Release.
