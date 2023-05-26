"""Implement Good Shepherd IO algorithm
with the Calvano environment and gradient estimation.
"""

from gsde import CalvanoAgent, CalvanoGradientAgent, CalvanoTorch, DummyAgent, MDP
import matplotlib.pyplot as plt

import torch
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is {device}')


def game(agent1, agent2, env, T, device):
    """Perform episode simulation to obtain the rewards"""
    ...


def simulate_learning(outer_agent, env, T, device):
    """Simulate learning and save history of actions"""
    # Train the inner model
    ...


if __name__ == "__main__":
    # # We start with defining two agents
    ...
