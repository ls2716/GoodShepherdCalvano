"""Check training simple Monte-Carlo agent
with the Calvano environment and a dummy agent with gradient estimation.
"""

from gsdis import DummyAgent, MDP, CalvanoDiscreteGEAgent, CalvanoDiscreteTorch
import matplotlib.pyplot as plt

import torch
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is {device}')


if __name__ == "__main__":
    device = 'cpu'
    possible_actions = [0.5, 1., 1.5]
    no_actions = len(possible_actions)
    # Define agents
    no_models = 20
    agent1 = CalvanoDiscreteGEAgent(10., no_actions, device, no_models)
    dummy_action = 1
    agent2 = DummyAgent(dummy_action, no_actions, device)
    # Define environment
    env = CalvanoDiscreteTorch(np.array([1., 1.]), 0.3, 1., np.array(
        [0., 0.]), possible_actions, device=device)

    # Define learning hyperparameters
    T = 1
    learning_steps = 100
    scale = 0.05

    # Run learning
    for i in tqdm(range(learning_steps)):
        running_rewards = MDP(
            agent1, agent2, env, T, device, 0)
        base_reward = running_rewards[0, 0].detach().cpu().numpy()
        rewards = []
        agent1.regenerate_models()
        # print('Regeneration')
        # print(agent1.parameters)
        # print(agent1.models[0])
        agent1.generate_perturbation(scale=scale)
        agent1.perturb_models()
        for model_index in range(agent1.no_models):
            running_rewards = MDP(
                agent1, agent2, env, T, device, model_index)
            rewards.append(running_rewards[0, 0].detach().cpu().numpy())
        agent1.get_best(rewards)

    print('Agent 1 probabilities')
    agent1.regenerate_models()
    probs = agent1.compute_probabilities(0)
    print(probs)
    print(agent1.parameters)
