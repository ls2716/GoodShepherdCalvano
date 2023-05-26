"""Implement Good Shepherd IO algorithm
with the Calvano environment and gradient estimation.
"""

from gsdis import CalvanoDiscreteGEAgent, CalvanoDiscreteADAgent, CalvanoDiscreteTorch, DummyAgent, MDP
import matplotlib.pyplot as plt

import torch
import numpy as np
from tqdm import tqdm
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is {device}')


def inner_loop(outer_agent, model_index, inner_learning_steps, env, device, T, no_actions, save_model=False):
    """Define reward of inner loop"""
    outer_running_reward = torch.Tensor([0])
    inner_agent = CalvanoDiscreteADAgent(10., no_actions, device)
    for inner_it in range(inner_learning_steps):
        inner_agent.zero_gradients()
        # Get rewards
        running_rewards = MDP(outer_agent, inner_agent,
                              env, T, device, model_index=model_index)
        inner_running_reward = torch.Tensor([0]) + running_rewards[0, 1]
        inner_running_reward.backward(retain_graph=False)
        # outer_running_reward += running_rewards[0, 0]/inner_learning_steps
        # Update inner agent
        inner_agent.update()
    outer_running_reward += running_rewards[0, 0]
    print(outer_running_reward)
    outer_reward = outer_running_reward.detach().cpu().flatten()[0]
    inner_reward = inner_running_reward.detach().cpu().flatten()[0]
    if save_model:
        inner_model_path = 'models/inner.pth'
        torch.save(inner_agent.parameters, inner_model_path)
    return outer_reward, inner_reward


if __name__ == "__main__":
    possible_actions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    possible_actions = [0.5, 0.75, 1.]
    no_actions = len(possible_actions)
    # # We start with defining two agents
    outer_agent = CalvanoDiscreteGEAgent(
        0.2, no_actions=no_actions, no_models=20, device=device)

    outer_model_path = 'models/outer_3_actions.pth'

    # # (optional) Load the model
    # outer_agent.parameters = torch.load(outer_model_path)

    # Define environment
    env = CalvanoDiscreteTorch(np.array([1., 1.]), 0.2, 1., np.array(
        [0., 0.]), actions=possible_actions, device=device)

    # We define parameters
    T = 5
    inner_learning_steps = 200
    outer_learning_steps = 1
    scale = 0.3

    # Outer loop
    start_time = time.time()
    for outer_it in tqdm(range(outer_learning_steps)):
        outer_agent.regenerate_models()
        outer_reward, inner_reward = inner_loop(
            outer_agent, model_index=0,
            inner_learning_steps=inner_learning_steps,
            env=env, T=T, no_actions=no_actions, device=device)
        base_reward = outer_reward
        print(
            f'\n Outer reward is {base_reward}, inner reward is {inner_reward}')

        outer_agent.generate_perturbation(scale)
        outer_agent.perturb_models()
        rewards = []
        params = []

        for model_index in range(outer_agent.no_models):
            outer_reward, inner_reward = inner_loop(
                outer_agent, model_index=model_index,
                inner_learning_steps=inner_learning_steps,
                env=env, no_actions=no_actions, T=T, device=device)
            rewards.append(outer_reward)

        rewards = torch.tensor(rewards)-base_reward
        outer_agent.get_best(rewards)
        # outer_agent.calculate_gradients(rewards)
        # outer_agent.update()
    end_time = time.time()

    outer_agent.regenerate_models()
    outer_reward, inner_reward = inner_loop(
        outer_agent, model_index=0,
        inner_learning_steps=inner_learning_steps,
        env=env, T=T, no_actions=no_actions, device=device, save_model=True)
    # Save outer model
    torch.save(outer_agent.parameters, outer_model_path)

    # Printing strategy
    print('Outer player strategy')
    print(outer_agent.compute_probabilities(model_index=0))

    print(f"Time elapsed {end_time-start_time}")
