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


def inner_loop(outer_agent, model_index, inner_learning_steps, inner_batch_size, env, device, T, save_model=False):
    """Define reward of inner loop"""
    outer_running_reward = torch.Tensor([0])
    inner_agent = CalvanoGradientAgent(0.05, 0.99, device)
    for inner_it in range(inner_learning_steps):
        inner_agent.zero_gradients()
        # Get rewards
        running_rewards, last_actions = MDP(outer_agent, inner_agent,
                                            env, inner_batch_size, T, device, model_index=model_index)
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
        torch.save(inner_agent.model.state_dict(), inner_model_path)
    return outer_reward, inner_reward


if __name__ == "__main__":
    # # We start with defining two agents
    outer_agent = CalvanoAgent(0.02, beta=0.99, no_models=10, device=device)

    outer_model_path = 'models/outer.pth'

    # (optional) Load the model
    outer_agent.model.load_state_dict(torch.load(outer_model_path))

    # Define environment
    env = CalvanoTorch(np.array([1., 1.]), 0.3, 1., np.array([0., 0.]), device)

    # We define parameters
    T = 5
    inner_learning_steps = 100
    inner_batch_size = 20
    outer_learning_steps = 5
    scale = 0.02

    # Outer loop
    for outer_it in tqdm(range(outer_learning_steps)):
        outer_agent.regenerate_models()
        outer_reward, inner_reward = inner_loop(
            outer_agent, model_index=0, inner_batch_size=inner_batch_size,
            inner_learning_steps=inner_learning_steps,
            env=env, T=T, device=device)
        base_reward = outer_reward
        print(
            f'\n Outer reward is {base_reward}, inner reward is {inner_reward}')

        outer_agent.get_perturbation(scale)
        outer_agent.perturb_models()
        rewards = []
        params = []

        for model_index in range(outer_agent.no_models):
            outer_reward, inner_reward = inner_loop(
                outer_agent, model_index=model_index, inner_batch_size=inner_batch_size,
                inner_learning_steps=inner_learning_steps,
                env=env, T=T, device=device)
            rewards.append(outer_reward)

        rewards = torch.tensor(rewards)-base_reward
        outer_agent.get_best(rewards)
        # outer_agent.calculate_gradients(rewards)
        # outer_agent.update()

    outer_agent.regenerate_models()
    outer_reward, inner_reward = inner_loop(
        outer_agent, model_index=0, inner_batch_size=inner_batch_size,
        inner_learning_steps=inner_learning_steps,
        env=env, T=T, device=device, save_model=True)
    # Save outer model
    torch.save(outer_agent.model.state_dict(), outer_model_path)

    # Plotting outer agent strtegy
    K = 10  # set number of samples per dimension
    agent_1_prices = torch.linspace(
        0, 3, K, device=device, requires_grad=False)
    agent_2_prices = torch.linspace(
        0, 4, K*2, device=device, requires_grad=False)
    grid_x, grid_y = torch.meshgrid(
        agent_1_prices, agent_2_prices, indexing='xy')

    states = torch.hstack((grid_x.flatten().reshape(-1, 1),
                          grid_y.flatten().reshape(-1, 1)))
    actions = outer_agent.get_action(states, 0)
    actions = actions.detach().cpu().numpy().reshape(K*2, K)
    grid_x = grid_x.cpu().numpy()
    grid_y = grid_y.cpu().numpy()
    ax = plt.axes(projection='3d')
    ax.plot_surface(grid_x, grid_y, actions)
    ax.set_xlabel('Outer agent')
    ax.set_ylabel('Inner agent')
    plt.savefig('outer_strategy.png')
    plt.show()
