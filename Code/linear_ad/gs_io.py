"""Implement Good Shepherd IO algorithm
with the Calvano environment and a dummy agent.
"""

from gs import CalvanoAgent, CalvanoTorch, DummyAgent, MDP
import matplotlib.pyplot as plt

import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is {device}')



if __name__=="__main__":
    # # We start with defining two agents
    outer_agent = CalvanoAgent(0.0001, beta=0.99, device=device)
    
    # Define environment
    env = CalvanoTorch(np.array([1., 1.]), 1, 1., np.array([0., 0.]), device)

    # We define parameters
    T = 6
    batch_size = 32
    inner_learning_steps = 30
    outer_learning_steps = 50

    # Outer loop
    for outer_it in range(outer_learning_steps):
        outer_running_reward = torch.Tensor([0])
        outer_agent.zero_gradients()
        # Inner loop
        inner_agent = CalvanoAgent(0.01, 0.99, device)
        for inner_it in range(inner_learning_steps):
            inner_agent.zero_gradients()
            # Get rewards
            running_rewards, last_actions = MDP(inner_agent, outer_agent,
                                                env, batch_size, T, device)
            inner_running_reward = torch.Tensor([0]) + running_rewards[0, 0]
            inner_running_reward.backward(retain_graph=True)
            outer_running_reward += running_rewards[0, 1]/inner_learning_steps
            last_outer_reward = torch.Tensor([0]) + running_rewards[0, 1]
            # Update inner agent
            inner_agent.update()
        # Print results
        print('outer iteration ', outer_it)
        print(inner_running_reward.item(), last_outer_reward.item(),
            last_actions.detach().numpy())
        outer_agent.zero_gradients()
        outer_running_reward.backward()
        outer_agent.update()
        print('outer agent l2')
        print(outer_agent.l2_loss(l2_lambda=1.))
    
    # Plotting outer agent strtegy
    K = 100 # set number of samples per dimension
    agent_1_prices = torch.linspace(0, 3, K, device=device, requires_grad=False)
    agent_2_prices = torch.linspace(0, 3, K, device=device, requires_grad=False)
    grid_x, grid_y = torch.meshgrid(agent_1_prices, agent_2_prices, indexing='xy')
    states = torch.hstack((grid_x.flatten().reshape(-1,1), grid_y.flatten().reshape(-1,1)))
    actions = outer_agent.get_action(states)
    actions = actions.detach().cpu().numpy().reshape(K,K)
    print(grid_x)
    grid_x = grid_x.cpu().numpy()
    grid_y = grid_y.cpu().numpy()
    ax = plt.axes(projection='3d')
    ax.plot_surface(grid_x, grid_y, actions)
    ax.set_xlabel('Inner agent')
    ax.set_ylabel('Outer agent')
    plt.show()
