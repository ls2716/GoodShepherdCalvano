"""Example for checking that meshgrid is done in good order"""

import matplotlib.pyplot as plt
import numpy as np
import torch

device = 'cpu'


def get_action(states):
    return states[:, 0]+states[:, 1]*2


# Plotting outer agent strategy
K = 10  # set number of samples per dimension
agent_1_prices = torch.linspace(
    0, 3, K, device=device, requires_grad=False)
agent_2_prices = torch.linspace(
    0, 4, K*2, device=device, requires_grad=False)
grid_x, grid_y = torch.meshgrid(
    agent_1_prices, agent_2_prices, indexing='xy')
# print(grid_x.flatten().reshape(K*2, K))
# print(grid_y.flatten().reshape(K*2, K))

states = torch.hstack((grid_x.flatten().reshape(-1, 1),
                       grid_y.flatten().reshape(-1, 1)))
# print(states)
actions = get_action(states)
actions = actions.detach().cpu().numpy().reshape(K*2, K)
grid_x = grid_x.cpu().numpy()
grid_y = grid_y.cpu().numpy()
ax = plt.axes(projection='3d')
ax.plot_surface(grid_x, grid_y, actions)
ax.set_xlabel('1 agent')
ax.set_ylabel('2 agent')
plt.savefig('meshgrid_strategy.png')
plt.show()
