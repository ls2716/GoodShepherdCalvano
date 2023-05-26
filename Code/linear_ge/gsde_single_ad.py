"""Check training simple Monte-Carlo agent
with the Calvano environment and a dummy agent with gradient estimation.
"""

from gsde import CalvanoAgent, CalvanoGradientAgent, CalvanoTorch, DummyAgent, MDP
import matplotlib.pyplot as plt

import torch
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is {device}')


if __name__ == "__main__":
    # Define agents
    dummy_price = 0.72
    agent1 = CalvanoGradientAgent(
        0.02, 0.99, device)
    agent2 = DummyAgent(dummy_price, device)
    # Define environment
    env = CalvanoTorch(np.array([1., 1.]), 0.3, 1., np.array([0., 0.]), device)

    # Define learning hyperparameters
    T = 5
    batch_size = 20
    learning_steps = 200
    scale = 0.05

    # Run learning
    for i in tqdm(range(learning_steps)):
        # print(f'Step {i}', end='\r')
        agent1.zero_gradients()
        running_rewards, actions = MDP(
            agent1, agent2, env, batch_size, T, device)
        running_rewards[0, 0].backward()
        agent1.update()

    # Get sample history
    actions = torch.linspace(0, 10, 5000, device=device).reshape(-1, 1)
    prices = torch.hstack((actions, actions*0 + dummy_price))
    rewards = env.step(prices)
    print('Best price', actions[np.argmax(
        rewards[:, 0].detach().cpu().numpy().flatten())])

    plt.plot(actions.detach().cpu().numpy().flatten(),
             rewards[:, 0].detach().cpu().numpy().flatten())
    plt.xlabel('actions')
    plt.ylabel('rewards')
    plt.title('rewards for each action')
    plt.grid()
    plt.savefig(f'images/gsde_rewards_dummy_{dummy_price}.png', dpi=300)
    plt.close()

    # Get sample actions
    states = torch.hstack((actions, actions*0 + dummy_price))
    agent_action = agent1.get_action(state=states).cpu()
    plt.plot(actions.detach().cpu().numpy().flatten(),
             agent_action.detach().cpu().numpy().flatten())
    plt.xlabel('state (previous action)')
    plt.ylabel('action')
    plt.title(
        f'Next actions based on previous action \n and dummy action {dummy_price}')
    plt.savefig(f'images/gsde_action_dummy_{dummy_price}_ad.png', dpi=300)
    plt.close()
