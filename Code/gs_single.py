"""Check training simple Monte-Carlo agent
with the Calvano environment and a dummy agent.
"""

from gs import CalvanoAgent, CalvanoTorch, DummyAgent, MDP
import matplotlib.pyplot as plt

import torch
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is {device}')



if __name__=="__main__":
    # Define agents
    dummy_price = 1.5
    agent1 = CalvanoAgent(0.01, device, 0.995)
    agent2 = DummyAgent(dummy_price, device)
    env = CalvanoTorch(np.array([1., 1.]), 1, 1., np.array([0., 0.]), device)

    # Define learning hyperparameters
    T = 6
    batch_size = 30
    learning_steps = 300

    # Run learning
    for i in range(learning_steps):
        agent1.zero_gradients()
        running_rewards, actions = MDP(agent1, agent2, env, batch_size, T, device)
        running_rewards[0, 0].backward(retain_graph=False)
        agent1.update()


    # Get sample history
    actions = torch.linspace(0, 5, 100, device=device).reshape(-1, 1)
    prices = torch.hstack((actions, actions*0 + 1.5), device=device)
    rewards = env.step(prices)


    plt.plot(actions, rewards[:, 0].flatten())
    plt.xlabel('actions')
    plt.ylabel('rewards')
    plt.title('rewards for each action')
    plt.savefig(f'rewards_dummy_{dummy_price}.png', dpi=300)
    plt.close()


    # Get sample actions
    states = torch.hstack((actions, actions*0 + 1.5))
    agent_action = agent1.get_action(state=states)
    plt.plot(actions.flatten(), agent_action.detach().numpy().flatten())
    plt.xlabel('state (previous action)')
    plt.ylabel('action')
    plt.title(f'Next actions based on previous action \n and dummy action {dummy_price}')
    plt.savefig(f'action_dummy_{dummy_price}.png', dpi=300)
    plt.close()