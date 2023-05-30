"""Check training simple Monte-Carlo agent
with the Calvano environment and a dummy agent with gradient estimation.
"""
import global_config

from gsdis import DummyAgent, MDP, CalvanoDiscreteADAgent, CalvanoDiscreteTorch
import matplotlib.pyplot as plt

import torch
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is {device}')


if __name__ == "__main__":
    device = 'cpu'
    possible_actions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    possible_actions = [3.]  # 4 actions

    possible_actions = [1.5,  1.8, 2.1, 2.4, 2.7, 3.]  # 6 actions Calvano
    # possible_actions = [1.5,  2.0,  2.5, 3.]  # 4 actions Calvano
    # possible_actions = [1.5,  2.25, 3.]  # 3 actions Calvano
    no_actions = len(possible_actions)
    # Define agents
    agent1 = CalvanoDiscreteADAgent(20., no_actions, device)
    dummy_action = 0
    agent2 = DummyAgent(dummy_action, no_actions, device)
    # Define environment
    env = CalvanoDiscreteTorch(np.array([2., 2.]), 0.25, 0., np.array(
        [1., 1.]), possible_actions, device=device)

    # Define learning hyperparameters
    T = 1
    learning_steps = 400

    # Run learning
    for i in tqdm(range(learning_steps)):
        running_rewards = MDP(
            agent1, agent2, env, T, device, None)
        agent1.zero_gradients()
        running_rewards[0, 0].backward()
        # print(agent1.parameters.grad)
        agent1.update()

    print('Agent 1 probabilities')
    agent1.compute_probabilities()
    print(agent1.probs)
    print(agent1.parameters)

    dummy_price = possible_actions[dummy_action]
    # Get sample history
    actions = torch.linspace(0, 3, 1000, device=device).reshape(-1, 1)
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
    plt.savefig(f'images/rewards_dummy_{dummy_price}.png', dpi=300)
    plt.close()

    print('Best monopolistic price', actions[np.argmax(
        rewards[:, 0].detach().cpu().numpy().flatten()+rewards[:, 1].detach().cpu().numpy().flatten())])

    plt.plot(actions.detach().cpu().numpy().flatten(),
             rewards[:, 0].detach().cpu().numpy().flatten()+rewards[:, 1].detach().cpu().numpy().flatten())
    plt.xlabel('actions')
    plt.ylabel('rewards')
    plt.title('rewards for each action')
    plt.grid()
    plt.savefig(f'images/mon_rewards_dummy_{dummy_price}.png', dpi=300)
    plt.close()

    # # Get sample actions
    # states = torch.hstack((actions, actions*0 + dummy_price))
    # agent_action = agent1.get_action(state=states).cpu()
    # plt.plot(actions.detach().cpu().numpy().flatten(),
    #          agent_action.detach().cpu().numpy().flatten())
    # plt.xlabel('state (previous action)')
    # plt.ylabel('action')
    # plt.title(
    #     f'Next actions based on previous action \n and dummy action {dummy_price}')
    # plt.savefig(f'images/gsde_action_dummy_{dummy_price}_ad.png', dpi=300)
    # plt.close()
