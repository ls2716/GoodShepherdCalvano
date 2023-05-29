"""Implement Good Shepherd IO algorithm
with the Calvano environment and gradient estimation.
"""

from gsdis import CalvanoDiscreteGEAgent, CalvanoDiscreteADAgent, CalvanoDiscreteTorch, DummyAgent, MDP
import matplotlib.pyplot as plt

import torch
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is {device}')


def game(agent1, agent2, env, possible_actions, T, device):
    """Perform episode simulation to obtain the rewards"""
    # print('Probabilities of agent1')
    # print(agent1.get_probabilities(0))
    # print('Probabilities of agent2')
    # print(agent2.get_probabilities())

    rewards_history = torch.zeros(size=(T, 2), device=device)
    actions_history = torch.zeros(size=(T, 2), device=device)

    state = torch.zeros(size=(1,),
                        requires_grad=False, device=device, dtype=torch.int32)
    agent1_rewards, agent2_rewards = env.get_reward_vectors()

    for t in range(T):
        action1_index = agent1.get_action(state, 0)
        action2_index = agent2.get_action(state, flip=True)

        action1 = possible_actions[action1_index]
        action2 = possible_actions[action2_index]

        actions = torch.tensor([action1, action2]).reshape(1, -1)
        actions_history[t, :] = actions[:, :]

        state = (action1_index*len(possible_actions) +
                 action2_index).reshape(-1)

        reward1 = agent1_rewards[state, 0]
        reward2 = agent2_rewards[state, 0]
        rewards = torch.tensor([reward1, reward2]).reshape(1, -1)

        rewards_history[t, :] = rewards[0, :]
        state = (action1_index*len(possible_actions) +
                 action2_index).reshape(-1)

    return rewards_history.detach().numpy(), actions_history.detach().numpy()


def simulate_learning(outer_agent, env, T, possible_actions, device):
    """Simulate learning and save history of actions"""
    # Train the inner model
    inner_learning_steps = 400
    rewards_learning = np.zeros(shape=(inner_learning_steps, T, 2))
    actions_learning = np.zeros(shape=(inner_learning_steps, T, 2))
    inner_agent = CalvanoDiscreteADAgent(10., no_actions, device)

    for inner_it in tqdm(range(inner_learning_steps)):
        # Run game and get outputs
        rewards_history, actions_history = game(
            outer_agent, inner_agent, env, possible_actions, T, device)
        rewards_learning[inner_it, :, :] = rewards_history[:, :]
        actions_learning[inner_it, :, :] = actions_history[:, :]

        inner_agent.zero_gradients()
        # Get rewards
        running_rewards = MDP(outer_agent, inner_agent,
                              env, T, device, model_index=0)
        inner_running_reward = torch.Tensor([0]) + running_rewards[0, 1]
        inner_running_reward.backward(retain_graph=False)
        # Update inner agent
        inner_agent.update()

    print(inner_agent.get_probabilities())
    print(running_rewards)

    return rewards_learning, actions_learning


if __name__ == "__main__":
    possible_actions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    possible_actions = [0.5, 0.75, 1.]
    possible_actions = [0.5,  0.7,  0.9, 1.1]  # 4 actions
    possible_actions = [1.5,  2.0,  2.5, 3.]  # 4 actions Calvano
    possible_actions = [1.5,  2.0, 2.25, 2.5, 2.75, 3.]  # 6 actions Calvano
    no_actions = len(possible_actions)
    # We start with defining two agents
    outer_agent = CalvanoDiscreteGEAgent(
        0.2,  no_actions=no_actions, no_models=10, device=device)
    outer_model_path = 'models/outer_6_actions_Calvano.pth'

    # Load the model
    outer_agent.parameters = torch.load(outer_model_path)

    # print(outer_agent.parameters)
    # outer_agent.parameters[8, 1] = 10.
    # outer_agent.parameters[8, 2] = 0.
    # outer_agent.parameters[5, 1] = 10.
    # outer_agent.parameters[5, 2] = 0.
    # exit(0)
    outer_agent.regenerate_models()
    print(outer_agent.get_probabilities(0))

    inner_agent = CalvanoDiscreteADAgent(0.02, no_actions, device)
    inner_model_path = 'models/inner.pth'
    # Load the model
    inner_agent.parameters = torch.load(inner_model_path)

    state = torch.zeros(size=(1,), dtype=torch.int32)
    action = outer_agent.get_action(state, 0)

    # Define environment
    env = CalvanoDiscreteTorch(
        np.array([2., 2.]), 0.25, 0., np.array([1., 1.]), possible_actions, device)

    T = 20

    if True:
        # Simulate a game (no randomness)
        rewards_history, actions_history = game(
            agent1=outer_agent, agent2=inner_agent, env=env, possible_actions=possible_actions, T=T, device=device)
        fig, axs = plt.subplots(2, 1, figsize=(14, 10))
        axs[0].plot(rewards_history[:, 0], label='outer rewards')
        axs[0].plot(rewards_history[:, 1], label='inner rewards')
        axs[0].set_xlabel('time step')
        axs[0].grid()
        axs[0].set_title('Rewards')
        axs[0].legend()

        axs[1].plot(actions_history[:, 0], label='outer actions')
        axs[1].plot(actions_history[:, 1], label='inner actions')
        axs[1].set_xlabel('time step')
        axs[1].grid()
        axs[1].set_title('Actions')
        axs[1].legend()
        plt.savefig('game_inner_outer.png')
        # plt.show()
        plt.close()

    if True:
        T = 5
        # outer_agent = DummyAgent(2, no_actions, device)
        rewards_learning, actions_learning = simulate_learning(
            outer_agent, env, T, possible_actions, device)
        steps_to_plot = list(range(0, 400, 2))
        for i, step in enumerate(steps_to_plot):
            fig, axs = plt.subplots(2, 1, figsize=(10, 9))
            axs[0].plot(rewards_learning[step, :, 0], label='outer rewards')
            axs[0].plot(rewards_learning[step, :, 1], label='inner rewards')
            # axs[0].set_xlabel('time step')
            axs[0].grid()
            axs[0].set_title('Rewards')
            axs[0].legend()

            axs[1].plot(actions_learning[step, :, 0], label='outer actions')
            axs[1].plot(actions_learning[step, :, 1], label='inner actions')
            axs[1].set_xlabel('time step')
            axs[1].grid()
            axs[1].set_title('Actions')
            axs[1].legend()
            plt.savefig(f'learning_images/learning_step_{step}.png')
            # plt.show()
            plt.close()
