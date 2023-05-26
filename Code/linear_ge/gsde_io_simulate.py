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
    rewards_history = torch.zeros(size=(T, 2), device=device)
    actions_history = torch.zeros(size=(T, 2), device=device)

    state = torch.zeros(size=(1, 2),
                        requires_grad=False, device=device, dtype=torch.float32)

    for t in range(T):
        action1 = agent1.get_action(state, 0)
        action2 = agent2.get_action(state)

        actions = torch.hstack((action1, action2))
        actions_history[t, :] = actions[:, :]
        rewards = env.step(actions)
        rewards_history[t, :] = rewards[0, :]
        state = torch.zeros_like(actions, requires_grad=False)
        state[:, :] = actions[:, :]

    return rewards_history.detach().numpy(), actions_history.detach().numpy()


def simulate_learning(outer_agent, env, T, device):
    """Simulate learning and save history of actions"""
    # Train the inner model
    inner_learning_steps = 100
    inner_batch_size = 20
    rewards_learning = np.zeros(shape=(inner_learning_steps, T, 2))
    actions_learning = np.zeros(shape=(inner_learning_steps, T, 2))
    inner_agent = CalvanoGradientAgent(0.05, 0.99, device)

    for inner_it in tqdm(range(inner_learning_steps)):
        # Run game and get outputs
        rewards_history, actions_history = game(
            outer_agent, inner_agent, env, T, device)
        rewards_learning[inner_it, :, :] = rewards_history[:, :]
        actions_learning[inner_it, :, :] = actions_history[:, :]

        inner_agent.zero_gradients()
        # Get rewards
        running_rewards, last_actions = MDP(outer_agent, inner_agent,
                                            env, inner_batch_size, T, device, model_index=0)
        inner_running_reward = torch.Tensor([0]) + running_rewards[0, 1]
        inner_running_reward.backward(retain_graph=False)
        # Update inner agent
        inner_agent.update()

    return rewards_learning, actions_learning


if __name__ == "__main__":
    # # We start with defining two agents
    outer_agent = CalvanoAgent(0.002, beta=0.99, no_models=10, device=device)
    outer_model_path = 'models/outer.pth'

    # Load the model
    outer_agent.model.load_state_dict(torch.load(outer_model_path))
    outer_agent.regenerate_models()

    inner_agent = CalvanoGradientAgent(0.02, 0.99, device)
    inner_model_path = 'models/inner.pth'
    # Load the model
    inner_agent.model.load_state_dict(torch.load(inner_model_path))

    state = torch.zeros(size=(1, 2), dtype=torch.float32)
    action = outer_agent.get_action(state, 0)

    # Define environment
    env = CalvanoTorch(np.array([1., 1.]), 0.3, 1., np.array([0., 0.]), device)

    T = 5

    if False:
        # outer_agent = DummyAgent(2.0, device)
        rewards_learning, actions_learning = simulate_learning(
            outer_agent, env, T, device)
        steps_to_plot = list(range(0, 96, 4))
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

    # train inner agent
    inner_learning_steps = 100
    inner_batch_size = 20
    inner_agent = CalvanoGradientAgent(0.05, 0.99, device)

    for inner_it in tqdm(range(inner_learning_steps)):
        inner_agent.zero_gradients()
        # Get rewards
        running_rewards, last_actions = MDP(outer_agent, inner_agent,
                                            env, inner_batch_size, T, device, model_index=0)
        inner_running_reward = torch.Tensor([0]) + running_rewards[0, 1]
        inner_running_reward.backward(retain_graph=False)
        # Update inner agent
        inner_agent.update()
    print(running_rewards)

    if True:
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
        # plt.show()
        plt.close()

    T = 20
    if True:
        K = 200
        # Get sample actions
        agent_2_prices = torch.linspace(
            0, 10, K, device=device, requires_grad=False).reshape(-1, 1)
        states = torch.hstack((agent_2_prices*0+6, agent_2_prices))
        agent_action = outer_agent.get_action(
            state=states, model_index=0).cpu()
        plt.plot(agent_2_prices.detach().cpu().numpy().flatten(),
                 agent_action.detach().cpu().numpy().flatten())
        plt.xlabel('state (previous action)')
        plt.ylabel('action')
        plt.grid()
        plt.title(
            f'Next actions based on previous opponent action and my previous action 2.0 \n')
        plt.savefig(f'outer_strategy_for6.0.png', dpi=300)
        plt.show()

    if True:
        # Simulate a game (no randomness)
        rewards_history, actions_history = game(
            agent1=outer_agent, agent2=inner_agent, env=env, T=T, device=device)
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
        plt.show()
