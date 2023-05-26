"""Check training simple Monte-Carlo agent
with the Calvano environment and a dummy agent with gradient estimation.
"""

from gsde import CalvanoAgent, CalvanoTorch, DummyAgent, MDP
import matplotlib.pyplot as plt

import torch
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is {device}')
torch.set_grad_enabled(False)


if __name__ == "__main__":
    # Define agents
    dummy_price = 2.0
    agent1 = CalvanoAgent(
        0.05, 0.99, 50, device)
    agent2 = DummyAgent(dummy_price, device)
    # Define environment
    env = CalvanoTorch(np.array([1., 1.]), 1, 1., np.array([0., 0.]), device)

    # Define learning hyperparameters
    T = 1
    batch_size = 20
    learning_steps = 300
    scale = 0.05

    # Run learning
    for i in tqdm(range(learning_steps)):
        # print(f'Step {i}', end='\r')
        agent1.regenerate_models()
        running_rewards, actions = MDP(
            agent1, agent2, env, batch_size, T, device, 0)
        base_reward = running_rewards.detach().numpy()[0, 0]

        agent1.get_perturbation(scale)
        agent1.perturb_models()
        rewards = []
        params = []
        for j in range(agent1.no_models):
            running_rewards, actions = MDP(
                agent1, agent2, env, batch_size, T, device, j)
            run_rewards = running_rewards.detach().numpy()
            rewards.append(run_rewards[0, 0])

        rewards = torch.tensor(rewards)-base_reward
        # print('Reward', torch.mean(rewards))
        agent1.calculate_gradients(rewards)
        # print('Gradients', agent1.gradients)
        # print('Parameter', end=' ')
        # for p in agent1.model.parameters():
        #     print(p.data)
        agent1.update()

    # Get sample history
    actions = torch.linspace(0, 5, 1000, device=device).reshape(-1, 1)
    prices = torch.hstack((actions, actions*0 + 2.0))
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
    states = torch.hstack((actions, actions*0 + 1.5))
    agent_action = agent1.get_action(state=states, model_index=0).cpu()
    plt.plot(actions.detach().cpu().numpy().flatten(),
             agent_action.detach().cpu().numpy().flatten())
    plt.xlabel('state (previous action)')
    plt.ylabel('action')
    plt.title(
        f'Next actions based on previous action \n and dummy action {dummy_price}')
    plt.savefig(f'images/gsde_action_dummy_{dummy_price}_ge.png', dpi=300)
    plt.close()
