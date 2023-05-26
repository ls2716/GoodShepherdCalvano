"""This file contains implementation of the Good Shepherd algorithm
agains bandits in a Calvano market environment.
"""
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

# Implement agents
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DummyAgent(object):

    def __init__(self, action, device) -> None:
        self.action = action

    def get_action(self, state):
        return self.action*torch.ones(size=(state.shape[0], 1), device=device)

    def update(self, *args):
        pass

    def reset(self):
        pass


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


class CalvanoAgent(object):
    """Implements calvano agent which takes previous history and outputs new price

    Arguments:
        - lr (float): learning rate
        - device (str): device
        - beta (float): discount factor for learning rate
    """

    def __init__(self, lr, beta, device) -> None:
        self.device = device
        self.model = MLP().to(device)
        self.lr = lr
        self.beta = beta
        # print("Model initialized")
        # print(self.model)

    def get_action(self, state):
        """Get action function based on the state.
        """
        state.reshape(-1, 2)
        return self.model(state)*3

    def zero_gradients(self):
        """Zero the gradients of the model"""
        self.model.zero_grad()

    def update_lr(self):
        """Update the learning rate"""
        self.lr = self.lr*self.beta

    def update(self):
        """Perform gradient descent"""
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 3.)
        for p in self.model.parameters():
            p.data = p.data + self.lr * p.grad

    def l2_loss(self, l2_lambda):
        """Compute l2 loss of the weights"""
        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.model.parameters())
        return -l2_norm*l2_lambda


class CalvanoTorch(object):
    """This class implements Calvano environment in PyTorch.

    Given n agents with product qualities a_i and prices p_i,
    the demands for ith product is:

    q_i = (exp((a_i-p_i)/mu)))/(\sum_{j=1}^n exp((a_j-p_j)/mu) + exp(a_0/mu)),
    where mu, a_0 are market parameters.   

    Then, the rewards are given by

    r_i = (p_i-c_i)q_i,
    where c_i are costs.

    """

    def __init__(self, A, mu, a_0, C, device) -> None:
        """Initialisation function.

        Arguments:
            - A (numpy array of floats): list of product quality indexes
            - mu (float): index of horizontal differentitation
            - a_0 (float) inverse index of aggregate demand
            - C (numpy array of floats): list of agent costs per product

        """
        self.A = torch.tensor(A, device=device)
        self.mu = mu
        self.a_0 = a_0
        self.C = torch.tensor(C, device=device)
        self.N = A.shape[0]

    def reset(self) -> None:
        """reset function

        Does nothing."""
        pass

    def step(self, P) -> torch.Tensor:
        """Step function.

        Given array of prices returns array of rewards.

        Arguments:
            - P (numpy array of floats): list of agent prices
        """
        # Fix prices shape
        P = P.reshape(-1, self.N)
        # Compute demands
        demands = torch.exp((self.A-P)/self.mu)
        demands = demands / \
            (torch.sum(torch.exp((self.A-P)/self.mu), axis=1)).reshape(-1, 1)
        # Return rewards
        return demands*(P-self.C)


def MDP(agent1, agent2, env, batch_size, T, device):
    """Perform episode simulation to obtain the rewards"""
    # Initialize running reward
    running_rewards = torch.zeros(size=(1, 2), device=device)
    # Initialize state using normal distribution with mean 2 and variance 2
    state = torch.randn(size=(batch_size, 2),
                        requires_grad=False, device=device)*2+2
    # Iterate over steps in episode
    for t in range(T):
        # Get actions for the states
        action1 = agent1.get_action(state)
        action2 = agent2.get_action(state)
        actions = torch.hstack((action1, action2))
        # Step actions and get rewards
        running_rewards += torch.mean(env.step(actions), dim=0, keepdim=True)
        state = actions

    return running_rewards, actions[0, :]
