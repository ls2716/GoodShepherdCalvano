"""This file contains implementation of the Good Shepherd algorithm
agains bandits in a Calvano market environment.

Training using parameter perturbation
"""
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

# Implement agents
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DummyAgent(object):

    def __init__(self, action, device) -> None:
        self.action = action

    def get_action(self, state, *args):
        return self.action*torch.ones(size=(state.shape[0], 1), device=device)

    def update(self, *args):
        pass

    def reset(self):
        pass


class MLP_SingleOutput(nn.Module):
    def __init__(self):
        super(MLP_SingleOutput, self).__init__()
        self.price = nn.Parameter(data=torch.tensor([0.]))

    def forward(self, x):
        out = x[:, 0][:, None]*0+self.price
        return out


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.model(x)


class CalvanoAgent(object):
    """Implements calvano agent which takes previous history and outputs new price

    Arguments:
        - lr (float): learning rate
        - device (str): device
        - beta (float): discount factor for learning rate
    """

    def __init__(self, lr, beta, no_models, device) -> None:
        self.device = device
        self.model = MLP().to(device)
        self.lr = lr
        self.beta = beta
        # print("Model initialized")
        # print(self.model)
        self.no_models = no_models
        self.models = []
        for i in range(no_models):
            self.models.append(deepcopy(self.model))
        self.parameter_perturbations = []
        self.gradients = []
        for p in self.model.parameters():
            self.parameter_perturbations.append(
                torch.zeros(size=(self.no_models, *p.data.shape),
                            device=device, requires_grad=False)
            )
            self.gradients.append(torch.zeros_like(
                (p.data), requires_grad=False))

    # Define perturbation
    def get_perturbation(self, scale):
        for i in range(len(self.parameter_perturbations)):
            self.parameter_perturbations[i] = torch.randn_like(
                self.parameter_perturbations[i])*scale
        self.scale = scale

    def perturb_models(self):
        for i, model in enumerate(self.models):
            for j, p in enumerate(model.parameters()):
                p.data += self.parameter_perturbations[j][i, :]

    def regenerate_models(self):
        for i, model in enumerate(self.models):
            for p, q in zip(model.parameters(), self.model.parameters()):
                p.data[:] = q.data[:]

    def get_action(self, state, model_index):
        """Get action function based on the state.
        """
        return self.models[model_index](state)*2

    def zero_gradients(self):
        for i in range(len(self.gradients)):
            self.gradients[i] *= 0

    def calculate_gradients(self, rewards):
        self.zero_gradients()
        for i in range(self.no_models):
            for j, pp in enumerate(self.parameter_perturbations):
                self.gradients[j] += 1/self.scale**2 / \
                    self.no_models*pp[i]*rewards[i]

    def clip_gradients(self, clip):
        l2_norm = np.mean(p.pow(2.0).mean()
                          for p in self.gradients())
        print('l2 norm of gradients before clip', l2_norm)
        mu = np.sqrt(clip/l2_norm)
        for j, g in enumerate(self.gradients):
            self.gradients[j] *= mu
        l2_norm = np.mean(p.pow(2.0).mean()
                          for p in self.gradients())
        print('l2 norm of gradients before clip', l2_norm)

    def update_lr(self):
        """Update the learning rate"""
        self.lr = self.lr*self.beta

    def update(self):
        """Perform gradient ascent"""
        for i, p in enumerate(self.model.parameters()):
            p.data = p.data + self.lr * self.gradients[i]

    def get_best(self, rewards):
        """Get the best model"""
        best_index = np.argmax(rewards)
        print(f'Best model is {best_index}')
        for p, q in zip(self.models[best_index].parameters(), self.model.parameters()):
            q.data[:] = p.data[:]

    def l2_loss(self, l2_lambda):
        l2_norm = sum(p.pow(2.0).sum()
                      for p in self.model.parameters())
        return -l2_norm*l2_lambda


class CalvanoGradientAgent(object):
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


def MDP(agent1, agent2, env, batch_size, T, device, model_index=None):
    """Perform episode simulation to obtain the rewards"""
    running_rewards = torch.zeros(size=(1, 2), device=device)
    state = torch.randn(size=(batch_size, 2),
                        requires_grad=False, device=device, dtype=torch.float32)*2+2
    for t in range(T):
        if model_index is None:
            action1 = agent1.get_action(state)
        else:
            action1 = agent1.get_action(state, model_index)

        action2 = agent2.get_action(state)
        actions = torch.hstack((action1, action2))
        running_rewards += torch.mean(env.step(actions), dim=0, keepdim=True)/T
        state = torch.zeros_like(actions, requires_grad=False)
        state[:, :] = actions[:, :]

    return running_rewards, actions[0, :]
