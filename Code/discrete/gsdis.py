"""This file contains implementation of the Good Shepherd algorithm
agains bandits in a Calvano market environment.

Discrete with gradient estimation
"""
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

# Implement agents
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DummyAgent(object):

    def __init__(self, action, no_actions, device) -> None:
        self.parameters = torch.zeros(size=(
            no_actions*no_actions, no_actions), requires_grad=False, dtype=torch.float64, device=device)
        self.parameters[:, action] += 10.
        self.no_actions = no_actions

    def compute_probabilities(self):
        self.probs = torch.softmax(self.parameters, dim=1)

    def get_probabilities(self, model_index=None):
        self.compute_probabilities()
        return self.probs

    def get_action(self, state, model_index=None, flip=False):
        no_actions = self.no_actions
        probs = self.get_probabilities()
        if flip:
            probs = probs.reshape(no_actions, no_actions, no_actions).permute(
                1, 0, 2).reshape(-1, no_actions)
        probs_state = probs[state, :]
        action_index = torch.multinomial(probs_state, 1)
        return action_index


class CalvanoDiscreteADAgent(object):
    """Implements calvano agent which takes previous history and outputs new price

    Arguments:
        - lr (float): learning rate
        - device (str): device
        - beta (float): discount factor for learning rate
    """

    def __init__(self, lr, no_actions, device, fixed_actions=None) -> None:
        self.device = device
        self.no_actions = no_actions
        self.lr = lr
        self.parameters = torch.ones(
            size=(no_actions*no_actions, no_actions),  dtype=torch.float64, requires_grad=True)
        if fixed_actions is not None:
            self.parameters = torch.tensor(
                fixed_actions, dtype=torch.float64, requires_grad=True)

    def compute_probabilities(self):
        self.probs = torch.softmax(self.parameters, dim=1)

    def get_probabilities(self):
        self.compute_probabilities()
        return self.probs

    def normalize_parameters(self):
        self.parameters = self.parameters/torch.norm(self.parameters, 1)

    def update(self):
        with torch.no_grad():
            self.parameters += self.lr * self.parameters.grad

    def zero_gradients(self):
        self.parameters.grad = None

    def get_action(self, state, flip=False):
        no_actions = self.no_actions
        probs = self.get_probabilities()
        if flip:
            probs = probs.reshape(no_actions, no_actions, no_actions).permute(
                1, 0, 2).reshape(-1, no_actions)
        probs_state = probs[state, :]
        action_index = torch.multinomial(probs_state, 1)
        return action_index


class CalvanoDiscreteGEAgent(object):
    """Implements calvano agent which takes previous history and outputs new price

    Arguments:
        - lr (float): learning rate
        - device (str): device
        - beta (float): discount factor for learning rate
    """

    def __init__(self, lr, no_actions, device, no_models, fixed_actions=None) -> None:
        self.device = device
        self.no_actions = no_actions
        self.lr = lr
        self.no_models = no_models
        self.parameters = torch.ones(
            size=(no_actions*no_actions, no_actions),  dtype=torch.float64, requires_grad=False)
        self.models = []
        self.gradients = torch.zeros(
            size=(no_actions*no_actions, no_actions),  dtype=torch.float64, requires_grad=False)
        for i in range(no_models):
            self.models.append(torch.ones(
                size=(no_actions*no_actions, no_actions),  dtype=torch.float64, requires_grad=False))
        if fixed_actions is not None:
            self.parameters = torch.tensor(
                fixed_actions, dtype=torch.float64, requires_grad=True)

    def generate_perturbation(self, scale):
        """Generate perturbations for the model"""
        self.perturbations = []
        for i in range(self.no_models):
            self.perturbations.append(torch.randn_like(
                self.parameters,  dtype=torch.float64, requires_grad=False)*scale)
        self.scale = scale

    def perturb_models(self):
        for i in range(self.no_models):
            self.models[i] += self.perturbations[i]

    def regenerate_models(self):
        for i in range(self.no_models):
            self.models[i][:] = self.parameters[:]

    def compute_probabilities(self, model_index):
        return torch.softmax(self.models[model_index], dim=1)

    def get_probabilities(self, model_index):
        return self.compute_probabilities(model_index=model_index)

    def normalize_parameters(self):
        self.parameters = self.parameters/torch.norm(self.parameters, 1)

    def zero_gradients(self):
        self.gradients *= 0

    def calculate_gradients(self, rewards):
        self.zero_gradients()
        # print('Gradient computation')
        # print(self.gradients)
        for i in range(self.no_models):
            self.gradients += 1/self.scale**2 / \
                self.no_models*self.perturbations[i]*rewards[i]
        # print(self.gradients)

    def update_lr(self):
        """Update the learning rate"""
        self.lr = self.lr*self.beta

    def update(self):
        """Perform gradient ascent"""
        self.parameters += self.lr * self.gradients

    def get_best(self, rewards):
        """Get the best model"""
        best_index = np.argmax(rewards)
        self.parameters[:] = self.models[best_index][:]

    def get_action(self, state, model_index, flip=False):
        no_actions = self.no_actions
        probs = self.get_probabilities(model_index)
        if flip:
            probs = probs.reshape(no_actions, no_actions, no_actions).permute(
                1, 0, 2).reshape(-1, no_actions)
        probs_state = probs[state, :]
        action_index = torch.multinomial(probs_state, 1)
        return action_index


class CalvanoDiscreteTorch(object):
    """This class implements Calvano environment in PyTorch.

    Given n agents with product qualities a_i and prices p_i,
    the demands for ith product is:

    q_i = (exp((a_i-p_i)/mu)))/(\sum_{j=1}^n exp((a_j-p_j)/mu) + exp(a_0/mu)),
    where mu, a_0 are market parameters.

    Then, the rewards are given by

    r_i = (p_i-c_i)q_i,
    where c_i are costs.

    """

    def __init__(self, A, mu, a_0, C, actions, device) -> None:
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
        self.actions = torch.tensor(
            actions, device=device, dtype=torch.float64)

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

    def get_reward_vectors(self):
        action1, action2 = torch.meshgrid(
            self.actions, self.actions, indexing='ij')
        joint_actions = torch.hstack((
            action1.reshape(-1, 1), action2.reshape(-1, 1)))
        rewards = self.step(joint_actions)
        return rewards[:, 0].reshape(-1, 1), rewards[:, 1].reshape(-1, 1)


def MDP(agent1, agent2, env: CalvanoDiscreteTorch, T, device, model_index=None):
    """Perform episode simulation to obtain the rewards"""
    no_actions = agent1.no_actions
    state = torch.ones(size=(no_actions*no_actions, 1),
                       requires_grad=True, device=device, dtype=torch.float64)/no_actions/no_actions

    if model_index is None:
        agent1_probs = agent1.get_probabilities()
    else:
        agent1_probs = agent1.get_probabilities(model_index)
    agent2_probs = agent2.get_probabilities()
    agent2_probs = agent2_probs.reshape(
        no_actions, no_actions, no_actions).permute(1, 0, 2).reshape(-1, no_actions)

    agent1_probs_rep = agent1_probs.reshape(-1).repeat_interleave(no_actions)
    agent2_probs_rep = agent2_probs.repeat_interleave(
        no_actions, 0).reshape(-1)

    transformation_matrix = (
        agent1_probs_rep*agent2_probs_rep).reshape(no_actions*no_actions, no_actions*no_actions).T
    # print(transformation_matrix)

    agent1_rewards, agent2_rewards = env.get_reward_vectors()
    # print(agent1_rewards, agent2_rewards)
    rewards = torch.zeros(
        size=(1, 2), dtype=torch.float64)
    # Define rewards as vectors
    for t in range(T):
        state = torch.matmul(transformation_matrix, state)
        rewards[0, 0] += torch.matmul(agent1_rewards.T, state).flatten()[0]/T
        rewards[0, 1] += torch.matmul(agent2_rewards.T, state).flatten()[0]/T

    return rewards


if __name__ == "__main__":
    fixed_actions1 = np.array([
        [10., 0.],
        [10., 0.],
        [0., 10.],
        [0., 10.]
    ])
    agent1 = CalvanoDiscreteADAgent(
        0.1, 2, 'cpu', fixed_actions=fixed_actions1)
    fixed_actions2 = np.array([
        [10., 0.],
        [0., 10.],
        [10., 0.],
        [0., 10.]
    ])
    # agent2 = DummyAgent(1, 2)
    agent2 = CalvanoDiscreteGEAgent(
        0.1, 2, 'cpu', no_models=5, fixed_actions=fixed_actions2)
    agent2.regenerate_models()

    possible_actions = [0., 1.]
    env = CalvanoDiscreteTorch(np.array([1., 1.]), 0.3, 1., np.array([
        0., 0.]), possible_actions, 'cpu')

    # rewards = MDP(agent1, agent2, env, 5, 'cpu', None)
    # print(rewards)
    # agent1.zero_gradients()
    # rewards[0, 1].backward(retain_graph=True)

    print(agent2.parameters)
    print(agent2.get_probabilities(0))

    for i in range(4):
        state = torch.zeros(size=(1,), dtype=torch.int32) + i
        for j in range(3):
            print(agent2.get_action(state, 0, flip=True), end=' ')
        print()
