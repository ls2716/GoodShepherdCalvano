"""Implement Good Shepherd IO algorithm
with the Calvano environment and gradient estimation
with discrete actions and using multiprocessing.
"""
import global_config

from gsdis import CalvanoDiscreteGEAgent, CalvanoDiscreteADAgent, CalvanoDiscreteTorch, DummyAgent, MDP
import matplotlib.pyplot as plt

import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os
from copy import deepcopy
import time
from datetime import datetime
import psutil

import logging
from logging.handlers import QueueHandler, QueueListener

# Set sharing strategy for torch
torch.multiprocessing.set_sharing_strategy('file_system')


def process_func(qin, qout, qlog):
    device = 'cpu'
    env = deepcopy(qin.get())
    outer_agent = deepcopy(qin.get())
    arguments = deepcopy(qin.get())
    core = deepcopy(qin.get())

    pid = os.getpid()
    p = psutil.Process(pid)
    p.cpu_affinity([core])

    inner_learning_steps = arguments["inner_learning_steps"]
    T = arguments["T"]
    no_actions = outer_agent.no_actions
    logging_level = arguments.get("logging_level", logging.INFO)

    qh = QueueHandler(qlog)
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    logger.addHandler(qh)

    logging.info(f'Started process - setup PID {os.getpid()}')
    logging.info(f'Arguments {arguments}')
    logging.debug(f"Core affinity set to {core}")

    logging.debug(f'Finished setup')
    qout.put(0)

    while True:
        new_parameters = qin.get()
        if new_parameters is None:
            return
        logging.debug('Got new parameters')
        outer_agent.models[0][:] = new_parameters[:]

        outer_reward, inner_reward = inner_loop(
            outer_agent=outer_agent, model_index=0,
            inner_learning_steps=inner_learning_steps,
            env=env, device=device, T=T, no_actions=no_actions)
        logging.debug('Finished')

        logging.debug(f'Outer reward {outer_reward}')

        qout.put((outer_reward, inner_reward))
    return


def inner_loop(outer_agent, model_index, inner_learning_steps, env, device, T, no_actions, save_model=False):
    """Define reward of inner loop"""
    outer_running_reward = torch.Tensor([0])
    inner_agent = CalvanoDiscreteADAgent(
        lr=10., no_actions=no_actions, device=device)
    for inner_it in range(inner_learning_steps):
        inner_agent.zero_gradients()
        # Get rewards
        running_rewards = MDP(agent1=outer_agent,
                              agent2=inner_agent,
                              env=env, T=T, device=device,
                              model_index=model_index)
        inner_running_reward = torch.Tensor([0]) + running_rewards[0, 1]
        inner_running_reward.backward(retain_graph=False)
        # outer_running_reward += running_rewards[0, 0]/inner_learning_steps
        # Update inner agent
        inner_agent.update()
    outer_running_reward += running_rewards[0, 0]
    outer_reward = outer_running_reward.detach().cpu().flatten()[0]
    inner_reward = inner_running_reward.detach().cpu().flatten()[0]
    if save_model:
        inner_model_path = 'models/inner.pth'
        torch.save(inner_agent.parameters, inner_model_path)
    return outer_reward, inner_reward


def logger_init(level=logging.INFO):
    q = mp.Queue()
    # this is the handler for all log records
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

    if not os.path.exists('logs'):
        os.mkdir('logs')
    # For timestamping
    # timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    # filename = f'logs/gsde_io_mp_{timestamp}.log'
    filename = 'logs/gsde_io_mp.log'
    fh = logging.FileHandler(filename, mode='w+')
    fh.setFormatter(logging.Formatter(
        "%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

    # ql gets records from the queue and sends them to the handler
    ql = QueueListener(q, fh)
    ql.start()

    logger = logging.getLogger()
    logger.setLevel(level)
    # Add the handler to the logger so records from this process are handled
    logger.addHandler(handler)
    logger.addHandler(fh)

    return ql, q


if __name__ == "__main__":
    # Setup spawn method
    mp.set_start_method('spawn')
    # Setup logger
    root_logging_level = logging.DEBUG
    process_logging_level = logging.INFO

    q_listener, qlog = logger_init(
        root_logging_level)
    logging.info('Initialized main thread')

    # Parameter setup

    # Setup device
    device = torch.device('cpu')
    logging.info(f'Device is {device}')

    # MP parameters
    num_processes = 8
    # Cores for processes
    cores = list(range(num_processes))
    # Add check for cores

    # Environment setup

    # Defining actions
    # possible_actions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    possible_actions = [0.5, 0.75, 1.]
    no_actions = len(possible_actions)

    # Market parameters
    # Quality indices
    A = np.array([1., 1.])
    # Substitution index
    mu = 0.2
    # Inverse aggregate demand
    a_0 = 1.
    # Costs
    C = np.array([0., 0.])

    # Game setup
    # Defining number of steps of environment
    T = 5
    # Learning length parameters
    inner_learning_steps = 200
    outer_learning_steps = 1

    # Number of models for GE agent
    ge_no_models = 24
    # Define scale for perturbation
    ge_scale = 0.03

    # Define method for learning ('gradient' or 'best')
    ge_learning_method = 'best'
    # Define learning rate if gradient (set to any value otherwise)
    ge_lr = 0.2
    # Define path for saving the outer ge model
    outer_model_path = 'models/outer_3_actions.pth'
    # Define whether to load the model
    load_model = False

    # Initialize outer agent
    outer_agent = CalvanoDiscreteGEAgent(
        lr=0.2, no_actions=no_actions,
        no_models=ge_no_models, device=device)

    # (optional) Load the model
    if load_model:
        try:
            logging.info("Loading model")
            outer_agent.parameters = torch.load(outer_model_path)
        except FileNotFoundError:
            logging.warning("Model weights not found and not loaded")
            logging.debug("Trying to create models directory")
            if not os.path.exists('models'):
                os.mkdir('models')
                logging.debug("Success")

    # Define environment
    env = CalvanoDiscreteTorch(A=A, mu=mu, a_0=a_0,
                               C=C, actions=possible_actions, device=device)

    arguments = {
        "inner_learning_steps": inner_learning_steps,
        "T": T,
        "logging_level": process_logging_level
    }

    # Initialize processes
    processes = []
    in_queues = []
    out_queues = []
    for i in range(num_processes):
        qin = mp.Queue()
        in_queues.append(qin)
        qout = mp.Queue()
        out_queues.append(qout)
        process = mp.Process(target=process_func, args=(qin, qout, qlog))
        processes.append(process)
        process.start()

    for i in range(num_processes):
        in_queues[i].put(env)
        in_queues[i].put(outer_agent)
        in_queues[i].put(arguments)
        in_queues[i].put(i)

    for i in range(num_processes):
        signal = out_queues[i].get()
        logging.debug(f'Process {i} signal {signal}')

    logging.info("Running outer learning")
    # Outer loop
    start_time = time.time()
    for outer_it in tqdm(range(outer_learning_steps)):
        outer_agent.regenerate_models()
        outer_reward, inner_reward = inner_loop(
            outer_agent, model_index=0,
            inner_learning_steps=inner_learning_steps,
            env=env, T=T, no_actions=no_actions, device=device)
        base_reward = outer_reward
        logging.info(
            f'\n Outer reward is {base_reward}, inner reward is {inner_reward}')

        outer_agent.generate_perturbation(ge_scale)
        outer_agent.perturb_models()
        rewards = []
        params = []

        # Sending for calculation
        for model_index in range(outer_agent.no_models):
            process_index = model_index % num_processes
            in_queues[process_index].put(outer_agent.models[model_index])

        # Receiving frewards
        for model_index in range(outer_agent.no_models):
            process_index = model_index % num_processes
            outer_reward, inner_reward = out_queues[process_index].get()
            logging.debug(
                f'Model index {model_index}, outer reward {outer_reward}')
            rewards.append(outer_reward)
        rewards = torch.tensor(rewards)-base_reward

        if ge_learning_method == 'best':
            best_index = np.argmax(rewards)
            logging.debug(f'Best model is {best_index}')
            outer_agent.get_best(rewards)
        else:
            outer_agent.calculate_gradients(rewards)
            outer_agent.update()
    end_time = time.time()

    # Stopping processes
    logging.info("Stopping processes")
    for i in range(num_processes):
        parameters = None
        in_queues[i].put(parameters)

    for i in range(num_processes):
        processes[i].join()

    outer_agent.regenerate_models()
    outer_reward, inner_reward = inner_loop(
        outer_agent, model_index=0,
        inner_learning_steps=inner_learning_steps,
        env=env, T=T, no_actions=no_actions, device=device, save_model=True)
    # Save outer model
    torch.save(outer_agent.parameters, outer_model_path)

    # Printing strategy
    logging.info('Outer player strategy')
    logging.info(f'\n {outer_agent.compute_probabilities(model_index=0)}')

    elapsed_time = end_time-start_time
    logging.info(f"Time elapsed {elapsed_time:.4} s")

    q_listener.stop()
