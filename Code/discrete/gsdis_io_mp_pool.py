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
import yaml

import multiprocessing as mp
import os
import sys
from copy import deepcopy
import time
from datetime import datetime
import psutil

import logging
from logging.handlers import QueueHandler, QueueListener


case_name = '3_actions_calvano'


def worker_init(qlog, env, outer_agent, arguments, cores):\

    logging_level = arguments.get("logging_level", logging.INFO)

    global worker_env
    global worker_outer_agent
    global inner_learning_steps
    global T
    global inner_lr
    global case_name
    worker_env = deepcopy(env)
    worker_outer_agent = deepcopy(outer_agent)
    inner_learning_steps = arguments["inner_learning_steps"]
    T = arguments["T"]
    inner_lr = arguments['inner_lr']
    case_name = arguments['case_name']
    torch.manual_seed(arguments['seed'])

    pid = os.getpid()
    psutil_process = psutil.Process(pid)
    mp_process_index = mp.Process()._identity[0]-1

    psutil_process.cpu_affinity([cores[mp_process_index]])

    qh = QueueHandler(qlog)
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    logger.addHandler(qh)

    logging.info(f'Started process - setup PID {os.getpid()}')
    logging.info(f'Arguments {arguments}')
    logging.debug(f"Core affinity set to {cores[mp_process_index]}")

    logging.debug(f'Finished setup')


def dummy_task():
    logging.debug('Dummy task')


def func(new_parameters):
    global worker_env
    global worker_outer_agent
    global inner_learning_steps
    global inner_lr
    global case_name
    global T
    device = 'cpu'
    no_actions = worker_outer_agent.no_actions

    logging.debug('Got new parameters')
    worker_outer_agent.models[0][:] = new_parameters[:]

    outer_reward, inner_reward = inner_loop(
        outer_agent=worker_outer_agent, model_index=0,
        inner_learning_steps=inner_learning_steps, inner_lr=inner_lr, case_name=case_name,
        env=worker_env, device=device, T=T, no_actions=no_actions)
    logging.debug('Finished')

    logging.debug(f'Outer reward {outer_reward}')

    return (outer_reward, inner_reward)


def inner_loop(outer_agent, model_index, inner_learning_steps, inner_lr, env, device, T, no_actions, case_name, save_model=False):
    """Define reward of inner loop"""
    outer_running_reward = torch.Tensor([0])
    inner_agent = CalvanoDiscreteADAgent(
        lr=inner_lr, no_actions=no_actions, device=device)
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
        inner_model_path = f'models/inner_{case_name}.pth'
        torch.save(inner_agent.parameters, inner_model_path)
    return outer_reward, inner_reward


def logger_init(level=logging.INFO, case_name=''):
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
    filename = f'logs/gsde_io_mp_pool_{case_name}.log'
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
    if len(sys.argv) != 2:
        raise ValueError("case name argument was not supplied")
    case_name = sys.argv[1]

    # Setup spawn method
    mp.set_start_method('spawn')
    # Setup logger
    root_logging_level = logging.DEBUG
    process_logging_level = logging.INFO

    q_listener, qlog = logger_init(
        root_logging_level, case_name=case_name)
    logging.info('Initialized main thread')
    logging.info(f'Case name {case_name}')

    # Parameter setup
    # Read configuration
    with open(f'run_cases/case_{case_name}.yml', 'r') as file:
        case_args = yaml.safe_load(file)
    logging.info(f'Configuration \n {case_args}')

    # Setup device
    device = torch.device('cpu')
    logging.info(f'Device is {device}')

    torch.manual_seed(case_args['seed'])

    # MP parameters
    num_processes = case_args['num_processes']
    # Cores for processes
    cores = list(range(num_processes))
    # Add check for cores

    # Environment setup

    # Defining actions
    possible_actions = case_args['possible_actions']
    no_actions = len(possible_actions)

    # Market parameters
    # Quality indices
    A = np.array([2., 2.])
    # Substitution index
    mu = 0.25
    # Inverse aggregate demand
    a_0 = 0.
    # Costs
    C = np.array([1., 1.])

    # Game setup
    # Defining number of steps of environment
    T = case_args['T']
    # Learning length parameters
    inner_learning_steps = case_args['inner_learning_steps']
    inner_lr = case_args['inner_lr']

    outer_learning_steps = case_args['outer_learning_steps']
    norm_scale = case_args['norm_scale']

    # Number of models for GE agent
    ge_no_models = case_args['ge_no_models']
    # Define scale for perturbation
    ge_scale = case_args['ge_scale']

    # Define method for learning ('gradient' or 'best')
    ge_learning_method = 'best'
    # Define learning rate if gradient (set to any value otherwise)
    ge_lr = None
    # Define path for saving the outer ge model
    outer_model_path = f'models/outer_{case_name}.pth'
    # Define whether to load the model
    load_model = False

    # Initialize outer agent
    outer_agent = CalvanoDiscreteGEAgent(
        lr=ge_lr, no_actions=no_actions,
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
        'inner_lr': inner_lr,
        "T": T,
        "logging_level": process_logging_level,
        'seed': case_args['seed'],
        'case_name': case_name
    }

    with mp.Pool(num_processes, initializer=worker_init,
                 initargs=[qlog, env, outer_agent, arguments, cores]) as pool:

        # Startup the processes
        params = list(range(num_processes))
        pool.apply(dummy_task)

        logging.info("Running outer learning")
        # Outer loop
        start_time = time.time()
        for outer_it in tqdm(range(outer_learning_steps)):
            outer_agent.regenerate_models()
            outer_reward, inner_reward = inner_loop(
                outer_agent, model_index=0,
                inner_learning_steps=inner_learning_steps, inner_lr=inner_lr,
                env=env, T=T, no_actions=no_actions, case_name=case_name, device=device)
            base_reward = outer_reward
            logging.info(
                f'\n Outer reward is {base_reward}, inner reward is {inner_reward}')

            outer_agent.generate_perturbation(ge_scale)
            outer_agent.perturb_models()
            rewards = []

            parameters = [outer_agent.models[index]
                          for index in range(outer_agent.no_models)]

            results = pool.map(func, parameters)

            rewards = [item[0] for item in results]
            for model_index in range(outer_agent.no_models):
                logging.debug(
                    f'Model index {model_index}, outer reward {rewards[model_index]}')

            rewards = torch.tensor(rewards)-base_reward
            if ge_learning_method == 'best':
                best_index = np.argmax(rewards)
                logging.debug(f'Best model is {best_index}')
                outer_agent.get_best(rewards)
            else:
                outer_agent.calculate_gradients(rewards)
                outer_agent.update()
            outer_agent.normalize_parameters(norm_scale=1.)

        end_time = time.time()

        # Stopping processes
        logging.info("Stopping processes")
        pool.close()
        pool.join()

    outer_agent.regenerate_models()
    outer_reward, inner_reward = inner_loop(
        outer_agent, model_index=0,
        inner_learning_steps=inner_learning_steps, inner_lr=inner_lr,
        env=env, T=T, no_actions=no_actions, device=device, case_name=case_name, save_model=True)
    # Save outer model
    torch.save(outer_agent.parameters, outer_model_path)

    # Printing parameters
    logging.info('Outer player parameters')
    logging.info(f'\n {outer_agent.parameters}')
    # Printing strategy
    logging.info('Outer player strategy')
    logging.info(f'\n {outer_agent.compute_probabilities(model_index=0)}')

    elapsed_time = end_time-start_time
    logging.info(f"Time elapsed {elapsed_time:.4} s")

    q_listener.stop()
