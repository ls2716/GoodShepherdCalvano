from multiprocessing import Process
from multiprocessing import Queue
from time import sleep, time
import numpy as np
from gsde import CalvanoTorch, CalvanoAgent, CalvanoGradientAgent, MDP, DummyAgent
from copy import deepcopy
import os


def process_func(qin, qout):

    print('Started process', os.getpid(), flush=True)
    device = 'cpu'
    T = 5
    batch_size = 20
    learning_steps = 300
    env = CalvanoTorch(np.array([1., 1.]), 0.3, 1., np.array([0., 0.]), 'cpu')

    qout.put(0)

    while True:
        dummy_price = qin.get()
        # print('Process', os.getpid(), 'Got dummy price')

        agent2 = DummyAgent(dummy_price, device)
        agent1 = CalvanoGradientAgent(
            0.01, 0.99, device)
        # Run learning
        # print('Process', os.getpid(), 'Learning')
        for i in range(learning_steps):
            # print(f'Step {i}', end='\r')
            agent1.zero_gradients()
            running_rewards, actions = MDP(
                agent1, agent2, env, batch_size, T, device)
            running_rewards[0, 0].backward()
            agent1.update()
        # print('Process', os.getpid(), 'Finished')
        qout.put((dummy_price, running_rewards.detach().numpy(),
                 actions.detach().numpy()))

    return


if __name__ == "__main__":
    num_processes = 4
    processes = []
    in_queues = []
    out_queues = []
    env = CalvanoTorch(np.array([1., 1.]), 1., 1., np.array([1., 1.]), 'cpu')
    for i in range(num_processes):
        qin = Queue()
        in_queues.append(qin)
        qout = Queue()
        out_queues.append(qout)
        process = Process(target=process_func, args=(qin, qout))
        processes.append(process)
        process.start()

    dummy_prices = [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5]
    for i in range(num_processes):
        signal = out_queues[i].get()
        print('Process', i, signal)

    time_start = time()
    for i, dummy_price in enumerate(dummy_prices):
        in_queues[i % num_processes].put(dummy_price)
    for i, dummy_price in enumerate(dummy_prices):
        # print('waiting for output')
        rewards_actions = out_queues[i % num_processes].get()
        print(f'Process {i % num_processes}, {rewards_actions}')
    time_end = time()
    print(f'{num_processes} processes - elapsed time = {time_end-time_start}')
    for i in range(num_processes):
        processes[i].kill()
