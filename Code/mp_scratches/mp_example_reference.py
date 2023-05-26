from multiprocessing import Process
from multiprocessing import Queue
from time import sleep, time
import numpy as np
from gsde import CalvanoTorch, CalvanoAgent, CalvanoGradientAgent, MDP, DummyAgent
from copy import deepcopy
import os


class Subclass(object):
    def __init__(self) -> None:
        self.a = 2


class TestClass(object):

    def __init__(self) -> None:
        self.a = 2
        self.b = Subclass()


def process_func(qin, qout):
    print('Started process', os.getpid(), flush=True)
    device = 'cpu'
    obj = qin.get()[0]
    print('Process ', os.getpid(), obj)
    obj.model = os.getpid()

    return


if __name__ == "__main__":
    num_processes = 4
    processes = []
    in_queues = []
    out_queues = []
    for i in range(num_processes):
        qin = Queue()
        in_queues.append(qin)
        qout = Queue()
        out_queues.append(qout)
        process = Process(target=process_func, args=(qin, qout))
        processes.append(process)
        process.start()
    obj = CalvanoGradientAgent(0.1, 0.99, 'cpu')
    for i in range(num_processes):
        in_queues[i].put((obj,))
    for i in range(num_processes):
        processes[i].join()
    print(obj)
    print(obj.model)
