import random
from collections import namedtuple

Transition = namedtuple('Transition',   # 默认经验元组格式
                        ('state', 'action', 'next_state', 'reward', 'done'))


# 经验回放缓存池
class ReplayBuffer(object):
    def __init__(self, capacity, transition=Transition):    # 可自行经验创建元组格式
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = transition

    # 存入经验
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    # 随机经验采样
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def claer(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)
