import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib
import matplotlib.pyplot as plt

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 500

# 这里使用的测试环境是平衡摆
env = gym.make('CartPole-v0')

env = env.unwrapped

# N_ACTIONS=2,分别表示向左或向右
N_ACTIONS = env.action_space.n

# N_STATES=4，表示当前的状态空间有4个维度
N_STATES = env.observation_space.shape[0]

# ENV_A_SHAPE用来表示动作的维度，这里由于只有2个动作，维度为1
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 50)

        self.fc1.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# 如何有效地根据优先级p来进行抽样呢？
# 如果每次抽样的时候都对经验池进行一次排序，这样会浪费大量的计算资源
# SumTree是一种树形结构，叶子结点储存每个样本的优先级p
class SumTree(object):
    # 指向当前应该的储存位置
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity

        # self.tree储存优先级
        self.tree = np.zeros(2 * capacity - 1)
        # 父节点共有capacity - 1个，用于储存样本的叶子结点共有capacity个

        #self.data储存经验
        self.data = np.zeros((capacity, N_STATES * 2 + N_ACTIONS))

    # 在SumTree中加入新的样本
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        # 经验池满了之后，用新的经验替代旧的经验
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    # 由下到上，更新节点优先级
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # 非递归的方式效率更高
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # 给定一个数v，取其对应的经验data
    # 这里思考一下，为什么这种储存优先级的方式，能让优先级更高的经验被取到的概率更大？
    # 我们可以将每个叶子结点的pi看成一个区间，pi的大小代表了区间长度
    # 我们在随机取v的时候，自然v落在大区间的概率更高
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # self.tree储存优先级
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        # 返回值为：储存优先级节点的序号，优先级和经验
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]

# 经验值类型
class Memory(object):
    # 避免其经验样本被选择的概率为0（见后）
    epsilon = 0.01

    # 用于将TD误差代表的重要性转化为概率，取值范围为[0~1]
    alpha = 0.6

    # beta用于计算权重，使用beta_increment_per_sampling不断增长，最大为1
    beta = 0.4
    beta_increment_per_sampling = 0.001

    # 初始的默认最高优先级
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    # 这里为什么要对每一个新插入的经验都赋予最高的优先级？
    # 当然了，刚刚学到的东西肯定是要用的，这样就不会出现反复学习某几个重复经验的情况
    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        # 如果现有的优先级都是0，赋予一个初始的默认最高优先级
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    # 从经验池中取n个样本，返回它们的优先级所在节点索引，经验数据以及权重
    def sample(self, n):
        b_idx, b_memory, ISWeights = \
            np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))

        # 将SumTree的叶子结点按照n分为不同的区间
        pri_seg = self.tree.total_p / n

        # beta不断变大，最大为1
        # 为什么需要不断变大？
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # 计算最小概率
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        if min_prob == 0:
            min_prob = 0.00001

        # 具体的采样过程
        for i in range(n):
            # 从n个区间中，均匀采样一个值v
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)

            # p为TD误差（优先级）
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            # 根据公式计算权重
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        # 在TD误差上加入一个ε，用来避免其经验样本被选择的概率为0
        abs_errors += self.epsilon

        # 不能超过默认的最高优先级
        clipped_errors = np.minimum(abs_errors.detach().numpy(), self.abs_err_upper)

        # 计算最终的p优先级
        ps = np.power(clipped_errors, self.alpha)

        # 更新节点权重
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

# 基于优先级的经验复用池
# 在传统的DQN的经验复用池中，选择batch的数据这个过程是随机的
# 但是，其实不同样本的优先级是不同的，需要给每一个样本一个优先级，并根据样本的优先级进行采样
class PRDQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # 当learn_step_counter=MEMORY_CAPACITY，
                                                                        # 会开始学习经验

        self.memory_counter = 0                                         # 经验池记忆计数器

        self.memory = Memory(capacity = MEMORY_CAPACITY)                # 初始化经验池

        # 选取DQN的梯度下降方法（Adam）
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)

        self.memory_counter += 1

    # 根据权重，定义新的损失函数
    def loss(self, y_output, y_true, ISWeights):
        value = y_output - y_true
        return torch.mean(value * value * ISWeights)

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 新的采样方法
        tree_idx, b_memory, ISWeights = self.memory.sample(BATCH_SIZE)

        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + N_ACTIONS -1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS -1:N_STATES + N_ACTIONS])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss(q_eval, q_target, torch.tensor(ISWeights))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 计算TD误差
        y = self.eval_net(b_s).gather(1, b_a)
        abs_errors = torch.abs(q_target - y)
        
        self.memory.batch_update(tree_idx, abs_errors)

prdqn = PRDQN()
Episode=[]
Score=[]

print('\nCollecting experience...')
for i_episode in range(80):
    s = env.reset()
    score = 0
    while True:
        env.render()
        a = prdqn.choose_action(s)

        s_, r, done, info = env.step(a)

        x, x_dot, theta, theta_dot = s_

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        score += r

        prdqn.store_transition(s, a, r, s_)

        if prdqn.memory_counter > MEMORY_CAPACITY:
            prdqn.learn()

        if done:
            break
        s = s_

    print(i_episode)
    print(prdqn.memory_counter)

    Episode.append(i_episode)
    Score.append(score)

# 绘制幕数-每幕得分图
plt.plot(Episode, Score)
plt.show()
