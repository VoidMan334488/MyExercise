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
ATOM_SIZE = 10                        # 分布中的采样次数（也就是分布分为几块表示）
VMIN = 0                              # 分布区间的最大价值（上界）
VMAX = 20                             # 分布区间的最小价值（下界）

# 这里使用的测试环境是平衡摆
env = gym.make('CartPole-v0')

env = env.unwrapped

# N_ACTIONS=2,分别表示向左或向右
N_ACTIONS = env.action_space.n

# N_STATES=4，表示当前的状态空间有4个维度
N_STATES = env.observation_space.shape[0]

# ENV_A_SHAPE用来表示动作的维度，这里由于只有2个动作，维度为1
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

# 分布式DQN的网络结构与DQN是有区别的
# 传统的DQN输入(s,a)，输出的是状态-动作对的价值
# 而分布式DQN输入(s,a)，输出的是状态-动作对的各个价值以多少的概率出现（价值分布）
# 相比较于分布式DQN，传统的DQN等于只是取了价值期望，这样会遗漏很多的信息
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        # 输入维度为状态维度
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.normal_(0, 0.1)

        # 输出维度为动作维度 * 区间分块（采样）的数量
        # 也就是输出的是 N_ACTIONS 个采样次数为 ATOM_SIZE 的价值分布
        self.out = nn.Linear(128, N_ACTIONS * ATOM_SIZE)
        self.out.weight.data.normal_(0, 0.1)

        # 这里的support向量有什么作用?
        # 其实就是价值权重，这里计算一下
        support=[0. for i in range(ATOM_SIZE)]
        IncreaMent = (VMAX - VMIN) / (ATOM_SIZE - 1)
        for i in range(1, ATOM_SIZE):
            support[i] = support[i - 1] + IncreaMent
        self.support=torch.FloatTensor(support)

    # 得到分布
    # 这一步的操作意义在于，将各个分布通过动作维度划分到不同的元素内
    def dist(self, x):
        q_atoms = self.fc1(x)
        q_atoms = F.relu(q_atoms)
        q_atoms = self.out(q_atoms)

        q_atoms = q_atoms.view(-1, N_ACTIONS, ATOM_SIZE)
        # softmax保证确实为一个标准化后的概率分布
        dist = F.softmax(q_atoms, dim=-1)
        # 防止出现NAN
        dist = dist.clamp(min=1e-3)
        return dist

    # 在这里，出现了新的知识
    # 为什么我还要定义Forward函数？明明我只需要用计算出的分布去计算损失函数即可
    # 原因就是我需要某个计算方法去选择贪心动作（笑），这里用价值权重 * 对应概率
    # 还有，我们定义的网络都是继承了nn.Module类，其中Net()的使用默认是调用Forward方法的
    # 也就是说，net() == net.foward()
    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

class Dist_DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # 当learn_step_counter=MEMORY_CAPACITY，
                                                                        # 会开始学习经验

        self.memory_counter = 0                                         # 经验池记忆计数器

        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS))     # 初始化经验池

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

        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + N_ACTIONS - 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS - 1:N_STATES + N_ACTIONS])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 代码的重点部分！
        # 原分布
        dist = self.eval_net.dist(b_s)

        # 计算更新后的分布
        # next_action为按照最大收益选取的下一个动作
        next_action = self.target_net(b_s_).argmax(1)

        # next_dist为b_s_的最大价值动作对应的价值分布
        next_dist = self.target_net.dist(b_s_)
        next_dist = next_dist[range(BATCH_SIZE), next_action]

        # 第一步，我们需要将两个分布对齐
        # 对应价值位置加上回报
        t_z = b_r + GAMMA * self.eval_net.support
        # 裁减掉超出范围的部分
        t_z = t_z.clamp(min=VMIN, max=VMAX)

        # 计算这个是为了找到相邻的两个采样点的位置
        b = (t_z - VMIN) / self.eval_net.support[1]
        l = b.floor().long()
        u = b.ceil().long()

        # offset的作用是什么？
        # offset用于计算高维数组中的索引偏移量
        offset = (
            torch.linspace(
                0, (BATCH_SIZE - 1) * ATOM_SIZE, BATCH_SIZE
            ).long()
                .unsqueeze(1)
                .expand(BATCH_SIZE, ATOM_SIZE)
        )
        # 初始化新的分布各个位置概率皆为0
        proj_dist = torch.zeros(next_dist.size())
        # 按照权重（该点距离上界采样点的距离）来给下界采样点分去概率
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        # 按照权重（该点距离下界采样点的距离）来给上界采样点分去概率
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

        # 第二步，计算两个分布间的距离（损失函数）
        # 损失函数为更新网络前后分布间KL散度的大小
        log_p = torch.log(proj_dist)
        log_q = torch.log(dist[range(BATCH_SIZE), b_a.view(-1)])

        # 好像把这一句注释掉，效果也还行？？？
        # log_p = torch.zeros(log_p.size())
        loss = (proj_dist * (log_p - log_q)).sum(1).mean()

        # 梯度下降更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dist_dqn = Dist_DQN()
Episode=[]
Score=[]

print('\nCollecting experience...')
for i_episode in range(200):
    s = env.reset()
    score = 0
    while True:
        env.render()
        a = dist_dqn.choose_action(s)

        s_, r, done, info = env.step(a)

        x, x_dot, theta, theta_dot = s_

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        score += r

        dist_dqn.store_transition(s, a, r, s_)

        if dist_dqn.memory_counter > MEMORY_CAPACITY:
            dist_dqn.learn()

        if done:
            break
        s = s_

    print(i_episode)
    print(dist_dqn.memory_counter)

    Episode.append(i_episode)
    Score.append(score)

# 绘制幕数-每幕得分图
plt.plot(Episode, Score)
plt.show()

# 写在后面的一些笔记：
# ATOM_SIZE，VMIN和VMAX都是超参数，设置需要一些先验知识，不然效果会很差
# 上述代码为C51的实现代码，使用的误差为KL散度
# 但是，怎么定义两个分布之间的距离，一般会使用Wasserstein Metric（瓦瑟斯坦度量距离）
# 也叫推土机距离(Earth Mover’s distance)，它的意思是, 将一个分布转变为另一个分布，所需要移动的最少的“土"（采样块）的量
# 这样可以保证我们的可以收敛到唯一的不动点（QR-DQN）
# 此外，还有IQN的方法，IQN更像是直接的学出了这个分布