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
STEPSIZE = 1                           # n的大小

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

class N_Step_DQN(object):
    def __init__(self):
        # 传统DQN使用当前的即时奖励和下一时刻的价值估计来判断目标的价值
        # 然而这种方法在训练的前期网络参数偏差较大时会导致得到的目标价值也会偏大
        # 进而导致目标价值的估计偏差较大，因此出现了多步学习来解决这个问题
        # 在多步学习中，即时奖励会通过与环境交互确切得到，所以训练前期的目标价值可以得到更准确的估计
        # 从而加快训练的速度
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # 当learn_step_counter=MEMORY_CAPACITY，
                                                                        # 会开始学习经验

        self.memory_counter = 0                                         # 经验池记忆计数器

        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS))     # 初始化经验池

        # 选取DQN的梯度下降方法（Adam）
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        # 选取损失函数
        self.loss_func = nn.MSELoss()

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

        q_eval = self.eval_net(b_s).gather(1, b_a)

        q_next = self.target_net(b_s_).detach()

        q_target = b_r + pow(GAMMA, STEPSIZE) * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

n_step_dqn = N_Step_DQN()
Episode=[]
Score=[]

print('\nCollecting experience...')
for i_episode in range(1000):
    s = env.reset()
    a = n_step_dqn.choose_action(s)
    s_old = [s for i in range(STEPSIZE)]
    a_old = [a for i in range(STEPSIZE)]
    score = 0
    Step_Count = 0
    Reward_Store = [0 for i in range(STEPSIZE)]
    while True:
        env.render()
        s_, r, done, info = env.step(a)

        x, x_dot, theta, theta_dot = s_

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        score += r

        a = n_step_dqn.choose_action(s_)

        if Step_Count < STEPSIZE - 1:
            Reward_Store[Step_Count] = r
            s_old[Step_Count + 1] = s_
            a_old[Step_Count + 1] = a

        else:
            # n步学习如何储存经验
            Reward_Store[-1] = r
            r_count = 0
            for i in range(STEPSIZE):
                r_count += pow(GAMMA ,i) * Reward_Store[STEPSIZE - 1 - i]

            n_step_dqn.store_transition(s_old[0], a_old[0], r_count, s_)

            for i in range(STEPSIZE - 1):
                Reward_Store[i] = Reward_Store[i + 1]
                s_old[i] = s_old[i + 1]
                a_old[i] = a_old[i + 1]

            a_old[-1] = a
            s_old[-1] = s_

        if n_step_dqn.memory_counter > MEMORY_CAPACITY:
            n_step_dqn.learn()

        if done:
            break

        Step_Count += 1

        s = s_

    print(i_episode)
    print(n_step_dqn.memory_counter)

    Episode.append(i_episode)
    Score.append(score)

# 绘制幕数-每幕得分图
plt.plot(Episode, Score)
plt.show()