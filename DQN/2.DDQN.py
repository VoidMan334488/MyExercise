import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
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

class DDQN(object):
    def __init__(self):
        # DDQN的整体结构中，包含2个网络
        # 为什么要设置两个网络？为了解决最大化偏差的问题
        # 以单独的Q学习为例，我们的更新公式为Q(S,A) = Q(S,A) + α[R + γ * max(A') Q(S',A') - Q(S,A)]
        # 如果Q(S',A')是一个overestimate，那就会会出现最大化偏差
        # 我们使用新的更新公式：(按概率来选择更新方式，Q1，Q2角色互换)
        # Q1(S,A) = Q1(S,A) + α{R + γ * Q2[S', argmax(A') Q1(S',A')] - Q1(S,A)}
        # Q2(S,A) = Q2(S,A) + α{R + γ * Q1[S', argmax(A') Q2(S',A')] - Q2(S,A)}
        # 这样，在Q1中可能会出现的最大化偏差，在Q2中也出现的概率就会很低
        # 如此便减小了最大化偏差的影响
        self.Q1_net, self.Q2_net = Net(), Net()

        self.learn_step_counter = 0                                     # 当learn_step_counter=MEMORY_CAPACITY，
                                                                        # 会开始学习经验

        self.memory_counter = 0                                         # 经验池记忆计数器

        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS))     # 初始化经验池

        # 选取DDQN的梯度下降方法（Adam）
        self.optimizer1 = torch.optim.Adam(self.Q1_net.parameters(), lr=LR)
        self.optimizer2 = torch.optim.Adam(self.Q2_net.parameters(), lr=LR)

        # 选取损失函数
        self.loss_func = nn.MSELoss()

    # 根据Q1+Q2（ε-greedy），选取动作
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        if np.random.uniform() < EPSILON:
            actions_value = self.Q1_net.forward(x)+self.Q2_net.forward(x)
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
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + N_ACTIONS - 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS - 1:N_STATES + N_ACTIONS])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # 0.5的概率，使用Q1(S,A) = Q1(S,A) + α{R + γ * Q2[S', argmax(A') Q1(S',A')] - Q1(S,A)}更新Q1
        if np.random.uniform() < 0.5:
            Q2_value=self.Q2_net(b_s_).gather(1, torch.max(self.Q1_net(b_s_), 1)[1].unsqueeze(1)).view(BATCH_SIZE, 1)

            loss = self.loss_func(self.Q1_net(b_s).gather(1, b_a), b_r + GAMMA * Q2_value)

            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()

        # 0.5的概率，使用Q2(S,A) = Q2(S,A) + α{R + γ * Q1[S', argmax(A') Q2(S',A')] - Q2(S,A)}更新Q2
        else:
            Q1_value = self.Q1_net(b_s_).gather(1, torch.max(self.Q2_net(b_s_), 1)[1].unsqueeze(1)).view(BATCH_SIZE, 1)

            loss = self.loss_func(self.Q2_net(b_s).gather(1, b_a), b_r + GAMMA * Q1_value)

            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step()

ddqn = DDQN()
Episode=[]
Score=[]

print('\nCollecting experience...')
for i_episode in range(80):
    s = env.reset()
    score = 0
    while True:
        env.render()
        a = ddqn.choose_action(s)

        s_, r, done, info = env.step(a)

        x, x_dot, theta, theta_dot = s_

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        score += r

        ddqn.store_transition(s, a, r, s_)

        if ddqn.memory_counter > MEMORY_CAPACITY:
            ddqn.learn()

        if done:
            break
        s = s_

    print(i_episode)
    print(ddqn.memory_counter)

    Episode.append(i_episode)
    Score.append(score)

# 绘制幕数-每幕得分图
plt.plot(Episode, Score)
plt.show()