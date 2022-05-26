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

        # 与普通的DQN相比，Dueling DQN唯一的区别就在于网络结构的不同
        # Dueling DQN将全连接层的输出分为两条流
        # 其中一条输出关于状态的价值V(S)，另一条输出关于动作的优势函数A(S,a)，最终合并为Q价值函数
        # 在一般的游戏场景中，经常会存在一些状态，在这些状态下采取动作对下一步状态转变没什么影响
        # 这时候，计算V(S)就比计算Q(S,a)更有意义
        self.outV = nn.Linear(50, 1)
        self.outV.weight.data.normal_(0, 0.1)
        self.outA = nn.Linear(50, N_ACTIONS)
        self.outA.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        State_Value = self.outV(x)
        Advantage = self.outA(x)

        # 由于同一个Q，可以有无数对不同的V和A组合表示，我们需要将值函数/优势函数固定
        # Q(s) = V(s) + [A(s,a) - sum(a')A(s,a') / N_ACTIONS]
        Actions_value = State_Value + Advantage - Advantage.mean()
        return Actions_value

class Dueling_DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # 当learn_step_counter=MEMORY_CAPACITY，
                                                                        # 会开始学习经验

        self.memory_counter = 0                                         # 经验池记忆计数器

        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS))     # 初始化经验池

        # 选取Dueling DQN的梯度下降方法（Adam）
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

        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dueling_dqn = Dueling_DQN()
Episode=[]
Score=[]

print('\nCollecting experience...')
for i_episode in range(80):
    s = env.reset()
    score = 0
    while True:
        env.render()
        a = dueling_dqn.choose_action(s)

        s_, r, done, info = env.step(a)

        x, x_dot, theta, theta_dot = s_

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        score += r

        dueling_dqn.store_transition(s, a, r, s_)

        if dueling_dqn.memory_counter > MEMORY_CAPACITY:
            dueling_dqn.learn()

        if done:
            break
        s = s_

    print(i_episode)
    print(dueling_dqn.memory_counter)

    Episode.append(i_episode)
    Score.append(score)

# 绘制幕数-每幕得分图
plt.plot(Episode, Score)
plt.show()