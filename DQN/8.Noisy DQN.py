import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import math
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

# 噪声层的实现
class NoisyFactorizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)

        # 初始的噪声权重大小（为什么要用这种方式设置初始值？）
        sigma_init = sigma_zero / math.sqrt(in_features)

        # 含义是将一个固定不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面
        # 经过类型转换变成了模型的一部分，成为了模型中根据训练可以改动的参数了
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化
        # 所以训练网络的时候，可以使用nn.Parameter()来转换一个固定的权重数值，使的其可以跟着网络训练一直调优下去，学习到一个最适合的权重值

        # 这里新加入的SigmaWeight是可以训练的参数，用来调整Weight噪声的大小
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))

        # pytorch一般情况下，是将网络中的参数保存成orderedDict形式的
        # 这里的参数其实包含两种，一种是模型中各种module含的参数，即nn.Parameter,我们当然可以在网络中定义其他的nn.Parameter参数
        # 另一种就是buffer,前者每次optim.step会得到更新，而不会更新后者
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            # 这里新加入的SigmaBias是可以训练的参数，用来调整Bias噪声的大小
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))

        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            # 假设linear layer有q个神经元（节点），上一层有p个神经元（节点）
            # 这两层中，每个神经元生成一个独立的高斯噪音，他们的乘积作为相应连接权重的噪音
            noise_v = torch.mul(eps_in, eps_out).detach()
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()

        # 真正的权重Weight = self.weight + self.sigma_weight * noise_v
        # 真正的偏差Bias = bias = bias + self.sigma_bias * eps_out.t()  (eps_out.t()其实也就是这一层的噪声)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)

class Net(nn.Module):
    def __init__(self, ):
    # 一般噪声会添加在全连接层
    # 产生噪声的方法一般使用分解高斯噪声（factorized Gaussian noise）
        super(Net, self).__init__()

        # fc1为一个全连接层，输入尺寸为N_STATES=4，输出尺寸为50
        self.fc1 = NoisyFactorizedLinear(N_STATES, 50)

        # 使用一个正态分布初始化fc1的权值
        self.fc1.weight.data.normal_(0, 0.1)

        # out为一个全连接层，输入尺寸为50，输出尺寸为N_ACTIONS=2
        self.out = NoisyFactorizedLinear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    # 前向传播的过程，也就是逐个经过网络层，输入为状态的表示向量，输出为选取动作的概率
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# RL过程中总会想办法增加智能体的探索能力
# 而噪声DQN是通过对网络参数增加噪声来增强模型的探索能力
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # 当learn_step_counter=MEMORY_CAPACITY，
                                                                        # 会开始学习经验

        self.memory_counter = 0                                         # 经验池记忆计数器

        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS))     # 初始化经验池

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # 噪声网络中，不再需要ε-贪心策略来进行探索
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
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

dqn = DQN()
Episode=[]
Score=[]

print('\nCollecting experience...')
for i_episode in range(70):
    s = env.reset()
    score = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        s_, r, done, info = env.step(a)

        x, x_dot, theta, theta_dot = s_

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        score += r

        dqn.store_transition(s, a, r, s_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:
            break
        s = s_

    print(i_episode)
    print(dqn.memory_counter)

    Episode.append(i_episode)
    Score.append(score)

# 绘制幕数-每幕得分图
plt.plot(Episode, Score)
plt.show()
