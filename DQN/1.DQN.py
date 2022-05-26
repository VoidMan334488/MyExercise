import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib
import matplotlib.pyplot as plt

#超参数设置
BATCH_SIZE = 32
LR = 0.01                   # 学习速率
EPSILON = 0.9               # 贪心策略中ε的大小
GAMMA = 0.9                 # 折扣因子
TARGET_REPLACE_ITER = 100   # 目标网络的更新频率
MEMORY_CAPACITY = 500       # 经验池的大小

# 这里使用的测试环境是平衡摆
env = gym.make('CartPole-v0')

# env其实并非CartPole类本身，而是一个经过包装的环境：
# gym的多数环境都用TimeLimit（源码）包装了，以限制Epoch，就是step的次数限制，比如限定为200次。所以小车保持平衡200步后，就会失败
# 用env.unwrapped可以得到原始的类，原始类想step多久就多久，不会200步后失败
env = env.unwrapped

# N_ACTIONS=2,分别表示向左或向右
N_ACTIONS = env.action_space.n

# N_STATES=4，表示当前的状态空间有4个维度
N_STATES = env.observation_space.shape[0]

# ENV_A_SHAPE用来表示动作的维度，这里由于只有2个动作，维度为1
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):
    def __init__(self, ):
        # 1. self指的是实例Instance本身，在Python类中规定，函数的第一个参数是实例对象本身，并且约定俗成，把其名字写为self
        #    也就是说，类中的方法的第一个参数一定要是self，而且不能省略

        # 2. 在python中创建类后，通常会创建一个__ init__ ()方法，这个方法会在创建类的实例的时候自动执行
        #    __ init__ ()方法必须包含一个self参数，而且要是第一个参数
        #    有时__ init__ ()方法中还需要传入其他参数，那在实例化一个对象的时候就要加上

        # 3. Python中的super(Net, self).__init__()是指首先找到Net的父类（比如是类NNet），然后把类Net的对象self转换为类NNet的对象，
        #    然后“被转换”的类NNet对象调用自己的init函数，其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西。
        #    回过头来看看我们的我们最上面的代码，Net类继承nn.Module，super(Net, self).__init__()就是对继承自父类nn.Module的属性进行初始化
        #    而且是用nn.Module的初始化方法来初始化继承的属性
        super(Net, self).__init__()

        # fc1为一个全连接层，输入尺寸为N_STATES=4，输出尺寸为50
        self.fc1 = nn.Linear(N_STATES, 50)

        # 使用一个正态分布初始化fc1的权值
        self.fc1.weight.data.normal_(0, 0.1)

        # out为一个全连接层，输入尺寸为50，输出尺寸为N_ACTIONS=2
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    # 前向传播的过程，也就是逐个经过网络层，输入为状态的表示向量，输出为选取动作的概率
    def forward(self, x):
        x = self.fc1(x)
        # 1.激活函数的主要作用是完成数据的非线性变换，解决线性模型的表达、分类能力不足的问题
        # 2.如果网络中全部是线性变换，则多层网络可以通过矩阵变换，直接转换成一层神经网络
        # 3.所以激活函数的存在，使得神经网络的“多层”有了实际的意义，使网络更加强大，增加网络的能力
        # 4.只有在输出层极小可能使用线性激活函数，在隐含层都使用非线性激活函数
        # 5.激活函数的另一个重要的作用是执行数据的归一化，将输入数据映射到某个范围内，再往下传递，这样可以限制数据的扩张
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        # DQN的整体结构中，包含2个网络，分别为价值网络和目标网络
        # 为什么要设置两个网络？为了解决自举带来的偏差
        # 以单独的Q学习为例，我们的更新公式为Q(S,A) = Q(S,A) + α[R + γ * max(A') Q(S',A') - Q(S,A)]
        # 假如在S状态下，我们选取A动作，只是偶然地得到了一个高于平均水平的回报，这样会使得我们对Q(S,A)作出过高的评价
        # 如果我们仅有一个Q网络，那由于自举，我们会将Q(S,A)不断设置为一个错误的值，这个值会一直影响其前序状态-动作对的更新，直到
        # Q(S,A)被再次更新，而且这样也会使得我们更倾向于探索一个错误的方向，这样等于是使得收敛出现了更大的波动
        # 如果我们有2个网络，那有可能，在网络传递参数之前，这个偏差就已经被修正了，这样实际采取的动作就不会有错误的倾向
        # 而且，在问题非凸的情况下，网络的波动很有可能会导致最终结果更倾向于陷入局部最优，这就更有减少方差的必要了
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # 当learn_step_counter=MEMORY_CAPACITY，
                                                                        # 会开始学习经验

        self.memory_counter = 0                                         # 经验池记忆计数器

        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS))     # 初始化经验池

        # 选取DQN的梯度下降方法（Adam）
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        # 选取损失函数
        self.loss_func = nn.MSELoss()

    # 根据价值网络选取动作
    def choose_action(self, x):
        # 1.torch.squeeze(x,N(default=0))
        # 假设x.size()=torch.Size([n0,n1,n2,...,nm])
        # 若nN=1,则变化后x.size()=torch.Size([n0,n1,n2,...,nN-1,nN+1,...,nm])
        # 反之则无变化
        # 2.torch.unsqueeze(x,N(default=0))
        # 给指定位置加上维数为1的维度
        # 因为我们要以BatchSize进行训练，所以要加上一个维度
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        #以ε的概率贪心地选取动作
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        # 以1-ε的概率随机地选取动作
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        # np.hstack将参数元组的元素数组按水平方向进行叠加
        # [[1,2],[3,4]]+[[5,6],[7,8]]=[[1,2,5,6],[3,4,7,8]]
        transition = np.hstack((s, [a, r], s_))

        # 经验池满了之后，用新的经验替代旧的经验
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 以TARGET_REPLACE_ITER的频率更新目标网络
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 从经验池中选取BATCH_SIZE进行学习
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + N_ACTIONS - 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS - 1:N_STATES + N_ACTIONS])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # Q学习！

        # gather的作用是根据索引b查找a，然后讲查找结果以张量矩阵的形式返回
        # a = tensor([                                b = tensor([
        #         [0, 1, 2, 3, 4],                            [1, 0, 0, 0, 0],
        #         [5, 6, 7, 8, 9],                            [0, 0, 1, 0, 0],
        #         [10, 11, 12, 13, 14]])                      [0, 0, 0, 0, 0]])
        # c = a.gather(0, b)，则                      d = a.gather(1, b)，则
        # c= tensor([                                 d = tensor([
        #         [5, 1, 2, 3, 4],                             [1, 0, 0, 0, 0],
        #         [0, 1, 7, 3, 4],                             [5, 5, 6, 5, 5],
        #         [0, 1, 2, 3, 4]])                            [10, 10, 10, 10, 10]])
        # 解释：gather是根据维度dim开始查找
        # 具体例子：
        # b[0][0]=1, c[0][0]=a[1][0]=5, d[0][0]=a[0][1]=1
        # b[0][1]=0, c[0][1]=a[0][1]=1, d[0][1]=a[0][0]=0
        # 也就是说，gather的维度是多少，则查找结果应该为：
        # 那一维Index替换为索引值，其他维度Index保持和索引值序号一致--->产生出的a[总Index]

        # 这里self.eval_net(b_s)会输出每个动作在该状态下对应的价值
        # self.eval_net(b_s).gather(1, b_a)会取出实际采取的那个动作对应的价值
        q_eval = self.eval_net(b_s).gather(1, b_a)

        # detach用于截断反向传播（不需要更新Target网络）
        q_next = self.target_net(b_s_).detach()

        # R + γ * max(A') Q(S',A')
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        # loss = R + γ * max(A') Q(S',A') - Q(S,A)
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

        # 采取动作，从环境中得到反馈
        s_, r, done, info = env.step(a)

        # 设置回报奖励函数
        # (位置x，x加速度, 偏移角度theta, 角加速度)
        x, x_dot, theta, theta_dot = s_

        # 变量env.x_threshold里存放着小车坐标的最大值（=2.4）
        # env.theta_threshold_radians里存放着theta最大角度的值
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
