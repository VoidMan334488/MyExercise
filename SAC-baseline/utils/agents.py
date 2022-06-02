import torch
from utils.nets import GaussianNet
from utils.nets import DeterministicNet
from torch.optim import Adam
from torch.distributions import Normal
from utils.misc import soft_update


class GaussianAgent(object):
    def __init__(self, env, lr=0.01, epsilon=1e-6, hidden_dim=64, device=torch.device("cpu")):
        self.obsize = env.observation_space.shape[0]
        self.acsize = env.action_space.shape[0]
        self.policies = GaussianNet(self.obsize, self.acsize, hidden_dim=hidden_dim).to(device)
        self.policy_optimizers = Adam(self.policies.parameters(), lr=lr, weight_decay=1e-3)
        self.epsilon = epsilon
        self.action_scale = torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.).to(device)
        self.action_bias = torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.).to(device)

    def step(self, obs):
        # 集合动作的高斯分布参数
        mean, log_std = self.policies(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class BaseAgent(object):
    def __init__(self, env, lr=0.01, epsilon=1e-6, hidden_dim=64, tau=0.005, device=torch.device("cpu")):
        self.obsize = env.observation_space.shape[0]
        self.acsize = env.action_space.shape[0]
        self.policy = DeterministicNet(self.obsize, self.acsize, hidden_dim=hidden_dim).to(device)
        self.target_policy = DeterministicNet(self.obsize, self.acsize, hidden_dim=hidden_dim).to(device)
        self.policy_optimizers = Adam(self.policy.parameters(), lr=lr)
        self.epsilon = epsilon
        self.tau = tau

    def step(self, obs):
        return self.policy(obs)

    def target_step(self, obs):
        return self.target_policy(obs)

    def soft_update(self):
        soft_update(self.target_policy, self.policy, self.tau)
