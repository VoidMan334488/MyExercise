from utils.nets import QNet
from utils.misc import hard_update
from utils.misc import soft_update
from torch.optim import Adam
import torch


class SoftBaseCritic(object):
    def __init__(self, env, lr=0.01, hidden_dim=64, tau=0.005, device=torch.device("cpu")):
        self.obsize = env.observation_space.shape[0]
        self.acsize = env.action_space.shape[0]
        self.value_net = QNet(self.obsize + self.acsize, 1, hidden_dim=hidden_dim).to(device)
        self.target_value_net = QNet(self.obsize + self.acsize, 1, hidden_dim=hidden_dim).to(device)
        hard_update(self.target_value_net, self.value_net)
        self.critic_optimizer = Adam(self.value_net.parameters(), lr=lr, weight_decay=1e-3)
        self.tau = tau

    def eval(self, obs, ac):
        ips = torch.cat([obs, ac], 1)
        return self.value_net(ips)

    def target_eval(self, obs, ac):
        ips = torch.cat([obs, ac], 1)
        return self.target_value_net(ips)

    def soft_update(self):
        soft_update(self.target_value_net, self.value_net, self.tau)


class BaseVCritic(object):
    def __init__(self, env, lr=0.01, hidden_dim=64, tau=0.005, device=torch.device("cpu")):
        self.obsize = env.observation_space.shape[0]
        self.acsize = env.action_space.shape[0]
        self.value_net = QNet(self.obsize, 1, hidden_dim=hidden_dim).to(device)
        self.target_value_net = QNet(self.obsize, 1, hidden_dim=hidden_dim).to(device)
        hard_update(self.target_value_net, self.value_net)
        self.critic_optimizer = Adam(self.value_net.parameters(), lr=lr, weight_decay=1e-3)
        self.tau = tau

    def eval(self, obs):
        return self.value_net(obs)

    def target_eval(self, obs):
        return self.target_value_net(obs)

    def soft_update(self):
        soft_update(self.target_value_net, self.value_net, self.tau)
