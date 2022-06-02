import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def weights_init_(m):
    """初始化网络权值"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)  # Xavier均匀分布
        torch.nn.init.constant_(m.bias, 0)  # 常数初始化


class QNet(nn.Module):
    """Q值网络"""
    def __init__(self, num_inputs, num_outputs, hidden_dim=64):
        super(QNet, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)

    def forward(self, inps):
        x = F.relu(self.linear1(inps))
        x = F.relu(self.linear2(x))
        q = self.linear3(x)
        return q


class DeterministicNet(nn.Module):
    """
    确定性策略网络
    :return: mean, log_std
    """
    def __init__(self, num_inputs, num_outputs, hidden_dim=64):
        super(DeterministicNet, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action = torch.tanh(self.linear3(x))
        return action


class GaussianNet(nn.Module):
    """
    高斯Q值网络
    :return: mean, log_std
    """
    def __init__(self, num_inputs, num_outputs, hidden_dim=64):
        super(GaussianNet, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_outputs)
        self.log_std_linear = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)  # 限制方差过大
        return mean, log_std

