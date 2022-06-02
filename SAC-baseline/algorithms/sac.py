import torch
import torch.nn.functional as F
import numpy
from utils.agents import GaussianAgent
from utils.critics import SoftBaseCritic

class SoftAC(object):
    def __init__(self, env,
                 tau=0.01, pi_lr=0.01, q_lr=0.01, gamma=0.99, alpha=0.2,
                 epsilon=1e-6, hidden_dim=128,
                 batch_size=128, device=torch.device("cpu")
                 ):
        self.obsize = env.observation_space.shape[0]
        self.acsize = env.action_space.shape[0]
        self.agent = GaussianAgent(env, lr=pi_lr, epsilon=epsilon, hidden_dim=hidden_dim, device=device)
        self.critic = SoftBaseCritic(env, lr=q_lr, hidden_dim=hidden_dim, tau=tau, device=device)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.updates = 0
        self.batch_size = batch_size

    def select_actions(self, obs, eval=False):
        torch_obs = torch.Tensor(obs).to(self.device).unsqueeze(0)
        if eval:    # 如果是评测采用mean值作为动作，采集经验时采用采样值
            _, _, torch_agent_actions = self.agent.step(torch_obs)
        else:
            torch_agent_actions, _, _ = self.agent.step(torch_obs)
        actions = torch_agent_actions.cpu().detach().numpy()[0]
        return actions

    def prep_training(self):
        self.critic.value_net.train()
        self.critic.target_value_net.train()
        self.agent.policies.train()

    def prep_rollouts(self, device='cpu'):
        self.critic.value_net.eval()
        self.critic.target_value_net.eval()
        self.agent.policies.eval()

    def update(self, replay_buffer, logger=None):
        self.prep_training()
        transitions = replay_buffer.sample(self.batch_size)
        batch = replay_buffer.transition(*zip(*transitions))  # 转换成为一批次
        self.updates += 1
        # 先将batch转换为array加速计算
        obs = torch.FloatTensor(numpy.array(batch.state)).to(self.device)
        acs = torch.FloatTensor(numpy.array(batch.action)).to(self.device)
        next_obs = torch.FloatTensor(numpy.array(batch.next_state)).to(self.device)
        rews = torch.FloatTensor(numpy.array(batch.reward)).to(self.device).unsqueeze(1)

        next_state_action, next_state_log_pi, _ = self.agent.step(next_obs)
        qf_next_target = self.critic.target_eval(next_obs, next_state_action)
        min_qf_next_target = qf_next_target - self.alpha * next_state_log_pi
        next_q_value = rews + self.gamma * (min_qf_next_target)
        qf = self.critic.eval(obs, acs)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf_loss = F.mse_loss(qf, next_q_value)

        self.critic.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic.critic_optimizer.step()

        pi, log_pi, _ = self.agent.step(obs)
        qf_pi = self.critic.eval(obs, pi)
        policy_loss = ((self.alpha * log_pi) - qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.agent.policy_optimizers.zero_grad()
        policy_loss.backward()
        self.agent.policy_optimizers.step()

        self.critic.soft_update()
        self.prep_rollouts()

        logger.add_scalar('loss/critic_loss', qf_loss, self.updates)
        logger.add_scalar('loss/policy', policy_loss, self.updates)
