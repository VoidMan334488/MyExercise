import argparse
import gym
import torch
import random
import os
import numpy as np
from utils.misc import make_dir, test_cuda
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from algorithms.sac import SoftAC
import datetime
from collections import namedtuple


def run(config):
    # 创建保存目录和日志
    run_dir, run_num = make_dir(config.env_id, config.model_name)
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    # 设置随机种子
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建mujoco环境
    env = gym.make(config.env_id)
    env.action_space.seed(seed)

    # 创建 sac 模型
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Model runs on " + torch.cuda.get_device_name() + ".")
    else:
        device = torch.device("cpu")
        print("Model runs on CPU.")
    model = SoftAC(env,
                   tau=config.tau,
                   pi_lr=config.pi_lr,
                   q_lr=config.q_lr,
                   gamma=config.gamma,
                   alpha=config.alpha,
                   epsilon=config.epsilon,
                   hidden_dim=config.hidden_dim,
                   batch_size=config.batch_size,
                   device=device
                   )
    # 创建经验回放池
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward', 'done'))
    replay_buffer = ReplayBuffer(config.buffer_length, transition=Transition)

    total_step_num = 0  # 运行总步数
    # 开始训练循环
    for ep_i in range(1, config.n_episodes):
        obs = env.reset(seed=seed)
        done = False
        episode_reward = 0  # 当局游戏总奖励
        episode_steps = 0  # 当局游戏总步数
        while not done:
            if config.start_steps > total_step_num:
                actions = env.action_space.sample()  # 随机动作
            else:
                actions = model.select_actions(obs)   # 模型采样动作

            next_obs, rewards, done, infos = env.step(actions)
            # env.render()
            replay_buffer.push(obs, actions, next_obs, rewards, done)
            obs = next_obs
            episode_reward += rewards
            episode_steps += 1
            total_step_num += 1

            # 经验回放模型更新
            if len(replay_buffer) >= config.batch_size:
                model.update(replay_buffer, logger=logger)

        # 进行测试
        if ep_i % 10 == 0 and config.if_eval:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                obs = env.reset(seed=seed)
                episode_reward = 0
                done = False
                while not done:
                    actions = model.select_actions(obs, eval=True)
                    next_obs, reward, done, _ = env.step(actions)
                    episode_reward += reward
                    obs = next_obs
                avg_reward += episode_reward
            avg_reward /= episodes

            logger.add_scalar('avg_reward/test', avg_reward, ep_i)
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

        # 训练日志
        logger.add_scalar('reward/train', episode_reward, ep_i)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"
              .format(ep_i, total_step_num, episode_steps, round(episode_reward, 2)))
        print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        if total_step_num > config.num_steps:
            break

    # model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="HalfCheetah-v3", help="Name of environment")
    parser.add_argument("--model_name", default="sac", help="Name of directory to store ")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=5000, type=int)
    parser.add_argument('--num_steps', type=int, default=500001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for training")
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--pi_lr", default=0.0003, type=float)
    parser.add_argument("--q_lr", default=0.0003, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--epsilon", default=1e-6, type=float)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--if_eval", default=True, type=bool)
    parser.add_argument("--use_gpu", default=True, type=bool)

    config = parser.parse_args()
    test_cuda()
    run(config)
