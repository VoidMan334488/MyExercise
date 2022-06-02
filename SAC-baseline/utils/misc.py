from pathlib import Path
import torch


def test_cuda():
    flag = torch.cuda.is_available()
    print("-------------------------------------------------------------------")
    print("cuda.is_available : " + str(flag))
    if flag:
        device_count = torch.cuda.device_count()
        print("Device_count : " + str(torch.cuda.device_count()))
        current_device = torch.cuda.current_device()
        print("Current_device : " + str(current_device))
        print("Device_name : " + str(torch.cuda.get_device_name()))
    print("-------------------------------------------------------------------")

def make_dir(env_id, model_name):
    """为保存的模型创建目录"""
    model_dir = Path('./models') / env_id / model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    return run_dir, run_num


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
