import torch
from numpy import exp
import random
from params import EPS_END, EPS_START, EPS_DECAY, DEVICE

steps_done = 0

def select_action(env, state, policy_net, device=DEVICE):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        exp(-steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            result = policy_net(state)
            # print(result)
            return result.max(0).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
