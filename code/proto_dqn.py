import gymnasium as gym # type: ignore
import matplotlib.pyplot as plt
from itertools import count
from params import DEVICE, BATCH_SIZE, LR, GAMMA, TAU, MEM_SIZE, RENDER_FPS
from network import DQN
from agent import ReplayMemory, select_action, Transition
from display import plot_loss, pretty_print

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def run():

    env = gym.make("FrozenLake-v1",
                render_mode = 'human',
                is_slippery=False
                )
    env.metadata['render_fps'] = RENDER_FPS


    n_actions = int(env.action_space.n) # Get number of actions from gym action space
    n_observations = 1 # Get the number of state observations
    state, info = env.reset()

    policy_net = DQN(n_observations, n_actions).to(DEVICE)
    target_net = DQN(n_observations, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEM_SIZE)


episode_durations = []
losses = []



def optimize_model():
    '''
    Brice : I have no idea where to put this function. In agent.py or in network.py ?
    Well, for now let's keep it here.
    '''
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=DEVICE, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch.unsqueeze(-1)).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

    with torch.no_grad():
        result = target_net(non_final_next_states.unsqueeze(-1))
        next_state_values[non_final_mask] = result.max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # pretty_print((state_batch, action_batch, reward_batch, next_state_values), )

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    losses.append(float(loss))

    print(state_action_values.shape)
    plot_loss()



    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 150

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    for t in count():
        action = select_action(env, state, policy_net, device=DEVICE)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=DEVICE)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break

print('Complete')
plot_loss(show_result=True)
plt.ioff()
plt.show()
