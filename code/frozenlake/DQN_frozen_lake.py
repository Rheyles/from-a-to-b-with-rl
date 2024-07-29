import json
import gymnasium as gym
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from collections import namedtuple, deque
from datetime import datetime
from colorama import Fore, Style

from params import *

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))

class ReplayMemory(object):
    """A class representing the replay memory 
    based on which our DQN agent will improve themselves"""

    def __init__(self, capacity=DQN_MEM_SIZE):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size=DQN_MEM_BATCH_SIZE):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    """ A class representing the DQN network of our 
    agent. We can call directly the forward method which
    is rather convenient !
    
    NOTE :  We are computing Q values, so if we have four possible actions we need to 
    output four elements with our network. """

    def __init__(self, num_inputs: int | np.ndarray, 
                 num_actions : int,
                 size=NN_SIZE):

        super(DQNNetwork, self).__init__()    

        self.linear1 = nn.Linear(num_inputs, size) 
        self.linear2 = nn.Linear(size, size) 
        self.linear3 = nn.Linear(size, num_actions) 
        
    def forward(self, x) -> torch.Tensor:
       
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class DQNAgent():
    def __init__(self, env:gym.Env) -> None:
        """
        Creates the DQN Agent, with its two networks (policy and target)
        that will kind of 'chase each other' and converge to faithfully
        represent the expected (discounted) reward corresponding to 
        all possible (state, action) pairs.

        ARGS
        ----
        * env : your gym environment
        """

        self.action_space = env.action_space
        self.num_actions = env.action_space.n
        self.num_inputs = 1 if len(env.observation_space.shape) == 0 else env.observation_space.shape[0]

        self.policy_net = DQNNetwork(self.num_inputs, self.num_actions)
        self.target_net = DQNNetwork(self.num_inputs, self.num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.update_strategy = DQN_NETWORK_REFRESH_STRATEGY
        self.optimizer = torch.optim.RMSprop(params=self.policy_net.parameters(), lr=DQN_LR, weight_decay=DQN_L2)

        self.memory = ReplayMemory()
        self.last_action = None
        self.global_steps = 0 # Will be useful to compute epsilon
        self.gamma = GAMMA        
    
    def act(self, state: int, deterministic=False) -> torch.Tensor:
        """
        Agent selects one of four actions to take either as a prediction of the model or randomly:
        The chances of picking a random action are high in the beginning and decrease with number of iterations

        Args:
            state (gym.ObsType): Observation
            deterministic (Bool) : Whether we allow exploration or not

        Returns:
            act ActType : Action that the agent performs
        """

        sample = random.random()
        action = self.action_space.sample()
        self.epsilon = DQN_EPS_END + (DQN_EPS_START - DQN_EPS_END) * np.exp(-self.global_steps / DQN_EPS_DECAY)
        
        if sample > self.epsilon or deterministic:
            with torch.no_grad():
                # torch.no_grad() used when inference on the model is done
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = self.policy_net.forward(state).argmax(0).item()

        return action

    def update(self) -> float:
        """
        This function runs the optimization of the model:
        it takes a batch from the buffer, creates the non final mask and computes:
        Q(s_t, a) and V(s_{t+1}) to compute the Loss, performs backprop and then clips gradient
        returns the computed loss

        Args:
            device (_type_, optional): Device to run computations on. Defaults to DEVICE.

        Returns:
            losses (list): Calculated Loss
        """
        if len(self.memory) < DQN_MEM_BATCH_SIZE: return 0
        transitions = self.memory.sample(DQN_MEM_BATCH_SIZE)
        batch = Transition(*zip(*transitions)) # We transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation)

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).unsqueeze(-1)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).unsqueeze(-1)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.int64).unsqueeze(-1)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32).unsqueeze(-1)
        term_batch = torch.tensor(np.array(batch.terminated), dtype=torch.float32).unsqueeze(-1)


        ### Updating the POLICY network based on the LOSS
        # We compute Q(s_t, a) for the actions taken in the batch 
        # We then compute the value of the next state from the point of view of the target net
        # i.e. the reward of (s_t, a) + the highest of Q(s_t+1,a) over the actions a (only if s_t+1 is not terminal)
        Q_state_action = self.policy_net.forward(state_batch).gather(1, action_batch)

        with torch.no_grad():    
            candidate_state_values = reward_batch + self.target_net.forward(next_state_batch) * (1 - term_batch) * GAMMA 
            V_future_state = torch.max(candidate_state_values, dim=1, keepdim=True).values
            
            # if any(reward_batch):
            #     data = np.hstack((state_batch.detach(), action_batch.detach(), next_state_batch.detach(), reward_batch.detach(), done_batch.detach(), Q_state_action.detach(), V_future_state.detach()))
            #     import time
                
            #     print(candidate_state_values)

            #     print('Before')
            #     print('\n')
            #     print(data.round(2).T)
            #     print('\n')
            

        # Compute Loss (Huber)
        loss = torch.sum(Q_state_action - V_future_state) ** 2

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # with torch.no_grad():
        #     Q_state_action = self.policy_net.forward(state_batch).gather(1, action_batch)
        #     candidate_state_values = reward_batch + self.target_net.forward(next_state_batch) * (1 - done_batch) * GAMMA 
        #     V_future_state = torch.max(candidate_state_values, dim=1, keepdim=True).values
            
        #     if any(reward_batch):
        #         data = np.hstack((state_batch.detach(), action_batch.detach(), next_state_batch.detach(), reward_batch.detach(), done_batch.detach(), Q_state_action.detach(), V_future_state.detach()))
        #         import time
                
        #         print('After')
        #         print('\n')
        #         print(data.round(2).T)
        #         print('\n')
        #         input()


        ### Updating the TARGET network based on the REFRESH STRATEGY
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        if self.update_strategy == 'soft':
            # Soft update of the target network's weights :  θ′ ← τ θ + (1 −τ )θ′

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*DQN_TAU 
                + target_net_state_dict[key]*(1-DQN_TAU)
            self.target_net.load_state_dict(target_net_state_dict)

        elif self.global_steps % int(1/DQN_TAU) == 0:
            self.target_net.load_state_dict(policy_net_state_dict)

        return loss.item()


# Initialize Environment and agent
env = gym.make("FrozenLake-v1" , map_name=MAP_NAME, render_mode='rgb_array', is_slippery=False)
env.metadata['render_fps'] = 150
agent = DQNAgent(env)
file_prefix = 'DQN_FrozenLake_' + MAP_NAME + "_" + datetime.now().strftime('%y-%m-%d_%H-%M')
 
# Init some monitoring stuff
cum_rewards, losses, steps, global_step = [], [], [], 0
best_model_score = -np.inf

try:
    # Init log file
    with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
        myfile.write(f'episode,step,cum_reward,loss,epsilon\n')

    # Save hyperparameters now
    print('./models/' + file_prefix + '_prms.json')
    with open('models/' + file_prefix + '_prms.json', 'w') as myfile:
        import params as pm
        prm_dict = {key : val for key, val in pm.__dict__.items() if '__' not in key}
        json.dump(prm_dict, myfile)

    for episode in range(NUM_EPISODES):
        
        state, _ = env.reset()
        cum_reward, loss, step, done, actions = 0, 0, 0, False, []

        while not done:
            # Let the agent act and face the consequences
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if NEG_REWARD_LAKE and terminated and reward == 0:
                reward = -1

            # Update thingies
            agent.memory.push(state, action, next_state, reward, terminated)
            loss += agent.update()
            agent.global_steps += 1
            cum_reward += reward
            state = next_state
            step += 1

            
        steps.append(step)
        losses.append(loss/step)
        cum_rewards.append(cum_reward)

        if episode % 100 == 0:
            with torch.no_grad():
                fig, ax = plt.subplots()
                all_states = torch.arange(0,16, dtype=torch.float32).
                all_values = agent.policy_net.forward(all_states)
                ax.imshow(all_values, cmin=-1, cmax=2)
                fig.draw()
                plt.pause(5)
                plt.close(fig)

        # Write some stuff in a file
        with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
            myfile.write(f'{episode},{step},{cum_reward},{loss/step}, {agent.epsilon}\n')
        
        # Print stuff in console
        bg, colr = Style.RESET_ALL, Fore.GREEN if cum_reward >= 1 else Fore.RED
        print(f'Ep. {episode:4d} | ' \
            + colr + f'Rw {cum_reward:2.0f} | Dur {step:3d}' + bg \
            + f' | AL {loss/step:7.3e} | EPS {agent.epsilon:5.2f} | GL_STP {agent.global_steps:5d}')

        # If model really good, save it
        if np.mean(cum_rewards[-20:]) > best_model_score:
            torch.save(agent.policy_net.state_dict() , './models/' + file_prefix + '_best.pkl')  
            best_model_score = np.mean(cum_rewards[-20:])
     
except KeyboardInterrupt:
    print('\nInterrupted w. Keyboard !')

finally:
    torch.save(agent.policy_net.state_dict() , './models/' + file_prefix + '_latest.pkl')
