import json
import glob
import sys
import gymnasium as gym
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import namedtuple, deque
from datetime import datetime
from colorama import Fore, Style

from params import *

### FROZEN MAPS

maps = {'4x4' : [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ],
        
    '5x5_easy' : [
        'SFFFF',
        'HFHFF',
        'FFFFH',
        'FFHFH',
        'HFFFG'
    ],

    '5x5_hard' : [
        'SFFFF',
        'HFHHF',
        'HFFFH',
        'HHFFF',
        'GFFFH'
    ],

    '6x6_easy' : [
        'SFFFFF',
        'HFHFFF',
        'FFHFHH',
        'HFFFHF',
        'HFFFFF',
        'HFHGHF'
    ],

    '6x6_hard' : [
        'SFFHFF',
        'HFFFFF',
        'FFHHHH',
        'FFHFHG',
        'FHFFFF',
        'FFFFHF'
    ],

    '8x8' : [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]
}

### CLASSES

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
    def __init__(self, env:gym.Env, can_observe=OBS_AGENT, nn_size=NN_SIZE) -> None:
        """
        Creates the DQN Agent, with its two networks (policy and target)
        that will kind of 'chase each other' and converge to faithfully
        represent the expected (discounted) reward corresponding to 
        all possible (state, action) pairs.

        ARGS
        ----
        * env : your gym environment
        """

        self.map = np.array(env.unwrapped.desc.astype(str))
        self.action_space = env.action_space
        self.num_actions = env.action_space.n
        self.can_observe = can_observe
        self.num_inputs = 5 if self.can_observe else 1 # Will not work for other environments, specific to Frozen Lake
        
        self.policy_net = DQNNetwork(self.num_inputs, self.num_actions, size=nn_size)
        self.target_net = DQNNetwork(self.num_inputs, self.num_actions, size=nn_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.update_strategy = DQN_NETWORK_REFRESH_STRATEGY
        self.optimizer = torch.optim.RMSprop(params=self.policy_net.parameters(), lr=DQN_LR, weight_decay=DQN_L2)

        # This guy will lower the learning rate when the model becomes good
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (optimizer=self.optimizer,
             mode='max', 
             factor= DQN_SCHEDULER_FACTOR, 
             min_lr = DQN_SCHEDULER_MIN_LR, 
             patience=DQN_SCHEDULER_PATIENCE)

        self.memory = ReplayMemory()
        self.last_action = None
        self.global_steps = 0 # Will be useful to compute epsilon
        self.gamma = GAMMA        


    def observe(self, state:int):
        """
        !!! SPECIFIC TO FROZEN LAKE !!!
        Transforms the observation of the agent (initially only their tile no)
        to include the type of the adjacent tiles [0 : Hole, 1 : Frozen, 1 : Start, 2 : Gift]

        Args:
            state (torch.Tensor): number of the tile where the agent stands

        Returns:
            np.ndarray: Array of the encoded close tiles
        """
        
        if self.can_observe: 
            ENCODING = {'H':0, 'F':1, 'S':1, 'G':2} # F : normal tile / S : start point / H : hole ('lake') / G : gift
            row, col = np.unravel_index(state, shape=self.map.shape)
            neighbours = np.array([[row, col+1], [row,col-1],[row+1,col],[row-1,col]])
            neighbours = np.clip(neighbours, a_min=0, a_max=self.map.shape[0]-1) # Note that it will fail miserably for non-square maps
            neighbours = [ENCODING[self.map[row, col]] for row, col in neighbours]
            neighbours.insert(0,state)
            return np.array(neighbours)
        
        return np.atleast_1d(state)

    
    def act(self, state, deterministic=False) -> torch.Tensor:
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
        self.epsilon = DQN_EPS_END + (DQN_EPS_START - DQN_EPS_END) * np.exp(-self.global_steps / DQN_EPS_DECAY)
        
        if sample > self.epsilon or deterministic:
            with torch.no_grad():
                # torch.no_grad() used when inference on the model is done
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state = torch.tensor(state, dtype=torch.float32)
                action = self.policy_net.forward(state).argmax(0).item()
        else: 
            action = self.action_space.sample()

        return action

    def update(self) -> float:
        """
        This function runs the optimization of the model:
        it takes a batch from the buffer, creates the non final mask and computes:
        Q(s_t, a) and V(s_{t+1}) to compute the Loss, performs backprop and
        returns the computed loss

        Args:
            device (_type_, optional): Device to run computations on. Defaults to DEVICE.

        Returns:
            losses (list): Calculated Loss
        """
        if len(self.memory) < DQN_MEM_BATCH_SIZE: return 0
        transitions = self.memory.sample(DQN_MEM_BATCH_SIZE)
        batch = Transition(*zip(*transitions)) # We transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation)

        state_batch = torch.tensor(np.vstack(batch.state), dtype=torch.float32)
        next_state_batch = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.int64).unsqueeze(-1)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.int64).unsqueeze(-1)
        term_batch = torch.tensor(np.array(batch.terminated), dtype=torch.bool).unsqueeze(-1)
        not_term_batch = torch.logical_not(term_batch)

        ### Updating the POLICY network based on the LOSS
        # We compute Q(s_t, a) for the actions taken in the batch 
        # We then compute the value of the next state from the point of view of the target net
        # i.e. the reward of (s_t, a) + the highest of Q(s_t+1,a) over the actions a (only if s_t+1 is not terminal)
        Q_state_action = self.policy_net.forward(state_batch).gather(1, action_batch)

        with torch.no_grad():    
            candidate_state_values = reward_batch + self.target_net.forward(next_state_batch) *  GAMMA * not_term_batch 
            best_action = torch.argmax(candidate_state_values, dim=1, keepdim=True)     
            V_future_state = candidate_state_values.gather(1, best_action)

        # Compute Loss (Huber)
        loss_fun = nn.SmoothL1Loss()
        loss = loss_fun(Q_state_action, V_future_state)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ### Updating the TARGET network based on the REFRESH STRATEGY
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        if self.update_strategy == 'soft':
            # Soft update of the target network's weights :  θ′ ← τ θ + (1 −τ )θ′
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*DQN_TAU + target_net_state_dict[key]*(1-DQN_TAU)
            self.target_net.load_state_dict(target_net_state_dict)

        elif self.update_strategy == 'hard' and self.global_steps % int(1/DQN_TAU) == 0:
            self.target_net.load_state_dict(policy_net_state_dict)

        return loss.item()




#########################################
# TRAIN PROGRAMME
#########################################

def train():
    # Initialize Environment and agent
    env = gym.make("FrozenLake-v1" , desc=maps[MAP_NAME], render_mode='ansi', is_slippery=False)
    env.metadata['render_fps'] = 150

    agent = DQNAgent(env, can_observe=OBS_AGENT)
    obs_str = '_OBS' if OBS_AGENT else '' 
    file_prefix = 'DQN_FrozenLake_' + MAP_NAME + obs_str + "_" + datetime.now().strftime('%y-%m-%d_%H-%M')
    
    # Init some monitoring stuff
    cum_rewards, losses, steps = [], [], []
    best_model_score = -np.inf

    try:
        # Init log file
        with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
            myfile.write(f'episode,step,cum_reward,loss,epsilon,learning_rate\n')

        # Save hyperparameters now
        print('./models/' + file_prefix + '_prms.json')
        with open('models/' + file_prefix + '_prms.json', 'w') as myfile:
            import params as pm
            prm_dict = {key : val for key, val in pm.__dict__.items() if '__' not in key}
            json.dump(prm_dict, myfile)

        for episode in range(NUM_EPISODES):
            
            state, _ = env.reset()
            state = agent.observe(state)
            cum_reward, ep_loss, step, done = 0, 0, 0, False

            while not done:
                # Let the agent act and face the consequences
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = agent.observe(next_state)
                if NEG_REWARD_LAKE and terminated and reward == 0: reward = -1 # Corresponds to falling into a lake
                done = terminated or truncated

                # Update thingies
                agent.memory.push(state, action, next_state, reward, terminated)
                ep_loss += agent.update()
                agent.global_steps += 1
                cum_reward += reward
                state = next_state
                step += 1
        
            agent.scheduler.step(metrics=cum_reward)
            lr = agent.scheduler.optimizer.param_groups[0]['lr']    
            steps.append(step)
            losses.append(ep_loss/step)
            cum_rewards.append(cum_reward)

            # Write some stuff in a file
            with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
                myfile.write(f'{episode},{step},{cum_reward},{ep_loss/step}, {agent.epsilon}, {lr}\n')
            
            # Print stuff in console
            bg, colr = Style.RESET_ALL, Fore.GREEN if cum_reward >= 1 else Fore.RED
            print(f'Ep. {episode:4d} | GL_STP {agent.global_steps:5d} | ' \
                + colr + f'Rw {cum_reward:2.0f} | Dur {step:3d}' + bg \
                + f' | AL {ep_loss/step:7.3e} | EPS {agent.epsilon:5.2f} | LR {lr:.2e}')

            # If model really good, save it
            if np.mean(cum_rewards[-20:]) > best_model_score:
                torch.save(agent.policy_net.state_dict() , './models/' + file_prefix + '_best.pkl')  
                best_model_score = np.mean(cum_rewards[-20:])
        
    except KeyboardInterrupt:
        print('\nInterrupted w. Keyboard !')

    finally:
        torch.save(agent.policy_net.state_dict(), './models/' + file_prefix + '_latest.pkl')





def evaluate(file, index=-1, mode='human'):
    """ Evaluates a trained model (default : the best version of the last trained model)"""

    model_file = glob.glob('./models/' + file + '*latest*')[index]
    json_file = glob.glob('./models/' + file + '*prms*')[index]
    base_name = '_'.join(json_file.split('_')[:-1])

    with open(json_file) as myfile:
        prms = json.load(myfile)
    
    env = gym.make('FrozenLake-v1', desc=maps[prms['MAP_NAME']], render_mode=mode, is_slippery=False)
    env.metadata['render_fps'] = 10
    if mode != 'human': 
        env = gym.wrappers.RecordVideo(env=env, 
                                    video_folder='',
                                    name_prefix=base_name,
                                    episode_trigger=lambda x: True)

    agent = DQNAgent(env, nn_size=prms['NN_SIZE'], can_observe=prms['OBS_AGENT'])
    agent.policy_net.load_state_dict(torch.load(model_file))
    agent.target_net.load_state_dict(torch.load(model_file))  
        
    # A bit of display
    print('\n\nOpening : ' + model_file)
    print('-'*40 + '\nHyperparameters')
    for key, val in prms.items():
        print(f'{key:>30s} : {val}') 
    print('-'*40 + '\n')   

    done, action = False, -1
    cum_rewards, steps = [], []
    state, _ = env.reset()

    if mode != 'human': 
        env.start_video_recorder()
    
    for episode in range(5):
        cum_reward, step, done = 0, 0, False
        state, _ = env.reset()
        state = agent.observe(state)
        
        while not done:
            # env.render()
            action = agent.act(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.observe(next_state)
            done = terminated or truncated

            state = next_state
            cum_reward += reward
            step += 1
            agent.global_steps += 1

        cum_rewards.append(cum_reward)
        steps.append(step)
    
    # Close the environment
    if mode != 'human':
        env.close_video_recorder()
        env.close()

    print(f'5 episodes : Rewards {cum_rewards}')    


if __name__ == '__main__':
    if len(sys.argv) > 1 and ('--eval' in sys.argv[1] or '-e' in sys.argv[1]):
        file = 'DQN_Frozenlake_' 
        print(file)
        if len(sys.argv) > 2:
            file += sys.argv[2]
        if len(sys.argv) > 3 and '--video' in sys.argv[3]:
            evaluate(file=file, mode='rgb_array')
        else: 
            evaluate(file=file, mode='human')
    else:
        train()