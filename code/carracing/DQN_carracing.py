import json
import glob
import sys
import time
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


class ConvDQN(nn.Module):

    def __init__(self, obs:tuple, num_actions:int, n_filters=16, dropout_rate=DROPOUT_RATE):

        super(ConvDQN, self).__init__()
        img_shape = obs.shape[-2:] # Dim 0 will be reserved for the batch
        n_imgs = obs.shape[-3]

        k1, k2, k3 = 4,2,1
        s1, s2, s3 = 4,2,1
        img_shape1 = [np.ceil((elem - (k1 - 1)) / (s1 * 2)).astype(int) for elem in img_shape]
        img_shape2 = [np.ceil((elem - (k2 - 1)) / (s2 * 2)).astype(int)   for elem in img_shape1]
        # img_shape3 = [(elem - (k3 - 1)) // (s3 * 2) for elem in img_shape2]

        n_linear = n_filters * 2 * np.prod(img_shape2) # Number of linear neurons when we flatten
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_imgs, n_filters, kernel_size=k1, stride=s1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout_rate))
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=k2, stride=s2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout_rate))
        self.lin1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_linear, n_linear // 4),
            nn.ReLU())
        
        self.lin2 = nn.Linear(n_linear // 4, num_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.lin1(out)
        out = self.lin2(out)
        return out


class DQNAgent():
    def __init__(self, env:gym.Env, n_filters=N_FILTERS, n_idle=N_IDLE, n_imgs=N_IMGS) -> None:
        """
        Creates the DQN Agent, with its two networks (policy and target)
        that will kind of 'chase each other' and converge to faithfully
        represent the expected (discounted) reward corresponding to 
        all possible (state, action) pairs.

        ARGS
        ----
        * env : your gym environment
        """
        
        self.n_imgs = n_imgs # Number of images that will go in the CNN
        self.n_idle = n_idle # Number of frames during which the agent keeps doing the same thang
        self.n_filters = n_filters # Number of CNN filters on the first layer
        self.epsilon = np.nan

        ini_state = env.reset()[0][:,:,1]/255 # We only consider the green channel for CarRacing
        ini_obs = [ini_state for _ in range(self.n_imgs)]
        self.obs = deque(ini_obs, maxlen=self.n_imgs)
        torch_obs = torch.tensor(np.array(self.obs), dtype=torch.float32).unsqueeze(0)
        
        self.action_space = env.action_space
        self.num_actions = env.action_space.n

        self.policy_net = ConvDQN(torch_obs, self.num_actions, n_filters=n_filters, dropout_rate=DROPOUT_RATE)
        self.target_net = ConvDQN(torch_obs, self.num_actions, n_filters=n_filters, dropout_rate=DROPOUT_RATE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.update_strategy = DQN_NETWORK_REFRESH_STRATEGY
        self.optimizer = torch.optim.AdamW(params=self.policy_net.parameters(), lr=DQN_LR, weight_decay=DQN_L2)

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

    def observe(self, state:np.ndarray):
        """Pre-processes the raw state of CarRacing to actually produce 
        a sequence of successive images."""

        self.obs.append(state[:,:,1]/255)   
        return torch.tensor(np.array(self.obs), dtype=torch.float32).unsqueeze(0)
    

    def step_idle(self, env:gym.Env, action:int | np.ndarray):
        """ A routine that plays n_idle steps from an environment with the same action. 
        The routine : 
        - discards the intermediate states (not put anywhere)
        - sums the rewards and returns them as a "collective" reward
        - returns early if the episode ends (truncated or )"""
        
        next_state, cum_reward, terminated, truncated, _ = env.step(action)

        for step in range(self.n_idle-1):
            if terminated or truncated:
                break
            next_state, reward, terminated, truncated, _ = env.step(action)
            cum_reward += reward

        return next_state, cum_reward, terminated, truncated, _

    
    def act(self, state:torch.Tensor, deterministic=False) -> torch.Tensor:
        """
        Agent selects one of four actions to take either as a prediction of the model or randomly:
        The chances of picking a random action are high in the beginning and decrease with number of iterations

        Args:
            state (torch.Tensor): here, since we are already stacking observations, we can 
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

                action_values = self.policy_net.forward(state)
                action = action_values.argmax(1).item()
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

        # Compute Loss (MSE here)
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
    env = gym.make("CarRacing-v2", continuous=False, render_mode='rgb_array')
    env.metadata['render_fps'] = 150

    agent = DQNAgent(env)
    file_prefix = 'DQN_Carracing_' + datetime.now().strftime('%y-%m-%d_%H-%M')
    
    # Init some monitoring stuff
    cum_rewards, losses, steps = [], [], []
    best_model_score = -np.inf
    time_start = time.time()

    try:
        # Init log file
        with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
            myfile.write(f'episode,step,time,cum_reward,loss,epsilon,action_0,action_1,action_2,action_3,action_4,learning_rate\n')

        # Save hyperparameters now
        print('./models/' + file_prefix + '_prms.json')
        with open('models/' + file_prefix + '_prms.json', 'w') as myfile:
            import params as pm
            prm_dict = {key : val for key, val in pm.__dict__.items() if '__' not in key}
            json.dump(prm_dict, myfile)

        for episode in range(NUM_EPISODES):
            
            state, _ = env.reset()
            for _ in range(N_START_SKIP): # At the beginning of each episode, do not train the agent since images are messed up
                action = 3
                env.step(action)

            state = agent.observe(state)
            cum_reward, ep_loss, step, done, actions = 0, 0, 0, False, []

            while not done:
                # Let the agent act and face the consequences
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = agent.step_idle(env, action) # Agent slacks off a bit ...
                next_state = agent.observe(next_state) # Update the image stack
                done = terminated or truncated

                # Update thingies
                agent.memory.push(state, action, next_state, reward, terminated)
                actions.append(action)
                ep_loss += agent.update()
                agent.global_steps += agent.n_idle
                cum_reward += reward
                state = next_state
                step += agent.n_idle
        
            time_now = time.time() - time_start
            act_frac = [actions.count(val)/len(actions) 
                            for val in range(env.action_space.n)]
            agent.scheduler.step(metrics=cum_reward)
            lr = agent.scheduler.optimizer.param_groups[0]['lr']    
            steps.append(step)
            losses.append(ep_loss/step)
            cum_rewards.append(cum_reward)

            # Write some stuff in a file
            with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
                myfile.write(f'{episode},{step},{time_now},{cum_reward},{ep_loss/step},{agent.epsilon},{act_frac[0]},{act_frac[1]},{act_frac[2]},{act_frac[3]},{act_frac[4]},{lr}\n')
                        
            # Fancier colors
            bg, colr = Style.RESET_ALL, Fore.RED
            if cum_reward > 200:
                colr = Fore.YELLOW
            if cum_reward > 400:
                colr = Fore.BLUE
            if cum_reward > 600:
                colr = Fore.GREEN
                
            print(f'Ep. {episode:4d} | ' \
                + colr + f'Rw {cum_reward:+8.2f} | EpLen {step:4d} | Time {time_now:7.1f}' + bg \
                + f' | LOSS {ep_loss/step:7.2f}' \
                + f' | EPS {agent.epsilon:5.2f}' \
                + f' | ACT0 {act_frac[0]:.3f}' \
                + f' | ACT1 {act_frac[1]:.3f}' \
                + f' | ACT2 {act_frac[2]:.3f}' \
                + f' | ACT3 {act_frac[3]:.3f}' \
                + f' | ACT4 {act_frac[4]:.3f}' \
                + f' | LR {lr:5.2e}')
            
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
    
    env = gym.make('CarRacing-v2', continuous=False, render_mode=mode)
    env.metadata['render_fps'] = 10
    if mode != 'human': 
        env = gym.wrappers.RecordVideo(env=env, 
                                    video_folder='',
                                    name_prefix=base_name,
                                    episode_trigger=lambda x: True)

    agent = DQNAgent(env, n_filters=prms['N_FILTERS'], n_idle=prms['N_IDLE'], n_imgs=prms['N_IMGS'])
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
        file = 'DQN_CarRacing' 
        print(file)
        if len(sys.argv) > 2:
            file += sys.argv[2]
        if len(sys.argv) > 3 and '--video' in sys.argv[3]:
            evaluate(file=file, mode='rgb_array')
        else: 
            evaluate(file=file, mode='human')
    else:
        train()
