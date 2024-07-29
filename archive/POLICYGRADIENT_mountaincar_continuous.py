import sys
import torch  
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import time
import keyboard
from colorama import Fore, Back, Style

import matplotlib.pyplot as plt

# Constants
GAMMA = 0.99

class Normalizer():
    """ A class used to (easily) normalize inputs (observations)
    from the environment"""
    def __init__(self, obs_space:gym.spaces.Box):
        obs_samples = np.array([obs_space.sample() for _ in range(10000)])
        self.std = np.std(obs_samples, axis=0)
        self.mu = np.mean(obs_samples, axis=0)
        
    def normalize_obs(self, observation=np.ndarray):
        """Normalises the observation"""
        normed_observation = (observation - self.mu / self.std)
        return normed_observation

class ContinuousActor(nn.Module):
    """ Implements a __continuous__ A2C actor
    """

    def __init__(self, obs_space:gym.spaces.Box, 
                 action_space:gym.spaces.Box, 
                 actor_size=16, learning_rate=1e-2,
                 dropout_rate=0.0, 
                 min_std=0.2):
        """ NB: action_space should be the limits [[min0, max0], 
        [min1, max1], ...] of the action space"""

        
        super(ContinuousActor, self).__init__()

        self.obs_min = torch.Tensor(obs_space.low)
        self.obs_max = torch.Tensor(obs_space.high)
        self.num_inputs = obs_space.shape[0]
        self.action_min = torch.Tensor(action_space.low)
        self.action_max = torch.Tensor(action_space.high)
        self.num_actions = action_space.shape[0]
        self.min_std_val = min_std
        self.log_probs = []
        
        self.dropout_rate = dropout_rate

        self.linear1 = nn.Linear(self.num_inputs, actor_size)
        self.linear2 = nn.Linear(actor_size, actor_size)
        self.avg = nn.Linear(actor_size, self.num_actions)
        self.std = nn.Linear(actor_size, self.num_actions)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.avg.weight)
        torch.nn.init.xavier_uniform_(self.std.weight)


        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)


    def act(self, obs:np.ndarray) -> tuple[torch.Tensor, torch.distributions.Normal]:
        obs = (torch.Tensor(obs).float() - self.obs_min)/(self.obs_max - self.obs_min)
        obs = torch.unsqueeze(obs, 0)
        x = F.elu(self.linear1(obs))
        x = F.elu(self.linear2(x))
        
        # We force the avgs to be in the valid action range 
        # So I use hard sigmoids and I then adapt their range to the individual actions 
        avgs = torch.clip(self.avg(x), self.action_min, self.action_max)
        
        # Stds only have to be positive; I still clip their value to something reasonable 
        # values : nothing less than 0.01 and nothing more than 2 x the max possible value.
        stds = F.softplus(self.std(x))
        min_std = self.min_std_val * torch.ones_like(self.action_max)
        max_std = self.action_max - self.action_min
        stds = stds.clip(min=min_std, max=max_std)
        
        distrib = Normal(avgs, stds)
        action = distrib.sample()
        self.log_probs.append(distrib.log_prob(action))
 
        return action.clip(self.action_min, self.action_max), distrib

    def update(self, critic_advantage:torch.Tensor):
        """ Updates the actor network based on critic advantage """
        
        log_probs = torch.cat(self.log_probs)
        loss = -torch.mean(log_probs * critic_advantage.detach())    # We don't update the parameters of the critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def cleanup(self):
        """ Deletes the rewards and the log-probabilities 
        (normally after one episode is over)"""
        del self.log_probs[:]


class Critic(nn.Module):
    def __init__(self, obs_space:gym.spaces.Box,  
                 critic_size=16, 
                 learning_rate=1e-2,
                 gamma=0.99,
                 dropout_rate=0.0):
        """ Critic part of the A2C agent. Can use a different 
        learning rate (usually higher than actor ?) and a different
        network size"""

        super(Critic, self).__init__()
        
        self.obs_min = torch.Tensor(obs_space.low)
        self.obs_max = torch.Tensor(obs_space.high)
        self.num_inputs = obs_space.shape[0]

        self.linear1 = nn.Linear(self.num_inputs, critic_size)
        self.linear2 = nn.Linear(critic_size, critic_size)
        self.linear3 = nn.Linear(critic_size, 1)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.rewards = []
        self.values = []
        self.advantage = None
        self.dropout_rate = dropout_rate

    def judge(self, obs:np.ndarray, done=False) -> torch.Tensor:
        """ Ask the critic to judge a state (estimates the V(s)) and the
        next state (estimates the V(s')). You can specify if the episode
        is terminated, in which case the value will automatically be
        set to 0 """
        
        if done: 
            self.values.append(0)
            return torch.Tensor([0])

        obs = (torch.Tensor(obs).float() - self.obs_min)/(self.obs_max - self.obs_min)
        obs = torch.unsqueeze(obs, 0)
        x = F.elu(F.dropout(self.linear1(obs)), self.dropout_rate)
        x = F.elu(F.dropout(self.linear2(x)), self.dropout_rate)
        out = self.linear3(x)
        self.values.append(out)    

        return out
    

    def update(self):
        """ Computes the advantage of the previous batch
        and uses it to improve itself"""

        # Computes the discounted rewards
        n_steps = len(self.rewards)
        advantage = deque(maxlen=n_steps)

        for index in range(n_steps-1, -1, -1):
            adv = self.rewards[index] + self.values[index+1] * self.gamma \
                - self.values[index]
            advantage.appendleft(adv)
            
        
        self.optimizer.zero_grad()
        self.advantage = torch.cat(tuple(advantage))
        loss = torch.mean(self.advantage ** 2)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def cleanup(self):
        """ Cleans up the critic network"""
        del self.rewards[:]
        del self.values[:]
        self.advantage = None




def train():
    
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
    env.metadata['render_fps'] = 150
    actor_net = ContinuousActor(obs_space=env.observation_space, 
                            action_space = env.action_space, 
                            actor_size=128,
                            learning_rate=0.0005, 
                            min_std=0.05)
    
    critic_net = Critic(obs_space=env.observation_space, 
                        critic_size=32, 
                        learning_rate=0.002,
                        gamma=GAMMA)
    
    normalizer = Normalizer(env.observation_space)
    
    max_episode_num = 1000
    numsteps = []
    avg_numsteps = []
    all_rewards = []
    avg_rewards = []

    for episode in range(max_episode_num):
        state, _ = env.reset()
        actor_net.cleanup()
        critic_net.cleanup()
        critic_net.judge(state)
        step, done = 0, False

        max_x, max_v = -1, 0

        while not done:
            
            # env.render()

            normed_state = normalizer.normalize_obs(state)
            action, distrib = actor_net.act(normed_state)

            new_state, reward, terminated, truncated, _ = env.step(action[0])
            done = terminated or truncated

            normed_new_state = normalizer.normalize_obs(new_state)
            critic_net.judge(normed_new_state, done=done)
            critic_net.rewards.append(reward)
            
            state = new_state

            step += 1
            if new_state[0] > max_x:
                max_x = new_state[0]
            if new_state[1] > max_v:
                max_v = new_state[1]
            
            print(f'\r ep {episode:3d}, stp {step:3d} | act {action.item():+.2f}, avg {distrib.loc.item():+.2f} std {distrib.scale.item():.2f} max_x {max_x:+.2f} max_v {max_v:.2f} |', end='')
            if keyboard.is_pressed('l'):
                time.sleep(0.25)


        c_loss = critic_net.update()
        a_loss = actor_net.update(critic_advantage=critic_net.advantage)

        numsteps.append(step)
        avg_numsteps.append(np.mean(numsteps[-10:]))
        all_rewards.append(np.sum(critic_net.rewards))
        avg_rewards.append(np.mean(all_rewards[-10:]))
        
        if all_rewards[-1] > 0:
            print(Fore.GREEN + f" rwd: {all_rewards[-1]:5.1f}", end='')
        else: 
            print(Fore.RED + f" rwd: {all_rewards[-1]:5.1f}", end='')

        print(Fore.WHITE + f' | c_loss {c_loss:.2f} a_loss {a_loss:.2f}')

        # print(f'Act : {np.sum(act_time):.2f}, judge : {np.sum(judge_time):.2f}, ' \
        #     + f'step : {np.sum(step_time):.2f}, update : {update_time:.2f}, render : {np.sum(render_time):.2f}')
        
            
    torch.save(actor_net.state_dict(), 'trained_actor_continuous.pkl')  
    torch.save(critic_net.state_dict(), 'trained_critic_continuous.pkl')

    plt.plot(all_rewards)
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.show()


if __name__ == '__main__':
    train()
    # evaluate()