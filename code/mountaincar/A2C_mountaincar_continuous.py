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
                 actor_size=32, 
                 learning_rate=1e-6):
        """ NB: action_space should be the limits [[min0, max0], 
        [min1, max1], ...] of the action space"""

        
        super(ContinuousActor, self).__init__()

        self.obs_min = torch.Tensor(obs_space.low)
        self.obs_max = torch.Tensor(obs_space.high)
        self.num_inputs = obs_space.shape[0]
        self.action_min = torch.Tensor(action_space.low)
        self.action_max = torch.Tensor(action_space.high)
        self.num_actions = action_space.shape[0]
        self.log_probs = []
        
        self.linear1 = nn.Linear(self.num_inputs, actor_size)
        self.linear2 = nn.Linear(actor_size, actor_size)
        self.avg = nn.Linear(actor_size, self.num_actions)
        self.std = nn.Linear(actor_size, self.num_actions)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.avg.weight)
        torch.nn.init.xavier_uniform_(self.std.weight)

        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.zeros_(self.linear2.bias)
        torch.nn.init.zeros_(self.avg.bias)
        torch.nn.init.zeros_(self.std.bias)        

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)


    def act(self, obs:np.ndarray) -> tuple[torch.Tensor, torch.distributions.Normal]:
        obs = torch.Tensor(obs)
        x = F.elu(self.linear1(obs))
        x = F.elu(self.linear2(x))
        
        avgs = self.avg(x) # I do not force anything for the average (the action will be clipped instead)
        stds = 0.01 + torch.exp(self.std(x)) # Stds only have to be positive --> softplus
        
        distrib = Normal(avgs, stds)
        action = distrib.sample()
        self.log_prob = torch.log(torch.exp(distrib.log_prob(action)) + 1e-5) # I know it is stupid ...
        
        action = action.clip(min=self.action_min, max=self.action_max)
 
        return action, distrib

    def update(self, critic_advantage:torch.Tensor):
        """ Updates the actor network based on critic advantage """
        
        self.optimizer.zero_grad()
        loss = -self.log_prob * critic_advantage.detach()    # We don't update the parameters of the critic
        loss.backward()
        self.optimizer.step()

        return loss.item()


class Critic(nn.Module):
    def __init__(self, obs_space:gym.spaces.Box,  
                 critic_size=32, 
                 learning_rate=1e-5,
                 gamma=0.99):
        """ Critic part of the A2C agent. Can use a different 
        learning rate (usually higher than actor ?) and a different
        network size"""

        super(Critic, self).__init__()
        
        self.obs_min = torch.Tensor(obs_space.low)
        self.obs_max = torch.Tensor(obs_space.high)
        self.num_inputs = obs_space.shape[0]
        self.advantage = None

        self.linear1 = nn.Linear(self.num_inputs, critic_size)
        self.linear2 = nn.Linear(critic_size, critic_size)
        self.linear3 = nn.Linear(critic_size, 1)

        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.zeros_(self.linear2.bias)
        torch.nn.init.zeros_(self.linear3.bias)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma


    def judge(self, obs:np.ndarray) -> torch.Tensor:
        """ Ask the critic to judge a state (estimates the V(s)) and the
        next state (estimates the V(s'))."""

        obs = torch.Tensor(obs)
        x = F.elu(self.linear1(obs))
        x = F.elu(self.linear2(x))
        out = self.linear3(x)

        return out
    

    def update(self, state: np.ndarray, new_state: np.ndarray, reward:float, done:bool):
        """ Computes the advantage of the current state
        and uses it to improve itself"""

        # Computes the discounted rewards

        self.advantage = reward + (1 - done) * self.gamma * self.judge(new_state) \
                            - self.judge(state)
        loss = self.advantage ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train():
    
    env = gym.envs.make('MountainCarContinuous-v0', render_mode='rgb_array')
    env.metadata['render_fps'] = 150
    actor_net = ContinuousActor(obs_space=env.observation_space, 
                            action_space=env.action_space)
    
    critic_net = Critic(obs_space=env.observation_space)
    
    normalizer = Normalizer(obs_space=env.observation_space)
    
    max_episode_num = 1000
    all_rewards = []

    for episode in range(max_episode_num):
        state, _ = env.reset()
        step, cum_reward, done = 0, 0, False
        max_x, max_v = -1, 0
        sts = []
        vals = []

        while not done:
            
            # env.render()

            action, distrib = actor_net.act(state)
            state_normed = normalizer.normalize_obs(state)

            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state_normed = normalizer.normalize_obs(new_state)

            done = terminated or truncated
            if new_state[0] > max_x: max_x = new_state[0]
            if new_state[1] > max_v: max_v = new_state[1]

            c_loss = critic_net.update(state_normed, new_state_normed, reward, done)
            a_loss = actor_net.update(critic_advantage=critic_net.advantage)

            with torch.no_grad():
                next_val = critic_net.judge(new_state)
            
            sts.append(new_state)
            vals.append(next_val)

            state = new_state
            step += 1
            cum_reward += reward
            
            
            print(f'\r ep {episode:3d}, stp {step:3d} | act {action.item():+.2f}, avg {distrib.loc.item():+.2f} std {distrib.scale.item():.2f} max_x {max_x:+.2f} max_v {max_v:.2f} |', end='')
            if keyboard.is_pressed('l'):
                time.sleep(0.25)

        # import matplotlib.pyplot as plt
        # plt.ion()
        # sts = np.vstack(sts)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.plot(sts[:,0], sts[:,1], np.squeeze(vals), 'o')
        # plt.pause(2)

        all_rewards.append(cum_reward)
        
        if all_rewards[-1] > 0:
            print(Fore.GREEN + f" rwd: {all_rewards[-1]:5.1f}", end='')
        else: 
            print(Fore.RED + f" rwd: {all_rewards[-1]:5.1f}", end='')

        print(Fore.WHITE + f' | c_loss {c_loss:.2f} a_loss {a_loss:.2f}')

        # print(f'Act : {np.sum(act_time):.2f}, judge : {np.sum(judge_time):.2f}, ' \
        #     + f'step : {np.sum(step_time):.2f}, update : {update_time:.2f}, render : {np.sum(render_time):.2f}')
        
            
    torch.save(actor_net.state_dict(), 'trained_actor_continuous.pkl')  
    torch.save(critic_net.state_dict(), 'trained_critic_continuous.pkl')

    
    avg_rewards = np.convolve(all_rewards, 0.1*np.ones(10), mode="same")
    plt.plot(all_rewards)
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.show()


if __name__ == '__main__':
    train()
    # evaluate()