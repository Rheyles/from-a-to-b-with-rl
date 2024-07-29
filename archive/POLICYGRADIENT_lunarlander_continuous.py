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
    def __init__(self, obs_space:gym.spaces.Box, mode='sample'):
        obs_samples = np.array([obs_space.sample() for _ in range(10000)])
        self.std = np.std(obs_samples, axis=0)
        self.mu = np.mean(obs_samples, axis=0)
        self.high = obs_space.high
        self.low = obs_space.low
        self.mode = mode
        
    def normalize_obs(self, observation=np.ndarray):
        """Normalises the observation. Either does it
        using "sample" option (from self.mode) :: gaussian normalization based on 10 000 observation
        (random) samples. Otherwise does it so that we bound all observations between -1 and 1"""

        if self.mode == 'sample':
            normed_observation = (observation - self.mu) / self.std
        else:
            normed_observation = 2 * (observation - self.low)/(self.high - self.low) - 1
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

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)


    def act(self, obs:np.ndarray) -> tuple[torch.Tensor, torch.distributions.Normal]:
        """ Asks the actor network to select an action based on an observation (obs)
        Returns the action and the probability distribution associated to this action
        [it is basically the product of the probability of the "individual" action along
        all the dimensions of the action space]."""

        obs = torch.Tensor(obs)
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        
        avgs = self.avg(x) # I do not force anything for the average (the action will be clipped instead)
        stds = 0.01 + F.softplus(self.std(x)) # Stds only have to be positive --> softplus
        distrib = Normal(avgs, stds)
        action = distrib.sample()
        log_prob = torch.sum(distrib.log_prob(action), axis=-1) 
        self.log_probs.append(log_prob.unsqueeze(0))
        action = action.clip(min=self.action_min, max=self.action_max)
 
        return action, distrib

    def update(self, critic_advantage:torch.Tensor):
        """ Updates the actor network based on critic advantage.
         This is done at the end of an episode. """
        
        log_probs = torch.cat(self.log_probs)
        self.optimizer.zero_grad()
        loss = -torch.sum(log_probs * critic_advantage.detach())    # We don't update the parameters of the critic
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def cleanup(self):
        del self.log_probs[:]


class Critic(nn.Module):
    def __init__(self, obs_space:gym.spaces.Box,  
                 critic_size=32, 
                 learning_rate=1e-4,
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

        self.rewards = []
        self.values  = []

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma


    def judge(self, obs:np.ndarray) -> torch.Tensor:
        """ Ask the critic to judge a state (estimates the V(s)) and the
        next state (estimates the V(s'))."""

        obs = torch.Tensor(obs)
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        out = self.linear3(x)

        self.values.append(out)

        return None
    

    def update(self):
        """ Computes the advantage of the previous batch
        and uses it to improve itself (the critic net)"""

        # Computes the discounted rewards
        n_steps = len(self.rewards)
        advantage = deque(maxlen=n_steps)

        for index in range(n_steps-1, -1, -1):
            adv = self.rewards[index] + self.values[index+1] * self.gamma - self.values[index]
            advantage.appendleft(adv)
            
        self.advantage = torch.cat(tuple(advantage))
        loss = (self.advantage ** 2).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def cleanup(self):
        """ Deletes variables that are no longer used"""
        del self.rewards[:]
        del self.values[:]
        self.advantage = None

















def evaluate(actor_file='trained_actor_continuous.pkl'):
    env = gym.make('LunarLander-v2', continuous=True, render_mode='human')
    state, _ = env.reset()

    actor_net = ContinuousActor(obs_space = env.observation_space, 
                               action_space = env.action_space, 
                               actor_size = 32)
    
    normalizer = Normalizer(obs_space=env.observation_space)

    actor_net.load_state_dict(torch.load(actor_file))

    done = False
    rewards, steps = 0, 0
    
    while not done:
        env.render()
        action, distrib = actor_net.act(normalizer.normalize_obs(state))
        new_state, reward, terminated, truncated, _ = env.step(action.detach().numpy())
        done = terminated or truncated
        
        state = new_state
        rewards += reward
        steps += 1

    print(f'Final : {steps} steps, {rewards} rewards')    





def train():
    
    env = gym.envs.make('LunarLander-v2', continuous=True, render_mode='human')
    env.metadata['render_fps'] = 150
    actor_net = ContinuousActor(obs_space=env.observation_space, 
                            action_space=env.action_space,
                            learning_rate=1e-3)
    
    critic_net = Critic(obs_space=env.observation_space,
                        learning_rate=1e-2)
    
    normalizer = Normalizer(obs_space=env.observation_space)
    
    max_episode_num = 500
    all_rewards = []
    duration = []
    actor_loss = []
    critic_loss = []

    for episode in range(max_episode_num):
        
        critic_net.cleanup()
        actor_net.cleanup()
        state, _ = env.reset()
        state_normed = normalizer.normalize_obs(state)
        critic_net.judge(state_normed)
        step, cum_reward, done = 0, 0, False
        


        while not done:
            
            # env.render()
            
            state_normed = normalizer.normalize_obs(state)
            action, distrib = actor_net.act(state_normed)
            
            new_state, reward, terminated, truncated, _ = env.step(action.detach().numpy())
            new_state_normed = normalizer.normalize_obs(new_state)
            done = terminated or truncated
        
            critic_net.rewards.append(reward)
            critic_net.judge(new_state_normed)


            state = new_state
            step += 1
            cum_reward += reward
            
            action_lst = action.tolist()
            loc_lst = distrib.loc.tolist()
            std_lst = distrib.scale.tolist()
            
            print(f'\r ep {episode:3d}, stp {step:3d} |'\
                + f' main {action_lst[0]:+.2f} ({loc_lst[0]:+.2f} +- {std_lst[0]:.2f})' \
                + f' lat  {action_lst[1]:+.2f} ({loc_lst[1]:+.2f} +- {std_lst[1]:.2f}) | ', end='')
            
            if keyboard.is_pressed('l'):
                time.sleep(0.25)

        
        c_loss = critic_net.update()
        a_loss = actor_net.update(critic_advantage=critic_net.advantage)

        all_rewards.append(cum_reward)
        duration.append(step)
        actor_loss.append(a_loss)
        critic_loss.append(c_loss)

        
        if all_rewards[-1] > 0:
            print(Fore.GREEN + f" rwd: {all_rewards[-1]:+7.1f}", end='')
        else: 
            print(Fore.RED + f" rwd: {all_rewards[-1]:+7.1f}", end='')

        print(Fore.WHITE + f' | c_loss {c_loss:.2f} a_loss {a_loss:+7.1f}')

            
    torch.save(actor_net.state_dict(), 'trained_actor_continuous.pkl')  
    torch.save(critic_net.state_dict(), 'trained_critic_continuous.pkl')

    
    avg_rewards = np.convolve(all_rewards, 0.1*np.ones(10), mode="same")
    fig, ax = plt.subplots(nrows=3, sharex=True)
    ax[0].plot(all_rewards, color='lightsalmon', label='rewards (inst.)')
    ax[0].plot(avg_rewards, 'r', label='rewards (avg)')
    ax[0].legend()
    ax[1].plot(duration, 'b', label='duration')
    ax[1].legend()
    ax[2].plot(actor_loss, color='purple', label='actor loss')
    ax[2].plot(critic_loss, color='orange', label='critic loss')
    ax[2].legend()
    plt.xlabel('Episode')
    plt.show()


if __name__ == '__main__':
    # train()
    evaluate()