import sys
import torch  
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

# Constants
GAMMA = 0.995


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
    

    
class DiscreteActor(nn.Module):
    def __init__(self, num_inputs:int, num_actions:int, 
                 actor_size=16, 
                 learning_rate=1e-5,
                 dropout_rate=0,
                 L2_alpha=0.0):
        
        super(DiscreteActor, self).__init__()

        self.num_actions = num_actions
        self.num_inputs = num_inputs 
        self.log_prob = None

        self.linear1 = nn.Linear(self.num_inputs, actor_size)
        self.linear2 = nn.Linear(actor_size, actor_size // 2)
        self.linear3 = nn.Linear(actor_size // 2 , actor_size // 4)
        self.linear4 = nn.Linear(actor_size // 4 , num_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=L2_alpha)
        self.dropout_rate = dropout_rate
    
    def act(self, state:np.ndarray, deterministic=False) -> torch.Tensor:
        """ Selects an action and computes the log probability associated with it
        Returns the action and stores in the "log_probs" attribute the log_probs.
        For test / evaluation purposes, you can set the `deterministic` flag to True
        so that the optimal / likeliest decision is always taken."""

        state = torch.Tensor(state).float().unsqueeze(0)
        x = F.relu(self.linear1(state))
        x = F.relu(F.dropout(self.linear2(x), self.dropout_rate))
        x = F.relu(F.dropout(self.linear3(x), self.dropout_rate))
        probs = F.softmax(self.linear4(x), dim=1)   # A squeeze(dim=0) might be needed here
        
        distrib = Categorical(probs)
        action = distrib.sample() if not deterministic else distrib.mode.unsqueeze(0)
        self.log_prob = distrib.log_prob(action)

        return action.item()
    
    
    def update(self, critic_advantage : torch.Tensor):
        """ Updates the actor network based on critic input """
        
        loss = -self.log_prob * critic_advantage.detach()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Critic(nn.Module):
    def __init__(self, num_inputs,  
                 critic_size=16, 
                 learning_rate=1e-2,
                 gamma=0.99,
                 dropout_rate = 0,
                 L2_alpha=0):
        """ Critic part of the A2C agent. Can use a different 
        learning rate (usually higher than actor ?) and a different
        network size. Also a dropout_rate and a L2 regularization coefficient."""

        
        super(Critic, self).__init__()

        self.gamma = gamma
        self.linear1 = nn.Linear(num_inputs, critic_size)
        self.linear2 = nn.Linear(critic_size, critic_size // 2)
        self.linear3 = nn.Linear(critic_size // 2, critic_size // 4 )
        self.linear4 = nn.Linear(critic_size // 4, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=L2_alpha)
        self.advantage = None
        self.dropout_rate = dropout_rate

    def judge(self, state:np.ndarray) -> torch.Tensor:
        """ Ask the critic to judge a state (estimates the V(s)) and the
        next state (estimates the V(s')). You can specify if the episode
        is terminated, in which case the value will automatically be
        set to 0 """
        
        state = torch.Tensor(state).unsqueeze(0)
        x = F.relu(self.linear1(state))
        x = F.relu(F.dropout(self.linear2(x), p=self.dropout_rate))
        x = F.relu(F.dropout(self.linear3(x), p=self.dropout_rate))
        return self.linear4(x)

    def update(self, rewards, normed_state, normed_next_state, done:bool):
        """ Computes the advantage of the previous batch
        and uses it to improve itself"""

        # Computes the estimator of the discounted rewards through TD
        value, next_value = self.judge(normed_state), self.judge(normed_next_state)
        self.advantage = rewards + next_value * self.gamma * (1 - done) - value
        loss = (self.advantage ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def evaluate(actor_file='trained_actor.pkl', critic_file='trained_critic.pkl'):
    env = gym.make('LunarLander-v2', render_mode='human')
    state, _ = env.reset()

    actor_net = DiscreteActor(num_inputs = env.observation_space.shape[0], 
                               num_actions= env.action_space.n, 
                               actor_size = 64)
    actor_net.load_state_dict(torch.load(actor_file))

    critic_net = Critic(num_inputs = env.observation_space.shape[0], 
                       critic_size = 64)
    critic_net.load_state_dict(torch.load(critic_file))

    done = False
    rewards, steps = 0, 0
    actions = []
    
    while not done:
        env.render()
        action = actor_net.act(state, deterministic=True)
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        critic_net.judge(state, done=done)
        state = new_state
        rewards += reward
        steps += 1
        actions.append(action)

    print(f'Final : {steps} steps, {rewards} rewards')    






def train(n_episodes = 10000, n_rolling_rewards=20):
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    num_inputs = env.observation_space.shape[0]
    actor_net = DiscreteActor(num_inputs, 
                               env.action_space.n, 
                               actor_size = 128,
                               learning_rate = 3e-5,
                               dropout_rate = 0,
                               L2_alpha= 1e-5)
    
    critic_net = Critic(num_inputs, 
                        critic_size = 128, 
                        learning_rate = 3e-4,
                        gamma = GAMMA, 
                        dropout_rate = 0,
                        L2_alpha = 1e-5)
    
    normalizer = Normalizer(obs_space=env.observation_space, mode='minmax')
    
    with open('train_log.csv', 'w') as myfile:
        myfile.write('ep,step,rew,a_loss,c_loss,action_0,action_1,action_2,action_3\n')
        rolling_rewards = []


    try:
        for episode in range(n_episodes):
            state, _ = env.reset()
            normed_state = normalizer.normalize_obs(state)
            done, step = False, 0
            actions, rewards = [], []
            c_losses, a_losses = [], []

            while not done:

                action = actor_net.act(normed_state)
                new_state, reward, terminated, truncated, _ = env.step(action)
                normed_new_state = normalizer.normalize_obs(new_state)
                done = terminated or truncated
              
                # Updating networks
                critic_loss = critic_net.update(reward, normed_state, normed_new_state, done)
                actor_loss = actor_net.update(critic_advantage=critic_net.advantage)

                # Updating variables
                a_losses.append(actor_loss)
                c_losses.append(critic_loss)
                actions.append(action)
                rewards.append(reward)
                state = new_state
                normed_state = normed_new_state
                step += 1
                

            episode_action_proba = [actions.count(val)/len(actions) 
                                for val in range(env.action_space.n)]
            critic_loss = np.mean(c_losses)
            actor_loss = np.mean(a_losses)
            sum_rewards = np.sum(rewards)
            rolling_rewards.append(sum_rewards)

            if len(rolling_rewards) > n_rolling_rewards:
                rolling_rewards.pop(0)
            
            print(f"episode: {episode:4d}, reward: {sum_rewards:+8.1f} " \
                + f"a_loss: {actor_loss:+8.1f}, c_loss : {critic_loss:+9.1f} "\
                + f"length: {step:6.0f}, p_0 {episode_action_proba[0]:.3f}")
                    

            with open('train_log.csv', 'a') as logfile:
                logfile.write(f'{episode},{step},{sum_rewards},{actor_loss},'\
                            + f'{critic_loss},{episode_action_proba[0]},{episode_action_proba[1]},'\
                            + f'{episode_action_proba[2]},{episode_action_proba[3]}\n')

            if np.mean(rolling_rewards) > 200:
                print(f'Exceeded 200 in average for {n_rolling_rewards} episodes')
                break

    except KeyboardInterrupt: 
        pass

    finally:
        print('Saving models ...')
        torch.save(actor_net.state_dict(), 'trained_actor.pkl')  
        torch.save(critic_net.state_dict(), 'trained_critic.pkl')

if __name__ == '__main__':
    train()
    # evaluate()