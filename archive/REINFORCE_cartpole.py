import sys
import torch  
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque


import matplotlib.pyplot as plt

# Constants
GAMMA = 0.999

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=1e-2):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.log_probs = []
        self.rewards = []
        
    def forward(self, state : torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def action(self, state:np.ndarray) -> tuple[int, np.ndarray]:
        """ Selects an action and returns it and the log_probability
        of taking that action.
        
        ARGS
        -----
        * state : the state (e.g. given by env.step() or env.reset())

        RETURNS
        -----
        * likeliest_action [int] : self-explanatory, isnt't it ? 
        * log_prob [torch.Tensor of an int] : the log-probability associated with the 
            likeliest action
        """
        print('action !')
        state = torch.Tensor(state).float().unsqueeze(0)
        probs = torch.squeeze(self.forward(state), dim=0)
        distrib = Categorical(probs)
        action = distrib.sample()
        self.log_probs.append(distrib.log_prob(action).unsqueeze(0))
        return action.item()
    
    def update(self):
        """ Updates the Policy network by computing the sum of discounted
        rewards, multiplied by the logarithm of the probability of the actions. Going
        up the gradient of the product of the two improves the chances that
        the agent gets the reward. 

        """
        # Computes the discounted rewards
        discounted = deque(maxlen=len(self.rewards))
        R = 0
        for rew in self.rewards[::-1]:
            R = R * GAMMA + rew
            discounted.appendleft(R)
        
        discounted = torch.Tensor(discounted)
        log_probs = torch.cat(self.log_probs)
        discounted = (discounted - torch.mean(discounted)) / (1e-6 + torch.std(discounted))
        policy_gradient = -torch.sum(log_probs * discounted)
        
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()

    def cleanup(self):
        """ Deletes the rewards and the log-probabilities 
        (normally after one episode is over)"""
        del self.log_probs[:]
        del self.rewards[:]


def evaluate(file='trained_model.pkl'):
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 16)
    policy_net.load_state_dict(torch.load(file))
    done = False
    rewards, steps = 0, 0
    
    while not done:
        env.render()
        action = policy_net.action(state)
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = new_state
        rewards += reward
        steps += 1

    print(f'Final : {steps} steps, {rewards} rewards')


    

def main():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    policy_net = PolicyNetwork(env.observation_space.shape[0], 
                               env.action_space.n, 
                               hidden_size=16,
                               learning_rate=0.01)
    
    max_episode_num = 2000
    max_steps = 500
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state, _ = env.reset()
        policy_net.cleanup()

        for steps in range(max_steps):
            env.render()
            print('main !')
            action = policy_net.action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            policy_net.rewards.append(reward)
            state = new_state

            if done:
                policy_net.update()
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(policy_net.rewards))
                avg_rewards = np.mean(all_rewards[-10:])
                
                print(f"episode: {episode:5d}, reward: {all_rewards[-1]:5.1f} " \
                    + f"average_reward: {avg_rewards:6.2f}, length: {steps:5.1f}")
                
                break
            
    torch.save(policy_net.state_dict(), 'trained_model.pkl')  

    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episode')
    plt.show()

if __name__ == '__main__':
    main()
    # evaluate()