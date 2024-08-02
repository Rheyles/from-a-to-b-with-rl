import sys
import glob
import json
import torch  
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
from colorama import Fore, Style, Back
from datetime import datetime

from params import *


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
    def __init__(self, env:gym.Env,
                 size=ACTOR_SIZE, 
                 learning_rate=ACTOR_LR,
                 dropout_rate=ACTOR_DROPOUT_RATE,
                 L2_alpha=ACTOR_L2_ALPHA,
                 entropy_beta=ACTOR_ENTROPY_BETA):
        
        super(DiscreteActor, self).__init__()

        self.num_actions = env.action_space.n
        self.num_inputs = env.observation_space.shape[0]
        self.log_prob = None
        self.entropy_beta = entropy_beta

        self.linear1 = nn.Linear(self.num_inputs, size)
        self.linear2 = nn.Linear(size, size)
        self.linear3 = nn.Linear(size, self.num_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=L2_alpha)
        self.dropout_rate = dropout_rate
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (optimizer=self.optimizer,
             mode='max', 
             factor = ACTOR_SCHEDULER_FACTOR, 
             min_lr = ACTOR_SCHEDULER_MIN_LR, 
             patience= ACTOR_SCHEDULER_PATIENCE)

    
    def act(self, state:np.ndarray, deterministic=False) -> torch.Tensor:
        """ Selects an action and computes the log probability associated with it
        Returns the action and stores in the "log_probs" attribute the log_probs.
        For test / evaluation purposes, you can set the `deterministic` flag to True
        so that the optimal / likeliest decision is always taken."""

        state = torch.Tensor(state).float().unsqueeze(0)
        x = F.relu(self.linear1(state))
        x = F.relu(F.dropout(self.linear2(x), self.dropout_rate))
        probs = F.softmax(self.linear3(x), dim=1)   # A squeeze(dim=0) might be needed here
        
        distrib = Categorical(probs)
        action = distrib.sample() if not deterministic else distrib.mode.unsqueeze(0)
        self.log_prob = distrib.log_prob(action)
        self.entropy = distrib.entropy()

        return action.item()
    
    
    def update(self, critic_advantage : torch.Tensor):
        """ Updates the actor network based on critic input """
        
        loss = -self.log_prob * critic_advantage.detach() - self.entropy_beta * self.entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Critic(nn.Module):
    def __init__(self, env:gym.Env,  
                size=CRITIC_SIZE, 
                learning_rate=CRITIC_LR,
                gamma=GAMMA,
                dropout_rate=CRITIC_DROPOUT_RATE,
                l2_alpha=CRITIC_L2_ALPHA):
        """ Critic part of the A2C agent. Can use a different 
        learning rate (usually higher than actor ?) and a different
        network size. Also a dropout_rate and a L2 regularization coefficient."""

        
        super(Critic, self).__init__()

        self.gamma = gamma
        self.num_inputs = env.observation_space.shape[0]
        self.linear1 = nn.Linear(self.num_inputs, size)
        self.linear2 = nn.Linear(size, size)
        self.linear3 = nn.Linear(size, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=l2_alpha)
        self.advantage = None
        self.dropout_rate = dropout_rate
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (optimizer=self.optimizer,
             mode='max', 
             factor = CRITIC_SCHEDULER_FACTOR, 
             min_lr = CRITIC_SCHEDULER_MIN_LR, 
             patience= CRITIC_SCHEDULER_PATIENCE)


    def judge(self, state:np.ndarray) -> torch.Tensor:
        """ Ask the critic to judge a state (estimates the V(s)) and the
        next state (estimates the V(s')). You can specify if the episode
        is terminated, in which case the value will automatically be
        set to 0 """
        
        state = torch.Tensor(state).unsqueeze(0)
        x = F.relu(self.linear1(state))
        x = F.relu(F.dropout(self.linear2(x), p=self.dropout_rate))
        return self.linear3(x)

    def update(self, rewards, normed_state, normed_next_state, done:bool):
        """ Computes the advantage of the previous batch
        and uses it to improve itself"""

        # Computes the estimator of the discounted rewards through TD
        value, next_value = self.judge(normed_state), self.judge(normed_next_state)
        self.advantage = rewards + next_value * self.gamma * (1 - done) - value
        loss = self.advantage ** 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()







def train(n_episodes = NUM_EPISODES):
    
    file_prefix = 'A2C_LunarLander_' + datetime.now().strftime('%y-%m-%d_%H-%M')
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    actor_net = DiscreteActor(env)
    critic_net = Critic(env)
    normalizer = Normalizer(obs_space=env.observation_space, mode='minmax')

    # Init some monitoring stuff
    cum_rewards = []
    best_model_score = -np.inf
    
    # Init log file
    with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
        myfile.write(f'episode,step,cum_reward,c_loss,a_loss,action_0,action_1,action_2,action_3,actor_lr,critic_lr\n')

    # Save hyperparameters now
    print('./models/' + file_prefix + '_prms.json')
    with open('models/' + file_prefix + '_prms.json', 'w') as myfile:
        import params as pm
        prm_dict = {key : val for key, val in pm.__dict__.items() if '__' not in key}
        json.dump(prm_dict, myfile)


    try:
        for episode in range(n_episodes):
            state, _ = env.reset()
            normed_state = normalizer.normalize_obs(state)
            done, step = False, 0
            actions, cum_reward = [], 0
            actor_loss, critic_loss = 0, 0

            while not done:

                action = actor_net.act(normed_state)
                new_state, reward, terminated, truncated, _ = env.step(action)
                normed_new_state = normalizer.normalize_obs(new_state)
                done = terminated or truncated
              
                # Updating networks
                c_loss = critic_net.update(reward, normed_state, normed_new_state, done)
                a_loss = actor_net.update(critic_advantage=critic_net.advantage)

                # Updating variables
                actor_loss += a_loss
                critic_loss += c_loss
                cum_reward += reward
                state = new_state
                normed_state = normed_new_state
                actions.append(action)
                step += 1
            
            act_frac = [actions.count(val)/len(actions) 
                                for val in range(env.action_space.n)]
            
            actor_net.scheduler.step(metrics=cum_reward)
            critic_net.scheduler.step(metrics=cum_reward)
            a_lr = actor_net.scheduler.optimizer.param_groups[0]['lr']    
            c_lr = critic_net.scheduler.optimizer.param_groups[0]['lr']
            cum_rewards.append(cum_reward)    
            
            # Fancier colors
            bg, colr = Style.RESET_ALL, Fore.RED
            if cum_reward > 0:
                colr = Fore.YELLOW
            if cum_reward > 100:
                colr = Fore.BLUE
            if cum_reward > 200:
                colr = Fore.GREEN
                

            print(f'Ep. {episode:4d} | ' \
                + colr + f'Rw {cum_reward:+8.2f} | Dur {step:4d}' + bg \
                + f' | AL {a_loss/step:7.2f}' \
                + f' | CL {c_loss/step:7.2f}' \
                + f' | ACT0 {act_frac[0]:.3f}' \
                + f' | ACT1 {act_frac[1]:.3f}' \
                + f' | ACT2 {act_frac[2]:.3f}' \
                + f' | ACT3 {act_frac[3]:.3f}' \
                + f' | A_LR {a_lr:5.2e}' \
                + f' | C_LR {c_lr:5.2e}')
                    

            # Write some stuff in a file
            with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
                myfile.write(f'{episode},{step},{cum_reward},{c_loss/step},{a_loss/step},{act_frac[0]},{act_frac[1]},{act_frac[2]},{act_frac[3]},{a_lr},{c_lr}\n')
            
            if len(cum_rewards) > 20 and np.mean(cum_rewards[-20:]) > 200:
                print(f'Exceeded 200 in average for 20 episodes. Early stopping')
                break

            # If model really good, save it
            if (len(cum_rewards) > 20) and (np.mean(cum_rewards[-20:]) > best_model_score):
                torch.save(actor_net.state_dict() , './models/' + file_prefix + '_best_actor.pkl')  
                torch.save(critic_net.state_dict(), './models/' + file_prefix + '_best_critic.pkl')
                best_model_score = np.mean(cum_rewards[-20:])

    except KeyboardInterrupt: 
        pass

    finally:
        print('Saving models ...')
        # torch.save(actor_net.state_dict(), 'trained_actor.pkl')  
        # torch.save(critic_net.state_dict(), 'trained_critic.pkl')






def evaluate(file='A2C_LunarLander_*', index=-1, mode='human'):
    """ Evaluates a trained model (default : the best version of the last trained model)"""

    actor_file = glob.glob('models/' + file + '*best_actor*')[index]
    json_file = glob.glob('models/' + file + '*prms*')[index]
    base_name = '_'.join(json_file.split('_')[:-1])
    env = gym.make('LunarLander-v2', render_mode=mode)

    with open(json_file) as myfile:
        prms = json.load(myfile)
        
    print('\n\nOpening file : ' + actor_file)
    print('-'*40 + '\nHyperparameters')
    for key, val in prms.items():
        print(f'{key:>25s} : {val}')    
    
    if mode != 'human': 
        env = gym.wrappers.RecordVideo(env=env, 
                                    video_folder='',
                                    name_prefix=base_name,
                                    episode_trigger=lambda x: True)

    normalizer = Normalizer(obs_space=env.observation_space, mode='minmax')
    actor_net = DiscreteActor(env, size=prms['ACTOR_SIZE'])
    actor_net.load_state_dict(torch.load(actor_file))

    done, cum_rewards = False, []
    state, _ = env.reset()

    if mode != 'human': 
        env.start_video_recorder()
    
    for episode in range(5):
        cum_reward, step, done = 0, 0, False
        state, _ = env.reset()
        normed_state = normalizer.normalize_obs(state)
        
        while not done:
            env.render()
            action = actor_net.act(normed_state, deterministic=True)
            new_state, reward, terminated, truncated, _ = env.step(action)
            normed_new_state = normalizer.normalize_obs(new_state)
            done = terminated or truncated
            state, normed_state = new_state, normed_new_state
            cum_reward += reward
            step += 1
        
        cum_rewards.append(cum_reward)
    
    # Close the environment
    if mode != 'human':
        env.close_video_recorder()
        env.close()

    print(f'5 episodes : Rewards {cum_rewards}')    


if __name__ == '__main__':
    if len(sys.argv) > 1 and ('--eval' in sys.argv[1] or '-e' in sys.argv[1]):
        if len(sys.argv) > 2 and '--video' in sys.argv[2]:
            evaluate(mode='rgb_array')
        else: 
            evaluate(mode='human')
    else:
        train()