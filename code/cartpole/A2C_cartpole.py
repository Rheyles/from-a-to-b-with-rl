import json
import torch  
import glob
import sys
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
from datetime import datetime
from colorama import Fore, Style

from params import *



class Normalizer():
    """ A class used to (easily) normalize inputs (observations)
    from the environment"""
    def __init__(self, obs_space:gym.spaces.Box, mode='standard', manual_low=None, manual_high=None):
        self.mode = mode.lower()

        if self.mode == 'standard':
            obs_samples = np.array([obs_space.sample() for _ in range(10000)])
            self.std = np.std(obs_samples, axis=0)
            self.mu = np.mean(obs_samples, axis=0)
            print(f'Normalizing with mu : {self.mu} and std : {self.std}')

        if self.mode == 'minmax':
            self.high = obs_space.high if manual_high is None else np.array(manual_high)
            self.low = obs_space.low if manual_low is None else np.array(manual_low)
            print(f'Scaling with min {self.low} and max {self.high}')

        
    def normalize_obs(self, observation=np.ndarray):
        """Normalises the observation. Either does it
        using "sample" option (from self.mode) :: gaussian normalization based on 10 000 observation
        (random) samples. Otherwise does it so that we bound all observations between -1 and 1"""

        if self.mode == 'standard':
            normed_observation = (observation - self.mu) / self.std
        elif self.mode == 'minmax':
            normed_observation = 2 * (observation - self.low)/(self.high - self.low) - 1
        else: 
            normed_observation = observation

        return normed_observation
    


class DiscreteActor(nn.Module):
    def __init__(self, env:gym.Env, 
                 size=ACTOR_SIZE, 
                 learning_rate=ACTOR_LR,
                 dropout_rate=ACTOR_DROPOUT_RATE,
                 l2_alpha=ACTOR_L2_ALPHA,
                 two_nn_layers=ACTOR_TWO_NN_LAYERS):
        '''
        Actor part of the A2C agent
        - size : number of neurons on the agent network
        - learning rate 
        - gamma : reward discount factor
        - dropout rate : if you want to deactivate some neurons
        - l2_alpha : L2 regularization coefficient
        - two_nn_layers [bool] : add another hidden layer of neurons
        '''

        super(DiscreteActor, self).__init__()

        self.num_actions = env.action_space.n
        self.num_inputs = env.observation_space.shape[0]
        self.log_probs = []
        self.dropout_rate = dropout_rate
        self.two_nn_layers = two_nn_layers

        self.linear1 = nn.Linear(self.num_inputs, size)
        self.linear2 = nn.Linear(size, size)
        self.linear3 = nn.Linear(size, self.num_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=l2_alpha)
    
    def act(self, state:np.ndarray, deterministic=False) -> torch.Tensor:
        """ Selects an action and computes the log probability associated with it
        Returns the action and stores in the "log_probs" attribute the log_probs.
        For test / evaluation purposes, you can set the `deterministic` flag to True
        so that the optimal / likeliest decision is always taken.
        NOTE : the state has to be normed by the normalizer first
        """

        state = torch.Tensor(state).float().unsqueeze(0)
        x = F.relu(self.linear1(state))
        if self.two_nn_layers:
            x = F.relu(F.dropout(self.linear2(x), self.dropout_rate))
        probs = F.softmax(self.linear3(x), dim=1)   # A squeeze(dim=0) might be needed here
        
        distrib = Categorical(probs)
        action = distrib.sample() if not deterministic else distrib.mode.unsqueeze(0)
        self.log_prob = distrib.log_prob(action)

        return action.item()
    
    
    def update(self, critic_advantage : torch.Tensor) -> float:
        """ Updates the actor network based on critic input. Returns actor loss """
        
        loss = -self.log_prob * critic_advantage.detach()
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
                 l2_alpha=CRITIC_L2_ALPHA,
                 two_nn_layers=CRITIC_TWO_NN_LAYERS):
        """ Critic part of the A2C agent. Options : 
        - size : number of neurons on the critic network
        - learning rate 
        - gamma : reward discount factor
        - dropout rate : if you want to deactivate some neurons
        - l2_alpha : L2 regularization coefficient
        - two_nn_layers [bool] : add another hidden layer of neurons
        """

        
        super(Critic, self).__init__()

        self.gamma = gamma
        print(env.observation_space.shape)
        self.num_inputs = env.observation_space.shape[0]
        self.linear1 = nn.Linear(self.num_inputs, size)
        self.linear2 = nn.Linear(size, size)
        self.linear3 = nn.Linear(size, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=l2_alpha)
        self.advantage = None
        self.dropout_rate = dropout_rate
        self.two_nn_layers = two_nn_layers

    def judge(self, state:np.ndarray) -> torch.Tensor:
        """ Ask the critic to judge a state (estimates the V(s)) and the
        next state (estimates the V(s')). You can specify if the episode
        is terminated, in which case the value will automatically be
        set to 0 """
        
        state = torch.Tensor(state).unsqueeze(0)
        x = F.dropout(F.relu(self.linear1(state)), self.dropout_rate)
        if self.two_nn_layers:
            x = F.dropout(F.relu(self.linear2(x)), self.dropout_rate)
        return self.linear3(x)
    

    def update(self, reward, normed_state, normed_next_state, done:bool) -> float:
        """ Computes the advantage of the previous batch
        and uses it to improve itself. Returns critic loss"""

        # Computes the estimator of the discounted rewards through TD
        value, next_value = self.judge(normed_state), self.judge(normed_next_state)
        self.advantage = reward + next_value * self.gamma * (1 - done) - value
        loss = (self.advantage ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


### Main programme 

def train():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    file_prefix = 'A2C_Cartpole_' + datetime.now().strftime('%y-%m-%d_%H-%M')
    normalizer = Normalizer(env.observation_space, 
                            mode=NORMALIZE_STRATEGY,
                            manual_low=[-4.8,-1,-4.2,-1],
                            manual_high=[4.8,1,4.2,1])
    actor_net = DiscreteActor(env)
    critic_net = Critic(env)

    # Init some monitoring stuff
    cum_rewards, c_losses, a_losses, steps = [], [], [], []
    best_model_score = -np.inf

    try:
        # Init log file
        with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
            myfile.write(f'episode,step,cum_reward,c_loss,a_loss,left_frac,right_frac\n')

        # Save hyperparameters now
        print('./models/' + file_prefix + '_prms.json')
        with open('models/' + file_prefix + '_prms.json', 'w') as myfile:
            import params as pm
            prm_dict = {key : val for key, val in pm.__dict__.items() if '__' not in key}
            json.dump(prm_dict, myfile)
        
        for episode in range(NUM_EPISODES):

            cum_reward, c_loss, a_loss, step, done, actions = 0, 0, 0, 0, False, []
            state, _ = env.reset()
            normed_state = normalizer.normalize_obs(state)
            critic_net.judge(normed_state)

            while not done:
                
                env.render()
                action = actor_net.act(normed_state)
                new_state, reward, terminated, truncated, _ = env.step(action)
                normed_new_state = normalizer.normalize_obs(new_state)
                done = terminated or truncated
                
                c_loss += critic_net.update(reward, normed_state, normed_new_state, done)
                a_loss += actor_net.update(critic_advantage=critic_net.advantage)
                
                state, normed_state = new_state, normed_new_state
                step += 1
                cum_reward += reward
                actions.append(action)

            act_frac = [actions.count(val)/len(actions) for val in range(env.action_space.n)]
            steps.append(step)
            cum_rewards.append(cum_reward)
            c_losses.append(c_loss/step)
            a_losses.append(a_loss/step)

            # Write some stuff in a file
            with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
                myfile.write(f'{episode},{step},{cum_reward},{c_loss/step},{a_loss/step},{act_frac[0]},{act_frac[1]}\n')
            
            # Print stuff in console
            bg, colr = Style.RESET_ALL, Fore.GREEN if step >= 499 else Fore.RED
            print(f'Ep. {episode:4d} | ' \
                + colr + f'Rw {cum_reward:7.2f} | Dur {step:5d}' + bg \
                + f' | AL {a_loss/step:7.2f}' \
                + f' | CL {c_loss/step:7.2f}' \
                + f' | LEFT {act_frac[0]:.3f}' \
                + f' | RIGHT {act_frac[1]:.3f}')

            # If model really good, save it
            if np.mean(cum_rewards[-20:]) > best_model_score:
                torch.save(actor_net.state_dict() , './models/' + file_prefix + '_best_actor.pkl')  
                torch.save(critic_net.state_dict(), './models/' + file_prefix + '_best_critic.pkl')
                best_model_score = np.mean(cum_rewards[-20:])

    except KeyboardInterrupt:
        print('Keyboard Interrupt.')

    finally:
        pass
        # torch.save(actor_net.state_dict() , './models/' + file_prefix + '_last_actor.pkl')  
        # torch.save(critic_net.state_dict(), './models/' + file_prefix + '_last_critic.pkl')





def evaluate(file='A2C_Cartpole_*', index=-1, mode='human'):
    """ Evaluates a trained model (default : the best version of the last trained model)"""

    actor_file = glob.glob('models/' + file + '*best_actor*')[index]
    json_file = glob.glob('models/' + file + '*prms*')[index]
    base_name = '_'.join(json_file.split('_')[:-1])
    
    with open(json_file) as myfile:
        prms = json.load(myfile)
        for key, val in prms.items():
            print(f'{key:25s} : {val}')    

    env = gym.make('CartPole-v1', render_mode=mode)

    if mode != 'human': 
        env = gym.wrappers.RecordVideo(env=env, 
                                    video_folder='',
                                    name_prefix=base_name,
                                    episode_trigger=lambda x: True)

    normalizer = Normalizer(env.observation_space, 
                            mode=prms['NORMALIZE_STRATEGY'],
                            manual_low=[-4.8,-1,-4.2,-1],
                            manual_high=[4.8,1,4.2,1])
    actor_net = DiscreteActor(env, size=prms['ACTOR_SIZE'])
    actor_net.load_state_dict(torch.load(actor_file))

    done = False
    cum_rewards, steps = [], []
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
        steps.append(step)
    
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