import json
import glob
import sys
import time
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from datetime import datetime
from colorama import Fore, Style
from torch.distributions import Categorical

from params_PPO import *


### CLASSES

class PPOActor(nn.Module):

    def __init__(self, env:gym.Env, 
                 size=ACTOR_SIZE, 
                 buffer_size=BUFFER_SIZE,
                 lr=ACTOR_LR,
                 l2_alpha=L2_ALPHA,
                 entropy_beta=ENTROPY_BETA,
                 clip_val=PPO_CLIP_VAL):

        super(PPOActor, self).__init__()

        self.n_obs = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.entropy_beta = entropy_beta
        self.clip_val = clip_val

        self.obs = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.log_probs = deque(maxlen=buffer_size)

        self.linear1 = nn.Linear(self.n_obs, size)
        self.linear2 = nn.Linear(size, self.n_actions)
            
        # These guys will lower the learning rate when the model becomes good
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=lr, weight_decay=l2_alpha)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
        (optimizer=self.optimizer,
            mode='max', 
            factor=SCHEDULER_FACTOR, 
            min_lr=SCHEDULER_MIN_LR, 
            patience=SCHEDULER_PATIENCE)

    def apply_policy(self, obs:torch.Tensor | np.ndarray, action=None, deterministic=False) -> torch.Tensor:
        """ 
        Applies the current network policy and 
        * selects an action at random if `action=None` and `deterministic=False`
        * selects the likeliest action if `action=None` and `determinstic=True` (e.g for evaluation)
        * does not select the action, just computes the probability associated to it if `action` is not None
        (action must have at least the first dimension equal to that of obs, by the way)
        """

        if isinstance(obs, np.ndarray):
            obs = torch.Tensor(obs).unsqueeze(0)

        x = F.relu(self.linear1(obs))
        probs = F.softmax(self.linear2(x), dim=1)
        distrib = Categorical(probs)
        if action is None: 
            action = distrib.sample() if not deterministic else distrib.mode.unsqueeze(0)
        log_probs = distrib.log_prob(action.squeeze()).unsqueeze(-1)
        entropy = distrib.entropy()

        return action, log_probs, entropy
    
    def update(self, normed_advantage=torch.Tensor, n_epochs=UPDATE_EPOCHS):
        """ Updates the critic network based on the clipped PPO loss"""

        # I AM PRETTY SURE WE HAVE TO DO MINIBATCHES HERE   

        actions = torch.cat(tuple(self.actions)).unsqueeze(-1)
        log_probs = torch.cat(tuple(self.log_probs)).detach()
        torch_obs = torch.tensor(np.array(self.obs))

        for epoch in range(n_epochs):
            _, new_log_probs, entropy = self.apply_policy(torch_obs, action=actions)
            new_log_probs = new_log_probs

            raw_prob_ratio = torch.exp(new_log_probs - log_probs)
            clipped_prob_ratio = torch.clip(raw_prob_ratio, 
                                            1 - self.clip_val,
                                            1 + self.clip_val)
            
            actor_loss = torch.maximum(-raw_prob_ratio * normed_advantage, 
                                       -clipped_prob_ratio * normed_advantage).mean()
            entropy_loss = torch.mean(entropy)
            loss = actor_loss + self.entropy_beta * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return actor_loss, entropy_loss
        


class PPOCritic(nn.Module):
    def __init__(self, 
                 env:gym.Env,
                 size=CRITIC_SIZE,
                 gamma=GAMMA,
                 lr=CRITIC_LR,
                 l2_alpha=L2_ALPHA,
                 buffer_size=BUFFER_SIZE
                 ):
        """
        CRITIC PART OF THE PPO AGENT
        """
        
        super(PPOCritic, self).__init__()

        self.n_obs = env.observation_space.shape[0]
        self.gamma = gamma

        self.obs = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        
        self.linear1 = nn.Linear(self.n_obs, size)
        self.linear2 = nn.Linear(size, 1)
            
        # These guys will lower the learning rate when the model becomes good
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=lr, weight_decay=l2_alpha)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
        (optimizer=self.optimizer,
            mode='max', 
            factor=SCHEDULER_FACTOR, 
            min_lr=SCHEDULER_MIN_LR, 
            patience=SCHEDULER_PATIENCE)
         
    def compute_est_state_values(self) -> torch.Tensor:
        """ Ask the critic to judge a state (estimates the V(s)) and the
        next state (estimates the V(s')). You can specify if the episode
        is terminated, in which case the value will automatically be
        set to 0 """
        
        torch_obs = torch.tensor(np.array(self.obs))
        x = F.relu(self.linear1(torch_obs))
        x = self.linear2(x)
        return x

    def compute_true_state_values(self):
        """ Computes the true state value 
         that will be compared to what is estimated from 
         the NN model to compute the critic loss"""
        
        state_values = deque(maxlen=len(self.rewards)) # I use it because of precious .appendleft method
        state_values.append(0)

        for reward, done in zip(reversed(self.rewards), 
                                  reversed(self.dones)):
            
            state_values.appendleft(reward + self.gamma * state_values[0] * (not done))

        return torch.Tensor(np.array(state_values)).unsqueeze(-1)
    
    def compute_advantage(self, detach=False, normalize=False):
        """ Computes the critic advantage for PPO. 
        Provides an option to detach the result and to normalise it (useful to give to the actor !)"""
    
        est_state_vals = self.compute_est_state_values()
        true_state_vals = self.compute_true_state_values()
        advantage = (true_state_vals - est_state_vals)
        
        if normalize:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)

        if detach:
            return advantage.detach()
        
        return advantage


    def update(self, n_epochs=UPDATE_EPOCHS):
        '''
        Updates the critic part of the PPO agent
        '''

        for _ in range(n_epochs):

            advantage = self.compute_advantage()
            loss = (advantage ** 2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss
        

        


#########################################
# TRAIN PROGRAMME
#########################################

def train():
    
    file_prefix = 'PPO_Cartpole_' + datetime.now().strftime('%y-%m-%d_%H-%M')

    # Initialize Environment and agent
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    env.metadata['render_fps'] = 150
    state, _ = env.reset()    
        
    # Introducing the agent in the field. We need the state (for its shape) to initialise the network
    actor = PPOActor(env, size=ACTOR_SIZE)
    critic = PPOCritic(env, size=CRITIC_SIZE)
    
    # Init some monitoring stuff
    actor_lr = actor.scheduler.optimizer.param_groups[0]['lr']    
    critic_lr = critic.scheduler.optimizer.param_groups[0]['lr']    
    cum_reward, actor_loss, critic_loss, entropy_loss = 0.0, 0.0, 0.0, 0.0
    episode, steps, global_steps, iters_since_last_update = 0, 0, 0, 0
    done, actions, cum_rewards, just_updated = False, [], [], False
    best_model_score = -np.inf
    time_start = time.time()

    try:
        # Init log file
        with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
            myfile.write(f'episode,step,time,cum_reward,a_loss,c_loss,e_loss,action_0,action_1,actor_lr, critic_lr,just_updated\n')

        # Save hyperparameters now
        print('./models/' + file_prefix + '_prms.json')
        with open('models/' + file_prefix + '_prms.json', 'w') as myfile:
            import params_PPO as pm
            prm_dict = {key : val for key, val in pm.__dict__.items() if '__' not in key}
            json.dump(prm_dict, myfile)

        while global_steps <= MAX_STEPS:

            # Let the agent act and face the consequences
            action, log_probs, entropy = actor.apply_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item()) 
            done = terminated or truncated

            # Update thingies
            actor.actions.append(action)
            actor.obs.append(state)
            actor.log_probs.append(log_probs)
            critic.obs.append(state)
            critic.rewards.append(reward)
            critic.dones.append(done)

            actions.append(action.item())
            iters_since_last_update += 1
            steps += 1
            global_steps += 1
            cum_reward += reward
            state = next_state

            # Network update routine
            if iters_since_last_update >= BUFFER_SIZE:
                
                iters_since_last_update = 0
                just_updated = True
                normed_advantage = critic.compute_advantage(detach=True, normalize=True)
                actor_loss, entropy_loss = actor.update(normed_advantage=normed_advantage)
                critic_loss = critic.update()

                actor.scheduler.step(metrics=cum_reward)
                critic.scheduler.step(metrics=cum_reward)
                actor_lr = actor.scheduler.optimizer.param_groups[0]['lr']    
                critic_lr = critic.scheduler.optimizer.param_groups[0]['lr']    
                
            # End of episode routine : collect episode info & write it down plus prepare next episode
            if done: 
 
                time_now = time.time() - time_start
                act_frac = [actions.count(val)/len(actions) 
                                for val in range(env.action_space.n)]
                cum_rewards.append(cum_reward)
                
               # Write some stuff in da file
                with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
                    myfile.write(f'{episode},{steps},{time_now},{cum_reward},{actor_loss},{critic_loss},{entropy_loss},' \
                                +f'{act_frac[0]},{act_frac[1]},{actor_lr},{critic_lr},{int(just_updated)}\n')
                            
                # Fancier colors for standard output
                bg, colr = Style.RESET_ALL, Fore.RED
                if cum_reward > 200: colr = Fore.YELLOW
                if cum_reward > 400: colr = Fore.BLUE
                if cum_reward == 500: colr = Fore.GREEN
                    
                print(f'Ep. {episode:4d} | Time {time_now:5.1f} | ' \
                    + colr + f'Rw {cum_reward:+8.2f}' + bg \
                    + f' | A_LOSS {actor_loss:5.2f}' \
                    + f' | C_LOSS {critic_loss:7.2f}' \
                    + f' | E_LOSS {entropy_loss:5.2f}' \
                    + f' | ACT0 {act_frac[0]:.3f}' \
                    + f' | ACT1 {act_frac[1]:.3f}' \
                    + f' | A_LR {actor_lr:5.2e}'\
                    + f' | C_LR {critic_lr:5.2e}'\
                    + f' | UPD {just_updated}')
                
                # Reset variables for next episode 
                env.reset()
                cum_reward, steps, done, actions = 0, 0, False, []
                just_updated = False
                episode += 1

                # If model really good, save it
                if len(cum_rewards) > 20 and (np.mean(cum_rewards[-20:]) > best_model_score):
                    torch.save(actor.state_dict() , './models/' + file_prefix + '_best_actor.pkl')  
                    torch.save(actor.state_dict() , './models/' + file_prefix + '_best_critic.pkl')  
                    best_model_score = np.mean(cum_rewards[-20:])
        
    except KeyboardInterrupt:
        print('\nInterrupted w. Keyboard !')

    finally:
        pass
        # torch.save(actor.state_dict(), './models/' + file_prefix + '_latest_actor.pkl')
        # torch.save(actor.state_dict() , './models/' + file_prefix + '_latest_critic.pkl')  


def evaluate(file, index=-1, mode='human'):
    """ Evaluates a trained model (default : the best version of the last trained model)"""

    model_file = glob.glob('./models/' + file + '*best_actor*')[index]
    json_file = glob.glob('./models/' + file + '*prms*')[index]
    base_name = '_'.join(json_file.split('_')[:-1])

    print(model_file)
    print(json_file)

    with open(json_file) as myfile:
        prms = json.load(myfile)
    
    env = gym.make('CartPole-v1', render_mode=mode)
    env.metadata['render_fps'] = 150
    if mode != 'human': 
        env = gym.wrappers.RecordVideo(env=env, 
                                    video_folder='',
                                    name_prefix=base_name,
                                    episode_trigger=lambda x: True)

    actor = PPOActor(env, size=prms['ACTOR_SIZE'], buffer_size=prms['BUFFER_SIZE'])
    actor.load_state_dict(torch.load(model_file))
        
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
        
        while not done:
            # env.render()
            action, _, _ = actor.apply_policy(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            state = next_state
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
        file = 'PPO_Cartpole' 
        if len(sys.argv) > 2:
            file += sys.argv[2]
        if len(sys.argv) > 3 and '--video' in sys.argv[3]:
            evaluate(file=file, mode='rgb_array')
        else: 
            evaluate(file=file, mode='human')
    else:
        train()
