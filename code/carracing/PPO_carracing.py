import json
import glob
import sys
import time
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

from collections import deque
from datetime import datetime
from colorama import Fore, Style
from torch.distributions import Categorical


from params_PPO import *


### CLASSES

class ConvNet(nn.Module):

    def __init__(self, obs:tuple, n_filters=16, num_actions=1, type='actor', dropout_rate=DROPOUT_RATE):

        super(ConvNet, self).__init__()
        img_shape = obs.shape[-2:] # Dim 0 will be reserved for the batch
        n_imgs = obs.shape[-3]

        k1, k2 = 4,2
        s1, s2 = 4,2
        img_shape1 = [(1 + (elem - k1) // (s1)) // 2 for elem in img_shape]
        img_shape2 = [(1 + (elem - k2) // (s2)) // 2 for elem in img_shape1]
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
        
        if type == 'actor': 
            # We get the probabilities for each action
            self.dense = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(n_linear, n_linear // 4),
                    nn.ReLU(),
                    nn.Linear(n_linear // 4, num_actions),
                    nn.Softmax(dim=1)
                )
            
        else:
            # We get the estimation of the value of a state
            self.dense = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(n_linear, n_linear // 4),
                    nn.ReLU(),
                    nn.Linear(n_linear // 4, 1)
                )        

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.dense(x)    

class PPOAgent():
    def __init__(self, 
                 env:gym.Env,
                 state:np.ndarray,
                 n_imgs=N_IMGS,
                 n_idle=N_IDLE,
                 n_filters=N_FILTERS,
                 gamma=GAMMA,
                 actor_lr=ACTOR_LR,
                 critic_lr=CRITIC_LR,
                 L2_alpha=L2_ALPHA,
                 entropy_beta=ENTROPY_BETA,
                 buffer_size=BUFFER_SIZE,
                 minibatch_size=MINIBATCH_SIZE
                 ):
        
        super(PPOAgent, self).__init__()

        # Save some parameters of the model for later 
        self.n_imgs = n_imgs 
        self.n_idle = n_idle
        self.entropy_beta = entropy_beta
        self.action_space = env.action_space
        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size

        # Preparing the agent observation and network
        self.current_obs = deque(maxlen=self.n_imgs) # This one is used to produce the "multi_image" observation
        for _ in range(self.n_imgs):
            self.current_obs.append(state[:,:,1]/255)
        torch_obs = torch.tensor(np.array(self.current_obs), dtype=torch.float32).unsqueeze(0)

        self.critic_net = ConvNet(torch_obs, n_filters=n_filters, type='critic')
        self.actor_net = ConvNet(torch_obs, n_filters=n_filters, type='actor', num_actions=self.num_actions)

        self.actor_optimizer = torch.optim.AdamW(params=self.actor_net.parameters(), lr=actor_lr, weight_decay=L2_alpha)
        self.critic_optimizer = torch.optim.AdamW(params=self.critic_net.parameters(), lr=critic_lr, weight_decay=L2_alpha)

        # The mighty queues storing information
        self.obs = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.log_probs = deque(maxlen=buffer_size) # Also known as "pi_old, old probas"
        self.entropy = deque(maxlen=buffer_size)
        
        # These guys will lower the learning rate when the model becomes good
        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (optimizer=self.actor_optimizer,
             mode='max', 
             factor=SCHEDULER_FACTOR, 
             min_lr=SCHEDULER_MIN_LR, 
             patience=SCHEDULER_PATIENCE)
        
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (optimizer=self.critic_optimizer,
             mode='max', 
             factor=SCHEDULER_FACTOR, 
             min_lr=SCHEDULER_MIN_LR, 
             patience=SCHEDULER_PATIENCE)

    def step_idle(self, env:gym.Env, action:int | np.ndarray):
        """ A routine that plays n_idle steps from an environment with the same action. 
        The routine : 
        - discards the intermediate states (not put anywhere)
        - sums the rewards and returns them as a "collective" reward
        - returns early if the episode ends (truncated or )
        - returns the _actual_ number of idle steps (can be less due to termination)"""
        
        next_state, cum_reward, terminated, truncated, _ = env.step(action)

        for i_step in range(1, self.n_idle):
            if terminated or truncated:
                break
            next_state, reward, terminated, truncated, _ = env.step(action)
            cum_reward += reward

        return next_state, cum_reward, terminated, truncated, i_step + 1

    
    def observe(self, state:np.ndarray):
        """Pre-processes the raw state of CarRacing to actually produce 
        a sequence of successive images."""

        self.current_obs.append(state[:,:,1]/255)   # We add one image, we ditch the oldest one, this is how we update
        return torch.tensor(np.array(self.current_obs), dtype=torch.float32).unsqueeze(0)
    
    
    def apply_policy(self, obs:torch.Tensor | np.ndarray, action=None, deterministic=False) -> torch.Tensor:
        """ 
        Applies the current network policy and 
        * selects an action at random if `action=None` and `deterministic=False`
        * selects the likeliest action if `action=None` and `determinstic=True` (e.g for evaluation)
        * does not select the action, just computes the probability associated to it if `action` is not None
        (action must have at least the first dimension equal to that of obs, by the way)
        """
 
        if isinstance(obs, np.ndarray):
            torch_obs = torch.Tensor(obs).unsqueeze(0)
        else:
            torch_obs = obs

        probs = self.actor_net.forward(torch_obs)
        distrib = Categorical(probs)
        if action is None: 
            action = distrib.sample() if not deterministic else distrib.mode.unsqueeze(0)

        log_probs = distrib.log_prob(action.squeeze()).unsqueeze(-1)
        entropy = distrib.entropy()

        return action, log_probs, entropy

   
    def est_state_values(self, torch_obs:torch.Tensor) -> torch.Tensor:
        """ Ask the critic to judge a state (estimates the V(s)) and the
        next state (estimates the V(s'))."""
        
        return self.critic_net(torch_obs)
    

    def true_state_values(self) -> torch.Tensor:
        """ Computes the true state value 
         that will be compared to what is estimated from 
         the NN model to compute the critic loss. 
         So the way I code things means I am doing Monte Carlo sampling"""
        
        state_values = deque(maxlen=len(self.rewards)) # I use it because of precious .appendleft method
        state_values.append(0)

        for reward, done in zip(reversed(self.rewards), 
                                  reversed(self.dones)):
            
            state_values.appendleft(reward + self.gamma * state_values[0] * (not done))

        return torch.Tensor(np.array(state_values)).unsqueeze(-1)
    

    def update(self, n_epochs=UPDATE_EPOCHS, clip_val=PPO_CLIP_VAL):
        '''
        Here I update both the critic net and the policy (actor) net at the same time.
        NOTE 1 : the mini_batching is a bit ugly
        NOTE 2 : Somehow the advantage should _not_ be updated for the actor during update ? 
        '''
        
        # These things should not change throughout the epochs 
        true_state_vals = self.true_state_values() 
        torch_obs = torch.cat(tuple(self.obs))
        actions = torch.cat(tuple(self.actions)).unsqueeze(-1)
        log_probs = torch.cat(tuple(self.log_probs)).detach()
        advantage = true_state_vals - self.est_state_values(torch_obs)  # Used for actor only
        adv_normed = (advantage - advantage.mean()) \
            / (1e-5 + advantage.std())  # Used for actor only
        
        critic_loss_record, actor_loss_record, entropy_loss_record = [],[],[]

        for _ in range(n_epochs):

            [obs_batches, actions_batches, log_probs_batches, adv_batches, tsv_batches] \
                = minibatch(torch_obs, actions, log_probs, adv_normed, true_state_vals, split_length=self.minibatch_size)

            for obs_batch, actions_batch, log_probs_batch, adv_batch, tsv_batch in \
                zip(obs_batches, actions_batches, log_probs_batches, adv_batches, tsv_batches):

                # Actor part
                _, new_log_probs_batch, entropy_batch = self.apply_policy\
                    (obs=obs_batch, action=actions_batch)
                
                raw_prob_ratio = torch.exp(new_log_probs_batch - log_probs_batch)
                clipped_prob_ratio = torch.clip(raw_prob_ratio, 1 - clip_val, 1 + clip_val)
                actor_loss = torch.maximum \
                    (-adv_batch.detach() * raw_prob_ratio, \
                     -adv_batch.detach() * clipped_prob_ratio).mean()
                entropy_loss = self.entropy_beta * entropy_batch.mean()
                actor_loss += entropy_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # # Critic part
                esv_batch = self.est_state_values(obs_batch)
                critic_loss = ((tsv_batch - esv_batch) ** 2).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                actor_loss_record.append(actor_loss.item())
                entropy_loss_record.append(entropy_loss.item())
                critic_loss_record.append(critic_loss.item())
            
        return sum(actor_loss_record)/len(actor_loss_record), \
            sum(critic_loss_record)/len(critic_loss_record), \
            sum(entropy_loss_record)/len(entropy_loss_record)


########################################
# HELPER FUNCTIONS
#########################################

def minibatch(*args, 
              split_length:int) -> list[torch.Tensor]:
    """Shuffles the data buffer and splits it into mini-batches of size
    `split_length`."""

    out_vars = []
    shuffled = torch.randperm(len(args[0]))
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            arg = torch.tensor(arg)
        out_vars.append(torch.split(arg[shuffled],split_length))

    return out_vars

#########################################
# TRAIN PROGRAMME
#########################################

def train():
    
    file_prefix = 'PPO_Carracing_mb_' + datetime.now().strftime('%y-%m-%d_%H-%M')

    # Initialize Environment and agent
    env = gym.make("CarRacing-v2", continuous=False, render_mode='rgb_array')
    env.metadata['render_fps'] = 300
    state, _ = env.reset()    
        
    # Introducing the agent in the field. We need the state (for its shape) to initialise the network
    agent = PPOAgent(env, state)
    for _ in range(N_START_SKIP): 
        action = 3
        state, _, _, _, _ = env.step(action)    
        obs = agent.observe(state) # Fills the observation deque and only works if N_START_SKIP > N_IMGS
    
    # Init some monitoring stuff
    a_lr = agent.actor_scheduler.optimizer.param_groups[0]['lr']  
    c_lr = agent.critic_scheduler.optimizer.param_groups[0]['lr']     
    cum_reward, actor_loss, critic_loss, entropy_loss = 0, 0, 0, 0
    episode, steps, global_steps, iters_since_last_update = 0, N_START_SKIP, N_START_SKIP, 0
    done, actions, cum_rewards = False, [], []
    best_model_score = -np.inf
    time_start = time.time()

    try:
        # Init log file
        with open('./models/' + file_prefix + '_log.csv', 'w') as myfile:
            myfile.write(f'episode,step,time,cum_reward,a_loss,c_loss,e_loss,action_0,action_1,action_2,action_3,action_4,a_lr,c_lr\n')

        # Save hyperparameters now
        print('./models/' + file_prefix + '_prms.json')
        with open('models/' + file_prefix + '_prms.json', 'w') as myfile:
            import params_PPO as pm
            prm_dict = {key : val for key, val in pm.__dict__.items() if '__' not in key}
            json.dump(prm_dict, myfile)

        while global_steps <= MAX_STEPS:

            # Let the agent act and face the consequences
            obs = agent.observe(state)
            action, probs, entropy = agent.apply_policy(obs)
            next_state, reward, terminated, truncated, n_idle \
                = agent.step_idle(env, action.item()) # Agent slacks off a bit ...
            done = terminated or truncated

            # Update thingies
            agent.actions.append(action)
            agent.obs.append(obs)
            agent.rewards.append(reward)
            agent.dones.append(done)
            agent.log_probs.append(probs)
            agent.entropy.append(entropy)

            actions.append(action.item())
            iters_since_last_update += 1
            steps += n_idle
            global_steps += n_idle
            cum_reward += reward
            state = next_state

            # Network update routine
            if iters_since_last_update >= BUFFER_SIZE:
                
                iters_since_last_update = 0
                actor_loss, critic_loss, entropy_loss \
                    = agent.update(n_epochs=UPDATE_EPOCHS)
                agent.actor_scheduler.step(metrics=cum_reward)
                agent.critic_scheduler.step(metrics=cum_reward)
                a_lr = agent.actor_scheduler.optimizer.param_groups[0]['lr']   
                c_lr = agent.critic_scheduler.optimizer.param_groups[0]['lr'] 
                
            # End of episode routine : collect episode info & write it down plus prepare next episode
            if done: 
 
                time_now = time.time() - time_start
                act_frac = [actions.count(val)/len(actions) 
                                for val in range(env.action_space.n)]
                cum_rewards.append(cum_reward)
                
               # Write some stuff in da file
                with open('./models/' + file_prefix + '_log.csv', 'a') as myfile:
                    myfile.write(f'{episode},{steps},{time_now},{cum_reward},{actor_loss},{critic_loss},{entropy_loss},' \
                                +f'{act_frac[0]},{act_frac[1]},{act_frac[2]},{act_frac[3]},{act_frac[4]},{a_lr},{c_lr}\n')
                            
                # Fancier colors for standard output
                bg, colr = Style.RESET_ALL, Fore.RED
                if cum_reward > 0:
                    colr = Fore.MAGENTA
                if cum_reward > 200:
                    colr = Fore.BLUE
                if cum_reward > 400:
                    colr = Fore.CYAN
                if cum_reward > 600:
                    colr = Fore.GREEN
                    
                print(f'Ep. {episode:4d} | T {time_now:7.1f} | ' \
                    + colr + f'Rw {cum_reward:+7.2f} | EpLen {steps:4d}' + bg \
                    + f' | A_LOSS {actor_loss:+5.2e}' \
                    + f' | C_LOSS {critic_loss:6.2f}' \
                    + f' | E_LOSS {entropy_loss:5.2f}' \
                    + f' | A0 {act_frac[0]:.2f}' \
                    + f' | A1 {act_frac[1]:.2f}' \
                    + f' | A2 {act_frac[2]:.2f}' \
                    + f' | A3 {act_frac[3]:.2f}' \
                    + f' | A4 {act_frac[4]:.2f}' \
                    + f' | A_LR {a_lr:5.2e}')
                
                # Reset variables for next episode 
                state, _ = env.reset()
                cum_reward, steps, done, actions = 0, N_START_SKIP, False, []
                global_steps += N_START_SKIP
                episode += 1
                for _ in range(N_START_SKIP): 
                    action = 3
                    agent.observe(state) # Fills the observation deque and only works if N_START_SKIP > N_IMGS
                    state, _, _, _, _ = env.step(action)    
                
                # If model really good, save it
                if len(cum_rewards) > 20 and (np.mean(cum_rewards[-20:]) > best_model_score):
                    torch.save(agent.actor_net.state_dict() , './models/' + file_prefix + '_best_actor.pkl')  
                    torch.save(agent.critic_net.state_dict() , './models/' + file_prefix + '_best_critic.pkl')  
                    best_model_score = np.mean(cum_rewards[-20:])
        
    except KeyboardInterrupt:
        print('\nInterrupted w. Keyboard !')

    finally:
        torch.save(agent.actor_net.state_dict(), './models/' + file_prefix + '_latest_actor.pkl')
        torch.save(agent.critic_net.state_dict(), './models/' + file_prefix + '_latest_critic.pkl')



def evaluate(file, index=-1, mode='human'):
    """ Evaluates a trained model (default : the best version of the last trained model)"""

    model_file = glob.glob('./models/' + file + '*best_actor*')[index]
    json_file = glob.glob('./models/' + file + '*prms*')[index]
    base_name = '_'.join(json_file.split('_')[:-1])
    print(json_file)
    print(model_file)

    with open(json_file) as myfile:
        prms = json.load(myfile)
    
    env = gym.make('CarRacing-v2', continuous=False, render_mode=mode)
    env.metadata['render_fps'] = 30
    state, _ = env.reset()

    if mode != 'human': 
        env = gym.wrappers.RecordVideo(env=env, 
                                    video_folder='',
                                    name_prefix=base_name,
                                    episode_trigger=lambda x: True)

    agent = PPOAgent(env, state=state, n_filters=prms['N_FILTERS'], n_idle=prms['N_IDLE'], n_imgs=prms['N_IMGS'])
    agent.actor_net.load_state_dict(torch.load(model_file))  
        
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
        for _ in range(N_START_SKIP): 
            action = 3
            state, _, _, _, _ = env.step(action)    
            obs = agent.observe(state) # Fills the observation deque and only works if N_START_SKIP > N_IMGS
        
        while not done:

            action, _, _ = agent.apply_policy(obs, deterministic=False) # Deterministic or not here, not sure ...
            next_state, reward, terminated, truncated, jump_steps = agent.step_idle(env, action.item())
            next_obs = agent.observe(next_state)
            done = terminated or truncated

            obs = next_obs
            cum_reward += reward
            step += jump_steps

        cum_rewards.append(cum_reward)
        steps.append(step)
    
    # Close the environment
    if mode != 'human':
        env.close_video_recorder()
        env.close()

    print(f'5 episodes : Rewards {cum_rewards}')    


if __name__ == '__main__':
    if len(sys.argv) > 1 and ('--eval' in sys.argv[1] or '-e' in sys.argv[1]):
        file = 'PPO_CarRacing' 
        print(file)
        if len(sys.argv) > 2:
            file += sys.argv[2]
        if len(sys.argv) > 3 and '--video' in sys.argv[3]:
            evaluate(file=file, mode='rgb_array')
        else: 
            evaluate(file=file, mode='human')
    else:
        train()
