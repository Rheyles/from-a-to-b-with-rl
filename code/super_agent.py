import json
import os
import glob
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from params import *
from buffer import ReplayMemory, TorchMemory

class SuperAgent():
    """The SuperAgent class. Basically an empty
    shell with plenty of methods that will actually
    be filled by its children classes (DQNAgent, A2CAgent, ...)"""

    def __init__(self, **kwargs) -> None:
        self.exploration = kwargs.get('exploration',True)
        self.training = kwargs.get('training',True)
        self.show_diagnostics = kwargs.get('show_diagnostics',False)
        self.creation_time = datetime.now()
        self.time = (datetime.now() - self.creation_time).total_seconds()
        self.log_buffer = []

        self.steps_done = 0
        self.episode = 0
        self.losses = [0]
        self.rewards = [0]
        self.episode_rewards = [0]
        self.episode_duration = [0]
        self.last_action = torch.tensor([[0]], dtype=torch.long, device=DEVICE)


    def select_action(self)-> torch.Tensor:
        pass
    def optimize_model(self) -> list:
        pass
    def soft_update_agent(self, *args, **kwargs):
        pass
    def save_model(self, **kwargs):
        pass
    def load_model(self):
        pass
    def logging(self):
        pass




class DQNAgent(SuperAgent):
    '''The DeepQNetwork Agent class.
    Works using :
    1/ A Memory buffer that stores states, decisions, rewards, ...
    that allow you to "replay the game" and find where you could have made
    better decisions through a Q score
    2/ A (sometimes Deep) neural network that takes observations and spits
    out an estimated Q score that will be regularly updated to reflect the
    true Q score that the memory buffer can evaluate.
    '''

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


        self.epsilon = EPS_START

        # Choosing my optimizer
        if OPTIMIZER.lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(),
                                                 lr=INI_LR,
                                                 weight_decay=REGULARIZATION)
        elif OPTIMIZER.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.policy_net.parameters(),
                                                 lr=INI_LR,
                                                 weight_decay=REGULARIZATION)
        else:
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                                 lr=INI_LR,
                                                 weight_decay=REGULARIZATION)

        # Choosing loss function
        if LOSS.lower() == 'mse':
            self.lossfun = nn.MSELoss()
        else:
            self.lossfun = nn.SmoothL1Loss()

        # Memory type
        if MEM_TYPE.lower() == 'legacy':
            self.memory = ReplayMemory(MEM_SIZE)
        elif MEM_TYPE.lower() == 'torch':
            self.memory = TorchMemory(MEM_SIZE)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
            (optimizer=self.optimizer,
             mode='max',
             factor=SCHEDULER_FACTOR,
             min_lr = MIN_LR,
             patience=SCHEDULER_PATIENCE)

        self.folder = 'models/' \
            + datetime.strftime(datetime.now(), "%m%d_%H%M_") \
            + str(self.__class__.__name__) + '/'

        os.makedirs(self.folder, exist_ok=True)

    def prepro(self, state: torch.Tensor) -> torch.Tensor:
        """ By default, let's do zero preprocessing.
        Children classes will overwrite it"""

        if self.steps_done == 0:
            print("NOTE : DQN agent default preprocessing >> Not doing anything")
        return state

    def update_memory(self, state:torch.Tensor,
                      action:torch.Tensor,
                      next_state:torch.Tensor,
                      reward:torch.Tensor,
                      not_done:torch.Tensor,
                      skip_steps=50) -> None:
        """ UPDATE_MEMORY (SUPER_AGENT) defines
        what to do in general when we update the
        agent memory.

        Optional arg: skip_step (default 50), forces the programme to not store
        any states at the beginning of each episode (e.g. for CarRace, when they
        are not useful)
        """

        state = self.prepro(state)
        next_state = self.prepro(next_state)
        reward = reward.unsqueeze(-1)
        not_done = not_done.unsqueeze(-1)

        self.rewards.append(reward[0].item())
        current_episode_rewards = sum(self.rewards[-self.episode_duration[-1]:])
        episode_is_done = current_episode_rewards < EARLY_STOPPING_SCORE

        if self.episode_duration[-1] < skip_steps: # On ne met pas en mémoire le zoom de début d'épisode
            return episode_is_done

        self.memory.push(state, action, next_state, reward, not_done)

        return episode_is_done


    def update_agent(self, strategy=NETWORK_REFRESH_STRATEGY):
        """
        Performs a soft update of the agent's networks: i.e. updates the weights of the target net according to the changes
        observed in the policy net. In initial code, updated at every episode
        """

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        if strategy == 'soft':
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            self.target_net.load_state_dict(target_net_state_dict)

        elif self.steps_done % int(1/TAU) == 0:
            # Hard refresh every 1/TAU
            self.target_net.load_state_dict(policy_net_state_dict)


    def save_model(self, add_episode=True) -> None:
        """
        Save the model (NN and hyperparameters).

        Args:
            folder (str, optional): Where to save the model. Defaults to 'models/'.
        """

        episode_str = ''
        if add_episode:
            episode_str = f'_{self.episode:05d}'

        os.makedirs(self.folder, exist_ok=True)
        torch.save(self.policy_net.state_dict(),
                   f'{self.folder}/policy{episode_str}.model')
        torch.save(self.target_net.state_dict(),
                   f'{self.folder}/target{episode_str}.model')

        with open(self.folder + '/params.json', 'w') as my_file:
            import params as prm
            my_dict = prm.__dict__
            my_dict['DEVICE'] = DEVICE.__str__()
            my_dict = {key : val for key, val in my_dict.items()
                       if '__' not in key
                       and key != 'torch'}
            json.dump(my_dict, my_file)


    def load_model(self, folder : str) -> None:
        """
        Load a model (ex: agt.load_model("./models/0605_1015DQNAgentObs"))
        Args:
            folder (str): Folder to the model
        """

        policy_files = sorted(glob.glob(folder + '/policy_*.model'))
        target_files = sorted(glob.glob(folder + '/target_*.model'))

        self.policy_net.load_state_dict(torch.load(policy_files[-1], map_location=DEVICE))
        self.target_net.load_state_dict(torch.load(target_files[-1], map_location=DEVICE))
        print(f'Loaded model weights : {policy_files[-1]} from {folder}')

    def logging(self):
        """Logs some statistics on the agent running as a function of time
        in a .csv file"""

        if not os.path.exists(self.folder + 'log.csv'):
            with open(self.folder + 'log.csv', 'w') as log_file:
                log_file.write('Time,Step,Episode,Loss,Reward,Eta,Epsilon,Action\n')

        lr = self.scheduler.optimizer.param_groups[0]['lr']

        self.log_buffer.append([self.time,
                                     self.steps_done,
                                     self.episode,
                                     self.losses[-1],
                                     self.rewards[-1],
                                     lr,
                                     self.epsilon,
                                     self.last_action.item()])

        if self.steps_done % LOG_EVERY == 0:
            array_test = np.vstack(self.log_buffer)
            self.log_buffer = []

            with open(self.folder + 'log.csv', 'a') as myfile:
                np.savetxt(myfile, array_test, delimiter=',',
                           fmt=["%7.2f", "%6d", "%4d",
                                "%5.3e", "%5.3e", "%5.3e",
                                "%5.3e", "%d"])
